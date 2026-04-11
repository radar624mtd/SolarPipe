using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using SolarPipe.Core.Interfaces;
using SolarPipe.Core.Models;
using SolarPipe.Data.DataFrame;
using SolarPipe.Training.Adapters;
using SolarPipe.Training.Evaluation;

namespace SolarPipe.Training.Sweep;

// ModelSweep: pre-flight gate + parallel hypothesis runner (RULE-160–163).
//
// Accepts SweepRunConfig (in SolarPipe.Core) to avoid a circular Training→Config dependency.
// SweepCommand (in Host) converts SweepConfig → SweepRunConfig before calling here.
//
// Pre-flight validates adapters, data frames, sidecar, config, and write permissions
// before any training begins (RULE-160).
//
// Checkpoint paths are hypothesis-scoped (RULE-162):
//   {cache}/sweeps/{sweep_id}/{hypothesis_id}/{stage_name}/
public sealed class ModelSweep
{
    private readonly IReadOnlyList<IFrameworkAdapter> _adapters;
    private readonly ComprehensiveMetricsEvaluator _metricsEvaluator;
    private readonly NnlsEnsembleOptimizer _nnlsOptimizer;
    private readonly string _cacheRoot;
    private readonly string _registryRoot;
    private readonly GrpcSidecarAdapter? _sidecarAdapter;

    public ModelSweep(
        IReadOnlyList<IFrameworkAdapter> adapters,
        ComprehensiveMetricsEvaluator metricsEvaluator,
        NnlsEnsembleOptimizer nnlsOptimizer,
        string cacheRoot,
        string registryRoot,
        GrpcSidecarAdapter? sidecarAdapter = null)
    {
        _adapters         = adapters          ?? throw new ArgumentNullException(nameof(adapters));
        _metricsEvaluator = metricsEvaluator  ?? throw new ArgumentNullException(nameof(metricsEvaluator));
        _nnlsOptimizer    = nnlsOptimizer     ?? throw new ArgumentNullException(nameof(nnlsOptimizer));
        _cacheRoot        = cacheRoot         ?? throw new ArgumentNullException(nameof(cacheRoot));
        _registryRoot     = registryRoot      ?? throw new ArgumentNullException(nameof(registryRoot));
        _sidecarAdapter   = sidecarAdapter;
    }

    // ── Public API ────────────────────────────────────────────────────────────

    // Run pre-flight validation gate. Returns success or list of failures. (RULE-160)
    public async Task<PreFlightResult> ValidatePreFlightAsync(
        SweepRunConfig config,
        IReadOnlyDictionary<string, IDataFrame> dataFrames,
        CancellationToken ct)
    {
        var failures = new List<PreFlightFailure>();

        // 1. Data sources: each frame must be non-null and non-empty
        foreach (var (name, frame) in dataFrames)
        {
            if (frame is null || frame.RowCount == 0)
                failures.Add(new PreFlightFailure("data_source",
                    $"Data source '{name}' is null or empty."));
        }

        // 2. Adapters: verify each referenced framework type resolves
        var frameworksNeeded = config.Stages.Values
            .Select(s => s.Framework)
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .ToList();

        foreach (var fw in frameworksNeeded)
        {
            var adapter = _adapters.FirstOrDefault(a =>
                string.Equals(a.FrameworkType.ToString(), fw, StringComparison.OrdinalIgnoreCase));

            if (adapter is null)
                failures.Add(new PreFlightFailure("adapter",
                    $"No adapter registered for framework '{fw}'."));
        }

        // 3. Sidecar health — required if any stage uses python_grpc (RULE-160)
        bool needsSidecar = config.Stages.Values.Any(s =>
            s.Framework.Equals("python_grpc", StringComparison.OrdinalIgnoreCase));

        if (needsSidecar)
        {
            if (_sidecarAdapter is null)
            {
                failures.Add(new PreFlightFailure("sidecar",
                    "python_grpc stage requires GrpcSidecarAdapter but none was registered."));
            }
            else
            {
                bool healthy;
                try { healthy = await _sidecarAdapter.CheckHealthAsync(TimeSpan.FromSeconds(10), ct); }
                catch (Exception ex)
                {
                    failures.Add(new PreFlightFailure("sidecar",
                        $"gRPC sidecar health check threw: {ex.Message}"));
                    healthy = false;
                }

                if (!healthy && failures.All(f => f.Component != "sidecar"))
                    failures.Add(new PreFlightFailure("sidecar",
                        "gRPC sidecar did not become healthy within 10 seconds."));
            }
        }

        // 4. Config validity: all hypothesis stage refs must exist
        var knownStages = config.Stages.Keys.ToHashSet(StringComparer.OrdinalIgnoreCase);
        foreach (var h in config.Hypotheses)
        {
            foreach (var s in h.Stages)
            {
                if (!knownStages.Contains(s))
                    failures.Add(new PreFlightFailure("config",
                        $"Hypothesis '{h.Id}' references unknown stage '{s}'."));
            }
        }

        // 5. Registry and cache write permissions
        if (!CanWrite(_registryRoot))
            failures.Add(new PreFlightFailure("registry",
                $"Registry path '{_registryRoot}' is not writable."));

        var sweepCachePath = Path.Combine(_cacheRoot, "sweeps");
        if (!CanWrite(sweepCachePath))
            failures.Add(new PreFlightFailure("cache",
                $"Sweep cache path '{sweepCachePath}' is not writable."));

        return failures.Count == 0
            ? PreFlightResult.Success()
            : PreFlightResult.Fail(failures);
    }

    // Run all hypotheses; returns full sweep result. (RULE-163, RULE-164)
    // sweepId: null = generate from config hash; provided = resume mode (RULE-162).
    public async Task<SweepResult> RunAsync(
        SweepRunConfig config,
        IDataFrame sweepData,
        double dbmBaselineMae,
        string? sweepId,
        CancellationToken ct)
    {
        string id = sweepId ?? ComputeSweepId(config);
        Console.WriteLine($"[sweep:{id}] Starting sweep '{config.Name}'");

        var folds = BuildFolds(sweepData, config.Folds, config.GapBufferDays, config.MinTestEvents);
        int trainingFoldCount = folds.Count - 1;  // last fold = calibration (RULE-164)

        var hypothesisResults = new List<HypothesisResult>();

        if (config.Parallel)
        {
            // Wrap each hypothesis in a try/catch so a single failure (e.g. sidecar unavailable)
            // doesn't abort the whole sweep via Task.WhenAll exception propagation.
            var tasks = config.Hypotheses.Select(async h =>
            {
                try
                {
                    return await EvaluateHypothesisAsync(h, config, folds, trainingFoldCount,
                        dbmBaselineMae, id, ct);
                }
                catch (OperationCanceledException) { throw; }
                catch (Exception ex)
                {
                    Console.Error.WriteLine(
                        $"[sweep:{h.Id}] SKIP — hypothesis failed: {ex.GetType().Name}: {ex.Message}");
                    return (HypothesisResult?)null;
                }
            }).ToList();

            foreach (var r in await Task.WhenAll(tasks))
                if (r is not null) hypothesisResults.Add(r);
        }
        else
        {
            foreach (var h in config.Hypotheses)
            {
                ct.ThrowIfCancellationRequested();
                try
                {
                    var result = await EvaluateHypothesisAsync(h, config, folds, trainingFoldCount,
                        dbmBaselineMae, id, ct);
                    hypothesisResults.Add(result);
                }
                catch (OperationCanceledException) { throw; }
                catch (Exception ex)
                {
                    Console.Error.WriteLine(
                        $"[sweep:{h.Id}] SKIP — hypothesis failed: {ex.GetType().Name}: {ex.Message}");
                }
            }
        }

        // Sort leaderboard by mean MAE ascending
        hypothesisResults.Sort((a, b) => a.AggregatedMetrics.MaeMean.CompareTo(b.AggregatedMetrics.MaeMean));

        Console.WriteLine(
            $"[sweep:{id}] Complete. Winner: {hypothesisResults[0].HypothesisId} " +
            $"MAE={hypothesisResults[0].AggregatedMetrics.MaeMean:F3}");

        return new SweepResult(id, config.Name, hypothesisResults);
    }

    // ── Fold building ─────────────────────────────────────────────────────────

    private static IReadOnlyList<(IDataFrame Train, IDataFrame Test)> BuildFolds(
        IDataFrame data, int totalFolds, int gapBufferDays, int minTestEvents)
    {
        var tsCol = data.Schema.Columns.FirstOrDefault(c => c.Type == ColumnType.DateTime)
                 ?? data.Schema.Columns.FirstOrDefault(c =>
                        c.Name.Equals("timestamp", StringComparison.OrdinalIgnoreCase));

        if (tsCol is null)
            throw new InvalidOperationException("Sweep data must have a timestamp column.");

        float[] ts = data.GetColumn(tsCol.Name);
        int n = ts.Length;

        float tMin = ts.Min();
        float tMax = ts.Max();
        float span = tMax - tMin;
        float gap  = (float)TimeSpan.FromDays(gapBufferDays).TotalSeconds;

        var folds = new List<(IDataFrame, IDataFrame)>(totalFolds);

        for (int k = 0; k < totalFolds; k++)
        {
            float testStart = tMin + span * (k + 1f) / (totalFolds + 1f);
            float testEnd   = k + 1 < totalFolds
                ? tMin + span * (k + 2f) / (totalFolds + 1f)
                : tMax;

            int[] trainIdx = Enumerable.Range(0, n)
                .Where(i => ts[i] < testStart - gap).ToArray();
            int[] testIdx  = Enumerable.Range(0, n)
                .Where(i => ts[i] >= testStart && ts[i] <= testEnd).ToArray();

            if (trainIdx.Length == 0)
                throw new InvalidOperationException(
                    $"Fold {k}: training set is empty after gap buffer.");

            folds.Add((SelectRows(data, trainIdx), SelectRows(data, testIdx)));
        }

        return folds;
    }

    // ── Hypothesis evaluation ─────────────────────────────────────────────────

    private async Task<HypothesisResult> EvaluateHypothesisAsync(
        SweepHypothesis hypothesis,
        SweepRunConfig config,
        IReadOnlyList<(IDataFrame Train, IDataFrame Test)> allFolds,
        int trainingFoldCount,
        double dbmBaselineMae,
        string sweepId,
        CancellationToken ct)
    {
        string tag = $"[sweep:{hypothesis.Id}]";
        Console.WriteLine($"{tag} Starting hypothesis '{hypothesis.ComposeExpression}'");

        var trainingFolds = allFolds.Take(trainingFoldCount).ToList();  // RULE-164
        var calibFold     = allFolds[trainingFoldCount];

        var foldMetricsList = new List<FoldMetrics>(trainingFoldCount);

        for (int foldIdx = 0; foldIdx < trainingFolds.Count; foldIdx++)
        {
            ct.ThrowIfCancellationRequested();

            var (trainFrame, testFrame) = trainingFolds[foldIdx];
            string stageTag = hypothesis.Stages[^1];  // last stage = final predictor
            Console.WriteLine($"{tag} stage:{stageTag} fold:{foldIdx} Training...");

            var stageConfig  = config.Stages[stageTag];
            var stageCols    = stageConfig.Features.Concat(new[] { stageConfig.Target }).ToArray();
            var stageTrainFrame = trainFrame.SelectColumns(stageCols);
            var stageTestFrame  = testFrame.SelectColumns(stageCols);
            var adapter      = ResolveAdapter(stageConfig.Framework);
            var model        = await adapter.TrainAsync(stageConfig, stageTrainFrame, null, ct);
            var preds        = await model.PredictAsync(stageTestFrame, ct);

            float[] obs = stageTestFrame.GetColumn(stageConfig.Target);
            var fm = _metricsEvaluator.EvaluateFold(foldIdx, obs, preds.Values, dbmBaselineMae);
            foldMetricsList.Add(fm);

            Console.WriteLine(
                $"{tag} stage:{stageTag} fold:{foldIdx} MAE={fm.Mae:F3} RMSE={fm.Rmse:F3}");

            (model as IDisposable)?.Dispose();
        }

        var aggregated = _metricsEvaluator.AggregateFolds(foldMetricsList);

        // NNLS weight optimization on calibration fold for ensemble hypotheses (RULE-164, RULE-165)
        IReadOnlyDictionary<string, float>? optimizedWeights = null;
        if (IsEnsembleHypothesis(hypothesis.ComposeExpression) && hypothesis.Stages.Count >= 2)
        {
            optimizedWeights = await OptimizeEnsembleWeightsAsync(
                hypothesis, config, calibFold, sweepId, ct);
        }

        Console.WriteLine($"{tag} Done. MeanMAE={aggregated.MaeMean:F3}±{aggregated.MaeStd:F3}");

        return new HypothesisResult(
            hypothesis.Id,
            hypothesis.ComposeExpression,
            foldMetricsList,
            aggregated,
            optimizedWeights);
    }

    private async Task<IReadOnlyDictionary<string, float>> OptimizeEnsembleWeightsAsync(
        SweepHypothesis hypothesis,
        SweepRunConfig config,
        (IDataFrame Train, IDataFrame Test) calibFold,
        string sweepId,
        CancellationToken ct)
    {
        var memberPredictions = new Dictionary<string, float[]>(StringComparer.OrdinalIgnoreCase);

        foreach (var stageName in hypothesis.Stages)
        {
            var stageConfig   = config.Stages[stageName];
            var stageCols     = stageConfig.Features.Concat(new[] { stageConfig.Target }).ToArray();
            var calibTrain    = calibFold.Train.SelectColumns(stageCols);
            var calibTest     = calibFold.Test.SelectColumns(stageCols);
            var adapter       = ResolveAdapter(stageConfig.Framework);
            var model         = await adapter.TrainAsync(stageConfig, calibTrain, null, ct);
            var preds         = await model.PredictAsync(calibTest, ct);
            memberPredictions[stageName] = preds.Values;
            (model as IDisposable)?.Dispose();
        }

        var lastStage  = config.Stages[hypothesis.Stages[^1]];
        var lastCols   = lastStage.Features.Concat(new[] { lastStage.Target }).ToArray();
        float[] observed = calibFold.Test.SelectColumns(lastCols).GetColumn(lastStage.Target);

        var weights = _nnlsOptimizer.Optimize(memberPredictions, observed);

        // RULE-165: log weights (not silently applied)
        var weightsStr = string.Join(", ", weights.Select(kvp => $"{kvp.Key}={kvp.Value:F4}"));
        Console.WriteLine($"[sweep:{hypothesis.Id} stage:ensemble] weights={{{weightsStr}}}");

        return weights;
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    private IFrameworkAdapter ResolveAdapter(string framework)
    {
        var adapter = _adapters.FirstOrDefault(a =>
            string.Equals(a.FrameworkType.ToString(), framework, StringComparison.OrdinalIgnoreCase));

        if (adapter is null)
            throw new InvalidOperationException(
                $"ModelSweep: no adapter for framework='{framework}'. " +
                $"Registered: [{string.Join(", ", _adapters.Select(a => a.FrameworkType))}].");

        return adapter;
    }

    private static bool IsEnsembleHypothesis(string compose) =>
        compose.Contains('+') && !compose.Contains('^') && !compose.Contains('→');

    // Deterministic sweep ID from config content hash (RULE-162)
    public static string ComputeSweepId(SweepRunConfig config)
    {
        var json = JsonSerializer.Serialize(config);
        var hash = SHA256.HashData(Encoding.UTF8.GetBytes(json));
        return "sweep_" + Convert.ToHexString(hash)[..12].ToLowerInvariant();
    }

    private static IDataFrame SelectRows(IDataFrame source, int[] indices)
    {
        var schema = source.Schema;
        var columns = new float[schema.Columns.Count][];
        for (int c = 0; c < schema.Columns.Count; c++)
        {
            float[] src = source.GetColumn(c);
            var col = new float[indices.Length];
            for (int r = 0; r < indices.Length; r++)
                col[r] = src[indices[r]];
            columns[c] = col;
        }
        return new InMemoryDataFrame(schema, columns);
    }

    private static bool CanWrite(string path)
    {
        try
        {
            Directory.CreateDirectory(path);
            var probe = Path.Combine(path, $".preflight_{Guid.NewGuid():N}");
            File.WriteAllText(probe, "ok");
            File.Delete(probe);
            return true;
        }
        catch { return false; }
    }
}

// ── Result types ──────────────────────────────────────────────────────────────

public sealed record PreFlightFailure(string Component, string Message);

public sealed class PreFlightResult
{
    public bool IsSuccess { get; }
    public bool HasFailures => !IsSuccess;
    public IReadOnlyList<PreFlightFailure> Failures { get; }

    private PreFlightResult(bool success, IReadOnlyList<PreFlightFailure> failures)
    {
        IsSuccess = success;
        Failures  = failures;
    }

    public static PreFlightResult Success() =>
        new(true, Array.Empty<PreFlightFailure>());

    public static PreFlightResult Fail(IReadOnlyList<PreFlightFailure> failures) =>
        new(false, failures);
}

public sealed record HypothesisResult(
    string HypothesisId,
    string ComposeExpression,
    IReadOnlyList<FoldMetrics> FoldMetrics,
    AggregatedMetrics AggregatedMetrics,
    IReadOnlyDictionary<string, float>? OptimizedWeights);

public sealed record SweepResult(
    string SweepId,
    string SweepName,
    IReadOnlyList<HypothesisResult> Hypotheses)
{
    public HypothesisResult Winner => Hypotheses[0];  // sorted by MaeMean ascending
}
