using System.Security.Cryptography;
using System.Text;
using SolarPipe.Config;
using SolarPipe.Core.Interfaces;
using SolarPipe.Core.Models;
using SolarPipe.Data;
using SolarPipe.Training.Checkpoint;

namespace SolarPipe.Host.Commands;

public sealed class TrainCommand : ICommand
{
    private readonly PipelineConfigLoader _loader;
    private readonly DataSourceRegistry _dataRegistry;
    private readonly IReadOnlyList<IFrameworkAdapter> _adapters;
    private readonly IModelRegistry _modelRegistry;
    private readonly CheckpointManager _checkpoints;

    public TrainCommand(
        PipelineConfigLoader loader,
        DataSourceRegistry dataRegistry,
        IReadOnlyList<IFrameworkAdapter> adapters,
        IModelRegistry modelRegistry,
        CheckpointManager checkpoints)
    {
        _loader = loader;
        _dataRegistry = dataRegistry;
        _adapters = adapters;
        _modelRegistry = modelRegistry;
        _checkpoints = checkpoints;
    }

    public async Task<int> ExecuteAsync(string[] args, CancellationToken ct)
    {
        string configPath;
        try { configPath = ArgParser.Require(args, "--config"); }
        catch (ArgumentException ex)
        {
            Console.Error.WriteLine($"TRAIN_ERROR type=MissingArguments message=\"{ex.Message}\"");
            return ExitCodes.MissingArguments;
        }

        var stageFilter = ArgParser.Get(args, "--stage");
        var resumeFrom = ArgParser.Get(args, "--resume-from-stage");
        bool noCache = Array.Exists(args, a => a.Equals("--no-cache", StringComparison.OrdinalIgnoreCase));

        try
        {
            var config = await _loader.LoadAsync(configPath, ct);

            foreach (var (key, _) in config.DataSources)
                _dataRegistry.RegisterSource(config.ToDataSourceConfig(key));

            if (noCache)
            {
                await _checkpoints.ClearAsync(config.Name, ct);
                Console.WriteLine($"CHECKPOINT cleared pipeline={config.Name}");
            }

            bool skipping = resumeFrom is not null;

            // Track trained models per stage so residual stages can reference them
            var trainedModels = new Dictionary<string, ITrainedModel>(StringComparer.OrdinalIgnoreCase);

            foreach (var (stageName, stageYaml) in config.Stages)
            {
                // --stage filter (single-stage run)
                if (stageFilter is not null
                    && !string.Equals(stageName, stageFilter, StringComparison.OrdinalIgnoreCase))
                    continue;

                // --resume-from-stage: skip stages before the named stage
                if (skipping)
                {
                    if (string.Equals(stageName, resumeFrom, StringComparison.OrdinalIgnoreCase))
                        skipping = false;
                    else
                    {
                        Console.WriteLine($"SKIP stage={stageName} reason=resuming_from={resumeFrom}");
                        continue;
                    }
                }

                var stageConfig = config.ToStageConfig(stageName);
                var adapter = ResolveAdapter(stageConfig);

                // Load only the columns needed for this stage (features + target).
                var dsOptions = config.DataSources[stageYaml.DataSource].Options;
                var data = await _dataRegistry.LoadAsync(
                    stageYaml.DataSource,
                    new DataQuery(Sql: BuildStageSql(stageConfig, dsOptions)),
                    ct);
                string inputFingerprint = ComputeInputFingerprint(data);

                // Check for valid checkpoint
                var cached = await _checkpoints.TryReadAsync(
                    config.Name, stageName, stageConfig, inputFingerprint, ct);

                if (cached is not null)
                {
                    Console.WriteLine(
                        $"CHECKPOINT_HIT stage={stageName} rows={cached.RowCount}");
                    cached.Dispose();
                    continue;
                }

                ITrainedModel model;
                using var stageCts = CancellationTokenSource.CreateLinkedTokenSource(ct);
                int timeoutMinutes = stageConfig.Hyperparameters is not null
                    && stageConfig.Hyperparameters.TryGetValue("timeout_minutes", out var tmObj)
                    && int.TryParse(tmObj?.ToString(), out var tm) ? tm : 30;
                stageCts.CancelAfter(TimeSpan.FromMinutes(timeoutMinutes)); // RULE-151

                // If this stage uses residual_calibration, find the preceding physics model,
                // run it on the training data, compute residuals, and train on those residuals.
                // This is the critical wiring: the ^ composition operator only works correctly
                // if the correction model was trained to predict (observed - baseline), not raw target.
                bool isResidualCalibration = string.Equals(
                    stageConfig.MockDataStrategy, "residual_calibration",
                    StringComparison.OrdinalIgnoreCase);

                if (isResidualCalibration && trainedModels.Count > 0)
                {
                    // Find the most recently trained physics baseline
                    var baselineEntry = trainedModels
                        .LastOrDefault(kvp => string.Equals(
                            config.Stages[kvp.Key].Framework, "Physics",
                            StringComparison.OrdinalIgnoreCase));

                    if (baselineEntry.Value is not null)
                    {
                        var (train, validation) = SplitTrainValidation(data, trainFraction: 0.8);

                        try
                        {
                            model = await TrainWithResidualCalibrationAsync(
                                stageConfig, adapter, baselineEntry.Value,
                                train, validation, stageCts.Token);
                        }
                        catch (OperationCanceledException) when (!ct.IsCancellationRequested)
                        {
                            Console.Error.WriteLine(
                                $"TRAIN_ERROR stage={stageName} type=Timeout " +
                                $"message=\"Stage exceeded 30 minutes.\"");
                            return ExitCodes.TrainingFailed;
                        }

                        Console.WriteLine(
                            $"OK stage={stageName} model_id={model.ModelId} " +
                            $"rmse={model.Metrics.Rmse:F4} mode=residual_calibration " +
                            $"baseline={baselineEntry.Key}");
                    }
                    else
                    {
                        // No physics baseline found — fall back to direct training and warn
                        Console.Error.WriteLine(
                            $"TRAIN_WARN stage={stageName} message=\"mock_data_strategy=residual_calibration " +
                            $"but no preceding Physics stage found; training on raw target instead\"");

                        var (train, validation) = SplitTrainValidation(data, trainFraction: 0.8);
                        try
                        {
                            model = await adapter.TrainAsync(stageConfig, train, validation, stageCts.Token);
                        }
                        catch (OperationCanceledException) when (!ct.IsCancellationRequested)
                        {
                            Console.Error.WriteLine(
                                $"TRAIN_ERROR stage={stageName} type=Timeout " +
                                $"message=\"Stage exceeded 30 minutes.\"");
                            return ExitCodes.TrainingFailed;
                        }
                        Console.WriteLine(
                            $"OK stage={stageName} model_id={model.ModelId} rmse={model.Metrics.Rmse:F4}");
                    }
                }
                else
                {
                    var (train, validation) = SplitTrainValidation(data, trainFraction: 0.8);
                    try
                    {
                        model = await adapter.TrainAsync(stageConfig, train, validation, stageCts.Token);
                    }
                    catch (OperationCanceledException) when (!ct.IsCancellationRequested)
                    {
                        Console.Error.WriteLine(
                            $"TRAIN_ERROR stage={stageName} type=Timeout " +
                            $"message=\"Stage exceeded 30 minutes. Check logs/dotnet_latest.json for last activity.\"");
                        return ExitCodes.TrainingFailed;
                    }
                    Console.WriteLine(
                        $"OK stage={stageName} model_id={model.ModelId} rmse={model.Metrics.Rmse:F4}");
                }

                trainedModels[stageName] = model;

                var artifact = new ModelArtifact
                {
                    ModelId = model.ModelId,
                    Version = "1.0.0",
                    StageName = stageName,
                    Config = stageConfig,
                    Metrics = model.Metrics,
                    DataFingerprint = inputFingerprint,
                    TrainedAt = DateTime.UtcNow,
                    ArtifactPath = string.Empty,
                };

                await _modelRegistry.RegisterAsync(artifact, model, ct);

                await _checkpoints.WriteAsync(
                    config.Name, stageName, data, stageConfig, inputFingerprint, ct);
            }

            return ExitCodes.Success;
        }
        catch (Exception ex)
        {
            var root = ex.InnerException ?? ex;
            Console.Error.WriteLine(
                $"TRAIN_ERROR stage={stageFilter ?? "all"} type={root.GetType().Name} message=\"{root.Message}\"");
            return ExitCodes.TrainingFailed;
        }
    }

    // Run physics baseline on training data, compute residuals, train correction on residuals.
    // The RF correction model's target becomes (observed_transit - physics_transit).
    // At inference time ResidualModel computes: baseline_prediction + correction_prediction.
    private static async Task<ITrainedModel> TrainWithResidualCalibrationAsync(
        StageConfig stageConfig,
        IFrameworkAdapter adapter,
        ITrainedModel baselineModel,
        IDataFrame train,
        IDataFrame? validation,
        CancellationToken ct)
    {
        // Run physics baseline on full training data to get per-row physics predictions
        var baselinePredictions = await baselineModel.PredictAsync(train, ct);

        // Compute residuals: observed_transit - physics_transit
        var observed = train.GetColumn(stageConfig.Target);
        var physicsValues = baselinePredictions.Values;

        int n = Math.Min(observed.Length, physicsValues.Length);
        var residuals = new float[n];
        int nonNan = 0;
        for (int i = 0; i < n; i++)
        {
            if (float.IsNaN(observed[i]) || float.IsNaN(physicsValues[i]))
                residuals[i] = float.NaN;
            else
            {
                residuals[i] = observed[i] - physicsValues[i];
                nonNan++;
            }
        }

        Console.WriteLine(
            $"RESIDUAL_CALIBRATION stage={stageConfig.Name} " +
            $"rows={n} non_nan={nonNan} " +
            $"mean_residual={residuals.Where(v => !float.IsNaN(v)).DefaultIfEmpty(0f).Average():F2}h");

        // Build residual training frame: same features, target = __residual__
        // AddColumn appends the residual column; update stage target to point to it
        using var residualTrain = train.AddColumn("__residual__", residuals);
        var residualStage = stageConfig with { Target = "__residual__" };

        // Compute residuals for validation set too if present
        IDataFrame? residualValidation = null;
        if (validation is not null && validation.RowCount > 0)
        {
            var valBaseline = await baselineModel.PredictAsync(validation, ct);
            var valObserved = validation.GetColumn(stageConfig.Target);
            var valPhysics  = valBaseline.Values;
            int nv = Math.Min(valObserved.Length, valPhysics.Length);
            var valResiduals = new float[nv];
            for (int i = 0; i < nv; i++)
                valResiduals[i] = float.IsNaN(valObserved[i]) || float.IsNaN(valPhysics[i])
                    ? float.NaN
                    : valObserved[i] - valPhysics[i];
            residualValidation = validation.AddColumn("__residual__", valResiduals);
        }

        try
        {
            return await adapter.TrainAsync(residualStage, residualTrain, residualValidation, ct);
        }
        finally
        {
            residualValidation?.Dispose();
        }
    }

    private IFrameworkAdapter ResolveAdapter(StageConfig config)
    {
        // Normalize: strip underscores so YAML "python_grpc" matches enum "PythonGrpc".
        static string Normalize(string s) => s.Replace("_", "", StringComparison.Ordinal);
        var adapter = _adapters.FirstOrDefault(a =>
            string.Equals(Normalize(a.FrameworkType.ToString()), Normalize(config.Framework), StringComparison.OrdinalIgnoreCase));

        if (adapter is null)
            throw new InvalidOperationException(
                $"TrainCommand: no adapter for framework='{config.Framework}' stage='{config.Name}'. " +
                $"Registered: [{string.Join(", ", _adapters.Select(a => a.FrameworkType))}].");

        return adapter;
    }

    private static (IDataFrame Train, IDataFrame Validation) SplitTrainValidation(
        IDataFrame data, double trainFraction)
    {
        // Temporal split — no random k-fold (RULE-051)
        int trainRows = (int)(data.RowCount * trainFraction);
        return (data.Slice(0, trainRows), data.Slice(trainRows, data.RowCount - trainRows));
    }

    private static string BuildStageSql(
        StageConfig stage,
        IReadOnlyDictionary<string, string>? dsOptions)
    {
        string table = dsOptions is not null
            && dsOptions.TryGetValue("table", out var t)
            && !string.IsNullOrWhiteSpace(t)
            ? t
            : stage.DataSource;

        static string Q(string c) => $"\"{c.Replace("\"", "")}\"";

        // TftPinn stages need activity_id for sequence Parquet join in the sidecar.
        // Prepend it so it's always the first extra column (not counted as a model feature).
        IEnumerable<string> extraCols = stage.ModelType.Equals("TftPinn", StringComparison.OrdinalIgnoreCase)
            ? new[] { Q("activity_id") }
            : Enumerable.Empty<string>();

        var cols = extraCols.Concat(
            stage.Features
                .Append(stage.Target)
                .Distinct(StringComparer.OrdinalIgnoreCase)
                .Select(f => ExpandFeatureToSql(f, Q)));

        string whereClause = dsOptions is not null
            && dsOptions.TryGetValue("filter", out var f)
            && !string.IsNullOrWhiteSpace(f)
            ? $" WHERE {f}"
            : string.Empty;

        return $"SELECT {string.Join(", ", cols)} FROM {Q(table)}{whereClause}";
    }

    private static string ExpandFeatureToSql(string feature, Func<string, string> quote)
    {
        // Derived features computed at query time — keeps pipeline declarative
        if (string.Equals(feature, "delta_v_kms", StringComparison.OrdinalIgnoreCase))
            return $"({quote("cme_speed_kms")} - {quote("sw_speed_ambient_kms")}) AS {quote("delta_v_kms")}";

        if (string.Equals(feature, "speed_ratio", StringComparison.OrdinalIgnoreCase))
            return $"({quote("cme_speed_kms")} / NULLIF({quote("sw_speed_ambient_kms")}, 0)) AS {quote("speed_ratio")}";

        if (string.Equals(feature, "speed_x_bz", StringComparison.OrdinalIgnoreCase))
            return $"({quote("cme_speed_kms")} * {quote("bz_gsm_proxy_nt")}) AS {quote("speed_x_bz")}";

        if (string.Equals(feature, "speed_x_density", StringComparison.OrdinalIgnoreCase))
            return $"({quote("cme_speed_kms")} * {quote("sw_density_n_cc")}) AS {quote("speed_x_density")}";

        return quote(feature);
    }

    private static string ComputeInputFingerprint(IDataFrame data)
    {
        // Lightweight fingerprint: row count + first/last value of first column
        // Full hash would require reading all float[] arrays — too slow for 9K rows
        var sb = new StringBuilder();
        sb.Append(data.RowCount);
        if (data.RowCount > 0 && data.Schema.Columns.Count > 0)
        {
            var col = data.GetColumn(0);
            sb.Append('|');
            sb.Append(col[0].ToString("G6"));
            sb.Append('|');
            sb.Append(col[^1].ToString("G6"));
        }
        return Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(sb.ToString())));
    }
}
