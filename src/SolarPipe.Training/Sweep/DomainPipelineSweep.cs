using SolarPipe.Core.Interfaces;
using SolarPipe.Core.Models;
using SolarPipe.Data.DataFrame;
using SolarPipe.Training.Features;

namespace SolarPipe.Training.Sweep;

// Outer CV loop for the three-domain CME prediction pipeline (Phase 8).
//
// Per fold: DomainFoldExecutor trains D1→D2→D3→meta sequentially then evaluates.
// Folds use expanding-window CV with gap buffer to prevent data leakage (same as ModelSweep).
// The final fold is both a training and evaluation fold (no calibration-only fold here,
// since Phase 8 has no NNLS ensemble to calibrate on).
//
// Physics baseline (DragBased-only MAE) is reported per fold for comparison against
// Phase 7 H1 (MAE=20.26h) and H3 (physics-only ensemble, MAE=191.7h).
public sealed class DomainPipelineSweep
{
    private readonly IReadOnlyList<IFrameworkAdapter> _adapters;

    public DomainPipelineSweep(IReadOnlyList<IFrameworkAdapter> adapters)
    {
        _adapters = adapters ?? throw new ArgumentNullException(nameof(adapters));
    }

    public async Task<DomainPipelineResult> RunAsync(
        DomainPipelineRunConfig config,
        IDataFrame data,
        CancellationToken ct)
    {
        Console.WriteLine($"[domain_sweep] Starting '{config.Name}' ({config.Folds} folds, {data.RowCount} rows)");

        var folds = config.HeldOutAfter is not null
            ? BuildHeldOutFold(data, config.HeldOutAfter, config.GapBufferDays)
            : BuildFolds(data, config.Folds, config.GapBufferDays, config.MinTestEvents);

        var executor = new DomainFoldExecutor(_adapters);
        var results  = new List<DomainFoldMetrics>(folds.Count);

        for (int k = 0; k < folds.Count; k++)
        {
            ct.ThrowIfCancellationRequested();

            var (trainFrame, testFrame) = folds[k];

            // Enrich both frames with missingness indicators and all static derived features.
            var enricher = new MissingnessFeatureEnricher();
            trainFrame = enricher.Enrich(trainFrame);
            testFrame  = enricher.Enrich(testFrame);

            trainFrame = DomainFeatureTransforms.AddOriginationFeatures(trainFrame);
            testFrame  = DomainFeatureTransforms.AddOriginationFeatures(testFrame);

            trainFrame = DomainFeatureTransforms.AddStormDuration(trainFrame);
            testFrame  = DomainFeatureTransforms.AddStormDuration(testFrame);

            var foldResult = await executor.ExecuteAsync(k, trainFrame, testFrame, config, ct);
            results.Add(foldResult);
        }

        double meanTransit  = results.Average(r => r.MaeTransit);
        double meanDst      = results.Average(r => r.MaeDst);
        double meanDuration = results.Average(r => r.MaeDuration);
        double meanBaseline = results.Average(r => r.MaePhysicsBaseline);

        Console.WriteLine(
            $"[domain_sweep] Complete. " +
            $"meta transit_mae={meanTransit:F3}h dst_mae={meanDst:F3}nT duration_mae={meanDuration:F3}h " +
            $"physics_baseline_mae={meanBaseline:F3}h");

        return new DomainPipelineResult(
            config.Name, results,
            meanTransit, meanDst, meanDuration, meanBaseline);
    }

    // Single train/test split at a given UTC date — all rows strictly before the date are training,
    // rows on/after (minus gap buffer) are test. Returns a single-element fold list.
    private static IReadOnlyList<(IDataFrame Train, IDataFrame Test)> BuildHeldOutFold(
        IDataFrame data, string heldOutAfter, int gapBufferDays)
    {
        var splitDate = DateTimeOffset.Parse(heldOutAfter, null, System.Globalization.DateTimeStyles.RoundtripKind);
        float splitTs  = (float)splitDate.ToUnixTimeSeconds();
        float gap      = (float)TimeSpan.FromDays(gapBufferDays).TotalSeconds;

        var tsCol = data.Schema.Columns.FirstOrDefault(c => c.Type == ColumnType.DateTime)
                 ?? data.Schema.Columns.FirstOrDefault(c =>
                        c.Name.Equals("launch_time", StringComparison.OrdinalIgnoreCase))
                 ?? throw new InvalidOperationException(
                        "DomainPipelineSweep: data must have a DateTime or launch_time column.");

        float[] ts = data.GetColumn(tsCol.Name);
        int n = ts.Length;

        int[] trainIdx = Enumerable.Range(0, n).Where(i => ts[i] < splitTs - gap).ToArray();
        int[] testIdx  = Enumerable.Range(0, n).Where(i => ts[i] >= splitTs).ToArray();

        if (trainIdx.Length == 0)
            throw new InvalidOperationException(
                $"DomainPipelineSweep: held-out split at {heldOutAfter} produced an empty training set.");
        if (testIdx.Length == 0)
            throw new InvalidOperationException(
                $"DomainPipelineSweep: held-out split at {heldOutAfter} produced an empty test set.");

        Console.WriteLine(
            $"[domain_sweep] held-out split: {trainIdx.Length} train, {testIdx.Length} test (split={heldOutAfter})");

        return [(SelectRows(data, trainIdx), SelectRows(data, testIdx))];
    }

    // Expanding-window CV fold builder — same logic as ModelSweep.BuildFolds.
    private static IReadOnlyList<(IDataFrame Train, IDataFrame Test)> BuildFolds(
        IDataFrame data, int totalFolds, int gapBufferDays, int minTestEvents)
    {
        var tsCol = data.Schema.Columns.FirstOrDefault(c => c.Type == ColumnType.DateTime)
                 ?? data.Schema.Columns.FirstOrDefault(c =>
                        c.Name.Equals("launch_time", StringComparison.OrdinalIgnoreCase));

        if (tsCol is null)
            throw new InvalidOperationException(
                "DomainPipelineSweep: data must have a DateTime or launch_time column.");

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
                    $"DomainPipelineSweep fold {k}: training set is empty after gap buffer.");

            if (testIdx.Length < minTestEvents)
            {
                Console.WriteLine(
                    $"[domain_sweep] fold {k}: only {testIdx.Length} test events " +
                    $"(min={minTestEvents}); skipping fold.");
                continue;
            }

            folds.Add((SelectRows(data, trainIdx), SelectRows(data, testIdx)));
        }

        if (folds.Count == 0)
            throw new InvalidOperationException(
                "DomainPipelineSweep: no valid folds were built. Check min_test_events and data span.");

        return folds;
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
}
