using SolarPipe.Core.Interfaces;
using SolarPipe.Core.Models;
using SolarPipe.Data.DataFrame;

namespace SolarPipe.Training.Validation;

// RULE-051: Expanding-window temporal CV with gap buffer.
// Never use random k-fold for time-series CME event data.
//
// Algorithm:
//   Sort events by timestamp.
//   Divide into (folds+1) segments: train_1 < gap < test_1, train_1+2 < gap < test_2, ...
//   Each fold's training set expands: fold k trains on events [0..cut_k].
//   Gap buffer removes events within gapBuffer of the test period boundary.
//   Validates ≥50 events per test fold when enforceMinTestEvents is true.
public sealed class ExpandingWindowCV
{
    private readonly int _folds;
    private readonly TimeSpan _gapBuffer;
    private readonly int _minTestEvents;
    private readonly bool _enforceMinTestEvents;

    // folds: number of CV folds (default 5 per RULE-051)
    // gapBuffer: events within this window of the test period are purged from training
    //            (default 5 days — CME transit time per RULE-051)
    // minTestEvents: minimum events required per test fold (default 50 per RULE-051)
    public ExpandingWindowCV(
        int folds = 5,
        TimeSpan? gapBuffer = null,
        int minTestEvents = 50,
        bool enforceMinTestEvents = true)
    {
        if (folds < 2)
            throw new ArgumentOutOfRangeException(nameof(folds), "At least 2 folds required.");
        if (minTestEvents < 1)
            throw new ArgumentOutOfRangeException(nameof(minTestEvents), "minTestEvents must be ≥1.");

        _folds = folds;
        _gapBuffer = gapBuffer ?? TimeSpan.FromDays(5);
        _minTestEvents = minTestEvents;
        _enforceMinTestEvents = enforceMinTestEvents;
    }

    // Run expanding-window CV.
    // data: full dataset, must contain a timestamp column (ColumnType.DateTime or named "timestamp")
    // timestampValues: parallel float[] of Unix seconds for each row (converted from datetime column)
    //   Pass null to use the frame's DateTime column; provide explicitly for synthetic timestamps.
    public async Task<CrossValidationResult> RunAsync(
        IDataFrame data,
        IFrameworkAdapter adapter,
        StageConfig stage,
        float[]? timestampValues,
        CancellationToken ct)
    {
        if (data.RowCount == 0)
            throw new ArgumentException(
                $"ExpandingWindowCV: data has 0 rows. Stage: {stage.Name}.", nameof(data));

        float[] ts = timestampValues ?? ExtractTimestamps(data, stage.Name);

        // Sort indices by timestamp
        int[] sortedIndices = Enumerable.Range(0, data.RowCount)
            .OrderBy(i => ts[i])
            .ToArray();

        float[] sortedTs = sortedIndices.Select(i => ts[i]).ToArray();

        // Compute fold cut points on sorted data
        // Folds: [0..cut0], [0..cut1], ..., [0..cut(n-2)] train; test on cut(k)+gap .. cut(k+1)
        int n = sortedTs.Length;
        var cutPoints = ComputeCutPoints(sortedTs, _folds);

        var foldResults = new List<CvFoldResult>();

        for (int k = 0; k < _folds; k++)
        {
            ct.ThrowIfCancellationRequested();

            float testStart = sortedTs[cutPoints[k]];
            float testEnd   = k + 1 < cutPoints.Length ? sortedTs[cutPoints[k + 1] - 1] : sortedTs[n - 1];
            float gapSeconds = (float)_gapBuffer.TotalSeconds;

            // Training: all events strictly before (testStart - gap)
            var trainIndices = sortedIndices
                .Where(i => ts[i] < testStart - gapSeconds)
                .ToArray();

            // Test: events in [testStart .. testEnd]
            var testIndices = sortedIndices
                .Where(i => ts[i] >= testStart && ts[i] <= testEnd)
                .ToArray();

            if (_enforceMinTestEvents && testIndices.Length < _minTestEvents)
                throw new InvalidOperationException(
                    $"ExpandingWindowCV fold {k}: test set has {testIndices.Length} events, " +
                    $"minimum {_minTestEvents} required (RULE-051). " +
                    $"Stage: {stage.Name}. Increase dataset size or reduce fold count.");

            if (trainIndices.Length == 0)
                throw new InvalidOperationException(
                    $"ExpandingWindowCV fold {k}: training set is empty after gap buffer applied. " +
                    $"Stage: {stage.Name}, gapBuffer: {_gapBuffer}.");

            IDataFrame trainFrame = SelectRows(data, trainIndices);
            IDataFrame testFrame  = SelectRows(data, testIndices);

            ITrainedModel model = await adapter.TrainAsync(stage, trainFrame, null, ct);
            PredictionResult preds = await model.PredictAsync(testFrame, ct);

            ModelMetrics metrics = ComputeMetrics(
                testFrame.GetColumn(stage.Target),
                preds.Values);

            foldResults.Add(new CvFoldResult(k, trainIndices.Length, testIndices.Length, metrics));

            trainFrame.Dispose();
            testFrame.Dispose();
        }

        return CrossValidationResult.Aggregate(foldResults);
    }

    // Cut points divide the sorted array into (folds+1) segments.
    // Returns indices of the start of each test segment (length == folds).
    private static int[] ComputeCutPoints(float[] sortedTs, int folds)
    {
        int n = sortedTs.Length;
        float tMin = sortedTs[0];
        float tMax = sortedTs[n - 1];
        float span = tMax - tMin;

        // Place cut points at equal time intervals (not equal count, preserving temporal structure)
        var cuts = new int[folds];
        for (int k = 0; k < folds; k++)
        {
            // Test segment k starts at fraction (k+1)/(folds+1) of the timeline
            float cutTime = tMin + span * (k + 1f) / (folds + 1f);
            // Find first index with ts >= cutTime
            int idx = Array.BinarySearch(sortedTs, cutTime);
            if (idx < 0) idx = ~idx;
            cuts[k] = Math.Clamp(idx, 1, n - 1);
        }
        return cuts;
    }

    // Extract timestamp column as float[] of Unix seconds
    private static float[] ExtractTimestamps(IDataFrame data, string stageName)
    {
        // Look for ColumnType.DateTime first, then case-insensitive "timestamp"
        var tsCol = data.Schema.Columns.FirstOrDefault(c => c.Type == ColumnType.DateTime)
                 ?? data.Schema.Columns.FirstOrDefault(c =>
                        c.Name.Equals("timestamp", StringComparison.OrdinalIgnoreCase));

        if (tsCol is null)
            throw new InvalidOperationException(
                $"ExpandingWindowCV: no timestamp column found in schema. " +
                $"Stage: {stageName}. Columns: [{string.Join(", ", data.Schema.Columns.Select(c => c.Name))}]");

        return data.GetColumn(tsCol.Name);
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

    private static ModelMetrics ComputeMetrics(float[] actuals, float[] predictions)
    {
        if (actuals.Length == 0)
            return new ModelMetrics(double.NaN, double.NaN, double.NaN);

        double n = actuals.Length;
        double ssRes = 0, sseTot = 0, sumAbs = 0;
        double mean = actuals.Average(a => a);

        for (int i = 0; i < actuals.Length; i++)
        {
            double diff = actuals[i] - predictions[i];
            ssRes  += diff * diff;
            sumAbs += Math.Abs(diff);
            double devFromMean = actuals[i] - mean;
            sseTot += devFromMean * devFromMean;
        }

        double rmse = Math.Sqrt(ssRes / n);
        double mae  = sumAbs / n;
        double r2   = sseTot < 1e-12 ? 0.0 : 1.0 - ssRes / sseTot;

        return new ModelMetrics(rmse, mae, r2);
    }
}
