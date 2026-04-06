using SolarPipe.Core.Interfaces;
using SolarPipe.Core.Models;
using SolarPipe.Data.DataFrame;

namespace SolarPipe.Training.Validation;

// Purged cross-validation (de Prado 2018).
// Removes training samples whose event windows overlap with test events,
// plus an embargo period after each test fold to prevent leakage through
// autocorrelated features (e.g. rolling statistics).
//
// WARNING: This is NOT appropriate for hyperparameter tuning on CME datasets
// (high variance). Use ExpandingWindowCV for tuning; use PurgedCV for final
// model comparison only.
public sealed class PurgedCV
{
    private readonly int _folds;
    private readonly TimeSpan _embargoBuffer;

    // folds: number of folds for purged CV (default 5)
    // embargoBuffer: time after each test fold's last event to exclude from next
    //   training fold (default 3 days — Dst recovery time per RULE-051)
    public PurgedCV(int folds = 5, TimeSpan? embargoBuffer = null)
    {
        if (folds < 2)
            throw new ArgumentOutOfRangeException(nameof(folds), "At least 2 folds required.");
        _folds = folds;
        _embargoBuffer = embargoBuffer ?? TimeSpan.FromDays(3);
    }

    public async Task<CrossValidationResult> RunAsync(
        IDataFrame data,
        IFrameworkAdapter adapter,
        StageConfig stage,
        float[]? timestampValues,
        CancellationToken ct)
    {
        if (data.RowCount == 0)
            throw new ArgumentException(
                $"PurgedCV: data has 0 rows. Stage: {stage.Name}.", nameof(data));

        float[] ts = timestampValues ?? ExtractTimestamps(data, stage.Name);
        float embargoSeconds = (float)_embargoBuffer.TotalSeconds;

        int[] sortedIndices = Enumerable.Range(0, data.RowCount)
            .OrderBy(i => ts[i])
            .ToArray();

        float[] sortedTs = sortedIndices.Select(i => ts[i]).ToArray();

        // Divide into _folds equal-time-span test segments
        int n = sortedTs.Length;
        float tMin = sortedTs[0];
        float tMax = sortedTs[n - 1];
        float span = tMax - tMin;

        var foldResults = new List<CvFoldResult>();

        for (int k = 0; k < _folds; k++)
        {
            ct.ThrowIfCancellationRequested();

            float testStart = tMin + span * k / _folds;
            float testEnd   = tMin + span * (k + 1f) / _folds;

            // Test indices: events in [testStart, testEnd]
            var testIndices = sortedIndices
                .Where(i => ts[i] >= testStart && ts[i] <= testEnd)
                .ToArray();

            if (testIndices.Length == 0)
                throw new InvalidOperationException(
                    $"PurgedCV fold {k}: test set is empty. Stage: {stage.Name}.");

            float embargoEnd = testEnd + embargoSeconds;

            // Purged training: exclude events overlapping test window and embargo period.
            // An event at time t overlaps the test window if t is within [testStart, embargoEnd].
            var trainIndices = sortedIndices
                .Where(i => ts[i] < testStart || ts[i] > embargoEnd)
                .ToArray();

            if (trainIndices.Length == 0)
                throw new InvalidOperationException(
                    $"PurgedCV fold {k}: training set empty after purge+embargo. " +
                    $"Stage: {stage.Name}, embargoBuffer: {_embargoBuffer}.");

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

    private static float[] ExtractTimestamps(IDataFrame data, string stageName)
    {
        var tsCol = data.Schema.Columns.FirstOrDefault(c => c.Type == ColumnType.DateTime)
                 ?? data.Schema.Columns.FirstOrDefault(c =>
                        c.Name.Equals("timestamp", StringComparison.OrdinalIgnoreCase));

        if (tsCol is null)
            throw new InvalidOperationException(
                $"PurgedCV: no timestamp column found. Stage: {stageName}. " +
                $"Columns: [{string.Join(", ", data.Schema.Columns.Select(c => c.Name))}]");

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
            double dev = actuals[i] - mean;
            sseTot += dev * dev;
        }

        double rmse = Math.Sqrt(ssRes / n);
        double mae  = sumAbs / n;
        double r2   = sseTot < 1e-12 ? 0.0 : 1.0 - ssRes / sseTot;

        return new ModelMetrics(rmse, mae, r2);
    }
}
