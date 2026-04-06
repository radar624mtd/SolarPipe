using SolarPipe.Core.Interfaces;
using SolarPipe.Core.Models;
using SolarPipe.Data.DataFrame;

namespace SolarPipe.Training.Validation;

// Standard k-fold cross-validation.
//
// CAVEAT (RULE-051): Do NOT use this for time-series CME event data.
// Random k-fold leaks future information into training folds for temporal data.
// Use ExpandingWindowCV for temporal data; only use KFoldCV for:
//   - Non-temporal feature importance analysis
//   - Cross-sectional (non-ordered) datasets
//   - Baseline comparisons where temporal leakage is documented and accepted
public sealed class KFoldCV
{
    private readonly int _folds;
    private readonly int _seed;

    public KFoldCV(int folds = 5, int seed = 42)
    {
        if (folds < 2)
            throw new ArgumentOutOfRangeException(nameof(folds), "At least 2 folds required.");
        _folds = folds;
        _seed = seed;
    }

    public async Task<CrossValidationResult> RunAsync(
        IDataFrame data,
        IFrameworkAdapter adapter,
        StageConfig stage,
        CancellationToken ct)
    {
        if (data.RowCount == 0)
            throw new ArgumentException(
                $"KFoldCV: data has 0 rows. Stage: {stage.Name}.", nameof(data));

        if (data.RowCount < _folds)
            throw new ArgumentException(
                $"KFoldCV: data has {data.RowCount} rows but {_folds} folds requested. " +
                $"Stage: {stage.Name}.", nameof(data));

        // Shuffle row indices with fixed seed for reproducibility
        var rng = new Random(_seed);
        int[] shuffled = Enumerable.Range(0, data.RowCount).ToArray();
        for (int i = shuffled.Length - 1; i > 0; i--)
        {
            int j = rng.Next(i + 1);
            (shuffled[i], shuffled[j]) = (shuffled[j], shuffled[i]);
        }

        var foldResults = new List<CvFoldResult>();

        for (int k = 0; k < _folds; k++)
        {
            ct.ThrowIfCancellationRequested();

            int testStart = k * data.RowCount / _folds;
            int testEnd   = (k + 1) * data.RowCount / _folds;

            var testIndices  = shuffled[testStart..testEnd];
            var trainIndices = shuffled[..testStart].Concat(shuffled[testEnd..]).ToArray();

            IDataFrame trainFrame = SelectRows(data, trainIndices);
            IDataFrame testFrame  = SelectRows(data, testIndices);

            ITrainedModel model = await adapter.TrainAsync(stage, trainFrame, null, ct);
            PredictionResult preds = await model.PredictAsync(testFrame, ct);

            ModelMetrics metrics = ComputeMetrics(
                testFrame.GetColumn(stage.Target),
                preds.Values);

            foldResults.Add(new CvFoldResult(k, trainIndices.Length, testIndices.Length, metrics));

            (model as IDisposable)?.Dispose();
            trainFrame.Dispose();
            testFrame.Dispose();
        }

        return CrossValidationResult.Aggregate(foldResults);
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
