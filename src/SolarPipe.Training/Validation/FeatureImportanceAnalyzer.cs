using SolarPipe.Core.Interfaces;
using SolarPipe.Core.Models;
using SolarPipe.Data.DataFrame;

namespace SolarPipe.Training.Validation;

public sealed record FeatureImportanceResult(
    string FeatureName,
    double MeanImportance,
    double StdImportance);

// Model-agnostic permutation feature importance.
//
// Algorithm: for each feature, randomly permute its values and measure
// the increase in RMSE. Large RMSE increase → feature is important.
// This avoids direct access to model internals (which ML.NET FastForest hides).
//
// Stability analysis: run multiple permutations per feature to estimate variance.
public sealed class FeatureImportanceAnalyzer
{
    private readonly int _permutations;
    private readonly int _seed;

    // permutations: number of shuffle repeats per feature (default 5 for stability)
    public FeatureImportanceAnalyzer(int permutations = 5, int seed = 42)
    {
        if (permutations < 1)
            throw new ArgumentOutOfRangeException(nameof(permutations),
                "permutations must be ≥ 1.");
        _permutations = permutations;
        _seed = seed;
    }

    // Compute permutation feature importance for the given model on test data.
    // Returns results sorted by mean importance descending (most important first).
    public async Task<IReadOnlyList<FeatureImportanceResult>> ComputeAsync(
        ITrainedModel model,
        IDataFrame testData,
        string targetColumn,
        IReadOnlyList<string> featureColumns,
        CancellationToken ct)
    {
        if (testData.RowCount == 0)
            throw new ArgumentException(
                $"FeatureImportanceAnalyzer: test data is empty. " +
                $"Model: {model.ModelId}, stage: {model.StageName}.");

        if (!testData.Schema.HasColumn(targetColumn))
            throw new ArgumentException(
                $"FeatureImportanceAnalyzer: target column '{targetColumn}' not found. " +
                $"Model: {model.ModelId}.");

        float[] actuals = testData.GetColumn(targetColumn);

        // Baseline RMSE with original (unshuffled) data
        PredictionResult baseline = await model.PredictAsync(testData, ct);
        double baselineRmse = ComputeRmse(actuals, baseline.Values);

        var results = new List<FeatureImportanceResult>();
        var rng = new Random(_seed);

        foreach (string feature in featureColumns)
        {
            ct.ThrowIfCancellationRequested();

            if (!testData.Schema.HasColumn(feature))
                continue;

            float[] original = testData.GetColumn(feature);
            var shuffledRmses = new double[_permutations];

            for (int p = 0; p < _permutations; p++)
            {
                float[] permuted = ShuffleColumn(original, rng);
                IDataFrame replaced = ReplaceColumn(testData, feature, permuted);
                PredictionResult preds = await model.PredictAsync(replaced, ct);
                shuffledRmses[p] = ComputeRmse(actuals, preds.Values);
                replaced.Dispose();
            }

            double[] importances = shuffledRmses
                .Select(r => r - baselineRmse)
                .ToArray();

            double mean = importances.Average();
            double std  = Std(importances);

            results.Add(new FeatureImportanceResult(feature, mean, std));
        }

        return results.OrderByDescending(r => r.MeanImportance).ToList();
    }

    private static float[] ShuffleColumn(float[] original, Random rng)
    {
        float[] copy = (float[])original.Clone();
        for (int i = copy.Length - 1; i > 0; i--)
        {
            int j = rng.Next(i + 1);
            (copy[i], copy[j]) = (copy[j], copy[i]);
        }
        return copy;
    }

    private static IDataFrame ReplaceColumn(IDataFrame source, string columnName, float[] newValues)
    {
        var schema = source.Schema;
        var columns = new float[schema.Columns.Count][];
        for (int c = 0; c < schema.Columns.Count; c++)
        {
            columns[c] = schema.Columns[c].Name.Equals(columnName, StringComparison.OrdinalIgnoreCase)
                ? newValues
                : source.GetColumn(c);
        }
        return new InMemoryDataFrame(schema, columns);
    }

    private static double ComputeRmse(float[] actuals, float[] predicted)
    {
        if (actuals.Length == 0) return double.NaN;
        double ss = 0;
        for (int i = 0; i < actuals.Length; i++)
        {
            double diff = actuals[i] - predicted[i];
            ss += diff * diff;
        }
        return Math.Sqrt(ss / actuals.Length);
    }

    private static double Std(double[] values)
    {
        double mean = values.Average();
        double variance = values.Average(v => (v - mean) * (v - mean));
        return Math.Sqrt(variance);
    }
}
