using SolarPipe.Core.Models;

namespace SolarPipe.Training.Validation;

// Result for a single CV fold
public sealed record CvFoldResult(
    int FoldIndex,
    int TrainRows,
    int TestRows,
    ModelMetrics Metrics);

// Aggregated result across all folds
public sealed record CrossValidationResult(
    IReadOnlyList<CvFoldResult> Folds,
    ModelMetrics MeanMetrics,
    ModelMetrics StdMetrics)
{
    public int FoldCount => Folds.Count;

    public static CrossValidationResult Aggregate(IReadOnlyList<CvFoldResult> folds)
    {
        if (folds.Count == 0)
            throw new ArgumentException("Cannot aggregate zero folds.", nameof(folds));

        double meanRmse = folds.Average(f => f.Metrics.Rmse);
        double meanMae  = folds.Average(f => f.Metrics.Mae);
        double meanR2   = folds.Average(f => f.Metrics.R2);

        double stdRmse = Std(folds.Select(f => f.Metrics.Rmse));
        double stdMae  = Std(folds.Select(f => f.Metrics.Mae));
        double stdR2   = Std(folds.Select(f => f.Metrics.R2));

        return new CrossValidationResult(
            folds,
            new ModelMetrics(meanRmse, meanMae, meanR2),
            new ModelMetrics(stdRmse, stdMae, stdR2));
    }

    private static double Std(IEnumerable<double> values)
    {
        var list = values.ToList();
        double mean = list.Average();
        double variance = list.Average(v => (v - mean) * (v - mean));
        return Math.Sqrt(variance);
    }
}
