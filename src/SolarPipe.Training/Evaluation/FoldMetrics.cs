namespace SolarPipe.Training.Evaluation;

// Per-fold result from ComprehensiveMetricsEvaluator (RULE-163: computed independently per fold).
public sealed record FoldMetrics(
    int FoldIndex,
    double Mae,
    double Rmse,
    double R2,
    double Bias,
    double SkillVsDbm,
    double HitRate6h,
    double HitRate12h,
    double PinballLoss10,
    double CoverageRate90,
    double KendallTau);

// Aggregated result across all folds (mean ± std per metric).
public sealed record AggregatedMetrics(
    double MaeMean,   double MaeStd,
    double RmseMean,  double RmseStd,
    double R2Mean,    double R2Std,
    double BiasMean,  double BiasStd,
    double SkillMean, double SkillStd,
    double HitRate6hMean,  double HitRate6hStd,
    double HitRate12hMean, double HitRate12hStd,
    double PinballMean,    double PinballStd,
    double CoverageMean,   double CoverageStd,
    double KendallMean,    double KendallStd);
