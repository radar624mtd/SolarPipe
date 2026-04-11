namespace SolarPipe.Training.Evaluation;

// Computes 10-metric suite per CV fold (RULE-163).
// Metrics are computed independently on each fold; aggregation happens only after all folds complete.
// DBM baseline MAE is sourced externally (from scoreboard CSV) and passed per fold.
//
// Coverage rate uses a trivial ±sigma(residuals)*1.645 interval when the model
// provides no explicit bounds — this is only used when LowerBound/UpperBound is null.
public sealed class ComprehensiveMetricsEvaluator
{
    // Evaluate a single fold's predictions against observed values.
    // dbmBaselineMae: the DBM baseline MAE for this fold (from scoreboard).
    // lowerBound / upperBound: 90% prediction interval endpoints (may be null → coverage = NaN).
    public FoldMetrics EvaluateFold(
        int foldIndex,
        float[] observed,
        float[] predicted,
        double dbmBaselineMae,
        float[]? lowerBound = null,
        float[]? upperBound = null)
    {
        if (observed is null) throw new ArgumentNullException(nameof(observed));
        if (predicted is null) throw new ArgumentNullException(nameof(predicted));
        if (observed.Length == 0)
            throw new ArgumentException("observed array must not be empty.", nameof(observed));
        if (observed.Length != predicted.Length)
            throw new ArgumentException(
                $"observed ({observed.Length}) and predicted ({predicted.Length}) must have equal length.");

        int n = observed.Length;
        double mae   = ComputeMae(observed, predicted, n);
        double rmse  = ComputeRmse(observed, predicted, n);
        double r2    = ComputeR2(observed, predicted, n);
        double bias  = ComputeBias(observed, predicted, n);
        double skill = dbmBaselineMae > 0 ? 1.0 - mae / dbmBaselineMae : double.NaN;
        double hr6   = ComputeHitRate(observed, predicted, n, 6.0);
        double hr12  = ComputeHitRate(observed, predicted, n, 12.0);
        double pinball = ComputePinball(observed, predicted, n, alpha: 0.10);
        double coverage = ComputeCoverage(observed, lowerBound, upperBound, n);
        double kendall = ComputeKendallTau(observed, predicted, n);

        return new FoldMetrics(foldIndex, mae, rmse, r2, bias, skill, hr6, hr12, pinball, coverage, kendall);
    }

    // Aggregate mean ± std across all folds (RULE-163).
    public AggregatedMetrics AggregateFolds(IReadOnlyList<FoldMetrics> folds)
    {
        if (folds is null || folds.Count == 0)
            throw new ArgumentException("Cannot aggregate zero folds.", nameof(folds));

        return new AggregatedMetrics(
            MaeMean:        Mean(folds, f => f.Mae),        MaeStd:        Std(folds, f => f.Mae),
            RmseMean:       Mean(folds, f => f.Rmse),       RmseStd:       Std(folds, f => f.Rmse),
            R2Mean:         Mean(folds, f => f.R2),         R2Std:         Std(folds, f => f.R2),
            BiasMean:       Mean(folds, f => f.Bias),       BiasStd:       Std(folds, f => f.Bias),
            SkillMean:      Mean(folds, f => f.SkillVsDbm), SkillStd:      Std(folds, f => f.SkillVsDbm),
            HitRate6hMean:  Mean(folds, f => f.HitRate6h),  HitRate6hStd:  Std(folds, f => f.HitRate6h),
            HitRate12hMean: Mean(folds, f => f.HitRate12h), HitRate12hStd: Std(folds, f => f.HitRate12h),
            PinballMean:    Mean(folds, f => f.PinballLoss10), PinballStd: Std(folds, f => f.PinballLoss10),
            CoverageMean:   Mean(folds, f => f.CoverageRate90), CoverageStd: Std(folds, f => f.CoverageRate90),
            KendallMean:    Mean(folds, f => f.KendallTau), KendallStd:    Std(folds, f => f.KendallTau));
    }

    // ── Metric implementations ──────────────────────────────────────────────────

    private static double ComputeMae(float[] obs, float[] pred, int n)
    {
        double sum = 0;
        int valid = 0;
        for (int i = 0; i < n; i++)
        {
            if (float.IsNaN(obs[i]) || float.IsNaN(pred[i])) continue;
            sum += Math.Abs(obs[i] - pred[i]);
            valid++;
        }
        return valid > 0 ? sum / valid : double.NaN;
    }

    private static double ComputeRmse(float[] obs, float[] pred, int n)
    {
        double ss = 0;
        int valid = 0;
        for (int i = 0; i < n; i++)
        {
            if (float.IsNaN(obs[i]) || float.IsNaN(pred[i])) continue;
            double diff = obs[i] - pred[i];
            ss += diff * diff;
            valid++;
        }
        return valid > 0 ? Math.Sqrt(ss / valid) : double.NaN;
    }

    private static double ComputeR2(float[] obs, float[] pred, int n)
    {
        double sumObs = 0;
        int valid = 0;
        for (int i = 0; i < n; i++)
        {
            if (float.IsNaN(obs[i]) || float.IsNaN(pred[i])) continue;
            sumObs += obs[i];
            valid++;
        }
        if (valid == 0) return double.NaN;

        double mean = sumObs / valid;
        double ssRes = 0, ssTot = 0;
        for (int i = 0; i < n; i++)
        {
            if (float.IsNaN(obs[i]) || float.IsNaN(pred[i])) continue;
            double diff = obs[i] - pred[i];
            ssRes += diff * diff;
            double dev = obs[i] - mean;
            ssTot += dev * dev;
        }
        return ssTot < 1e-12 ? 0.0 : 1.0 - ssRes / ssTot;
    }

    private static double ComputeBias(float[] obs, float[] pred, int n)
    {
        double sum = 0;
        int valid = 0;
        for (int i = 0; i < n; i++)
        {
            if (float.IsNaN(obs[i]) || float.IsNaN(pred[i])) continue;
            sum += pred[i] - obs[i];
            valid++;
        }
        return valid > 0 ? sum / valid : double.NaN;
    }

    private static double ComputeHitRate(float[] obs, float[] pred, int n, double windowHours)
    {
        int hits = 0, valid = 0;
        for (int i = 0; i < n; i++)
        {
            if (float.IsNaN(obs[i]) || float.IsNaN(pred[i])) continue;
            if (Math.Abs(obs[i] - pred[i]) <= windowHours) hits++;
            valid++;
        }
        return valid > 0 ? (double)hits / valid : double.NaN;
    }

    // Asymmetric pinball (quantile) loss at quantile alpha.
    // L = mean( max(alpha*(y - yhat), (alpha-1)*(y - yhat)) )
    private static double ComputePinball(float[] obs, float[] pred, int n, double alpha)
    {
        double sum = 0;
        int valid = 0;
        for (int i = 0; i < n; i++)
        {
            if (float.IsNaN(obs[i]) || float.IsNaN(pred[i])) continue;
            double r = obs[i] - pred[i];
            sum += Math.Max(alpha * r, (alpha - 1.0) * r);
            valid++;
        }
        return valid > 0 ? sum / valid : double.NaN;
    }

    // Coverage rate: fraction of observed values within [lowerBound, upperBound].
    // Returns NaN if bounds are null (model didn't produce intervals).
    private static double ComputeCoverage(
        float[] obs,
        float[]? lower,
        float[]? upper,
        int n)
    {
        if (lower is null || upper is null) return double.NaN;
        int hits = 0, valid = 0;
        for (int i = 0; i < n; i++)
        {
            if (float.IsNaN(obs[i]) || float.IsNaN(lower[i]) || float.IsNaN(upper[i])) continue;
            if (obs[i] >= lower[i] && obs[i] <= upper[i]) hits++;
            valid++;
        }
        return valid > 0 ? (double)hits / valid : double.NaN;
    }

    // Kendall's tau-b rank correlation.
    // O(n²) — acceptable for CV fold sizes (~50–200 events).
    internal static double ComputeKendallTau(float[] obs, float[] pred, int n)
    {
        // Collect valid pairs
        var pairs = new List<(float o, float p)>(n);
        for (int i = 0; i < n; i++)
        {
            if (!float.IsNaN(obs[i]) && !float.IsNaN(pred[i]))
                pairs.Add((obs[i], pred[i]));
        }

        int m = pairs.Count;
        if (m < 2) return double.NaN;

        long concordant = 0, discordant = 0;
        long tiesO = 0, tiesP = 0;

        for (int i = 0; i < m - 1; i++)
        {
            for (int j = i + 1; j < m; j++)
            {
                double dO = pairs[j].o - pairs[i].o;
                double dP = pairs[j].p - pairs[i].p;
                double product = dO * dP;
                if (product > 0) concordant++;
                else if (product < 0) discordant++;
                else
                {
                    if (dO == 0 && dP != 0) tiesO++;
                    else if (dP == 0 && dO != 0) tiesP++;
                    // both zero: tied pair — not counted
                }
            }
        }

        double denom = Math.Sqrt((concordant + discordant + tiesO) * (double)(concordant + discordant + tiesP));
        return denom < 1e-12 ? double.NaN : (concordant - discordant) / denom;
    }

    // ── Aggregation helpers ─────────────────────────────────────────────────────

    private static double Mean(IReadOnlyList<FoldMetrics> folds, Func<FoldMetrics, double> selector)
    {
        var values = folds.Select(selector).Where(v => !double.IsNaN(v)).ToList();
        return values.Count > 0 ? values.Average() : double.NaN;
    }

    private static double Std(IReadOnlyList<FoldMetrics> folds, Func<FoldMetrics, double> selector)
    {
        var values = folds.Select(selector).Where(v => !double.IsNaN(v)).ToList();
        if (values.Count < 2) return double.NaN;
        double mean = values.Average();
        double variance = values.Average(v => (v - mean) * (v - mean));
        return Math.Sqrt(variance);
    }
}
