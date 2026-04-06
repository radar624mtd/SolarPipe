namespace SolarPipe.Training.Validation;

// Split conformal prediction — provides finite-sample coverage guarantees
// regardless of underlying model distribution.
//
// Usage:
//   1. Train model on training split.
//   2. Calibrate on held-out calibration split.
//   3. Call GetIntervalWidth(alpha) to get ±half-width for new predictions.
//
// Coverage guarantee: P(Y ∈ [ŷ ± width]) ≥ 1 − α for any distribution
// (Vovk et al. 2005, Angelopoulos & Bates 2021).
//
// Fallback for non-temporal use cases. For time-series / solar cycle data
// use EnbPiPredictor which adapts to distribution shifts.
public sealed class SplitConformalPredictor
{
    private float[]? _sortedResiduals;

    // Calibrate on a held-out set by computing and sorting absolute residuals.
    // actual and predicted must be the same length and non-empty.
    public void Calibrate(float[] actual, float[] predicted)
    {
        if (actual is null) throw new ArgumentNullException(nameof(actual));
        if (predicted is null) throw new ArgumentNullException(nameof(predicted));
        if (actual.Length != predicted.Length)
            throw new ArgumentException(
                $"SplitConformalPredictor.Calibrate: actual length {actual.Length} " +
                $"≠ predicted length {predicted.Length}.");
        if (actual.Length == 0)
            throw new ArgumentException(
                "SplitConformalPredictor.Calibrate: calibration set is empty.");

        _sortedResiduals = actual
            .Zip(predicted, (a, p) => MathF.Abs(a - p))
            .OrderBy(r => r)
            .ToArray();
    }

    // Get prediction interval half-width for desired miscoverage level alpha.
    // alpha = 0.1 → 90% coverage, alpha = 0.05 → 95% coverage.
    // Returns the (1−α)·(1 + 1/n) quantile of calibration absolute residuals.
    public float GetIntervalWidth(float alpha = 0.1f)
    {
        if (_sortedResiduals is null)
            throw new InvalidOperationException(
                "SplitConformalPredictor: call Calibrate before GetIntervalWidth.");
        if (alpha is <= 0f or >= 1f)
            throw new ArgumentOutOfRangeException(nameof(alpha), "alpha must be in (0, 1).");

        int n = _sortedResiduals.Length;
        // Finite-sample correction: use (1−α)·(1 + 1/n) quantile
        double quantile = (1.0 - alpha) * (1.0 + 1.0 / n);
        int index = (int)Math.Ceiling(quantile * n) - 1;
        index = Math.Clamp(index, 0, n - 1);
        return _sortedResiduals[index];
    }

    // Returns (lower, upper) prediction intervals for each prediction value.
    public (float[] Lower, float[] Upper) GetIntervals(float[] predictions, float alpha = 0.1f)
    {
        float width = GetIntervalWidth(alpha);
        return (
            predictions.Select(p => p - width).ToArray(),
            predictions.Select(p => p + width).ToArray()
        );
    }
}
