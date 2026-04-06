namespace SolarPipe.Training.Validation;

// EnbPI: Ensemble Predictors with Prediction Intervals (Xu & Xie, 2021).
// Adaptive conformal prediction for time-series — adapts to distribution shifts
// (solar cycle phase changes) via a sliding window of recent residuals.
//
// Unlike standard split conformal (which assumes exchangeability), EnbPI provides
// asymptotic coverage guarantees for non-exchangeable data by discarding stale
// residuals when the distribution shifts.
//
// Reference: Xu & Xie (2021). "Conformal prediction interval for dynamic time-series."
//   ICML 2021. https://arxiv.org/abs/2010.09107
//
// Calibration protocol:
//   1. Build sequence of (actual, predicted) pairs in temporal order.
//   2. Call UpdateResidual() for each pair as they arrive.
//   3. Call GetIntervalWidth(alpha) to query current interval width.
//
// The sliding window automatically down-weights old residuals from different
// solar cycle phases without requiring explicit phase labels.
public sealed class EnbPiPredictor
{
    private readonly Queue<float> _residualWindow;
    private readonly int _windowSize;

    // windowSize: number of recent residuals to keep (default 100 events per ADR-012).
    //   Smaller → faster adaptation to phase changes; larger → more stable coverage.
    public EnbPiPredictor(int windowSize = 100)
    {
        if (windowSize < 2)
            throw new ArgumentOutOfRangeException(nameof(windowSize),
                "EnbPI windowSize must be ≥ 2.");
        _windowSize = windowSize;
        _residualWindow = new Queue<float>(windowSize + 1);
    }

    public int WindowCount => _residualWindow.Count;

    // Add a new observed residual |actual - predicted| to the sliding window.
    // Call in temporal order as new predictions arrive.
    public void UpdateResidual(float actual, float predicted)
    {
        float absResidual = MathF.Abs(actual - predicted);
        if (float.IsNaN(absResidual))
            return; // Skip NaN residuals (sentinel values already converted at load time)

        _residualWindow.Enqueue(absResidual);
        if (_residualWindow.Count > _windowSize)
            _residualWindow.Dequeue();
    }

    // Batch initialization from a calibration split.
    // Adds residuals in the order given — use temporal order.
    public void Calibrate(float[] actuals, float[] predicted)
    {
        if (actuals is null) throw new ArgumentNullException(nameof(actuals));
        if (predicted is null) throw new ArgumentNullException(nameof(predicted));
        if (actuals.Length != predicted.Length)
            throw new ArgumentException(
                $"EnbPiPredictor.Calibrate: actual length {actuals.Length} " +
                $"≠ predicted length {predicted.Length}.");

        // Keep only the most recent windowSize residuals from the calibration set
        int start = Math.Max(0, actuals.Length - _windowSize);
        for (int i = start; i < actuals.Length; i++)
            UpdateResidual(actuals[i], predicted[i]);
    }

    // Get prediction interval half-width for desired miscoverage level alpha.
    // alpha = 0.1 → 90% coverage.
    // Throws if no residuals have been added yet.
    public float GetIntervalWidth(float alpha = 0.1f)
    {
        if (_residualWindow.Count == 0)
            throw new InvalidOperationException(
                "EnbPiPredictor: no residuals in window. Call Calibrate or UpdateResidual first.");
        if (alpha is <= 0f or >= 1f)
            throw new ArgumentOutOfRangeException(nameof(alpha), "alpha must be in (0, 1).");

        float[] sorted = _residualWindow.OrderBy(r => r).ToArray();
        int n = sorted.Length;

        // Adaptive quantile: (1−α)·(1 + 1/n) quantile of sliding window residuals
        double quantile = (1.0 - alpha) * (1.0 + 1.0 / n);
        int index = (int)Math.Ceiling(quantile * n) - 1;
        index = Math.Clamp(index, 0, n - 1);
        return sorted[index];
    }

    // Returns (lower, upper) adaptive prediction intervals for each value.
    public (float[] Lower, float[] Upper) GetIntervals(float[] predictions, float alpha = 0.1f)
    {
        float width = GetIntervalWidth(alpha);
        return (
            predictions.Select(p => p - width).ToArray(),
            predictions.Select(p => p + width).ToArray()
        );
    }

    // Reset the residual window (e.g. when solar cycle phase is known to change).
    public void Reset() => _residualWindow.Clear();
}
