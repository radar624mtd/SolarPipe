using SolarPipe.Core.Interfaces;
using SolarPipe.Core.Models;

namespace SolarPipe.Prediction;

// ResidualModel: right model learns to predict (observed - baseline).
// Operator: baseline ^ correction
//
// Predict flow:
//   1. Run baseline on input → get baseline_values
//   2. Augment input with "residual_baseline" column = baseline_values
//   3. Run correction on augmented frame → get correction_values (= residual predictions)
//   4. Final = baseline_values + correction_values (element-wise)
//   5. Uncertainty = sqrt(baseline_ub² + correction_ub²) if both have bounds
public sealed class ResidualModel : IComposedModel
{
    private readonly ITrainedModel _baseline;
    private readonly ITrainedModel _correction;
    private readonly string _baselineColumnName;

    public string Name { get; }

    public ResidualModel(ITrainedModel baseline, ITrainedModel correction, string name = "")
    {
        _baseline = baseline ?? throw new ArgumentNullException(nameof(baseline));
        _correction = correction ?? throw new ArgumentNullException(nameof(correction));
        _baselineColumnName = $"residual_baseline_{_baseline.StageName}";
        Name = string.IsNullOrWhiteSpace(name)
            ? $"{_baseline.ModelId}^{_correction.ModelId}"
            : name;
    }

    public async Task<PredictionResult> PredictAsync(IDataFrame input, CancellationToken ct)
    {
        if (input is null) throw new ArgumentNullException(nameof(input),
            $"ResidualModel '{Name}': input DataFrame must not be null.");

        var baselineResult = await _baseline.PredictAsync(input, ct).ConfigureAwait(false);

        if (baselineResult.Values.Length != input.RowCount)
            throw new InvalidOperationException(
                $"ResidualModel '{Name}': baseline model '{_baseline.ModelId}' produced " +
                $"{baselineResult.Values.Length} predictions but input has {input.RowCount} rows.");

        var augmented = input.AddColumn(_baselineColumnName, baselineResult.Values);

        PredictionResult correctionResult;
        try
        {
            correctionResult = await _correction.PredictAsync(augmented, ct).ConfigureAwait(false);
        }
        finally
        {
            augmented.Dispose();
        }

        if (correctionResult.Values.Length != baselineResult.Values.Length)
            throw new InvalidOperationException(
                $"ResidualModel '{Name}': correction model '{_correction.ModelId}' produced " +
                $"{correctionResult.Values.Length} predictions but expected {baselineResult.Values.Length}.");

        var finalValues = new float[baselineResult.Values.Length];
        for (int i = 0; i < finalValues.Length; i++)
        {
            float b = baselineResult.Values[i];
            float c = correctionResult.Values[i];
            // NaN propagation: if either is NaN, result is NaN
            finalValues[i] = float.IsNaN(b) || float.IsNaN(c) ? float.NaN : b + c;
        }

        // Propagate uncertainty bounds if both models provide them
        float[]? lower = null;
        float[]? upper = null;
        if (baselineResult.LowerBound != null && correctionResult.LowerBound != null &&
            baselineResult.UpperBound != null && correctionResult.UpperBound != null)
        {
            lower = new float[finalValues.Length];
            upper = new float[finalValues.Length];
            for (int i = 0; i < finalValues.Length; i++)
            {
                float bLo = baselineResult.LowerBound[i];
                float cLo = correctionResult.LowerBound[i];
                float bHi = baselineResult.UpperBound[i];
                float cHi = correctionResult.UpperBound[i];
                // Combined interval: half-widths added in quadrature
                float bHalf = (bHi - bLo) / 2f;
                float cHalf = (cHi - cLo) / 2f;
                float combinedHalf = MathF.Sqrt(bHalf * bHalf + cHalf * cHalf);
                lower[i] = finalValues[i] - combinedHalf;
                upper[i] = finalValues[i] + combinedHalf;
            }
        }

        return new PredictionResult(
            finalValues, lower, upper,
            ModelId: $"{Name}_residual",
            GeneratedAt: DateTime.UtcNow);
    }
}
