using SolarPipe.Core.Interfaces;
using SolarPipe.Core.Models;

namespace SolarPipe.Prediction;

// EnsembleModel: weighted average of multiple model outputs.
// Operator: m1 + m2 (equal weights), or constructed directly with explicit weights.
// RULE-150: predict calls can be CPU-bound (ML.NET inference); run concurrently via LongRunning tasks.
public sealed class EnsembleModel : IComposedModel
{
    private readonly IReadOnlyList<ITrainedModel> _models;
    private readonly IReadOnlyList<float> _weights; // normalized to sum = 1.0

    public string Name { get; }

    public EnsembleModel(IReadOnlyList<ITrainedModel> models, IReadOnlyList<float>? weights = null, string name = "")
    {
        if (models is null) throw new ArgumentNullException(nameof(models));
        if (models.Count == 0) throw new ArgumentException(
            "EnsembleModel requires at least one model.", nameof(models));

        _models = models;
        _weights = NormalizeWeights(weights, models.Count);
        Name = string.IsNullOrWhiteSpace(name)
            ? string.Join("+", models.Select(m => m.ModelId))
            : name;
    }

    public async Task<PredictionResult> PredictAsync(IDataFrame input, CancellationToken ct)
    {
        if (input is null) throw new ArgumentNullException(nameof(input),
            $"EnsembleModel '{Name}': input DataFrame must not be null.");

        // RULE-150: Run each model predict on a LongRunning task to avoid thread-pool starvation.
        var tasks = _models.Select(model =>
            Task.Factory.StartNew(
                () => model.PredictAsync(input, ct).GetAwaiter().GetResult(),
                ct,
                TaskCreationOptions.LongRunning,
                TaskScheduler.Default)).ToArray();

        var results = await Task.WhenAll(tasks).ConfigureAwait(false);

        // Validate all results have same length
        int n = results[0].Values.Length;
        for (int i = 1; i < results.Length; i++)
        {
            if (results[i].Values.Length != n)
                throw new InvalidOperationException(
                    $"EnsembleModel '{Name}': model '{_models[i].ModelId}' produced " +
                    $"{results[i].Values.Length} predictions but expected {n} (from model '{_models[0].ModelId}').");
        }

        var finalValues = new float[n];
        for (int row = 0; row < n; row++)
        {
            float sum = 0f;
            bool anyNaN = false;
            for (int m = 0; m < results.Length; m++)
            {
                float v = results[m].Values[row];
                if (float.IsNaN(v)) { anyNaN = true; break; }
                sum += _weights[m] * v;
            }
            finalValues[row] = anyNaN ? float.NaN : sum;
        }

        // Propagate bounds if all models provide them
        float[]? lower = null;
        float[]? upper = null;
        if (results.All(r => r.LowerBound != null && r.UpperBound != null))
        {
            lower = new float[n];
            upper = new float[n];
            for (int row = 0; row < n; row++)
            {
                float lo = 0f, hi = 0f;
                for (int m = 0; m < results.Length; m++)
                {
                    lo += _weights[m] * results[m].LowerBound![row];
                    hi += _weights[m] * results[m].UpperBound![row];
                }
                lower[row] = lo;
                upper[row] = hi;
            }
        }

        return new PredictionResult(
            finalValues, lower, upper,
            ModelId: $"{Name}_ensemble",
            GeneratedAt: DateTime.UtcNow);
    }

    private static IReadOnlyList<float> NormalizeWeights(IReadOnlyList<float>? weights, int count)
    {
        if (weights == null)
            return Enumerable.Repeat(1f / count, count).ToList();

        if (weights.Count != count)
            throw new ArgumentException(
                $"EnsembleModel: {weights.Count} weights provided but {count} models given.", nameof(weights));

        float sum = weights.Sum();
        if (sum <= 0f)
            throw new ArgumentException(
                $"EnsembleModel: weight sum must be positive, got {sum}.", nameof(weights));

        return weights.Select(w => w / sum).ToList();
    }
}
