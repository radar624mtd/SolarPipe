using SolarPipe.Core.Interfaces;
using SolarPipe.Core.Models;

namespace SolarPipe.Prediction;

// GatedModel: classifier routes each row to one of two models.
// Operator: classifier ? (ifTrue, ifFalse)
//
// The classifier model returns Values in [0,1] per row (soft gate weight for ifTrue).
// If a row's gate value >= threshold, ifTrue is the primary model; below threshold → ifFalse.
//
// Soft gating: output = gate * ifTrue + (1 - gate) * ifFalse for smooth transitions.
// Uncertainty = routing-entropy-weighted combination of model bounds.
public sealed class GatedModel : IComposedModel
{
    private readonly ITrainedModel _classifier;
    private readonly ITrainedModel _ifTrue;
    private readonly ITrainedModel _ifFalse;
    private readonly float _threshold;

    public string Name { get; }

    public GatedModel(
        ITrainedModel classifier,
        ITrainedModel ifTrue,
        ITrainedModel ifFalse,
        float threshold = 0.5f,
        string name = "")
    {
        if (threshold <= 0f || threshold >= 1f)
            throw new ArgumentOutOfRangeException(nameof(threshold),
                $"GatedModel threshold must be in (0, 1), got {threshold}.");

        _classifier = classifier ?? throw new ArgumentNullException(nameof(classifier));
        _ifTrue = ifTrue ?? throw new ArgumentNullException(nameof(ifTrue));
        _ifFalse = ifFalse ?? throw new ArgumentNullException(nameof(ifFalse));
        _threshold = threshold;
        Name = string.IsNullOrWhiteSpace(name)
            ? $"{classifier.ModelId}?({ifTrue.ModelId},{ifFalse.ModelId})"
            : name;
    }

    public async Task<PredictionResult> PredictAsync(IDataFrame input, CancellationToken ct)
    {
        if (input is null) throw new ArgumentNullException(nameof(input),
            $"GatedModel '{Name}': input DataFrame must not be null.");

        // Run all three concurrently — classifier and both branch models
        var classifierTask = Task.Factory.StartNew(
            () => _classifier.PredictAsync(input, ct).GetAwaiter().GetResult(),
            ct, TaskCreationOptions.LongRunning, TaskScheduler.Default);

        var ifTrueTask = Task.Factory.StartNew(
            () => _ifTrue.PredictAsync(input, ct).GetAwaiter().GetResult(),
            ct, TaskCreationOptions.LongRunning, TaskScheduler.Default);

        var ifFalseTask = Task.Factory.StartNew(
            () => _ifFalse.PredictAsync(input, ct).GetAwaiter().GetResult(),
            ct, TaskCreationOptions.LongRunning, TaskScheduler.Default);

        await Task.WhenAll(classifierTask, ifTrueTask, ifFalseTask).ConfigureAwait(false);

        var gateValues = classifierTask.Result.Values;
        var trueValues = ifTrueTask.Result.Values;
        var falseValues = ifFalseTask.Result.Values;

        int n = input.RowCount;
        ValidateLength(gateValues.Length, n, _classifier.ModelId);
        ValidateLength(trueValues.Length, n, _ifTrue.ModelId);
        ValidateLength(falseValues.Length, n, _ifFalse.ModelId);

        var finalValues = new float[n];
        for (int i = 0; i < n; i++)
        {
            float gate = gateValues[i];
            if (float.IsNaN(gate) || float.IsNaN(trueValues[i]) || float.IsNaN(falseValues[i]))
            {
                finalValues[i] = float.NaN;
                continue;
            }
            // Clamp gate to [0,1] — classifier output should be in this range
            gate = Math.Clamp(gate, 0f, 1f);
            // Soft gating: smooth interpolation
            finalValues[i] = gate * trueValues[i] + (1f - gate) * falseValues[i];
        }

        // Propagate uncertainty if all three models provide bounds
        float[]? lower = null;
        float[]? upper = null;
        var trueResult = ifTrueTask.Result;
        var falseResult = ifFalseTask.Result;
        if (trueResult.LowerBound != null && trueResult.UpperBound != null &&
            falseResult.LowerBound != null && falseResult.UpperBound != null)
        {
            lower = new float[n];
            upper = new float[n];
            for (int i = 0; i < n; i++)
            {
                float gate = Math.Clamp(gateValues[i], 0f, 1f);
                lower[i] = gate * trueResult.LowerBound[i] + (1f - gate) * falseResult.LowerBound[i];
                upper[i] = gate * trueResult.UpperBound[i] + (1f - gate) * falseResult.UpperBound[i];
            }
        }

        return new PredictionResult(
            finalValues, lower, upper,
            ModelId: $"{Name}_gated",
            GeneratedAt: DateTime.UtcNow);
    }

    private void ValidateLength(int actual, int expected, string modelId)
    {
        if (actual != expected)
            throw new InvalidOperationException(
                $"GatedModel '{Name}': model '{modelId}' produced {actual} predictions " +
                $"but input has {expected} rows.");
    }
}
