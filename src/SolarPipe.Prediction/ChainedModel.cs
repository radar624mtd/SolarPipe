using SolarPipe.Core.Interfaces;
using SolarPipe.Core.Models;

namespace SolarPipe.Prediction;

// ChainedModel: output of left model feeds as an additional column into right model.
// Operator: left -> right
// The left model's prediction Values are appended as column "chained_prediction_{leftModelId}"
// before passing the augmented frame to the right model.
public sealed class ChainedModel : IComposedModel
{
    private readonly ITrainedModel _left;
    private readonly ITrainedModel _right;
    private readonly string _outputColumnName;

    public string Name { get; }

    public ChainedModel(ITrainedModel left, ITrainedModel right, string name = "")
    {
        _left = left ?? throw new ArgumentNullException(nameof(left));
        _right = right ?? throw new ArgumentNullException(nameof(right));
        _outputColumnName = $"chained_{_left.StageName}";
        Name = string.IsNullOrWhiteSpace(name) ? $"{_left.ModelId}→{_right.ModelId}" : name;
    }

    public async Task<PredictionResult> PredictAsync(IDataFrame input, CancellationToken ct)
    {
        if (input is null) throw new ArgumentNullException(nameof(input),
            $"ChainedModel '{Name}': input DataFrame must not be null.");

        var leftResult = await _left.PredictAsync(input, ct).ConfigureAwait(false);

        if (leftResult.Values.Length != input.RowCount)
            throw new InvalidOperationException(
                $"ChainedModel '{Name}': left model '{_left.ModelId}' produced {leftResult.Values.Length} predictions " +
                $"but input has {input.RowCount} rows.");

        // Propagate NaN from left: if left output is NaN, pass NaN through right input column.
        var augmented = input.AddColumn(_outputColumnName, leftResult.Values);

        try
        {
            return await _right.PredictAsync(augmented, ct).ConfigureAwait(false);
        }
        finally
        {
            augmented.Dispose();
        }
    }
}
