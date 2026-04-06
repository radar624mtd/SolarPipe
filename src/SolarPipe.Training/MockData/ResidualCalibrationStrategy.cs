using SolarPipe.Core.Interfaces;
using SolarPipe.Core.Models;
using SolarPipe.Data.DataFrame;

namespace SolarPipe.Training.MockData;

// Strategy: train on synthetic data, compute (obs - synthetic_prediction) residuals,
// then train a correction model on those residuals.
// The ResidualCalibrator wraps both models so prediction = base_prediction + correction.
internal sealed class ResidualCalibrationStrategy : IMockDataStrategy
{
    private readonly IFrameworkAdapter _adapter;

    internal ResidualCalibrationStrategy(IFrameworkAdapter adapter)
    {
        _adapter = adapter;
    }

    public async Task<ITrainedModel> TrainAsync(
        StageConfig stage,
        IDataFrame syntheticData,
        IDataFrame observationalData,
        CancellationToken ct)
    {
        ct.ThrowIfCancellationRequested();

        if (syntheticData.RowCount == 0)
            throw new ArgumentException(
                $"ResidualCalibration: syntheticData is empty. Stage: '{stage.Name}'.",
                nameof(syntheticData));
        if (observationalData.RowCount == 0)
            throw new ArgumentException(
                $"ResidualCalibration: observationalData is empty. Stage: '{stage.Name}'.",
                nameof(observationalData));
        if (!observationalData.Schema.HasColumn(stage.Target))
            throw new ArgumentException(
                $"ResidualCalibration: target column '{stage.Target}' not found in observationalData. Stage: '{stage.Name}'.",
                nameof(observationalData));

        // Step 1: train base model on synthetic data
        var baseModel = await _adapter.TrainAsync(stage, syntheticData, null, ct);
        ct.ThrowIfCancellationRequested();

        // Step 2: generate base predictions on observational data
        var basePredictions = await baseModel.PredictAsync(observationalData, ct);
        ct.ThrowIfCancellationRequested();

        // Step 3: compute residuals = observed - synthetic_predicted
        var observed = observationalData.GetColumn(stage.Target);
        var predicted = basePredictions.Values;

        int n = Math.Min(observed.Length, predicted.Length);
        var residuals = new float[n];
        for (int i = 0; i < n; i++)
            residuals[i] = float.IsNaN(observed[i]) || float.IsNaN(predicted[i])
                ? float.NaN
                : observed[i] - predicted[i];

        // Step 4: train correction model on residuals
        // Use same features but target = residual
        var residualStage = stage with { Target = "__residual__" };
        var residualData = observationalData.AddColumn("__residual__", residuals);

        var correctionModel = await _adapter.TrainAsync(residualStage, residualData, null, ct);

        return new ResidualCalibratorModel(baseModel, correctionModel, stage.Name);
    }
}

// Wraps base + correction models: final prediction = base + correction
internal sealed class ResidualCalibratorModel : ITrainedModel
{
    private readonly ITrainedModel _base;
    private readonly ITrainedModel _correction;

    public string ModelId => $"residual_calibrator/{_base.ModelId}+{_correction.ModelId}";
    public string StageName { get; }
    public ModelMetrics Metrics => _base.Metrics;

    internal ResidualCalibratorModel(ITrainedModel baseModel, ITrainedModel correctionModel, string stageName)
    {
        _base = baseModel;
        _correction = correctionModel;
        StageName = stageName;
    }

    public async Task<PredictionResult> PredictAsync(IDataFrame input, CancellationToken ct)
    {
        var baseResult = await _base.PredictAsync(input, ct);
        ct.ThrowIfCancellationRequested();

        // Augment input with base prediction so correction model can use it
        var augmented = input.AddColumn("__base_prediction__", baseResult.Values);
        var corrResult = await _correction.PredictAsync(augmented, ct);

        int n = baseResult.Values.Length;
        var finalValues = new float[n];
        for (int i = 0; i < n; i++)
            finalValues[i] = float.IsNaN(baseResult.Values[i]) || float.IsNaN(corrResult.Values[i])
                ? float.NaN
                : baseResult.Values[i] + corrResult.Values[i];

        float[]? lb = null, ub = null;
        if (baseResult.LowerBound != null && corrResult.LowerBound != null)
        {
            lb = new float[n];
            ub = new float[n];
            for (int i = 0; i < n; i++)
            {
                lb[i] = baseResult.LowerBound[i] + corrResult.LowerBound![i];
                ub[i] = baseResult.UpperBound![i] + corrResult.UpperBound![i];
            }
        }

        return new PredictionResult(finalValues, lb, ub, ModelId, DateTime.UtcNow);
    }

    public Task SaveAsync(string path, CancellationToken ct) =>
        throw new NotSupportedException(
            $"ResidualCalibratorModel.SaveAsync not yet implemented. " +
            $"Stage: '{StageName}'. Implement composite save when persistence is needed.");

    public Task LoadAsync(string path, CancellationToken ct) =>
        throw new NotSupportedException(
            $"ResidualCalibratorModel.LoadAsync not yet implemented. " +
            $"Stage: '{StageName}'.");
}
