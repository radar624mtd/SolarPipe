using Microsoft.ML.OnnxRuntime;
using SolarPipe.Core.Interfaces;
using SolarPipe.Core.Models;

namespace SolarPipe.Training.Adapters;

// OnnxAdapter: loads a pre-exported .onnx file and wraps it as ITrainedModel.
// "Training" = load model from disk; no fitting occurs here.
//
// ModelType "NeuralOde" → RULE-070: loads dynamics network f(y,t,θ) only;
//   full ODE integration (Dormand-Prince) happens in OnnxTrainedModel.PredictAsync.
// All other ModelTypes → standard ONNX inference (single forward pass).
//
// Hyperparameter key "model_path" (snake_case, OrdinalIgnoreCase) is required.
public sealed class OnnxAdapter : IFrameworkAdapter
{
    public FrameworkType FrameworkType => FrameworkType.Onnx;

    public IReadOnlyList<string> SupportedModels => ["Standard", "NeuralOde"];

    public Task<ITrainedModel> TrainAsync(
        StageConfig config,
        IDataFrame trainingData,
        IDataFrame? validationData,
        CancellationToken ct)
    {
        if (!SupportedModels.Contains(config.ModelType, StringComparer.OrdinalIgnoreCase))
            throw new NotSupportedException(
                $"OnnxAdapter does not support model type '{config.ModelType}'. " +
                $"Supported: [{string.Join(", ", SupportedModels)}] (stage={config.Name}).");

        var modelPath = FindHyperString(config, "model_path")
            ?? throw new ArgumentException(
                $"OnnxAdapter requires hyperparameter 'model_path' pointing to a .onnx file. " +
                $"Stage: '{config.Name}'.");

        if (!File.Exists(modelPath))
            throw new FileNotFoundException(
                $"ONNX model file not found. Stage: '{config.Name}', path: '{modelPath}'.", modelPath);

        bool isNeuralOde = config.ModelType.Equals("NeuralOde", StringComparison.OrdinalIgnoreCase);

        var options = new SessionOptions();
        // Deterministic execution where possible
        options.ExecutionMode = ExecutionMode.ORT_SEQUENTIAL;

        var session = new InferenceSession(modelPath, options);

        var metrics = new ModelMetrics(double.NaN, double.NaN, double.NaN);
        ITrainedModel model = new OnnxTrainedModel(config, session, isNeuralOde, metrics);
        return Task.FromResult(model);
    }

    internal static string? FindHyperString(StageConfig config, string key)
    {
        if (config.Hyperparameters == null) return null;
        foreach (var kv in config.Hyperparameters)
            if (string.Equals(kv.Key, key, StringComparison.OrdinalIgnoreCase))
                return kv.Value?.ToString();
        return null;
    }
}
