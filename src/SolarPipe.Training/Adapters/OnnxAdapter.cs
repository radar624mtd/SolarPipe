using Microsoft.ML.OnnxRuntime;
using SolarPipe.Core.Interfaces;
using SolarPipe.Core.Models;

namespace SolarPipe.Training.Adapters;

// OnnxAdapter: loads a pre-exported .onnx file and wraps it as ITrainedModel.
// "Training" = load model from disk; no fitting occurs here.
//
// ModelType "NeuralOde" → RULE-070: loads dynamics network f(y,t,θ) only;
//   full ODE integration (Dormand-Prince) happens in OnnxTrainedModel.PredictAsync.
// ModelType "TftPinn" → G5 named-input graph:
//   inputs  x_flat (B, N_FLAT), m_flat (B, N_FLAT), x_seq (B, T, C), m_seq (B, T, C)
//   outputs p10 (B, 1), p50 (B, 1), p90 (B, 1)
//   OnnxTrainedModel.PredictAsync returns P50 in Values, P10/P90 in LowerBound/UpperBound.
// All other ModelTypes → standard ONNX inference (single forward pass).
//
// Hyperparameter key "model_path" (snake_case, OrdinalIgnoreCase) is required.
public sealed class OnnxAdapter : IFrameworkAdapter
{
    public FrameworkType FrameworkType => FrameworkType.Onnx;

    public IReadOnlyList<string> SupportedModels => ["Standard", "NeuralOde", "TftPinn"];

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
        bool isTftPinn = config.ModelType.Equals("TftPinn", StringComparison.OrdinalIgnoreCase);

        var options = new SessionOptions();
        // Deterministic execution where possible
        options.ExecutionMode = ExecutionMode.ORT_SEQUENTIAL;

        bool useCudaEp = string.Equals(
            FindHyperString(config, "use_cuda_ep"), "true",
            StringComparison.OrdinalIgnoreCase);
        if (useCudaEp)
        {
            // RULE-300: must use MatMul-surgery ONNX (model_matmul_reduce.onnx).
            // Raw torch.onnx.export output crashes on Maxwell sm_52 via cuDNN ReduceSum.
            // OrtCudaProviderOptions defaults: device_id=0, arena=1, mem_limit=0 (unlimited).
            options.AppendExecutionProvider_CUDA(0);
        }

        var session = new InferenceSession(modelPath, options);

        var metrics = new ModelMetrics(double.NaN, double.NaN, double.NaN);
        var mode = isNeuralOde
            ? OnnxInferenceMode.NeuralOde
            : isTftPinn
                ? OnnxInferenceMode.TftPinn
                : OnnxInferenceMode.Standard;
        ITrainedModel model = new OnnxTrainedModel(config, session, mode, metrics);
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
