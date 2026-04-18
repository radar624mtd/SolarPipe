using Grpc.Net.Client;
using SolarPipe.Core.Interfaces;
using SolarPipe.Core.Models;
using SolarPipe.Grpc;

namespace SolarPipe.Training.Adapters;

// GrpcSidecarAdapter: dispatches training and prediction to the Python sidecar via gRPC.
// Phase 2: validates proto schema + Arrow IPC contract against deterministic stub.
// Phase 4: replaces stub with real PyTorch server.
//
// RULE-061: Use server-streaming RPCs for long-running training.
// RULE-063: Enforce float32 Arrow IPC schema at both ends (via ArrowIpcHelper).
// RULE-125: Large arrays always via Arrow IPC files, never inline proto bytes.
public sealed class GrpcSidecarAdapter : IFrameworkAdapter, IDisposable
{
    private readonly GrpcChannel _channel;
    private readonly PythonTrainer.PythonTrainerClient _client;
    private readonly string _tempDir;
    private bool _disposed;

    public FrameworkType FrameworkType => FrameworkType.PythonGrpc;
    // "TftPinn" is routed to _train_tft_pinn in the sidecar (G5 wiring).
    // "TFT" + hyperparameters[use_tft_pinn]=true is equivalent (server-side dispatch).
    public IReadOnlyList<string> SupportedModels => ["TFT", "TftPinn", "NeuralOde"];

    public GrpcSidecarAdapter(string address, string? tempDir = null)
    {
        if (string.IsNullOrWhiteSpace(address))
            throw new ArgumentException("gRPC address must not be empty.", nameof(address));

        _channel = GrpcChannel.ForAddress(address);
        _client = new PythonTrainer.PythonTrainerClient(_channel);
        _tempDir = tempDir ?? Path.Combine(Path.GetTempPath(), "solarpipe_grpc");
        Directory.CreateDirectory(_tempDir);
    }

    // CheckHealthAsync: attempts to connect to the gRPC channel within a timeout.
    // Returns true if the channel reaches the Ready state, false on timeout/error.
    // Used by SidecarLifecycleService and for pre-flight checks before TrainAsync.
    public async Task<bool> CheckHealthAsync(TimeSpan timeout, CancellationToken ct)
    {
        try
        {
            using var cts = CancellationTokenSource.CreateLinkedTokenSource(ct);
            cts.CancelAfter(timeout);
            await _channel.ConnectAsync(cts.Token);
            return true;
        }
        catch (OperationCanceledException)
        {
            return false;
        }
        catch (Exception)
        {
            return false;
        }
    }

    public async Task<ITrainedModel> TrainAsync(
        StageConfig config,
        IDataFrame trainingData,
        IDataFrame? validationData,
        CancellationToken ct)
    {
        if (!SupportedModels.Contains(config.ModelType, StringComparer.OrdinalIgnoreCase))
            throw new NotSupportedException(
                $"GrpcSidecarAdapter does not support model type '{config.ModelType}'. " +
                $"Supported: [{string.Join(", ", SupportedModels)}] (stage={config.Name}).");

        string arrowPath = Path.Combine(_tempDir, $"{config.Name}_{Guid.NewGuid():N}_train.arrow");

        try
        {
            // RULE-125: write features to Arrow IPC file; pass path via gRPC, not bytes.
            await ArrowIpcHelper.WriteAsync(trainingData, arrowPath, ct);

            var request = new TrainRequest
            {
                StageName = config.Name,
                ModelType = config.ModelType,
                ArrowIpcPath = arrowPath,
                TargetColumn = config.Target,
                ModelOutputDir = _tempDir,
            };
            foreach (var feature in config.Features)
                request.FeatureColumns.Add(feature);
            if (config.Hyperparameters != null)
                foreach (var kv in config.Hyperparameters)
                    request.Hyperparameters[kv.Key] = kv.Value?.ToString() ?? "";

            // RULE-061: Use server-streaming for training to avoid deadline issues.
            string modelId = "";
            using var stream = _client.StreamTrain(request, cancellationToken: ct);
            while (await stream.ResponseStream.MoveNext(ct))
            {
                var progress = stream.ResponseStream.Current;
                if (!string.IsNullOrEmpty(progress.ErrorMessage))
                    throw new InvalidOperationException(
                        $"gRPC sidecar training error: {progress.ErrorMessage} " +
                        $"(stage={config.Name}, model_type={config.ModelType}).");

                if (progress.IsFinal)
                    modelId = progress.ModelId;
            }

            if (string.IsNullOrEmpty(modelId))
                throw new InvalidOperationException(
                    $"gRPC sidecar did not return a model_id for stage '{config.Name}'.");

            return new GrpcSidecarTrainedModel(
                config, modelId, _client, _tempDir,
                new ModelMetrics(double.NaN, double.NaN, double.NaN));
        }
        finally
        {
            TryDeleteFile(arrowPath);
        }
    }

    private static void TryDeleteFile(string path)
    {
        try { File.Delete(path); } catch { /* best-effort cleanup */ }
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _channel.Dispose();
    }
}
