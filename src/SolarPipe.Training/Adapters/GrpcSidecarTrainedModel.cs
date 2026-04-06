using Apache.Arrow.Ipc;
using SolarPipe.Core.Interfaces;
using SolarPipe.Core.Models;
using SolarPipe.Grpc;

namespace SolarPipe.Training.Adapters;

// ITrainedModel backed by a running gRPC Python sidecar.
// Prediction: writes features as Arrow IPC, calls Predict RPC, reads output Arrow IPC.
// RULE-063: Validates float32 schema on both write and read sides.
// RULE-125: Large arrays via file path, not inline proto bytes.
public sealed class GrpcSidecarTrainedModel : ITrainedModel
{
    private readonly PythonTrainer.PythonTrainerClient _client;
    private readonly string _tempDir;
    private readonly ModelMetrics _metrics;
    private readonly StageConfig _config;

    public string ModelId { get; }
    public string StageName => _config.Name;
    public ModelMetrics Metrics => _metrics;
    public IReadOnlyList<string> SupportedModels => ["TFT", "NeuralOde"];

    internal GrpcSidecarTrainedModel(
        StageConfig config,
        string modelId,
        PythonTrainer.PythonTrainerClient client,
        string tempDir,
        ModelMetrics metrics)
    {
        _config = config;
        ModelId = modelId;
        _client = client;
        _tempDir = tempDir;
        _metrics = metrics;
    }

    public async Task<PredictionResult> PredictAsync(IDataFrame features, CancellationToken ct)
    {
        string featurePath = Path.Combine(_tempDir, $"{ModelId}_{Guid.NewGuid():N}_feat.arrow");
        string outputPath = Path.Combine(_tempDir, $"{ModelId}_{Guid.NewGuid():N}_pred.arrow");

        try
        {
            await ArrowIpcHelper.WriteAsync(features, featurePath, ct);

            var request = new PredictRequest
            {
                ModelId = ModelId,
                ArrowIpcPath = featurePath,
                OutputArrowIpcPath = outputPath,
            };
            foreach (var col in _config.Features)
                request.FeatureColumns.Add(col);

            var response = await _client.PredictAsync(request, cancellationToken: ct);

            if (!string.IsNullOrEmpty(response.ErrorMessage))
                throw new InvalidOperationException(
                    $"gRPC sidecar prediction error: {response.ErrorMessage} " +
                    $"(stage={StageName}, model_id={ModelId}).");

            float[] values = await ReadPredictionsAsync(response.OutputArrowIpcPath, ct);
            return new PredictionResult(values, null, null, ModelId, DateTime.UtcNow);
        }
        finally
        {
            TryDeleteFile(featurePath);
            TryDeleteFile(outputPath);
        }
    }

    // ReadPredictionsAsync: reads the Arrow IPC prediction file and enforces float32 (RULE-063).
    private static async Task<float[]> ReadPredictionsAsync(string path, CancellationToken ct)
    {
        if (!File.Exists(path))
            throw new FileNotFoundException(
                $"gRPC sidecar did not write prediction output to '{path}'.");

        await using var stream = File.OpenRead(path);
        using var reader = new ArrowFileReader(stream);
        var batch = await reader.ReadNextRecordBatchAsync(ct);

        if (batch is null || batch.ColumnCount == 0)
            throw new InvalidOperationException(
                $"Prediction Arrow IPC file '{path}' contains no columns or batches.");

        var col = batch.Column(0);

        // RULE-063: Validate float32 schema on read side.
        if (col.Data.DataType.TypeId != Apache.Arrow.Types.ArrowTypeId.Float)
            throw new InvalidOperationException(
                $"Prediction column has type '{col.Data.DataType}', expected float32 (RULE-063). " +
                $"Python stub must enforce pa.float32() schema.");

        var floatArray = (Apache.Arrow.FloatArray)col;
        var values = new float[floatArray.Length];
        for (int i = 0; i < floatArray.Length; i++)
            values[i] = floatArray.IsNull(i) ? float.NaN : floatArray.GetValue(i) ?? float.NaN;

        return values;
    }

    public Task SaveAsync(string directory, CancellationToken ct)
    {
        // Phase 2 stub: model artifacts live on Python side; no C# serialization needed.
        // Phase 4: call ExportOnnx RPC or copy artifacts from model_output_dir.
        return Task.CompletedTask;
    }

    public Task LoadAsync(string directory, CancellationToken ct) => Task.CompletedTask;

    private static void TryDeleteFile(string path)
    {
        try { File.Delete(path); } catch { /* best-effort cleanup */ }
    }
}
