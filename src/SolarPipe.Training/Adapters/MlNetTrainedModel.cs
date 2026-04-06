using Microsoft.ML;
using SolarPipe.Core.Interfaces;
using SolarPipe.Core.Models;

namespace SolarPipe.Training.Adapters;

public sealed class MlNetTrainedModel : ITrainedModel
{
    private readonly ITransformer _model;
    private readonly MLContext _mlContext;
    private readonly StageConfig _config;

    public string ModelId { get; }
    public string StageName => _config.Name;
    public ModelMetrics Metrics { get; }

    public MlNetTrainedModel(
        StageConfig config,
        ITransformer model,
        MLContext mlContext,
        ModelMetrics metrics)
    {
        _config = config ?? throw new ArgumentNullException(nameof(config));
        _model = model ?? throw new ArgumentNullException(nameof(model));
        _mlContext = mlContext ?? throw new ArgumentNullException(nameof(mlContext));
        Metrics = metrics ?? throw new ArgumentNullException(nameof(metrics));

        ModelId = $"{config.Name}_{config.ModelType}";
    }

    public Task<PredictionResult> PredictAsync(IDataFrame input, CancellationToken ct)
    {
        var dataView = input.ToDataView(_mlContext);
        var predictions = _model.Transform(dataView);

        // Extract the "Score" column produced by ML.NET regression trainers
        var scoreColumn = predictions.Schema["Score"];
        var cursor = predictions.GetRowCursor([scoreColumn]);

        var values = new List<float>();
        var getter = cursor.GetGetter<float>(scoreColumn);

        while (cursor.MoveNext())
        {
            float score = 0f;
            getter(ref score);
            values.Add(score);
        }

        var result = new PredictionResult(
            Values: values.ToArray(),
            LowerBound: null,
            UpperBound: null,
            ModelId: ModelId,
            GeneratedAt: DateTime.UtcNow);

        return Task.FromResult(result);
    }

    public Task SaveAsync(string path, CancellationToken ct)
    {
        if (string.IsNullOrWhiteSpace(path))
            throw new ArgumentException("Save path must not be empty. Stage: " + StageName, nameof(path));

        Directory.CreateDirectory(Path.GetDirectoryName(path) ?? ".");
        _mlContext.Model.Save(_model, null, path);
        return Task.CompletedTask;
    }

    public Task LoadAsync(string path, CancellationToken ct)
    {
        throw new NotSupportedException(
            $"LoadAsync is not supported on MlNetTrainedModel instances. " +
            $"Use FileSystemModelRegistry.LoadAsync to deserialize a saved model. Stage: '{StageName}'.");
    }
}
