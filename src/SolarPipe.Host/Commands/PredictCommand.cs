using System.Text.Json;
using SolarPipe.Config;
using SolarPipe.Core.Interfaces;
using SolarPipe.Core.Models;
using SolarPipe.Data;
using SolarPipe.Prediction;

namespace SolarPipe.Host.Commands;

public sealed class PredictCommand : ICommand
{
    private readonly PipelineConfigLoader _loader;
    private readonly DataSourceRegistry _dataRegistry;
    private readonly IModelRegistry _modelRegistry;

    public PredictCommand(
        PipelineConfigLoader loader,
        DataSourceRegistry dataRegistry,
        IModelRegistry modelRegistry)
    {
        _loader = loader;
        _dataRegistry = dataRegistry;
        _modelRegistry = modelRegistry;
    }

    public async Task<int> ExecuteAsync(string[] args, CancellationToken ct)
    {
        string configPath, inputPath, outputPath;
        try
        {
            configPath = ArgParser.Require(args, "--config");
            inputPath  = ArgParser.Require(args, "--input");
            outputPath = ArgParser.Require(args, "--output");
        }
        catch (ArgumentException ex)
        {
            Console.Error.WriteLine($"PREDICT_ERROR type=MissingArguments message=\"{ex.Message}\"");
            return ExitCodes.MissingArguments;
        }

        try
        {
            var config = await _loader.LoadAsync(configPath, ct);

            // Register pipeline data sources (needed for schema discovery; not used for input)
            foreach (var (key, _) in config.DataSources)
                _dataRegistry.RegisterSource(config.ToDataSourceConfig(key));

            // Load input CSV — the new, unseen events to predict
            var inputSource = new DataSourceConfig("__predict_input__", "csv", inputPath);
            _dataRegistry.RegisterSource(inputSource);
            var rawInput = await _dataRegistry.LoadAsync("__predict_input__", new DataQuery(), ct);

            // Compute derived features that TrainCommand generates via SQL but CSV doesn't have.
            // These mirror ExpandFeatureToSql in TrainCommand exactly.
            var input = AddDerivedFeatures(rawInput);

            // Load the latest trained model for each stage
            var stageModels = new Dictionary<string, ITrainedModel>(StringComparer.OrdinalIgnoreCase);
            foreach (var (stageName, _) in config.Stages)
            {
                var artifacts = await _modelRegistry.ListAsync(stageName, ct);
                if (artifacts.Count == 0)
                {
                    Console.Error.WriteLine(
                        $"PREDICT_ERROR stage={stageName} type=NoModel " +
                        $"message=\"No trained model found for stage '{stageName}'. Run train first.\"");
                    return ExitCodes.PredictionFailed;
                }
                var latest = artifacts.OrderByDescending(a => a.TrainedAt).First();
                stageModels[stageName] = await _modelRegistry.LoadAsync(latest.ModelId, latest.Version, ct);
                Console.WriteLine($"PREDICT_LOAD stage={stageName} model={latest.ModelId}");
            }

            // Build the composed model from the compose: directive.
            // This is what makes the prediction correct — the ^ operator produces
            // baseline_prediction + correction_prediction (where correction was trained on residuals).
            IComposedModel composed;
            if (!string.IsNullOrWhiteSpace(config.Compose))
            {
                composed = BuildComposedModel(config.Compose, stageModels);
                Console.WriteLine($"PREDICT_COMPOSE expression=\"{config.Compose}\" model={composed.Name}");
            }
            else
            {
                // No compose directive — fallback: run all stages independently
                Console.Error.WriteLine(
                    "PREDICT_WARN message=\"No compose directive in config; running stages independently\"");
                var results = new List<object>();
                foreach (var (stageName, model) in stageModels)
                {
                    using var stageCts = CancellationTokenSource.CreateLinkedTokenSource(ct);
                    stageCts.CancelAfter(TimeSpan.FromSeconds(60));
                    var result = await model.PredictAsync(input, stageCts.Token);
                    results.Add(new
                    {
                        stage = stageName,
                        model_id = result.ModelId,
                        generated_at = result.GeneratedAt,
                        values = result.Values,
                    });
                }
                return WriteOutput(results, outputPath, ct);
            }

            // Run the composed prediction
            using var predCts = CancellationTokenSource.CreateLinkedTokenSource(ct);
            predCts.CancelAfter(TimeSpan.FromSeconds(60));

            PredictionResult predResult;
            try
            {
                predResult = await composed.PredictAsync(input, predCts.Token);
            }
            catch (OperationCanceledException) when (!ct.IsCancellationRequested)
            {
                Console.Error.WriteLine(
                    "PREDICT_ERROR type=Timeout message=\"Composed prediction exceeded 60s.\"");
                return ExitCodes.PredictionFailed;
            }

            Console.WriteLine(
                $"PREDICT_OK model={predResult.ModelId} rows={predResult.Values.Length}");

            var output = new[]
            {
                new
                {
                    composed_model = composed.Name,
                    expression = config.Compose,
                    model_id = predResult.ModelId,
                    generated_at = predResult.GeneratedAt,
                    values = predResult.Values,
                    lower_bound = predResult.LowerBound,
                    upper_bound = predResult.UpperBound,
                }
            };

            return WriteOutput(output, outputPath, ct);
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine(
                $"PREDICT_ERROR type={ex.GetType().Name} message=\"{ex.Message}\"");
            return ExitCodes.PredictionFailed;
        }
    }

    // Compute derived features from raw CSV columns. Must mirror TrainCommand.ExpandFeatureToSql.
    // Called before prediction so the RF model finds the columns it was trained on.
    private static IDataFrame AddDerivedFeatures(IDataFrame raw)
    {
        int n = raw.RowCount;
        var result = raw;

        float[]? speed    = raw.Schema.HasColumn("cme_speed_kms")        ? raw.GetColumn("cme_speed_kms")        : null;
        float[]? swSpeed  = raw.Schema.HasColumn("sw_speed_ambient_kms") ? raw.GetColumn("sw_speed_ambient_kms") : null;
        float[]? bz       = raw.Schema.HasColumn("bz_gsm_proxy_nt")      ? raw.GetColumn("bz_gsm_proxy_nt")      : null;
        float[]? density  = raw.Schema.HasColumn("sw_density_n_cc")      ? raw.GetColumn("sw_density_n_cc")      : null;

        // delta_v_kms = cme_speed_kms - sw_speed_ambient_kms
        if (!raw.Schema.HasColumn("delta_v_kms") && speed != null && swSpeed != null)
        {
            var dv = new float[n];
            for (int i = 0; i < n; i++)
                dv[i] = (float.IsNaN(speed[i]) || float.IsNaN(swSpeed[i]))
                    ? float.NaN : speed[i] - swSpeed[i];
            result = result.AddColumn("delta_v_kms", dv);
        }

        // speed_ratio = cme_speed_kms / sw_speed_ambient_kms
        if (!raw.Schema.HasColumn("speed_ratio") && speed != null && swSpeed != null)
        {
            var sr = new float[n];
            for (int i = 0; i < n; i++)
                sr[i] = (float.IsNaN(speed[i]) || float.IsNaN(swSpeed[i]) || swSpeed[i] == 0f)
                    ? float.NaN : speed[i] / swSpeed[i];
            result = result.AddColumn("speed_ratio", sr);
        }

        // speed_x_bz = cme_speed_kms * bz_gsm_proxy_nt
        if (!raw.Schema.HasColumn("speed_x_bz") && speed != null && bz != null)
        {
            var sxbz = new float[n];
            for (int i = 0; i < n; i++)
                sxbz[i] = (float.IsNaN(speed[i]) || float.IsNaN(bz[i]))
                    ? float.NaN : speed[i] * bz[i];
            result = result.AddColumn("speed_x_bz", sxbz);
        }

        // speed_x_density = cme_speed_kms * sw_density_n_cc
        if (!raw.Schema.HasColumn("speed_x_density") && speed != null && density != null)
        {
            var sxd = new float[n];
            for (int i = 0; i < n; i++)
                sxd[i] = (float.IsNaN(speed[i]) || float.IsNaN(density[i]))
                    ? float.NaN : speed[i] * density[i];
            result = result.AddColumn("speed_x_density", sxd);
        }

        return result;
    }

    // Walk the parsed compose AST and construct the matching IComposedModel tree.
    private static IComposedModel BuildComposedModel(
        string expression,
        IReadOnlyDictionary<string, ITrainedModel> stageModels)
    {
        var parser = new ComposeExpressionParser();
        var ast = parser.Parse(expression);
        return BuildFromNode(ast, stageModels);
    }

    private static IComposedModel BuildFromNode(
        ComposeNode node,
        IReadOnlyDictionary<string, ITrainedModel> stageModels)
    {
        return node switch
        {
            IdentifierNode id => new SingleModelAdapter(
                stageModels.TryGetValue(id.Name, out var m)
                    ? m
                    : throw new InvalidOperationException(
                        $"Compose references stage '{id.Name}' which has no trained model. " +
                        $"Available: [{string.Join(", ", stageModels.Keys)}]."),
                id.Name),

            ResidualNode rn =>
                new ResidualModel(
                    ExtractModel(BuildFromNode(rn.Left, stageModels), rn.Left),
                    ExtractModel(BuildFromNode(rn.Right, stageModels), rn.Right),
                    $"{rn.Left}^{rn.Right}"),

            ChainNode cn =>
                new ChainedModel(
                    ExtractModel(BuildFromNode(cn.Left, stageModels), cn.Left),
                    ExtractModel(BuildFromNode(cn.Right, stageModels), cn.Right),
                    $"{cn.Left}->{cn.Right}"),

            EnsembleNode en =>
                new EnsembleModel(
                    [
                        ExtractModel(BuildFromNode(en.Left, stageModels), en.Left),
                        ExtractModel(BuildFromNode(en.Right, stageModels), en.Right),
                    ],
                    name: $"{en.Left}+{en.Right}"),

            GateNode gn =>
                new GatedModel(
                    ExtractModel(BuildFromNode(gn.Classifier, stageModels), gn.Classifier),
                    ExtractModel(BuildFromNode(gn.IfTrue, stageModels), gn.IfTrue),
                    ExtractModel(BuildFromNode(gn.IfFalse, stageModels), gn.IfFalse),
                    name: $"{gn.Classifier}?({gn.IfTrue},{gn.IfFalse})"),

            _ => throw new InvalidOperationException(
                $"Unhandled compose AST node type: {node.GetType().Name}")
        };
    }

    // ResidualModel and ChainedModel take ITrainedModel for one operand.
    // Unwrap SingleModelAdapter; for nested composites, wrap them so the algebra holds.
    private static ITrainedModel ExtractModel(IComposedModel composed, ComposeNode sourceNode)
    {
        if (composed is SingleModelAdapter adapter)
            return adapter.Model;
        // Wrap composite in a trainable-model shim so it can be used as a composed sub-tree
        return new ComposedModelTrainedShim(composed);
    }

    private static int WriteOutput(object output, string outputPath, CancellationToken ct)
    {
        var json = JsonSerializer.Serialize(output,
            new JsonSerializerOptions
            {
                WriteIndented = true,
                // Infinity values are not valid JSON and not physically meaningful predictions.
                // This option writes them as the JSON string "Infinity" — the backtest reader
                // treats non-numeric values as prediction failures (correct behavior).
                NumberHandling = System.Text.Json.Serialization.JsonNumberHandling.AllowNamedFloatingPointLiterals
            });
        var tempPath = outputPath + $".tmp_{Guid.NewGuid():N}";
        File.WriteAllText(tempPath, json);
        File.Move(tempPath, outputPath, overwrite: true);
        Console.WriteLine($"OK output={outputPath}");
        return ExitCodes.Success;
    }
}

// Wraps a single ITrainedModel as an IComposedModel so it can be used in the compose tree.
internal sealed class SingleModelAdapter(ITrainedModel model, string name) : IComposedModel
{
    public ITrainedModel Model { get; } = model;
    public string Name { get; } = name;
    public Task<PredictionResult> PredictAsync(IDataFrame input, CancellationToken ct) =>
        Model.PredictAsync(input, ct);
}

// Wraps an IComposedModel as an ITrainedModel shim for use in ResidualModel/ChainedModel constructors.
internal sealed class ComposedModelTrainedShim(IComposedModel inner) : ITrainedModel
{
    public string ModelId => inner.Name;
    public string StageName => inner.Name;
    public ModelMetrics Metrics => new(0, 0, 0);
    public Task<PredictionResult> PredictAsync(IDataFrame input, CancellationToken ct) =>
        inner.PredictAsync(input, ct);
    public Task SaveAsync(string path, CancellationToken ct) =>
        throw new NotSupportedException("ComposedModelTrainedShim.SaveAsync not supported.");
    public Task LoadAsync(string path, CancellationToken ct) =>
        throw new NotSupportedException("ComposedModelTrainedShim.LoadAsync not supported.");
}
