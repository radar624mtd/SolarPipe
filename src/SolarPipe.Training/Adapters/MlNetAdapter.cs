using Microsoft.ML;
using Microsoft.ML.Trainers.FastTree;
using SolarPipe.Core.Interfaces;
using SolarPipe.Core.Models;

namespace SolarPipe.Training.Adapters;

public sealed class MlNetAdapter : IFrameworkAdapter
{
    public FrameworkType FrameworkType => FrameworkType.MlNet;

    public IReadOnlyList<string> SupportedModels => new[] { "FastForest", "FastTree" };

    public async Task<ITrainedModel> TrainAsync(
        StageConfig config,
        IDataFrame trainingData,
        IDataFrame? validationData,
        CancellationToken ct)
    {
        // RULE-011: Fresh MLContext per training call with pinned seed
        var mlContext = new MLContext(seed: 42);

        var trainView = trainingData.ToDataView(mlContext);

        var featurePipeline = mlContext.Transforms.Concatenate(
            "Features", config.Features.ToArray());

        var trainer = SelectTrainer(mlContext, config);

        var fullPipeline = featurePipeline.Append(trainer);

        // RULE-150: LongRunning for CPU-bound ML.NET training — avoids thread pool starvation
        var model = await Task.Factory.StartNew(
            () => fullPipeline.Fit(trainView),
            ct,
            TaskCreationOptions.LongRunning,
            TaskScheduler.Default);

        Microsoft.ML.Data.RegressionMetrics? regressionMetrics = null;
        if (validationData != null)
        {
            var valView = validationData.ToDataView(mlContext);
            regressionMetrics = mlContext.Regression.Evaluate(
                model.Transform(valView),
                labelColumnName: config.Target);
        }

        var metrics = regressionMetrics != null
            ? new ModelMetrics(
                regressionMetrics.RootMeanSquaredError,
                regressionMetrics.MeanAbsoluteError,
                regressionMetrics.RSquared)
            : new ModelMetrics(double.NaN, double.NaN, double.NaN);

        return new MlNetTrainedModel(config, model, mlContext, metrics);
    }

    private static IEstimator<ITransformer> SelectTrainer(MLContext mlContext, StageConfig config)
    {
        return config.ModelType switch
        {
            "FastForest" => mlContext.Regression.Trainers.FastForest(
                new FastForestRegressionTrainer.Options
                {
                    LabelColumnName = config.Target,
                    NumberOfTrees = GetHyperInt(config, "number_of_trees", 100),
                    NumberOfLeaves = GetHyperInt(config, "number_of_leaves", 31),
                    // RULE-010: Never default — 1.0 disables RF diversity mechanism
                    FeatureFraction = GetHyperFloat(config, "feature_fraction", 0.7f),
                    // RULE-011: Pin trainer-specific seed for reproducibility
                    FeatureSelectionSeed = GetHyperInt(config, "feature_selection_seed", 42),
                    // RULE-011: Single-threaded for deterministic results
                    NumberOfThreads = GetHyperInt(config, "number_of_threads", 1),
                }),
            "FastTree" => mlContext.Regression.Trainers.FastTree(
                new FastTreeRegressionTrainer.Options
                {
                    LabelColumnName = config.Target,
                    NumberOfTrees = GetHyperInt(config, "number_of_trees", 100),
                    NumberOfLeaves = GetHyperInt(config, "number_of_leaves", 31),
                    // RULE-010: explicit feature fraction
                    FeatureFraction = GetHyperFloat(config, "feature_fraction", 0.7f),
                    FeatureSelectionSeed = GetHyperInt(config, "feature_selection_seed", 42),
                    NumberOfThreads = GetHyperInt(config, "number_of_threads", 1),
                }),
            _ => throw new NotSupportedException(
                $"Model type '{config.ModelType}' is not supported by MlNetAdapter. " +
                $"Supported: {string.Join(", ", new[] { "FastForest", "FastTree" })}. " +
                $"Stage: '{config.Name}'.")
        };
    }

    internal static int GetHyperInt(StageConfig config, string key, int defaultValue)
    {
        if (config.Hyperparameters == null) return defaultValue;
        var raw = FindHyperValue(config.Hyperparameters, key);
        if (raw == null) return defaultValue;
        return raw switch
        {
            int i => i,
            long l => (int)l,
            string s when int.TryParse(s, out var parsed) => parsed,
            _ => Convert.ToInt32(raw)
        };
    }

    private static object? FindHyperValue(IReadOnlyDictionary<string, object> hypers, string key)
    {
        foreach (var kv in hypers)
            if (string.Equals(kv.Key, key, StringComparison.OrdinalIgnoreCase))
                return kv.Value;
        return null;
    }

    internal static float GetHyperFloat(StageConfig config, string key, float defaultValue)
    {
        if (config.Hyperparameters == null) return defaultValue;
        var raw = FindHyperValue(config.Hyperparameters, key);
        if (raw == null) return defaultValue;
        return raw switch
        {
            float f => f,
            double d => (float)d,
            string s when float.TryParse(s, System.Globalization.NumberStyles.Float,
                System.Globalization.CultureInfo.InvariantCulture, out var parsed) => parsed,
            _ => Convert.ToSingle(raw)
        };
    }
}
