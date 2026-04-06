using SolarPipe.Core.Interfaces;
using SolarPipe.Core.Models;
using SolarPipe.Data.DataFrame;

namespace SolarPipe.Training.MockData;

public enum MockDataStrategyType
{
    PretrainThenFinetune,
    MixedTraining,
    ResidualCalibration,
}

public sealed record MockDataConfig(
    MockDataStrategyType StrategyType,
    float SyntheticWeight = 0.5f,
    string? SyntheticTimestampColumn = null);

public static class MockDataStrategyFactory
{
    public static IMockDataStrategy Create(MockDataConfig config, IFrameworkAdapter adapter) =>
        config.StrategyType switch
        {
            MockDataStrategyType.PretrainThenFinetune =>
                new PretrainThenFinetuneStrategy(adapter),
            MockDataStrategyType.MixedTraining =>
                new MixedTrainingStrategy(adapter, config.SyntheticWeight),
            MockDataStrategyType.ResidualCalibration =>
                new ResidualCalibrationStrategy(adapter),
            _ => throw new ArgumentOutOfRangeException(
                nameof(config),
                $"Unknown MockDataStrategyType: {config.StrategyType}. Stage: unknown, Config: {config}")
        };
}

public interface IMockDataStrategy
{
    Task<ITrainedModel> TrainAsync(
        StageConfig stage,
        IDataFrame syntheticData,
        IDataFrame observationalData,
        CancellationToken ct);
}
