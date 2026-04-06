using SolarPipe.Core.Interfaces;
using SolarPipe.Core.Models;

namespace SolarPipe.Training.MockData;

// Strategy: pretrain on synthetic data, then finetune on observational data.
// The adapter's TrainAsync is called twice — first with synthetic data as training
// and observational as validation, then again with observational data only.
// The second call produces the returned model (fine-tuned).
internal sealed class PretrainThenFinetuneStrategy : IMockDataStrategy
{
    private readonly IFrameworkAdapter _adapter;

    internal PretrainThenFinetuneStrategy(IFrameworkAdapter adapter)
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
                $"PretrainThenFinetune: syntheticData is empty. Stage: '{stage.Name}'.",
                nameof(syntheticData));
        if (observationalData.RowCount == 0)
            throw new ArgumentException(
                $"PretrainThenFinetune: observationalData is empty. Stage: '{stage.Name}'.",
                nameof(observationalData));

        // Phase 1: pretrain on synthetic
        await _adapter.TrainAsync(stage, syntheticData, observationalData, ct);

        ct.ThrowIfCancellationRequested();

        // Phase 2: finetune on observational data only — this is the returned model
        return await _adapter.TrainAsync(stage, observationalData, null, ct);
    }
}
