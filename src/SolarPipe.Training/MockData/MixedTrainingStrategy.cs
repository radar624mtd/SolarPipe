using SolarPipe.Core.Interfaces;
using SolarPipe.Core.Models;
using SolarPipe.Data.DataFrame;

namespace SolarPipe.Training.MockData;

// Strategy: blend synthetic and observational data using sample weighting.
// Synthetic rows are duplicated proportionally to synthetic_weight / (1 - synthetic_weight)
// so the adapter sees a single merged dataset with natural class balance reflecting the weight.
// RULE-053: ENLIL temporal isolation is the caller's responsibility — this strategy only
// merges what it receives; the TrainCommand must ensure only pre-test-period ENLIL data
// is passed as syntheticData.
internal sealed class MixedTrainingStrategy : IMockDataStrategy
{
    private readonly IFrameworkAdapter _adapter;
    private readonly float _syntheticWeight;

    internal MixedTrainingStrategy(IFrameworkAdapter adapter, float syntheticWeight)
    {
        if (syntheticWeight is <= 0f or >= 1f)
            throw new ArgumentOutOfRangeException(
                nameof(syntheticWeight),
                $"SyntheticWeight must be in (0, 1); got {syntheticWeight}.");
        _adapter = adapter;
        _syntheticWeight = syntheticWeight;
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
                $"MixedTraining: syntheticData is empty. Stage: '{stage.Name}'.",
                nameof(syntheticData));
        if (observationalData.RowCount == 0)
            throw new ArgumentException(
                $"MixedTraining: observationalData is empty. Stage: '{stage.Name}'.",
                nameof(observationalData));

        var merged = MergeWithWeights(syntheticData, observationalData, _syntheticWeight, stage);
        return await _adapter.TrainAsync(stage, merged, null, ct);
    }

    internal static IDataFrame MergeWithWeights(
        IDataFrame synthetic, IDataFrame observational, float syntheticWeight, StageConfig stage)
    {
        // Compute how many synthetic rows to include relative to observational
        // Target: synthRows / (synthRows + obsRows) ≈ syntheticWeight
        int obsCount = observational.RowCount;
        int targetSynthCount = (int)Math.Round(obsCount * syntheticWeight / (1f - syntheticWeight));
        targetSynthCount = Math.Max(1, Math.Min(targetSynthCount, synthetic.RowCount));

        // Validate schema compatibility
        foreach (var col in stage.Features)
        {
            if (!synthetic.Schema.HasColumn(col))
                throw new InvalidOperationException(
                    $"MixedTraining: feature '{col}' missing from syntheticData. Stage: '{stage.Name}'.");
            if (!observational.Schema.HasColumn(col))
                throw new InvalidOperationException(
                    $"MixedTraining: feature '{col}' missing from observationalData. Stage: '{stage.Name}'.");
        }
        if (!synthetic.Schema.HasColumn(stage.Target))
            throw new InvalidOperationException(
                $"MixedTraining: target '{stage.Target}' missing from syntheticData. Stage: '{stage.Name}'.");
        if (!observational.Schema.HasColumn(stage.Target))
            throw new InvalidOperationException(
                $"MixedTraining: target '{stage.Target}' missing from observationalData. Stage: '{stage.Name}'.");

        // Build merged dataset using all columns present in observational schema
        var schema = observational.Schema;
        var numCols = schema.Columns.Count;
        var buffers = new float[numCols][];

        for (int c = 0; c < numCols; c++)
        {
            var colName = schema.Columns[c].Name;
            var obsCol = observational.GetColumn(colName);

            float[] synthCol;
            if (synthetic.Schema.HasColumn(colName))
            {
                var raw = synthetic.GetColumn(colName);
                synthCol = raw.Length > targetSynthCount
                    ? raw[..targetSynthCount]
                    : raw;
            }
            else
            {
                // Column not in synthetic data — fill with NaN
                synthCol = Enumerable.Repeat(float.NaN, targetSynthCount).ToArray();
            }

            int actualSynth = Math.Min(synthCol.Length, targetSynthCount);
            var merged = new float[actualSynth + obsCol.Length];
            synthCol.AsSpan(0, actualSynth).CopyTo(merged.AsSpan(0, actualSynth));
            obsCol.CopyTo(merged.AsSpan(actualSynth));
            buffers[c] = merged;
        }

        return new InMemoryDataFrame(schema, buffers);
    }
}
