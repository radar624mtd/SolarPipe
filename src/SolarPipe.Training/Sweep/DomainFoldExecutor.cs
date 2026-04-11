using SolarPipe.Core.Interfaces;
using SolarPipe.Core.Models;
using SolarPipe.Data.DataFrame;
using SolarPipe.Training.Features;
using SolarPipe.Training.Physics;

namespace SolarPipe.Training.Sweep;

// Executes one CV fold of the three-domain pipeline:
//   D1 (Origination RF) → D2 (DragBased + RF residual) → D3 (BurtonOde + RF residual)
//   → Meta-learner (three separate RF models)
//
// All frames are immutable IDataFrame: AddColumn returns new instances.
// Physics models are instantiated per fold (stateless parameter extraction, no ML fitting).
// ML models are trained per fold.
//
// Returns per-fold MAE for transit_time_hours, dst_min_nt, storm_duration_hours,
// and physics-only baseline MAE on transit_time_hours for comparison.
internal sealed class DomainFoldExecutor
{
    private readonly IReadOnlyList<IFrameworkAdapter> _adapters;

    public DomainFoldExecutor(IReadOnlyList<IFrameworkAdapter> adapters)
    {
        _adapters = adapters ?? throw new ArgumentNullException(nameof(adapters));
    }

    public async Task<DomainFoldMetrics> ExecuteAsync(
        int foldIndex,
        IDataFrame trainFrame,
        IDataFrame testFrame,
        DomainPipelineRunConfig config,
        CancellationToken ct)
    {
        string tag = $"[domain_fold:{foldIndex}]";

        // ── Domain 1: Origination ─────────────────────────────────────────────
        // Generate DragBasedV2 ODE pseudo-labels as training targets for origination RF.
        // The RF learns to predict arrival speed from source magnetic/geometric properties.
        Console.WriteLine($"{tag} D1:origination training...");
        var originConfig = config.Stages[config.Origination.Stage!];
        var dragV2Config = config.Stages[config.Transit.PhysicsStage!];
        var physicsAdapter = ResolveAdapter("Physics");

        var dragV2 = await physicsAdapter.TrainAsync(
            new StageConfig("drag_v2_pseudo", "Physics", "DragBasedV2", "",
                dragV2Config.Features, "arrival_speed_kms", dragV2Config.Hyperparameters),
            trainFrame, null, ct);

        var d1PseudoLabelsTrain = await dragV2.PredictAsync(trainFrame, ct);
        var d1PseudoLabelsTest  = await dragV2.PredictAsync(testFrame, ct);

        // Append pseudo-label as the training target column
        var d1TrainWithTarget = AppendColumn(trainFrame, "arrival_speed_kms", d1PseudoLabelsTrain.Values);

        var mlAdapter = ResolveAdapter("MlNet");
        var d1Model = await mlAdapter.TrainAsync(originConfig, d1TrainWithTarget, null, ct);
        var d1PredsTrain = await d1Model.PredictAsync(trainFrame, ct);
        var d1PredsTest  = await d1Model.PredictAsync(testFrame, ct);

        // ── Domain 2: Transit ─────────────────────────────────────────────────
        Console.WriteLine($"{tag} D2:transit physics training...");
        var dragConfig    = config.Stages[config.Transit.PhysicsStage!];
        var transitRfConf = config.Stages[config.Transit.ResidualStage!];

        var dragModel = await physicsAdapter.TrainAsync(dragConfig, trainFrame, null, ct);
        var dragPredsTrain = await dragModel.PredictAsync(trainFrame, ct);
        var dragPredsTest  = await dragModel.PredictAsync(testFrame, ct);

        // Build augmented transit frames with D1 output + derived transit features
        var d2TrainAug = BuildTransitFrame(trainFrame, d1PredsTrain.Values);
        var d2TestAug  = BuildTransitFrame(testFrame,  d1PredsTest.Values);

        // Compute residual target: observed - physics prediction
        float[] transitObs   = trainFrame.GetColumn("transit_time_hours");
        float[] residualTgt  = SubtractNaN(transitObs, dragPredsTrain.Values);
        var d2TrainForRf = AppendColumn(d2TrainAug, "transit_residual", residualTgt);

        Console.WriteLine($"{tag} D2:transit RF residual training...");
        var transitRfForTrain = transitRfConf with { Target = "transit_residual" };
        var transitRfModel = await mlAdapter.TrainAsync(transitRfForTrain, d2TrainForRf, null, ct);
        var d2RfPredsTrain = await transitRfModel.PredictAsync(d2TrainAug, ct);
        var d2RfPredsTest  = await transitRfModel.PredictAsync(d2TestAug, ct);

        // D2 final = physics + RF residual
        float[] d2PredsTrain = AddNaN(dragPredsTrain.Values, d2RfPredsTrain.Values);
        float[] d2PredsTest  = AddNaN(dragPredsTest.Values,  d2RfPredsTest.Values);

        // ── Domain 3: Impact ──────────────────────────────────────────────────
        Console.WriteLine($"{tag} D3:impact physics training...");
        var burtonConfig  = config.Stages[config.Impact.PhysicsStage!];
        var impactRfConf  = config.Stages[config.Impact.ResidualStage!];

        var burtonModel = await physicsAdapter.TrainAsync(burtonConfig, trainFrame, null, ct);
        var burtonPredsTrain = await burtonModel.PredictAsync(trainFrame, ct);
        var burtonPredsTest  = await burtonModel.PredictAsync(testFrame, ct);

        // Build augmented impact frames with D1+D2 outputs + derived impact features
        var d3TrainAug = BuildImpactFrame(trainFrame, d1PredsTrain.Values, d2PredsTrain);
        var d3TestAug  = BuildImpactFrame(testFrame,  d1PredsTest.Values,  d2PredsTest);

        // Residual target: observed Dst - physics prediction
        float[] dstObs      = trainFrame.GetColumn("dst_min_nt");
        float[] dstResidual = SubtractNaN(dstObs, burtonPredsTrain.Values);
        var d3TrainForRf = AppendColumn(d3TrainAug, "dst_residual", dstResidual);

        Console.WriteLine($"{tag} D3:impact RF residual training...");
        var impactRfForTrain = impactRfConf with { Target = "dst_residual" };
        var impactRfModel = await mlAdapter.TrainAsync(impactRfForTrain, d3TrainForRf, null, ct);
        var d3RfPredsTrain = await impactRfModel.PredictAsync(d3TrainAug, ct);
        var d3RfPredsTest  = await impactRfModel.PredictAsync(d3TestAug, ct);

        float[] d3PredsTrain = AddNaN(burtonPredsTrain.Values, d3RfPredsTrain.Values);
        float[] d3PredsTest  = AddNaN(burtonPredsTest.Values,  d3RfPredsTest.Values);

        // ── Meta-Learner ──────────────────────────────────────────────────────
        Console.WriteLine($"{tag} meta-learner training...");
        var metaTrainFrame = BuildMetaFrame(trainFrame, d1PredsTrain.Values, d2PredsTrain, d3PredsTrain);
        var metaTestFrame  = BuildMetaFrame(testFrame,  d1PredsTest.Values,  d2PredsTest,  d3PredsTest);

        // Train and predict for each meta output
        float[] metaArrivalTest  = await TrainAndPredictAsync(mlAdapter, config, "arrival_time_hours",
            "transit_time_hours", metaTrainFrame, metaTestFrame, ct);
        float[] metaIntensityTest = await TrainAndPredictAsync(mlAdapter, config, "storm_intensity_nt",
            "dst_min_nt", metaTrainFrame, metaTestFrame, ct);
        float[] metaDurationTest  = await TrainAndPredictAsync(mlAdapter, config, "storm_duration_hours",
            "storm_duration_hours", metaTrainFrame, metaTestFrame, ct);

        // ── Physics baseline comparison ───────────────────────────────────────
        // DragBased-only MAE on test set transit_time_hours (for Phase 7 H1 comparison)
        double physicsBaselineMae = ComputeMae(
            testFrame.GetColumn("transit_time_hours"),
            dragPredsTest.Values);

        // ── Fold metrics ──────────────────────────────────────────────────────
        float[] obsTransit = testFrame.GetColumn("transit_time_hours");
        float[] obsDst     = testFrame.GetColumn("dst_min_nt");

        double maeTransit  = ComputeMae(obsTransit,       metaArrivalTest);
        double maeDst      = ComputeMae(obsDst,           metaIntensityTest);
        double maeDuration = ComputeMae(testFrame.GetColumn("storm_duration_hours"), metaDurationTest);

        Console.WriteLine($"{tag} done. MAE: transit={maeTransit:F2}h dst={maeDst:F2}nT " +
                          $"duration={maeDuration:F2}h baseline={physicsBaselineMae:F2}h");

        // Per-event predictions (keyed by launch_time Unix seconds for held-out evaluation)
        float[] launchTimes = testFrame.Schema.HasColumn("launch_time")
            ? testFrame.GetColumn("launch_time")
            : new float[testFrame.RowCount];

        var eventPreds = new List<EventPrediction>(testFrame.RowCount);
        for (int i = 0; i < testFrame.RowCount; i++)
        {
            eventPreds.Add(new EventPrediction(
                LaunchTimeUnix: launchTimes[i],
                ObsTransit:     obsTransit[i],
                PredTransit:    metaArrivalTest[i],
                ObsDst:         obsDst[i],
                PredDst:        metaIntensityTest[i],
                PhysicsTransit: dragPredsTest.Values[i]));
        }

        return new DomainFoldMetrics(foldIndex, maeTransit, maeDst, maeDuration, physicsBaselineMae, eventPreds);
    }

    // ── Frame builders ────────────────────────────────────────────────────────

    private static IDataFrame BuildTransitFrame(IDataFrame frame, float[] d1Preds)
    {
        var f = AppendColumn(frame, "pred_arrival_speed_kms", d1Preds);
        f = DomainFeatureTransforms.AddTransitFeatures(f);
        f = DomainFeatureTransforms.AddDeltaV(f);
        return f;
    }

    private static IDataFrame BuildImpactFrame(IDataFrame frame, float[] d1Preds, float[] d2Preds)
    {
        var f = AppendColumn(frame, "pred_arrival_speed_kms", d1Preds);
        f = AppendColumn(f, "pred_transit_time_hours", d2Preds);
        f = DomainFeatureTransforms.AddImpactFeatures(f);
        return f;
    }

    private static IDataFrame BuildMetaFrame(
        IDataFrame frame,
        float[] d1Preds,
        float[] d2Preds,
        float[] d3Preds)
    {
        var f = AppendColumn(frame, "pred_origination_arrival_speed_kms", d1Preds);
        f = AppendColumn(f, "pred_transit_time_hours", d2Preds);
        f = AppendColumn(f, "pred_dst_min_nt",         d3Preds);
        f = DomainFeatureTransforms.AddImpactFeatures(f);
        return f;
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    private async Task<float[]> TrainAndPredictAsync(
        IFrameworkAdapter adapter,
        DomainPipelineRunConfig config,
        string outputName,
        string targetColumn,
        IDataFrame trainFrame,
        IDataFrame testFrame,
        CancellationToken ct)
    {
        if (!config.MetaLearnerStages.TryGetValue(outputName, out var stageName))
            throw new InvalidOperationException(
                $"DomainFoldExecutor: no meta-learner stage mapped to output '{outputName}'.");

        var stageConf = config.Stages[stageName] with { Target = targetColumn };
        var model = await adapter.TrainAsync(stageConf, trainFrame, null, ct);
        var preds = await model.PredictAsync(testFrame, ct);
        return preds.Values;
    }

    private IFrameworkAdapter ResolveAdapter(string framework)
    {
        var adapter = _adapters.FirstOrDefault(a =>
            string.Equals(a.FrameworkType.ToString(), framework, StringComparison.OrdinalIgnoreCase));
        return adapter ?? throw new InvalidOperationException(
            $"DomainFoldExecutor: no adapter for framework='{framework}'.");
    }

    // Appends a column if not already present; returns same frame if already present.
    private static IDataFrame AppendColumn(IDataFrame frame, string name, float[] values)
    {
        if (frame.Schema.HasColumn(name)) return frame;
        return frame.AddColumn(name, values);
    }

    // Element-wise subtraction: NaN if either operand is NaN.
    private static float[] SubtractNaN(float[] a, float[] b)
    {
        var result = new float[a.Length];
        for (int i = 0; i < a.Length; i++)
            result[i] = (float.IsNaN(a[i]) || float.IsNaN(b[i])) ? float.NaN : a[i] - b[i];
        return result;
    }

    // Element-wise addition: NaN if either operand is NaN.
    private static float[] AddNaN(float[] a, float[] b)
    {
        var result = new float[a.Length];
        for (int i = 0; i < a.Length; i++)
            result[i] = (float.IsNaN(a[i]) || float.IsNaN(b[i])) ? float.NaN : a[i] + b[i];
        return result;
    }

    private static double ComputeMae(float[] observed, float[] predicted)
    {
        double sum = 0.0;
        int count = 0;
        for (int i = 0; i < observed.Length; i++)
        {
            if (float.IsNaN(observed[i]) || float.IsNaN(predicted[i])) continue;
            sum += Math.Abs(observed[i] - predicted[i]);
            count++;
        }
        return count > 0 ? sum / count : double.NaN;
    }
}
