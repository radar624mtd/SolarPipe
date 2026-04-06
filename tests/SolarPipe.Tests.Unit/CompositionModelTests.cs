using FluentAssertions;
using NSubstitute;
using SolarPipe.Core.Interfaces;
using SolarPipe.Core.Models;
using SolarPipe.Data.DataFrame;
using SolarPipe.Prediction;

namespace SolarPipe.Tests.Unit;

[Trait("Category", "Unit")]
public class CompositionModelTests
{
    // ─── helpers ────────────────────────────────────────────────────────────────

    private static InMemoryDataFrame MakeFrame(string colName, float[] values)
    {
        var schema = new DataSchema([new ColumnInfo(colName, ColumnType.Float, false)]);
        return new InMemoryDataFrame(schema, [values]);
    }

    private static InMemoryDataFrame MakeTwoColFrame(
        string col1, float[] v1, string col2, float[] v2)
    {
        var schema = new DataSchema([
            new ColumnInfo(col1, ColumnType.Float, false),
            new ColumnInfo(col2, ColumnType.Float, false)
        ]);
        return new InMemoryDataFrame(schema, [v1, v2]);
    }

    private static ITrainedModel MakeModel(string id, float[] returnValues,
        float[]? lower = null, float[]? upper = null)
    {
        var model = Substitute.For<ITrainedModel>();
        model.ModelId.Returns(id);
        model.StageName.Returns(id);
        model.Metrics.Returns(new ModelMetrics(0, 0, 0));
        model.PredictAsync(Arg.Any<IDataFrame>(), Arg.Any<CancellationToken>())
            .Returns(Task.FromResult(new PredictionResult(
                returnValues, lower, upper, id, DateTime.UtcNow)));
        return model;
    }

    // ─── ChainedModel ────────────────────────────────────────────────────────────

    [Fact]
    public async Task ChainedModel_OutputFromLeftAppendsColumnForRight()
    {
        // Left returns [10f, 20f]. Right should see a frame with an extra column.
        var leftValues = new float[] { 10f, 20f };
        var rightValues = new float[] { 100f, 200f };

        IDataFrame? capturedFrame = null;
        var right = Substitute.For<ITrainedModel>();
        right.ModelId.Returns("right");
        right.StageName.Returns("right");
        right.Metrics.Returns(new ModelMetrics(0, 0, 0));
        right.PredictAsync(Arg.Do<IDataFrame>(f => capturedFrame = f), Arg.Any<CancellationToken>())
            .Returns(Task.FromResult(new PredictionResult(rightValues, null, null, "right", DateTime.UtcNow)));

        var left = MakeModel("left", leftValues);
        var chained = new ChainedModel(left, right, "test_chain");

        using var input = MakeFrame("speed", [500f, 800f]);
        var result = await chained.PredictAsync(input, CancellationToken.None);

        result.Values.Should().BeEquivalentTo(rightValues);
        capturedFrame!.Schema.HasColumn("chained_left").Should().BeTrue(
            "left output must be appended as 'chained_left' column");
        capturedFrame.RowCount.Should().Be(2);
    }

    [Fact]
    public async Task ChainedModel_NanFromLeft_PropagatesAsColumn()
    {
        var leftValues = new float[] { float.NaN, 20f };
        var left = MakeModel("left", leftValues);

        IDataFrame? capturedFrame = null;
        var right = Substitute.For<ITrainedModel>();
        right.ModelId.Returns("right");
        right.StageName.Returns("right");
        right.Metrics.Returns(new ModelMetrics(0, 0, 0));
        right.PredictAsync(Arg.Do<IDataFrame>(f => capturedFrame = f), Arg.Any<CancellationToken>())
            .Returns(Task.FromResult(new PredictionResult([50f, 60f], null, null, "right", DateTime.UtcNow)));

        var chained = new ChainedModel(left, right, "nan_chain");
        using var input = MakeFrame("speed", [float.NaN, 800f]);

        await chained.PredictAsync(input, CancellationToken.None);

        var chainedCol = capturedFrame!.GetColumn("chained_left");
        float.IsNaN(chainedCol[0]).Should().BeTrue("NaN from left must pass through to right input");
        chainedCol[1].Should().Be(20f);
    }

    [Fact]
    public void ChainedModel_NullLeft_Throws()
    {
        var right = MakeModel("right", [1f]);
        var act = () => new ChainedModel(null!, right);
        act.Should().Throw<ArgumentNullException>();
    }

    [Fact]
    public async Task ChainedModel_MismatchedOutputLength_Throws()
    {
        var left = MakeModel("left", [1f, 2f, 3f]); // returns 3 values for 2-row input → mismatch
        var right = MakeModel("right", [10f, 20f]);
        var chained = new ChainedModel(left, right);
        using var input = MakeFrame("speed", [500f, 800f]);

        var act = async () => await chained.PredictAsync(input, CancellationToken.None);
        await act.Should().ThrowAsync<InvalidOperationException>().WithMessage("*3 predictions*");
    }

    // ─── ResidualModel ───────────────────────────────────────────────────────────

    [Fact]
    public async Task ResidualModel_SumBasePlusCorrection()
    {
        var baseline = MakeModel("physics", [30f, 50f, 70f]);
        var correction = MakeModel("rf", [5f, -3f, 2f]);
        var residual = new ResidualModel(baseline, correction, "physics^rf");

        using var input = MakeFrame("speed", [800f, 1200f, 2000f]);
        var result = await residual.PredictAsync(input, CancellationToken.None);

        result.Values[0].Should().BeApproximately(35f, 0.001f);
        result.Values[1].Should().BeApproximately(47f, 0.001f);
        result.Values[2].Should().BeApproximately(72f, 0.001f);
    }

    [Fact]
    public async Task ResidualModel_BaselineColumnAppendedForCorrection()
    {
        IDataFrame? capturedFrame = null;
        var correction = Substitute.For<ITrainedModel>();
        correction.ModelId.Returns("rf");
        correction.StageName.Returns("rf");
        correction.Metrics.Returns(new ModelMetrics(0, 0, 0));
        correction.PredictAsync(Arg.Do<IDataFrame>(f => capturedFrame = f), Arg.Any<CancellationToken>())
            .Returns(Task.FromResult(new PredictionResult([1f, 2f], null, null, "rf", DateTime.UtcNow)));

        var baseline = MakeModel("physics", [30f, 50f]);
        var residual = new ResidualModel(baseline, correction);

        using var input = MakeFrame("speed", [800f, 1200f]);
        await residual.PredictAsync(input, CancellationToken.None);

        capturedFrame!.Schema.HasColumn("residual_baseline_physics").Should().BeTrue(
            "baseline predictions must be appended as feature for correction model");
        capturedFrame.GetColumn("residual_baseline_physics")
            .Should().BeEquivalentTo([30f, 50f]);
    }

    [Fact]
    public async Task ResidualModel_NanBaseline_PropagatesNanToOutput()
    {
        var baseline = MakeModel("physics", [float.NaN, 50f]);
        var correction = MakeModel("rf", [5f, -3f]);
        var residual = new ResidualModel(baseline, correction);

        using var input = MakeFrame("speed", [float.NaN, 800f]);
        var result = await residual.PredictAsync(input, CancellationToken.None);

        float.IsNaN(result.Values[0]).Should().BeTrue("NaN baseline must produce NaN output");
        result.Values[1].Should().BeApproximately(47f, 0.001f);
    }

    [Fact]
    public async Task ResidualModel_WithUncertaintyBounds_CombinesInQuadrature()
    {
        // baseline half-width = 4, correction half-width = 3 → combined = sqrt(16+9) = 5
        var baseline = MakeModel("physics", [30f],
            lower: [26f], upper: [34f]); // half-width 4
        var correction = MakeModel("rf", [5f],
            lower: [2f], upper: [8f]); // half-width 3
        var residual = new ResidualModel(baseline, correction);

        using var input = MakeFrame("speed", [800f]);
        var result = await residual.PredictAsync(input, CancellationToken.None);

        result.Values[0].Should().BeApproximately(35f, 0.001f);
        result.LowerBound.Should().NotBeNull();
        result.UpperBound.Should().NotBeNull();
        float expectedHalf = MathF.Sqrt(16f + 9f); // = 5
        result.LowerBound![0].Should().BeApproximately(35f - expectedHalf, 0.001f);
        result.UpperBound![0].Should().BeApproximately(35f + expectedHalf, 0.001f);
    }

    [Fact]
    public async Task ResidualModel_NanCorrection_PropagatesNan()
    {
        var baseline = MakeModel("physics", [30f, 50f]);
        var correction = MakeModel("rf", [float.NaN, -3f]);
        var residual = new ResidualModel(baseline, correction);

        using var input = MakeFrame("speed", [800f, 1200f]);
        var result = await residual.PredictAsync(input, CancellationToken.None);

        float.IsNaN(result.Values[0]).Should().BeTrue("NaN correction must produce NaN output");
        result.Values[1].Should().BeApproximately(47f, 0.001f);
    }

    // ─── EnsembleModel ───────────────────────────────────────────────────────────

    [Fact]
    public async Task EnsembleModel_EqualWeights_AveragesOutputs()
    {
        var m1 = MakeModel("m1", [10f, 20f]);
        var m2 = MakeModel("m2", [20f, 40f]);
        var ensemble = new EnsembleModel([m1, m2]);

        using var input = MakeFrame("speed", [500f, 800f]);
        var result = await ensemble.PredictAsync(input, CancellationToken.None);

        result.Values[0].Should().BeApproximately(15f, 0.001f);
        result.Values[1].Should().BeApproximately(30f, 0.001f);
    }

    [Fact]
    public async Task EnsembleModel_ExplicitWeights_WeightedAverage()
    {
        var m1 = MakeModel("m1", [10f]);
        var m2 = MakeModel("m2", [30f]);
        // weights 1:3 → normalized 0.25, 0.75 → expected = 0.25*10 + 0.75*30 = 25
        var ensemble = new EnsembleModel([m1, m2], weights: [1f, 3f]);

        using var input = MakeFrame("speed", [800f]);
        var result = await ensemble.PredictAsync(input, CancellationToken.None);

        result.Values[0].Should().BeApproximately(25f, 0.001f);
    }

    [Fact]
    public void EnsembleModel_EmptyModelList_Throws()
    {
        var act = () => new EnsembleModel([]);
        act.Should().Throw<ArgumentException>().WithMessage("*at least one*");
    }

    [Fact]
    public async Task EnsembleModel_NanInAnyModel_PropagatesNan()
    {
        var m1 = MakeModel("m1", [float.NaN, 20f]);
        var m2 = MakeModel("m2", [10f, 30f]);
        var ensemble = new EnsembleModel([m1, m2]);

        using var input = MakeFrame("speed", [500f, 800f]);
        var result = await ensemble.PredictAsync(input, CancellationToken.None);

        float.IsNaN(result.Values[0]).Should().BeTrue("NaN in any model must propagate");
        result.Values[1].Should().BeApproximately(25f, 0.001f);
    }

    // ─── GatedModel ──────────────────────────────────────────────────────────────

    [Fact]
    public async Task GatedModel_HighGate_WeightsIfTrueHeavily()
    {
        // gate = 0.9 → output ≈ 0.9 * 100 + 0.1 * 50 = 95
        var classifier = MakeModel("clf", [0.9f]);
        var ifTrue = MakeModel("true_model", [100f]);
        var ifFalse = MakeModel("false_model", [50f]);
        var gated = new GatedModel(classifier, ifTrue, ifFalse);

        using var input = MakeFrame("speed", [800f]);
        var result = await gated.PredictAsync(input, CancellationToken.None);

        result.Values[0].Should().BeApproximately(95f, 0.001f);
    }

    [Fact]
    public async Task GatedModel_LowGate_WeightsIfFalseHeavily()
    {
        // gate = 0.1 → output = 0.1 * 100 + 0.9 * 50 = 55
        var classifier = MakeModel("clf", [0.1f]);
        var ifTrue = MakeModel("true_model", [100f]);
        var ifFalse = MakeModel("false_model", [50f]);
        var gated = new GatedModel(classifier, ifTrue, ifFalse);

        using var input = MakeFrame("speed", [400f]);
        var result = await gated.PredictAsync(input, CancellationToken.None);

        result.Values[0].Should().BeApproximately(55f, 0.001f);
    }

    [Fact]
    public async Task GatedModel_NanGate_PropagatesNan()
    {
        var classifier = MakeModel("clf", [float.NaN]);
        var ifTrue = MakeModel("true_model", [100f]);
        var ifFalse = MakeModel("false_model", [50f]);
        var gated = new GatedModel(classifier, ifTrue, ifFalse);

        using var input = MakeFrame("speed", [800f]);
        var result = await gated.PredictAsync(input, CancellationToken.None);

        float.IsNaN(result.Values[0]).Should().BeTrue("NaN gate must produce NaN output");
    }

    [Fact]
    public void GatedModel_InvalidThreshold_Throws()
    {
        var m = MakeModel("m", [1f]);
        var act = () => new GatedModel(m, m, m, threshold: 0f);
        act.Should().Throw<ArgumentOutOfRangeException>().WithMessage("*threshold*");
    }
}
