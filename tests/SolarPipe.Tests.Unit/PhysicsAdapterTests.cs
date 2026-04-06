using FluentAssertions;
using SolarPipe.Core.Models;
using SolarPipe.Data.DataFrame;
using SolarPipe.Tests.Unit.Fixtures;
using SolarPipe.Training.Adapters;

namespace SolarPipe.Tests.Unit;

[Trait("Category", "Unit")]
public class PhysicsAdapterTests
{
    private readonly PhysicsAdapter _adapter = new();

    private static InMemoryDataFrame MakeSpeedFrame(float[] speeds)
    {
        var schema = new DataSchema([new ColumnInfo("radial_speed_km_s", ColumnType.Float, false)]);
        return new InMemoryDataFrame(schema, [speeds]);
    }

    private static StageConfig MakeDragConfig(IReadOnlyDictionary<string, object>? hp = null) =>
        new StageConfig(
            Name: "drag_stage",
            Framework: "Physics",
            ModelType: "DragBased",
            DataSource: "src",
            Features: ["radial_speed_km_s"],
            Target: "arrival_time_hours",
            Hyperparameters: hp);

    [Fact]
    public void FrameworkType_IsPhysics()
    {
        _adapter.FrameworkType.Should().Be(FrameworkType.Physics);
    }

    [Fact]
    public void SupportedModels_ContainsDragBased()
    {
        _adapter.SupportedModels.Should().Contain("DragBased");
    }

    [Fact]
    public async Task TrainAsync_DragBased_ReturnsDragBasedModel()
    {
        using var frame = MakeSpeedFrame([800f, 1200f]);
        var model = await _adapter.TrainAsync(MakeDragConfig(), frame, null, CancellationToken.None);

        model.Should().NotBeNull();
        model.StageName.Should().Be("drag_stage");
    }

    [Fact]
    public async Task TrainAsync_UnsupportedModelType_Throws()
    {
        var config = new StageConfig("s", "Physics", "NeuralOde", "src", [], "t");
        using var frame = MakeSpeedFrame([800f]);

        var act = async () => await _adapter.TrainAsync(config, frame, null, CancellationToken.None);
        await act.Should().ThrowAsync<NotSupportedException>().WithMessage("*NeuralOde*");
    }

    [Fact]
    public async Task TrainThenPredict_EndToEnd_ProducesFiniteResult()
    {
        using var frame = MakeSpeedFrame([
            PhysicsTestFixtures.ModerateEvent.RadialSpeed.KmPerSec,
            PhysicsTestFixtures.HalloweenStorm2003.RadialSpeed.KmPerSec
        ]);

        var model = await _adapter.TrainAsync(
            MakeDragConfig(new Dictionary<string, object>
            {
                ["gamma_km_inv"] = PhysicsTestFixtures.DragParameters.GammaTypicalKmInv,
                ["ambient_wind_km_s"] = (double)PhysicsTestFixtures.ModerateEvent.AmbientSolarWindKmPerSec,
                ["start_distance_solar_radii"] = (double)PhysicsTestFixtures.DragParameters.StartDistanceSolarRadii,
                ["target_distance_solar_radii"] = (double)PhysicsTestFixtures.DragParameters.EarthDistanceSolarRadii
            }),
            frame, null, CancellationToken.None);

        var result = await model.PredictAsync(frame, CancellationToken.None);

        result.Values.Should().HaveCount(2);
        result.Values.Should().OnlyContain(v => !float.IsNaN(v) && v > 0f,
            "All predictions must be finite positive arrival times");

        // Halloween storm faster → arrives before moderate event
        result.Values[1].Should().BeLessThan(result.Values[0],
            "Halloween storm (2459 km/s) must arrive before moderate CME (800 km/s)");
    }
}
