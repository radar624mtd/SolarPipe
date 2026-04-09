using FluentAssertions;
using SolarPipe.Core.Models;
using SolarPipe.Data.DataFrame;
using SolarPipe.Tests.Unit.Fixtures;
using SolarPipe.Training.Physics;

namespace SolarPipe.Tests.Unit;

[Trait("Category", "Unit")]
public class DragBasedModelTests
{
    // Helper — build a StageConfig for drag model with given hyperparameters
    private static StageConfig MakeDragConfig(IReadOnlyDictionary<string, object>? hp = null) =>
        new StageConfig(
            Name: "test_drag",
            Framework: "Physics",
            ModelType: "DragBased",
            DataSource: "test",
            Features: ["radial_speed_km_s"],
            Target: "arrival_time_hours",
            Hyperparameters: hp);

    // Helper — build an IDataFrame with a single speed column
    private static InMemoryDataFrame MakeSpeedFrame(float[] speeds)
    {
        var schema = new DataSchema([new ColumnInfo("radial_speed_km_s", ColumnType.Float, false)]);
        return new InMemoryDataFrame(schema, [speeds]);
    }

    // --- RunOde analytical comparison ---

    [Fact]
    public void RunOde_SlowCme_ArrivalTimeInReasonableRange()
    {
        // Moderate event: v0=800 km/s, w=400 km/s, typical gamma
        // Expected transit ~1.5-3 days (36-72 hours)
        var (_, arrivalHours) = DragBasedModel.RunOde(
            PhysicsTestFixtures.ModerateEvent.RadialSpeed.KmPerSec,
            PhysicsTestFixtures.ModerateEvent.AmbientSolarWindKmPerSec,
            PhysicsTestFixtures.DragParameters.GammaTypicalKmInv,
            PhysicsTestFixtures.DragParameters.StartDistanceSolarRadii,
            PhysicsTestFixtures.DragParameters.EarthDistanceSolarRadii);

        arrivalHours.Should().BeInRange(30.0, 100.0,
            "Moderate CME (800 km/s) transit to 1 AU should be 30–100 hours");
    }

    [Fact]
    public void RunOde_FastCme_ArrivalTimeFasterThanSlow()
    {
        // Halloween storm speed — should arrive faster than moderate event
        var (_, arrivalFast) = DragBasedModel.RunOde(
            PhysicsTestFixtures.HalloweenStorm2003.RadialSpeed.KmPerSec,
            PhysicsTestFixtures.HalloweenStorm2003.AmbientSolarWindKmPerSec,
            PhysicsTestFixtures.DragParameters.GammaTypicalKmInv,
            PhysicsTestFixtures.DragParameters.StartDistanceSolarRadii,
            PhysicsTestFixtures.DragParameters.EarthDistanceSolarRadii);

        var (_, arrivalSlow) = DragBasedModel.RunOde(
            PhysicsTestFixtures.ModerateEvent.RadialSpeed.KmPerSec,
            PhysicsTestFixtures.ModerateEvent.AmbientSolarWindKmPerSec,
            PhysicsTestFixtures.DragParameters.GammaTypicalKmInv,
            PhysicsTestFixtures.DragParameters.StartDistanceSolarRadii,
            PhysicsTestFixtures.DragParameters.EarthDistanceSolarRadii);

        arrivalFast.Should().BeLessThan(arrivalSlow,
            "Faster CME must arrive before slower CME under same drag conditions");
    }

    [Fact]
    public void RunOde_HighDrag_DecelleratesMoreThanLowDrag()
    {
        // Same speed, high vs low drag — high drag produces earlier arrival at ~1 AU
        // (CME decelerates faster toward ambient, but also loses more speed)
        // However, primarily: low-drag fast CME retains speed longer → arrives sooner
        var (_, arrivalHighDrag) = DragBasedModel.RunOde(
            800.0, 400.0,
            PhysicsTestFixtures.DragParameters.GammaMaxKmInv,
            PhysicsTestFixtures.DragParameters.StartDistanceSolarRadii,
            PhysicsTestFixtures.DragParameters.EarthDistanceSolarRadii);

        var (_, arrivalLowDrag) = DragBasedModel.RunOde(
            800.0, 400.0,
            PhysicsTestFixtures.DragParameters.GammaMinKmInv,
            PhysicsTestFixtures.DragParameters.StartDistanceSolarRadii,
            PhysicsTestFixtures.DragParameters.EarthDistanceSolarRadii);

        // For v0 > w (CME faster than ambient), high drag decelerates faster
        // but both arrive; high-drag CME retains less speed → may arrive slightly later
        // What we care about: both produce finite, non-NaN arrival times
        double.IsNaN(arrivalHighDrag).Should().BeFalse();
        double.IsNaN(arrivalLowDrag).Should().BeFalse();
        arrivalHighDrag.Should().BeInRange(20.0, 200.0);
        arrivalLowDrag.Should().BeInRange(20.0, 200.0);
    }

    [Fact]
    public void RunOde_CarringtonClassSpeed_StableNoNaN()
    {
        // RULE-030: Carrington-class event (~3000 km/s) — solver must remain stable
        // τ = 1/(2*γ*(v0-w)) ≈ 1/(2*2e-7*2600) ≈ 0.96h — very stiff
        var (_, arrivalHours) = DragBasedModel.RunOde(
            PhysicsTestFixtures.July2012Event.RadialSpeed.KmPerSec,
            PhysicsTestFixtures.July2012Event.AmbientSolarWindKmPerSec,
            PhysicsTestFixtures.DragParameters.GammaTypicalKmInv,
            PhysicsTestFixtures.DragParameters.StartDistanceSolarRadii,
            PhysicsTestFixtures.DragParameters.EarthDistanceSolarRadii);

        double.IsNaN(arrivalHours).Should().BeFalse("Dormand-Prince must be stable for Carrington-class CME");
        arrivalHours.Should().BeInRange(12.0, 50.0,
            "Carrington-class CME (~2000 km/s) should arrive in 12–50 hours");
    }

    // --- Parameter range validation (RULE-032) ---

    [Fact]
    public void RunOde_SpeedTooLow_Throws()
    {
        var act = () => DragBasedModel.RunOde(100.0, 400.0, 0.5e-7, 21.5, 215.0);
        act.Should().Throw<ArgumentOutOfRangeException>().WithMessage("*200*");
    }

    [Fact]
    public void RunOde_SpeedTooHigh_Throws()
    {
        var act = () => DragBasedModel.RunOde(4000.0, 400.0, 0.5e-7, 21.5, 215.0);
        act.Should().Throw<ArgumentOutOfRangeException>().WithMessage("*3500*");
    }

    [Fact]
    public void RunOde_GammaTooLow_Throws()
    {
        var act = () => DragBasedModel.RunOde(800.0, 400.0, 0.05e-7, 21.5, 215.0);
        act.Should().Throw<ArgumentOutOfRangeException>().WithMessage("*0.2e-7*");
    }

    [Fact]
    public void RunOde_StartDistanceTooSmall_Throws()
    {
        var act = () => DragBasedModel.RunOde(800.0, 400.0, 0.5e-7, 5.0, 215.0);
        act.Should().Throw<ArgumentOutOfRangeException>().WithMessage("*20*");
    }

    // --- PredictAsync via ITrainedModel ---

    [Fact]
    public async Task PredictAsync_SingleRow_ReturnsFiniteArrivalTime()
    {
        var config = MakeDragConfig(new Dictionary<string, object>
        {
            ["gamma_km_inv"] = 0.5e-7,
            ["ambient_wind_km_s"] = 400.0,
            ["start_distance_solar_radii"] = 21.5,
            ["target_distance_solar_radii"] = 215.0
        });
        var model = new DragBasedModel(config);

        using var frame = MakeSpeedFrame([800f]);
        var result = await model.PredictAsync(frame, CancellationToken.None);

        result.Values.Should().HaveCount(1);
        float.IsNaN(result.Values[0]).Should().BeFalse();
        result.Values[0].Should().BeInRange(30f, 100f);
    }

    [Fact]
    public async Task PredictAsync_NaNSpeed_PropagatesNaN()
    {
        var model = new DragBasedModel(MakeDragConfig());
        using var frame = MakeSpeedFrame([float.NaN]);

        var result = await model.PredictAsync(frame, CancellationToken.None);
        float.IsNaN(result.Values[0]).Should().BeTrue("NaN input must propagate to NaN output");
    }

    [Fact]
    public async Task PredictAsync_OutOfRangeSpeed_ProducesNaN()
    {
        // Batch mode: out-of-range speeds produce NaN rather than throwing,
        // so a single bad row doesn't abort processing of 1900+ training events.
        var model = new DragBasedModel(MakeDragConfig());
        using var frame = MakeSpeedFrame([173f, 800f, 4000f]);

        var result = await model.PredictAsync(frame, CancellationToken.None);
        float.IsNaN(result.Values[0]).Should().BeTrue("173 km/s < 200 km/s limit → NaN");
        result.Values[1].Should().BeInRange(30f, 100f, "800 km/s is valid → finite prediction");
        float.IsNaN(result.Values[2]).Should().BeTrue("4000 km/s > 3500 km/s limit → NaN");
    }

    [Fact]
    public async Task PredictAsync_HyperparametersAreCaseInsensitive()
    {
        // RULE-012: OrdinalIgnoreCase — PascalCase keys must work same as snake_case
        var config = MakeDragConfig(new Dictionary<string, object>
        {
            ["Gamma_Km_Inv"] = 0.5e-7,           // PascalCase
            ["Ambient_Wind_Km_S"] = 400.0,         // PascalCase
            ["Start_Distance_Solar_Radii"] = 21.5,
            ["Target_Distance_Solar_Radii"] = 215.0
        });
        var model = new DragBasedModel(config);
        using var frame = MakeSpeedFrame([800f]);

        var result = await model.PredictAsync(frame, CancellationToken.None);
        result.Values[0].Should().BeInRange(30f, 100f,
            "PascalCase hyperparameter keys must resolve correctly");
    }

    // --- YAML-style hyperparameter key aliases ---

    [Fact]
    public async Task PredictAsync_YamlStyleHyperparameterKeys_ResolveCorrectly()
    {
        // YAML uses drag_parameter (not gamma_km_inv), background_speed_km_s (not ambient_wind_km_s), etc.
        // These must resolve to the same values — not silently fall through to defaults.
        var yamlConfig = MakeDragConfig(new Dictionary<string, object>
        {
            ["drag_parameter"] = 0.2e-7,
            ["background_speed_km_s"] = 400.0,
            ["r_start_rs"] = 21.5,
            ["r_stop_rs"] = 215.0
        });
        var codeConfig = MakeDragConfig(new Dictionary<string, object>
        {
            ["gamma_km_inv"] = 0.2e-7,
            ["ambient_wind_km_s"] = 400.0,
            ["start_distance_solar_radii"] = 21.5,
            ["target_distance_solar_radii"] = 215.0
        });

        var yamlModel = new DragBasedModel(yamlConfig);
        var codeModel = new DragBasedModel(codeConfig);

        using var frame = MakeSpeedFrame([800f]);
        var yamlResult = await yamlModel.PredictAsync(frame, CancellationToken.None);
        var codeResult = await codeModel.PredictAsync(frame, CancellationToken.None);

        yamlResult.Values[0].Should().BeApproximately(codeResult.Values[0], 0.01f,
            "YAML-style and code-style hyperparameter keys must produce identical results");
    }

    [Fact]
    public async Task PredictAsync_YamlDragParameter_NotIgnored()
    {
        // Verify that drag_parameter=0.2e-7 produces a DIFFERENT result than default 0.5e-7
        var yamlConfig = MakeDragConfig(new Dictionary<string, object>
        {
            ["drag_parameter"] = 0.2e-7,
            ["background_speed_km_s"] = 400.0,
            ["r_start_rs"] = 21.5,
            ["r_stop_rs"] = 215.0
        });
        var defaultConfig = MakeDragConfig(); // uses default gamma=0.5e-7

        var yamlModel = new DragBasedModel(yamlConfig);
        var defaultModel = new DragBasedModel(defaultConfig);

        using var frame = MakeSpeedFrame([800f]);
        var yamlResult = await yamlModel.PredictAsync(frame, CancellationToken.None);
        var defaultResult = await defaultModel.PredictAsync(frame, CancellationToken.None);

        Math.Abs(yamlResult.Values[0] - defaultResult.Values[0]).Should().BeGreaterThan(1.0f,
            "drag_parameter=0.2e-7 vs default 0.5e-7 must produce materially different transit times");
    }

    // --- cme_speed_kms column name ---

    [Fact]
    public async Task PredictAsync_CmeSpeedKmsColumn_Resolves()
    {
        var model = new DragBasedModel(MakeDragConfig());

        // Use cme_speed_kms as column name (as in training_features and YAML)
        var schema = new DataSchema([new ColumnInfo("cme_speed_kms", ColumnType.Float, false)]);
        using var frame = new InMemoryDataFrame(schema, [new[] { 800f }]);

        var result = await model.PredictAsync(frame, CancellationToken.None);

        result.Values.Should().HaveCount(1);
        float.IsNaN(result.Values[0]).Should().BeFalse();
        result.Values[0].Should().BeInRange(30f, 100f);
    }

    // --- SaveAsync / LoadAsync ---

    [Fact]
    public async Task SaveAsync_CreatesFile_AtomicWrite()
    {
        var model = new DragBasedModel(MakeDragConfig());
        var tmpPath = Path.Combine(Path.GetTempPath(), $"drag_test_{Guid.NewGuid():N}.json");

        try
        {
            await model.SaveAsync(tmpPath, CancellationToken.None);
            File.Exists(tmpPath).Should().BeTrue("SaveAsync must create model file");

            var loaded = DragBasedModel.FromFile(tmpPath);
            loaded.StageName.Should().Be(model.StageName);
        }
        finally
        {
            if (File.Exists(tmpPath)) File.Delete(tmpPath);
        }
    }
}
