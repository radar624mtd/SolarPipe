using FluentAssertions;
using SolarPipe.Config;

namespace SolarPipe.Tests.Unit.Phase7;

[Trait("Category", "Unit")]
public sealed class SweepConfigLoaderTests
{
    private readonly SweepConfigLoader _sut = new();

    // ── Valid config parsing ──────────────────────────────────────────────────

    [Fact]
    public void LoadFromString_ValidYaml_ParsesName()
    {
        var yaml = ValidSweepYaml();
        var config = _sut.LoadFromString(yaml);
        config.Sweep.Name.Should().Be("test_sweep");
    }

    [Fact]
    public void LoadFromString_ValidYaml_ParsesHypotheses()
    {
        var config = _sut.LoadFromString(ValidSweepYaml());
        config.Sweep.Hypotheses.Should().HaveCount(2);
        config.Sweep.Hypotheses[0].Id.Should().Be("H1");
        config.Sweep.Hypotheses[1].Id.Should().Be("H2");
    }

    [Fact]
    public void LoadFromString_ValidYaml_ParsesStages()
    {
        var config = _sut.LoadFromString(ValidSweepYaml());
        config.Stages.Should().ContainKey("stage_a");
        config.Stages.Should().ContainKey("stage_b");
    }

    [Fact]
    public void LoadFromString_ValidYaml_ParsesCvSettings()
    {
        var config = _sut.LoadFromString(ValidSweepYaml());
        config.Sweep.Cv.Folds.Should().Be(5);
        config.Sweep.Cv.GapBufferDays.Should().Be(5);
        config.Sweep.Cv.CalibrationFold.Should().Be("last");
    }

    // ── Invalid hypothesis references ─────────────────────────────────────────

    [Fact]
    public void LoadFromString_HypothesisReferencesUnknownStage_Throws()
    {
        // Only the hypothesis stages list references "ghost_stage"; it's not defined in stages
        var yaml = @"
sweep:
  name: test_bad_ref
  cv:
    folds: 5
    gap_buffer_days: 5
    min_test_events: 50
    calibration_fold: last
  hypotheses:
    - id: H1
      compose: ghost_stage
      stages: [ghost_stage]
stages:
  stage_a:
    framework: Physics
    model_type: DragBased
    features: [f1]
    target: t1
";
        var act = () => _sut.LoadFromString(yaml);
        act.Should().Throw<InvalidOperationException>()
            .WithMessage("*ghost_stage*");
    }

    // ── Missing required fields ───────────────────────────────────────────────

    [Fact]
    public void LoadFromString_MissingSweepName_Throws()
    {
        // Name is empty string after deserialization
        var yaml = @"
sweep:
  name: """"
  cv:
    folds: 3
    gap_buffer_days: 5
    min_test_events: 50
    calibration_fold: last
  hypotheses:
    - id: H1
      compose: stage_a
      stages: [stage_a]
stages:
  stage_a:
    framework: Physics
    model_type: DragBased
    features: [f1]
    target: t1
";
        var act = () => _sut.LoadFromString(yaml);
        act.Should().Throw<InvalidOperationException>()
            .WithMessage("*name is required*");
    }

    [Fact]
    public void LoadFromString_FoldsBelowTwo_Throws()
    {
        var yaml = @"
sweep:
  name: test
  cv:
    folds: 1
    gap_buffer_days: 5
    min_test_events: 50
    calibration_fold: last
  hypotheses:
    - id: H1
      compose: s1
      stages: [s1]
stages:
  s1:
    framework: Physics
    model_type: DragBased
    features: [f1]
    target: t1
";
        var act = () => _sut.LoadFromString(yaml);
        act.Should().Throw<InvalidOperationException>()
            .WithMessage("*folds*");
    }

    // ── Helper ────────────────────────────────────────────────────────────────

    private static string ValidSweepYaml() => @"
sweep:
  name: test_sweep
  parallel: true
  log_tag_prefix: sweep
  cv:
    strategy: expanding_window
    folds: 5
    gap_buffer_days: 5
    min_test_events: 50
    calibration_fold: last
  hypotheses:
    - id: H1
      compose: ""stage_a ^ stage_b""
      stages: [stage_a, stage_b]
    - id: H2
      compose: ""stage_b""
      stages: [stage_b]
stages:
  stage_a:
    framework: Physics
    model_type: DragBased
    features: [cme_speed_kms, sw_speed_ambient_kms]
    target: transit_hours_observed
    hyperparameters:
      drag_parameter: 0.2e-7
  stage_b:
    framework: MlNet
    model_type: FastForest
    features: [cme_speed_kms, bz_gsm_proxy_nt]
    target: transit_hours_observed
    hyperparameters:
      number_of_trees: 100
      feature_fraction: 0.7
";
}
