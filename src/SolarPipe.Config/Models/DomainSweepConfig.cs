namespace SolarPipe.Config.Models;

// YAML model for phase8_domain_sweep.yaml.
//
// Top-level structure:
//   domain_pipeline:   orchestration config (CV settings, domain assignments, meta-learner)
//   stages:            stage definitions (same SweepStageYaml as Phase 7)
//
// Three-domain pipeline:
//   Origination → transit_target (latent arrival speed via DragBasedV2 pseudo-labels)
//   Transit     → transit_time_hours (DragBased physics + RF residual)
//   Impact      → dst_min_nt (BurtonOde physics + RF residual)
//   Meta-learner → arrival_time_hours + storm_intensity_nt + storm_duration_hours
public sealed class DomainSweepConfig
{
    public DomainPipelineMeta DomainPipeline { get; set; } = new();
    public Dictionary<string, SweepStageYaml> Stages { get; set; } = new();
}

public sealed class DomainPipelineMeta
{
    public string Name { get; set; } = string.Empty;
    public DomainCvConfig Cv { get; set; } = new();
    public DomainDataSourceConfig DataSource { get; set; } = new();
    public DomainGroupConfig Origination { get; set; } = new();
    public DomainGroupConfig Transit { get; set; } = new();
    public DomainGroupConfig Impact { get; set; } = new();
    public MetaLearnerConfig MetaLearner { get; set; } = new();
    public PhysicsBaselineConfig? PhysicsBaseline { get; set; }
}

public sealed class DomainCvConfig
{
    public int Folds { get; set; } = 5;
    public int GapBufferDays { get; set; } = 5;
    public int MinTestEvents { get; set; } = 30;
    // When set, bypasses CV and does a single train/test split at this UTC date (ISO-8601).
    public string? HeldOutAfter { get; set; }
}

public sealed class DomainDataSourceConfig
{
    public string Provider { get; set; } = "sqlite";
    public string ConnectionString { get; set; } = string.Empty;
    public string Table { get; set; } = string.Empty;
    public string Filter { get; set; } = string.Empty;
}

public sealed class DomainGroupConfig
{
    public string Target { get; set; } = string.Empty;
    public string? Stage { get; set; }           // single-stage domain (origination)
    public string? PhysicsStage { get; set; }    // physics baseline stage name
    public string? ResidualStage { get; set; }   // RF residual stage name
}

public sealed class MetaLearnerConfig
{
    // stage name per output — arrival_time_hours, storm_intensity_nt, storm_duration_hours
    public Dictionary<string, string> Stages { get; set; } = new();
}

public sealed class PhysicsBaselineConfig
{
    public string Compose { get; set; } = string.Empty;
    public List<string> Stages { get; set; } = new();
}
