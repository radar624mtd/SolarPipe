using SolarPipe.Core.Models;

namespace SolarPipe.Config.Models;

public class SweepConfig
{
    public SweepMeta Sweep { get; set; } = new();
    public Dictionary<string, SweepStageYaml> Stages { get; set; } = new();

    // Build a StageConfig from the sweep's shared stage definition
    public StageConfig ToStageConfig(string key)
    {
        var s = Stages[key];
        return new StageConfig(key, s.Framework, s.ModelType, string.Empty,
            s.Features, s.Target, s.Hyperparameters);
    }
}

public class SweepMeta
{
    public string Name { get; set; } = string.Empty;
    public bool Parallel { get; set; } = true;
    public string LogTagPrefix { get; set; } = "sweep";
    public SweepCvConfig Cv { get; set; } = new();
    public List<HypothesisConfig> Hypotheses { get; set; } = new();
}

public class SweepCvConfig
{
    public string Strategy { get; set; } = "expanding_window";
    public int Folds { get; set; } = 5;
    public int GapBufferDays { get; set; } = 5;
    public int MinTestEvents { get; set; } = 50;
    public string CalibrationFold { get; set; } = "last";
}

public class HypothesisConfig
{
    public string Id { get; set; } = string.Empty;
    public string Compose { get; set; } = string.Empty;
    public List<string> Stages { get; set; } = new();
}

public class SweepStageYaml
{
    public string Framework { get; set; } = string.Empty;
    public string ModelType { get; set; } = string.Empty;
    public List<string> Features { get; set; } = new();
    public string Target { get; set; } = string.Empty;
    public Dictionary<string, object>? Hyperparameters { get; set; }
}
