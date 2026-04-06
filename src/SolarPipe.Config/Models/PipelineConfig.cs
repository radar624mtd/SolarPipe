using SolarPipe.Core.Models;

namespace SolarPipe.Config.Models;

public class PipelineConfig
{
    public string Name { get; set; } = string.Empty;
    public Dictionary<string, DataSourceYaml> DataSources { get; set; } = new();
    public Dictionary<string, StageYaml> Stages { get; set; } = new();
    public string? Compose { get; set; }
    public EvaluationYaml? Evaluation { get; set; }
    public OutputYaml? Output { get; set; }

    public DataSourceConfig ToDataSourceConfig(string key)
    {
        var ds = DataSources[key];
        return new DataSourceConfig(key, ds.Provider, ds.ConnectionString, ds.Options);
    }

    public StageConfig ToStageConfig(string key)
    {
        var s = Stages[key];
        return new StageConfig(key, s.Framework, s.ModelType, s.DataSource,
            s.Features, s.Target, s.Hyperparameters, s.MockDataStrategy);
    }
}

public class DataSourceYaml
{
    public string Provider { get; set; } = string.Empty;
    public string ConnectionString { get; set; } = string.Empty;
    public Dictionary<string, string>? Options { get; set; }
}

public class StageYaml
{
    public string Framework { get; set; } = string.Empty;
    public string ModelType { get; set; } = string.Empty;
    public string DataSource { get; set; } = string.Empty;
    public List<string> Features { get; set; } = new();
    public string Target { get; set; } = string.Empty;
    public Dictionary<string, object>? Hyperparameters { get; set; }
    public string? MockDataStrategy { get; set; }
}

public class EvaluationYaml
{
    public string Strategy { get; set; } = string.Empty;
    public int GapDays { get; set; }
}

public class OutputYaml
{
    public string Format { get; set; } = string.Empty;
    public string Path { get; set; } = string.Empty;
}
