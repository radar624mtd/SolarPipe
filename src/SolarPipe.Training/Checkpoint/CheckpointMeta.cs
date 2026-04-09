namespace SolarPipe.Training.Checkpoint;

public sealed class CheckpointMeta
{
    public string StageName { get; set; } = string.Empty;
    public string PipelineName { get; set; } = string.Empty;
    public string ConfigFingerprint { get; set; } = string.Empty;
    public string InputFingerprint { get; set; } = string.Empty;
    public int RowCount { get; set; }
    public List<string> ColumnNames { get; set; } = [];
    public DateTime WrittenAt { get; set; }
}
