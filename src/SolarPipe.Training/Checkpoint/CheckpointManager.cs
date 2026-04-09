using System.Security.Cryptography;
using System.Text.Json;
using SolarPipe.Core.Interfaces;
using SolarPipe.Core.Models;
using SolarPipe.Training.Adapters;

namespace SolarPipe.Training.Checkpoint;

// CheckpointManager — per-stage Arrow IPC checkpointing for crash recovery (ADR-015).
//
// Layout: cache/{pipelineName}/{stageName}.checkpoint   (Arrow IPC data)
//         cache/{pipelineName}/{stageName}.meta.json    (fingerprints + schema)
//
// Staleness: checkpoint is invalid if either configFingerprint or inputFingerprint
//            differs from what was stored. Invalid checkpoints are deleted on read.
//
// Thread safety: single-threaded callers (TrainCommand iterates stages sequentially).
//                No mutex needed beyond the File.Move atomic write guarantee.
public sealed class CheckpointManager
{
    private readonly string _cacheRoot;

    private static readonly JsonSerializerOptions _jsonOpts = new()
    {
        WriteIndented = true,
        PropertyNameCaseInsensitive = true,
    };

    public CheckpointManager(string cacheRoot)
    {
        _cacheRoot = cacheRoot;
    }

    public async Task WriteAsync(
        string pipelineName,
        string stageName,
        IDataFrame output,
        StageConfig config,
        string inputFingerprint,
        CancellationToken ct)
    {
        var dir = GetPipelineDir(pipelineName);
        Directory.CreateDirectory(dir);

        var checkpointPath = GetCheckpointPath(pipelineName, stageName);
        var metaPath = GetMetaPath(pipelineName, stageName);
        var tmpCheckpoint = checkpointPath + $".tmp_{Guid.NewGuid():N}";
        var tmpMeta = metaPath + $".tmp_{Guid.NewGuid():N}";

        string configFingerprint = ComputeConfigFingerprint(config);

        var meta = new CheckpointMeta
        {
            StageName = stageName,
            PipelineName = pipelineName,
            ConfigFingerprint = configFingerprint,
            InputFingerprint = inputFingerprint,
            RowCount = output.RowCount,
            ColumnNames = output.Schema.Columns.Select(c => c.Name).ToList(),
            WrittenAt = DateTime.UtcNow,
        };

        // Write Arrow IPC to temp, then meta to temp, then atomic rename both
        await ArrowIpcHelper.WriteAsync(output, tmpCheckpoint, ct);

        await File.WriteAllTextAsync(tmpMeta, JsonSerializer.Serialize(meta, _jsonOpts), ct);

        File.Move(tmpCheckpoint, checkpointPath, overwrite: true);
        File.Move(tmpMeta, metaPath, overwrite: true);
    }

    public async Task<IDataFrame?> TryReadAsync(
        string pipelineName,
        string stageName,
        StageConfig config,
        string inputFingerprint,
        CancellationToken ct)
    {
        var checkpointPath = GetCheckpointPath(pipelineName, stageName);
        var metaPath = GetMetaPath(pipelineName, stageName);

        if (!File.Exists(metaPath) || !File.Exists(checkpointPath))
            return null;

        CheckpointMeta? meta;
        try
        {
            var json = await File.ReadAllTextAsync(metaPath, ct);
            meta = JsonSerializer.Deserialize<CheckpointMeta>(json, _jsonOpts);
        }
        catch (JsonException)
        {
            // Corrupt meta — invalidate
            DeleteCheckpoint(pipelineName, stageName);
            return null;
        }

        if (meta is null)
        {
            DeleteCheckpoint(pipelineName, stageName);
            return null;
        }

        string configFingerprint = ComputeConfigFingerprint(config);

        if (meta.ConfigFingerprint != configFingerprint || meta.InputFingerprint != inputFingerprint)
        {
            // Stale — config or data changed; discard
            DeleteCheckpoint(pipelineName, stageName);
            return null;
        }

        try
        {
            return await ArrowIpcHelper.ReadAsync(checkpointPath, ct);
        }
        catch (Exception)
        {
            // Corrupt data file
            DeleteCheckpoint(pipelineName, stageName);
            return null;
        }
    }

    public Task ClearAsync(string pipelineName, CancellationToken ct)
    {
        var dir = GetPipelineDir(pipelineName);
        if (!Directory.Exists(dir))
            return Task.CompletedTask;

        foreach (var file in Directory.EnumerateFiles(dir, "*.checkpoint")
            .Concat(Directory.EnumerateFiles(dir, "*.meta.json")))
        {
            try { File.Delete(file); }
            catch (IOException) { /* already gone */ }
        }

        return Task.CompletedTask;
    }

    public string GetCheckpointPath(string pipelineName, string stageName) =>
        Path.Combine(GetPipelineDir(pipelineName), $"{stageName}.checkpoint");

    public string GetMetaPath(string pipelineName, string stageName) =>
        Path.Combine(GetPipelineDir(pipelineName), $"{stageName}.meta.json");

    private string GetPipelineDir(string pipelineName) =>
        Path.Combine(_cacheRoot, pipelineName);

    private void DeleteCheckpoint(string pipelineName, string stageName)
    {
        TryDelete(GetCheckpointPath(pipelineName, stageName));
        TryDelete(GetMetaPath(pipelineName, stageName));
    }

    private static void TryDelete(string path)
    {
        try { File.Delete(path); }
        catch (IOException) { /* ignore */ }
    }

    private static string ComputeConfigFingerprint(StageConfig config)
    {
        var json = JsonSerializer.Serialize(config, _jsonOpts);
        var bytes = System.Text.Encoding.UTF8.GetBytes(json);
        return Convert.ToHexString(SHA256.HashData(bytes));
    }
}
