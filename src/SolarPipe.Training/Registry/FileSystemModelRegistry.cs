using System.Security.Cryptography;
using System.Text.Json;
using Microsoft.ML;
using SolarPipe.Core.Interfaces;
using SolarPipe.Core.Models;
using SolarPipe.Training.Adapters;

namespace SolarPipe.Training.Registry;

public sealed class FileSystemModelRegistry : IModelRegistry, IDisposable
{
    private readonly string _basePath;
    private readonly Mutex _registryMutex = new(false, "Global\\SolarPipeRegistry");
    private bool _disposed;

    private static readonly JsonSerializerOptions _jsonOptions = new()
    {
        WriteIndented = true,
        PropertyNameCaseInsensitive = true,
        NumberHandling = System.Text.Json.Serialization.JsonNumberHandling.AllowNamedFloatingPointLiterals,
    };

    public FileSystemModelRegistry(string basePath)
    {
        if (string.IsNullOrWhiteSpace(basePath))
            throw new ArgumentException("Registry base path must not be empty.", nameof(basePath));
        _basePath = basePath;
        Directory.CreateDirectory(_basePath);
    }

    public async Task RegisterAsync(ModelArtifact artifact, ITrainedModel model, CancellationToken ct)
    {
        if (artifact is null) throw new ArgumentNullException(nameof(artifact));
        if (model is null) throw new ArgumentNullException(nameof(model));

        var version = ResolveVersion(artifact.ModelId, artifact.Version);
        var modelDir = Path.Combine(_basePath, artifact.ModelId, version);
        Directory.CreateDirectory(modelDir);

        var modelPath = Path.Combine(modelDir, "model.bin");
        var metaPath = Path.Combine(modelDir, "metadata.json");

        // Step 1: Save model to temp file
        var tempModelPath = Path.Combine(modelDir, $".tmp_{Guid.NewGuid():N}_model.bin");
        await model.SaveAsync(tempModelPath, ct);

        // Step 2: Compute SHA-256 fingerprint of the saved model file (RULE-041)
        string fileFingerprint;
        await using (var fs = new FileStream(tempModelPath, FileMode.Open, FileAccess.Read, FileShare.None))
        {
            fileFingerprint = Convert.ToHexString(await SHA256.HashDataAsync(fs, ct));
        }

        var finalArtifact = artifact with
        {
            Version = version,
            ArtifactPath = modelPath,
            DataFingerprint = fileFingerprint,
        };

        // Step 3: Serialize metadata to temp file
        var tempMetaPath = Path.Combine(modelDir, $".tmp_{Guid.NewGuid():N}_metadata.json");
        var json = JsonSerializer.Serialize(finalArtifact, _jsonOptions);
        await File.WriteAllTextAsync(tempMetaPath, json, ct);

        // Step 4: Atomic move both files under mutex (RULE-040)
        _registryMutex.WaitOne();
        try
        {
            File.Move(tempModelPath, modelPath, overwrite: true);
            File.Move(tempMetaPath, metaPath, overwrite: true);
        }
        finally
        {
            _registryMutex.ReleaseMutex();
        }
    }

    public async Task<ITrainedModel> LoadAsync(string modelId, string version, CancellationToken ct)
    {
        if (string.IsNullOrWhiteSpace(modelId))
            throw new ArgumentException("Model ID must not be empty.", nameof(modelId));

        var resolvedVersion = string.IsNullOrWhiteSpace(version)
            ? GetLatestVersion(modelId)
            : version;

        var metaPath = Path.Combine(_basePath, modelId, resolvedVersion, "metadata.json");
        if (!File.Exists(metaPath))
            throw new FileNotFoundException(
                $"Metadata not found for model '{modelId}' version '{resolvedVersion}'. Path: {metaPath}");

        var json = await File.ReadAllTextAsync(metaPath, ct);
        ModelArtifact? artifact;
        try
        {
            artifact = JsonSerializer.Deserialize<ModelArtifact>(json, _jsonOptions);
        }
        catch (JsonException ex)
        {
            throw new InvalidOperationException(
                $"Corrupted metadata for model '{modelId}' version '{resolvedVersion}'. " +
                $"Path: {metaPath}. Parse error: {ex.Message}", ex);
        }

        if (artifact is null)
            throw new InvalidOperationException(
                $"Metadata deserialized to null for model '{modelId}' version '{resolvedVersion}'.");

        if (!File.Exists(artifact.ArtifactPath))
            throw new FileNotFoundException(
                $"Model artifact file missing for '{modelId}' version '{resolvedVersion}'. " +
                $"Expected: {artifact.ArtifactPath}");

        return LoadModelForFramework(artifact);
    }

    public async Task<IReadOnlyList<ModelArtifact>> ListAsync(string? stageName = null, CancellationToken ct = default)
    {
        var artifacts = new List<ModelArtifact>();

        if (!Directory.Exists(_basePath))
            return artifacts;

        foreach (var modelDir in Directory.EnumerateDirectories(_basePath))
        {
            foreach (var versionDir in Directory.EnumerateDirectories(modelDir))
            {
                var metaPath = Path.Combine(versionDir, "metadata.json");
                if (!File.Exists(metaPath))
                    continue;

                var json = await File.ReadAllTextAsync(metaPath, ct);
                ModelArtifact? artifact;
                try
                {
                    artifact = JsonSerializer.Deserialize<ModelArtifact>(json, _jsonOptions);
                }
                catch (JsonException)
                {
                    continue; // Skip corrupted entries during listing
                }

                if (artifact is null) continue;
                if (stageName is null || artifact.StageName == stageName)
                    artifacts.Add(artifact);
            }
        }

        return artifacts.OrderByDescending(a => a.TrainedAt).ToList();
    }

    private string ResolveVersion(string modelId, string requestedVersion)
    {
        var modelDir = Path.Combine(_basePath, modelId);
        if (!Directory.Exists(modelDir))
            return requestedVersion;

        // Auto-increment if version already exists
        if (!Directory.Exists(Path.Combine(modelDir, requestedVersion)))
            return requestedVersion;

        // Parse semver major.minor.patch and increment patch
        if (!TryParseSemVer(requestedVersion, out var major, out var minor, out var patch))
            return requestedVersion + $"_{DateTime.UtcNow:yyyyMMddHHmmss}";

        // Find next available patch version
        int nextPatch = patch + 1;
        while (Directory.Exists(Path.Combine(modelDir, $"{major}.{minor}.{nextPatch}")))
            nextPatch++;

        return $"{major}.{minor}.{nextPatch}";
    }

    private string GetLatestVersion(string modelId)
    {
        var modelDir = Path.Combine(_basePath, modelId);
        if (!Directory.Exists(modelDir))
            throw new DirectoryNotFoundException(
                $"No versions found for model '{modelId}'. Directory: {modelDir}");

        var versions = Directory.EnumerateDirectories(modelDir)
            .Select(Path.GetFileName)
            .Where(v => v is not null && TryParseSemVer(v!, out _, out _, out _))
            .OrderByDescending(v =>
            {
                TryParseSemVer(v!, out var maj, out var min, out var pat);
                return (maj, min, pat);
            })
            .FirstOrDefault();

        if (versions is null)
            throw new InvalidOperationException(
                $"No valid versioned artifacts found for model '{modelId}'. Directory: {modelDir}");

        return versions;
    }

    private static bool TryParseSemVer(string version, out int major, out int minor, out int patch)
    {
        major = minor = patch = 0;
        var parts = version.Split('.');
        if (parts.Length != 3) return false;
        return int.TryParse(parts[0], out major)
            && int.TryParse(parts[1], out minor)
            && int.TryParse(parts[2], out patch);
    }

    private static ITrainedModel LoadModelForFramework(ModelArtifact artifact)
    {
        return artifact.Config.Framework switch
        {
            "mlnet" or "MlNet" or "ml.net" => LoadMlNetModel(artifact),
            _ => throw new NotSupportedException(
                $"Cannot load model for framework '{artifact.Config.Framework}'. " +
                $"ModelId: '{artifact.ModelId}', Version: '{artifact.Version}'. " +
                $"Supported: mlnet")
        };
    }

    private static MlNetTrainedModel LoadMlNetModel(ModelArtifact artifact)
    {
        var mlContext = new MLContext(seed: 42);
        var loadedModel = mlContext.Model.Load(artifact.ArtifactPath, out _);

        return new MlNetTrainedModel(
            artifact.Config,
            loadedModel,
            mlContext,
            artifact.Metrics);
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _registryMutex.Dispose();
    }
}
