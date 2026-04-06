namespace SolarPipe.Training.Adapters;

// Configuration for SidecarLifecycleService.
// Read from environment variables so the CLI host can configure via env.
public sealed class SidecarOptions
{
    public bool Enabled { get; init; } = true;
    public int Port { get; init; } = 50051;
    public string PythonExecutable { get; init; } = "";
    public string ServerScript { get; init; } = "";
    public string WorkingDirectory { get; init; } = Directory.GetCurrentDirectory();
    public string ModelDir { get; init; } = "./models";
    public string LogDir { get; init; } = "logs";
    public TimeSpan StartupTimeout { get; init; } = TimeSpan.FromSeconds(30);

    public static SidecarOptions FromEnvironment()
    {
        string root = Environment.GetEnvironmentVariable("SOLARPIPE_ROOT")
                   ?? Directory.GetCurrentDirectory();

        string defaultPython = OperatingSystem.IsWindows()
            ? Path.Combine(root, "python", ".venv", "Scripts", "python.exe")
            : Path.Combine(root, "python", ".venv", "bin", "python");

        string defaultScript = Path.Combine(root, "python", "solarpipe_server.py");

        return new SidecarOptions
        {
            Enabled  = !string.Equals(
                Environment.GetEnvironmentVariable("SOLARPIPE_SIDECAR_DISABLED"),
                "true", StringComparison.OrdinalIgnoreCase),
            Port     = int.TryParse(
                Environment.GetEnvironmentVariable("SOLARPIPE_SIDECAR_PORT"), out int p) ? p : 50051,
            PythonExecutable = Environment.GetEnvironmentVariable("SOLARPIPE_PYTHON") ?? defaultPython,
            ServerScript     = Environment.GetEnvironmentVariable("SOLARPIPE_SERVER_SCRIPT") ?? defaultScript,
            WorkingDirectory = root,
            ModelDir         = Environment.GetEnvironmentVariable("SOLARPIPE_MODEL_DIR")
                            ?? Path.Combine(root, "models"),
            LogDir           = Path.Combine(root, "logs"),
            StartupTimeout   = TimeSpan.FromSeconds(
                double.TryParse(
                    Environment.GetEnvironmentVariable("SOLARPIPE_SIDECAR_STARTUP_TIMEOUT_S"),
                    out double s) ? s : 30),
        };
    }
}
