using System.ComponentModel;
using System.Diagnostics;
using System.Runtime.InteropServices;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;

namespace SolarPipe.Training.Adapters;

// SidecarLifecycleService: IHostedService that spawns and monitors the Python gRPC sidecar.
//
// RULE-060: Lifecycle managed here — NOT via bare Process.Start in command code.
//           On Windows: Job Object ensures child dies when .NET host exits (even on crash).
//           Heartbeat is also sent as --parent-pid to the Python server.
// RULE-062: Python must be invoked via workspace-relative venv path, never bare "python".
//
// Registration (Program.cs):
//   services.AddHostedService<SidecarLifecycleService>();
//   services.AddSingleton(SidecarOptions.FromEnvironment());
public sealed class SidecarLifecycleService : IHostedService, IAsyncDisposable
{
    private readonly SidecarOptions _options;
    private readonly IHostApplicationLifetime _lifetime;
    private readonly ILogger<SidecarLifecycleService> _logger;

    private Process? _process;
    private JobObjectHandle? _jobObject;
    private TaskCompletionSource<bool>? _readyTcs;
    private CancellationTokenSource? _exitCts;
    private bool _disposed;

    public SidecarLifecycleService(
        SidecarOptions options,
        IHostApplicationLifetime lifetime,
        ILogger<SidecarLifecycleService> logger)
    {
        _options = options;
        _lifetime = lifetime;
        _logger = logger;
    }

    public bool IsRunning => _process is { HasExited: false };

    public async Task StartAsync(CancellationToken cancellationToken)
    {
        if (!_options.Enabled)
        {
            _logger.LogInformation("Sidecar disabled — skipping launch");
            return;
        }

        if (!File.Exists(_options.PythonExecutable))
            throw new FileNotFoundException(
                $"Python executable not found: {_options.PythonExecutable}. " +
                $"Set SOLARPIPE_ROOT and create venv at ${{SOLARPIPE_ROOT}}/python/.venv (RULE-062).",
                _options.PythonExecutable);

        if (!File.Exists(_options.ServerScript))
            throw new FileNotFoundException(
                $"Sidecar script not found: {_options.ServerScript}.",
                _options.ServerScript);

        _exitCts = new CancellationTokenSource();
        _readyTcs = new TaskCompletionSource<bool>(TaskCreationOptions.RunContinuationsAsynchronously);

        await Task.Run(() => Launch(), cancellationToken);

        // Register graceful teardown on host shutdown
        _lifetime.ApplicationStopping.Register(() =>
        {
            _logger.LogInformation("ApplicationStopping — stopping sidecar");
            StopProcess(grace: TimeSpan.FromSeconds(5));
        });

        // Wait for server to be ready (port open) or timeout
        using var timeoutCts = new CancellationTokenSource(_options.StartupTimeout);
        using var linked = CancellationTokenSource.CreateLinkedTokenSource(
            cancellationToken, timeoutCts.Token);

        try
        {
            await _readyTcs.Task.WaitAsync(linked.Token);
            _logger.LogInformation("Sidecar ready on port {Port}", _options.Port);
        }
        catch (OperationCanceledException) when (timeoutCts.IsCancellationRequested)
        {
            throw new TimeoutException(
                $"Sidecar did not become ready within {_options.StartupTimeout.TotalSeconds}s " +
                $"(stage=SidecarLifecycleService, port={_options.Port}).");
        }
    }

    public Task StopAsync(CancellationToken cancellationToken)
    {
        StopProcess(grace: TimeSpan.FromSeconds(5));
        return Task.CompletedTask;
    }

    public async ValueTask DisposeAsync()
    {
        if (_disposed) return;
        _disposed = true;
        StopProcess(grace: TimeSpan.FromSeconds(2));
        _exitCts?.Dispose();
        if (OperatingSystem.IsWindows()) _jobObject?.Dispose();
        if (_process != null)
        {
            await Task.Run(() => _process.WaitForExit(2000));
            _process.Dispose();
        }
    }

    // ─── Process launch ───────────────────────────────────────────────────────

    private void Launch()
    {
        var psi = new ProcessStartInfo
        {
            FileName = _options.PythonExecutable,
            Arguments = BuildArguments(),
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            CreateNoWindow = true,
            WorkingDirectory = _options.WorkingDirectory,
        };

        // Pass SOLARPIPE_ROOT so the server can locate config files
        psi.Environment["SOLARPIPE_ROOT"] = _options.WorkingDirectory;

        _process = new Process { StartInfo = psi, EnableRaisingEvents = true };
        _process.OutputDataReceived += OnOutput;
        _process.ErrorDataReceived += OnError;
        _process.Exited += OnProcessExited;

        _process.Start();
        _process.BeginOutputReadLine();
        _process.BeginErrorReadLine();

        // RULE-060: On Windows, assign to a Job Object so the child is killed if the parent crashes
        if (OperatingSystem.IsWindows())
        {
            try
            {
                _jobObject = JobObjectHandle.CreateAndAssign(_process);
            }
            catch (Exception ex)
            {
                _logger.LogWarning("Job Object creation failed — orphan protection unavailable: {Msg}", ex.Message);
            }
        }

        _logger.LogInformation(
            "Sidecar launched: PID={Pid} port={Port} script={Script}",
            _process.Id, _options.Port, _options.ServerScript);

        // Poll for readiness: try TCP connect to the gRPC port
        _ = Task.Run(() => WaitForPortAsync(_exitCts!.Token));
    }

    private string BuildArguments()
    {
        var script = _options.ServerScript.Replace("\\", "/");
        return $"\"{script}\" " +
               $"--port {_options.Port} " +
               $"--parent-pid {Environment.ProcessId} " +
               $"--model-dir \"{_options.ModelDir}\" " +
               $"--log-dir \"{_options.LogDir}\"";
    }

    private async Task WaitForPortAsync(CancellationToken ct)
    {
        using var httpClient = new System.Net.Http.HttpClient
        {
            Timeout = TimeSpan.FromSeconds(2),
        };

        while (!ct.IsCancellationRequested)
        {
            try
            {
                // gRPC server accepts HTTP/2 — check TCP via a simple socket probe
                using var tcp = new System.Net.Sockets.TcpClient();
                await tcp.ConnectAsync("127.0.0.1", _options.Port, ct);
                _readyTcs?.TrySetResult(true);
                return;
            }
            catch
            {
                await Task.Delay(250, ct).ConfigureAwait(false);
            }
        }

        _readyTcs?.TrySetCanceled(ct);
    }

    // ─── Process output / lifecycle ───────────────────────────────────────────

    private void OnOutput(object sender, DataReceivedEventArgs e)
    {
        if (e.Data != null)
            _logger.LogDebug("[sidecar stdout] {Line}", e.Data);
    }

    private void OnError(object sender, DataReceivedEventArgs e)
    {
        if (e.Data != null)
            _logger.LogWarning("[sidecar stderr] {Line}", e.Data);
    }

    private void OnProcessExited(object? sender, EventArgs e)
    {
        int code = _process?.ExitCode ?? -1;
        string codeDesc = TranslateExitCode(code);

        _logger.LogError(
            "Sidecar exited unexpectedly: exit_code={Code} ({Desc}) (RULE-141)",
            code, codeDesc);

        _readyTcs?.TrySetException(new InvalidOperationException(
            $"Sidecar process exited before becoming ready: exit_code={code} ({codeDesc})."));
    }

    private void StopProcess(TimeSpan grace)
    {
        _exitCts?.Cancel();
        var proc = _process;
        if (proc == null || proc.HasExited) return;

        try
        {
            proc.Kill(entireProcessTree: true);
            proc.WaitForExit((int)grace.TotalMilliseconds);
        }
        catch (Exception ex)
        {
            _logger.LogWarning("Error stopping sidecar: {Msg}", ex.Message);
        }
    }

    // ─── Exit code translation (RULE-141) ─────────────────────────────────────

    public static string TranslateExitCode(int code) => code switch
    {
        0   => "clean exit",
        1   => "Python unhandled exception",
        137 => "Out of Memory (SIGKILL / OOM killer)",
        139 => "Segmentation fault — check Arrow/PyTorch native handles",
        -1  => "unknown (process not started or exit code unavailable)",
        _   => $"exit code {code}",
    };
}
