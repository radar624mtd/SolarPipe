using FluentAssertions;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging.Abstractions;
using NSubstitute;
using SolarPipe.Training.Adapters;

namespace SolarPipe.Tests.Integration;

// Phase 4 Task 11.2 — SidecarLifecycleService integration tests.
//
// Tests do NOT require a running Python server. They validate:
//   1. SidecarOptions.FromEnvironment() honours env-var overrides
//   2. Disabled sidecar skips launch without errors
//   3. Missing Python executable raises FileNotFoundException on StartAsync
//   4. Exit code translation follows RULE-141 (137=OOM, 139=SIGSEGV, etc.)
[Trait("Category", "Integration")]
public sealed class SidecarLifecycleTests
{
    // ─── 1. SidecarOptions env-var overrides ─────────────────────────────────

    [Fact]
    public void SidecarOptions_FromEnvironment_ReadsPortOverride()
    {
        Environment.SetEnvironmentVariable("SOLARPIPE_SIDECAR_PORT", "50099");
        try
        {
            var opts = SidecarOptions.FromEnvironment();
            opts.Port.Should().Be(50099, "SOLARPIPE_SIDECAR_PORT env var must override default port");
        }
        finally
        {
            Environment.SetEnvironmentVariable("SOLARPIPE_SIDECAR_PORT", null);
        }
    }

    [Fact]
    public void SidecarOptions_FromEnvironment_DefaultPort_Is50051()
    {
        Environment.SetEnvironmentVariable("SOLARPIPE_SIDECAR_PORT", null);

        var opts = SidecarOptions.FromEnvironment();

        opts.Port.Should().Be(50051, "default gRPC port is 50051");
    }

    [Fact]
    public void SidecarOptions_FromEnvironment_DisabledFlag_SetsEnabledFalse()
    {
        Environment.SetEnvironmentVariable("SOLARPIPE_SIDECAR_DISABLED", "true");
        try
        {
            var opts = SidecarOptions.FromEnvironment();
            opts.Enabled.Should().BeFalse("SOLARPIPE_SIDECAR_DISABLED=true must disable the sidecar");
        }
        finally
        {
            Environment.SetEnvironmentVariable("SOLARPIPE_SIDECAR_DISABLED", null);
        }
    }

    // ─── 2. Disabled sidecar is a no-op ──────────────────────────────────────

    [Fact]
    public async Task StartAsync_WhenDisabled_DoesNotThrow()
    {
        var opts = new SidecarOptions { Enabled = false };
        var lifetime = Substitute.For<IHostApplicationLifetime>();
        await using var svc = new SidecarLifecycleService(
            opts, lifetime, NullLogger<SidecarLifecycleService>.Instance);

        Func<Task> act = () => svc.StartAsync(CancellationToken.None);

        await act.Should().NotThrowAsync("disabled sidecar must skip launch silently");
        svc.IsRunning.Should().BeFalse("no process is started when disabled");
    }

    // ─── 3. Missing executable raises FileNotFoundException ──────────────────

    [Fact]
    public async Task StartAsync_MissingPythonExecutable_ThrowsFileNotFoundException()
    {
        var opts = new SidecarOptions
        {
            Enabled          = true,
            PythonExecutable = "/nonexistent/python",
            ServerScript     = "/nonexistent/server.py",
        };
        var lifetime = Substitute.For<IHostApplicationLifetime>();
        await using var svc = new SidecarLifecycleService(
            opts, lifetime, NullLogger<SidecarLifecycleService>.Instance);

        Func<Task> act = () => svc.StartAsync(CancellationToken.None);

        await act.Should().ThrowAsync<FileNotFoundException>()
            .WithMessage("*Python executable not found*",
                "must report the missing executable path so the operator knows how to fix it");
    }

    // ─── 4. Exit code translation (RULE-141) ─────────────────────────────────

    [Theory]
    [InlineData(0,   "clean exit")]
    [InlineData(1,   "Python unhandled exception")]
    [InlineData(137, "Out of Memory")]
    [InlineData(139, "Segmentation fault")]
    [InlineData(-1,  "unknown")]
    public void TranslateExitCode_ReturnsHumanReadableDescription(int code, string expected)
    {
        string desc = SidecarLifecycleService.TranslateExitCode(code);

        desc.Should().ContainEquivalentOf(expected,
            $"exit code {code} must map to a human-readable description (RULE-141)");
    }
}
