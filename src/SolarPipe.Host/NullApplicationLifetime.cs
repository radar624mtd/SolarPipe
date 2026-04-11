using Microsoft.Extensions.Hosting;

namespace SolarPipe.Host;

// Minimal IHostApplicationLifetime for CLI commands that don't use a generic host.
// ApplicationStopping is never cancelled — the command owns shutdown via CancellationToken.
internal sealed class NullApplicationLifetime : IHostApplicationLifetime
{
    public CancellationToken ApplicationStarted  => CancellationToken.None;
    public CancellationToken ApplicationStopping => CancellationToken.None;
    public CancellationToken ApplicationStopped  => CancellationToken.None;
    public void StopApplication() { }
}
