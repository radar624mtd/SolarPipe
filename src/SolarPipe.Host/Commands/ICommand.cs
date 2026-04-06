namespace SolarPipe.Host.Commands;

public interface ICommand
{
    Task<int> ExecuteAsync(string[] args, CancellationToken ct);
}
