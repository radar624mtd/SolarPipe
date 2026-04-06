using System.Text.Json;
using SolarPipe.Core.Interfaces;

namespace SolarPipe.Host.Commands;

public sealed class InspectCommand : ICommand
{
    private readonly IModelRegistry _modelRegistry;

    public InspectCommand(IModelRegistry modelRegistry)
    {
        _modelRegistry = modelRegistry;
    }

    public async Task<int> ExecuteAsync(string[] args, CancellationToken ct)
    {
        var stageFilter = ArgParser.Get(args, "--stage");

        try
        {
            var artifacts = await _modelRegistry.ListAsync(stageFilter, ct);

            if (artifacts.Count == 0)
            {
                Console.WriteLine(stageFilter is null
                    ? "No models registered in the registry."
                    : $"No models registered for stage '{stageFilter}'.");
                return ExitCodes.Success;
            }

            var json = JsonSerializer.Serialize(artifacts, new JsonSerializerOptions { WriteIndented = true });
            Console.WriteLine(json);
            return ExitCodes.Success;
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine(
                $"INSPECT_ERROR type={ex.GetType().Name} message=\"{ex.Message}\"");
            return ExitCodes.InspectFailed;
        }
    }
}
