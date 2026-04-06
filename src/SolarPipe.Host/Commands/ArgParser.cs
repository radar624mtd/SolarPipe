namespace SolarPipe.Host.Commands;

public static class ArgParser
{
    public static string? Get(string[] args, string flag)
    {
        for (var i = 0; i < args.Length - 1; i++)
        {
            if (string.Equals(args[i], flag, StringComparison.OrdinalIgnoreCase))
                return args[i + 1];
        }
        return null;
    }

    public static string Require(string[] args, string flag)
    {
        return Get(args, flag)
            ?? throw new ArgumentException(
                $"Required argument '{flag}' is missing. " +
                $"Usage includes: {flag} <value>");
    }
}
