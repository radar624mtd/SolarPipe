namespace SolarPipe.Host;

public static class ExitCodes
{
    public const int Success = 0;
    public const int ValidationFailed = 1;
    public const int TrainingFailed = 2;
    public const int PredictionFailed = 3;
    public const int InspectFailed = 4;
    public const int UnknownCommand = 5;
    public const int MissingArguments = 6;
    public const int CheckpointCorrupt = 7;
}
