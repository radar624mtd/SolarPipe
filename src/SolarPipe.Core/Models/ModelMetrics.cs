namespace SolarPipe.Core.Models;

public record ModelMetrics(
    double Rmse,
    double Mae,
    double R2,
    IReadOnlyDictionary<string, double>? AdditionalMetrics = null);
