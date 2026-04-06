namespace SolarPipe.Core.Domain;

public static class SpaceWeatherTimeParser
{
    // OMNI format: "2003 301 0000" (year, doy, hhmm)
    private static readonly string[] OmniFormats = ["yyyy DDD HHmm", "yyyy DDD HH mm"];

    // CDAW catalog format: "2003/10/28 11:30" or "2003-10-28T11:30:00"
    private static readonly string[] CdawFormats =
    [
        "yyyy/MM/dd HH:mm", "yyyy/MM/dd HH:mm:ss",
        "yyyy-MM-ddTHH:mm:ss", "yyyy-MM-ddTHH:mm:ssZ",
        "yyyy-MM-dd HH:mm:ss", "yyyy-MM-dd HH:mm"
    ];

    public static DateTime ParseOmni(string value)
    {
        if (string.IsNullOrWhiteSpace(value))
            throw new FormatException("OMNI timestamp is null or empty.");

        var parts = value.Trim().Split(' ', StringSplitOptions.RemoveEmptyEntries);
        if (parts.Length < 3)
            throw new FormatException($"OMNI timestamp '{value}' does not have expected 3 parts (year doy hhmm).");

        if (!int.TryParse(parts[0], out int year) ||
            !int.TryParse(parts[1], out int doy) ||
            !int.TryParse(parts[2], out int hhmm))
            throw new FormatException($"Cannot parse OMNI timestamp '{value}'.");

        int hour = hhmm / 100;
        int min = hhmm % 100;
        return new DateTime(year, 1, 1, hour, min, 0, DateTimeKind.Utc).AddDays(doy - 1);
    }

    public static DateTime ParseCdaw(string value)
    {
        if (string.IsNullOrWhiteSpace(value))
            throw new FormatException("CDAW timestamp is null or empty.");

        if (DateTime.TryParseExact(value.Trim(), CdawFormats,
            System.Globalization.CultureInfo.InvariantCulture,
            System.Globalization.DateTimeStyles.AssumeUniversal | System.Globalization.DateTimeStyles.AdjustToUniversal,
            out DateTime result))
            return result;

        throw new FormatException($"Cannot parse CDAW timestamp '{value}'. Expected formats: yyyy/MM/dd HH:mm or ISO 8601.");
    }

    public static DateTime ParseIso8601(string value)
    {
        if (DateTime.TryParse(value, System.Globalization.CultureInfo.InvariantCulture,
            System.Globalization.DateTimeStyles.RoundtripKind, out DateTime result))
            return result;

        throw new FormatException($"Cannot parse ISO 8601 timestamp '{value}'.");
    }
}
