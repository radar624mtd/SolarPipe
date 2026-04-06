namespace SolarPipe.Core.Models;

public record ColumnInfo(string Name, ColumnType Type, bool IsNullable, string? MissingReason = null);

public enum ColumnType { Float, Int, String, DateTime }

public record DataSchema(IReadOnlyList<ColumnInfo> Columns)
{
    public bool HasColumn(string name) =>
        Columns.Any(c => c.Name.Equals(name, StringComparison.OrdinalIgnoreCase));

    public int IndexOf(string name)
    {
        for (int i = 0; i < Columns.Count; i++)
            if (Columns[i].Name.Equals(name, StringComparison.OrdinalIgnoreCase))
                return i;
        return -1;
    }
}
