using System.Buffers;
using Microsoft.ML;
using SolarPipe.Core.Interfaces;
using SolarPipe.Core.Models;

namespace SolarPipe.Data.DataFrame;

public sealed partial class InMemoryDataFrame : IDataFrame
{
    private readonly float[][] _data;          // column-major: _data[col][row]
    private readonly DataSchema _schema;
    private readonly Dictionary<string, string[]>? _stringColumns;
    private bool _disposed;

    public DataSchema Schema => _schema;
    public int RowCount { get; }

    public InMemoryDataFrame(DataSchema schema, float[][] data,
        Dictionary<string, string[]>? stringColumns = null)
    {
        if (schema is null) throw new ArgumentNullException(nameof(schema));
        if (data is null) throw new ArgumentNullException(nameof(data));
        if (data.Length != schema.Columns.Count)
            throw new ArgumentException(
                $"Data has {data.Length} columns but schema has {schema.Columns.Count}.",
                nameof(data));

        _schema = schema;
        _data = data;
        _stringColumns = stringColumns;
        RowCount = data.Length == 0 ? 0 : data[0].Length;

        ValidateColumnLengths();
    }

    public string[]? GetStringColumn(string name) =>
        _stringColumns is not null && _stringColumns.TryGetValue(name, out var arr) ? arr : null;

    public float[] GetColumn(string name)
    {
        int idx = _schema.IndexOf(name);
        if (idx < 0)
            throw new KeyNotFoundException($"Column '{name}' not found in schema.");
        return _data[idx];
    }

    public float[] GetColumn(int index)
    {
        if ((uint)index >= (uint)_data.Length)
            throw new ArgumentOutOfRangeException(nameof(index), $"Column index {index} out of range [0, {_data.Length}).");
        return _data[index];
    }

    internal float[] GetColumnSpan(string name) => GetColumn(name);

    private void ValidateColumnLengths()
    {
        for (int i = 0; i < _data.Length; i++)
        {
            if (_data[i].Length != RowCount)
                throw new InvalidOperationException(
                    $"Column '{_schema.Columns[i].Name}' has {_data[i].Length} rows but expected {RowCount}. " +
                    $"Stage: InMemoryDataFrame construction, schema: [{string.Join(", ", _schema.Columns.Select(c => c.Name))}].");
        }
    }

    public static InMemoryDataFrame FromColumns(DataSchema schema, IEnumerable<float[]> columns) =>
        new(schema, columns.ToArray());

    public static InMemoryDataFrame Empty(DataSchema schema) =>
        new(schema, schema.Columns.Select(_ => Array.Empty<float>()).ToArray());

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        // Return rented arrays to pool if they were allocated from ArrayPool.
        // Convention: arrays exactly matching 2^n sizes from ArrayPool.Shared.Rent are returned.
        // For simplicity in MVP, data arrays are standard allocations (not rented).
        // Phase 3 will add explicit ArrayPool tracking via a flag field.
    }
}
