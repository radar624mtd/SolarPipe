using SolarPipe.Core.Interfaces;
using SolarPipe.Core.Models;

namespace SolarPipe.Data.DataFrame;

public sealed partial class InMemoryDataFrame
{
    public IDataFrame Slice(int startRow, int count)
    {
        if (startRow < 0 || startRow > RowCount)
            throw new ArgumentOutOfRangeException(nameof(startRow),
                $"startRow {startRow} out of range [0, {RowCount}]. Schema: [{string.Join(", ", _schema.Columns.Select(c => c.Name))}].");
        if (count < 0 || startRow + count > RowCount)
            throw new ArgumentOutOfRangeException(nameof(count),
                $"Slice [{startRow}, {startRow + count}) exceeds RowCount={RowCount}.");

        var sliced = new float[_data.Length][];
        for (int i = 0; i < _data.Length; i++)
        {
            sliced[i] = new float[count];
            Array.Copy(_data[i], startRow, sliced[i], 0, count);
        }
        return new InMemoryDataFrame(_schema, sliced);
    }

    public IDataFrame SelectColumns(params string[] columns)
    {
        if (columns is null || columns.Length == 0)
            throw new ArgumentException("Must select at least one column.", nameof(columns));

        var newCols = new List<ColumnInfo>(columns.Length);
        var newData = new float[columns.Length][];

        for (int i = 0; i < columns.Length; i++)
        {
            int srcIdx = _schema.IndexOf(columns[i]);
            if (srcIdx < 0)
                throw new KeyNotFoundException($"Column '{columns[i]}' not found in schema.");
            newCols.Add(_schema.Columns[srcIdx]);
            newData[i] = _data[srcIdx];
        }

        return new InMemoryDataFrame(new DataSchema(newCols), newData);
    }

    public IDataFrame AddColumn(string name, float[] values)
    {
        if (string.IsNullOrWhiteSpace(name))
            throw new ArgumentException("Column name must not be empty.", nameof(name));
        if (values is null)
            throw new ArgumentNullException(nameof(values));
        if (RowCount > 0 && values.Length != RowCount)
            throw new ArgumentException(
                $"AddColumn: values.Length={values.Length} does not match RowCount={RowCount}. Column='{name}'.");
        if (_schema.HasColumn(name))
            throw new InvalidOperationException($"Column '{name}' already exists in the schema.");

        var newCols = _schema.Columns.Append(new ColumnInfo(name, ColumnType.Float, true)).ToList();
        var newData = _data.Append(values).ToArray();
        return new InMemoryDataFrame(new DataSchema(newCols), newData);
    }

    public IDataFrame ResampleAndAlign(TimeSpan cadence)
    {
        // Requires a timestamp column. Locate it by checking for DateTimeKind columns or
        // a column named "timestamp" (case-insensitive).
        // Phase 2 full implementation. Phase 1: throw NotSupportedException with clear context.
        throw new NotSupportedException(
            $"ResampleAndAlign is not implemented in Phase 1. " +
            $"Schema columns: [{string.Join(", ", _schema.Columns.Select(c => c.Name))}]. " +
            $"Requested cadence: {cadence}. Implement in Phase 2 (Task IDataFrame.ResampleAndAlign).");
    }
}
