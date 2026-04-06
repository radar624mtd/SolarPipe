using System.Diagnostics;
using Microsoft.ML;
using Microsoft.ML.Data;
using SolarPipe.Core.Models;

namespace SolarPipe.Data.DataFrame;

public sealed partial class InMemoryDataFrame
{
    public IDataView ToDataView(MLContext mlContext)
    {
        if (mlContext is null) throw new ArgumentNullException(nameof(mlContext));

        // RULE-002: Validate all column lengths match RowCount before IDataView creation.
        Debug.Assert(_data.All(col => col.Length == RowCount),
            $"ToDataView: column length mismatch. RowCount={RowCount}.");
        ValidateColumnLengths();

        return new DataFrameDataView(_schema, _data, RowCount);
    }

    public float[][] ToArray()
    {
        var result = new float[_data.Length][];
        for (int i = 0; i < _data.Length; i++)
        {
            result[i] = new float[RowCount];
            Array.Copy(_data[i], result[i], RowCount);
        }
        return result;
    }
}

internal sealed class DataFrameDataView : IDataView
{
    private readonly DataSchema _schema;
    private readonly float[][] _data;
    private readonly int _rowCount;
    private readonly DataViewSchema _dvSchema;

    internal DataFrameDataView(DataSchema schema, float[][] data, int rowCount)
    {
        _schema = schema;
        _data = data;
        _rowCount = rowCount;
        _dvSchema = BuildDvSchema(schema);
    }

    public bool CanShuffle => false;
    public DataViewSchema Schema => _dvSchema;

    public long? GetRowCount() => _rowCount;

    public DataViewRowCursor GetRowCursor(IEnumerable<DataViewSchema.Column> columnsNeeded, Random? rand = null)
        => new DataFrameCursor(_dvSchema, _data, _rowCount);

    public DataViewRowCursor[] GetRowCursorSet(IEnumerable<DataViewSchema.Column> columnsNeeded, int n, Random? rand = null)
        => [GetRowCursor(columnsNeeded, rand)];

    private static DataViewSchema BuildDvSchema(DataSchema schema)
    {
        var builder = new DataViewSchema.Builder();
        foreach (var col in schema.Columns)
        {
            var dvType = col.Type switch
            {
                ColumnType.Float => NumberDataViewType.Single,
                ColumnType.Int => NumberDataViewType.Int32,
                _ => (DataViewType)TextDataViewType.Instance
            };
            builder.AddColumn(col.Name, dvType);
        }
        return builder.ToSchema();
    }
}

internal sealed class DataFrameCursor : DataViewRowCursor
{
    private readonly DataViewSchema _schema;
    private readonly float[][] _data;
    private readonly int _rowCount;
    private long _position = -1;

    internal DataFrameCursor(DataViewSchema schema, float[][] data, int rowCount)
    {
        _schema = schema;
        _data = data;
        _rowCount = rowCount;
    }

    public override DataViewSchema Schema => _schema;
    public override long Position => _position;
    public override long Batch => 0;
    public override bool IsColumnActive(DataViewSchema.Column column) => true;

    public override bool MoveNext()
    {
        if (_position + 1 >= _rowCount) return false;
        _position++;
        return true;
    }

    public override ValueGetter<TValue> GetGetter<TValue>(DataViewSchema.Column column)
    {
        int colIdx = column.Index;
        float[] colData = _data[colIdx];

        if (typeof(TValue) == typeof(float))
        {
            return (ValueGetter<TValue>)(Delegate)(ValueGetter<float>)((ref float value) =>
            {
                value = colData[_position];
            });
        }
        throw new InvalidOperationException(
            $"GetGetter<{typeof(TValue).Name}> not supported for column '{column.Name}' (index {colIdx}). " +
            $"DataFrameDataView only supports float columns.");
    }

    public override ValueGetter<DataViewRowId> GetIdGetter() =>
        (ref DataViewRowId value) => { value = new DataViewRowId((ulong)_position, 0); };

    protected override void Dispose(bool disposing) { }
}
