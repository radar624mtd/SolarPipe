using ParquetSharp;
using SolarPipe.Core.Interfaces;
using SolarPipe.Core.Models;

namespace SolarPipe.Data.Providers;

public sealed class ParquetProvider : IDataSourceProvider
{
    public string ProviderName => "parquet";

    public bool CanHandle(DataSourceConfig config) =>
        config.Provider.Equals("parquet", StringComparison.OrdinalIgnoreCase)
        && !string.IsNullOrWhiteSpace(config.ConnectionString);

    public Task<DataSchema> DiscoverSchemaAsync(DataSourceConfig config, CancellationToken ct)
    {
        ct.ThrowIfCancellationRequested();

        if (!File.Exists(config.ConnectionString))
            throw new FileNotFoundException(
                $"ParquetProvider: file not found: '{config.ConnectionString}'");

        using var fileReader = new ParquetFileReader(config.ConnectionString);
        var schema = BuildSchema(fileReader);
        return Task.FromResult(schema);
    }

    public Task<IDataFrame> LoadAsync(DataSourceConfig config, DataQuery query, CancellationToken ct)
    {
        ct.ThrowIfCancellationRequested();

        if (!File.Exists(config.ConnectionString))
            throw new FileNotFoundException(
                $"ParquetProvider: file not found: '{config.ConnectionString}'");

        using var fileReader = new ParquetFileReader(config.ConnectionString);
        var schema = BuildSchema(fileReader);

        int numCols = schema.Columns.Count;
        var columnBuffers = new List<float>[numCols];
        for (int i = 0; i < numCols; i++)
            columnBuffers[i] = new List<float>();

        int numRowGroups = fileReader.FileMetaData.NumRowGroups;
        int rowsLoaded = 0;

        for (int rg = 0; rg < numRowGroups; rg++)
        {
            ct.ThrowIfCancellationRequested();

            if (query.Limit.HasValue && rowsLoaded >= query.Limit.Value)
                break;

            using var rowGroupReader = fileReader.RowGroup(rg);
            int rowsInGroup = (int)rowGroupReader.MetaData.NumRows;

            int rowsToTake = query.Limit.HasValue
                ? Math.Min(rowsInGroup, query.Limit.Value - rowsLoaded)
                : rowsInGroup;

            for (int col = 0; col < numCols; col++)
            {
                using var columnReader = rowGroupReader.Column(col);
                var values = ReadColumnAsFloat(columnReader, rowsInGroup, schema.Columns[col].Type);

                for (int r = 0; r < rowsToTake; r++)
                    columnBuffers[col].Add(values[r]);
            }

            rowsLoaded += rowsToTake;
        }

        var data = columnBuffers.Select(b => b.ToArray()).ToArray();
        return Task.FromResult<IDataFrame>(new DataFrame.InMemoryDataFrame(schema, data));
    }

    // --- helpers ---

    private static DataSchema BuildSchema(ParquetFileReader fileReader)
    {
        var descriptor = fileReader.FileMetaData.Schema;
        var cols = new List<ColumnInfo>();

        // ColumnDescriptor is not IDisposable — no using needed
        for (int i = 0; i < descriptor.NumColumns; i++)
        {
            var colDesc = descriptor.Column(i);
            var name = colDesc.Name;
            var colType = MapPhysicalType(colDesc.PhysicalType);
            cols.Add(new ColumnInfo(name, colType, IsNullable: true));
        }

        return new DataSchema(cols);
    }

    private static ColumnType MapPhysicalType(PhysicalType pt) =>
        pt switch
        {
            PhysicalType.Float  => ColumnType.Float,
            PhysicalType.Double => ColumnType.Float,   // downcast to float
            PhysicalType.Int32  => ColumnType.Float,
            PhysicalType.Int64  => ColumnType.Float,
            PhysicalType.Int96  => ColumnType.DateTime, // Spark timestamp (Int96 nanoseconds)
            PhysicalType.ByteArray => ColumnType.String,
            PhysicalType.FixedLenByteArray => ColumnType.String,
            PhysicalType.Boolean => ColumnType.Float,
            _ => ColumnType.Float
        };

    private static float[] ReadColumnAsFloat(ColumnReader columnReader, int rowCount, ColumnType colType)
    {
        var result = new float[rowCount];

        switch (columnReader)
        {
            case ColumnReader<float> floatReader:
                ReadFloatColumn(floatReader, result, rowCount);
                break;

            case ColumnReader<double> doubleReader:
                ReadDoubleColumn(doubleReader, result, rowCount);
                break;

            case ColumnReader<int> intReader:
                ReadIntColumn(intReader, result, rowCount);
                break;

            case ColumnReader<long> longReader:
                ReadLongColumn(longReader, result, rowCount);
                break;

            case ColumnReader<bool> boolReader:
                ReadBoolColumn(boolReader, result, rowCount);
                break;

            default:
                // String / binary: fill NaN
                for (int i = 0; i < result.Length; i++)
                    result[i] = float.NaN;
                break;
        }

        return result;
    }

    private static void ReadFloatColumn(ColumnReader<float> reader, float[] result, int rowCount)
    {
        var defLevels = new short[rowCount];
        var repLevels = new short[rowCount];
        long rowsRead = reader.ReadBatch(rowCount, defLevels, repLevels, result, out long valuesRead);
        ApplyNullAndSentinel(result, defLevels, (int)rowsRead, (int)valuesRead);
    }

    private static void ReadDoubleColumn(ColumnReader<double> reader, float[] result, int rowCount)
    {
        var defLevels = new short[rowCount];
        var repLevels = new short[rowCount];
        var doubles = new double[rowCount];
        long rowsRead = reader.ReadBatch(rowCount, defLevels, repLevels, doubles, out long valuesRead);

        for (int i = 0; i < (int)valuesRead; i++)
            result[i] = (float)doubles[i];

        ApplyNullAndSentinel(result, defLevels, (int)rowsRead, (int)valuesRead);
    }

    private static void ReadIntColumn(ColumnReader<int> reader, float[] result, int rowCount)
    {
        var defLevels = new short[rowCount];
        var repLevels = new short[rowCount];
        var ints = new int[rowCount];
        long rowsRead = reader.ReadBatch(rowCount, defLevels, repLevels, ints, out long valuesRead);

        for (int i = 0; i < (int)valuesRead; i++)
            result[i] = (float)ints[i];

        ApplyNullAndSentinel(result, defLevels, (int)rowsRead, (int)valuesRead);
    }

    private static void ReadLongColumn(ColumnReader<long> reader, float[] result, int rowCount)
    {
        var defLevels = new short[rowCount];
        var repLevels = new short[rowCount];
        var longs = new long[rowCount];
        long rowsRead = reader.ReadBatch(rowCount, defLevels, repLevels, longs, out long valuesRead);

        for (int i = 0; i < (int)valuesRead; i++)
            result[i] = (float)longs[i];

        ApplyNullAndSentinel(result, defLevels, (int)rowsRead, (int)valuesRead);
    }

    private static void ReadBoolColumn(ColumnReader<bool> reader, float[] result, int rowCount)
    {
        var defLevels = new short[rowCount];
        var repLevels = new short[rowCount];
        var bools = new bool[rowCount];
        long rowsRead = reader.ReadBatch(rowCount, defLevels, repLevels, bools, out long valuesRead);

        for (int i = 0; i < (int)valuesRead; i++)
            result[i] = bools[i] ? 1f : 0f;

        ApplyNullAndSentinel(result, defLevels, (int)rowsRead, (int)valuesRead);
    }

    // Definition level 0 = null → NaN. Applies sentinel conversion (RULE-120).
    // For required columns (all def levels == 1), valueCount == rowCount and there are no gaps.
    private static void ApplyNullAndSentinel(float[] result, short[] defLevels, int rowCount, int valueCount)
    {
        if (valueCount == rowCount)
        {
            // No nulls — just apply sentinel conversion in-place
            for (int i = 0; i < rowCount; i++)
                result[i] = IsSentinel(result[i]) ? float.NaN : result[i];
            return;
        }

        // Nulls present: rebuild array right-to-left so values shift correctly
        // defLevels[row] == 0 means null
        int vi = valueCount - 1;
        for (int row = rowCount - 1; row >= 0; row--)
        {
            if (defLevels[row] == 0)
            {
                result[row] = float.NaN;
            }
            else
            {
                float v = vi >= 0 ? result[vi--] : float.NaN;
                result[row] = IsSentinel(v) ? float.NaN : v;
            }
        }
    }

    private static bool IsSentinel(float v) =>
        v is 9999.9f or 999.9f or 999f or -1e31f;
}
