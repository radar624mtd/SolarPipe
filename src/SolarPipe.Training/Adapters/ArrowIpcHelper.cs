using Apache.Arrow;
using Apache.Arrow.Ipc;
using SolarPipe.Core.Interfaces;
using SolarPipe.Core.Models;
using SolarPipe.Data.DataFrame;

namespace SolarPipe.Training.Adapters;

// Shared Arrow IPC write/read helpers used by GrpcSidecarAdapter and tests.
// RULE-063: All numeric columns are written as float32 (never float64).
// RULE-125: Data travels as file path, not inline bytes.
public static class ArrowIpcHelper
{
    public static async Task WriteAsync(
        IDataFrame frame,
        string outputPath,
        CancellationToken ct)
    {
        var schema = BuildSchema(frame);
        var arrays = BuildArrays(frame);
        var batch = new RecordBatch(schema, arrays, frame.RowCount);

        await using var stream = File.Create(outputPath);
        using var writer = new ArrowFileWriter(stream, schema);
        await writer.WriteRecordBatchAsync(batch, ct);
        await writer.WriteEndAsync(ct);
    }

    // ReadAsync: deserializes an Arrow IPC file into an InMemoryDataFrame.
    // RULE-063: Validates that all columns are float32; null Arrow values → float.NaN (RULE-120).
    public static async Task<IDataFrame> ReadAsync(string path, CancellationToken ct)
    {
        if (!File.Exists(path))
            throw new FileNotFoundException(
                $"ArrowIpcHelper.ReadAsync: file not found at '{path}'.");

        await using var stream = File.OpenRead(path);
        using var reader = new ArrowFileReader(stream);

        var cols = new List<ColumnInfo>();
        var bufs = new List<List<float>>();

        RecordBatch? batch;
        bool schemaRead = false;

        while ((batch = await reader.ReadNextRecordBatchAsync(ct)) is not null)
        {
            using (batch)
            {
                if (!schemaRead)
                {
                    for (int c = 0; c < batch.ColumnCount; c++)
                    {
                        var field = batch.Schema.GetFieldByIndex(c);
                        // RULE-063: enforce float32 on read side
                        if (field.DataType.TypeId != Apache.Arrow.Types.ArrowTypeId.Float)
                            throw new InvalidOperationException(
                                $"ArrowIpcHelper.ReadAsync: column '{field.Name}' has type " +
                                $"'{field.DataType}', expected float32 (RULE-063). " +
                                "Python sidecar must enforce pa.float32() schema.");
                        cols.Add(new ColumnInfo(field.Name, ColumnType.Float, IsNullable: true));
                        bufs.Add(new List<float>());
                    }
                    schemaRead = true;
                }

                for (int c = 0; c < batch.ColumnCount; c++)
                {
                    var arr = (FloatArray)batch.Column(c);
                    for (int r = 0; r < arr.Length; r++)
                        bufs[c].Add(arr.IsNull(r) ? float.NaN : arr.GetValue(r) ?? float.NaN);
                }
            }
        }

        if (!schemaRead)
        {
            // Empty file — return empty frame with no columns
            return new InMemoryDataFrame(new DataSchema([]), []);
        }

        var dfSchema = new DataSchema(cols);
        float[][] arrays = bufs.Select(b => b.ToArray()).ToArray();
        return new InMemoryDataFrame(dfSchema, arrays);
    }

    private static Apache.Arrow.Schema BuildSchema(IDataFrame frame)
    {
        var builder = new Apache.Arrow.Schema.Builder();
        foreach (var col in frame.Schema.Columns)
        {
            // RULE-063: All columns are float32. Never emit float64 across the gRPC boundary.
            builder.Field(new Field(col.Name, Apache.Arrow.Types.FloatType.Default, nullable: true));
        }
        return builder.Build();
    }

    private static IReadOnlyList<IArrowArray> BuildArrays(IDataFrame frame)
    {
        var result = new List<IArrowArray>(frame.Schema.Columns.Count);
        for (int c = 0; c < frame.Schema.Columns.Count; c++)
        {
            float[] data = frame.GetColumn(c);
            var builder = new FloatArray.Builder();
            foreach (float v in data)
                builder.Append(float.IsNaN(v) ? null : (float?)v);
            result.Add(builder.Build());
        }
        return result;
    }
}
