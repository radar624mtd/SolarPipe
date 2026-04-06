using Apache.Arrow;
using Apache.Arrow.Ipc;
using SolarPipe.Core.Interfaces;

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
