using Microsoft.ML;
using SolarPipe.Core.Models;

namespace SolarPipe.Core.Interfaces;

public interface IDataFrame : IDisposable
{
    DataSchema Schema { get; }
    int RowCount { get; }

    float[] GetColumn(string name);
    float[] GetColumn(int index);

    // GetStringColumn: returns raw string values for a column with ColumnType.String.
    // Returns null if the column is not a string type or not found.
    // Default implementation returns null — override in concrete frames that store strings.
    string[]? GetStringColumn(string name) => null;

    IDataFrame Slice(int startRow, int count);
    IDataFrame SelectColumns(params string[] columns);
    IDataFrame AddColumn(string name, float[] values);

    IDataView ToDataView(MLContext mlContext);
    float[][] ToArray();

    IDataFrame ResampleAndAlign(TimeSpan cadence);
}
