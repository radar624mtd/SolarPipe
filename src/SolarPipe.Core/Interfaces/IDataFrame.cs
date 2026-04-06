using Microsoft.ML;
using SolarPipe.Core.Models;

namespace SolarPipe.Core.Interfaces;

public interface IDataFrame : IDisposable
{
    DataSchema Schema { get; }
    int RowCount { get; }

    float[] GetColumn(string name);
    float[] GetColumn(int index);

    IDataFrame Slice(int startRow, int count);
    IDataFrame SelectColumns(params string[] columns);
    IDataFrame AddColumn(string name, float[] values);

    IDataView ToDataView(MLContext mlContext);
    float[][] ToArray();

    IDataFrame ResampleAndAlign(TimeSpan cadence);
}
