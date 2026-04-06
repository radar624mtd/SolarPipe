using SolarPipe.Core.Interfaces;
using SolarPipe.Core.Models;
using SolarPipe.Data.DataFrame;

namespace SolarPipe.Data.Transforms;

// TransformEngine applies a sequence of named transforms to an IDataFrame.
// Each transform produces a new column added to the frame (non-destructive).
// RULE-120: sentinel values must already be NaN before reaching transforms.
// RULE-121: NaN propagation — any NaN in input produces NaN in output for that row.
public sealed class TransformEngine
{
    private readonly List<ITransform> _transforms = [];

    public TransformEngine Add(ITransform transform)
    {
        _transforms.Add(transform ?? throw new ArgumentNullException(nameof(transform)));
        return this;
    }

    public IDataFrame Apply(IDataFrame input)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        IDataFrame current = input;
        foreach (var t in _transforms)
        {
            var next = t.Apply(current);
            if (!ReferenceEquals(next, current))
            {
                // If transform returned a new frame, dispose the intermediate
                // only if we created it (i.e., not the original input).
                if (!ReferenceEquals(current, input))
                    current.Dispose();
            }
            current = next;
        }
        return current;
    }
}

public interface ITransform
{
    string OutputColumnName { get; }
    IDataFrame Apply(IDataFrame input);
}

// ─── Normalize ──────────────────────────────────────────────────────────────────
// Scales column to [0, 1] using provided or computed min/max.
public sealed class NormalizeTransform : ITransform
{
    private readonly string _sourceColumn;
    private readonly float? _fixedMin;
    private readonly float? _fixedMax;

    public string OutputColumnName { get; }

    public NormalizeTransform(string sourceColumn, string? outputColumn = null,
        float? min = null, float? max = null)
    {
        _sourceColumn = sourceColumn ?? throw new ArgumentNullException(nameof(sourceColumn));
        OutputColumnName = outputColumn ?? sourceColumn + "_norm";
        _fixedMin = min;
        _fixedMax = max;
    }

    public IDataFrame Apply(IDataFrame input)
    {
        var col = input.GetColumn(_sourceColumn);
        float min = _fixedMin ?? ComputeMin(col);
        float max = _fixedMax ?? ComputeMax(col);
        float range = max - min;

        var result = new float[col.Length];
        for (int i = 0; i < col.Length; i++)
        {
            if (float.IsNaN(col[i])) { result[i] = float.NaN; continue; }
            result[i] = range < 1e-10f ? 0f : (col[i] - min) / range;
        }
        return input.AddColumn(OutputColumnName, result);
    }

    private static float ComputeMin(float[] col)
    {
        float min = float.MaxValue;
        foreach (float v in col)
            if (!float.IsNaN(v) && v < min) min = v;
        return min == float.MaxValue ? 0f : min;
    }

    private static float ComputeMax(float[] col)
    {
        float max = float.MinValue;
        foreach (float v in col)
            if (!float.IsNaN(v) && v > max) max = v;
        return max == float.MinValue ? 0f : max;
    }
}

// ─── Standardize ────────────────────────────────────────────────────────────────
// Zero-mean, unit-variance standardization (z-score).
public sealed class StandardizeTransform : ITransform
{
    private readonly string _sourceColumn;

    public string OutputColumnName { get; }

    public StandardizeTransform(string sourceColumn, string? outputColumn = null)
    {
        _sourceColumn = sourceColumn ?? throw new ArgumentNullException(nameof(sourceColumn));
        OutputColumnName = outputColumn ?? sourceColumn + "_std";
    }

    public IDataFrame Apply(IDataFrame input)
    {
        var col = input.GetColumn(_sourceColumn);
        ComputeMeanStd(col, out float mean, out float std);

        var result = new float[col.Length];
        for (int i = 0; i < col.Length; i++)
        {
            if (float.IsNaN(col[i])) { result[i] = float.NaN; continue; }
            result[i] = std < 1e-10f ? 0f : (col[i] - mean) / std;
        }
        return input.AddColumn(OutputColumnName, result);
    }

    internal static void ComputeMeanStd(float[] col, out float mean, out float std)
    {
        double sum = 0, sum2 = 0;
        int count = 0;
        foreach (float v in col)
        {
            if (float.IsNaN(v)) continue;
            sum += v; sum2 += (double)v * v; count++;
        }
        if (count == 0) { mean = 0f; std = 0f; return; }
        mean = (float)(sum / count);
        double variance = sum2 / count - (double)mean * mean;
        std = (float)Math.Sqrt(Math.Max(0.0, variance));
    }
}

// ─── LogScale ───────────────────────────────────────────────────────────────────
// Natural log transform: log(|x| + offset). NaN for negative values if strict=true.
public sealed class LogScaleTransform : ITransform
{
    private readonly string _sourceColumn;
    private readonly float _offset;
    private readonly bool _strict;

    public string OutputColumnName { get; }

    public LogScaleTransform(string sourceColumn, string? outputColumn = null,
        float offset = 1f, bool strict = false)
    {
        _sourceColumn = sourceColumn ?? throw new ArgumentNullException(nameof(sourceColumn));
        OutputColumnName = outputColumn ?? sourceColumn + "_log";
        _offset = offset;
        _strict = strict;
    }

    public IDataFrame Apply(IDataFrame input)
    {
        var col = input.GetColumn(_sourceColumn);
        var result = new float[col.Length];
        for (int i = 0; i < col.Length; i++)
        {
            if (float.IsNaN(col[i])) { result[i] = float.NaN; continue; }
            if (_strict && col[i] < 0f) { result[i] = float.NaN; continue; }
            result[i] = MathF.Log(MathF.Abs(col[i]) + _offset);
        }
        return input.AddColumn(OutputColumnName, result);
    }
}

// ─── Lag ────────────────────────────────────────────────────────────────────────
// Produces lagged version of a column: result[i] = col[i - lag_steps * step].
// Rows where the look-back goes below index 0 are filled with NaN.
public sealed class LagTransform : ITransform
{
    private readonly string _sourceColumn;
    private readonly int _lagSteps;
    private readonly int _step;

    public string OutputColumnName { get; }

    public LagTransform(string sourceColumn, int lagSteps = 1, int step = 1,
        string? outputColumn = null)
    {
        if (lagSteps < 1) throw new ArgumentOutOfRangeException(nameof(lagSteps),
            $"LagTransform: lagSteps must be >= 1, got {lagSteps}.");
        if (step < 1) throw new ArgumentOutOfRangeException(nameof(step),
            $"LagTransform: step must be >= 1, got {step}.");

        _sourceColumn = sourceColumn ?? throw new ArgumentNullException(nameof(sourceColumn));
        _lagSteps = lagSteps;
        _step = step;
        OutputColumnName = outputColumn ?? $"{sourceColumn}_lag{lagSteps}";
    }

    public IDataFrame Apply(IDataFrame input)
    {
        var col = input.GetColumn(_sourceColumn);
        int offset = _lagSteps * _step;
        var result = new float[col.Length];
        for (int i = 0; i < col.Length; i++)
        {
            int srcIdx = i - offset;
            result[i] = srcIdx < 0 ? float.NaN : col[srcIdx];
        }
        return input.AddColumn(OutputColumnName, result);
    }
}

// ─── WindowStats ────────────────────────────────────────────────────────────────
// Rolling window statistics. Supported ops: mean, std, min, max.
// Window is causal (only past rows): window covers [i-size+1, i].
// Rows with fewer than minCount non-NaN values emit NaN.
public sealed class WindowStatsTransform : ITransform
{
    public enum StatOp { Mean, Std, Min, Max }

    private readonly string _sourceColumn;
    private readonly int _windowSize;
    private readonly StatOp _op;
    private readonly int _minCount;

    public string OutputColumnName { get; }

    public WindowStatsTransform(string sourceColumn, int windowSize, StatOp op,
        string? outputColumn = null, int minCount = 1)
    {
        if (windowSize < 1) throw new ArgumentOutOfRangeException(nameof(windowSize),
            $"WindowStatsTransform: windowSize must be >= 1, got {windowSize}.");

        _sourceColumn = sourceColumn ?? throw new ArgumentNullException(nameof(sourceColumn));
        _windowSize = windowSize;
        _op = op;
        _minCount = minCount;
        OutputColumnName = outputColumn ?? $"{sourceColumn}_win{windowSize}_{op.ToString().ToLowerInvariant()}";
    }

    public IDataFrame Apply(IDataFrame input)
    {
        var col = input.GetColumn(_sourceColumn);
        var result = new float[col.Length];
        for (int i = 0; i < col.Length; i++)
        {
            int start = Math.Max(0, i - _windowSize + 1);
            result[i] = ComputeWindow(col, start, i + 1);
        }
        return input.AddColumn(OutputColumnName, result);
    }

    private float ComputeWindow(float[] col, int start, int end)
    {
        double sum = 0, sum2 = 0;
        float wMin = float.MaxValue, wMax = float.MinValue;
        int count = 0;
        for (int j = start; j < end; j++)
        {
            if (float.IsNaN(col[j])) continue;
            sum += col[j]; sum2 += (double)col[j] * col[j];
            if (col[j] < wMin) wMin = col[j];
            if (col[j] > wMax) wMax = col[j];
            count++;
        }
        if (count < _minCount) return float.NaN;
        return _op switch
        {
            StatOp.Mean => (float)(sum / count),
            StatOp.Std  => (float)Math.Sqrt(Math.Max(0.0, sum2 / count - (sum / count) * (sum / count))),
            StatOp.Min  => wMin,
            StatOp.Max  => wMax,
            _ => throw new InvalidOperationException($"Unknown StatOp: {_op}")
        };
    }
}
