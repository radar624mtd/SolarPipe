using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using ParquetSharp;
using SolarPipe.Core.Interfaces;
using SolarPipe.Core.Models;

namespace SolarPipe.Training.Adapters;

// OnnxInferenceMode: selects the forward-pass shape used by OnnxTrainedModel.
public enum OnnxInferenceMode
{
    // Standard: single forward pass on a (rows, features) tensor. First output is the scalar prediction.
    Standard,
    // NeuralOde: loaded ONNX is f(y,t) — dynamics only. C# runs Dormand-Prince RK4(5) per row.
    NeuralOde,
    // TftPinn (G5): 4 named inputs (x_flat, m_flat, x_seq, m_seq) → 3 named outputs (p10, p50, p90).
    // Values = P50, LowerBound = P10, UpperBound = P90. Sequence tensor is loaded from
    // the Parquet file referenced by hyperparameter `sequences_path` using the activity_id column.
    TftPinn,
}

// OnnxTrainedModel wraps an InferenceSession for three modes:
//   Standard: single forward pass per prediction batch.
//   NeuralOde: RULE-070 — dynamics network only is loaded; Dormand-Prince ODE solver
//              runs in C# calling ORT ~200 times per prediction (RK4(5) adaptive steps).
//   TftPinn: G5 ONNX graph with named inputs x_flat/m_flat/x_seq/m_seq → p10/p50/p90.
//
// IDisposable: InferenceSession holds native ORT resources — must Dispose promptly.
public sealed class OnnxTrainedModel : ITrainedModel, IDisposable
{
    private readonly StageConfig _config;
    private readonly InferenceSession _session;
    private readonly OnnxInferenceMode _mode;
    private bool _disposed;

    public string ModelId { get; }
    public string StageName => _config.Name;
    public ModelMetrics Metrics { get; }

    public OnnxTrainedModel(
        StageConfig config,
        InferenceSession session,
        OnnxInferenceMode mode,
        ModelMetrics metrics)
    {
        _config = config ?? throw new ArgumentNullException(nameof(config));
        _session = session ?? throw new ArgumentNullException(nameof(session));
        _mode = mode;
        Metrics = metrics ?? throw new ArgumentNullException(nameof(metrics));
        ModelId = $"{config.Name}_{config.ModelType}_onnx";
    }

    public Task<PredictionResult> PredictAsync(IDataFrame input, CancellationToken ct)
    {
        ct.ThrowIfCancellationRequested();

        var result = _mode switch
        {
            OnnxInferenceMode.NeuralOde => PredictNeuralOde(input, ct),
            OnnxInferenceMode.TftPinn => PredictTftPinn(input, ct),
            _ => PredictStandard(input, ct),
        };

        return Task.FromResult(result);
    }

    // Standard ONNX inference: flatten all feature columns into a (rows × features) tensor,
    // run one forward pass, extract the first output as float predictions.
    private PredictionResult PredictStandard(IDataFrame input, CancellationToken ct)
    {
        ct.ThrowIfCancellationRequested();

        int rows = input.RowCount;
        int featureCount = _config.Features.Count;

        // Build input tensor: row-major [rows, featureCount]
        var data = new float[rows * featureCount];
        for (int f = 0; f < featureCount; f++)
        {
            var col = input.GetColumn(_config.Features[f]);
            for (int r = 0; r < rows; r++)
                data[r * featureCount + f] = col[r];
        }

        var inputName = _session.InputMetadata.Keys.First();
        var tensor = new DenseTensor<float>(data, [rows, featureCount]);
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(inputName, tensor)
        };

        using var outputs = _session.Run(inputs);
        var outputTensor = outputs.First().AsTensor<float>();

        var values = new float[rows];
        for (int i = 0; i < rows; i++)
            values[i] = outputTensor[i];

        return new PredictionResult(
            Values: values,
            LowerBound: null,
            UpperBound: null,
            ModelId: ModelId,
            GeneratedAt: DateTime.UtcNow);
    }

    // NeuralOde inference: RULE-070 pattern.
    // The loaded ONNX is f(y, t, θ) — the dynamics network only.
    // We run a Dormand-Prince RK4(5) C# solver that calls ORT each step.
    //
    // Input convention (from config.Features):
    //   "state"  — current state variable (scalar per row)
    //   "t_start" — initial time (hours)
    //   "t_end"   — integration endpoint (hours)
    //
    // If the exact column names are absent, we fall back to the first three feature columns.
    private PredictionResult PredictNeuralOde(IDataFrame input, CancellationToken ct)
    {
        int rows = input.RowCount;
        var values = new float[rows];

        // Attempt to resolve state/time columns by name (OrdinalIgnoreCase).
        float[] stateCol = ResolveColumn(input, "state", 0);
        float[] tStartCol = ResolveColumn(input, "t_start", 1);
        float[] tEndCol = ResolveColumn(input, "t_end", 2);

        for (int r = 0; r < rows; r++)
        {
            ct.ThrowIfCancellationRequested();
            values[r] = (float)IntegrateOde(
                y0: stateCol[r],
                t0: tStartCol[r],
                t1: tEndCol[r]);
        }

        return new PredictionResult(
            Values: values,
            LowerBound: null,
            UpperBound: null,
            ModelId: ModelId,
            GeneratedAt: DateTime.UtcNow);
    }

    // TftPinn inference: 4 named inputs → 3 named outputs.
    //   x_flat (B, N_FLAT) = feature values with NaN→0
    //   m_flat (B, N_FLAT) = 1.0 where observed, else 0.0
    //   x_seq  (B, T, C)   = OMNI sequences loaded from Parquet keyed by activity_id
    //   m_seq  (B, T, C)   = 1.0 where observed, else 0.0
    // Outputs: p10 / p50 / p90 → Values = P50, LowerBound = P10, UpperBound = P90.
    private PredictionResult PredictTftPinn(IDataFrame input, CancellationToken ct)
    {
        ct.ThrowIfCancellationRequested();

        int rows = input.RowCount;
        int nFlat = _config.Features.Count;

        // Build x_flat + m_flat (NaN → 0, mask 0; observed → value, mask 1).
        var xFlat = new float[rows * nFlat];
        var mFlat = new float[rows * nFlat];
        for (int f = 0; f < nFlat; f++)
        {
            var col = input.GetColumn(_config.Features[f]);
            for (int r = 0; r < rows; r++)
            {
                float v = col[r];
                bool observed = !float.IsNaN(v);
                xFlat[r * nFlat + f] = observed ? v : 0f;
                mFlat[r * nFlat + f] = observed ? 1f : 0f;
            }
        }

        // Resolve sequence file path + split from hyperparameters / input row
        string sequencesPath = FindHyperString("sequences_path")
            ?? Environment.GetEnvironmentVariable("SOLARPIPE_SEQUENCES_PATH")
            ?? "data/sequences";
        string split = FindHyperString("split") ?? "holdout";

        string seqFile = Path.Combine(sequencesPath, $"{split}_sequences.parquet");
        if (!File.Exists(seqFile))
            throw new FileNotFoundException(
                $"TftPinn ONNX predict: sequence parquet not found at '{seqFile}'. " +
                $"Set hyperparameter 'sequences_path' or env SOLARPIPE_SEQUENCES_PATH.", seqFile);

        // Prefer string activity_id column (set by SqliteProvider for ColumnType.String).
        // Fall back to float column for backward compat (older code paths).
        string[] activityIds = input.GetStringColumn("activity_id")
            ?? ResolveColumn(input, "activity_id", 0)
               .Select(FormatActivityId).ToArray();

        // Load shape (T, C) per activity; per-row lookup populates (B, T, C).
        var (tSeq, cSeq, perAidX, perAidM) = LoadPinnSequencesParquet(seqFile, activityIds, ct);

        var xSeq = new float[rows * tSeq * cSeq];
        var mSeq = new float[rows * tSeq * cSeq];
        for (int r = 0; r < rows; r++)
        {
            string key = activityIds[r];
            if (perAidX.TryGetValue(key, out var xArr) && perAidM.TryGetValue(key, out var mArr))
            {
                Buffer.BlockCopy(xArr, 0, xSeq, r * tSeq * cSeq * sizeof(float), xArr.Length * sizeof(float));
                Buffer.BlockCopy(mArr, 0, mSeq, r * tSeq * cSeq * sizeof(float), mArr.Length * sizeof(float));
            }
            // Rows missing from parquet: zeros + mask 0 (model can still infer from flat branch).
        }

        var xFlatTensor = new DenseTensor<float>(xFlat, [rows, nFlat]);
        var mFlatTensor = new DenseTensor<float>(mFlat, [rows, nFlat]);
        var xSeqTensor = new DenseTensor<float>(xSeq, [rows, tSeq, cSeq]);
        var mSeqTensor = new DenseTensor<float>(mSeq, [rows, tSeq, cSeq]);

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("x_flat", xFlatTensor),
            NamedOnnxValue.CreateFromTensor("m_flat", mFlatTensor),
            NamedOnnxValue.CreateFromTensor("x_seq",  xSeqTensor),
            NamedOnnxValue.CreateFromTensor("m_seq",  mSeqTensor),
        };

        using var outputs = _session.Run(inputs);
        var p10 = new float[rows];
        var p50 = new float[rows];
        var p90 = new float[rows];
        foreach (var o in outputs)
        {
            var tensor = o.AsTensor<float>();
            float[] dst = o.Name switch
            {
                "p10" => p10,
                "p50" => p50,
                "p90" => p90,
                _ => throw new InvalidOperationException(
                    $"TftPinn ONNX graph emitted unexpected output '{o.Name}' " +
                    $"(expected p10/p50/p90).")
            };
            for (int r = 0; r < rows; r++) dst[r] = tensor[r, 0];
        }

        return new PredictionResult(
            Values: p50,
            LowerBound: p10,
            UpperBound: p90,
            ModelId: ModelId,
            GeneratedAt: DateTime.UtcNow);
    }

    private string? FindHyperString(string key)
    {
        if (_config.Hyperparameters is null) return null;
        foreach (var kv in _config.Hyperparameters)
            if (string.Equals(kv.Key, key, StringComparison.OrdinalIgnoreCase))
                return kv.Value?.ToString();
        return null;
    }

    // Activity IDs in the Arrow/frame layer are cast to float — preserve integer printable form
    // where possible (CDAW activityIDs are large ints) so the Parquet's string keys line up.
    private static string FormatActivityId(float v)
    {
        if (float.IsNaN(v)) return "NaN";
        double d = v;
        if (Math.Abs(d - Math.Round(d)) < 1e-6)
            return ((long)Math.Round(d)).ToString(System.Globalization.CultureInfo.InvariantCulture);
        return d.ToString("G9", System.Globalization.CultureInfo.InvariantCulture);
    }

    // LoadPinnSequencesParquet: long-form Parquet reader matching python/build_pinn_sequences.py output.
    // Schema (expected): activity_id (string), window (string), timestep (int), plus OMNI channel columns.
    // Only rows with window == "pre_launch" are used. Missing channels → zero + mask 0.
    // Returns (T, C, xMap, mMap): T timesteps, C channels, per-aid flat (T*C) float arrays.
    private (int T, int C, Dictionary<string, float[]> xMap, Dictionary<string, float[]> mMap)
        LoadPinnSequencesParquet(string path, string[] targetAids, CancellationToken ct)
    {
        using var fileReader = new ParquetFileReader(path);
        var descriptor = fileReader.FileMetaData.Schema;

        // Discover channel columns (everything not in the {activity_id, window, timestep} key set).
        int nCols = descriptor.NumColumns;
        int aidCol = -1, winCol = -1, stepCol = -1;
        var channelCols = new List<(int Index, string Name)>();
        for (int i = 0; i < nCols; i++)
        {
            var name = descriptor.Column(i).Name;
            if (name.Equals("activity_id", StringComparison.OrdinalIgnoreCase)) aidCol = i;
            else if (name.Equals("window", StringComparison.OrdinalIgnoreCase)) winCol = i;
            else if (name.Equals("timestep", StringComparison.OrdinalIgnoreCase)) stepCol = i;
            else channelCols.Add((i, name));
        }
        if (aidCol < 0 || winCol < 0 || stepCol < 0)
            throw new InvalidOperationException(
                $"TftPinn sequence parquet '{path}' missing required columns " +
                $"(activity_id, window, timestep).");

        // Index target aids for O(1) membership check.
        var wantedAids = new HashSet<string>(targetAids, StringComparer.Ordinal);

        // Collect all pre_launch rows for wanted aids.
        // Long-form layout: each row group has aid/window/step/channel_1..channel_K.
        var perAidRows = new Dictionary<string, List<(int Step, float[] Vals, short[] DefLevels)>>();

        int totalRowGroups = fileReader.FileMetaData.NumRowGroups;
        int tMin = int.MaxValue, tMax = int.MinValue;
        int C = channelCols.Count;

        for (int rg = 0; rg < totalRowGroups; rg++)
        {
            ct.ThrowIfCancellationRequested();
            using var rgReader = fileReader.RowGroup(rg);
            int rowsInGroup = (int)rgReader.MetaData.NumRows;

            var aids = ReadStringColumn(rgReader.Column(aidCol), rowsInGroup);
            var windows = ReadStringColumn(rgReader.Column(winCol), rowsInGroup);
            var steps = ReadIntColumn(rgReader.Column(stepCol), rowsInGroup);

            var channelData = new float[C][];
            var channelDef  = new short[C][];
            for (int c = 0; c < C; c++)
                (channelData[c], channelDef[c]) = ReadChannelColumn(rgReader.Column(channelCols[c].Index), rowsInGroup);

            for (int r = 0; r < rowsInGroup; r++)
            {
                if (!string.Equals(windows[r], "pre_launch", StringComparison.Ordinal)) continue;
                string aid = aids[r];
                if (!wantedAids.Contains(aid)) continue;

                int step = steps[r];
                if (step < tMin) tMin = step;
                if (step > tMax) tMax = step;

                var vals = new float[C];
                var def  = new short[C];
                for (int c = 0; c < C; c++)
                {
                    vals[c] = channelData[c][r];
                    def[c]  = channelDef[c][r];
                }

                if (!perAidRows.TryGetValue(aid, out var list))
                {
                    list = new List<(int, float[], short[])>();
                    perAidRows[aid] = list;
                }
                list.Add((step, vals, def));
            }
        }

        int T = (tMin == int.MaxValue) ? 0 : (tMax - tMin + 1);
        var xMap = new Dictionary<string, float[]>(perAidRows.Count);
        var mMap = new Dictionary<string, float[]>(perAidRows.Count);
        foreach (var kv in perAidRows)
        {
            var xArr = new float[T * C];
            var mArr = new float[T * C];
            foreach (var (step, vals, def) in kv.Value)
            {
                int tIdx = step - tMin;
                if (tIdx < 0 || tIdx >= T) continue;
                for (int c = 0; c < C; c++)
                {
                    float v = vals[c];
                    bool observed = def[c] != 0 && !float.IsNaN(v);
                    xArr[tIdx * C + c] = observed ? v : 0f;
                    mArr[tIdx * C + c] = observed ? 1f : 0f;
                }
            }
            xMap[kv.Key] = xArr;
            mMap[kv.Key] = mArr;
        }

        return (T, C, xMap, mMap);
    }

    private static string[] ReadStringColumn(ColumnReader columnReader, int rowCount)
    {
        var result = new string[rowCount];
        if (columnReader is ColumnReader<ByteArray> baReader)
        {
            var defLevels = new short[rowCount];
            var repLevels = new short[rowCount];
            var values = new ByteArray[rowCount];
            long rowsRead = baReader.ReadBatch(rowCount, defLevels, repLevels, values, out long valuesRead);
            int vi = 0;
            for (int r = 0; r < (int)rowsRead; r++)
            {
                if (defLevels[r] == 0)
                    result[r] = string.Empty;
                else if (vi < (int)valuesRead)
                    result[r] = DecodeByteArray(values[vi++]);
                else
                    result[r] = string.Empty;
            }
        }
        return result;
    }

    private static string DecodeByteArray(ByteArray ba)
    {
        if (ba.Length == 0 || ba.Pointer == IntPtr.Zero) return string.Empty;
        var bytes = new byte[ba.Length];
        System.Runtime.InteropServices.Marshal.Copy(ba.Pointer, bytes, 0, ba.Length);
        return System.Text.Encoding.UTF8.GetString(bytes);
    }

    private static int[] ReadIntColumn(ColumnReader columnReader, int rowCount)
    {
        var result = new int[rowCount];
        if (columnReader is ColumnReader<int> intReader)
        {
            var defLevels = new short[rowCount];
            var repLevels = new short[rowCount];
            var values = new int[rowCount];
            long rowsRead = intReader.ReadBatch(rowCount, defLevels, repLevels, values, out long valuesRead);
            int vi = 0;
            for (int r = 0; r < (int)rowsRead; r++)
                result[r] = defLevels[r] == 0 ? 0 : (vi < (int)valuesRead ? values[vi++] : 0);
        }
        else if (columnReader is ColumnReader<long> longReader)
        {
            var defLevels = new short[rowCount];
            var repLevels = new short[rowCount];
            var values = new long[rowCount];
            long rowsRead = longReader.ReadBatch(rowCount, defLevels, repLevels, values, out long valuesRead);
            int vi = 0;
            for (int r = 0; r < (int)rowsRead; r++)
                result[r] = defLevels[r] == 0 ? 0 : (vi < (int)valuesRead ? (int)values[vi++] : 0);
        }
        return result;
    }

    private static (float[] Values, short[] DefLevels) ReadChannelColumn(ColumnReader columnReader, int rowCount)
    {
        var values = new float[rowCount];
        var defLevels = new short[rowCount];
        var repLevels = new short[rowCount];

        if (columnReader is ColumnReader<float> floatReader)
        {
            floatReader.ReadBatch(rowCount, defLevels, repLevels, values, out _);
        }
        else if (columnReader is ColumnReader<double> doubleReader)
        {
            var doubles = new double[rowCount];
            doubleReader.ReadBatch(rowCount, defLevels, repLevels, doubles, out long valsRead);
            for (int i = 0; i < (int)valsRead; i++) values[i] = (float)doubles[i];
        }
        else
        {
            // Unsupported type — treat all as null.
            for (int i = 0; i < rowCount; i++) { values[i] = float.NaN; defLevels[i] = 0; }
        }

        // Realign values into row positions when nulls are present (ReadBatch packs non-null values).
        bool hasNulls = false;
        for (int i = 0; i < rowCount; i++) if (defLevels[i] == 0) { hasNulls = true; break; }
        if (hasNulls)
        {
            int vi = 0;
            int nonNull = 0;
            for (int i = 0; i < rowCount; i++) if (defLevels[i] != 0) nonNull++;
            // values currently holds the first nonNull values packed at [0..nonNull); expand in-place right-to-left.
            vi = nonNull - 1;
            for (int r = rowCount - 1; r >= 0; r--)
            {
                if (defLevels[r] == 0) values[r] = float.NaN;
                else values[r] = vi >= 0 ? values[vi--] : float.NaN;
            }
        }

        return (values, defLevels);
    }

    // Dormand-Prince RK4(5) adaptive ODE solver.
    // Each call to Dynamics() invokes the ONNX session — approximately 50 steps × 4 evaluations
    // = ~200 ORT calls per prediction (feasible for SolarPipe batch sizes per ADR-006).
    private double IntegrateOde(double y0, double t0, double t1)
    {
        if (t1 <= t0) return y0;

        // Dormand-Prince coefficients
        const double a21 = 1.0 / 5.0;
        const double a31 = 3.0 / 40.0, a32 = 9.0 / 40.0;
        const double a41 = 44.0 / 45.0, a42 = -56.0 / 15.0, a43 = 32.0 / 9.0;
        const double a51 = 19372.0 / 6561.0, a52 = -25360.0 / 2187.0, a53 = 64448.0 / 6561.0, a54 = -212.0 / 729.0;
        const double a61 = 9017.0 / 3168.0, a62 = -355.0 / 33.0, a63 = 46732.0 / 5247.0, a64 = 49.0 / 176.0, a65 = -5103.0 / 18656.0;

        // 5th-order weights
        const double b1 = 35.0 / 384.0, b3 = 500.0 / 1113.0, b4 = 125.0 / 192.0, b5 = -2187.0 / 6784.0, b6 = 11.0 / 84.0;

        // 4th-order weights (for error estimate)
        const double e1 = 71.0 / 57600.0, e3 = -71.0 / 16695.0, e4 = 71.0 / 1920.0, e5 = -17253.0 / 339200.0, e6 = 22.0 / 525.0, e7 = -1.0 / 40.0;

        const double relTol = 1e-4;
        const double absTol = 1e-6;
        const int maxSteps = 500;

        double t = t0;
        double y = y0;
        double h = (t1 - t0) / 50.0;
        double hMin = (t1 - t0) * 1e-8;

        for (int step = 0; step < maxSteps && t < t1; step++)
        {
            if (t + h > t1) h = t1 - t;
            if (h < hMin) h = hMin;

            double k1 = h * Dynamics(y, t);
            double k2 = h * Dynamics(y + a21 * k1, t + h / 5.0);
            double k3 = h * Dynamics(y + a31 * k1 + a32 * k2, t + 3.0 * h / 10.0);
            double k4 = h * Dynamics(y + a41 * k1 + a42 * k2 + a43 * k3, t + 4.0 * h / 5.0);
            double k5 = h * Dynamics(y + a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4, t + 8.0 * h / 9.0);
            double k6 = h * Dynamics(y + a61 * k1 + a62 * k2 + a63 * k3 + a64 * k4 + a65 * k5, t + h);

            double y5 = y + b1 * k1 + b3 * k3 + b4 * k4 + b5 * k5 + b6 * k6;

            // Error estimate requires k7 = f(y5, t+h)
            double k7 = h * Dynamics(y5, t + h);
            double err = Math.Abs(e1 * k1 + e3 * k3 + e4 * k4 + e5 * k5 + e6 * k6 + e7 * k7);

            double scale = absTol + relTol * Math.Max(Math.Abs(y), Math.Abs(y5));
            double errNorm = (scale > 0) ? err / scale : err;

            if (errNorm <= 1.0 || h <= hMin)
            {
                t += h;
                y = y5;
            }

            // PI step-size control
            double factor = (errNorm > 0) ? 0.9 * Math.Pow(1.0 / errNorm, 0.2) : 5.0;
            h *= Math.Clamp(factor, 0.1, 5.0);

            if (double.IsNaN(y) || double.IsInfinity(y))
                return double.NaN;
        }

        return y;
    }

    // Calls the ONNX dynamics network f(y, t) → dy/dt.
    // Input: scalar tensor [1, 2] = [y, t]. Output: scalar tensor [1, 1].
    private double Dynamics(double y, double t)
    {
        var inputName = _session.InputMetadata.Keys.First();
        var data = new float[] { (float)y, (float)t };
        var tensor = new DenseTensor<float>(data, [1, 2]);

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(inputName, tensor)
        };

        using var outputs = _session.Run(inputs);
        return outputs.First().AsTensor<float>()[0];
    }

    private static float[] ResolveColumn(IDataFrame frame, string name, int fallbackIndex)
    {
        foreach (var col in frame.Schema.Columns)
            if (string.Equals(col.Name, name, StringComparison.OrdinalIgnoreCase))
                return frame.GetColumn(col.Name);

        // Fallback: use positional index if schema has enough columns
        if (fallbackIndex < frame.Schema.Columns.Count)
            return frame.GetColumn(fallbackIndex);

        return new float[frame.RowCount]; // zeros
    }

    public Task SaveAsync(string path, CancellationToken ct)
    {
        // ONNX models are already on disk; write a zero-byte placeholder so the
        // registry can compute a fingerprint without requiring a full re-serialize.
        File.WriteAllBytes(path, []);
        return Task.CompletedTask;
    }

    public Task LoadAsync(string path, CancellationToken ct)
    {
        throw new NotSupportedException(
            $"OnnxTrainedModel does not support LoadAsync. Use OnnxAdapter.TrainAsync with 'model_path' to load. " +
            $"Stage: '{StageName}'.");
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _session.Dispose();
        GC.SuppressFinalize(this);
    }
}
