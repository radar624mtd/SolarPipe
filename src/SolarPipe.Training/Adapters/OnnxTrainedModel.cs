using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SolarPipe.Core.Interfaces;
using SolarPipe.Core.Models;

namespace SolarPipe.Training.Adapters;

// OnnxTrainedModel wraps an InferenceSession for two modes:
//   Standard: single forward pass per prediction batch.
//   NeuralOde: RULE-070 — dynamics network only is loaded; Dormand-Prince ODE solver
//              runs in C# calling ORT ~200 times per prediction (RK4(5) adaptive steps).
//
// IDisposable: InferenceSession holds native ORT resources — must Dispose promptly.
public sealed class OnnxTrainedModel : ITrainedModel, IDisposable
{
    private readonly StageConfig _config;
    private readonly InferenceSession _session;
    private readonly bool _isNeuralOde;
    private bool _disposed;

    public string ModelId { get; }
    public string StageName => _config.Name;
    public ModelMetrics Metrics { get; }

    public OnnxTrainedModel(
        StageConfig config,
        InferenceSession session,
        bool isNeuralOde,
        ModelMetrics metrics)
    {
        _config = config ?? throw new ArgumentNullException(nameof(config));
        _session = session ?? throw new ArgumentNullException(nameof(session));
        _isNeuralOde = isNeuralOde;
        Metrics = metrics ?? throw new ArgumentNullException(nameof(metrics));
        ModelId = $"{config.Name}_{config.ModelType}_onnx";
    }

    public Task<PredictionResult> PredictAsync(IDataFrame input, CancellationToken ct)
    {
        ct.ThrowIfCancellationRequested();

        var result = _isNeuralOde
            ? PredictNeuralOde(input, ct)
            : PredictStandard(input, ct);

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
        throw new NotSupportedException(
            $"OnnxTrainedModel does not support SaveAsync — ONNX models are loaded from disk, not serialized here. " +
            $"Stage: '{StageName}'.");
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
