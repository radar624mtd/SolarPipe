using System.Text.Json;
using SolarPipe.Core.Interfaces;
using SolarPipe.Core.Models;

namespace SolarPipe.Prediction;

// CompositionDecomposer: error attribution for the winning hypothesis (RULE-167).
//
// Runs only on the winning hypothesis — not inside the sweep loop.
// For each stage, collects raw predictions (bypassing composition) to attribute
// how much each stage reduces error.
//
// Output: composition_decomposition_{hypothesis_id}.json written to outputDir.
public sealed class CompositionDecomposer
{
    // Decompose a multi-stage hypothesis.
    // stages: ordered list of (stageName, model) — same order as compose expression.
    // data: full dataset (training + validation combined, or just validation — caller's choice).
    // targetColumn: name of the observed values column.
    // hypothesisId: used for output filename.
    // outputDir: directory to write result JSON.
    public async Task<DecompositionResult> DecomposeAsync(
        string hypothesisId,
        IReadOnlyList<(string Name, ITrainedModel Model)> stages,
        IDataFrame data,
        string targetColumn,
        string outputDir,
        CancellationToken ct)
    {
        if (stages is null || stages.Count == 0)
            throw new ArgumentException("At least one stage required.", nameof(stages));
        if (data is null) throw new ArgumentNullException(nameof(data));
        if (data.RowCount == 0)
            throw new ArgumentException("Data must not be empty.", nameof(data));

        float[] observed = data.GetColumn(targetColumn);

        // Per-stage raw predictions (independent of composition)
        var stagePredictions = new Dictionary<string, float[]>(StringComparer.OrdinalIgnoreCase);
        foreach (var (name, model) in stages)
        {
            ct.ThrowIfCancellationRequested();
            var pred = await model.PredictAsync(data, ct);
            stagePredictions[name] = pred.Values;
        }

        // 1. Per-stage scatter metrics (each stage vs observed, independently)
        var stageMetrics = new Dictionary<string, StageMetrics>();
        foreach (var (name, preds) in stagePredictions)
        {
            stageMetrics[name] = ComputeStageMetrics(observed, preds);
        }

        // 2. Residual reduction: error at each composition step
        //    Step 0 = first stage (baseline); subsequent steps show improvement
        var residualReductions = new List<ResidualReductionEntry>();
        double? previousMae = null;
        foreach (var (name, _) in stages)
        {
            double mae = stageMetrics[name].Mae;
            double reductionPct = previousMae.HasValue && previousMae.Value > 1e-12
                ? (previousMae.Value - mae) / previousMae.Value * 100.0
                : double.NaN;
            residualReductions.Add(new ResidualReductionEntry(name, mae, reductionPct));
            previousMae = mae;
        }

        // 3. Stage correlation matrix (Pearson between all stage predictions)
        var correlationMatrix = BuildCorrelationMatrix(stages, stagePredictions);

        // 4. Error attribution: fraction of total variance explained by each stage
        double totalVariance = ComputeVariance(observed);
        var errorAttribution = new Dictionary<string, double>();
        foreach (var (name, preds) in stagePredictions)
        {
            double stageVariance = ComputeVariance(preds);
            errorAttribution[name] = totalVariance > 1e-12
                ? stageVariance / totalVariance
                : double.NaN;
        }

        var result = new DecompositionResult(
            hypothesisId,
            DateTime.UtcNow,
            stageMetrics,
            residualReductions,
            correlationMatrix,
            errorAttribution);

        // Write to output file (RULE-167)
        Directory.CreateDirectory(outputDir);
        var outPath = Path.Combine(outputDir, $"composition_decomposition_{hypothesisId}.json");
        var json = JsonSerializer.Serialize(result, new JsonSerializerOptions
        {
            WriteIndented = true,
            NumberHandling = System.Text.Json.Serialization.JsonNumberHandling.AllowNamedFloatingPointLiterals,
        });
        await File.WriteAllTextAsync(outPath, json, ct);

        Console.WriteLine(
            $"[decomposer hypothesis:{hypothesisId}] " +
            $"Written to {outPath}");

        return result;
    }

    // ── Metric helpers ────────────────────────────────────────────────────────

    private static StageMetrics ComputeStageMetrics(float[] observed, float[] predicted)
    {
        int n = Math.Min(observed.Length, predicted.Length);
        if (n == 0) return new StageMetrics(double.NaN, double.NaN, double.NaN, double.NaN);

        double sumAbs = 0, ssRes = 0, sumObs = 0, sumBias = 0;
        int valid = 0;

        for (int i = 0; i < n; i++)
        {
            if (float.IsNaN(observed[i]) || float.IsNaN(predicted[i])) continue;
            double diff = observed[i] - predicted[i];
            sumAbs += Math.Abs(diff);
            ssRes  += diff * diff;
            sumObs += observed[i];
            sumBias += predicted[i] - observed[i];
            valid++;
        }

        if (valid == 0) return new StageMetrics(double.NaN, double.NaN, double.NaN, double.NaN);

        double meanObs = sumObs / valid;
        double ssTot = 0;
        for (int i = 0; i < n; i++)
        {
            if (float.IsNaN(observed[i]) || float.IsNaN(predicted[i])) continue;
            double dev = observed[i] - meanObs;
            ssTot += dev * dev;
        }

        return new StageMetrics(
            Mae:  sumAbs / valid,
            Rmse: Math.Sqrt(ssRes / valid),
            R2:   ssTot < 1e-12 ? 0.0 : 1.0 - ssRes / ssTot,
            Bias: sumBias / valid);
    }

    private static Dictionary<string, Dictionary<string, double>> BuildCorrelationMatrix(
        IReadOnlyList<(string Name, ITrainedModel Model)> stages,
        IReadOnlyDictionary<string, float[]> preds)
    {
        var matrix = new Dictionary<string, Dictionary<string, double>>(
            StringComparer.OrdinalIgnoreCase);

        foreach (var (nameA, _) in stages)
        {
            var row = new Dictionary<string, double>(StringComparer.OrdinalIgnoreCase);
            foreach (var (nameB, _) in stages)
            {
                row[nameB] = nameA.Equals(nameB, StringComparison.OrdinalIgnoreCase)
                    ? 1.0
                    : PearsonCorrelation(preds[nameA], preds[nameB]);
            }
            matrix[nameA] = row;
        }

        return matrix;
    }

    private static double PearsonCorrelation(float[] x, float[] y)
    {
        int n = Math.Min(x.Length, y.Length);
        if (n < 2) return double.NaN;

        double sumX = 0, sumY = 0;
        int valid = 0;
        for (int i = 0; i < n; i++)
        {
            if (float.IsNaN(x[i]) || float.IsNaN(y[i])) continue;
            sumX += x[i]; sumY += y[i]; valid++;
        }
        if (valid < 2) return double.NaN;

        double meanX = sumX / valid, meanY = sumY / valid;
        double cov = 0, varX = 0, varY = 0;
        for (int i = 0; i < n; i++)
        {
            if (float.IsNaN(x[i]) || float.IsNaN(y[i])) continue;
            double dx = x[i] - meanX, dy = y[i] - meanY;
            cov  += dx * dy;
            varX += dx * dx;
            varY += dy * dy;
        }

        double denom = Math.Sqrt(varX * varY);
        return denom < 1e-12 ? double.NaN : cov / denom;
    }

    private static double ComputeVariance(float[] values)
    {
        int n = values.Length;
        if (n < 2) return double.NaN;
        double sum = 0;
        int valid = 0;
        for (int i = 0; i < n; i++)
        {
            if (float.IsNaN(values[i])) continue;
            sum += values[i]; valid++;
        }
        if (valid < 2) return double.NaN;
        double mean = sum / valid;
        double variance = 0;
        for (int i = 0; i < n; i++)
        {
            if (float.IsNaN(values[i])) continue;
            double d = values[i] - mean;
            variance += d * d;
        }
        return variance / valid;
    }
}

// ── Result types ──────────────────────────────────────────────────────────────

public sealed record StageMetrics(double Mae, double Rmse, double R2, double Bias);

public sealed record ResidualReductionEntry(
    string StageName,
    double Mae,
    double ReductionPct);

public sealed record DecompositionResult(
    string HypothesisId,
    DateTime GeneratedAt,
    Dictionary<string, StageMetrics> StageMetrics,
    IReadOnlyList<ResidualReductionEntry> ResidualReduction,
    Dictionary<string, Dictionary<string, double>> CorrelationMatrix,
    Dictionary<string, double> ErrorAttribution);
