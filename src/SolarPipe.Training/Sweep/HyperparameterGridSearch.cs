using SolarPipe.Core.Interfaces;
using SolarPipe.Core.Models;
using SolarPipe.Training.Evaluation;

namespace SolarPipe.Training.Sweep;

// Hyperparameter grid search with automatic Latin Hypercube fallback (RULE-166).
//
// If full grid > 200 combinations → auto-switch to LHS(100, seed=42).
// Both paths produce identical output schema: GridSearchResult.
// Grid search evaluates on training folds only (RULE-164 — calibration fold excluded).
public sealed class HyperparameterGridSearch
{
    private const int MaxFullGridSize  = 200;
    private const int LhsSampleSize    = 100;
    private const int LhsSeed          = 42;

    private readonly IFrameworkAdapter _adapter;
    private readonly ComprehensiveMetricsEvaluator _evaluator;

    public HyperparameterGridSearch(
        IFrameworkAdapter adapter,
        ComprehensiveMetricsEvaluator evaluator)
    {
        _adapter   = adapter   ?? throw new ArgumentNullException(nameof(adapter));
        _evaluator = evaluator ?? throw new ArgumentNullException(nameof(evaluator));
    }

    // parameterGrid: each key maps to a list of candidate values (same types as hyperparameter dict).
    // trainingFolds: list of (train, test) pairs — calibration fold MUST be excluded (RULE-164).
    // baseConfig: stage config whose Hyperparameters will be overridden per candidate.
    // dbmBaselineMae: per-fold DBM MAE for skill score (index-aligned with trainingFolds).
    public async Task<GridSearchResult> SearchAsync(
        StageConfig baseConfig,
        IReadOnlyDictionary<string, IReadOnlyList<object>> parameterGrid,
        IReadOnlyList<(IDataFrame Train, IDataFrame Test)> trainingFolds,
        IReadOnlyList<double> dbmBaselineMae,
        CancellationToken ct)
    {
        if (parameterGrid is null) throw new ArgumentNullException(nameof(parameterGrid));
        if (trainingFolds is null || trainingFolds.Count == 0)
            throw new ArgumentException("At least one training fold required.", nameof(trainingFolds));

        var candidates = BuildCandidates(parameterGrid, out bool usedLhs);

        if (usedLhs)
            Console.WriteLine(
                $"[grid:LHS combinations=100 reason=grid_too_large stage={baseConfig.Name}]");

        var results = new List<GridSearchEntry>(candidates.Count);
        foreach (var hp in candidates)
        {
            ct.ThrowIfCancellationRequested();

            var stageWithHp = baseConfig with
            {
                Hyperparameters = MergeHyperparameters(baseConfig.Hyperparameters, hp)
            };

            var foldMetrics = new List<FoldMetrics>(trainingFolds.Count);
            for (int foldIdx = 0; foldIdx < trainingFolds.Count; foldIdx++)
            {
                ct.ThrowIfCancellationRequested();

                var (train, test) = trainingFolds[foldIdx];
                var model = await _adapter.TrainAsync(stageWithHp, train, null, ct);
                var preds = await model.PredictAsync(test, ct);

                float[] obs  = test.GetColumn(stageWithHp.Target);
                double dbmMae = foldIdx < dbmBaselineMae.Count
                    ? dbmBaselineMae[foldIdx] : double.NaN;

                foldMetrics.Add(_evaluator.EvaluateFold(foldIdx, obs, preds.Values, dbmMae));

                (model as IDisposable)?.Dispose();
            }

            var aggregated = _evaluator.AggregateFolds(foldMetrics);
            results.Add(new GridSearchEntry(hp, foldMetrics, aggregated));
        }

        // Rank by mean MAE ascending (primary ranking per spec)
        results.Sort((a, b) => a.Aggregated.MaeMean.CompareTo(b.Aggregated.MaeMean));

        return new GridSearchResult(
            results,
            BestHyperparameters: results[0].Hyperparameters,
            UsedLatinHypercube: usedLhs);
    }

    // ── Candidate generation ────────────────────────────────────────────────────

    private static List<IReadOnlyDictionary<string, object>> BuildCandidates(
        IReadOnlyDictionary<string, IReadOnlyList<object>> grid,
        out bool usedLhs)
    {
        long totalCombinations = 1;
        foreach (var values in grid.Values)
            totalCombinations = checked(totalCombinations * values.Count);

        if (totalCombinations > MaxFullGridSize)
        {
            usedLhs = true;
            return LatinHypercubeSample(grid, LhsSampleSize, LhsSeed);
        }

        usedLhs = false;
        return ExpandFullGrid(grid);
    }

    private static List<IReadOnlyDictionary<string, object>> ExpandFullGrid(
        IReadOnlyDictionary<string, IReadOnlyList<object>> grid)
    {
        var keys   = grid.Keys.ToArray();
        var values = keys.Select(k => grid[k]).ToArray();
        var result = new List<IReadOnlyDictionary<string, object>>();
        ExpandGridRecursive(keys, values, 0, new Dictionary<string, object>(), result);
        return result;
    }

    private static void ExpandGridRecursive(
        string[] keys,
        IReadOnlyList<object>[] values,
        int depth,
        Dictionary<string, object> current,
        List<IReadOnlyDictionary<string, object>> result)
    {
        if (depth == keys.Length)
        {
            result.Add(new Dictionary<string, object>(current));
            return;
        }
        foreach (var v in values[depth])
        {
            current[keys[depth]] = v;
            ExpandGridRecursive(keys, values, depth + 1, current, result);
        }
        current.Remove(keys[depth]);
    }

    // Latin Hypercube Sampling: divide each dimension into n equal strata,
    // assign one sample per stratum per dimension, shuffle independently.
    private static List<IReadOnlyDictionary<string, object>> LatinHypercubeSample(
        IReadOnlyDictionary<string, IReadOnlyList<object>> grid,
        int n,
        int seed)
    {
        var keys   = grid.Keys.ToArray();
        var values = keys.Select(k => grid[k]).ToArray();
        var rng    = new Random(seed);

        // For each dimension, create a permutation of indices into that dimension's values,
        // scaled to pick from the full value list using stratum-based selection.
        var dimPerms = new int[keys.Length][];
        for (int d = 0; d < keys.Length; d++)
        {
            int m = values[d].Count;
            // Create n indices uniformly drawn from [0, m) without replacement per stratum
            var indices = new int[n];
            for (int i = 0; i < n; i++)
                indices[i] = (int)Math.Floor((double)i * m / n);
            // Shuffle
            for (int i = n - 1; i > 0; i--)
            {
                int j = rng.Next(i + 1);
                (indices[i], indices[j]) = (indices[j], indices[i]);
            }
            dimPerms[d] = indices;
        }

        var result = new List<IReadOnlyDictionary<string, object>>(n);
        for (int i = 0; i < n; i++)
        {
            var point = new Dictionary<string, object>(keys.Length);
            for (int d = 0; d < keys.Length; d++)
                point[keys[d]] = values[d][dimPerms[d][i]];
            result.Add(point);
        }
        return result;
    }

    private static IReadOnlyDictionary<string, object> MergeHyperparameters(
        IReadOnlyDictionary<string, object>? baseHp,
        IReadOnlyDictionary<string, object> overrides)
    {
        var merged = baseHp is not null
            ? new Dictionary<string, object>(baseHp, StringComparer.OrdinalIgnoreCase)
            : new Dictionary<string, object>(StringComparer.OrdinalIgnoreCase);

        foreach (var (k, v) in overrides)
            merged[k] = v;

        return merged;
    }
}

// Result of a single hyperparameter candidate evaluated across all training folds.
public sealed record GridSearchEntry(
    IReadOnlyDictionary<string, object> Hyperparameters,
    IReadOnlyList<FoldMetrics> FoldMetrics,
    AggregatedMetrics Aggregated);

// Full grid search result, ordered by mean MAE ascending (best first).
public sealed record GridSearchResult(
    IReadOnlyList<GridSearchEntry> Entries,
    IReadOnlyDictionary<string, object> BestHyperparameters,
    bool UsedLatinHypercube);
