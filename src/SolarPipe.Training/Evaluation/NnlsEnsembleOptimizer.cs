namespace SolarPipe.Training.Evaluation;

// Non-negative least squares ensemble weight optimizer (RULE-165).
//
// Solves: min ‖Aw - y‖² subject to w >= 0, sum(w) = 1
// where A = [predictions_member1 | predictions_member2 | ...] (n x k matrix)
//       y = observed values (n x 1)
//       w = weights (k x 1)
//
// Method: projected gradient descent with simplex projection.
// The simplex constraint (w >= 0, sum = 1) is enforced after each gradient step
// using the Duchi et al. O(k log k) algorithm.
//
// Weights are returned as an immutable dictionary keyed by stage name.
// Caller (SweepCommand) must write weights back to config (RULE-165).
public sealed class NnlsEnsembleOptimizer
{
    private readonly int _maxIterations;
    private readonly double _tolerance;
    private readonly double _learningRate;

    public NnlsEnsembleOptimizer(
        int maxIterations = 2000,
        double tolerance = 1e-8,
        double learningRate = 0.01)
    {
        if (maxIterations < 1)
            throw new ArgumentOutOfRangeException(nameof(maxIterations));
        _maxIterations = maxIterations;
        _tolerance = tolerance;
        _learningRate = learningRate;
    }

    // memberPredictions: dict keyed by stage name → predictions on calibration fold.
    // observed: ground truth on calibration fold.
    // Returns optimal weights summing to 1, all >= 0. (RULE-165)
    public IReadOnlyDictionary<string, float> Optimize(
        IReadOnlyDictionary<string, float[]> memberPredictions,
        float[] observed)
    {
        if (memberPredictions is null) throw new ArgumentNullException(nameof(memberPredictions));
        if (observed is null) throw new ArgumentNullException(nameof(observed));
        if (memberPredictions.Count == 0)
            throw new ArgumentException("At least one member prediction is required.", nameof(memberPredictions));

        var stageNames = memberPredictions.Keys.ToArray();
        int k = stageNames.Length;
        int n = observed.Length;

        if (n == 0)
            throw new ArgumentException("Observed array must not be empty.", nameof(observed));

        foreach (var (name, pred) in memberPredictions)
        {
            if (pred.Length != n)
                throw new ArgumentException(
                    $"Member '{name}' prediction length {pred.Length} does not match observed length {n}.");
        }

        // Filter rows where any member prediction or observed value is NaN.
        // Physics models produce NaN for rows with missing input features (e.g. null ambient wind).
        var validRows = Enumerable.Range(0, n)
            .Where(i => !float.IsNaN(observed[i])
                     && stageNames.All(s => !float.IsNaN(memberPredictions[s][i])))
            .ToArray();

        if (validRows.Length == 0)
            throw new ArgumentException(
                "All rows contain NaN in predictions or observed — cannot optimize weights.");

        int nValid = validRows.Length;

        // Build matrix A (nValid x k) as flat row-major using only non-NaN rows
        var A = new double[nValid, k];
        var obsValid = new double[nValid];
        for (int j = 0; j < k; j++)
        {
            var pred = memberPredictions[stageNames[j]];
            for (int r = 0; r < nValid; r++)
                A[r, j] = pred[validRows[r]];
        }
        for (int r = 0; r < nValid; r++)
            obsValid[r] = observed[validRows[r]];

        n = nValid;
        observed = obsValid.Select(v => (float)v).ToArray();

        // Degenerate case: single member gets weight 1
        if (k == 1)
        {
            return new Dictionary<string, float> { [stageNames[0]] = 1.0f };
        }

        // Initialize with uniform weights on the simplex
        var w = new double[k];
        for (int j = 0; j < k; j++) w[j] = 1.0 / k;

        // Precompute AtA and Aty for gradient: grad = AtA*w - Aty (= At(Aw - y))
        var AtA = new double[k, k];
        var Aty = new double[k];

        for (int j1 = 0; j1 < k; j1++)
        {
            for (int j2 = 0; j2 < k; j2++)
            {
                double dot = 0;
                for (int i = 0; i < n; i++) dot += A[i, j1] * A[i, j2];
                AtA[j1, j2] = dot;
            }
            double rhs = 0;
            for (int i = 0; i < n; i++) rhs += A[i, j1] * observed[i];
            Aty[j1] = rhs;
        }

        // Step size: 1 / (2 * spectral_norm_approx). Use max diagonal of AtA as upper bound.
        double maxDiag = 0;
        for (int j = 0; j < k; j++) maxDiag = Math.Max(maxDiag, AtA[j, j]);
        double stepSize = maxDiag > 1e-12 ? _learningRate / maxDiag : _learningRate;

        var grad = new double[k];
        for (int iter = 0; iter < _maxIterations; iter++)
        {
            // Gradient: AtA*w - Aty
            for (int j = 0; j < k; j++)
            {
                double g = -Aty[j];
                for (int j2 = 0; j2 < k; j2++) g += AtA[j, j2] * w[j2];
                grad[j] = g;
            }

            // Gradient step
            var wNew = new double[k];
            for (int j = 0; j < k; j++)
                wNew[j] = w[j] - stepSize * grad[j];

            // Project onto probability simplex
            SimplexProject(wNew);

            // Convergence check
            double delta = 0;
            for (int j = 0; j < k; j++) delta += (wNew[j] - w[j]) * (wNew[j] - w[j]);
            w = wNew;

            if (Math.Sqrt(delta) < _tolerance) break;
        }

        // Build result dictionary (RULE-165: returned, not applied silently)
        var result = new Dictionary<string, float>(StringComparer.OrdinalIgnoreCase);
        for (int j = 0; j < k; j++)
            result[stageNames[j]] = (float)Math.Max(0.0, w[j]);

        return result;
    }

    // Project vector v onto the probability simplex {w : w >= 0, sum(w) = 1}.
    // In-place. Duchi et al. 2008 algorithm — O(k log k).
    internal static void SimplexProject(double[] v)
    {
        int k = v.Length;
        var sorted = (double[])v.Clone();
        Array.Sort(sorted);
        Array.Reverse(sorted);   // descending

        double cumSum = 0;
        double rho = 0;
        for (int j = 0; j < k; j++)
        {
            cumSum += sorted[j];
            if (sorted[j] - (cumSum - 1.0) / (j + 1) > 0)
                rho = j + 1;
        }

        double theta = 0;
        cumSum = 0;
        for (int j = 0; j < rho; j++) cumSum += sorted[j];
        theta = (cumSum - 1.0) / rho;

        for (int j = 0; j < k; j++)
            v[j] = Math.Max(v[j] - theta, 0.0);
    }
}
