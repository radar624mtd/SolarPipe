namespace SolarPipe.Training.Physics;

// Dormand-Prince RK4(5) adaptive ODE solver — FSAL tableau, embedded 4th/5th order error estimation.
// RULE-030: Required for Burton ODE stability when τ < 0.36h (Carrington-class events).
// Reference: Dormand & Prince (1980), J. Comput. Appl. Math. 6:19-26.
public static class DormandPrinceSolver
{
    // Butcher tableau — Dormand-Prince coefficients
    // a[i][j]: stage weights; c[i]: time fractions; b5[i]: 5th-order weights; e[i]: error weights (b5-b4)
    private static readonly double[] C = [0.0, 1.0 / 5.0, 3.0 / 10.0, 4.0 / 5.0, 8.0 / 9.0, 1.0, 1.0];

    private static readonly double[][] A =
    [
        [],
        [1.0 / 5.0],
        [3.0 / 40.0,         9.0 / 40.0],
        [44.0 / 45.0,       -56.0 / 15.0,       32.0 / 9.0],
        [19372.0 / 6561.0,  -25360.0 / 2187.0,  64448.0 / 6561.0,  -212.0 / 729.0],
        [9017.0 / 3168.0,   -355.0 / 33.0,      46732.0 / 5247.0,   49.0 / 176.0,  -5103.0 / 18656.0],
        [35.0 / 384.0,       0.0,                500.0 / 1113.0,     125.0 / 192.0, -2187.0 / 6784.0,   11.0 / 84.0]
    ];

    // 5th-order solution weights (same as last A row for FSAL)
    private static readonly double[] B5 = [35.0 / 384.0, 0.0, 500.0 / 1113.0, 125.0 / 192.0, -2187.0 / 6784.0, 11.0 / 84.0, 0.0];

    // Error coefficients (B5 - B4), where B4 is the embedded 4th-order formula
    private static readonly double[] E =
    [
         71.0 / 57600.0,
          0.0,
        -71.0 / 16695.0,
         71.0 / 1920.0,
       -17253.0 / 339200.0,
         22.0 / 525.0,
         -1.0 / 40.0
    ];

    public const double DefaultAbsTol = 1e-8;
    public const double DefaultRelTol = 1e-6;
    public const double MaxStepScale = 5.0;
    public const double MinStepScale = 0.2;
    public const double SafetyFactor = 0.9;

    // Integrate scalar ODE: dy/dt = f(t, y) from t0 to tEnd with initial value y0.
    // Returns (tFinal, yFinal). Throws if step size collapses or NaN propagates.
    public static (double T, double Y) Integrate(
        Func<double, double, double> f,
        double t0,
        double tEnd,
        double y0,
        double h0,
        double absTol = DefaultAbsTol,
        double relTol = DefaultRelTol)
    {
        if (double.IsNaN(y0))
            throw new ArgumentException($"Initial value is NaN (stage=DormandPrince, t0={t0}).");
        if (h0 <= 0.0)
            throw new ArgumentOutOfRangeException(nameof(h0), $"Initial step size must be positive, got {h0}.");

        double t = t0;
        double y = y0;
        double h = Math.Min(h0, tEnd - t0);

        // FSAL: k7 from previous step becomes k1 of next step
        double k1 = f(t, y);
        GuardNaN(k1, t, y, 1);

        const int maxSteps = 100_000;
        for (int step = 0; step < maxSteps && t < tEnd; step++)
        {
            if (t + h > tEnd)
                h = tEnd - t;
            if (h <= 0.0)
                break;

            // Compute stages k2..k7
            double k2 = f(t + C[1] * h, y + h * (A[1][0] * k1));
            GuardNaN(k2, t, y, 2);

            double k3 = f(t + C[2] * h, y + h * (A[2][0] * k1 + A[2][1] * k2));
            GuardNaN(k3, t, y, 3);

            double k4 = f(t + C[3] * h, y + h * (A[3][0] * k1 + A[3][1] * k2 + A[3][2] * k3));
            GuardNaN(k4, t, y, 4);

            double k5 = f(t + C[4] * h, y + h * (A[4][0] * k1 + A[4][1] * k2 + A[4][2] * k3 + A[4][3] * k4));
            GuardNaN(k5, t, y, 5);

            double k6 = f(t + C[5] * h, y + h * (A[5][0] * k1 + A[5][1] * k2 + A[5][2] * k3 + A[5][3] * k4 + A[5][4] * k5));
            GuardNaN(k6, t, y, 6);

            // 5th-order solution (also serves as k7 seed for next FSAL step)
            double yNext = y + h * (B5[0] * k1 + B5[2] * k3 + B5[3] * k4 + B5[4] * k5 + B5[5] * k6);
            GuardNaN(yNext, t, y, 7);

            double k7 = f(t + h, yNext);
            GuardNaN(k7, t, y, 7);

            // Error estimate: difference between 5th and 4th order solutions
            double errEst = h * (E[0] * k1 + E[2] * k3 + E[3] * k4 + E[4] * k5 + E[5] * k6 + E[6] * k7);

            // Tolerance-scaled error norm
            double sc = absTol + relTol * Math.Max(Math.Abs(y), Math.Abs(yNext));
            double err = Math.Abs(errEst) / sc;

            if (err <= 1.0)
            {
                // Accept step
                t += h;
                y = yNext;
                k1 = k7; // FSAL reuse
            }

            // Adapt step size: h_new = h * min(5, max(0.2, 0.9 * (1/err)^0.2))
            double scale = err > 0.0
                ? SafetyFactor * Math.Pow(1.0 / err, 0.2)
                : MaxStepScale;
            scale = Math.Min(MaxStepScale, Math.Max(MinStepScale, scale));
            h *= scale;

            if (h < 1e-15)
                throw new InvalidOperationException(
                    $"DormandPrince: step size collapsed to {h:E3} at t={t:F4}, y={y:E6}. " +
                    $"Check ODE parameters (stage=DormandPrince).");
        }

        return (t, y);
    }

    // Integrate vector ODE: dy/dt = f(t, y[]) from t0 to tEnd.
    public static (double T, double[] Y) IntegrateVector(
        Func<double, double[], double[]> f,
        double t0,
        double tEnd,
        double[] y0,
        double h0,
        double absTol = DefaultAbsTol,
        double relTol = DefaultRelTol)
    {
        int n = y0.Length;

        for (int i = 0; i < n; i++)
            if (double.IsNaN(y0[i]))
                throw new ArgumentException($"Initial state y0[{i}] is NaN (stage=DormandPrince, t0={t0}).");

        if (h0 <= 0.0)
            throw new ArgumentOutOfRangeException(nameof(h0), $"Initial step size must be positive, got {h0}.");

        double t = t0;
        double[] y = (double[])y0.Clone();
        double h = Math.Min(h0, tEnd - t0);

        double[] k1 = f(t, y);
        GuardVectorNaN(k1, t, 1);

        double[] tmp = new double[n];

        const int maxSteps = 100_000;
        for (int step = 0; step < maxSteps && t < tEnd; step++)
        {
            if (t + h > tEnd)
                h = tEnd - t;
            if (h <= 0.0)
                break;

            double[] k2 = EvalStage(f, t, C[1], h, y, [k1], [A[1][0]], tmp, n);
            GuardVectorNaN(k2, t, 2);

            double[] k3 = EvalStage(f, t, C[2], h, y, [k1, k2], [A[2][0], A[2][1]], tmp, n);
            GuardVectorNaN(k3, t, 3);

            double[] k4 = EvalStage(f, t, C[3], h, y, [k1, k2, k3], [A[3][0], A[3][1], A[3][2]], tmp, n);
            GuardVectorNaN(k4, t, 4);

            double[] k5 = EvalStage(f, t, C[4], h, y, [k1, k2, k3, k4], [A[4][0], A[4][1], A[4][2], A[4][3]], tmp, n);
            GuardVectorNaN(k5, t, 5);

            double[] k6 = EvalStage(f, t, C[5], h, y, [k1, k2, k3, k4, k5], [A[5][0], A[5][1], A[5][2], A[5][3], A[5][4]], tmp, n);
            GuardVectorNaN(k6, t, 6);

            // 5th-order step
            double[] yNext = new double[n];
            for (int i = 0; i < n; i++)
                yNext[i] = y[i] + h * (B5[0] * k1[i] + B5[2] * k3[i] + B5[3] * k4[i] + B5[4] * k5[i] + B5[5] * k6[i]);
            GuardVectorNaN(yNext, t, 7);

            double[] k7 = f(t + h, yNext);
            GuardVectorNaN(k7, t, 7);

            // Error: max-norm across components
            double err = 0.0;
            for (int i = 0; i < n; i++)
            {
                double errEst = h * (E[0] * k1[i] + E[2] * k3[i] + E[3] * k4[i] + E[4] * k5[i] + E[5] * k6[i] + E[6] * k7[i]);
                double sc = absTol + relTol * Math.Max(Math.Abs(y[i]), Math.Abs(yNext[i]));
                err = Math.Max(err, Math.Abs(errEst) / sc);
            }

            if (err <= 1.0)
            {
                t += h;
                y = yNext;
                k1 = k7;
            }

            double scale = err > 0.0
                ? SafetyFactor * Math.Pow(1.0 / err, 0.2)
                : MaxStepScale;
            scale = Math.Min(MaxStepScale, Math.Max(MinStepScale, scale));
            h *= scale;

            if (h < 1e-15)
                throw new InvalidOperationException(
                    $"DormandPrince: step size collapsed to {h:E3} at t={t:F4}. " +
                    $"Check ODE parameters (stage=DormandPrince).");
        }

        return (t, y);
    }

    private static double[] EvalStage(
        Func<double, double[], double[]> f,
        double t, double ci, double h,
        double[] y, double[][] ks, double[] aij,
        double[] tmp, int n)
    {
        for (int i = 0; i < n; i++)
        {
            double sum = 0.0;
            for (int j = 0; j < ks.Length; j++)
                sum += aij[j] * ks[j][i];
            tmp[i] = y[i] + h * sum;
        }
        return f(t + ci * h, tmp);
    }

    private static void GuardNaN(double v, double t, double y, int stage)
    {
        if (double.IsNaN(v) || double.IsInfinity(v))
            throw new InvalidOperationException(
                $"DormandPrince: NaN/Inf at stage k{stage}, t={t:F4}, y={y:E6}. " +
                "Check ODE right-hand side and input parameters.");
    }

    private static void GuardVectorNaN(double[] v, double t, int stage)
    {
        for (int i = 0; i < v.Length; i++)
            if (double.IsNaN(v[i]) || double.IsInfinity(v[i]))
                throw new InvalidOperationException(
                    $"DormandPrince: NaN/Inf at stage k{stage}[{i}], t={t:F4}. " +
                    "Check ODE right-hand side and input parameters.");
    }
}
