using SolarPipe.Core.Domain;

namespace SolarPipe.Training.Physics;

// Phase 9 M1: step-by-step backtest scaffolding around the existing DragBased ODE.
// Runs the same dv/dt = -γ(v - w)|v - w| integration as DragBasedModel.RunOde, but
// exposes per-hour state so downstream code can log trajectories and (in M2) swap
// γ for a time-varying γ_eff(t) driven by L1 observations.
//
// RULE-030: uses the Dormand-Prince solver (not MathNet).
// RULE-032: same parameter ranges as DragBasedModel.
//
// The M1 implementation uses a constant γ throughout (StaticDragCoefficient), which
// must reproduce DragBasedModel.RunOde within solver tolerance. Tested by
// ProgressiveDragPropagatorTests.StaticReference_MatchesDragBasedModel.
public sealed class ProgressiveDragPropagator
{
    private const double HourSeconds = 3600.0;
    private const double KmPerHourPerSolarRadius = HourSeconds / PhysicalConstants.SolarRadiusKm;

    // Per-hour integration cadence (spec §6.3: step size never > 1 hour).
    private const double StepHours = 1.0;

    // Spec §6.3 v(t) clip range.
    private const double MinSpeedKmPerSec = 200.0;
    private const double MaxSpeedKmPerSec = 3000.0;

    private readonly IDragCoefficientProvider _gammaProvider;
    private readonly double _ambientWindKmPerSec;
    private readonly double _startSolarRadii;
    private readonly double _targetSolarRadii;
    private readonly double _maxHours;

    public ProgressiveDragPropagator(
        IDragCoefficientProvider gammaProvider,
        double ambientWindKmPerSec,
        double startSolarRadii = 21.5,
        double targetSolarRadii = 215.0,
        double maxHours = 72.0)
    {
        if (ambientWindKmPerSec < 200.0 || ambientWindKmPerSec > 1000.0)
            throw new ArgumentOutOfRangeException(nameof(ambientWindKmPerSec),
                $"Ambient wind {ambientWindKmPerSec} km/s outside [200, 1000] (stage=ProgressiveDragPropagator).");
        if (startSolarRadii < 20.0)
            throw new ArgumentOutOfRangeException(nameof(startSolarRadii),
                $"Start distance {startSolarRadii} R☉ must be ≥ 20 (stage=ProgressiveDragPropagator).");
        if (targetSolarRadii <= startSolarRadii)
            throw new ArgumentOutOfRangeException(nameof(targetSolarRadii),
                $"Target distance {targetSolarRadii} R☉ must exceed start {startSolarRadii} R☉.");
        if (maxHours <= 0.0 || maxHours > 200.0)
            throw new ArgumentOutOfRangeException(nameof(maxHours),
                $"maxHours {maxHours} outside (0, 200] (stage=ProgressiveDragPropagator).");

        _gammaProvider = gammaProvider ?? throw new ArgumentNullException(nameof(gammaProvider));
        _ambientWindKmPerSec = ambientWindKmPerSec;
        _startSolarRadii = startSolarRadii;
        _targetSolarRadii = targetSolarRadii;
        _maxHours = maxHours;
    }

    // Propagate from (r = startSolarRadii, v = v0) forward in hourly segments.
    // At each whole-hour boundary, queries gammaProvider for γ_eff(hourIndex, v, r)
    // and integrates one hour with that constant γ. Records per-step state.
    //
    // Termination:
    //   - r ≥ target → arrival by propagation
    //   - shockProvider (if non-null) fires → arrival by observation
    //   - total elapsed ≥ maxHours → timeout
    public ProgressiveTrajectory Propagate(
        double v0KmPerSec,
        IShockDetector? shockDetector = null)
    {
        if (v0KmPerSec < 200.0 || v0KmPerSec > 3500.0)
            throw new ArgumentOutOfRangeException(nameof(v0KmPerSec),
                $"Initial speed {v0KmPerSec} km/s outside [200, 3500] (stage=ProgressiveDragPropagator).");

        var steps = new List<ProgressiveStep>();
        double t = 0.0;
        double[] y = [_startSolarRadii, v0KmPerSec];

        // Record step 0 — initial state, no γ evaluated yet.
        var gamma0 = _gammaProvider.GammaAtHour(hourIndex: 0, speedKmPerSec: y[1], rSolarRadii: y[0]);
        steps.Add(new ProgressiveStep(
            HourIndex: 0, TimeHours: 0.0,
            RSolarRadii: y[0], SpeedKmPerSec: y[1],
            GammaKmInv: gamma0.GammaKmInv,
            NObs: gamma0.NObs, VObs: gamma0.VObs,
            GammaFellBack: gamma0.FellBack));

        double? arrivalHours = null;
        double? arrivalSpeed = null;
        string terminationReason = "timeout";
        bool shockArrived = false;
        int nMissingHours = gamma0.FellBack ? 1 : 0;

        int maxHourSteps = (int)Math.Ceiling(_maxHours / StepHours);
        for (int hourIndex = 1; hourIndex <= maxHourSteps; hourIndex++)
        {
            double tSegEnd = Math.Min(t + StepHours, _maxHours);
            if (tSegEnd <= t) break;

            // Evaluate γ at the *start* of the hour (observations are hourly-stamped at hour boundary).
            var gStep = _gammaProvider.GammaAtHour(hourIndex, y[1], y[0]);
            double gamma = Clip(gStep.GammaKmInv, 1e-9, 1e-6);
            if (gStep.FellBack) nMissingHours++;

            // Integrate one hour with constant γ.
            double[] yStart = (double[])y.Clone();
            var (tNew, yNew) = DormandPrinceSolver.IntegrateVector(
                (tau, yv) =>
                {
                    double v = yv[1];
                    double dDist = v * KmPerHourPerSolarRadius;
                    double dSpeed = -gamma * (v - _ambientWindKmPerSec) * Math.Abs(v - _ambientWindKmPerSec) * HourSeconds;
                    if (double.IsNaN(dDist) || double.IsNaN(dSpeed))
                        throw new InvalidOperationException(
                            $"NaN in drag ODE at t={tau:F3}h, v={v:F1} km/s, γ={gamma:E3} (stage=ProgressiveDragPropagator).");
                    return [dDist, dSpeed];
                },
                t, tSegEnd, y, h0: 0.1);

            // Clip speed to physical range (spec §6.3).
            yNew[1] = Clip(yNew[1], MinSpeedKmPerSec, MaxSpeedKmPerSec);

            // Target-distance crossing check (linear interp between yStart and yNew).
            if (yStart[0] < _targetSolarRadii && yNew[0] >= _targetSolarRadii)
            {
                double frac = (_targetSolarRadii - yStart[0]) / (yNew[0] - yStart[0]);
                arrivalHours = t + frac * (tNew - t);
                arrivalSpeed = yStart[1] + frac * (yNew[1] - yStart[1]);
                terminationReason = "target_reached";
                // Record final step at crossing point, not at hour boundary.
                steps.Add(new ProgressiveStep(
                    HourIndex: hourIndex, TimeHours: arrivalHours.Value,
                    RSolarRadii: _targetSolarRadii, SpeedKmPerSec: arrivalSpeed.Value,
                    GammaKmInv: gamma,
                    NObs: gStep.NObs, VObs: gStep.VObs,
                    GammaFellBack: gStep.FellBack));
                break;
            }

            t = tNew;
            y = yNew;

            steps.Add(new ProgressiveStep(
                HourIndex: hourIndex, TimeHours: t,
                RSolarRadii: y[0], SpeedKmPerSec: y[1],
                GammaKmInv: gamma,
                NObs: gStep.NObs, VObs: gStep.VObs,
                GammaFellBack: gStep.FellBack));

            // Shock-arrival check (hour-over-hour in the observation stream). M1 leaves
            // shockDetector null; M2 will wire in the Δv ≥ 200 + density×3 rule.
            if (shockDetector != null && shockDetector.ShockDetectedAtHour(hourIndex))
            {
                arrivalHours = t;
                arrivalSpeed = y[1];
                terminationReason = "shock_detected";
                shockArrived = true;
                break;
            }

            if (t >= _maxHours)
            {
                terminationReason = "timeout";
                break;
            }
        }

        return new ProgressiveTrajectory(
            Steps: steps,
            ArrivalTimeHours: arrivalHours,
            ArrivalSpeedKmPerSec: arrivalSpeed,
            TerminationReason: terminationReason,
            ShockArrived: shockArrived,
            NMissingHours: nMissingHours,
            TotalHours: Math.Max(0, steps.Count - 1));
    }

    private static double Clip(double value, double min, double max) =>
        value < min ? min : (value > max ? max : value);
}

// Provides γ at each hour of the propagation. M1 uses StaticDragCoefficient
// (returns γ₀ always). M2 will add DensityModulatedDrag which reads n_obs from
// an L1 stream and returns γ₀ · (n_obs / n_ref).
public interface IDragCoefficientProvider
{
    GammaStep GammaAtHour(int hourIndex, double speedKmPerSec, double rSolarRadii);
}

public readonly record struct GammaStep(
    double GammaKmInv,
    double? NObs,
    double? VObs,
    bool FellBack);

// M1 default: constant γ₀ for every hour.
public sealed class StaticDragCoefficient : IDragCoefficientProvider
{
    private readonly double _gammaKmInv;

    public StaticDragCoefficient(double gammaKmInv)
    {
        if (gammaKmInv < 1e-9 || gammaKmInv > 1e-6)
            throw new ArgumentOutOfRangeException(nameof(gammaKmInv),
                $"γ₀ {gammaKmInv:E3} km⁻¹ outside [1e-9, 1e-6] (stage=StaticDragCoefficient).");
        _gammaKmInv = gammaKmInv;
    }

    public GammaStep GammaAtHour(int hourIndex, double speedKmPerSec, double rSolarRadii) =>
        new(GammaKmInv: _gammaKmInv, NObs: null, VObs: null, FellBack: false);
}

// Per-hour shock detector hook. M1 leaves this null; M2 supplies one backed by
// the L1ObservationStream.
public interface IShockDetector
{
    bool ShockDetectedAtHour(int hourIndex);
}

// Per-step record written to the trajectory JSON.
public sealed record ProgressiveStep(
    int HourIndex,
    double TimeHours,
    double RSolarRadii,
    double SpeedKmPerSec,
    double GammaKmInv,
    double? NObs,
    double? VObs,
    bool GammaFellBack);

public sealed record ProgressiveTrajectory(
    IReadOnlyList<ProgressiveStep> Steps,
    double? ArrivalTimeHours,
    double? ArrivalSpeedKmPerSec,
    string TerminationReason,
    bool ShockArrived,
    int NMissingHours,
    int TotalHours);
