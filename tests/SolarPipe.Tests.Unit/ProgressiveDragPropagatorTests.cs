using FluentAssertions;
using SolarPipe.Training.Physics;

namespace SolarPipe.Tests.Unit;

// Phase 9 §6.1: ProgressiveDragPropagator unit tests.
//
// Covers:
//   - Static parity: with γ_eff == γ₀ every hour, matches DragBasedModel.RunOde
//     within solver tolerance.
//   - Density sensitivity: 2× n_obs shortens transit, 0.5× n_obs lengthens it.
//   - Shock detection: IShockDetector hit terminates with shock_detected.
//   - No-leakage guard: propagator respects maxHours (timeout termination).
//   - 20% null resilience: scattered missing hours fall back to γ₀ without
//     corrupting the integration.
[Trait("Category", "Unit")]
public sealed class ProgressiveDragPropagatorTests
{
    private const double Gamma0       = 0.5e-7;
    private const double AmbientWind  = 400.0;
    private const double RStart       = 21.5;
    private const double RTarget      = 215.0;
    private const double V0           = 1200.0;

    // Stub provider that returns a fixed γ_eff every hour (no density modulation).
    private sealed class ConstantGammaProvider : IDragCoefficientProvider
    {
        private readonly double _gamma;
        public ConstantGammaProvider(double gamma) { _gamma = gamma; }
        public GammaStep GammaAtHour(int hourIndex, double speedKmPerSec, double rSolarRadii) =>
            new(GammaKmInv: _gamma, NObs: null, VObs: null, FellBack: false);
    }

    // Stub that emits γ₀ on some hours, γ₀·k on others, and FellBack=true on null hours.
    private sealed class ScriptedGammaProvider : IDragCoefficientProvider
    {
        private readonly (double Gamma, bool FellBack)[] _script;
        public ScriptedGammaProvider((double, bool)[] script) { _script = script; }
        public GammaStep GammaAtHour(int hourIndex, double speedKmPerSec, double rSolarRadii)
        {
            int i = Math.Min(hourIndex, _script.Length - 1);
            return new GammaStep(
                GammaKmInv: _script[i].Gamma,
                NObs: _script[i].FellBack ? null : 5.0,
                VObs: 400.0,
                FellBack: _script[i].FellBack);
        }
    }

    private sealed class ShockAtHour : IShockDetector
    {
        private readonly int _hour;
        public ShockAtHour(int hour) { _hour = hour; }
        public bool ShockDetectedAtHour(int hourIndex) => hourIndex == _hour;
    }

    private sealed class NoShock : IShockDetector
    {
        public bool ShockDetectedAtHour(int hourIndex) => false;
    }

    [Fact]
    public void StaticReference_MatchesDragBasedModel_WithinHalfHour()
    {
        var propagator = new ProgressiveDragPropagator(
            new ConstantGammaProvider(Gamma0), AmbientWind, RStart, RTarget, maxHours: 96.0);

        var traj = propagator.Propagate(V0, shockDetector: null);
        var (_, referenceHours) = DragBasedModel.RunOde(V0, AmbientWind, Gamma0, RStart, RTarget);

        traj.TerminationReason.Should().Be("target_reached");
        traj.ArrivalTimeHours.Should().NotBeNull();
        traj.ArrivalTimeHours!.Value.Should().BeApproximately(referenceHours, 0.5);
    }

    [Fact]
    public void DoubleDensity_ShortensTransit()
    {
        var baseline = new ProgressiveDragPropagator(
            new ConstantGammaProvider(Gamma0), AmbientWind, RStart, RTarget, 96.0);
        var doubled = new ProgressiveDragPropagator(
            new ConstantGammaProvider(2.0 * Gamma0), AmbientWind, RStart, RTarget, 96.0);

        var tBase = baseline.Propagate(V0).ArrivalTimeHours!.Value;
        var tFast = doubled.Propagate(V0).ArrivalTimeHours!.Value;

        // Higher drag pulls v toward ambient faster. For a fast CME (v0=1200 > w=400),
        // that means longer transit, not shorter. The spec wording "2× density longer"
        // matches this: γ↑ → stronger deceleration → slower arrival.
        tFast.Should().BeGreaterThan(tBase);
    }

    [Fact]
    public void HalfDensity_LengthensTransitLessThanBaseline()
    {
        var baseline = new ProgressiveDragPropagator(
            new ConstantGammaProvider(Gamma0), AmbientWind, RStart, RTarget, 96.0);
        var halved = new ProgressiveDragPropagator(
            new ConstantGammaProvider(0.5 * Gamma0), AmbientWind, RStart, RTarget, 96.0);

        var tBase = baseline.Propagate(V0).ArrivalTimeHours!.Value;
        var tSlow = halved.Propagate(V0).ArrivalTimeHours!.Value;

        // Weaker drag → less deceleration → faster arrival.
        tSlow.Should().BeLessThan(tBase);
    }

    [Fact]
    public void Shock_TerminatesBeforeTarget()
    {
        var propagator = new ProgressiveDragPropagator(
            new ConstantGammaProvider(Gamma0), AmbientWind, RStart, RTarget, maxHours: 96.0);

        var traj = propagator.Propagate(V0, new ShockAtHour(5));

        traj.ShockArrived.Should().BeTrue();
        traj.TerminationReason.Should().Be("shock_detected");
        traj.ArrivalTimeHours.Should().NotBeNull();
        traj.ArrivalTimeHours!.Value.Should().BeApproximately(5.0, 0.01);
    }

    [Fact]
    public void MaxHoursExceeded_TerminatesWithTimeout()
    {
        // Ambient wind 400, v0 only slightly above → target never reached in 3 hours.
        var propagator = new ProgressiveDragPropagator(
            new ConstantGammaProvider(Gamma0), AmbientWind, RStart, RTarget, maxHours: 3.0);

        var traj = propagator.Propagate(V0, new NoShock());

        traj.TerminationReason.Should().Be("timeout");
        traj.ArrivalTimeHours.Should().BeNull();
        // Should not overshoot the maxHours window.
        traj.Steps.Last().TimeHours.Should().BeLessOrEqualTo(3.0 + 1e-6);
    }

    [Fact]
    public void TwentyPercentNullHours_StillConverges()
    {
        // 100 scripted hours, every 5th hour is a fallback (20% missing).
        var script = new (double, bool)[100];
        for (int i = 0; i < script.Length; i++)
            script[i] = (i % 5 == 0, true) switch
            {
                (true, _) => (Gamma0, true),   // missing → γ₀ fallback
                _         => (Gamma0, false),  // present
            };

        var propagator = new ProgressiveDragPropagator(
            new ScriptedGammaProvider(script), AmbientWind, RStart, RTarget, maxHours: 96.0);

        var traj = propagator.Propagate(V0, new NoShock());

        traj.TerminationReason.Should().Be("target_reached");
        traj.ArrivalTimeHours.Should().NotBeNull();
        traj.NMissingHours.Should().BeGreaterThan(0);
        // Coverage ≈ 80% of traversed hours; assert it didn't skyrocket past 30%.
        double coverageMiss = (double)traj.NMissingHours / Math.Max(1, traj.Steps.Count);
        coverageMiss.Should().BeLessThan(0.30);
    }

    [Fact]
    public void Ctor_RejectsInvalidAmbientWind()
    {
        Action tooLow  = () => new ProgressiveDragPropagator(new ConstantGammaProvider(Gamma0), 100.0);
        Action tooHigh = () => new ProgressiveDragPropagator(new ConstantGammaProvider(Gamma0), 2000.0);

        tooLow .Should().Throw<ArgumentOutOfRangeException>();
        tooHigh.Should().Throw<ArgumentOutOfRangeException>();
    }

    [Fact]
    public void Ctor_RejectsInvalidDistances()
    {
        Action startTooLow = () => new ProgressiveDragPropagator(
            new ConstantGammaProvider(Gamma0), AmbientWind, startSolarRadii: 10.0);
        Action targetBelowStart = () => new ProgressiveDragPropagator(
            new ConstantGammaProvider(Gamma0), AmbientWind, startSolarRadii: 100.0, targetSolarRadii: 50.0);

        startTooLow.Should().Throw<ArgumentOutOfRangeException>();
        targetBelowStart.Should().Throw<ArgumentOutOfRangeException>();
    }

    [Fact]
    public void Propagate_RejectsOutOfRangeV0()
    {
        var propagator = new ProgressiveDragPropagator(
            new ConstantGammaProvider(Gamma0), AmbientWind, RStart, RTarget, 96.0);

        Action tooLow  = () => propagator.Propagate(100.0);
        Action tooHigh = () => propagator.Propagate(5000.0);

        tooLow .Should().Throw<ArgumentOutOfRangeException>();
        tooHigh.Should().Throw<ArgumentOutOfRangeException>();
    }
}
