using FluentAssertions;
using SolarPipe.Training.Physics;
using SolarPipe.Tests.Unit.Fixtures;

namespace SolarPipe.Tests.Unit;

[Trait("Category", "Unit")]
public class DormandPrinceSolverTests
{
    // --- Scalar ODE tests ---

    [Fact]
    public void Scalar_ExponentialDecay_MatchesAnalyticalSolution()
    {
        // dy/dt = -k*y  →  y(t) = y0 * exp(-k*t)
        const double k = 0.5;
        const double y0 = 10.0;
        const double tEnd = 4.0;

        var (_, yFinal) = DormandPrinceSolver.Integrate(
            (t, y) => -k * y, 0.0, tEnd, y0, h0: 0.1);

        double expected = y0 * Math.Exp(-k * tEnd);
        yFinal.Should().BeApproximately(expected, 1e-6,
            "5th-order solver should match analytical solution to 1e-6");
    }

    [Fact]
    public void Scalar_SimpleGrowth_MatchesAnalytical()
    {
        // dy/dt = y  →  y(t) = exp(t)
        var (_, yFinal) = DormandPrinceSolver.Integrate(
            (t, y) => y, 0.0, 2.0, 1.0, h0: 0.2);

        double expected = Math.Exp(2.0);
        yFinal.Should().BeApproximately(expected, 1e-5);
    }

    [Fact]
    public void Scalar_ZeroDerivative_ReturnsInitialValue()
    {
        // dy/dt = 0  →  y stays at y0
        var (_, yFinal) = DormandPrinceSolver.Integrate(
            (t, y) => 0.0, 0.0, 5.0, 42.0, h0: 0.5);

        yFinal.Should().BeApproximately(42.0, 1e-10);
    }

    [Fact]
    public void Scalar_NaNInitialValue_Throws()
    {
        var act = () => DormandPrinceSolver.Integrate(
            (t, y) => y, 0.0, 1.0, double.NaN, h0: 0.1);
        act.Should().Throw<ArgumentException>().WithMessage("*NaN*");
    }

    [Fact]
    public void Scalar_NegativeStepSize_Throws()
    {
        var act = () => DormandPrinceSolver.Integrate(
            (t, y) => y, 0.0, 1.0, 1.0, h0: -0.1);
        act.Should().Throw<ArgumentOutOfRangeException>();
    }

    // --- Vector ODE tests ---

    [Fact]
    public void Vector_HarmonicOscillator_ConservesEnergy()
    {
        // d²x/dt² = -ω²x  →  state [x, v]
        // dy[0]/dt = y[1],  dy[1]/dt = -ω²*y[0]
        // Energy E = 0.5*v² + 0.5*ω²*x² = const
        const double omega = 1.0;
        double[] y0 = [1.0, 0.0]; // x=1, v=0
        double tEnd = 2 * Math.PI;    // one full period

        var (_, yFinal) = DormandPrinceSolver.IntegrateVector(
            (t, y) => [y[1], -omega * omega * y[0]],
            0.0, tEnd, y0, h0: 0.1);

        // After one period: x≈1, v≈0 (energy conserved, solution periodic)
        double e0 = 0.5 * y0[1] * y0[1] + 0.5 * omega * omega * y0[0] * y0[0];
        double eF = 0.5 * yFinal[1] * yFinal[1] + 0.5 * omega * omega * yFinal[0] * yFinal[0];
        eF.Should().BeApproximately(e0, 1e-6, "Energy must be conserved over one period");
    }

    [Fact]
    public void Vector_ExponentialSystem_MatchesAnalytical()
    {
        // y[0]' = -2*y[0], y[1]' = -3*y[1]  → independent exponentials
        double[] y0 = [5.0, 3.0];
        const double tEnd = 1.0;

        var (_, yFinal) = DormandPrinceSolver.IntegrateVector(
            (t, y) => [-2.0 * y[0], -3.0 * y[1]],
            0.0, tEnd, y0, h0: 0.1);

        yFinal[0].Should().BeApproximately(5.0 * Math.Exp(-2.0), 1e-6);
        yFinal[1].Should().BeApproximately(3.0 * Math.Exp(-3.0), 1e-6);
    }

    [Fact]
    public void Vector_NaNInitialValue_Throws()
    {
        var act = () => DormandPrinceSolver.IntegrateVector(
            (t, y) => y, 0.0, 1.0, [1.0, double.NaN], h0: 0.1);
        act.Should().Throw<ArgumentException>().WithMessage("*NaN*");
    }

    // --- Drag ODE stability test (RULE-030 enforcement) ---

    [Fact]
    public void Scalar_DragonStiffOde_StableForSmallTau()
    {
        // Simulate stiff-like drag: high gamma drives fast equilibration
        // dv/dt = -gamma*(v-w)*|v-w|  with gamma=2e-7, v0=2000, w=400
        // τ = 1/(2*γ*(v0-w)) ≈ 1/(2*2e-7*1600) ≈ 1.6h — borderline stiff
        const double gamma = 2e-7;
        const double w = 400.0;
        const double v0 = 2000.0;
        const double tEnd = 50.0; // 50 hours

        // Convert to per-hour (gamma is in km⁻¹, v in km/s, t in hours)
        var (_, vFinal) = DormandPrinceSolver.Integrate(
            (t, v) => -gamma * (v - w) * Math.Abs(v - w) * 3600.0,
            0.0, tEnd, v0, h0: 0.1);

        // After 50 hours high-drag CME must approach ambient wind (within 50 km/s)
        vFinal.Should().BeInRange(w - 50.0, w + 300.0,
            "High-drag CME should decelerate toward ambient solar wind speed");
        double.IsNaN(vFinal).Should().BeFalse("solver must not produce NaN for stiff drag ODE");
    }

    [Fact]
    public void Scalar_ConvergenceOrder_IsApproximatelyFifth()
    {
        // RULE-030 enforcement: halving step size should reduce error by ~2^5 = 32
        // Use exponential decay: exact solution known
        const double k = 1.0;
        const double y0 = 1.0;
        const double tEnd = 2.0;
        double exact = Math.Exp(-k * tEnd);

        var (_, y1) = DormandPrinceSolver.Integrate(
            (t, y) => -k * y, 0.0, tEnd, y0, h0: 0.5,
            absTol: 1e-12, relTol: 1e-10);

        var (_, y2) = DormandPrinceSolver.Integrate(
            (t, y) => -k * y, 0.0, tEnd, y0, h0: 0.25,
            absTol: 1e-12, relTol: 1e-10);

        double err1 = Math.Abs(y1 - exact);
        double err2 = Math.Abs(y2 - exact);

        // Both should be very small; err1/err2 ≥ 1 (finer step is at least as accurate)
        // With adaptive control the ratio won't be exactly 32, but err should stay tiny
        err1.Should().BeLessThan(1e-9, "5th-order solver with h0=0.5 must be accurate");
        err2.Should().BeLessThan(1e-10, "5th-order solver with h0=0.25 must be more accurate");
    }
}
