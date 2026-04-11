using SolarPipe.Data;

namespace SolarPipe.Training.Physics;

// Phase 9 §3.3: density-modulated drag coefficient.
//
//   γ_eff(t) = γ₀ · (n_obs(t) / n_ref)
//
// Where n_ref is quiet-sun reference proton density (OMNI climatology median,
// 5 cm⁻³ by default), γ₀ is the static per-event drag parameter from the
// trained Phase 8 DragBasedModel, and n_obs(t) comes from L1ObservationStream.
//
// Missing n_obs → fall back to γ₀ for that hour.
// γ_eff is hard-clipped to [1e-9, 1e-6] km⁻¹ to guard against sensor glitches.
public sealed class DensityModulatedDrag : IDragCoefficientProvider
{
    public const double DefaultNReference = 5.0;   // cm⁻³, OMNI 1963-2023 quiet-sun median
    public const double MinGammaKmInv     = 1e-9;
    public const double MaxGammaKmInv     = 1e-6;

    private readonly double _gamma0;
    private readonly double _nRef;
    private readonly L1ObservationStream _stream;

    public DensityModulatedDrag(double gamma0KmInv, L1ObservationStream stream, double nRef = DefaultNReference)
    {
        if (gamma0KmInv < MinGammaKmInv || gamma0KmInv > MaxGammaKmInv)
            throw new ArgumentOutOfRangeException(nameof(gamma0KmInv),
                $"γ₀ {gamma0KmInv:E3} km⁻¹ outside [{MinGammaKmInv:E0}, {MaxGammaKmInv:E0}] (stage=DensityModulatedDrag).");
        if (nRef <= 0.0 || nRef > 100.0)
            throw new ArgumentOutOfRangeException(nameof(nRef),
                $"n_ref {nRef} cm⁻³ outside (0, 100] (stage=DensityModulatedDrag).");
        _gamma0 = gamma0KmInv;
        _nRef = nRef;
        _stream = stream ?? throw new ArgumentNullException(nameof(stream));
    }

    public GammaStep GammaAtHour(int hourIndex, double speedKmPerSec, double rSolarRadii)
    {
        if (!_stream.TryGetAtHour(hourIndex, out var obs))
            return new GammaStep(GammaKmInv: _gamma0, NObs: null, VObs: null, FellBack: true);

        if (!obs.ProtonDensity.HasValue)
            return new GammaStep(GammaKmInv: _gamma0, NObs: null, VObs: obs.FlowSpeed, FellBack: true);

        double scaled = _gamma0 * (obs.ProtonDensity.Value / _nRef);
        double clipped = scaled < MinGammaKmInv ? MinGammaKmInv
                       : (scaled > MaxGammaKmInv ? MaxGammaKmInv : scaled);
        return new GammaStep(GammaKmInv: clipped, NObs: obs.ProtonDensity, VObs: obs.FlowSpeed, FellBack: false);
    }
}

// Phase 9 §3.3 step 4: shock detector backed by L1ObservationStream's
// hour-over-hour Δv + density-ratio rule. Separated from the stream so
// ProgressiveDragPropagator can remain data-source-agnostic.
public sealed class L1ShockDetector : IShockDetector
{
    private readonly L1ObservationStream _stream;

    public L1ShockDetector(L1ObservationStream stream)
    {
        _stream = stream ?? throw new ArgumentNullException(nameof(stream));
    }

    public bool ShockDetectedAtHour(int hourIndex) => _stream.ShockDetectedAtHour(hourIndex);
}
