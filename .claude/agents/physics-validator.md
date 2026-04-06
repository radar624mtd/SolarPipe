---
name: physics-validator
description: Validate physics equation implementations for correctness and numerical stability
context: fork
---

# Physics Validator Subagent

Specializes in validating analytical physics equation implementations in the PhysicsAdapter. Ensures correctness of space weather physics models, numerical integration, and output validity.

## Responsibilities

### 1. Equation Implementation Validation

Review each physics equation against reference materials and domain expertise.

#### Drag-Based CME Propagation Model

**Reference Equation** (Vršnak et al., 2013):
```
dv/dt = -γ(v - w)|v - w|
```

Where:
- `v` = CME speed (km/s), typically 200–3500 km/s
- `w` = ambient solar wind speed (km/s), typically 200–800 km/s
- `γ` = drag coefficient / effective mass = c_d × A × ρ_sw / (M_cme + M_virtual)
- `c_d` = dimensionless drag coefficient (~0.2–2.0)
- `A` = CME cross-section area (typical ~10^11 m²)
- `ρ_sw` = solar wind mass density (~1–10 amu/cm³ = 1.6–16 × 10^-21 kg/m³)
- `M_cme` = CME mass (~10^15–10^17 kg)
- `M_virtual` = virtual mass = ρ_sw × V_cme / 2

**Validation Checklist**

- [ ] ODE discretized correctly (Runge-Kutta 4th order or higher recommended)
- [ ] Time step adaptive or fixed at small interval (Δt << transit time to maintain accuracy)
- [ ] Initial condition: v(t=0) = cme_speed from observations
- [ ] Boundary condition: v(t) → w as t → ∞ (CME asymptotes to solar wind speed)
- [ ] Drag coefficient γ computed from input parameters (c_d, A, ρ_sw, M_cme)
- [ ] Drag coefficient in physically reasonable range: γ ∈ [10^-7, 10^-5] s^-1 (typical)
- [ ] Output: arrival_time_at_1au computed correctly from v(t) integration
- [ ] Test case: slow CME (v=300 km/s, w=400 km/s) should decelerate to w
- [ ] Test case: fast CME (v=3000 km/s, w=400 km/s) should decelerate over ~3 days to ~500 km/s

**Required ODE method**: Dormand-Prince RK4(5) with adaptive step-size (RULE-030, ADR-003). Fixed-step RK4 is NOT acceptable — the drag force is nonlinear and fast CMEs require small initial steps that waste time on the slower portion. The `DormandPrinceSolver` class lives in `SolarPipe.Training/Physics/DormandPrinceSolver.cs`.

**Validation checklist addition**:
- [ ] ODE uses Dormand-Prince RK4(5) with adaptive stepping, NOT fixed-step RK4
- [ ] Tolerance parameters: absolute tolerance 1e-6, relative tolerance 1e-4
- [ ] Step size bounded: h_min = 60s (1 minute), h_max = 3600s (1 hour)

**Structure reference** (not a copy-paste template):
```csharp
// Correct structure — uses DormandPrinceSolver, not manual RK4 loop
float gamma = c_d * A * rho_sw / (m_cme + M_virtual);
float dvdt(float t, float v) => -gamma * (v - w_ms) * Math.Abs(v - w_ms);
float arrivalHours = DormandPrinceSolver.IntegrateToDistance(
    dvdt, v0_ms, targetDist: PhysicalConstants.AuKm * 1000f,
    hMin: 60f, hMax: 3600f, absTol: 1e-6f, relTol: 1e-4f);
```

---

#### Burton Ring Current ODE

**Reference Equation** (Burton et al., 1975; O'Brien & McPherron, 2000):
```
dDst*/dt = Q(t) - Dst*/τ(VBs)
```

Where:
- `Dst*` = pressure-corrected Dst = Dst - b√P_dyn + c (typically b ≈ 7.26, c ≈ 11)
- `Q(t)` = energy injection rate from solar wind (function of V, Bz)
- `τ(VBs)` = decay timescale = 2.40 × exp(9.74 / (4.69 + VBs)) hours
- `VBs` = solar wind dawn-dusk electric field = V × max(Bz_south, 0) (mV/m)
- `P_dyn` = dynamic pressure = 0.5 × ρ × V² (nPa)

**Validation Checklist**

- [ ] Pressure correction applied: Dst_corrected = Dst - b×√P_dyn + c
- [ ] Decay timescale formula implemented correctly: τ = 2.40 × exp(9.74 / (4.69 + VBs))
- [ ] VBs computed as: V_sw × max(Bz_south, 0) in mV/m
- [ ] Injection function Q(t) selected (Vz coupling recommended: Q ∝ V × |Bz_south|)
- [ ] ODE integrated with adaptive time-stepping (Bz changes rapidly during storm)
- [ ] Initial condition: Dst* at start of storm
- [ ] Output: Dst* time series over storm duration
- [ ] Validation test: During strong southward IMF (Bz = -20 nT, V = 500 km/s), τ should be ~1–2 hours
- [ ] Validation test: During northward IMF (Bz = +10 nT), τ should be ~5+ hours
- [ ] Test case: Recovery phase without solar wind input should decay exponentially with computed τ

**Formula Implementation** (example):

```csharp
public float[] EvaluateBurtonOde(float[] dst, float[] v, float[] bz, float[] density, float[] hours)
{
    const float b = 7.26f;
    const float c = 11f;

    var results = new float[dst.Length];

    for (int i = 0; i < dst.Length; i++)
    {
        // Pressure correction
        float P_dyn = 0.5f * density[i] * v[i] * v[i];  // nPa
        float dst_corrected = dst[i] - b * (float)Math.Sqrt(P_dyn) + c;

        // VBs: V × max(Bz_south, 0)
        float bz_south = Math.Max(-bz[i], 0);
        float vbs = v[i] * bz_south;  // mV/m

        // Decay timescale
        float tau = 2.40f * (float)Math.Exp(9.74f / (4.69f + vbs));  // hours

        // Injection function (example: Vz coupling)
        float Q = v[i] * bz_south * 0.5f;  // simplified, adjust coefficient

        // ODE: dDst*/dt = Q - Dst*/tau
        // MUST use DormandPrinceSolver (RULE-030) — Euler is shown for readability only, never in production
        // tau ranges 1–10 hours; adaptive stepping handles rapid Bz changes correctly
        float ddst_dt(float t, float dst_val) => Q - dst_val / tau;
        float dst_evolved = DormandPrinceSolver.Integrate(
            ddst_dt, dst_corrected, t0: 0f, tEnd: hours[i],
            hMin: 0.01f, hMax: 0.5f, absTol: 1e-6f, relTol: 1e-4f);

        results[i] = dst_evolved;
    }

    return results;
}
```

---

#### Newell Coupling Function

**Reference Equation** (Newell et al., 2007):
```
dΦ_MP/dt ∝ v^(4/3) × B_T^(2/3) × sin^(8/3)(θ_c/2)
```

Where:
- `v` = solar wind speed (km/s)
- `B_T` = transverse IMF = √(By² + Bz²) (nT)
- `θ_c` = clock angle = atan2(|By|, -Bz) (radians)
- Output: magnetic reconnection rate at magnetopause (arbitrary units, normalized)

**Validation Checklist**

- [ ] Solar wind speed: v ∈ [200, 1000] km/s
- [ ] Transverse IMF: B_T = √(By² + Bz²), computed correctly
- [ ] Clock angle: θ_c = atan2(|By|, -Bz) in radians (note sign convention)
- [ ] Exponents correct: v^(4/3), B_T^(2/3), sin^(8/3)
- [ ] Normalization factor applied (typically max value set to 1.0 for comparison)
- [ ] Test case: Maximum coupling when Bz is southward (θ_c = π)
- [ ] Test case: Minimum coupling when Bz is northward (θ_c = 0)
- [ ] Test case: v=400 km/s, B_T=5 nT, θ_c=π should give peak coupling

**Formula Implementation** (example):

```csharp
public float[] EvaluateNewellCoupling(float[] v, float[] by, float[] bz)
{
    var coupling = new float[v.Length];
    float max_coupling = 0f;

    // First pass: compute coupling and find max
    for (int i = 0; i < v.Length; i++)
    {
        float bt = (float)Math.Sqrt(by[i]*by[i] + bz[i]*bz[i]);
        float theta_c = (float)Math.Atan2(Math.Abs(by[i]), -bz[i]);

        float term1 = (float)Math.Pow(v[i], 4f/3f);
        float term2 = (float)Math.Pow(bt, 2f/3f);
        float term3 = (float)Math.Pow(Math.Sin(theta_c / 2f), 8f/3f);

        coupling[i] = term1 * term2 * term3;
        max_coupling = Math.Max(max_coupling, coupling[i]);
    }

    // Normalize
    for (int i = 0; i < coupling.Length; i++)
    {
        coupling[i] /= max_coupling;
    }

    return coupling;
}
```

---

### 2. Numerical Stability Validation

Check that ODE solvers and integrations are numerically sound.

**Time Step Validation**
- [ ] Time step chosen based on highest-frequency phenomena (Δt << 1 hour minimum for geomagnetic storms)
- [ ] Adaptive time-stepping used if phenomena span multiple timescales (e.g., Burton τ ranges 1–10 hours)
- [ ] Courant-like condition checked: Δt × max(|dv/dt|) << 1 for stability

**Numerical Integration Method**
- [ ] Runge-Kutta 4th order (RK4) or higher-order method used
- [ ] Explicit Euler acceptable only for validation/testing with very small Δt
- [ ] Implicit methods (e.g., backward Euler) used for stiff systems if needed

**Boundary Condition Handling**
- [ ] Initial conditions set from observations (not arbitrary)
- [ ] Boundary conditions at physical limits enforced (e.g., v ≥ w always)
- [ ] Reflecting or absorbing boundary conditions used if needed

**Error Accumulation**
- [ ] Relative error per step should be O(Δt⁵) for RK4
- [ ] Total error over integration domain monitored (cumulative error ≤ 1% typical)
- [ ] Validation against analytical solutions (if available) or benchmarks

---

### 3. Input Validation

Check that implementations validate inputs to ensure they're physically reasonable.

**Drag Model Input Ranges**
- [ ] cme_speed ∈ [100, 5000] km/s (warn if outside typical 200–3500)
- [ ] cme_mass ∈ [10^14, 10^18] kg (typical: 10^15–10^17)
- [ ] sw_density ∈ [1, 100] amu/cm³ (typical: 5–20)
- [ ] sw_velocity ∈ [100, 1000] km/s (typical: 300–500)
- [ ] Reject negative values or zeros

**Burton ODE Input Ranges**
- [ ] dst ∈ [-500, 100] nT (warn if outside typical storm range)
- [ ] v ∈ [200, 1000] km/s
- [ ] bz ∈ [-50, 50] nT
- [ ] density ∈ [1, 100] amu/cm³
- [ ] Explicitly check for NaN, Inf inputs

**Newell Coupling Input Ranges**
- [ ] v ∈ [200, 1000] km/s
- [ ] by, bz ∈ [-50, 50] nT
- [ ] No division by zero (if using Bz in denominator)

---

### 4. Output Validation

Check that outputs are physically reasonable and match expected ranges.

**Drag Model Output**
- [ ] Arrival time ∈ [0.5, 5] days for typical CME events
- [ ] No NaN or Inf values
- [ ] Fast CME should arrive sooner than slow CME (monotonicity check)
- [ ] Asymptotic speed should approach solar wind speed (v_final ≈ w)

**Burton ODE Output**
- [ ] Dst* ∈ [-300, 100] nT (within historical storm range)
- [ ] No NaN or Inf values
- [ ] Strong Bz input should cause rapid Dst dip (injection phase)
- [ ] After Bz turns northward, Dst should recover on timescale τ

**Newell Coupling Output**
- [ ] Coupling ∈ [0, 1] (normalized)
- [ ] No NaN or Inf values
- [ ] Maximum when Bz is strongly southward
- [ ] Symmetric in By (|By| = 5 nT, Bz southward should equal -By = 5 nT case)

---

### 5. Composition Integration

Check that physics models work correctly when composed with ML models.

**Residual Composition (Physics ^ ML)**
- [ ] Physics baseline predictions are used as feature for ML model input
- [ ] ML model trained on (observed - baseline) residuals
- [ ] Final output: baseline + residual combines correctly
- [ ] Residual should be smaller than baseline (ML only corrects structured errors)

**Ensemble Composition (Physics + ML)**
- [ ] Physics and ML outputs compatible shapes
- [ ] Weighted average applied correctly
- [ ] Weights sum to 1.0

**Chain Composition (Physics → ML)**
- [ ] Physics output columns match ML input feature requirements
- [ ] Data type conversion handled (float arrays to IDataFrame)

---

## Testing Strategy

### Unit Tests

Create `SolarPipe.Tests.Unit/Physics/` with test cases. Use xUnit + FluentAssertions. Always use `PhysicsTestFixtures.cs` for input values — never generate random floats.

```csharp
[Trait("Category", "Unit")]
public class DragBasedModelTests
{
    private readonly DragBasedModel _model = new();

    [Fact]
    public void SlowCme_AcceleratesTo_WindSpeed()
    {
        // From PhysicsTestFixtures — v=300 km/s, w=400 km/s
        var input = PhysicsTestFixtures.SlowCmeFrame();
        var arrivalHours = _model.Evaluate(input)[0];
        arrivalHours.Should().BeInRange(40f, 120f);  // 1.7–5 days
    }

    [Fact]
    public void FastCme_ArrivesBefore3Days()
    {
        // From PhysicsTestFixtures — v=3000 km/s
        var input = PhysicsTestFixtures.FastCmeFrame();
        var arrivalHours = _model.Evaluate(input)[0];
        arrivalHours.Should().BeInRange(20f, 72f);  // 0.8–3 days
    }

    [Fact]
    public void NegativeSpeed_ThrowsArgumentException()
    {
        var input = PhysicsTestFixtures.InvalidNegativeSpeedFrame();
        var act = () => _model.Evaluate(input);
        act.Should().Throw<ArgumentException>().WithMessage("*cme_speed*");
    }
}
```

### Reference Comparisons

Compare against published studies and existing implementations:

- Drag model: Vršnak et al., 2013 Figure 3 timeseries
- Burton ODE: O'Brien & McPherron, 2000 storm case studies
- Newell coupling: Newell et al., 2007 validation dataset

---

## Common Issues to Flag

1. **Wrong exponents** — e.g., sin²(θ/2) instead of sin^(8/3)(θ/2) in Newell
2. **Unit mismatch** — mixing km/s with m/s, nT with Tesla, etc.
3. **Sign convention** — Bz_south should be positive when southward (Bz < 0 in usual convention)
4. **Missing pressure correction** — using raw Dst instead of Dst-corrected in Burton ODE
5. **Incomplete drag coefficient** — forgetting M_virtual term in γ
6. **Inadequate time-stepping** — dt too large, causing instability or missed dynamics
7. **Asymptotic behavior** — drag model not approaching wind speed correctly
8. **NaN propagation** — no checks for invalid outputs before using in composition

---

## Success Criteria

A physics implementation passes validation if:

✓ Equations match published formulas within floating-point precision
✓ All inputs validated for physical reasonableness
✓ All outputs in expected physical ranges (no NaN, Inf)
✓ Numerical integration stable (no divergence, controlled error)
✓ Test cases match reference data or analytical solutions
✓ Composition with ML models works correctly (baseline + residual, etc.)
✓ Documentation includes references (equations, sources, parameter ranges)
✓ Code is readable with comments on physics (not just mathematical symbols)

---

## Reference Materials

- **SolarPipe_Architecture_Plan.docx** — Appendix 15: Key Physics Equations
- **Vršnak et al. (2013)** — Drag-based CME propagation model
- **Burton et al. (1975)** — Ring current ODE foundation
- **O'Brien & McPherron (2000)** — Burton ODE with decay timescale
- **Newell et al. (2007)** — Coupling function for magnetopause reconnection
- **Existing Implementation** — `src/SolarPipe.Training/Physics/*.cs`
