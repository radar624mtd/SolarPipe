---
name: physics-validator
description: Validate physics equation implementations for correctness and numerical stability. Use when editing DragBasedModel, DormandPrinceSolver, BurtonOde, NewellCoupling, CoordinateTransform, ProgressiveDragPropagator, DensityModulatedDrag, L1ObservationStream, or any file in SolarPipe.Training/Physics/.
---

Review the code changes for the following violations. Report each issue with the file, line reference, and a concrete fix.

## Coordinate Frame Rules
- Bz values must use GSM frame, not GSE. Any Bz-dependent calculation must be preceded by `CoordinateTransform.GseToGsm()`. Flag any `Bz` variable sourced directly from raw data without coordinate conversion.
- Spatial vectors must use `GseVector` or `GsmVector` typed structs. Bare `float[]`, `double[]`, or unnamed tuples for coordinate data are violations.

## ODE Solver Rules
- ODE integration must use `DormandPrinceSolver` (Dormand-Prince RK4(5)). References to MathNet ODE solvers (`RungeKutta`, `OdeIntegrator`, etc.) are prohibited.
- Step size adaptation must respect the solver's error tolerance parameters — hardcoded fixed steps defeat the adaptive scheme.

## Density-Modulated Drag Rules (Phase 9)
- `γ_eff(t)` must be computed as `γ₀ · (n_obs(t) / n_ref)` with `n_ref = 5 cm⁻³` (physical climatology constant, not tunable).
- `γ_eff` must be hard-clipped to `[1e-9, 1e-6] km⁻¹` at every timestep. Out-of-range γ causes ODE divergence.
- When `proton_density` is NULL in `omni_hourly`, the propagator must fall back to `γ_eff = γ₀` (not zero, not NaN, not skipping the step).
- If `proton_density` is NULL for ≥ 20% of in-transit hours, the event must be flagged as `fallback=true` and the Phase 8 static prediction must be used instead. The propagator must never silently return a degraded result without setting this flag.
- Shock-arrival detection: `v_obs` jump ≥ 200 km/s AND density jump ≥ 3× within the same hour. Both conditions required — flagging on speed alone is a violation.
- `DormandPrinceSolver` instances must not be shared across parallel hypothesis runs or across progressive steps. Construct a fresh solver per event.

## Test Data Rules
- Physics test data must come from `PhysicsTestFixtures.cs`. Any use of `Random.NextDouble()`, `new Random()`, or hardcoded magic numbers for speeds, densities, or field values in test files is a violation.
- Sentinel values (9999.9, -1e31) must be converted to `NaN` at data load time, not filtered mid-computation.
- Integration test for `ProgressiveDragPropagator` must verify Jan-18 X1.9 prediction within ±0.5h of 25.00h (Python reference). This is the parity gate.

## Error Reporting Rules
- All exceptions thrown from physics stages must include: stage name, input dimensions, and parameter values in the message. A bare `throw new Exception("failed")` is not acceptable.

## Numerical Stability
- Flag any division where the denominator could be zero without a guard.
- Flag any `Math.Sqrt` or `Math.Log` call where the argument could be negative or zero.
- Flag any accumulator that could produce NaN through unchecked arithmetic on sentinel-converted values.
- In the density-modulated drag loop: if `γ_eff · dt · |v - w|` exceeds `|v - w|` (ODE would reverse velocity), log a warning and clamp the step. Do not allow the ODE to produce negative relative velocity.

## Phase 7 Physics-in-Sweep Rules
- When a physics stage (DragBased, BurtonOde, NewellCoupling) is used inside a hypothesis sweep, it must still validate GSM-frame Bz before each ODE call — the sweep runner does not add coordinate validation.
- `DormandPrinceSolver` instances must not be shared across parallel hypothesis runs. Each hypothesis scope must construct its own solver instance.
- If a physics stage returns NaN for a CV fold event, the event must be excluded from that fold's metrics with a logged warning `[sweep:HN stage:X] nan_event cme_id=Y`. It must not propagate as a valid prediction.

## Neural Physics Integration
- When the TFT or any neural model produces a transit time prediction that will be combined with a physics ODE baseline, the physics residual (truth - ODE) must be used as the training target, not the raw transit time. This prevents the neural model from re-learning the physics prior.
- Phase 9 `γ_eff` trajectory (hourly values) may be passed as a scalar summary feature (mean, std) to the neural ensemble head, but the full trajectory must not be passed as-is — it is event-length variable and would require padding.
