---
name: physics-validator
description: Validate physics equation implementations for correctness and numerical stability. Use when editing DragBasedModel, DormandPrinceSolver, BurtonOde, NewellCoupling, CoordinateTransform, or any file in SolarPipe.Training/Physics/.
---

Review the code changes for the following violations. Report each issue with the file, line reference, and a concrete fix.

## Coordinate Frame Rules
- Bz values must use GSM frame, not GSE. Any Bz-dependent calculation must be preceded by `CoordinateTransform.GseToGsm()`. Flag any `Bz` variable sourced directly from raw data without coordinate conversion.
- Spatial vectors must use `GseVector` or `GsmVector` typed structs. Bare `float[]`, `double[]`, or unnamed tuples for coordinate data are violations.

## ODE Solver Rules
- ODE integration must use `DormandPrinceSolver` (Dormand-Prince RK4(5)). References to MathNet ODE solvers (`RungeKutta`, `OdeIntegrator`, etc.) are prohibited.
- Step size adaptation must respect the solver's error tolerance parameters — hardcoded fixed steps defeat the adaptive scheme.

## Test Data Rules
- Physics test data must come from `PhysicsTestFixtures.cs`. Any use of `Random.NextDouble()`, `new Random()`, or hardcoded magic numbers for speeds, densities, or field values in test files is a violation.
- Sentinel values (9999.9, -1e31) must be converted to `NaN` at data load time, not filtered mid-computation.

## Error Reporting Rules
- All exceptions thrown from physics stages must include: stage name, input dimensions, and parameter values in the message. A bare `throw new Exception("failed")` is not acceptable.

## Numerical Stability
- Flag any division where the denominator could be zero without a guard.
- Flag any `Math.Sqrt` or `Math.Log` call where the argument could be negative or zero.
- Flag any accumulator that could produce NaN through unchecked arithmetic on sentinel-converted values.

## Phase 7 Physics-in-Sweep Rules
- When a physics stage (DragBased, BurtonOde, NewellCoupling) is used inside a hypothesis sweep, it must still validate GSM-frame Bz before each ODE call — the sweep runner does not add coordinate validation.
- `DormandPrinceSolver` instances must not be shared across parallel hypothesis runs. Each hypothesis scope must construct its own solver instance.
- If a physics stage returns NaN for a CV fold event, the event must be excluded from that fold's metrics with a logged warning `[sweep:HN stage:X] nan_event cme_id=Y`. It must not propagate as a valid prediction.
