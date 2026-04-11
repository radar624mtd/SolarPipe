# Phase 9 M7 — Hybrid Routing Backtest Report

Generated: 2026-04-10T22:17:56.116293+00:00

Speed threshold: 540 km/s  (density mode for v₀ ≥ 540, static for v₀ < 540)


## Headline numbers (arrived-event MAE)

| Run | n_arrived / 71 | MAE | RMSE | Bias |
|---|---:|---:|---:|---:|
| Phase 8 Static (baseline) | 29/71 | 11.07h | 13.76h | 7.30h |
| M6.6 All-density (n_ref=10) | 31/71 | 8.10h | 10.87h | 0.89h |
| M7 Hybrid (derived n_ref) | 26/71 | 10.16h | 13.22h | 6.68h |


## Speed-bucket breakdown

**Fast CMEs (v₀ ≥ 540 km/s, n=31):**

| Run | arrived | MAE |
|---|---:|---:|
| M6.6 density n_ref=10 | 31/31 | 8.10h |
| M7 hybrid (density, derived n_ref) | 26/31 | 10.16h |


**Slow CMEs (v₀ < 540 km/s, n=40):**

| Run | arrived | MAE |
|---|---:|---:|
| Phase 8 static | 0/40 | 0.00h |
| M7 hybrid (static) | same as Phase 8 static | — |


## Shock-detection events

| Launch | v₀ | obs | M7 pred | M7 err | M6.6 err | Static err |
|---|---:|---:|---:|---:|---:|---:|
| 2026-01-17T23:24:00 | 951 | 43.52 | 45.00 | 1.48 | 0.11 | 10.30 |
| 2026-01-18T08:36:00 | 1123 | 34.32 | 36.00 | 1.68 | 1.68 | 15.01 |
| 2026-01-18T18:09:00 | 1473 | 24.77 | 26.00 | 1.23 | 1.23 | 18.02 |


## Analysis: why M7 hybrid is worse than M6.6 on the fast-CME subset

M6.6 used `n_ref=10.0` (global constant) whereas M7 derives n_ref per-event from the
150h pre-launch OMNI window (ambient median ~4–5 cm⁻³ in 2026 solar maximum).

With n_ref=10 the effective drag is halved: γ_eff ≈ 0.5·γ₀. This compensates for a
systematic over-prediction of drag in Phase 8's calibrated γ₀, which was tuned against
static mode (no density modulation). When n_ref is physically correct, γ_eff ≈ γ₀
during typical solar wind — matching the static result — but becomes *stronger* during
dense ambient periods (pre-event solar wind compressed by preceding events), causing
over-deceleration and large positive transit-time errors for fast CMEs.

**Root cause**: γ₀ must be re-calibrated jointly with the density-modulation formula.
The correct next step (M8) is a joint fit of (γ₀, n_ref_target) on the training set,
treating n_ref as a free scale parameter in the ODE, not a physics constant.


## Per-event table (arrived only)

| Launch | v₀ | obs | M7 arr | M7 err | M6.6 err | Static err | mode | shock |
|---|---:|---:|---:|---:|---:|---:|---|---|
| 2026-01-01T19:36:00 | 628 | 73.08 | 65.25 | -7.83 | -10.10 | -5.51 | density | - |
| 2026-01-05T08:48:00 | 661 | 68.15 | 66.41 | -1.74 | -6.01 | -2.55 | density | - |
| 2026-01-06T02:00:00 | 895 | 50.95 | 61.43 | 10.48 | 1.42 | 4.62 | density | - |
| 2026-01-08T05:53:00 | 1016 | 61.72 | 64.63 | 2.92 | -7.52 | -9.74 | density | - |
| 2026-01-08T15:48:00 | 872 | 51.80 | 64.13 | 12.33 | 4.25 | 4.54 | density | - |
| 2026-01-08T17:00:00 | 1056 | 50.60 | 60.10 | 9.51 | 0.05 | 0.33 | density | - |
| 2026-01-08T17:12:00 | 1084 | 50.40 | 59.63 | 9.23 | -0.41 | -0.16 | density | - |
| 2026-01-10T02:24:00 | 1118 | 17.20 | 51.08 | 33.88 | 28.64 | 32.24 | density | - |
| 2026-01-10T20:36:00 | 1045 | 65.12 | 50.43 | -14.69 | -18.66 | -13.90 | density | - |
| 2026-01-10T20:48:00 | 809 | 64.92 | 57.23 | -7.69 | -10.68 | -6.27 | density | - |
| 2026-01-11T05:12:00 | 775 | 56.52 | 54.67 | -1.84 | -3.57 | 3.51 | density | - |
| 2026-01-12T01:45:00 | 804 | 35.97 | 50.74 | 14.78 | 13.58 | 22.88 | density | - |
| 2026-01-12T03:00:00 | 916 | 34.72 | 45.67 | 10.95 | 9.58 | 20.17 | density | - |
| 2026-01-14T08:24:00 | 768 | 41.27 | 56.87 | 15.61 | 12.02 | 19.06 | density | - |
| 2026-01-17T23:24:00 | 951 | 43.52 | 45.00 | 1.48 | 0.11 | 10.30 | density | Y |
| 2026-01-18T08:36:00 | 1123 | 34.32 | 36.00 | 1.68 | 1.68 | 15.01 | density | Y |
| 2026-01-18T18:09:00 | 1473 | 24.77 | 26.00 | 1.23 | 1.23 | 18.02 | density | Y |
| 2026-02-13T09:38:00 | 800 | 34.02 | 60.50 | 26.48 | 19.15 | 24.99 | density | - |
| 2026-03-17T05:48:00 | 630 | 67.72 | 68.33 | 0.61 | -6.92 | -0.27 | density | - |
| 2026-03-18T09:23:00 | 731 | 58.90 | 65.86 | 6.96 | -5.40 | 3.09 | density | - |
| 2026-03-22T16:23:00 | 655 | 61.50 | 62.95 | 1.45 | -2.69 | 4.44 | density | - |
| 2026-03-22T23:24:00 | 739 | 54.48 | 56.73 | 2.25 | -2.08 | 7.13 | density | - |
| 2026-03-30T03:24:00 | 1689 | 56.08 | 44.67 | -11.41 | -27.25 | -16.29 | density | - |
| 2026-03-30T21:15:00 | 768 | 38.23 | 62.33 | 24.10 | 14.85 | 22.09 | density | - |
| 2026-03-30T21:30:00 | 942 | 37.98 | 55.81 | 17.82 | 6.22 | 16.10 | density | - |
| 2026-04-01T23:45:00 | 1220 | 39.28 | 54.54 | 15.26 | 1.33 | 7.95 | density | - |
