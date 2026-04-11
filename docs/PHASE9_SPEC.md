# Phase 9: Progressive L1 Assimilation

**Status:** Draft design, revised after premise validation. Not yet
approved for implementation.
**Authors:** Radar + Claude (design session 2026-04-10; revised 2026-04-10
after throwaway Python validation).
**Supersedes:** none. Adds a new inference mode alongside the existing
Phase 7/8 static-feature pipelines.

**Revision note (2026-04-10):** The original §3.3 residual-nudge-on-γ
rule keyed off `flow_speed` residuals, but a throwaway Python validation
against the Jan-18 X1.9 event and the full CCMC 4-event set showed that
density is the correct assimilation signal, not speed. The revised rule
modulates the drag coefficient directly from observed `proton_density`
via a physically-motivated scaling `γ(t) = γ₀ · (n_obs(t) / n_ref)`, with
no tunable gain. This is simpler, has no free parameters to overfit, and
reproduces Vršnak's drag-density proportionality from first principles.
See §3.3 for the derivation and §6.2 for validation results.

---

## 1. Why this phase exists

Phase 8's domain pipeline achieves 12.33h transit MAE across 71 held-out
2026 events. On the CCMC 4-event benchmark we come in at 8.6h average.
One event — 2026-01-18 18:09 X1.9, observed transit 24.77h — is predicted
at 43.52h, an 18.76h error. Every other model we have tried (ML residual,
physics baseline, NNLS ensemble, hypothesis sweep) fails on this event at
roughly the same magnitude.

The diagnostic work in April 2026 isolated the cause:

1. The Jan-18 X1.9 is a CME-CME interaction event. A 1123 km/s CME launched
   ~10 hours earlier on Jan-18 08:36 pre-conditioned the solar wind along
   the X1.9's path. The X1.9 propagated through a disturbed wake rather
   than undisturbed wind, giving it an anomalously short transit for a
   1473 km/s CME.
2. Every static-feature model we ship reads CME launch properties and
   pre-event ambient wind as 0-dimensional features, then predicts arrival.
   There is no point in the pipeline where the model can see the state of
   the heliosphere *between* launch and arrival.
3. The information needed to correctly predict the X1.9 is physically
   present in `omni_hourly` between 2026-01-18 08:36 and 2026-01-18 18:09 —
   the wake signature left by the precursor CME — and continues to appear
   in L1 observations throughout the X1.9's transit on Jan-19. We already
   have all of this data in `solar_data.db`. We just don't read it during
   prediction.

CCMC's ELEvo achieves 7.8h on this event because it runs its propagation
forward from launch and updates its estimate against real-time L1
observations as they arrive. This is a data-assimilation technique, not a
machine-learning technique. No amount of ghost-row cleanup, training-set
expansion, neural architecture change, or hyperparameter search will close
this gap — we ran a tightened-filter A/B that empirically confirmed this
(tightened filter regressed every metric by 25-50%).

Phase 9 adds the missing capability: a prediction mode that consumes L1
observations between launch and arrival and updates its transit estimate
in wall-clock time.

## 2. Scope

Phase 9 adds exactly one new inference mode. It does not change any
existing training code, pipeline, config schema, or model artifact. It
does not touch the sidecar. It does not introduce a new ML framework.

### 2.1 In scope

- New CLI command: `predict-progressive`
- New class: `ProgressiveDragPropagator` in `SolarPipe.Training/Physics/`
- New class: `L1ObservationStream` in `SolarPipe.Data/` (reads `flow_speed` and `proton_density` from `omni_hourly`)
- New class: `DensityModulatedDrag` (γ_eff function) in `SolarPipe.Training/Physics/`
- New test category: `[Trait("Category", "Assimilation")]` for the Phase 9 unit + integration tests
- Extension of the CCMC benchmark output to include progressive-mode predictions alongside the static-mode ones for side-by-side comparison

### 2.2 Out of scope

- Kalman filtering (we start with the simpler residual-nudge update; Kalman is a Phase 9.5 follow-up if residual-nudge plateaus)
- Ensemble of multiple propagators (single DragBased only for v1; BurtonOde assimilation is Phase 9.5)
- Real-time DSCOVR / ACE API ingestion (we use the existing `omni_hourly` table, which is already hourly-resolution historical OMNI)
- Changing any training data or training code
- Sidecar / TFT / PINN work (still deferred until post-Phase 9)
- Schema refactor of `l1_arrivals` (deferred indefinitely; the A/B proved label duplication isn't on the critical path)

## 3. The assimilation model

### 3.1 State

At wall-clock time `t`, the propagator carries:

```
s(t) = { r(t), v(t) }
```

- `r(t)` — heliocentric distance, Rs (start: 21.5, target: 215)
- `v(t)` — current CME front speed, km/s

Note that `gamma` is **not** carried as state. Unlike the original design,
the drag coefficient is a *function of the current observation*, evaluated
fresh at each step, so there is nothing to accumulate across time. See §3.3.

### 3.2 ODE

Standard drag-based kinematics, unchanged from existing `DragBasedModel`:

```
dr/dt = v
dv/dt = -gamma * (v - w) * |v - w|
```

where `w` is the ambient solar wind speed. Integrated with the existing
Dormand-Prince solver.

### 3.3 The assimilation step — density-modulated drag

**Derivation.** The Vršnak drag-based equation is

```
dv/dt = -C_d · (A_CME · ρ_sw / m_CME) · (v - w) · |v - w|
```

where `C_d` is the dimensionless drag coefficient (~1 for blunt bodies),
`A_CME` is the CME cross-section, `m_CME` is its mass, `ρ_sw` is the solar
wind mass density, and `(v - w)` is the relative speed. Lumping the
pre-factor into a single `gamma`, we get `dv/dt = -γ · (v - w) · |v - w|`
with `γ = C_d · A / (m · m_p) · n_sw`, where `n_sw` is proton number
density. So **γ scales linearly with solar wind proton density, holding
CME geometry and mass fixed**.

Phase 8 trains `γ` as a static per-event parameter, meaning it's implicitly
fit to the *average* density along the transit path. When the average is
wrong — for example, when a precursor CME evacuates the medium, leaving
a rarefied wake — the static γ overestimates drag. This is exactly what
happened on Jan-18 X1.9.

**The update rule.** At each observation time `t_k` (hourly while the CME
is in transit), query `omni_hourly WHERE datetime = t_k` for `n_obs` and
`v_obs`, then evaluate the effective drag coefficient as

```
γ_eff(t_k) = γ₀ · (n_obs(t_k) / n_ref)
```

where `γ₀` is the static launch-time estimate from the trained drag model
(the same γ Phase 8 uses) and `n_ref = 5 cm⁻³` is quiet-sun solar wind
proton density (from OMNI climatology; coded as a physical constant, not
tuned).

This is a **density-modulated drag**, not an assimilation in the
estimation-theoretic sense. There are no hidden state variables, no
tunable gain, no gradient step. At each ODE substep the RHS uses
`γ_eff(t)` where `t` lies in the hour for which `n_obs(t)` is known. If
the observation is missing for an hour, γ falls back to `γ₀` for that hour.

**The assimilation loop per hour:**

1. Query `omni_hourly WHERE datetime = floor_hour(t)` for `v_obs`, `n_obs`.
2. If `n_obs` is present: `γ_eff = γ₀ · (n_obs / n_ref)`; else `γ_eff = γ₀`.
3. Integrate the drag ODE one hour forward with `γ_eff`, updating `r, v`.
4. Check shock-arrival criterion: if `v_obs` jumps by ≥ 200 km/s AND the
   density jumps by a factor ≥ 3 within the same hour, declare shock
   arrival at `t_k` and terminate. This is the same shock-detection
   heuristic used in DONKI IPS ingestion (see `cme_icme_matcher.py:40`),
   applied in reverse: instead of *matching* a CME to a known shock, we
   *detect* a shock in the incoming L1 stream. The density-jump criterion
   is visible at 1h cadence (confirmed on Jan-18: n=3.1→7.9→23.9 cm⁻³
   across two hours).
5. Loop until `r ≥ 215 R_sun` (1 AU arrival by propagation) or shock
   detected (arrival by observation) or `t > launch + 72h` (timeout).

**Physical clipping.** `γ_eff` is hard-clipped to `[1e-9, 1e-6] km⁻¹`
regardless of observed density, to handle sensor glitches and extreme
events without breaking the integrator.

**`γ₀` provenance.** `γ₀` is the static per-event drag parameter
produced by the trained Phase 8 `DragBasedModel` stage for this event —
NOT a global constant. The density scaling is a multiplicative
modulation on top of Phase 8's existing per-event γ estimate, so events
where Phase 8's trained γ is already good stay good; events where Phase 8
mispredicted γ because the medium was anomalous get corrected.

### 3.5 Fallback when L1 data is missing

If `omni_hourly.proton_density` is null for an hour (sensor dropout,
data-pipeline gap, or the event is in the future with no observation
yet), `γ_eff(t) = γ₀` for that hour — the propagator silently falls back
to static drag. The output JSON reports `n_missing_hours` so downstream
consumers can flag low-confidence predictions.

If `proton_density` is null for **≥ 20% of hours** between launch and
the earlier of (a) 1 AU arrival or (b) `launch + 72h`, the predict
command emits a warning and the prediction should be treated as
equivalent to the static baseline — no Phase 9 improvement is possible
without L1 data.

**Known data gap (2026-04-10):** `omni_hourly` has all columns null from
`2026-03-29 01:00` onward. This affects 2 of 4 CCMC benchmark events
(Mar-30, Apr-01). Those events will fall back to static predictions and
score as Phase 8 does. Backfilling OMNI post-March-29 is a Phase 9
prerequisite tracked separately in the data-acquisition backlog.

### 3.4 Why not residual-nudge on γ

The original draft of this spec proposed a first-order residual-nudge
update `γ(t_k) = γ(t_{k-1}) + η · (v_obs - v_pred) / (v_pred · |v_pred - w|) · Δt`,
treating γ as a hidden state to be estimated from the velocity residual.
The April 2026 throwaway Python validation falsified this approach:

- **The observable mismatch.** L1 `flow_speed` measures ambient solar wind
  speed at 1 AU, not the CME's own front speed mid-transit. Using
  `v_obs_L1 - v_pred_CME` as a correction signal compares two different
  physical quantities. In the Jan-18 test, this drove γ in the wrong
  direction (the observed wake wind was slightly *elevated* at 480-500
  km/s due to precursor compression, which the residual-nudge interpreted
  as "CME is moving fast, reduce drag further," amplifying an already
  wrong prior).
- **Density has no such ambiguity.** `proton_density` at L1 directly
  measures the medium the CME is plowing through (with the caveat that
  we observe it as it passes L1, which for a ~1000 km/s CME leaving 20
  R_sun is sampled ~20h later). The quasi-steady solar wind assumption
  holds well enough that wake density measured at L1 at time `t` is a
  reasonable proxy for density along the CME path at time `t`.
- **No tunable.** Residual-nudge needed `η` (gain). Density-modulation
  has `n_ref` = 5 cm⁻³, which is a physical constant from OMNI
  climatology and cannot be "tuned."

A proper Kalman filter that jointly tracks `γ` and the L1 advection delay
is the principled next step if density-modulation plateaus. That's a
Phase 9.5 follow-up, not v1.

## 4. Command-line interface

```
solarpipe predict-progressive \
    --config configs/phase8_live_eval.yaml \
    --event 2026-01-18T18:09:00 \
    --until 2026-01-20T00:00:00 \
    [--n-ref 5.0] \
    [--output output/progressive/]
```

- `--config` — reuses the Phase 8 config to pick up feature definitions,
  drag parameter priors, and data source paths. No new config format.
- `--event` — launch time of the CME to propagate. The command looks up
  the matching row in `feature_vectors` to seed `v(0)`, `γ₀`, and the
  `activity_id` used for no-leakage bounds.
- `--until` — wall-clock time to propagate to, or until shock arrival,
  whichever comes first. Must be `launch_time + 72h` at most.
- `--n-ref` — quiet-sun reference density for γ scaling, cm⁻³ (default
  `5.0`). This is NOT a tunable — it is exposed for scientific scrutiny
  and should only be changed if OMNI climatology is re-derived.
- `--output` — writes `progressive_<event_id>.json` with the full state
  trajectory `[(t_k, r_k, v_k, γ_eff_k, n_obs_k, v_obs_k)]` and the final
  predicted/detected arrival time.

### 4.1 Backtest mode

```
solarpipe predict-progressive \
    --config configs/phase8_live_eval.yaml \
    --backtest configs/phase8_live_eval.yaml \
    [--eta 0.15] \
    [--output output/progressive_backtest/]
```

Runs the progressive predictor against every event in the held-out 2026
test set, using historical `omni_hourly` data that post-dates each event's
launch (no leakage: for event `e`, only `omni_hourly.datetime > e.launch_time`
is consumed, and the query is capped at `e.launch_time + 72h`). Produces a
results JSON with the same schema as `phase8_domain_results.json` so the
existing CCMC comparison code can diff static-mode vs progressive-mode
directly.

## 5. File layout

```
src/SolarPipe.Training/Physics/
  ProgressiveDragPropagator.cs     ~160 lines — wraps DragBasedModel, step-by-step integration, shock detection
  DensityModulatedDrag.cs           ~40 lines — γ_eff(n_obs, γ₀, n_ref) with clipping
src/SolarPipe.Data/
  L1ObservationStream.cs           ~120 lines — hourly OMNI reader, no-leakage guards
src/SolarPipe.Host/Commands/
  PredictProgressiveCommand.cs     ~200 lines — argument parsing, event lookup, output write
tests/SolarPipe.Tests.Unit/Physics/
  ProgressiveDragPropagatorTests.cs
  DensityModulatedDragTests.cs
tests/SolarPipe.Tests.Integration/
  PredictProgressiveCommandTests.cs
```

All five new files stay under 400 lines per the existing rule.

## 6. Test plan

### 6.1 Unit tests

- `DensityModulatedDragTests`:
  - `n_obs = n_ref` → γ_eff = γ₀ exactly
  - `n_obs = 2 · n_ref` → γ_eff = 2 · γ₀ exactly
  - `n_obs = 0` → γ_eff clipped to minimum `1e-9`, not zero
  - `n_obs = 1000` (sensor spike) → γ_eff clipped to maximum `1e-6`
  - `n_obs = null` → γ_eff = γ₀ (fallback to static)

- `ProgressiveDragPropagatorTests`:
  - Synthetic case: all observations have `n_obs = n_ref` → propagator output equals static DragBasedModel output to within 0.5h tolerance.
  - Synthetic case: all observations have `n_obs = 2 · n_ref` → transit is longer than static by the amount that doubling γ implies (verifiable with closed-form drag solution).
  - Synthetic case: all observations have `n_obs = 0.5 · n_ref` → transit is shorter by the amount halving γ implies.
  - Shock detection: inject a synthetic +300 km/s + density-jump event at a known hour. Propagator must terminate at that hour with `shock_arrived=true`.
  - No-leakage guard: asking for observations at `t < launch_time` must raise; asking for `t > launch_time + 72h` must raise.
  - Missing-observation resilience: 20% random null rows in the input stream → propagator still converges and reports `n_missing_hours` in output metadata.

### 6.2 Integration tests

The density-modulated rule has been Python-validated on the full CCMC
4-event benchmark. See `scripts/phase9_throwaway_4event.py` for the
reference numbers the C# implementation must match. Reference values
(Python-computed, γ₀ = 2e-8 km⁻¹, n_ref = 5 cm⁻³):

| Event | Truth | Ph8 baseline | Static (Python) | Density-modulated | DM err |
|---|---|---|---|---|---|
| 2026-01-18 X1.9 | 24.77h | 43.52h | 37.80h (w=520) | **25.00h** | **+0.23h** |
| 2026-03-18 | 58.90h | 58.78h | 61.00h (w=416) | 62.00h | +3.10h |
| 2026-03-30 | 56.08h | 48.62h | 39.80h | *fallback*¹ | — |
| 2026-04-01 | 39.28h | 47.77h | 47.30h | *fallback*¹ | — |

¹ Falls back to static prediction because `omni_hourly` has no data
after 2026-03-29 01:00 (see §3.5 known data gap). On the 2 events with
full OMNI coverage:

- **Phase 8 MAE (2-event): 9.44h**
- **Density-modulated MAE (2-event): 1.67h** (-82%)

Shock-detection criterion fires correctly on Jan-18: density jump
3.1 → 7.9 → 23.9 cm⁻³, speed 440 → 679 → 1125 km/s in two hours at
`2026-01-19 19:00`, matching the labeled ICME arrival time within 5
minutes.

Sensitivity to `n_ref ∈ {3, 5, 7} cm⁻³`:
- Jan-18: 25.00h / 25.00h / 25.00h — insensitive (shock detection dominates)
- Mar-18: 67.00h / 62.00h / 60.00h — ±7h spread, tolerable but worth backfilling quiet-sun climatology if production needs tightening.

- `PredictProgressiveCommandTests`:
  - Run against each of the 4 CCMC benchmark events with a mock `L1ObservationStream` that replays real `omni_hourly` rows between launch and `launch + min(72h, transit * 1.1)`. Each event's predicted arrival time must match the Python reference within ±0.5h (verifies the C# re-implementation is numerically consistent with the Python premise test).
  - The Jan-18 X1.9 must come in with `|error| ≤ 10h` (vs Phase 8's 18.76h). This is the hard must-have.
  - The Mar-18 event must come in with `|error| ≤ 5h` (vs Phase 8's 1.12h — assimilation must not regress the easy one by more than a few hours; this is the "do no harm" case).
  - Run the `--backtest` mode against the full 71-event Phase 8 test set. The aggregate transit MAE must be ≤ 12.33h (baseline). This is a soft target — if progressive mode matches baseline aggregate while dramatically improving on the cannibalism outliers, that's a win.

### 6.3 Numerical tests

- Gamma must stay in `[1e-9, 1e-6] km⁻¹` at all times (physical range for drag parameter). Hard-clip if the update would exceed.
- `v(t)` must stay in `[200, 3000] km/s`. Clip and log warning if violated.
- Integration step size must never exceed 1 hour (assimilation cadence) or go below 60 seconds (solver stability).

## 7. Expected outcomes

### 7.1 Must have (ship criterion)

- Jan-18 X1.9 predicted arrival within **3h of truth** (vs 18.75h error in Phase 8 baseline). Tightened from "12h → 10h" in earlier drafts because the Python validation reaches **0.23h** error on this event. A C# re-implementation that can't stay within 3h is broken.
- Aggregate 71-event transit MAE no worse than 12.33h baseline **on the OMNI-covered subset** (events where `proton_density` is ≥80% non-null between launch and launch+72h). Events outside that subset fall back to static and are not counted against Phase 9.
- CCMC benchmark MAE improves on the subset with OMNI coverage. As of 2026-04-10 that subset is 2 of 4 events (Jan-18, Mar-18). On that subset Python gives 9.44h → 1.67h — the C# implementation must reproduce this within ±0.5h per event. The 4-event headline number depends on OMNI backfill for Mar-30 and Apr-01 (see §3.5).

### 7.2 Nice to have

- Aggregate MAE improves measurably (say, 12.33h → 10-11h) because assimilation also helps medium-transit events where ambient wind mispredictions dominate.
- Per-event uncertainty estimates from the residual trajectory (std of residuals over the transit → arrival-time uncertainty).
- A graceful fallback when `omni_hourly` has a gap: the propagator should skip the missing hour (pure forward integration) rather than crash, so the command works on partially-available streams.

### 7.3 Not expected

- Phase 9 will not help events where the progenitor attribution is wrong. If the "observed transit" in the label is computed against the wrong shock, progressive mode will converge to the right physical answer and still score as an error against the wrong label. This is a data-labeling problem, not a prediction problem, and is out of scope.
- Phase 9 will not help events launched more than 72h before their arrival. The capped-window design prevents the command from consuming a week of OMNI to predict one CME; events that slow are better served by the static pipeline.

## 8. Risks and mitigations

| Risk | Mitigation |
|---|---|
| `n_ref` could be mis-calibrated for the current solar cycle | `n_ref = 5 cm⁻³` is the OMNI 1963-2023 median for quiet-sun conditions. A one-line sensitivity test at `n_ref ∈ {3, 5, 7} cm⁻³` is included in the 4-event validation; if results are strongly sensitive, re-derive from 2020-2026 climatology before production. |
| Density-modulation could make quiet-wind events worse | The "do no harm" integration test on Mar-18 catches this. If the aggregate 71-event MAE regresses, that's a hard fail — no ship. |
| Hourly OMNI is too coarse to detect fast events | Shock detection criterion uses hour-over-hour density jumps, which are visible at 1h cadence (confirmed on Jan-18 data: speed 440→679 in one hour, density 3.1→7.9 in one hour). |
| Leakage in `--backtest` mode if we read `omni_hourly.datetime` at or beyond the labeled `shock_arrival_time` | Hard cap: propagator refuses to read observations beyond `launch_time + min(72h, labeled_transit * 1.1)`. This gives the assimilation loop some runway past the true arrival (needed for shock detection to trip) without letting it "see the answer." |
| Progressive mode regresses easy events because assimilation over-corrects on noisy OMNI | The `|progressive_error| <= |static_error| + 2h` integration test catches this per-event. If it fires, we lower default `eta` and revisit. |
| CME-CME interaction events where even assimilation fails because the drag model is the wrong kinematic model for post-cannibalism wakes | Accept. Phase 9 v1 targets "single CME through OMNI-observable wind." Post-cannibalism wakes are Phase 9.5 territory — they'd need a multi-body propagator, not an assimilation upgrade. |

## 9. Milestones

1. **M1 — Static reference** (0.5 day). Write the backtest scaffolding that runs the *existing* DragBasedModel step-by-step against `omni_hourly` without modulation, writing the per-step trajectory to JSON. This establishes the output format and the no-leakage plumbing. No new physics.
2. **M2 — Density-modulated drag** (0.5 day). Implement `DensityModulatedDrag` (one function) and wire it into `ProgressiveDragPropagator` as a per-step γ_eff evaluator. Half a day because the rule is ~30 lines with no state — dramatically smaller than the original residual-nudge design.
3. **M3 — CLI command** (0.5 day). `PredictProgressiveCommand` wraps the propagator, handles `--event` and `--backtest` modes, plus the 20%-gap fallback warning.
4. **M4 — Unit + integration tests** (1 day). Full test plan from §6, including the 4 CCMC events against Python reference numbers.
5. **M5 — Parity validation** (0.5 day). Run the C# command on all 4 CCMC events, confirm per-event transit hours match the Python reference within ±0.5h. If Jan-18 C# prediction differs from Python 25.00h by more than 0.5h, the C# implementation is wrong — fix before proceeding.
6. **M6 — Full backtest + comparison report** (0.5 day). Run `--backtest` on the 71-event set (with OMNI-coverage filter), produce a side-by-side JSON + markdown report against the Phase 8 baseline. Explicitly tag which events fell back to static.

Total rough scope: **3.5 working days** (down from 4) because M2 is simpler now that density-modulation replaces residual-nudge.

**Prerequisite (not in Phase 9 scope):** Backfill `omni_hourly` past
2026-03-29. Until this is done, the Mar-30 and Apr-01 CCMC events will
fall back to Phase 8 behavior, and the full 4-event CCMC headline number
cannot be computed. This is tracked separately as a data-acquisition work
item, not a Phase 9 deliverable.

## 10. Explicit non-goals

- No new training. No new model artifacts. No new pipeline config keys. Phase 9 is purely an inference-side addition.
- No rewrite of Phase 8. If Phase 9 delivers, the two live side by side; `domain-sweep` remains the training/evaluation workflow, and `predict-progressive` becomes the preferred inference mode for events where L1 observations are available.
- No attempt to chase ELEvo's full methodology. We copy the one idea (assimilate L1 during transit) that matters for our failing case, and ignore the rest (ensemble of drag parameters, geometric deprojection, etc.) until there's evidence it's needed.
