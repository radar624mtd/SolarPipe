# Neural Ensemble Implementation Plan (PINN + TFT + Ensemble Head)

**Status document — updated every session. The last agent to touch a gate MUST update this file before ending the session. No exceptions.**

- **Created:** 2026-04-17
- **Owner:** radar (project), Claude (implementing agent)
- **Target:** MAE ≤ 3h ± 2h on the 90-event holdout (`training_features WHERE split='holdout'`)
- **Data contract:** `staging.db` → `training_features` VIEW — **133 columns × 9,418 rows** (1,974 with `split IS NOT NULL`: 1,884 train + 90 holdout)
- **Training framework:** PyTorch sidecar (existing gRPC path). TorchSharp rejected — see ADR below.
- **Inference framework:** ONNX Runtime via existing `OnnxAdapter`.
- **Null strategy:** value + per-column mask + learnable null embedding; phantom cols dropped at load.

---

## 0. How to use this document (inter-session protocol)

1. **At session start**, open `docs/NEURAL_ENSEMBLE_PLAN.md` and the memory pointer at `memory/project_neural_ensemble_plan.md`. Read §1 (current gate), §4 (last session log), §5 (open issues).
2. **Before writing any code**, identify which gate (G1–G7) you are advancing. If unclear, stop and ask.
3. **Each gate has a checklist in §3**. Every checkbox must be ticked with the commit hash that satisfied it. Do not skip.
4. **At session end**, append to §4 the Session Log: date, agent identity, gate touched, files changed, commit hashes, blockers. If a gate advanced, mark it in §2.
5. **Memory sync:** update `memory/project_neural_ensemble_plan.md` with a one-line status only (link back here for detail).
6. **Rule of continuity:** never delete entries from §4. Append only. Corrections go in new entries referencing old ones.

---

## 1. Current Gate

**→ G1 — Schema Contract** (not started as of 2026-04-17)

Next action: create `python/feature_schema.py`.

---

## 2. Gate Status Board

| Gate | Name | Status | Exit commit | Notes |
|---|---|---|---|---|
| G1 | Schema contract | ☐ not started | — | `python/feature_schema.py` enumerating all 133 cols |
| G2 | Masked dataset loader | ☐ blocked on G1 | — | `python/datasets/training_features_loader.py` |
| G3 | TFT + PINN model | ☐ blocked on G2 | — | `python/tft_pinn_model.py` replacing `_SimpleTftModel` |
| G4 | Physics loss | ☐ blocked on G3 | — | `python/physics_loss.py` (drag ODE, monotonicity, quantile) |
| G5 | ONNX export | ☐ blocked on G4 | — | extend existing `ExportOnnx` RPC, opset ≤ 20 |
| G6 | C# wiring + YAML | ☐ blocked on G5 | — | `configs/neural_ensemble_v1.yaml` + `OnnxAdapter` round-trip |
| G7 | Holdout quality gate | ☐ blocked on G6 | — | MAE ≤ 6h (Tier 1+2 baseline) before P6 ensemble head |

Legend: ☐ not started · ◐ in progress · ✅ complete · ❌ failed/rolled back

---

## 3. Gate-by-gate checklists

Every box must be ticked with (commit hash, date) before the gate is declared complete.

### G1 — Schema contract
Goal: a single Python module that is the authoritative list of features, enforced at every training run.

- [ ] Create `python/feature_schema.py` with a `FEATURE_SCHEMA: list[ColumnSpec]` where each entry has `{name, dtype, tier (0/1/2), branch ("flat" | "sequence" | "label" | "key" | "drop"), null_policy ("dense" | "sparse" | "phantom" | "mnar"), default_value}`.
- [ ] Enumerate all **133** columns from `training_features`; total assignment: 7 key/label, 127 flat, 6 drop (phantoms). **The count must sum to 133 exactly** — unit test enforces this.
- [ ] List phantom-drop columns explicitly: `rc_bz_min`, `dimming_area`, `dimming_asymmetry`, `eit_wave_speed_kms`, `phase8_pred_transit_hours`, `pinn_v1_pred_transit_hours` (the prediction slots re-enter at P6, not now).
- [ ] Add `SEQUENCE_CHANNELS: list[str]` = 20 OMNI channels + 2 GOES MAG channels (Hp, Bt) → **22 total**. Leave room for SuperMAG (+2 → 24) per RULE-213.
- [ ] Add `assert_schema_matches_db(conn)` that queries `PRAGMA table_info(training_features)` and aborts with a diff if drift detected.
- [ ] Unit test `tests/test_feature_schema.py`: (a) len == 133, (b) phantom count == 6, (c) flat count == 127, (d) no duplicate names, (e) `assert_schema_matches_db` passes against live `staging.db`.
- [ ] `python -m py_compile` + `ruff check --select E,F` clean.
- [ ] Update §2 gate board and §4 session log with exit commit.

### G2 — Masked dataset loader
Goal: produce `(x_flat, m_flat, x_seq, m_seq, y)` batches directly from `training_features` + OMNI Parquet, with correct split handling.

- [ ] Create `python/datasets/training_features_loader.py` exposing `TrainingFeaturesDataset(split: "train"|"holdout", db_path, sequences_path)`.
- [ ] Flat branch: read via SQL `SELECT * FROM training_features WHERE split = ? AND exclude = 0 ORDER BY launch_time`. Apply schema: drop phantoms, cast to float32 where numeric, categorical → integer codes with NULL → -1 sentinel, then mask NaN → 0.0 with mask=0.
- [ ] Sequence branch: join to pre-built Parquet from `scripts/build_pinn_sequences.py` by `activity_id`. Shape assert `(B, 150, 22)` after Tier 2 GOES MAG channels included (run the updated builder first — track in Session Log).
- [ ] Split-leak guard: `assert set(holdout.activity_id) ∩ set(train.activity_id) == ∅`.
- [ ] Deterministic order: sort by `launch_time` ascending so temporal splits remain reproducible.
- [ ] Unit tests: (a) no NaN in output tensors, (b) mask==0 everywhere x==0-by-imputation, (c) shapes match schema, (d) `len(train)==1884`, `len(holdout)==90`.
- [ ] Update §2 and §4.

### G3 — TFT + PINN model
Goal: replace `_SimpleTftModel` with a real TFT that consumes masks and emits quantile predictions.

- [ ] Add `pytorch-forecasting` to `python/requirements.txt`; verify `python3.12 -m pip install -r python/requirements.txt` succeeds.
- [ ] Create `python/tft_pinn_model.py` with:
  - Flat encoder: MLP over concat(x_flat, m_flat, learnable_null_embedding), LayerNorm, 2 residual blocks.
  - Sequence encoder: pytorch-forecasting TFT with masked attention (additive −inf on `m_seq==0`).
  - Head: concat → MLP → **3 outputs** (P10, P50, P90 transit-time in hours).
- [ ] Wire into `solarpipe_server.py` behind a feature flag `use_tft_pinn` (default off) so old LSTM stub remains until G5 green.
- [ ] Overfit test `tests/test_tft_pinn_overfit.py`: 32 rows → MAE < 0.5h after 200 steps. Deterministic seed.
- [ ] Shape tests: input `(B, 150, 22)` + flat `(B, 127)` + masks → output `(B, 3)`.
- [ ] Update §2 and §4.

### G4 — Physics loss
Goal: add PINN residual losses enforcing physics plausibility.

- [ ] Create `python/physics_loss.py` with four losses:
  1. **Drag ODE residual**: integrate `dv/dt = -γ_eff·|v−v_sw|·(v−v_sw)` with γ_eff from density modulation (mirror `ProgressiveDragPropagator`). Use `torchdiffeq.odeint` if lightweight; otherwise explicit RK4 in torch for autodiff.
  2. **Monotonic decay hinge**: penalize `max(0, v(t+1)−v(t))` while `v(t) > v_sw(t)`.
  3. **Transit bound**: `max(0, 12−ΔT)² + max(0, ΔT−120)²`.
  4. **Quantile ordering**: pinball loss + `max(0, P10−P50)² + max(0, P50−P90)²`.
- [ ] Total loss = `L_pinball + λ₁·L_ode + λ₂·L_mono + λ₃·L_bound + λ₄·L_qorder`, λ configurable in YAML, defaults documented in this file.
- [ ] Gradient finite-difference test for each loss term on synthetic data.
- [ ] Update §2 and §4.

### G5 — ONNX export
Goal: export the trained model including mask inputs; verify parity with PyTorch.

- [ ] Extend proto `ExportOnnx` path to emit a graph with **named inputs**: `x_flat`, `m_flat`, `x_seq`, `m_seq` → outputs `p10`, `p50`, `p90`.
- [ ] Clamp opset to 20 (already enforced server-side per CLAUDE.md; re-verify).
- [ ] Parity test: load exported ONNX via `OnnxAdapter`, run 10 holdout rows, compare to PyTorch within `1e-4` max abs diff.
- [ ] Store test artifact in `tests/fixtures/tft_pinn_smoketest.onnx` (git-ignored; regenerated by test).
- [ ] Update §2 and §4.

### G6 — C# wiring + YAML
Goal: full train → export → predict round-trip through the CLI.

- [ ] Create `configs/neural_ensemble_v1.yaml` with `table: training_features`, `filter: split IS NOT NULL`, `features: [<127 flat cols>]`, `sequence_channels: [<22>]`, sidecar training stage + onnx predict stage.
- [ ] Run `dotnet run --project src/SolarPipe.Host -- validate --config configs/neural_ensemble_v1.yaml` — passes.
- [ ] Run `train` then `predict` end-to-end; capture holdout MAE in `output/neural_ensemble_v1/`.
- [ ] `dotnet test --filter Category=Pipeline` passes.
- [ ] Update §2 and §4.

### G7 — Holdout quality gate
Goal: confirm Tier 1+2 baseline beats or matches PINN V1 before advancing to P6 ensemble head.

- [ ] Holdout MAE logged, must be **≤ 6h**. (PINN V1 baseline: 8.69h.)
- [ ] Quantile calibration check: holdout coverage for P10–P90 interval within [0.70, 0.90].
- [ ] If MAE regressed: file an issue in §5, roll back to last green commit, do NOT advance to P6.
- [ ] If green: mark plan §1 "All gates green — proceed to Ensemble Head (P6) per CLAUDE.md roadmap."

---

## 4. Session Log (append only — never delete)

Every session that touches any gate must add an entry. One entry per session, in reverse-chronological order (newest at top).

### 2026-04-17 — Plan authored
- Agent: Claude (this session, summary id 6dd739c0-...)
- Actions: Verified DB inventory (`training_features` = 133 cols × 9,418 rows; `pinn_expanded_flat` = 108 × 1,974). Authored this plan. No code written yet.
- Files: `docs/NEURAL_ENSEMBLE_PLAN.md` (new), `memory/project_neural_ensemble_plan.md` (new).
- Gate touched: plan only. G1 still not started.
- Blockers: none for G1. Note for G2: sequence Parquet must be rebuilt with 22 channels before loader tests run — see RULE-213.
- Next session: start G1 checklist from top.

---

## 5. Open issues / decisions pending

| ID | Date | Issue | Decision/status |
|---|---|---|---|
| I-01 | 2026-04-17 | SuperMAG (SME/SMR) blocked on account activation — affects G2 sequence channel count | Hold at 22 channels. When unblocked, expand to 24 per RULE-213 as a single atomic update (sequences + model + server). No intermediate 23-channel state. |
| I-02 | 2026-04-17 | `phase8_pred_transit_hours` and `pinn_v1_pred_transit_hours` columns exist but are 100% NULL | Drop at G1 (phantom). Re-introduce at P6 ensemble head by backfilling these columns via separate pipeline run; not this plan's scope. |
| I-03 | 2026-04-17 | HEK `dimming_area`, `dimming_asymmetry`, `eit_wave_speed_kms` are phantoms — HEK FRMs don't populate them | Drop permanently. Existence flags (`has_coronal_dimming` etc.) carry the usable signal. |
| I-04 | 2026-04-17 | Tier 3 is mandatory per `feedback_tier3_is_mandatory.md`, but the neural ensemble plan does not block on it | Tier 3 integrates post-G7 as an additive expansion; it does not block G1–G7. Record any Tier 3 columns added by adjusting `FEATURE_SCHEMA` at that time. |

---

## 6. Architectural decision record

### ADR-NE-001: Training in PyTorch sidecar, not TorchSharp
- **Context:** Need full TFT + PINN with masked attention and autodiff through an ODE integrator.
- **Options considered:**
  1. TorchSharp in-process.
  2. Existing gRPC sidecar → PyTorch + pytorch-forecasting.
  3. ML.NET / ONNX Runtime only.
- **Decision:** Option 2. The sidecar is already wired (`GrpcSidecarAdapter`, `solarpipe_server.py`, `ExportOnnx` RPC). `pytorch-forecasting` provides TFT out of the box. Option 1 doubles the native dependency surface and re-implements what already works. Option 3 cannot train — inference only.
- **Consequences:** Training requires Python 3.12 sidecar. Inference in C# uses ONNX Runtime (already referenced). Proto stubs must be recompiled after any `.proto` change (per CLAUDE.md). Opset clamped to ≤ 20.

### ADR-NE-002: Value + mask + learnable embedding for nulls
- **Context:** 108-col feature matrix is bimodally null — 6 phantoms (100%), ~12 MNAR sparse (87–95%), dense core (0–10%).
- **Decision:** At every null-possible column, concatenate `(value_with_NaN→0, observed_mask, learnable_null_embedding)`. Phantoms dropped at load, not masked.
- **Rejected alternatives:** Mean/median imputation (loses MNAR signal), forward-fill (not temporally meaningful for flat features), drop-rows (kills 70%+ of training set due to Tier 2 sparsity).

### ADR-NE-003: `training_features` VIEW is the only C# access point
- **Context:** `pinn_expanded_flat` (1,974 × 108) and `feature_vectors` (9,418 × 58) both exist. Neural head wants per-activity rich features.
- **Decision:** All C# YAML configs reference `table: training_features` + `filter: split IS NOT NULL`. Do not bypass the view.
- **Consequences:** Any new flat feature added to either input table is automatically picked up by rebuilding the view. Schema contract (G1) must be re-validated after any view DDL change.

---

## 7. Cross-reference map

| Concept | Canonical source |
|---|---|
| Ingest endpoint audit, bugs, sentinels | `docs/ENDPOINT_AUDIT_2026_04_15.md` |
| Per-channel source → schema mapping | `docs/SOURCE_TO_SCHEMA_REFERENCE.md` |
| Python → C# promotion lifecycle | `docs/ENDPOINT_CSHARP_MAPPING.md` |
| DB schema, row counts, alias table | `docs/DATA_SCHEMA_REFERENCE.md` |
| Tier 1/2 current state | `memory/project_tier1_feature_matrix_state.md` |
| Tier 3 mandate | `memory/feedback_tier3_is_mandatory.md` |
| Project-wide rules | `CLAUDE.md` §ABSOLUTE IMPERATIVE (RULE-200–216) |
| This plan | `docs/NEURAL_ENSEMBLE_PLAN.md` (you are here) |
| Memory pointer | `memory/project_neural_ensemble_plan.md` |
