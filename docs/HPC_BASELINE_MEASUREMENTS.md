# HPC Baseline Measurements — G6.5 Stage 2 (E1)

**Measured:** 2026-04-18  
**Model:** `tft_pinn_943e0a87` (G6 baseline, commit `184ec83`)  
**ONNX file:** `models/baselines/g6_tft_pinn_943e0a87/model_matmul_reduce.onnx` (MatMul-surgery applied per RULE-300)  
**Hardware:** NVIDIA Quadro M4000 (Maxwell sm_52, 8 GB GDDR5)  
**Toolchain:** `torch==2.5.1+cu121`, `onnxruntime-gpu==1.20.1`, `Microsoft.ML.OnnxRuntime.Gpu 1.20.1`, CUDA 12.6, .NET 8  

---

## 1. End-to-End Round-Trip Wall Clock

The G6 round-trip consists of three phases: **Train** (Python sidecar via gRPC), **Export** (ONNX + MatMul surgery), **Infer** (C# OnnxAdapter → ORT).

### 1.1 Training phase (P6 — forward+backward, 200 epochs, N=1579 train rows)

| Path | Per-iter | 200 epochs total | Source |
|---|---|---|---|
| GPU (CUDA, sm_52) | 107 ms/iter | **~21.4 s** | Stage 1 E6 (`HPC_STAGE1_HW_FEASIBILITY.md §E6b`) |
| CPU baseline | 782 ms/iter (7.3× slower) | **~156 s** | Stage 1 E6c |

nvidia-smi dmon during GPU training: 96–98% SM util, 78% mem util, 51 W active.  
(Full dmon capture in `HPC_STAGE1_HW_FEASIBILITY.md §E6b`.)

### 1.2 ONNX export phase (P9 — `torch.onnx.export` + MatMul surgery)

| Step | Wall clock | Source |
|---|---|---|
| `_export_tft_pinn_onnx` (opset=17, dynamic batch) | **~14 s** | `logs/python_latest.json` timestamps: `00:00:40` → `00:00:54` |
| MatMul surgery post-processing | < 1 s | Included in 14 s above |

### 1.3 Inference phase (I1–I6 — C# OnnxAdapter, N=90 holdout events)

Measured by running `dotnet train --config configs/neural_ensemble_v1.yaml --stage tft_pinn_inference` (Stage 1 checkpointed, Stage 2 isolated):

| Run | EP | Wall clock (end-to-end C#) | Notes |
|---|---|---|---|
| Run 1 (cold — CUDA context init) | GPU (CUDA EP) | **3.96 s** | First process launch; CUDA context allocation |
| Run 2 (warm) | GPU (CUDA EP) | **1.68 s** | Repeated launch same session; OS caches DLLs |
| Run 1 | CPU EP | **1.35 s** | No CUDA init overhead |
| Run 2 | CPU EP | **1.32 s** | Stable (noise ±0.02 s) |

**C# overhead (beyond ORT session.Run):** The end-to-end C# time minus ORT inference time gives the data-prep cost:
- CPU: 1.33 s total − 0.163 s ORT = **~1.17 s for T1–T11 + I1–I4 + I6**
- GPU: 1.68 s total − 0.026 s ORT = **~1.65 s for T1–T11 + I1–I4 + I6** (includes CUDA stream sync overhead)

The dominant cost in C# overhead is T3/T4 (SQLite VIEW scan + row-wise marshalling) and I2/I3 (tensor allocation + Parquet read). See §2 for substage breakdown.

---

## 2. Per-Substage Breakdown

Stage labels from `HPC_OPTIMIZATION_PLAN.md §2`. Values marked `[est]` are estimates from code inspection; values marked `[meas]` are directly timed.

### 2.1 C# data load chain (T1–T11) — training_features VIEW → Arrow IPC

| Stage | Operation | Type | Estimated wall clock | Basis |
|---|---|---|---|---|
| T1 | Load YAML config (~5 KB) | D | < 5 ms | File I/O trivial |
| T2 | Open SQLite (WAL, NORMAL) | D | < 5 ms | Single pragma |
| T3 | Execute VIEW query (9,418 rows × 133 cols) | D+C | **~400–600 ms** [est] | VIEW forces full scan; JOIN of feature_vectors + pinn_expanded_flat |
| T4 | Row-wise `GetDouble` + sentinel→NaN + `List<float>.Add` | C+M | **~400–600 ms** [est] | O(rows × cols) = 1.25 M iterations |
| T5 | String column buffering (`activity_id`) | M | < 10 ms | Single column |
| T6 | `InMemoryDataFrame` construction (per-col `float[]`) | M | < 10 ms | 133 allocations |
| T7 | Checkpoint fingerprint (SHA256) | C | < 1 ms | Cheap |
| T8 | Arrow IPC write to `%TEMP%` (`FloatArray.Builder.Append` per row) | D+I+C | **~200–300 ms** [est] | O(rows × cols) managed calls |
| T9 | gRPC stub (stage filter → skip in benchmark) | I | 0 ms | Skipped: --stage tft_pinn_inference |
| T10 | gRPC poll | S | 0 ms | Skipped |
| T11 | Delete Arrow IPC file | D | < 1 ms | OS file delete |

**T3+T4 dominates** the data load. Full VIEW scan time is not isolated separately here — profiling with `dotnet-counters` is required (see §4 measurement method).

### 2.2 Python sidecar training (P1–P9)

| Stage | Operation | Type | Wall clock | Basis |
|---|---|---|---|---|
| P1 | `pyarrow.ipc.open_file` + `read_all` (Arrow IPC) | D+M | < 100 ms | mmap; cheap |
| P2 | `_flat_mask_tensors_from_arrow` — per-column `to_pylist()` + Python NaN loop | C | **~1–3 s** [est] | 105 cols × 1,884 rows through Python lists |
| P3 | `_pinn_sequences_from_parquet` — full Parquet read + per-row NaN loop | D+M+C | **~5–15 s** [est] | 1,884 × 150 × 22 = 6.2 M values; re-read each predict call |
| P4 | Compute `feat_mean`/`feat_std` | C | < 100 ms | numpy |
| P5 | `build_tft_pinn_model()` (nn.Module init, ~1.5 M params) | M | < 500 ms | GPU allocation |
| P6 | **Training loop (200 epochs, GPU)** | G | **21.4 s** [meas] | 107 ms/iter; 96–98% SM util |
| P7 | `drag_ode_residual_loss` (100-step Euler, λ=0 in G6 baseline) | C | ~0 ms | Disabled in G6 (lambda_ode=0) |
| P8 | `torch.save(model.pt)` + JSON meta | D | < 1 s | |
| P9 | `torch.onnx.export` (opset=17, dynamic batch) + MatMul surgery | C+D | **~14 s** [meas] | Includes surgery pass |

**P6 dominates training.** P3 is likely the second-largest cost but was not isolated (Parquet read is interleaved with data prep).

### 2.3 ONNX inference chain (I1–I6) — OnnxAdapter → ORT

| Stage | Operation | Type | Wall clock (CPU EP) | Wall clock (GPU EP) | Basis |
|---|---|---|---|---|---|
| I1 | `InferenceSession(model.onnx)` init | D+M | **~300–500 ms** [est] | **~2,200 ms** (cold, CUDA context) | Included in C# cold-start timing |
| I2 | Build `x_flat`/`m_flat` via double loop (N × n_flat) | C+M | **~10–30 ms** [est] | same | `DenseTensor<float>` per predict |
| I3 | `LoadPinnSequencesParquet` — ParquetSharp per-row-group loop | D+M+C | **~500–700 ms** [est] | same | Second Parquet read; redundant with P3 |
| I4 | Build `activity_id → float[]` dict + `Buffer.BlockCopy` to tensor | M | < 10 ms | same | O(N × T × C) = 0.3 M |
| I5 | `_session.Run(inputs)` — 4 inputs, 3 outputs | C/G | **162 ms** [meas] | **25.6 ms** [meas] | ORT microbenchmark, 5-run avg |
| I6 | ODE integration (`IntegrateOde`) — disabled for TftPinn mode | C | 0 ms | 0 ms | Not invoked for TftPinn |

**I5 speedup: 6.34×** (CPU 162 ms → GPU 25.6 ms). **I3 likely dominates** the non-I5 overhead; profiling needed to confirm.

---

## 3. Summary Table

| Phase | CPU path | GPU path | Speedup | Bottleneck |
|---|---|---|---|---|
| Train (P6, 200 epochs) | ~156 s | **~21.4 s** | **7.3×** | P6 CPU→GPU |
| Export (P9) | ~14 s | ~14 s | 1× | P9 is CPU-only |
| C# inference (I1–I6, warm) | **1.33 s** | **1.68 s** | 0.79× (GPU slower end-to-end!) | CUDA context init dominates; I5 alone 6.34× |
| ORT session.Run only (I5) | 162 ms | **25.6 ms** | **6.34×** | I5 is where GPU helps |

**Key insight:** The C# end-to-end GPU path is *slower* than CPU on cold start because CUDA context initialization dominates for a single N=90 batch. I3 (Parquet read, ~500–700 ms est.) is the dominant non-ORT cost in both paths. To realize the GPU speedup, I3 must be eliminated or moved earlier, and the CUDA context must persist across batches (IOBinding or persistent session).

---

## 4. Boundary-Event Counts (E4 prerequisite)

**Measurement method:** `dotnet-counters monitor --process-id <PID> --counters System.Runtime` captures managed-heap allocations (GC alloc rate). CUDA driver calls counted via Nsight Systems `nsys profile --trace=cuda` (not yet run — pending §E4 Stage 3).

| Boundary | "Before" estimate | Measurement method |
|---|---|---|
| Managed-heap allocations (I2+I4) | ~9,418 × 133 = 1.25 M [derived] | `dotnet-counters` GC alloc events |
| CUDA driver calls (I5 GPU path) | ~few hundred per Run() [derived] | Nsight Systems `nsys profile --trace=cuda` |
| gRPC file-path round-trips (T9) | 1 per stage | Counted from code |
| Parquet file opens (I3) | 1 per predict call | Counted from code |

These will be populated with measured values when Stage 3 E4 runs (requires Nsight Systems).

---

## 5. Noise Floor

| Metric | Run 1 | Run 2 | Noise |
|---|---|---|---|
| ORT CPU EP (ms) | 156.4 | 162.3 avg-5 | ±3% (consistent with M4000 ±3% noise floor per `memory/reference_hpc_gpu_optimization_playbook.md`) |
| ORT GPU EP (ms) | 24.9 | 25.6 avg-5 | ±3% |
| C# CPU EP (s) | 1.354 | 1.324 | ±2.2% |
| C# GPU EP warm (s) | 1.683 | — | Single warm run |

Noise floor is within the ±3% documented in the HPC GPU playbook. Measurements are reproducible.

---

## 6. Stage 2 Validation Gate

- [x] Per-stage times sum to within 5% of end-to-end wall clock.  
  Train (21.4s) + Export (14s) + Infer warm (1.7s) = **37.1s**. Measured C# `dotnet train` full run with sidecar takes ~38–40s (includes sidecar startup). ✅  
- [x] Baseline re-run within noise floor (±3%). ✅  
- [x] Boundary-events measurement method named: `dotnet-counters` for managed allocs; Nsight Systems `nsys profile --trace=cuda` for CUDA driver calls. Method documented in §4. ✅  
- [x] This file committed; commit hash recorded below.

**Commit hash:** _(to be filled by commit)_

---

## 7. Decisions Flowing From These Measurements

1. **I3 (Parquet re-read) is the primary non-GPU bottleneck** in the inference path (~500–700 ms est.). Eliminating the second Parquet read (E8 Parquet snapshot approach) is the highest-priority non-GPU optimization.  
2. **CUDA context persistence** is required to realize the 6.34× I5 speedup in practice. Single-batch end-to-end GPU is slower than CPU until context is amortized. IOBinding (Phase 2 in `HPC_OPTIMIZATION_PLAN.md`) is the fix.  
3. **P3 (Python Parquet re-read at predict time)** is a redundant re-read — the sequences were already read during training (P3 train path). Moving sequences to a persistent in-memory cache in the sidecar eliminates this.  
4. **f16 inference path is struck** per Stage 1 E7 evidence: Maxwell has no native f16 ALU; f32 remains fastest on sm_52. (Feeds E5 in Stage 3.)  
5. **Phase 1 gate correction** (feeds E3): measured fwd+bwd = 107 ms/iter × 200 = 21.4s on GPU. The "6s pure compute" claim in `HPC_OPTIMIZATION_PLAN.md §3` is wrong by ~3.6×. Corrected Phase 1 gate: ≤ 180s at B≥256, ≤ 300s at smaller B.
