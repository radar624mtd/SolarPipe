# HPC Plan — Red-Team Review

**Status:** Authoritative. Blocks G6 closure and G7 development.
**Reviewed artifacts:**
- [`docs/HPC_OPTIMIZATION_PLAN.md`](HPC_OPTIMIZATION_PLAN.md) (strategy)
- [`docs/HPC_EXECUTION_MAPPING.md`](HPC_EXECUTION_MAPPING.md) (per-action ledger)
**Date:** 2026-04-18
**Reviewer:** Claude Opus 4.7 (red-team pass)
**Verdict:** **Ratify after E1–E10 are addressed.** G6 reverts to ◐ (in progress) until the HPC validation gate (this document) passes. No G7 hyperparameter tuning may begin until then.

---

## Verdict (TL;DR)

Solid tactical artifact, ratify with three quantitative corrections + four feasibility caveats before it gates G7. The mapping discipline (per-action ledger, silo rules, cross-boundary budget) is exactly the right shape. But several headline numbers don't survive a calculator, and at least three "decisions" rest on assumptions that haven't been load-bearing-tested on this hardware.

---

## A. Completeness — what's missing

**A1. No measured baseline anywhere in the document.** Every "before" number in §6 (`N·133 = 1.25 M`, `200 D2H per epoch`, `90 ORT Run on CPU`) is a *derivation*, not a *measurement*. Phase 0 of the strategy doc explicitly requires a baseline file (`docs/HPC_BASELINE_MEASUREMENTS.md`) — that file does not exist yet. Ratifying this plan before Phase 0 lands inverts the playbook the user codified for PFSS (`memory/reference_hpc_gpu_optimization_playbook.md`: "validate correctness before timing"). **Block G7 ratification until §6 has measured columns alongside the modeled ones.**

**A2. No accuracy-preservation gate inside the ledger.** §3.1 action 1.6 fuses sentinel→NaN with mask production via SIMD on host. §3.2 action 1.11 does the same on the seq tensor. There is no row in the ledger that says "verify post-rewrite numerical parity vs current pipeline ≤ 1e-5". The strategy doc has gates (Phase 2: ≤1e-4, Phase 4: ≤1e-5), but the action-by-action ledger is the artifact engineers will follow — **add a "Parity gate" column** so each cross-rewrite action has a paired correctness check.

**A3. Backward-pass and optimizer memory not in §1.5.** VO-E1/E2 (Adam m,v) only sized at "= |params|". With the corrected param count (see B1), optimizer state alone is ~12 MB, not 4 MB. Activation memory in VO-E3/E4 is "≈ 0.26 KB × blocks × B" — that's wrong by ~20× because Transformer activations include `(B, T, d)` per layer (`B·222·128·4 = 114 KB per layer per B=1`, times 2 layers + saved attention scores `B·heads·T²` ≈ 198 KB per layer). At B=64 the activation footprint is closer to **40 MB**, not 8 MB. Still trivial on 8 GB Maxwell, but if anyone ever bumps `seq_n_layers` or `T`, the table mis-scales.

**A4. No mention of `cudnn_conv_algo_search` warm-up cost.** §3.7 action 6.1 says "build InferenceSession". With `cudnn_conv_algo_search = EXHAUSTIVE` (mentioned in strategy doc §6) the first `Run` can take 5–30 s on Maxwell. Phase 5 §8 sets a "cold-start ≤ 2 s" gate that is **mutually exclusive** with that ORT setting. Pick one.

**A5. Sequence layout claim is internally inconsistent.** §1.2 row VO-B1 says "Column-major (N, C, T) for cache-friendly Transformer input proj", but the actual model code (`tft_pinn_model.py:220`) consumes `(B, T, C)` and immediately concatenates along dim=-1. A `(N, C, T)` host layout means every minibatch must be transposed before upload — that's a 2.5 MB transpose per batch you didn't account for. Either keep `(N, T, C)` everywhere or add an explicit "transpose at H2D boundary" action with its own kernel assignment.

**A6. No row for the 80 % train / 20 % val split.** `solarpipe_server.py:916` does `torch.randperm(N)` and slices. That index buffer is per-epoch, lives on host, and triggers per-batch `tensor[idx]` gathers. Currently those gathers are CPU→CPU; after Phase 1 they become CPU→GPU advanced indexing, which on Maxwell can be slower than full-batch upload + on-device shuffle. **Add as VO-E7 with explicit decision: pre-shuffle on device once per epoch.**

**A7. CC-3 silo rule is violated by §3.4 action 3.4.** "u8 mask → f32 expand in **pinned**" with kernel "C#-ILGPU host-side **or** SIMD". ILGPU host-side compiles to PTX and runs on device — that's a CC-3 → CC-4 leak. The "or SIMD" disambiguates, but the row should commit to one kernel. Pick SIMD for this; ILGPU launch overhead per batch (~30 µs) dominates a 9 KB expand.

**A8. No fallback for ParquetSharp + NativeAOT.** §3.1 row 1.3 mandates ParquetSharp; Phase 5 mandates NativeAOT. ParquetSharp 14.x has reflection paths that are not AOT-clean on .NET 8. **Add a row in §9 (risks) — and either pin a known-AOT-clean ParquetSharp version, write the trim descriptor, or move the cache (action 1.10) to a flat `.bin` file produced once and never touched again** (which is closer to what action 1.9 already does, so action 1.3 is partly obsolete).

**A9. `meta.bin` schema not specified.** §3.3 row 2.4 names a binary file but doesn't define its layout. Without a versioned header (magic + version + per-section length-prefixed offsets), this becomes the next breaking-change source. **Add a §1.8 schema table or a `python/meta_schema.py` reference.** Same applies to `sequences_v{hash}.bin` (action 1.9).

**A10. No row for the OnnxRuntime EP arena allocator.** ORT CUDA EP defaults to a per-session arena that can grow to GBs on a fragmented allocation pattern. Maxwell with 8 GB shared with the OS can OOM under `cudnn_conv_algo_search=EXHAUSTIVE` if other processes are present. **Add an explicit `OrtCUDAProviderOptions.gpu_mem_limit` and an OOM-fallback row.**

---

## B. Quantitative errors

**B1. Model parameter count is wrong by ~3×.** Doc claims **498 K total params (≈ 2 MB f32)**. Direct enumeration from the actual `tft_pinn_model.py` code:

| Sub-module | Doc claim | Actual |
|---|---|---|
| FlatEncoder | ~412 K | **~1.01 M** |
| SeqEncoder | ~80 K | **~402 K** |
| Head | ~6 K | **~75 K** |
| **Total** | **~498 K** | **~1.49 M** |

Each ResidualBlock is `(d → 2d → d)` (`tft_pinn_model.py:75-76`), not `(d → d → d)`, so each block has `~263 K` params, not the implied `~50 K`. SeqEncoder has 2 transformer layers each with full attn (`4·d²` for QKVO) + FFN (`8·d²` for `d→4d→d`) = ~200 K each. Head is LN+`(384→192→3)` = ~75 K, not 6 K.

This propagates: **VO-E1/E2 are 6 MB each, not 2 MB. Total GPU resident at training is ~30+ MB, not 19 MB.** Still trivial on 8 GB, but f16 inference is now 3 MB not 1 MB, and any "params fit in L2" assumption fails (Maxwell L2 = 2 MB on M4000).

**B2. Per-epoch FLOPs are ~10× under-counted.** Doc (`HPC_OPTIMIZATION_PLAN.md` §3) claims `200 epochs × 77 GFLOPs ≈ 15.4 TFLOPs ≈ 6 s on Maxwell`. Recomputing with correct params and including backward (=2× forward):

- Forward per sample: ~117 MFLOPs (attn QK^T and softmax·V at T²·d on 222² is ~13 MFLOPs/layer; FFN 8·T·d² is ~29 MFLOPs/layer; doc missed that T² scales)
- Per-epoch (1,884 samples × 3× for fwd+bwd): **~660 GFLOPs**
- 200 epochs: **~132 TFLOPs**
- Maxwell: **~50–90 s** at 1.5–2.6 TFLOPs sustained, **>4 min** if launch-overhead-bound at small B

The Phase 1 gate of "≤ 120 s" is plausible but tight, **not "trivially 6 s"**. If anyone uses the 6 s figure to justify "GPU is way oversized so don't bother with launch-overhead optimizations", they will miss the actual bottleneck. **Restate the Phase 1 gate as "< 180 s with B≥256, < 300 s otherwise".**

**B3. "Cross-boundary events: 1.25 M → 100" is rhetorical, not measurable.** The 1.25 M figure (= N·133 SQLite reads) double-counts: reading column-major *was already* one boundary event per column in the current code, not per cell — the cell-level cost is RAM bandwidth, not boundary crossings. This is conflating "boxing-equivalent managed allocations" with "kernel launches" with "syscalls". **Pick one definition** (suggest: count CUDA driver calls + syscalls separately from managed-heap allocations), and re-do §6 with two columns.

**B4. "Maxwell has limited f16 ALU; likely keep f32 for safety, accept 2 MB" (VO-D row note).** Maxwell has **no native f16 ALU at all** for compute (f16 is storage-only on sm_52). ORT CUDA EP's `kFloat16` path on Maxwell falls back to f32 compute via casts — net effect is **slower** than pure f32, not faster. **Strike the f16 inference column entirely; it will only mislead.**

**B5. "Maxwell L2 = 2 MB" affects the cache-friendly claim.** §1.2 VO-B1 wants `(N, C, T)` to be cache-friendly. With N=1,974, even at u8, the seq tensor is 8 MB — **never resident in L2**. The cache-line argument is for the per-event slice (T·C·4 = 17.8 KB), which fits easily in L1 regardless of memory order. The layout choice should be driven by what cuBLAS GEMM expects (row-major `(B, T, C·2)` after concat-with-mask), not by host cache. **Reword the rationale or drop it.**

---

## C. Feasibility risks

**C1. PyTorch 2.7 + CUDA 12.6 + sm_52.** The strategy doc pins this; the execution mapping assumes it. **PyTorch 2.7 stable wheels do not currently ship sm_52 PTX**. The last confirmed sm_52-bearing CUDA wheel is PyTorch 2.4.x (CUDA 12.1). 2.5/2.6/2.7 dropped Maxwell from the official wheel matrix. You may need: (a) build PyTorch from source with `TORCH_CUDA_ARCH_LIST="5.2"`, or (b) downgrade to PT 2.4 + CUDA 12.1 (which Maxwell M4000 fully supports), or (c) use the JIT PTX compile path (slow first-run, fine after). **This is a precondition that should be empirically verified in Phase 0**, not assumed. If false, Phase 1 gate ("≤ 120 s on M4000") cannot even be measured.

**C2. ORT 1.18 GPU + CUDA 12.6 + sm_52.** The C# `Microsoft.ML.OnnxRuntime.Gpu 1.18.0` package ships against CUDA 11.8 (1.18 line) or CUDA 12.x (1.19+). On Maxwell + CUDA 12.6, you are committing to a *combination ORT does not test*. Risks: (a) some kernel falls back to CPU silently (you get correctness but no speedup), (b) IOBinding "BindOutputToDevice" path triggers a known regression where ORT allocates a transient host buffer anyway. **Add a Phase 2 sub-gate: prove `Run` actually executes on GPU via `nvidia-smi dmon` evidence captured in the same commit as the code change.**

**C3. CUDA Graph capture (§3.7 row 6.3) on Maxwell.** CUDA Graph capture for ORT requires `cudnn_conv_algo_search != EXHAUSTIVE` (search is non-deterministic; capture fails). It also requires fixed shapes — your inference batch is exactly 90 once, but a future training-time eval might hit B=1,884. **State explicitly: graph capture is for the holdout eval call only. Add a non-graph code path for ad-hoc inference.**

**C4. Action 1.2 (materialize VIEW to Parquet at ingest time) introduces a staleness contract.** The VIEW joins `feature_vectors` (9,418 rows) with `pinn_expanded_flat` (1,974 rows). If a Tier-2/3 ingest runs and updates `pinn_expanded_flat` but the snapshot Parquet isn't rebuilt, training silently uses stale features. **Tie the snapshot's lifetime to a content hash of both source tables and check-and-rebuild on every train, or wire it into the ingest scripts that mutate the source.** Currently the doc has no such hook.

**C5. ILGPU-emitted PTX path on M4000 driver 582.16.** Driver 582.16 is the recent Studio driver. ILGPU 1.5 emits PTX targeting up to compute_75 unless capped; needs `Context.Builder().Cuda(...).Architecture(CudaArchitecture.SM_52)`. **Add as a row in §9 risks** — silent wrong-PTX-target compiles with no error and runs ~50 % slower.

**C6. Pinned memory exhaustion.** §3.4 row 3.1 allocates pinned staging "36 KB × B". For B=64 that's 2.3 MB, fine. For a hypothetical full-corpus eval (B=1,884) it's 68 MB pinned — Windows pinned-memory ceiling is per-process and OS-set; 68 MB is OK, but if the model is later run alongside other CUDA workloads you can OOM the pinned pool with no clear error message. **Document the per-allocation upper bound and the failure mode.**

**C7. `key_padding_mask` clone in `SeqEncoder.forward` (line 234).** The current model does an `if all_masked.any(): clone()`. This is a host-conditional inside the forward graph — it breaks CUDA Graph capture (graphs require unconditional control flow). **The execution mapping never mentions this.** Either (a) make the masking unconditional (fix-up always runs), or (b) accept that Phase 2's CUDA Graph gate cannot ship without a model code change.

---

## D. Expected performance impact — sober estimate

The doc implicitly suggests "training drops from 15 min to ~6 s; inference 90-event batch from O(seconds) to <200 ms". My recomputation:

| Phase | Doc claim | My estimate (90 % CI) | Why |
|---|---|---|---|
| Phase 1 (training on GPU) | 6 s pure compute, ≤120 s wall | **45–180 s wall** | FLOPs ~10× under-counted; small-batch launch-overhead-bound on Maxwell at B=64 |
| Phase 2 (90-event holdout inference, GPU) | ≤200 ms | **80–250 ms** | 1.5 M params + 4 inputs + 3 outputs + IOBinding setup; CPU baseline is probably ~300–500 ms, so 2–4× speedup is realistic, not 10× |
| Phase 3 (eliminate Arrow + double Parquet) | ≥5× SQLite→Arrow | **3–8×** | Realistic; row-wise managed allocations + Builder.Append is genuinely the hot spot, dominates over actual disk read |
| Phase 4 (ODE → ILGPU) | ≥10× physics-loss compute | **2–5×** if PT GPU is well-vectorized; ≥10× only if Phase 1 stays on CPU | The Python loop in `physics_loss.py:217` is a Python-level for-loop — once on GPU, 100 iterations of vectorized `torch` ops on B×100 tensors is fast. ILGPU rewrite gives marginal extra unless the loop is fused. |
| Phase 5 (NativeAOT + SIMD) | ≤2 s cold-start | **1.5–4 s** depending on AOT trim warnings and ORT init |
| Phase 6 (G7 quality gate) | MAE ≤ 6 h | **No change from optimization** | HPC work changes wall clock, not loss. Conflating perf gate with quality gate is the riskiest framing in the plan. |

**Total realistic end-state for the G7 round-trip (train + holdout eval):** current ≈ 18–20 min wall → optimized **≈ 90–240 s wall**. That's **5–13× speedup**, not 50×+ as the cumulative numbers in the doc would suggest. Still well worth doing.

**Most important caveat:** none of this changes MAE. The doc's framing — "HPC plan gates G7 advancement" — is a process choice, not a technical necessity. If MAE is the user-facing goal and current G6 baseline is 30 h vs target 6 h, the binding constraint is **model + data quality, not throughput**. A 10× faster training loop that still produces 30 h MAE is still a fail. **Recommend: re-frame as "HPC plan unblocks G7 *iteration speed*; G7 quality is gated by model architecture + Tier 2/3 features".**

---

## E. Ratification checklist (HPC validation gate — blocks G7)

These ten items must be addressed (commit, file, or measurement) before G7 hyperparameter tuning begins. **Items are grouped into four sequential stages; each stage has a validation gate. Skipping ahead is forbidden — later items depend on earlier measurements.** Total estimated effort: ~3 days.

The authoritative copy of this checklist lives here; `NEURAL_ENSEMBLE_PLAN.md` §G6.5 mirrors it for gate-board visibility. Update both when ticking boxes.

---

### Stage 1 — Hardware feasibility (~½ day, load-bearing)

Do these first. If either fails, the whole HPC plan re-plans around a different toolchain — no point measuring a baseline against hardware that can't run the target kernels.

- [x] **E6.** ✅ PASS (2026-04-18). PyTorch 2.5.1+cu121 executes on M4000 sm_52: device_capability=(5,2), 96–98% GPU util, finite loss, 107 ms/iter fwd+bwd. PyTorch 2.7 dropped sm_52 — **pinned to 2.5.1+cu121** in `python/requirements.txt`. See `docs/HPC_STAGE1_HW_FEASIBILITY.md`.
- [x] **E7.** ✅ PASS (2026-04-18, with graph surgery). Original G6 ONNX fails on ORT CUDA EP (`CUDNN_STATUS_EXECUTION_FAILED_CUDART` on seq_encoder ReduceSum; cuDNN ReduceTensor is broken on sm_52). **MatMul graph surgery** (replace 3 ReduceSum nodes with MatMul+Squeeze) routes reductions through cuBLAS GEMM, which works on Maxwell. Patched model: **25.3 ms for N=90 (6.56× CPU EP), max parity diff 6.1e-05, GPU 678 MiB / 51 W active**. Einsum replacement does NOT work — ORT's CUDA Einsum lowers to the same cuDNN reduction path. See `docs/HPC_STAGE1_HW_FEASIBILITY.md` §E7.

**Stage 1 validation gate:**
- [x] `torch.cuda.is_available() == True` and `get_device_capability(0) == (5, 2)` captured.
- [x] PyTorch fwd+bwd on `TftPinnModel` completes without NaN and shows >0 % GPU utilization.
- [x] ORT CUDA EP holdout inference shows >0 % GPU utilization and ≤ 1e-4 parity vs CPU EP. ← **PASS with MatMul-surgery ONNX** (25.3 ms, 678 MiB, 6.1e-05 parity).
- [x] Toolchain versions pinned and committed. `torch==2.5.1` + `onnxruntime-gpu==1.20.1` in `python/requirements.txt`; `Microsoft.ML.OnnxRuntime.Gpu 1.20.1` in `Directory.Packages.props` (all 3 csproj refs updated). Build: 0 warnings, 0 errors.
- [x] `docs/HPC_STAGE1_HW_FEASIBILITY.md` committed.

If this gate fails: stop, document in `NEURAL_ENSEMBLE_PLAN.md` §5, do not proceed to Stage 2.

---

### Stage 2 — Baseline measurement (~1 day, depends on Stage 1)

Only meaningful once the GPU path is confirmed. Produces the single source of truth every downstream number compares against.

- [ ] **E1.** `docs/HPC_BASELINE_MEASUREMENTS.md` committed with: (a) end-to-end G6 round-trip wall clock on both CPU path and GPU path, (b) per-stage breakdown for T1–T11, P1–P9, I1–I6 (from `HPC_OPTIMIZATION_PLAN.md` §2), (c) `nvidia-smi dmon -s ucvmet` capture during the training window, (d) measured counts for the §6 boundary-events table (managed allocs from allocation profiler; CUDA driver calls from Nsight Systems or equivalent), (e) at least one re-run establishing noise floor.

**Stage 2 validation gate:**
- [ ] Per-stage times sum to within 5 % of end-to-end (accounting complete).
- [ ] Baseline re-run within documented noise floor (reproducible).
- [ ] Boundary-events measurement method named in the file (tool + command).
- [ ] File committed; commit hash recorded next to this checkbox.

---

### Stage 3 — Document corrections (~½ day, depends on Stage 2)

- [ ] **E2.** Param count corrected in `HPC_EXECUTION_MAPPING.md` §1.4 (≈ 1.5 M, not 498 K; see §B1 of this review for breakdown). §1.5 Adam state resized (≈ 12 MB not 4 MB). §1.7 "Total GPU memory resident" recomputed for B=64 train and B=90 infer.
- [ ] **E3.** FLOPs and Phase 1 gate corrected in `HPC_OPTIMIZATION_PLAN.md` §3 (≈ 130 TFLOPs total over 200 epochs including backward; gate ≤ 180 s at B≥256, ≤ 300 s at smaller B). Strike "6 s" claim. Cite Stage 1 measured fwd+bwd time where possible.
- [ ] **E4.** `HPC_EXECUTION_MAPPING.md` §6 cross-boundary table rebuilt with two distinct columns: (a) managed-heap allocations, (b) CUDA driver calls + syscalls. "Before" column populated from Stage 2 E1 measurements, not derivations.
- [ ] **E5.** f16 inference path in §1.4 "dtype (infer)" column **struck** unless Stage 1 E7 evidence shows ORT 1.18 f16 is faster than f32 on sm_52 (unlikely per §B4). If struck, update §1.4 total infer-memory rows to use f32.

**Stage 3 validation gate:**
- [ ] Every numeric claim in `HPC_EXECUTION_MAPPING.md` §1.4, §1.5, §1.7, §6 cross-references Stage 2 measurement or direct code enumeration.
- [ ] Every Phase gate in `HPC_OPTIMIZATION_PLAN.md` §8 compatible with measured Maxwell throughput — no gate demands an impossible wall time.
- [ ] `git diff` reviewed; no stale numbers remain.

---

### Stage 4 — Process + framing (~½ day, depends on Stage 3)

No code. Durable rules future sessions will inherit.

- [ ] **E8.** Action 1.2 (VIEW → Parquet snapshot) hash-based staleness check designed in `HPC_EXECUTION_MAPPING.md` §3.1: snapshot filename embeds SHA256 of `(feature_vectors.row_count, pinn_expanded_flat.row_count, MAX(launch_time))`; training scripts validate hash before reading and rebuild on mismatch. Hash construction and validation-function location named in the doc.
- [ ] **E9.** Per-action "Parity gate" column added to `HPC_EXECUTION_MAPPING.md` §3 ledger. For each rewrite action: (a) reference Python/SQL implementation, (b) numerical tolerance (1e-5 SIMD, 1e-4 GPU), (c) test file path asserting parity. Novel actions marked "N/A (novel)".
- [ ] **E10.** Framing language corrected across `HPC_OPTIMIZATION_PLAN.md`, `HPC_EXECUTION_MAPPING.md`, and `CLAUDE.md`: **"HPC plan unblocks G7 iteration speed; G7 quality (MAE ≤ 6h) depends on model architecture + Tier 2/3 features, not throughput."** CLAUDE.md already corrected — verify both HPC docs match.

**Stage 4 validation gate:**
- [ ] Staleness check (E8) has written hash construction + named validation function.
- [ ] Parity-gate column populated for every rewrite action; novel actions marked N/A.
- [ ] `grep`-level check: no doc implies HPC improves MAE.

---

**Closure rule:** When all four stage gates pass, append a session-log row to `NEURAL_ENSEMBLE_PLAN.md` §4 with the closure commit hash; update §2 to re-mark G6 ✅ and G6.5 ✅; update §1 current gate to G7; sync `memory/reference_hpc_redteam_review.md` and `memory/project_neural_ensemble_plan.md`. **Only then** may G7 hyperparameter tuning begin.

**Forbidden during G6.5:** G7 hyperparameter sweeps, "preview" holdout runs, or any multi-hour wall-clock training loop. The entire point of G6.5 is that the next wall-clock run is cheap.

---

## F. What's right and shouldn't change

- The seven-family variable taxonomy (VO-A through VO-G) is excellent and reusable. Keep it.
- The silo-rule formulation in §2 is the right discipline; mirrors what worked for PFSS.
- u8 masks (§5 rule 1), FrozenDictionary (rule 3), pre-baked PE (rule 4), `stackalloc` ≤ 1 KB (rule 9), `ArrayPool` (rule 10) are all correct and high-value low-risk wins.
- The decision tree in §7 is the kind of artifact that scales — future contributors can apply it without re-reading the whole doc.
- The worked example in §8 is the right shape; replicate that format for at least the train-step path too.
- Pinning toolchain versions in §3 of the strategy doc (CUDA 12.6, cuDNN 8.9, ORT 1.18) is correct in spirit even if the specific version may need bumping after Phase 0.

**Bottom line:** ratify after items E1–E10 are addressed. The mapping discipline is sound; the headline numbers need a calculator pass; and the gate-of-G7 framing is over-promising on what HPC alone can deliver.

---

## G. Status changes triggered by this review

1. **G6 reverts from ✅ to ◐** in `docs/NEURAL_ENSEMBLE_PLAN.md` §2. G6 closure was technically structural (round-trip ran end-to-end), but the optimization plan it depends on is unratified. Mark G6 complete only after this gate passes.
2. **New gate: HPC validation (this document, §E).** Sits between G6 and G7. Blocks all G7 work.
3. **`CLAUDE.md`** updated with a pointer to this file under the "Active Phase / HPC optimization" section so every agent sees it at session start.

---

## H. Cross-references

- [`docs/HPC_OPTIMIZATION_PLAN.md`](HPC_OPTIMIZATION_PLAN.md) — strategy doc under review.
- [`docs/HPC_EXECUTION_MAPPING.md`](HPC_EXECUTION_MAPPING.md) — tactical ledger under review.
- [`docs/NEURAL_ENSEMBLE_PLAN.md`](NEURAL_ENSEMBLE_PLAN.md) — gate board updated by this review.
- [`memory/reference_hpc_gpu_optimization_playbook.md`](../memory/reference_hpc_gpu_optimization_playbook.md) — prior PFSS GPU lessons; correctness-before-timing discipline applied here.
- [`memory/reference_g6_baseline.md`](../memory/reference_g6_baseline.md) — DO-NOT-OVERWRITE baseline artifacts for perf comparison.
- `python/tft_pinn_model.py` — authoritative param/shape source used in §B1 recomputation.
- `python/physics_loss.py` — authoritative ODE-loop source used in §D Phase 4 estimate.
