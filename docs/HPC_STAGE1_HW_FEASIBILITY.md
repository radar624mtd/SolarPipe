# HPC Stage 1 — Hardware Feasibility Report

**Date:** 2026-04-18  
**Hardware:** NVIDIA Quadro M4000 (Maxwell, sm_52, 8 GB GDDR5)  
**Toolchain:** Python 3.12, PyTorch 2.5.1+cu121, CUDA 12.1

---

## E6 — PyTorch CUDA feasibility on M4000 sm_52

### E6a — CUDA device probe

```
torch version: 2.5.1+cu121
torch.cuda.is_available(): True
device_capability(0): (5, 2)
device_name: Quadro M4000
```

**Result: ✅ PASS**

### E6b — Forward + backward on TftPinnModel (GPU)

- Batch size: 16  
- Seq shape: (16, 150, 22), flat shape: (16, 105)  
- Optimizer: Adam  
- GPU utilization during run: **96–98%**  
- GPU memory: **806 MiB**  
- GPU power draw: **76–81 W**  
- Loss: finite (no NaN)  
- Throughput: **~107 ms/iter**

**Result: ✅ PASS**

### E6c — CPU forward + backward (GPU must stay flat)

- GPU utilization during CPU run: **19–21%** (idle, OS background only)  
- GPU memory: **305 MiB** (runtime resident only)  
- CPU run was **7.3× slower** than GPU

**Result: ✅ PASS** — CPU and GPU paths are independent; GPU confirmed not used in CPU mode.

**Stage 1 E6 conclusion:** PyTorch 2.5.1+cu121 executes TftPinnModel on M4000 sm_52 with full GPU utilization. The plan's preferred PyTorch 2.7 has dropped sm_52 from official wheels (red-team §C1); **pin to PyTorch 2.5.1+cu121** in `python/requirements.txt`. This satisfies the "2.4 minimum" gate.

---

## E7 — ORT CUDA EP inference on G6 ONNX model

### E7a — ORT package check

- ORT 1.23.2 (`onnxruntime-gpu`): `CUDAExecutionProvider` listed in `get_available_providers()`.

**Result: ✅ PASS** (package present)

### E7b — ORT CUDA EP on G6 ONNX (holdout inference)

**Tested ORT versions:** 1.23.2, 1.18.1, 1.20.1

All three versions fail at runtime with:

```
CUDNN failure 5003: CUDNN_STATUS_EXECUTION_FAILED_CUDART
Node: /inner/seq_encoder/ReduceSum_1
File: reduction_ops.cc:575
Op: cudnnReduceTensor
```

**Root cause:**  
The ONNX model (opset 17) exports `ReduceSum` in the opset-13+ style — axes are a second input tensor (from a `Constant` node), not an attribute. ORT's CUDA EP routes dynamic-axis `ReduceSum` through cuDNN's `cudnnReduceTensor`. cuDNN's implementation of this kernel crashes at runtime on Maxwell (sm_52) with `CUDNN_STATUS_EXECUTION_FAILED_CUDART`.

Three `ReduceSum` nodes are affected (all in `seq_encoder`):
- `/inner/seq_encoder/ReduceSum` — axis=[-1] (over channels; inputs: `m_seq`)
- `/inner/seq_encoder/ReduceSum_1` — axis=[1] (over time; inputs: masked transformer output)
- `/inner/seq_encoder/ReduceSum_2` — axis=[1] (over time; inputs: mask count)

**Attempted workarounds:**
- ORT downgrade (1.18.1, 1.20.1): same failure. ORT 1.18.1 additionally fails to even load `onnxruntime_providers_cuda.dll` (CUDA 11.8 wheel vs CUDA 12.1 runtime).
- `CUDAExecutionProvider` with `cudnn_conv_algo_search=DEFAULT`: same failure.
- TensorRT EP: `onnxruntime_providers_tensorrt.dll` fails to load (TRT runtime not installed).
- ONNX opset 11 downgrade (would put `axes` as attribute): blocked by `LayerNormalization` opset-17 dependency in model.

**Disabling cuDNN investigated:**  
ORT's CUDA EP does not expose a "disable cuDNN" provider option. The `cudnn_conv_algo_search` option controls only convolution algorithm selection, not reduction kernels. Dynamic-axis `ReduceSum` is always routed to `cudnnReduceTensor` regardless of provider options tested — there is no Python/C API flag to override this.

**Graph surgery investigated:**  
Attempted to patch the 3 `ReduceSum` nodes to use `axes` as an attribute (opset-11 style). Both the ONNX checker and ORT reject this because the model is opset 17, where `ReduceSum` does not accept `axes` as an attribute.

Re-export at opset 11 attempted but blocked: `nn.TransformerEncoder` uses `aten::_transformer_encoder_layer_fwd` which only exports at opset 17+.

**Graph surgery attempts (2026-04-18):**

1. **Einsum surgery — FAILED.** Replaced the 3 ReduceSum nodes with `Einsum` equivalents (`btc->bt`, `btd->bd`, `btc->bc`). ONNX checker passed, model loaded on GPU EP, but runtime failed with the **same cuDNN 5003 error at `reduction_ops.cc:801`** — ORT's CUDA Einsum kernel internally dispatches to `cudnnReduceTensor` for reduction-only Einsums. Einsum does NOT escape cuDNN on ORT CUDA EP.

2. **Explicit MatMul surgery — ✅ PASSED.** Replaced each `ReduceSum` with explicit `MatMul(x, ones)` + `Squeeze`:
   - `ReduceSum(x_(B,150,22), axis=-1)` → `MatMul(x, ones_(22,1))` + `Squeeze(-1)` → `(B,150)`
   - `ReduceSum(x_(B,150,128), axis=1)` → `Transpose(0,2,1)` + `MatMul(x, ones_(150,1))` + `Squeeze(-1)` → `(B,128)`
   - `ReduceSum(x_(B,150,1), axis=1)` → `Transpose(0,2,1)` + `MatMul(x, ones_(150,1))` + `Squeeze(1)` → `(B,1)`

   `MatMul` dispatches to cuBLAS GEMM (separate kernel from cuDNN ReductionOps). GPU EP executes cleanly.

### E7b revised result — ✅ PASS (with surgery)

| Metric | CPU EP (orig) | GPU EP (patched) |
|---|---|---|
| N=90 holdout inference | 166.1 ms | **25.3 ms** |
| Speedup | 1.0× | **6.56×** |
| GPU memory | 305 MiB idle | 678 MiB active |
| GPU power | 16 W | 51 W |
| Parity vs CPU EP (max abs diff) | — | **≤ 6.1e-05** (≤ 1e-4 gate) |
| Providers active | CPUExecutionProvider | CUDAExecutionProvider + CPU fallback |

Patched model: `models/baselines/g6_tft_pinn_943e0a87/model_matmul_reduce.onnx`.

**CPU EP baseline (confirmed working):**
- N=90 holdout inference: **160 ms/run** on CPU EP
- GPU stays at 20% idle (OS background)
- Outputs finite and numerically stable

**Result: ✅ PASS via graph surgery — CUDA EP executes the patched ONNX graph on M4000 at 6.56× CPU EP speed with ≤ 6.1e-05 parity.**

The original G6 ONNX cannot run on Maxwell CUDA EP. The **MatMul-surgery variant** (`model_matmul_reduce.onnx`) runs correctly on GPU. Pipeline decision: ONNX export step must apply the MatMul surgery pass before the model is published for inference.

---

## Stage 1 Gate Assessment

| Gate | Status | Notes |
|------|--------|-------|
| `torch.cuda.is_available() == True` and `device_capability == (5, 2)` | ✅ | Confirmed |
| PyTorch fwd+bwd completes without NaN, >0% GPU util | ✅ | 96–98%, 107 ms/iter |
| ORT CUDA EP holdout inference >0% GPU util, ≤1e-4 parity | ✅ | 25.3 ms, 678 MiB, 51 W, max_diff 6.1e-05 (patched ONNX with MatMul surgery) |
| Toolchain versions pinned and committed | ⏳ | See actions below |
| This file committed | ⏳ | Pending commit |

**Stage 1 result: FULL PASS — E6 ✅, E7 ✅ (with MatMul graph surgery).**

---

## Decision and Next Steps

### Decision: Use GPU CUDA EP for inference via MatMul-surgery ONNX

**Actions required before G7:**

1. **Bake MatMul surgery into the ONNX export pipeline.** Add a post-export pass in `python/solarpipe_server.py` (or a standalone script) that replaces the 3 `seq_encoder` ReduceSum nodes with MatMul+Squeeze. The ONNX file published to `models/` must already be patched before C# loads it.
2. **Strike f16 inference path** from `HPC_EXECUTION_MAPPING.md` §1.4 (red-team E5): Maxwell has no native f16 ALU so f32 remains fastest. This is unchanged by E7 success.
3. **Pin toolchain in `python/requirements.txt`:** `torch==2.5.1+cu121`, `onnxruntime-gpu==1.20.1` (CUDA 12 wheel).
4. **Pin `Microsoft.ML.OnnxRuntime.Gpu`** (not CPU-only) in `Directory.Packages.props` for the C# side — matching the Python sidecar's CUDA EP path.
5. **Update `NEURAL_ENSEMBLE_PLAN.md` §G6.5:** mark E7 PASS with MatMul surgery artifact reference.
6. **CUDA Graph capture (§3.7 row 6.3):** revisit in Phase 2 — patched graph now routes through cuBLAS + fixed shapes, which is capture-compatible.

### Revised inference architecture (updates `HPC_OPTIMIZATION_PLAN.md` §6 CC-6)

```
Python sidecar export:
  PyTorch → torch.onnx.export → apply_matmul_reduce_surgery() → model.onnx

C# OnnxAdapter.PredictAsync():
  → InferenceSession(model.onnx, providers=['CUDAExecutionProvider','CPUExecutionProvider'])
  → Run(feeds_f32)  // ~25 ms for N=90 holdout (6.56× speedup)
  → PredictionResult{P10, P50, P90}
```

IOBinding and pinned-memory optimizations from Phase 2 remain worthwhile. Training continues to use PyTorch CUDA directly.

---

*Append-only. Do not edit previous entries — add new findings below this line.*
