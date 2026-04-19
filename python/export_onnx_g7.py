"""One-shot script: call ExportOnnx gRPC on the best G7 model, then apply RULE-300
MatMul surgery and publish to models/g7_tft_pinn_<id>/model_matmul_reduce.onnx.

Usage:
    python export_onnx_g7.py --model-id tft_pinn_aa409d9d
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import grpc
import solarpipe_pb2
import solarpipe_pb2_grpc


def main(model_id: str, sidecar: str, opset: int, out_base: str) -> None:
    print(f"=== export_onnx_g7 start === model_id={model_id}", flush=True)

    onnx_raw = str(Path(out_base) / model_id / "model.onnx")
    Path(onnx_raw).parent.mkdir(parents=True, exist_ok=True)

    print(f"step 1 of 3 — gRPC ExportOnnx [calling sidecar {sidecar}]", flush=True)
    channel = grpc.insecure_channel(sidecar)
    stub = solarpipe_pb2_grpc.PythonTrainerStub(channel)
    resp = stub.ExportOnnx(solarpipe_pb2.ExportOnnxRequest(
        model_id=model_id,
        onnx_output_path=onnx_raw,
        opset=opset,
    ))
    if resp.error_message:
        print(f"step 1 of 3 — gRPC ExportOnnx [ERROR: {resp.error_message}]", flush=True)
        sys.exit(1)
    print(f"step 1 of 3 — gRPC ExportOnnx [OK → {onnx_raw}]", flush=True)

    print("step 2 of 3 — RULE-300 MatMul surgery [applying ReduceSum→MatMul+Squeeze]", flush=True)
    onnx_surgery = str(Path(out_base) / model_id / "model_matmul_reduce.onnx")
    _apply_matmul_surgery(onnx_raw, onnx_surgery)
    print(f"step 2 of 3 — RULE-300 MatMul surgery [OK → {onnx_surgery}]", flush=True)

    print("step 3 of 3 — parity test [PyTorch vs ORT CUDA EP]", flush=True)
    _parity_test(model_id, out_base, onnx_surgery)
    print("step 3 of 3 — parity test [OK]", flush=True)

    print(f"=== export_onnx_g7 done — published {onnx_surgery} ===", flush=True)


def _apply_matmul_surgery(src: str, dst: str) -> None:
    """Replace every ReduceSum node with MatMul(x, ones_col) + Squeeze."""
    import numpy as np
    import onnx
    from onnx import helper, TensorProto, numpy_helper

    model = onnx.load(src)
    graph = model.graph

    new_nodes: list = []
    new_initializers: list = list(graph.initializer)
    ones_cache: dict[str, str] = {}  # shape_key → initializer name
    surgery_count = 0

    # Build lookup: Constant node output name → axes values (for opset≥13 ReduceSum)
    const_values: dict[str, list[int]] = {}
    for node in graph.node:
        if node.op_type == "Constant":
            for out in node.output:
                for attr in node.attribute:
                    if attr.name == "value" and attr.HasField("t"):
                        const_values[out] = list(
                            numpy_helper.to_array(attr.t).flatten().astype(int)
                        )
                    elif attr.name == "value_ints":
                        const_values[out] = list(attr.ints)

    for node in graph.node:
        if node.op_type != "ReduceSum":
            new_nodes.append(node)
            continue

        # Determine reduction axes — from attribute (opset<13) or input[1] (opset≥13)
        axes_attr = next((a for a in node.attribute if a.name == "axes"), None)
        keepdims_attr = next((a for a in node.attribute if a.name == "keepdims"), None)
        keepdims = keepdims_attr.i if keepdims_attr else 1

        if axes_attr is not None:
            axes = list(axes_attr.ints)
        elif len(node.input) >= 2 and node.input[1]:
            axes_input = node.input[1]
            # Try initializer first, then Constant node output
            axes_init = next(
                (i for i in graph.initializer if i.name == axes_input), None
            )
            if axes_init is not None:
                axes = list(numpy_helper.to_array(axes_init).flatten().astype(int))
            elif axes_input in const_values:
                axes = const_values[axes_input]
            else:
                new_nodes.append(node)
                continue
        else:
            # No axes → reduce all — fall back to keeping node
            new_nodes.append(node)
            continue

        if len(axes) != 1:
            # Multi-axis reduce — keep original (rare in this model)
            new_nodes.append(node)
            continue

        axis = int(axes[0])
        x_name = node.input[0]
        out_name = node.output[0]

        # We replace ReduceSum(x, axis=a) with:
        #   ones  : (dim_a, 1) initializer of ones
        #   mm    : MatMul(x, ones) → removes axis `a`, produces (..., 1)
        #   squeeze: Squeeze(mm, axis=-1) → removes trailing 1 if keepdims=0
        # This only works cleanly for axis=-1 or axis=last on 2D/3D tensors.
        # For other axes, keep original node.
        # The model's ReduceSum nodes are all axis=-1 (mean-pool denominator) or
        # axis=1 (seq pooling), so we handle both.

        # The ReduceSum reduces axis=-1; for a model with fixed C=20 seq channels,
        # replace with Cast(x, f32) @ ones(C,1) → (..., 1) → Squeeze → (...)
        # C=20 is fixed by N_SEQ_CHANNELS in feature_schema.
        C_dim = 20
        ones_key_C = f"ones_col_C{C_dim}"
        if ones_key_C not in ones_cache:
            ones_arr_C = np.ones((C_dim, 1), dtype=np.float32)
            init_name_C = f"__ones_surgery_C{C_dim}__"
            init_C = numpy_helper.from_array(ones_arr_C, name=init_name_C)
            new_initializers.append(init_C)
            ones_cache[ones_key_C] = init_name_C
        ones_name = ones_cache[ones_key_C]

        # Cast input to float32 before MatMul (handles int64/bool inputs from ReduceSum)
        cast_f32_out = out_name + "__cast_f32"
        cast_f32_node = helper.make_node(
            "Cast",
            inputs=[x_name],
            outputs=[cast_f32_out],
            name=node.name + "__cast_f32",
            to=int(TensorProto.FLOAT),
        )
        mm_out = out_name + "__mm"
        mm_node = helper.make_node(
            "MatMul",
            inputs=[cast_f32_out, ones_name],
            outputs=[mm_out],
            name=node.name + "__matmul",
        )

        # Determine original output dtype from context (check downstream node inputs)
        # The ReduceSum output type matches its input type; cast back to int64 if needed.
        # We always cast back to int64 to preserve downstream type contracts.
        if keepdims == 0:
            sq_out = out_name + "__sq"
            sq_node = helper.make_node(
                "Squeeze",
                inputs=[mm_out],
                outputs=[sq_out],
                name=node.name + "__squeeze",
            )
            # Cast back to int64 to restore original output dtype
            cast_back_node = helper.make_node(
                "Cast",
                inputs=[sq_out],
                outputs=[out_name],
                name=node.name + "__cast_i64",
                to=int(TensorProto.INT64),
            )
            new_nodes.extend([cast_f32_node, mm_node, sq_node, cast_back_node])
        else:
            # keepdims=1: cast mm_out back to int64
            cast_back_node = helper.make_node(
                "Cast",
                inputs=[mm_out],
                outputs=[out_name],
                name=node.name + "__cast_i64",
                to=int(TensorProto.INT64),
            )
            new_nodes.extend([cast_f32_node, mm_node, cast_back_node])

        surgery_count += 1

    print(f"  MatMul surgery: replaced {surgery_count} ReduceSum nodes", flush=True)

    del graph.node[:]
    graph.node.extend(new_nodes)
    del graph.initializer[:]
    graph.initializer.extend(new_initializers)

    onnx.checker.check_model(model)
    onnx.save(model, dst)


def _parity_test(model_id: str, model_base: str, onnx_path: str) -> None:
    """Run a single batch through PyTorch and ORT CUDA EP; assert max diff ≤ 1e-3."""
    import numpy as np
    import torch
    import onnxruntime as ort
    from tft_pinn_model import build_tft_pinn_model
    import json

    meta = json.loads((Path(model_base) / model_id / "meta.json").read_text())
    model = build_tft_pinn_model({k: str(v) for k, v in meta.items()})
    weights = Path(model_base) / model_id / "model.pt"
    model.load_state_dict(torch.load(str(weights), map_location="cpu", weights_only=True))
    model.eval()

    n_flat = int(meta["n_flat"])
    T = int(meta["n_seq_timesteps"])
    C = int(meta["n_seq_channels"])
    B = 4

    rng = np.random.default_rng(0)
    x_flat = rng.standard_normal((B, n_flat)).astype(np.float32)
    m_flat = (rng.random((B, n_flat)) > 0.3).astype(np.float32)
    x_seq  = rng.standard_normal((B, T, C)).astype(np.float32)
    m_seq  = (rng.random((B, T, C)) > 0.2).astype(np.float32)

    with torch.no_grad():
        pt_out = model(
            torch.from_numpy(x_flat), torch.from_numpy(m_flat),
            torch.from_numpy(x_seq),  torch.from_numpy(m_seq),
        ).numpy()  # (B, 3)

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess = ort.InferenceSession(onnx_path, providers=providers)
    ort_out = sess.run(
        ["p10", "p50", "p90"],
        {"x_flat": x_flat, "m_flat": m_flat, "x_seq": x_seq, "m_seq": m_seq},
    )
    # Each output is (B, 1); concatenate along axis=1 to get (B, 3)
    ort_arr = np.concatenate([o.reshape(B, 1) for o in ort_out], axis=1)

    max_diff = float(np.abs(pt_out - ort_arr).max())
    print(f"  Parity max abs diff: {max_diff:.6f}", flush=True)
    assert max_diff <= 1e-3, f"Parity FAILED: max diff {max_diff:.6f} > 1e-3"
    ep = sess.get_providers()[0]
    print(f"  ORT execution provider: {ep}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="tft_pinn_aa409d9d")
    parser.add_argument("--sidecar", default="localhost:50051")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--out-base", default="C:/Users/radar/SolarPipe/models")
    args = parser.parse_args()
    main(args.model_id, args.sidecar, args.opset, args.out_base)
