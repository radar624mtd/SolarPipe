"""
G6.5 Stage 2 — E1 GPU round-trip benchmark.

Measures wall-clock time for each stage of the G6 pipeline:
  T-sidecar : gRPC sidecar startup
  T-train   : TFT+PINN training (200 epochs, GPU)
  T-export  : ONNX export via ExportOnnx gRPC call
  T-infer   : C# OnnxAdapter holdout predict (90 events, CUDA EP)

Usage:
  python scripts/benchmark_g6_gpu.py [--runs N] [--skip-train]

Outputs per-run timing JSON and nvidia-smi dmon capture.
"""
import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).parent.parent
PYTHON = r"C:/Users/radar/AppData/Local/Programs/Python/Python312/python.exe"
SIDECAR = REPO / "python/solarpipe_server.py"
CONFIG = REPO / "configs/neural_ensemble_v1.yaml"
DMON_LOG = REPO / "logs/dmon_train.log"
RESULTS_JSON = REPO / "logs/g6_benchmark_results.json"
SIDECAR_PORT = int(os.environ.get("SOLARPIPE_SIDECAR_ADDRESS", "http://localhost:50051").split(":")[-1])
DOTNET_ARGS_BASE = ["dotnet", "run", "--project", str(REPO / "src/SolarPipe.Host"), "--no-build", "--"]


def wait_for_port(port: int, timeout: int = 60) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            s = socket.create_connection(("localhost", port), timeout=2)
            s.close()
            return True
        except OSError:
            time.sleep(1)
    return False


def start_dmon(out_path: Path) -> subprocess.Popen:
    out_path.parent.mkdir(exist_ok=True)
    fh = open(out_path, "w")
    flags = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
    return subprocess.Popen(
        ["nvidia-smi", "dmon", "-s", "ucvmet", "-d", "1"],
        stdout=fh, stderr=subprocess.DEVNULL, creationflags=flags,
    )


def stop_dmon(proc: subprocess.Popen) -> None:
    try:
        proc.terminate()
        proc.wait(timeout=5)
    except Exception:
        proc.kill()


def parse_dmon(path: Path) -> dict:
    sm, mem, pwr = [], [], []
    try:
        for line in path.read_text().splitlines():
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            if len(parts) < 6:
                continue
            try:
                sm.append(int(parts[1]))
                mem.append(int(parts[2]))
                pwr.append(int(parts[5]))
            except (ValueError, IndexError):
                continue
    except FileNotFoundError:
        pass

    def s(lst: list) -> dict:
        return {"avg": round(sum(lst) / len(lst), 1), "peak": max(lst)} if lst else {}

    return {"sm_util_pct": s(sm), "mem_util_pct": s(mem), "power_w": s(pwr), "n_samples": len(sm)}


def run_cmd(cmd: list, label: str) -> tuple[float, str, int]:
    print(f"  -> {label}", flush=True)
    t0 = time.perf_counter()
    r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    elapsed = round(time.perf_counter() - t0, 2)
    status = "OK" if r.returncode == 0 else f"ERROR rc={r.returncode}"
    print(f"     {label} — {elapsed}s [{status}]", flush=True)
    if r.returncode != 0:
        print(r.stderr[-1500:], flush=True)
    return elapsed, r.stdout, r.returncode


def run_single(run_idx: int, n_runs: int, skip_train: bool) -> dict:
    print(f"\n{'='*60}", flush=True)
    print(f"=== RUN {run_idx}/{n_runs} {'(skip-train mode)' if skip_train else ''} ===", flush=True)
    timings: dict = {}

    # Step 1: start sidecar
    print(f"\nstep 1 of 4 — start gRPC sidecar [starting]", flush=True)
    slog = open(REPO / "logs/benchmark_sidecar.log", "w")
    t0 = time.perf_counter()
    sidecar = subprocess.Popen([PYTHON, str(SIDECAR), "--port", str(SIDECAR_PORT)],
                                cwd=REPO, stdout=slog, stderr=slog)
    ok = wait_for_port(SIDECAR_PORT, timeout=30)
    timings["sidecar_startup_s"] = round(time.perf_counter() - t0, 2)
    if not ok:
        print("  [ERROR] sidecar did not start within 30s", flush=True)
        sidecar.terminate()
        return {"run": run_idx, "error": "sidecar_timeout"}
    print(f"step 1 of 4 — sidecar ready [{timings['sidecar_startup_s']}s]", flush=True)

    try:
        if not skip_train:
            # Step 2: train with dmon
            print(f"\nstep 2 of 4 — train TFT+PINN 200 epochs GPU [starting]", flush=True)
            dmon = start_dmon(DMON_LOG)
            time.sleep(0.5)
            t_train, _, rc_train = run_cmd(
                DOTNET_ARGS_BASE + ["train", "--config", str(CONFIG)], "dotnet train")
            stop_dmon(dmon)
            timings["train_s"] = t_train
            timings["train_dmon"] = parse_dmon(DMON_LOG)
            timings["train_ok"] = rc_train == 0
            print(f"step 2 of 4 — train [{'OK' if rc_train==0 else 'ERROR'}]", flush=True)
        else:
            timings["train_s"] = None
            timings["train_dmon"] = {}
            timings["train_ok"] = None
            print(f"\nstep 2 of 4 — train [SKIPPED]", flush=True)

        # Step 3: ONNX export (uses last trained model stored by sidecar)
        # The ExportOnnx gRPC path is triggered via the C# CLI export-onnx command if it exists;
        # otherwise we time the Python-side export directly.
        print(f"\nstep 3 of 4 — ONNX export + MatMul surgery [starting]", flush=True)
        t_export, _, rc_export = run_cmd(
            DOTNET_ARGS_BASE + ["export-onnx", "--config", str(CONFIG)], "dotnet export-onnx")
        if rc_export != 0:
            # export-onnx not yet a CLI verb; record as N/A
            print("  [INFO] export-onnx CLI not implemented — timing from sidecar ExportOnnx RPC directly", flush=True)
            t_export_py, _, _ = run_cmd(
                [PYTHON, "-c",
                 "import time, sys; sys.path.insert(0, str('python')); "
                 "from solarpipe_server import _export_tft_pinn_onnx; "
                 "t0=time.perf_counter(); "
                 "_export_tft_pinn_onnx('tft_pinn_943e0a87', 'models/baselines/g6_tft_pinn_943e0a87', "
                 "'models/baselines/g6_tft_pinn_943e0a87/model_bench.onnx', 17); "
                 "print(f'export_s={time.perf_counter()-t0:.2f}')"],
                "python export-onnx direct")
            timings["export_s"] = t_export_py
            timings["export_method"] = "python_direct"
        else:
            timings["export_s"] = t_export
            timings["export_method"] = "dotnet_cli"
        print(f"step 3 of 4 — ONNX export [{timings['export_s']}s]", flush=True)

        # Step 4: holdout predict (CUDA EP via use_cuda_ep: true in YAML)
        print(f"\nstep 4 of 4 — holdout predict CUDA EP [starting]", flush=True)
        t_pred, out_pred, rc_pred = run_cmd(
            DOTNET_ARGS_BASE + ["predict", "--config", str(CONFIG)], "dotnet predict")
        timings["predict_s"] = t_pred
        timings["predict_ok"] = rc_pred == 0
        print(f"step 4 of 4 — predict [{'OK' if rc_pred==0 else 'ERROR'}]", flush=True)

    finally:
        sidecar.terminate()
        try:
            sidecar.wait(timeout=10)
        except Exception:
            sidecar.kill()

    valid = [v for v in [
        timings.get("sidecar_startup_s", 0),
        timings.get("train_s") or 0,
        timings.get("export_s") or 0,
        timings.get("predict_s") or 0,
    ] if v]
    timings["total_s"] = round(sum(valid), 2)

    print(f"\n--- Run {run_idx} summary ---", flush=True)
    print(json.dumps({k: v for k, v in timings.items() if k != "train_dmon"}, indent=2), flush=True)
    return {"run": run_idx, **timings}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=2, help="Number of benchmark runs (default 2)")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip training (measure export+predict only, using cached model)")
    args = parser.parse_args()

    print("=== G6 GPU round-trip benchmark start ===", flush=True)
    results = [run_single(i, args.runs, args.skip_train) for i in range(1, args.runs + 1)]
    data = {"benchmark": "G6_gpu_roundtrip", "runs": results}
    Path(RESULTS_JSON).write_text(json.dumps(data, indent=2))
    print(f"\n=== G6 GPU round-trip benchmark done — {RESULTS_JSON} ===", flush=True)


if __name__ == "__main__":
    main()
