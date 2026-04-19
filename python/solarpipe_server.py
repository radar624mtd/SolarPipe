"""
SolarPipe gRPC sidecar server — Phase 4 full implementation.

Supports TFT and NeuralODE model types via PyTorch (when available).
Falls back to stub behaviour if PyTorch is not installed, so the server
can be exercised in CI without a GPU environment.

RULE-060: Lifecycle managed by .NET IHostedService (SidecarLifecycleService).
RULE-061: Server-streaming RPCs for training (StreamTrain).
RULE-062: Binds to 0.0.0.0, not localhost.
RULE-063: Explicit pa.float32() schema at both ends of gRPC channel.
RULE-125: All large arrays transferred via Arrow IPC files, never inline proto.

Usage (from workspace root):
    ${SOLARPIPE_ROOT}/python/.venv/bin/python python/solarpipe_server.py \\
        [--port 50051] [--parent-pid <PID>]

Generate gRPC stubs from proto (once, after proto changes):
    python -m grpc_tools.protoc \\
        -I python/ --python_out=python/ --grpc_python_out=python/ \\
        python/solarpipe.proto
"""

import argparse
import json
import logging
import os
import signal
import sys
import threading
import time
from concurrent import futures
from pathlib import Path

import pyarrow as pa
import pyarrow.ipc as pa_ipc

try:
    import grpc
    from grpc_health.v1 import health, health_pb2, health_pb2_grpc
    import solarpipe_pb2
    import solarpipe_pb2_grpc
    _GRPC_AVAILABLE = True
except ImportError as _grpc_err:
    _GRPC_AVAILABLE = False
    _grpc_err_msg = str(_grpc_err)

try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

# Import full TFT implementation (requires PyTorch + tft_model.py in same dir)
_TFT_TRANSIT_AVAILABLE = False
if _TORCH_AVAILABLE:
    try:
        import sys as _sys
        import os as _os
        _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
        from tft_model import TransitTimeTFT, TFTTrainer  # noqa: E402
        _TFT_TRANSIT_AVAILABLE = True
    except ImportError as _tft_err:
        _tft_transit_err_msg = str(_tft_err)

# Import TFT+PINN ensemble (G3/G4; requires PyTorch + feature_schema + tft_pinn_model)
_TFT_PINN_AVAILABLE = False
if _TORCH_AVAILABLE:
    try:
        from tft_pinn_model import build_tft_pinn_model  # noqa: E402
        from physics_loss import build_pinn_loss  # noqa: E402
        from feature_schema import (  # noqa: E402
            FLAT_COLS as _PINN_FLAT_COLS,
            N_SEQ_CHANNELS as _PINN_N_SEQ_CHANNELS,
            SEQUENCE_CHANNELS as _PINN_SEQ_CHANNELS,
        )
        _TFT_PINN_AVAILABLE = True
    except ImportError as _tft_pinn_err:
        _tft_pinn_err_msg = str(_tft_pinn_err)

# ─── Structured JSON logging to logs/python_latest.json (ADR-010) ────────────

def _make_json_formatter() -> logging.Formatter:
    class _JsonFmt(logging.Formatter):
        def format(self, record: logging.LogRecord) -> str:
            return json.dumps({
                "time": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
                "level": record.levelname,
                "logger": record.name,
                "msg": record.getMessage(),
                "pid": os.getpid(),
            })
    return _JsonFmt()


def _configure_logging(log_dir: str = "logs") -> None:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(log_dir) / "python_latest.json"

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # File handler — structured JSON (ADR-010)
    fh = logging.FileHandler(str(log_path), mode="w", encoding="utf-8")
    fh.setFormatter(_make_json_formatter())
    root.addHandler(fh)

    # Console — minimal, human-readable
    ch = logging.StreamHandler(sys.stderr)
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    ch.setLevel(logging.WARNING)
    root.addHandler(ch)


logger = logging.getLogger("solarpipe.server")

# ─── Schema constants (RULE-063) ──────────────────────────────────────────────

PREDICTION_SCHEMA = pa.schema([pa.field("prediction", pa.float32())])


# ─── Arrow IPC helpers (RULE-125, RULE-063) ───────────────────────────────────

def _read_arrow_ipc(path: str) -> pa.Table:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Arrow IPC file not found: {path!r}")
    with pa_ipc.open_file(pa.memory_map(str(p), "r")) as reader:
        table = reader.read_all()
    # Enforce float32 (RULE-063)
    for col in table.schema:
        if pa.types.is_floating(col.type) and col.type != pa.float32():
            raise TypeError(
                f"Arrow IPC schema violation: column {col.name!r} has type {col.type}, "
                f"expected float32. Use explicit pa.float32() schema at write time (RULE-063)."
            )
    return table


def _write_arrow_ipc(path: str, table: pa.Table) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with pa_ipc.new_file(str(p), table.schema) as writer:
        writer.write_table(table)


# ─── Parent-process heartbeat (RULE-060) ─────────────────────────────────────

def _start_parent_heartbeat(parent_pid: int, poll_interval_s: float = 5.0) -> None:
    """Terminate self if the parent .NET process (parent_pid) dies."""

    def _check() -> None:
        while True:
            time.sleep(poll_interval_s)
            try:
                # os.kill with signal 0 checks process existence without killing it
                os.kill(parent_pid, 0)
            except (ProcessLookupError, PermissionError):
                logger.warning(
                    "Parent process %d no longer exists — self-terminating (RULE-060)",
                    parent_pid,
                )
                os.kill(os.getpid(), signal.SIGTERM)
                break

    t = threading.Thread(target=_check, daemon=True, name="parent-heartbeat")
    t.start()
    logger.info("Parent heartbeat started: monitoring PID %d", parent_pid)


# ─── TFT trainer (requires PyTorch) ──────────────────────────────────────────

class _SimpleTftModel(nn.Module if _TORCH_AVAILABLE else object):
    """Minimal single-layer LSTM as stand-in for full TFT architecture.

    Full TFT implementation (variable selection, gating, temporal attention)
    would require pytorch-forecasting. This stub demonstrates the training
    loop contract and ONNX export pathway.
    """

    def __init__(self, input_size: int, hidden: int = 64) -> None:
        super().__init__()
        if _TORCH_AVAILABLE:
            self.lstm = nn.LSTM(input_size, hidden, batch_first=True)
            self.head = nn.Linear(hidden, 1)

    def forward(self, x):  # type: ignore[override]
        if not _TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available")
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


def _train_tft(
    table: pa.Table,
    feature_columns: list[str],
    target_column: str,
    hyperparameters: dict[str, str],
    model_output_dir: str,
    progress_cb,
    ct_event: threading.Event,
) -> str:
    """Train a TFT model; emit progress via progress_cb(epoch, train_loss, val_loss)."""
    if not _TORCH_AVAILABLE:
        logger.warning("PyTorch not available — using stub TFT trainer")
        return _stub_train(table, "TFT", model_output_dir)

    epochs = int(hyperparameters.get("epochs", "20"))
    lr = float(hyperparameters.get("learning_rate", "1e-3"))
    hidden = int(hyperparameters.get("hidden_size", "64"))

    # Build tensors from Arrow table — replace Arrow nulls (None) with NaN, then filter
    import math
    feat_arrays = [[float("nan") if v is None else float(v)
                    for v in table.column(c).to_pylist()]
                   for c in feature_columns]
    tgt_array = [float("nan") if v is None else float(v)
                 for v in table.column(target_column).to_pylist()]
    n_raw = len(tgt_array)
    valid = [i for i in range(n_raw)
             if not math.isnan(tgt_array[i])
             and all(not math.isnan(feat_arrays[j][i]) for j in range(len(feature_columns)))]
    if not valid:
        raise ValueError("TFT trainer: all rows contain NaN — cannot train")
    feat_arrays = [[fa[i] for i in valid] for fa in feat_arrays]
    tgt_array = [tgt_array[i] for i in valid]
    n = len(tgt_array)

    X = torch.tensor(feat_arrays, dtype=torch.float32).T.unsqueeze(1)  # (n, 1, features)
    y = torch.tensor(tgt_array, dtype=torch.float32).unsqueeze(1)      # (n, 1)

    model = _SimpleTftModel(len(feature_columns), hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    split = max(1, int(0.8 * n))

    for epoch in range(1, epochs + 1):
        if ct_event.is_set():
            logger.info("TFT training cancelled at epoch %d", epoch)
            break

        model.train()
        opt.zero_grad()
        pred = model(X[:split])
        train_loss = loss_fn(pred, y[:split])
        train_loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X[split:]), y[split:]) if split < n else train_loss

        progress_cb(epoch, float(train_loss), float(val_loss))

    # Save model artifact
    import uuid
    model_id = f"tft_{uuid.uuid4().hex[:8]}"
    out_dir = Path(model_output_dir) / model_id
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(out_dir / "model.pt"))
    (out_dir / "meta.json").write_text(
        json.dumps({"model_type": "TFT", "input_size": len(feature_columns),
                    "hidden": hidden, "epochs": epochs})
    )
    logger.info("TFT model saved: %s", model_id)
    return model_id


# ─── TFT_TRANSIT trainer (full sequence-aware TFT — Phase 3+4) ───────────────

def _load_sequences_for_ids(sequences_path: str, activity_ids: list[str],
                             split: str) -> dict:
    """Load pre-launch + in-transit tensors from Parquet for the given activity_ids.

    Returns dict with keys:
        pre_seq:      (N, 150, 20) float32 tensor
        transit_seq:  (N, 72, 20)  float32 tensor  (NaN-padded where shorter)
        transit_mask: (N, 72)      bool tensor
        has_full_omni: (N,)        bool tensor
    """
    import pyarrow.parquet as pq
    import numpy as np

    SEQ_CHANNELS = [
        "Bz_GSM", "flow_speed", "proton_density", "flow_pressure", "AE_nT",
        "Dst_nT", "Kp_x10", "B_scalar_avg", "By_GSM", "electric_field",
        "plasma_beta", "alfven_mach", "sigma_Bz", "sigma_N", "sigma_V",
        "flow_longitude", "flow_latitude", "alpha_proton_ratio", "F10_7_index",
        "Bx_GSE",
    ]
    PRE_LEN = 150
    TRANSIT_LEN = 72
    C = len(SEQ_CHANNELS)  # 20

    seq_file = str(Path(sequences_path) / f"{split}_sequences.parquet")
    tbl = pq.read_table(seq_file)

    aid_index: dict[str, int] = {aid: i for i, aid in enumerate(activity_ids)}
    N = len(activity_ids)

    pre_np = np.full((N, PRE_LEN, C), float("nan"), dtype=np.float32)
    transit_np = np.full((N, TRANSIT_LEN, C), float("nan"), dtype=np.float32)
    has_full_omni = np.zeros(N, dtype=bool)

    aid_col = tbl.column("activity_id").to_pylist()
    window_col = tbl.column("window").to_pylist()
    timestep_col = tbl.column("timestep").to_pylist()
    omni_col = tbl.column("has_full_omni").to_pylist()

    channel_arrays = [tbl.column(ch).to_pylist() for ch in SEQ_CHANNELS]

    for row_i in range(tbl.num_rows):
        aid = aid_col[row_i]
        if aid not in aid_index:
            continue
        n_idx = aid_index[aid]
        w = window_col[row_i]
        ts = int(timestep_col[row_i])
        vals = [channel_arrays[c][row_i] for c in range(C)]

        if w == "pre_launch":
            # timestep in [-150, -1] → array index ts + 150
            t_idx = ts + PRE_LEN
            if 0 <= t_idx < PRE_LEN:
                pre_np[n_idx, t_idx] = [float("nan") if v is None else float(v) for v in vals]
        elif w == "in_transit":
            # timestep in [0, 71]
            if 0 <= ts < TRANSIT_LEN:
                transit_np[n_idx, ts] = [float("nan") if v is None else float(v) for v in vals]
        if omni_col[row_i]:
            has_full_omni[n_idx] = True

    transit_mask = ~np.all(np.isnan(transit_np), axis=-1)  # (N, TRANSIT_LEN)

    return {
        "pre_seq": torch.from_numpy(pre_np),
        "transit_seq": torch.from_numpy(transit_np),
        "transit_mask": torch.from_numpy(transit_mask),
        "has_full_omni": torch.from_numpy(has_full_omni),
    }


def _build_tft_dataset(
    table: pa.Table,
    feature_columns: list[str],
    target_column: str,
    existing_pred_columns: list[str],
    sequences_path: str,
    split: str,
) -> dict:
    """Assemble static + sequence tensors for TFTTrainer.

    Returns dict with keys: static, pre_seq, transit_seq, transit_mask,
    existing_preds, target, activity_ids
    """
    import numpy as np

    activity_ids = [str(v) for v in table.column("activity_id").to_pylist()]
    N = len(activity_ids)

    # Static features
    static_np = np.zeros((N, len(feature_columns)), dtype=np.float32)
    for j, col in enumerate(feature_columns):
        vals = table.column(col).to_pylist()
        static_np[:, j] = [float("nan") if v is None else float(v) for v in vals]

    # Target
    target_vals = table.column(target_column).to_pylist()
    target_np = np.array([float("nan") if v is None else float(v)
                          for v in target_vals], dtype=np.float32)

    # Existing predictions (may be NaN)
    if existing_pred_columns:
        ep_np = np.full((N, len(existing_pred_columns)), float("nan"), dtype=np.float32)
        for j, col in enumerate(existing_pred_columns):
            if col in table.schema.names:
                vals = table.column(col).to_pylist()
                ep_np[:, j] = [float("nan") if v is None else float(v) for v in vals]
        existing_preds = torch.from_numpy(ep_np)
    else:
        existing_preds = None

    # Sequences
    seq_data = _load_sequences_for_ids(sequences_path, activity_ids, split)

    return {
        "static": torch.from_numpy(static_np),
        "pre_seq": seq_data["pre_seq"],
        "transit_seq": seq_data["transit_seq"],
        "transit_mask": seq_data["transit_mask"],
        "existing_preds": existing_preds,
        "target": torch.from_numpy(target_np),
        "activity_ids": activity_ids,
    }


def _train_tft_transit(
    table: pa.Table,
    feature_columns: list[str],
    target_column: str,
    hyperparameters: dict[str, str],
    model_output_dir: str,
    progress_cb,
    ct_event: threading.Event,
) -> str:
    """Train the full sequence-aware TransitTimeTFT.

    Expected hyperparameters:
        sequences_path   — path to directory containing *_sequences.parquet
        existing_preds   — comma-separated column names of existing model predictions
        d_model          — (default 64)
        n_heads          — (default 4)
        n_lstm_layers    — (default 2)
        epochs           — (default 50)
        lr               — (default 1e-3)
        batch_size       — (default 64)
        device           — (default "cuda" if available else "cpu")
    """
    if not _TFT_TRANSIT_AVAILABLE:
        logger.warning("TFT_TRANSIT unavailable — falling back to stub")
        return _stub_train(table, "TFT_TRANSIT", model_output_dir)

    sequences_path = hyperparameters.get(
        "sequences_path",
        os.environ.get("SOLARPIPE_SEQUENCES_PATH", "data/sequences")
    )
    if not Path(sequences_path).exists():
        raise FileNotFoundError(
            f"TFT_TRANSIT: sequences_path not found: {sequences_path!r}. "
            "Set SOLARPIPE_SEQUENCES_PATH or pass sequences_path hyperparameter."
        )

    existing_pred_cols_raw = hyperparameters.get("existing_preds", "")
    existing_pred_cols = [c.strip() for c in existing_pred_cols_raw.split(",") if c.strip()]

    d_model = int(hyperparameters.get("d_model", "64"))
    n_heads = int(hyperparameters.get("n_heads", "4"))
    n_lstm_layers = int(hyperparameters.get("n_lstm_layers", "2"))
    epochs = int(hyperparameters.get("epochs", "50"))
    lr = float(hyperparameters.get("lr", "1e-3"))
    batch_size = int(hyperparameters.get("batch_size", "64"))

    device_str = hyperparameters.get(
        "device",
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    logger.info(
        "TFT_TRANSIT training: n_static=%d n_existing=%d device=%s epochs=%d",
        len(feature_columns), len(existing_pred_cols), device_str, epochs,
    )

    # Determine split from table (use "train" if not present)
    split = "train"
    if "split" in table.schema.names:
        splits = set(table.column("split").to_pylist())
        split = next(iter(splits), "train")

    dataset = _build_tft_dataset(
        table, feature_columns, target_column,
        existing_pred_cols, sequences_path, split
    )

    # Simple 80/20 temporal split for single-call Train (not fold CV)
    N = dataset["target"].shape[0]
    split_idx = max(1, int(0.8 * N))
    train_data = {k: v[:split_idx] if torch.is_tensor(v) else v
                  for k, v in dataset.items()}
    val_data = {k: v[split_idx:] if torch.is_tensor(v) else v
                for k, v in dataset.items()}

    model_kwargs = dict(
        n_static=len(feature_columns),
        n_seq_channels=20,
        n_existing_preds=len(existing_pred_cols),
        d_model=d_model,
        n_heads=n_heads,
        n_lstm_layers=n_lstm_layers,
    )

    trainer = TFTTrainer(
        model_kwargs=model_kwargs,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        device=device_str,
    )

    model = trainer.train_fold(train_data, val_data, progress_cb, ct_event)

    import uuid
    model_id = f"tft_transit_{uuid.uuid4().hex[:8]}"
    out_dir = Path(model_output_dir) / model_id
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(out_dir / "model.pt"))
    (out_dir / "meta.json").write_text(json.dumps({
        "model_type": "TFT_TRANSIT",
        "model_kwargs": model_kwargs,
        "feature_columns": feature_columns,
        "existing_pred_columns": existing_pred_cols,
        "sequences_path": sequences_path,
        "epochs": epochs,
    }))
    logger.info("TFT_TRANSIT model saved: %s", model_id)
    return model_id


def _predict_tft_transit(model_id: str, model_output_dir: str, table: pa.Table) -> pa.Table:
    """Run TFT_TRANSIT inference; returns Arrow table with transit_p10/p50/p90 columns."""
    import numpy as np

    meta_path = Path(model_output_dir) / model_id / "meta.json"
    meta = json.loads(meta_path.read_text())
    model_kwargs = meta["model_kwargs"]
    feature_columns = meta["feature_columns"]
    existing_pred_cols = meta.get("existing_pred_columns", [])
    sequences_path = meta.get("sequences_path",
                              os.environ.get("SOLARPIPE_SEQUENCES_PATH", "data/sequences"))

    model = TransitTimeTFT(**model_kwargs)
    weights_path = Path(model_output_dir) / model_id / "model.pt"
    model.load_state_dict(torch.load(str(weights_path), map_location="cpu", weights_only=True))
    model.eval()

    split = "holdout"
    if "split" in table.schema.names:
        splits = set(table.column("split").to_pylist())
        split = next(iter(splits), "holdout")

    # Build a dummy target (unused for prediction)
    N = table.num_rows
    dummy_target = [0.0] * N
    aug_table = table.append_column(
        pa.field("_dummy_target", pa.float32()),
        pa.array(dummy_target, type=pa.float32())
    ) if "_dummy_target" not in table.schema.names else table

    dataset = _build_tft_dataset(
        aug_table, feature_columns, "_dummy_target",
        existing_pred_cols, sequences_path, split
    )

    trainer = TFTTrainer(model_kwargs=model_kwargs)
    preds = trainer.predict(model, dataset)  # (N, 3)

    p10 = pa.array(preds[:, 0].numpy().astype(np.float32), type=pa.float32())
    p50 = pa.array(preds[:, 1].numpy().astype(np.float32), type=pa.float32())
    p90 = pa.array(preds[:, 2].numpy().astype(np.float32), type=pa.float32())

    return pa.table({
        "transit_p10": p10,
        "transit_p50": p50,
        "transit_p90": p90,
        "prediction": p50,  # gRPC contract: single "prediction" column = P50
    }, schema=pa.schema([
        pa.field("transit_p10", pa.float32()),
        pa.field("transit_p50", pa.float32()),
        pa.field("transit_p90", pa.float32()),
        pa.field("prediction", pa.float32()),
    ]))


# ─── NeuralODE trainer (dynamics network only — RULE-070) ────────────────────

class _NeuralOdeDynamics(nn.Module if _TORCH_AVAILABLE else object):
    """f(y, t, θ): the dynamics network exported to ONNX (RULE-070)."""

    def __init__(self, dim: int = 4, hidden: int = 64) -> None:
        super().__init__()
        if _TORCH_AVAILABLE:
            self.net = nn.Sequential(
                nn.Linear(dim + 1, hidden),
                nn.Tanh(),
                nn.Linear(hidden, dim),
            )

    def forward(self, y, t):  # type: ignore[override]
        if not _TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available")
        if t.dim() == 0:
            t_expand = t.expand(y.shape[0], 1)
        elif t.dim() == 1:
            t_expand = t.unsqueeze(-1)
        else:
            t_expand = t
        return self.net(torch.cat([y, t_expand], dim=-1))


def _train_neural_ode(
    table: pa.Table,
    feature_columns: list[str],
    target_column: str,
    hyperparameters: dict[str, str],
    model_output_dir: str,
    progress_cb,
    ct_event: threading.Event,
) -> str:
    if not _TORCH_AVAILABLE:
        logger.warning("PyTorch not available — using stub NeuralODE trainer")
        return _stub_train(table, "NeuralOde", model_output_dir)

    epochs = int(hyperparameters.get("epochs", "15"))
    lr = float(hyperparameters.get("learning_rate", "1e-3"))
    dim = len(feature_columns)

    dynamics = _NeuralOdeDynamics(dim=dim)
    opt = torch.optim.Adam(dynamics.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    import math
    feat_arrays = [[float("nan") if v is None else float(v)
                    for v in table.column(c).to_pylist()]
                   for c in feature_columns]
    tgt_array = [float("nan") if v is None else float(v)
                 for v in table.column(target_column).to_pylist()]
    n_raw = len(tgt_array)
    valid = [i for i in range(n_raw)
             if not math.isnan(tgt_array[i])
             and all(not math.isnan(feat_arrays[j][i]) for j in range(len(feature_columns)))]
    if not valid:
        raise ValueError("NeuralODE trainer: all rows contain NaN — cannot train")
    feat_arrays = [[fa[i] for i in valid] for fa in feat_arrays]
    tgt_array = [tgt_array[i] for i in valid]
    n = len(tgt_array)
    X = torch.tensor(feat_arrays, dtype=torch.float32).T   # (n, dim)
    y = torch.tensor(tgt_array, dtype=torch.float32).unsqueeze(1)
    t = torch.zeros(n, 1)

    split = max(1, int(0.8 * n))

    for epoch in range(1, epochs + 1):
        if ct_event.is_set():
            logger.info("NeuralODE training cancelled at epoch %d", epoch)
            break

        dynamics.train()
        opt.zero_grad()
        # Simplified: use dynamics output directly as prediction proxy
        pred = dynamics(X[:split], t[:split]).mean(dim=-1, keepdim=True)
        train_loss = loss_fn(pred, y[:split])
        train_loss.backward()
        opt.step()

        dynamics.eval()
        with torch.no_grad():
            val_pred = dynamics(X[split:], t[split:]).mean(dim=-1, keepdim=True)
            val_loss = loss_fn(val_pred, y[split:]) if split < n else train_loss

        progress_cb(epoch, float(train_loss), float(val_loss))

    import uuid
    model_id = f"neural_ode_{uuid.uuid4().hex[:8]}"
    out_dir = Path(model_output_dir) / model_id
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(dynamics.state_dict(), str(out_dir / "dynamics.pt"))
    (out_dir / "meta.json").write_text(
        json.dumps({"model_type": "NeuralOde", "dim": dim, "hidden": 64, "epochs": epochs})
    )
    logger.info("NeuralODE dynamics saved: %s", model_id)
    return model_id


def _export_neural_ode_onnx(model_id: str, model_output_dir: str, onnx_path: str, opset: int) -> None:
    """Export dynamics network only to ONNX (RULE-070)."""
    if not _TORCH_AVAILABLE:
        # Write a valid placeholder so the .NET side can detect success
        Path(onnx_path).parent.mkdir(parents=True, exist_ok=True)
        Path(onnx_path).write_bytes(b"\x00" * 8)
        return

    meta_path = Path(model_output_dir) / model_id / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"NeuralODE meta not found: {meta_path}")

    meta = json.loads(meta_path.read_text())
    dim = meta["dim"]
    hidden = meta.get("hidden", 64)
    dynamics = _NeuralOdeDynamics(dim=dim, hidden=hidden)

    weights = Path(model_output_dir) / model_id / "dynamics.pt"
    dynamics.load_state_dict(torch.load(str(weights), map_location="cpu", weights_only=True))
    dynamics.eval()

    # Export f(y, t) — dynamics network only (RULE-070)
    dummy_y = torch.zeros(1, dim)
    dummy_t = torch.zeros(1, 1)
    Path(onnx_path).parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        dynamics,
        (dummy_y, dummy_t),
        onnx_path,
        opset_version=min(opset if opset > 0 else 20, 20),
        input_names=["y", "t"],
        output_names=["dydt"],
        dynamic_axes={"y": {0: "batch"}, "t": {0: "batch"}, "dydt": {0: "batch"}},
    )
    logger.info("NeuralODE dynamics exported to ONNX: %s (opset=%d)", onnx_path, opset)


# ─── TFT + PINN trainer / predictor / ONNX export (G3/G4/G5) ────────────────

def _pinn_sequences_from_parquet(
    sequences_path: str,
    split: str,
    activity_ids: list[str],
) -> "tuple[torch.Tensor, torch.Tensor]":
    """Load (x_seq, m_seq) pair for the TFT+PINN trainer.

    Rebuilds the (N, T, C) float32 sequences from the Parquet file produced
    by ``scripts/build_pinn_sequences.py``.  NaNs become 0.0 in x_seq with
    mask 0.0 in m_seq, matching TrainingFeaturesDataset semantics.

    Channels missing from the Parquet are padded with zeros and mask=0
    (supports the RULE-213 SuperMAG expansion path without a code change).
    """
    import numpy as np
    import pyarrow.parquet as pq

    seq_file = Path(sequences_path) / f"{split}_sequences.parquet"
    if not seq_file.exists():
        raise FileNotFoundError(f"Sequence parquet not found: {seq_file}")

    tbl = pq.read_table(str(seq_file))
    aid_col      = tbl.column("activity_id").to_pylist()
    window_col   = tbl.column("window").to_pylist()
    timestep_col = tbl.column("timestep").to_pylist()

    # Determine T from the pre_launch timestep range; build index map
    pre_timesteps: list[int] = [
        int(timestep_col[i]) for i in range(tbl.num_rows)
        if window_col[i] == "pre_launch"
    ]
    if not pre_timesteps:
        raise RuntimeError("No pre_launch rows in sequence parquet")
    t_min, t_max = min(pre_timesteps), max(pre_timesteps)
    T = t_max - t_min + 1
    C = _PINN_N_SEQ_CHANNELS

    aid_index: dict[str, int] = {aid: i for i, aid in enumerate(activity_ids)}
    N = len(activity_ids)
    x_np = np.zeros((N, T, C), dtype=np.float32)
    m_np = np.zeros((N, T, C), dtype=np.float32)

    schema_names = set(tbl.schema.names)
    ch_arrays: dict[str, list] = {}
    for ch in _PINN_SEQ_CHANNELS:
        if ch in schema_names:
            ch_arrays[ch] = tbl.column(ch).to_pylist()
        else:
            ch_arrays[ch] = None  # padded with zeros / mask 0

    for row_i in range(tbl.num_rows):
        if window_col[row_i] != "pre_launch":
            continue
        aid = aid_col[row_i]
        if aid not in aid_index:
            continue
        n_idx = aid_index[aid]
        t_idx = int(timestep_col[row_i]) - t_min
        if t_idx < 0 or t_idx >= T:
            continue
        for c_idx, ch in enumerate(_PINN_SEQ_CHANNELS):
            arr = ch_arrays[ch]
            if arr is None:
                continue
            v = arr[row_i]
            if v is None:
                continue
            fv = float(v)
            if fv != fv:   # NaN check
                continue
            x_np[n_idx, t_idx, c_idx] = fv
            m_np[n_idx, t_idx, c_idx] = 1.0

    return torch.from_numpy(x_np), torch.from_numpy(m_np)


def _flat_mask_tensors_from_arrow(
    table: pa.Table,
    feature_columns: list[str],
) -> "tuple[torch.Tensor, torch.Tensor]":
    """Project Arrow table to (x_flat, m_flat) float32 tensors.

    NaN / Arrow null -> x_flat=0.0, m_flat=0.0; observed -> m_flat=1.0.
    Columns must match feature_schema.FLAT_COLS order on the caller side.
    """
    import math
    import numpy as np

    N = table.num_rows
    C = len(feature_columns)
    x_np = np.zeros((N, C), dtype=np.float32)
    m_np = np.zeros((N, C), dtype=np.float32)
    for j, col in enumerate(feature_columns):
        vals = table.column(col).to_pylist()
        for i, v in enumerate(vals):
            if v is None:
                continue
            fv = float(v)
            if math.isnan(fv):
                continue
            x_np[i, j] = fv
            m_np[i, j] = 1.0
    return torch.from_numpy(x_np), torch.from_numpy(m_np)


def _train_tft_pinn(
    table: pa.Table,
    feature_columns: list[str],
    target_column: str,
    hyperparameters: dict[str, str],
    model_output_dir: str,
    progress_cb,
    ct_event: threading.Event,
) -> str:
    """Train the TFT+PINN two-headed ensemble (G3 model, G4 loss).

    Required hyperparameters:
        sequences_path:  dir containing {train,holdout}_sequences.parquet
        activity_ids:    comma-separated list in same order as Arrow rows
                         (caller-supplied; falls back to table[activity_id])

    All model + loss hyperparameters are YAML keys consumed by
    build_tft_pinn_model() and build_pinn_loss().
    """
    if not _TORCH_AVAILABLE or not _TFT_PINN_AVAILABLE:
        logger.warning("TFT+PINN unavailable -- stub train")
        return _stub_train(table, "TFT_PINN", model_output_dir)

    # Check expected feature column order matches feature_schema
    if list(feature_columns) != list(_PINN_FLAT_COLS):
        raise ValueError(
            f"TFT+PINN feature_columns must match feature_schema.FLAT_COLS "
            f"(got {len(feature_columns)} cols, expected {len(_PINN_FLAT_COLS)}; "
            f"first diff at index "
            f"{next((i for i, (a, b) in enumerate(zip(feature_columns, _PINN_FLAT_COLS)) if a != b), 'n/a')})"
        )

    sequences_path = hyperparameters.get("sequences_path", "data/sequences")
    split = hyperparameters.get("split", "train")
    epochs = int(hyperparameters.get("max_epochs", hyperparameters.get("epochs", "10")))
    lr = float(hyperparameters.get("learning_rate", "1e-3"))
    batch_size = int(hyperparameters.get("batch_size", "64"))
    grad_clip = float(hyperparameters.get("gradient_clip_norm", "1.0"))
    seed = int(hyperparameters.get("seed", "42"))

    _device_pref = hyperparameters.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(_device_pref if torch.cuda.is_available() or _device_pref == "cpu" else "cpu")
    logger.info("TFT+PINN device: %s (requested=%s)", device, _device_pref)

    torch.manual_seed(seed)

    # ---- Build x_flat / m_flat / y from Arrow table ------------------------
    import math
    x_flat, m_flat = _flat_mask_tensors_from_arrow(table, feature_columns)

    tgt_raw = table.column(target_column).to_pylist()
    y_list = [float("nan") if v is None else float(v) for v in tgt_raw]
    valid_idx = [i for i, v in enumerate(y_list) if not math.isnan(v)]
    if not valid_idx:
        raise ValueError("TFT+PINN trainer: all targets NaN")
    y = torch.tensor([y_list[i] for i in valid_idx], dtype=torch.float32).unsqueeze(1)
    x_flat = x_flat[valid_idx]
    m_flat = m_flat[valid_idx]

    # ---- Load sequences from Parquet ---------------------------------------
    aid_col = [str(v) for v in table.column("activity_id").to_pylist()] \
        if "activity_id" in table.schema.names else []
    if aid_col:
        aids = [aid_col[i] for i in valid_idx]
    else:
        raise ValueError("TFT+PINN trainer: Arrow table must include activity_id column")

    x_seq, m_seq = _pinn_sequences_from_parquet(sequences_path, split, aids)

    # Drop rows where the sequence is fully unobserved (no info at all)
    seq_obs = m_seq.sum(dim=(1, 2)) > 0
    if not seq_obs.all():
        keep = seq_obs.nonzero(as_tuple=True)[0]
        x_flat = x_flat[keep]
        m_flat = m_flat[keep]
        x_seq  = x_seq[keep]
        m_seq  = m_seq[keep]
        y      = y[keep]

    N = y.size(0)
    if N == 0:
        raise ValueError("TFT+PINN trainer: no rows after sequence obs filter")

    # Ambient wind + initial speed — extract from RAW x_flat before normalization
    cme_speed_idx = _PINN_FLAT_COLS.index("cme_speed_kms")
    sw_speed_idx  = _PINN_FLAT_COLS.index("sw_speed_ambient")
    v0   = torch.where(m_flat[:, cme_speed_idx] > 0,
                       x_flat[:, cme_speed_idx],
                       torch.full((N,), 500.0))
    v_sw = torch.where(m_flat[:, sw_speed_idx] > 0,
                       x_flat[:, sw_speed_idx],
                       torch.full((N,), 400.0))

    # ---- Standardize x_flat (per-column z-score on observed values) ---------
    # Raw physical columns span 24 orders of magnitude — unnormalized input
    # causes gradient explosion and NaN weights within a few epochs.
    feat_mean = torch.zeros(x_flat.shape[1])
    feat_std  = torch.ones(x_flat.shape[1])
    for j in range(x_flat.shape[1]):
        obs_vals = x_flat[m_flat[:, j] > 0, j]
        if obs_vals.numel() > 1:
            feat_mean[j] = obs_vals.mean()
            s = obs_vals.std()
            feat_std[j] = s if s > 1e-8 else 1.0
    x_flat = (x_flat - feat_mean) / feat_std

    # ---- Move all tensors to device -----------------------------------------
    x_flat = x_flat.to(device)
    m_flat = m_flat.to(device)
    x_seq  = x_seq.to(device)
    m_seq  = m_seq.to(device)
    y      = y.to(device)
    v0     = v0.to(device)
    v_sw   = v_sw.to(device)

    # Fixed batch shapes → benchmark mode finds fastest cuDNN kernel per shape.
    # AMP disabled: Maxwell (sm_52) has no native f16 ALU — f32 only (E5).
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    # ---- Model + loss ------------------------------------------------------
    model = build_tft_pinn_model(hyperparameters).to(device)
    loss_fn = build_pinn_loss(hyperparameters)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    # Cosine annealing: smoothly decays lr from lr → lr_min over T_max epochs.
    # eta_min = lr/100 keeps final LR non-zero for fine-grained convergence.
    lr_min = float(hyperparameters.get("lr_min", lr / 100.0))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=epochs, eta_min=lr_min
    )

    split_n = max(1, int(0.8 * N))
    rng = torch.Generator().manual_seed(seed)
    perm = torch.randperm(N, generator=rng)
    idx_train = perm[:split_n]
    idx_val   = perm[split_n:]

    n_batches = max(1, (len(idx_train) + batch_size - 1) // batch_size)

    patience = int(hyperparameters.get("early_stopping_patience", "20"))
    best_val_loss = float("inf")
    best_state: dict | None = None
    epochs_no_improve = 0

    # Physics loss warm-start: ramp lambda_bound/lambda_qorder from 0 → target
    # over `physics_warmup_epochs`. During warmup, the model sees only pinball
    # loss so the quantile head can anchor P50 before physics regularisation fires.
    physics_warmup = int(hyperparameters.get("physics_warmup_epochs", "50"))
    target_lambda_bound  = loss_fn.lambda_bound
    target_lambda_qorder = loss_fn.lambda_qorder

    for epoch in range(1, epochs + 1):
        if ct_event.is_set():
            logger.info("TFT+PINN training cancelled at epoch %d", epoch)
            break

        # Linearly ramp physics lambdas from 0 → target over warmup window
        if physics_warmup > 0 and epoch <= physics_warmup:
            ramp = epoch / physics_warmup
            loss_fn.lambda_bound  = target_lambda_bound  * ramp
            loss_fn.lambda_qorder = target_lambda_qorder * ramp
        else:
            loss_fn.lambda_bound  = target_lambda_bound
            loss_fn.lambda_qorder = target_lambda_qorder

        model.train()
        epoch_losses = []
        batch_perm = idx_train[torch.randperm(len(idx_train), generator=rng)]
        for b in range(n_batches):
            bi = batch_perm[b * batch_size : (b + 1) * batch_size]
            if len(bi) == 0:
                continue
            opt.zero_grad()
            preds = model(x_flat[bi], m_flat[bi], x_seq[bi], m_seq[bi])
            loss_d = loss_fn(preds, y[bi], v0[bi], v_sw[bi], x_seq[bi], m_seq[bi])
            loss_val = loss_d["total"]
            if not torch.isfinite(loss_val):
                logger.warning("NaN/Inf loss at epoch %d batch %d — skipping step", epoch, b)
                continue
            loss_val.backward()
            if grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            epoch_losses.append(float(loss_val.detach()))

        train_loss = sum(epoch_losses) / max(1, len(epoch_losses))

        model.eval()
        with torch.no_grad():
            if len(idx_val) > 0:
                v_preds = model(
                    x_flat[idx_val], m_flat[idx_val],
                    x_seq[idx_val], m_seq[idx_val],
                )
                v_loss_d = loss_fn(
                    v_preds, y[idx_val], v0[idx_val], v_sw[idx_val],
                    x_seq[idx_val], m_seq[idx_val],
                )
                val_loss = float(v_loss_d["total"])
            else:
                val_loss = train_loss

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        scheduler.step()
        progress_cb(epoch, train_loss, val_loss)

        if patience > 0 and epochs_no_improve >= patience:
            logger.info(
                "Early stopping at epoch %d — no improvement for %d epochs (best val=%.4f)",
                epoch, patience, best_val_loss,
            )
            break

    # Restore best weights before saving
    if best_state is not None:
        model.load_state_dict(best_state)

    # ---- Save artifact -----------------------------------------------------
    import uuid
    model_id = f"tft_pinn_{uuid.uuid4().hex[:8]}"
    out_dir = Path(model_output_dir) / model_id
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(out_dir / "model.pt"))
    (out_dir / "meta.json").write_text(json.dumps({
        "model_type":            "TFT_PINN",
        "n_flat":                len(_PINN_FLAT_COLS),
        "n_seq_channels":        _PINN_N_SEQ_CHANNELS,
        "n_seq_timesteps":       int(x_seq.shape[1]),
        "flat_hidden_dim":       int(hyperparameters.get("flat_hidden_dim", 256)),
        "flat_n_residual_blocks": int(hyperparameters.get("flat_n_residual_blocks", 2)),
        "null_embedding_dim":    int(hyperparameters.get("null_embedding_dim", 16)),
        "seq_hidden_dim":        int(hyperparameters.get("seq_hidden_dim", 128)),
        "seq_n_heads":           int(hyperparameters.get("seq_n_heads", 4)),
        "seq_n_layers":          int(hyperparameters.get("seq_n_layers", 2)),
        "seq_dropout":           float(hyperparameters.get("seq_dropout", 0.1)),
        "epochs":                epochs,
        "feat_mean":             feat_mean.tolist(),
        "feat_std":              feat_std.tolist(),
    }))
    logger.info("TFT+PINN model saved: %s (epochs=%d, N=%d)", model_id, epochs, N)
    return model_id


def _predict_tft_pinn(model_id: str, model_output_dir: str, table: pa.Table) -> pa.Table:
    """Run inference for TFT+PINN; returns Arrow table with P50 prediction."""
    if not _TORCH_AVAILABLE or not _TFT_PINN_AVAILABLE:
        values = pa.array([48.0] * table.num_rows, type=pa.float32())
        return pa.table({"prediction": values}, schema=PREDICTION_SCHEMA)

    meta_path = Path(model_output_dir) / model_id / "meta.json"
    meta = json.loads(meta_path.read_text())
    model = build_tft_pinn_model({
        "flat_hidden_dim":       meta["flat_hidden_dim"],
        "flat_n_residual_blocks": meta["flat_n_residual_blocks"],
        "null_embedding_dim":    meta["null_embedding_dim"],
        "seq_hidden_dim":        meta["seq_hidden_dim"],
        "seq_n_heads":           meta["seq_n_heads"],
        "seq_n_layers":          meta["seq_n_layers"],
        "seq_dropout":           meta["seq_dropout"],
    })
    weights = Path(model_output_dir) / model_id / "model.pt"
    model.load_state_dict(torch.load(str(weights), map_location="cpu", weights_only=True))
    model.eval()

    # Build x_flat / m_flat from Arrow; apply training-time standardization
    x_flat, m_flat = _flat_mask_tensors_from_arrow(table, list(_PINN_FLAT_COLS))
    if "feat_mean" in meta and "feat_std" in meta:
        feat_mean = torch.tensor(meta["feat_mean"], dtype=torch.float32)
        feat_std  = torch.tensor(meta["feat_std"],  dtype=torch.float32)
        x_flat = (x_flat - feat_mean) / feat_std
    aids = [str(v) for v in table.column("activity_id").to_pylist()]
    seq_path_col = "prediction_sequences_path"
    if seq_path_col in table.schema.names:
        seq_path = str(table.column(seq_path_col)[0].as_py())
    else:
        seq_path = "data/sequences"
    split = "holdout"
    if "split" in table.schema.names:
        split = str(table.column("split")[0].as_py())
    x_seq, m_seq = _pinn_sequences_from_parquet(seq_path, split, aids)

    with torch.no_grad():
        out = model(x_flat, m_flat, x_seq, m_seq)  # (B, 3)
    p50 = out[:, 1].tolist()
    values = pa.array(p50, type=pa.float32())
    return pa.table({"prediction": values}, schema=PREDICTION_SCHEMA)


def _export_tft_pinn_onnx(
    model_id: str, model_output_dir: str, onnx_path: str, opset: int
) -> None:
    """Export the TFT+PINN model to ONNX (opset <= 20).

    Named inputs:  x_flat (B, N_FLAT), m_flat (B, N_FLAT),
                   x_seq (B, T, C_SEQ), m_seq (B, T, C_SEQ)
    Named outputs: p10 (B, 1), p50 (B, 1), p90 (B, 1)
    Batch axis is dynamic; T and channel counts are fixed from the trained meta.
    """
    if not _TORCH_AVAILABLE or not _TFT_PINN_AVAILABLE:
        Path(onnx_path).parent.mkdir(parents=True, exist_ok=True)
        Path(onnx_path).write_bytes(b"\x00" * 8)
        return

    meta_path = Path(model_output_dir) / model_id / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"TFT+PINN meta not found: {meta_path}")

    meta = json.loads(meta_path.read_text())
    model = build_tft_pinn_model({
        "flat_hidden_dim":       meta["flat_hidden_dim"],
        "flat_n_residual_blocks": meta["flat_n_residual_blocks"],
        "null_embedding_dim":    meta["null_embedding_dim"],
        "seq_hidden_dim":        meta["seq_hidden_dim"],
        "seq_n_heads":           meta["seq_n_heads"],
        "seq_n_layers":          meta["seq_n_layers"],
        "seq_dropout":           meta["seq_dropout"],
    })

    weights = Path(model_output_dir) / model_id / "model.pt"
    model.load_state_dict(torch.load(str(weights), map_location="cpu", weights_only=True))
    model.eval()

    n_flat = int(meta["n_flat"])
    T_seq  = int(meta["n_seq_timesteps"])
    C_seq  = int(meta["n_seq_channels"])

    # Split the 3-way output into three (B, 1) tensors for named outputs
    class _OnnxWrapper(nn.Module):
        def __init__(self, inner: nn.Module) -> None:
            super().__init__()
            self.inner = inner

        def forward(self, x_flat, m_flat, x_seq, m_seq):
            preds = self.inner(x_flat, m_flat, x_seq, m_seq)  # (B, 3)
            return preds[:, 0:1], preds[:, 1:2], preds[:, 2:3]

    wrapper = _OnnxWrapper(model)
    wrapper.eval()

    dummy_x_flat = torch.zeros(1, n_flat, dtype=torch.float32)
    dummy_m_flat = torch.ones(1,  n_flat, dtype=torch.float32)
    dummy_x_seq  = torch.zeros(1, T_seq, C_seq, dtype=torch.float32)
    dummy_m_seq  = torch.ones(1,  T_seq, C_seq, dtype=torch.float32)

    Path(onnx_path).parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        (dummy_x_flat, dummy_m_flat, dummy_x_seq, dummy_m_seq),
        onnx_path,
        opset_version=min(opset if opset > 0 else 20, 20),
        input_names=["x_flat", "m_flat", "x_seq", "m_seq"],
        output_names=["p10", "p50", "p90"],
        dynamic_axes={
            "x_flat": {0: "batch"},
            "m_flat": {0: "batch"},
            "x_seq":  {0: "batch"},
            "m_seq":  {0: "batch"},
            "p10":    {0: "batch"},
            "p50":    {0: "batch"},
            "p90":    {0: "batch"},
        },
    )
    logger.info(
        "TFT+PINN exported to ONNX: %s (opset=%d, n_flat=%d, T=%d, C=%d)",
        onnx_path, min(opset if opset > 0 else 20, 20), n_flat, T_seq, C_seq,
    )


def _stub_train(table: pa.Table, model_type: str, model_output_dir: str) -> str:
    """Deterministic stub when PyTorch is unavailable."""
    import uuid
    model_id = f"stub_{model_type.lower()}_{uuid.uuid4().hex[:8]}"
    out_dir = Path(model_output_dir) / model_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "meta.json").write_text(
        json.dumps({"model_type": model_type, "stub": True, "rows": table.num_rows})
    )
    return model_id


def _predict(model_id: str, model_output_dir: str, table: pa.Table) -> pa.Table:
    """Run inference; returns Arrow table with prediction column (float32)."""
    meta_path = Path(model_output_dir) / model_id / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Model not found: {model_id!r}")

    meta = json.loads(meta_path.read_text())
    n = table.num_rows

    if meta.get("stub") or not _TORCH_AVAILABLE:
        # Stub: 48h deterministic (matches Phase 2 contract)
        values = pa.array([48.0] * n, type=pa.float32())
        return pa.table({"prediction": values}, schema=PREDICTION_SCHEMA)
    elif meta["model_type"] == "TFT_TRANSIT":
        if not _TFT_TRANSIT_AVAILABLE:
            raise RuntimeError("TFT_TRANSIT model requires tft_model.py and PyTorch")
        return _predict_tft_transit(model_id, model_output_dir, table)
    elif meta["model_type"] == "TFT_PINN":
        if not _TFT_PINN_AVAILABLE:
            raise RuntimeError("TFT_PINN model requires tft_pinn_model.py and PyTorch")
        return _predict_tft_pinn(model_id, model_output_dir, table)
    elif meta["model_type"] == "TFT":
        # Use stored input_size (not table.num_columns) — prediction frame may include target col.
        import math
        input_size = meta["input_size"]
        feat_names = table.schema.names[:input_size]
        # Replace Arrow null (None) with NaN; predict only on valid rows, NaN elsewhere.
        feat_lists = [[float("nan") if v is None else float(v)
                       for v in table.column(c).to_pylist()]
                      for c in feat_names]
        valid = [i for i in range(n)
                 if all(not math.isnan(feat_lists[j][i]) for j in range(input_size))]
        model = _SimpleTftModel(input_size=input_size, hidden=meta.get("hidden", 64))
        weights = Path(model_output_dir) / model_id / "model.pt"
        model.load_state_dict(torch.load(str(weights), map_location="cpu", weights_only=True))
        model.eval()
        if valid:
            X_v = torch.tensor(
                [[feat_lists[j][i] for i in valid] for j in range(input_size)],
                dtype=torch.float32,
            ).T.unsqueeze(1)
            with torch.no_grad():
                raw_v = model(X_v).squeeze(-1).tolist()
        else:
            raw_v = []
        raw = [float("nan")] * n
        for out_pos, row_idx in enumerate(valid):
            raw[row_idx] = raw_v[out_pos]
        values = pa.array(raw, type=pa.float32())
    elif meta["model_type"] == "NeuralOde":
        # Use stored dim (not table.num_columns) — prediction frame may include target col.
        import math
        dim = meta["dim"]
        feat_names = table.schema.names[:dim]
        feat_lists = [[float("nan") if v is None else float(v)
                       for v in table.column(c).to_pylist()]
                      for c in feat_names]
        valid = [i for i in range(n)
                 if all(not math.isnan(feat_lists[j][i]) for j in range(dim))]
        dynamics = _NeuralOdeDynamics(dim=dim, hidden=meta.get("hidden", 64))
        weights = Path(model_output_dir) / model_id / "dynamics.pt"
        dynamics.load_state_dict(torch.load(str(weights), map_location="cpu", weights_only=True))
        dynamics.eval()
        if valid:
            X_v = torch.tensor(
                [[feat_lists[j][i] for i in valid] for j in range(dim)],
                dtype=torch.float32,
            ).T
            t_v = torch.zeros(len(valid), 1)
            with torch.no_grad():
                raw_v = dynamics(X_v, t_v).mean(dim=-1).tolist()
        else:
            raw_v = []
        raw = [float("nan")] * n
        for out_pos, row_idx in enumerate(valid):
            raw[row_idx] = raw_v[out_pos]
        values = pa.array(raw, type=pa.float32())
    else:
        raise ValueError(f"Unknown model_type in meta: {meta['model_type']!r}")

    return pa.table({"prediction": values}, schema=PREDICTION_SCHEMA)


# ─── gRPC service implementation ──────────────────────────────────────────────

class _PythonTrainerServicer(solarpipe_pb2_grpc.PythonTrainerServicer if _GRPC_AVAILABLE else object):

    def __init__(self, model_dir: str) -> None:
        self._model_dir = model_dir

    def _dispatch_train(self, request, ct_event: threading.Event, progress_cb) -> str:
        table = _read_arrow_ipc(request.arrow_ipc_path)
        features = list(request.feature_columns)
        hyperparams = dict(request.hyperparameters)
        model_type = request.model_type.strip()

        # TFT+PINN routing: model_type=="TftPinn" OR (TFT + use_tft_pinn=true flag)
        use_tft_pinn = str(hyperparams.get("use_tft_pinn", "false")).strip().lower() == "true"
        if model_type.upper() in ("TFTPINN", "TFT_PINN") or (
            model_type.upper() == "TFT" and use_tft_pinn
        ):
            return _train_tft_pinn(table, features, request.target_column,
                                   hyperparams, self._model_dir, progress_cb, ct_event)

        if model_type.upper() == "TFT_TRANSIT":
            return _train_tft_transit(table, features, request.target_column,
                                      hyperparams, self._model_dir, progress_cb, ct_event)
        elif model_type.upper() == "TFT":
            return _train_tft(table, features, request.target_column,
                              hyperparams, self._model_dir, progress_cb, ct_event)
        elif model_type.lower() in ("neuralode", "neural_ode"):
            return _train_neural_ode(table, features, request.target_column,
                                     hyperparams, self._model_dir, progress_cb, ct_event)
        else:
            raise ValueError(
                f"Unsupported model_type {model_type!r}. "
                f"Supported: TFT, TftPinn, NeuralOde."
            )

    def StreamTrain(self, request, context):  # type: ignore[override]
        logger.info("StreamTrain: stage=%s model=%s", request.stage_name, request.model_type)
        ct_event = threading.Event()

        # Cooperative cancellation: watch gRPC context
        def _watch_cancel() -> None:
            while not ct_event.is_set():
                if context.is_active() is False:
                    ct_event.set()
                    break
                time.sleep(1.0)

        threading.Thread(target=_watch_cancel, daemon=True).start()

        progress_messages: list = []
        lock = threading.Lock()

        def _on_progress(epoch: int, train_loss: float, val_loss: float) -> None:
            with lock:
                progress_messages.append((epoch, train_loss, val_loss))

        train_err: Exception | None = None
        model_id: str = ""

        def _run() -> None:
            nonlocal train_err, model_id
            try:
                model_id = self._dispatch_train(request, ct_event, _on_progress)
            except Exception as exc:
                train_err = exc
            finally:
                ct_event.set()

        t = threading.Thread(target=_run, daemon=True, name=f"train-{request.stage_name}")
        t.start()

        # Stream progress while training runs
        last_sent = 0
        while not ct_event.is_set() or last_sent < len(progress_messages):
            with lock:
                new_msgs = progress_messages[last_sent:]

            for epoch, tl, vl in new_msgs:
                yield solarpipe_pb2.TrainProgress(
                    epoch=epoch, train_loss=tl, val_loss=vl, is_final=False
                )
            last_sent += len(new_msgs)

            if ct_event.is_set():
                break
            time.sleep(0.1)

        t.join(timeout=5.0)

        if train_err is not None:
            logger.error("StreamTrain error: %s", train_err)
            yield solarpipe_pb2.TrainProgress(
                is_final=True, error_message=str(train_err)
            )
            return

        yield solarpipe_pb2.TrainProgress(is_final=True, model_id=model_id)
        logger.info("StreamTrain complete: model_id=%s", model_id)

    def Train(self, request, context):  # type: ignore[override]
        logger.info("Train: stage=%s model=%s", request.stage_name, request.model_type)
        try:
            losses: list[tuple[int, float, float]] = []
            model_id = self._dispatch_train(request, threading.Event(),
                                            lambda e, tl, vl: losses.append((e, tl, vl)))
            tl = losses[-1][1] if losses else float("nan")
            vl = losses[-1][2] if losses else float("nan")
            return solarpipe_pb2.TrainResponse(
                model_id=model_id,
                final_train_loss=tl,
                final_val_loss=vl,
                epochs_completed=len(losses),
            )
        except Exception as exc:
            logger.error("Train error: %s", exc)
            return solarpipe_pb2.TrainResponse(error_message=str(exc))

    def Predict(self, request, context):  # type: ignore[override]
        logger.info("Predict: model_id=%s", request.model_id)
        try:
            table_in = _read_arrow_ipc(request.arrow_ipc_path)
            table_out = _predict(request.model_id, self._model_dir, table_in)
            _write_arrow_ipc(request.output_arrow_ipc_path, table_out)
            return solarpipe_pb2.PredictResponse(
                output_arrow_ipc_path=request.output_arrow_ipc_path,
                row_count=table_out.num_rows,
            )
        except Exception as exc:
            logger.error("Predict error: %s", exc)
            return solarpipe_pb2.PredictResponse(error_message=str(exc))

    def ExportOnnx(self, request, context):  # type: ignore[override]
        logger.info("ExportOnnx: model_id=%s opset=%d", request.model_id, request.opset)
        try:
            meta_path = Path(self._model_dir) / request.model_id / "meta.json"
            if not meta_path.exists():
                raise FileNotFoundError(f"Model not found: {request.model_id!r}")
            meta = json.loads(meta_path.read_text())
            mt = str(meta.get("model_type", "")).upper()
            if mt == "TFT_PINN":
                _export_tft_pinn_onnx(
                    request.model_id, self._model_dir,
                    request.onnx_output_path, request.opset,
                )
            elif mt in ("NEURALODE", "NEURAL_ODE"):
                _export_neural_ode_onnx(
                    request.model_id, self._model_dir,
                    request.onnx_output_path, request.opset,
                )
            else:
                raise ValueError(
                    f"ExportOnnx: model_type {mt!r} not supported for ONNX export. "
                    f"Supported: TFT_PINN, NeuralOde."
                )
            return solarpipe_pb2.ExportOnnxResponse(
                onnx_output_path=request.onnx_output_path
            )
        except Exception as exc:
            logger.error("ExportOnnx error: %s", exc)
            return solarpipe_pb2.ExportOnnxResponse(error_message=str(exc))


# ─── Server bootstrap ─────────────────────────────────────────────────────────

def serve(port: int = 50051, parent_pid: int | None = None, model_dir: str = "./models") -> None:
    if not _GRPC_AVAILABLE:
        logger.error(
            "gRPC imports failed (%s). Install: pip install grpcio grpcio-health-checking pyarrow. "
            "Also compile proto: python -m grpc_tools.protoc -I python/ "
            "--python_out=python/ --grpc_python_out=python/ python/solarpipe.proto",
            _grpc_err_msg if not _GRPC_AVAILABLE else "",
        )
        sys.exit(1)

    Path(model_dir).mkdir(parents=True, exist_ok=True)

    if parent_pid is not None:
        _start_parent_heartbeat(parent_pid)

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=4),
        options=[
            ("grpc.max_send_message_length", 4 * 1024 * 1024),
            ("grpc.max_receive_message_length", 4 * 1024 * 1024),
        ],
    )

    # Register main service
    solarpipe_pb2_grpc.add_PythonTrainerServicer_to_server(
        _PythonTrainerServicer(model_dir), server
    )

    # Register health service (grpc.health.v1.Health — ADR-011)
    health_servicer = health.HealthServicer()
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)
    health_servicer.set(
        "solarpipe.PythonTrainer",
        health_pb2.HealthCheckResponse.SERVING,
    )
    health_servicer.set(
        "",
        health_pb2.HealthCheckResponse.SERVING,
    )

    # RULE-062: bind to 0.0.0.0, not localhost
    addr = f"0.0.0.0:{port}"
    server.add_insecure_port(addr)
    server.start()
    logger.info("SolarPipe sidecar listening on %s (torch=%s)", addr, _TORCH_AVAILABLE)

    def _handle_sigterm(_sig, _frame) -> None:
        logger.info("SIGTERM received — graceful shutdown")
        server.stop(grace=5)

    signal.signal(signal.SIGTERM, _handle_sigterm)

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt — stopping server")
        server.stop(grace=2)

    logger.info("SolarPipe sidecar stopped")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SolarPipe Python gRPC sidecar")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument(
        "--parent-pid", type=int, default=None,
        help="PID of .NET host process; sidecar exits when parent dies (RULE-060)"
    )
    parser.add_argument(
        "--model-dir", default="./models",
        help="Directory for model artifacts"
    )
    parser.add_argument(
        "--log-dir", default="logs",
        help="Directory for structured JSON logs (ADR-010)"
    )
    args = parser.parse_args()
    _configure_logging(args.log_dir)
    serve(port=args.port, parent_pid=args.parent_pid, model_dir=args.model_dir)
