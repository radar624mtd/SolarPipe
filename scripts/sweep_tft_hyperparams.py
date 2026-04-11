"""LHS hyperparameter sweep for TransitTimeTFT.

Samples up to 100 configurations via Latin Hypercube Sampling over:
  d_model:      [32, 64, 128, 256]
  n_heads:      [2, 4, 8]
  n_lstm_layers:[1, 2, 3]
  dropout:      [0.05, 0.10, 0.20, 0.30]
  lr:           [5e-4, 1e-3, 2e-3, 5e-3]
  batch_size:   [32, 64, 128]

Each config is evaluated on a single temporal hold-out fold (last 20% of
training events chronologically) with 80 max epochs + early stopping.
Best config is saved to output/tft_v1/sweep_best.json.

Usage:
  python scripts/sweep_tft_hyperparams.py [--n-configs 50] [--device cuda]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sqlite3
import sys
import threading
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

_script_dir = Path(__file__).resolve().parent
_worktree_root = _script_dir.parent   # eloquent-benz dir (has python/ subdir)
ROOT = _worktree_root
if not (ROOT / "data" / "data" / "staging" / "staging.db").exists():
    for _p in _script_dir.parents:
        if (_p / "data" / "data" / "staging" / "staging.db").exists():
            ROOT = _p
            break
# tft_model.py lives in the worktree python/ dir (has CUDA TDR fix)
sys.path.insert(0, str(_worktree_root / "python"))
sys.path.insert(0, str(_script_dir))

from tft_model import TransitTimeTFT, TFTTrainer
from train_tft_model import (
    load_existing_predictions, build_dataset, impute_and_normalize,
    N_STATIC, N_SEQ_CH, N_EXISTING, STAGING_DB,
)

OUT_DIR = ROOT / "output" / "tft_v1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Hyperparameter grid ───────────────────────────────────────────────────────

GRID = {
    "d_model":       [32, 64, 128, 256],
    "n_heads":       [2, 4, 8],
    "n_lstm_layers": [1, 2, 3],
    "dropout":       [0.05, 0.10, 0.20, 0.30],
    "lr":            [5e-4, 1e-3, 2e-3, 5e-3],
    "batch_size":    [32, 64, 128],
}


def lhs_sample(grid: dict, n: int, seed: int = 42) -> list[dict]:
    """Latin Hypercube Sampling over discrete grid.

    Divides each parameter's range into n equal strata, samples one value
    from each stratum, then randomly pairs across parameters.
    """
    rng = random.Random(seed)
    keys = list(grid.keys())
    n_params = len(keys)

    # For each parameter, create n strata indices then shuffle
    strata = []
    for key in keys:
        choices = grid[key]
        n_choices = len(choices)
        # Map n strata → choice indices (cyclic)
        indices = [i % n_choices for i in range(n)]
        rng.shuffle(indices)
        strata.append(indices)

    configs = []
    for sample_i in range(n):
        cfg = {keys[p]: grid[keys[p]][strata[p][sample_i]] for p in range(n_params)}
        # Constraint: n_heads must divide d_model
        while cfg["d_model"] % cfg["n_heads"] != 0:
            cfg["n_heads"] = rng.choice([h for h in grid["n_heads"]
                                         if cfg["d_model"] % h == 0] or [1])
        configs.append(cfg)
    return configs


def evaluate_config(
    cfg: dict,
    train_data: dict,
    val_data: dict,
    device: str,
    max_epochs: int = 80,
    patience: int = 15,
) -> tuple[float, float]:
    """Train one config on train_data, evaluate on val_data.

    Returns (val_mae, best_val_loss).
    """
    model_kwargs = {
        "n_static": N_STATIC,
        "n_seq_channels": N_SEQ_CH,
        "n_existing_preds": N_EXISTING,
        "d_model": cfg["d_model"],
        "n_heads": cfg["n_heads"],
        "n_lstm_layers": cfg["n_lstm_layers"],
        "dropout": cfg["dropout"],
    }

    trainer = TFTTrainer(
        model_kwargs=model_kwargs,
        epochs=max_epochs,
        lr=cfg["lr"],
        batch_size=cfg["batch_size"],
        device=device,
        patience=patience,
    )

    ct_event = threading.Event()

    def _silent_cb(epoch, tl, vl):
        pass

    try:
        model = trainer.train_fold(train_data, val_data, _silent_cb, ct_event)
        preds = trainer.predict(model, val_data).numpy()
        labels = val_data["target"].squeeze().numpy()
        valid = ~np.isnan(labels) & ~np.isnan(preds[:, 1])
        if not valid.any():
            return float("nan"), float("nan")
        mae = float(np.mean(np.abs(preds[:, 1][valid] - labels[valid])))
        # Compute val_loss on CPU to avoid WDDM TDR with large val sets
        preds_t = torch.from_numpy(preds)
        tgt_t = val_data["target"].cpu()
        q_cpu = trainer.q_loss.q.cpu()
        if tgt_t.dim() == 1:
            tgt_t = tgt_t.unsqueeze(1)
        err = tgt_t - preds_t
        val_loss = float(torch.where(err >= 0, q_cpu * err, (q_cpu - 1.0) * err).mean())
        return mae, val_loss
    except Exception as exc:
        print(f"    ERROR: {exc}")
        return float("nan"), float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-configs", type=int, default=50,
                    help="Number of LHS configs to evaluate (default 50, max 100)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    n_configs = min(args.n_configs, 100)  # RULE-166: cap at 100

    print(f"=== TFT Hyperparameter Sweep (LHS, n={n_configs}) ===")
    print(f"device: {args.device}")
    print(f"N_STATIC={N_STATIC}, N_SEQ_CH={N_SEQ_CH}, N_EXISTING={N_EXISTING}")

    # Load data once
    print("\n[1/3] Loading datasets...")
    existing_preds = load_existing_predictions()
    train_ds = build_dataset("train", existing_preds)
    holdout_ds = build_dataset("holdout", existing_preds)

    # Normalize (fit on train)
    tr_s, ho_s, sm, ss = impute_and_normalize(train_ds["static"].numpy(),
                                               holdout_ds["static"].numpy())
    train_ds["static"] = torch.from_numpy(tr_s)
    holdout_ds["static"] = torch.from_numpy(ho_s)

    tr_pre_np = train_ds["pre_seq"].numpy()
    N_tr, T_pre, C = tr_pre_np.shape
    ho_pre_np = holdout_ds["pre_seq"].numpy()
    tr_pn, ho_pn, _, _ = impute_and_normalize(tr_pre_np.reshape(-1, C),
                                               ho_pre_np.reshape(-1, C))
    train_ds["pre_seq"] = torch.from_numpy(tr_pn.reshape(N_tr, T_pre, C))
    holdout_ds["pre_seq"] = torch.from_numpy(ho_pn.reshape(ho_pre_np.shape[0], T_pre, C))

    tr_tr_np = train_ds["transit_seq"].numpy()
    ho_tr_np = holdout_ds["transit_seq"].numpy()
    _, T_tr, _ = tr_tr_np.shape
    tr_tn, ho_tn, _, _ = impute_and_normalize(tr_tr_np.reshape(-1, C),
                                               ho_tr_np.reshape(-1, C))
    train_ds["transit_seq"] = torch.from_numpy(tr_tn.reshape(N_tr, T_tr, C))
    holdout_ds["transit_seq"] = torch.from_numpy(ho_tn.reshape(ho_tr_np.shape[0], T_tr, C))

    tr_ep, ho_ep, _, _ = impute_and_normalize(train_ds["existing_preds"].numpy(),
                                               holdout_ds["existing_preds"].numpy())
    train_ds["existing_preds"] = torch.from_numpy(tr_ep)
    holdout_ds["existing_preds"] = torch.from_numpy(ho_ep)

    # Temporal hold-out split: use last 600 events only (400 train + 200 val)
    # Smaller slice makes each config ~4x faster while still ranking hyperparams correctly
    N_all = train_ds["target"].shape[0]
    SWEEP_TRAIN = 400
    SWEEP_VAL = 200
    sweep_start = max(0, N_all - SWEEP_TRAIN - SWEEP_VAL)
    split_idx = sweep_start + SWEEP_TRAIN
    sweep_train = {k: v[sweep_start:split_idx] if torch.is_tensor(v) else v
                   for k, v in train_ds.items() if k != "activity_ids"}
    sweep_val = {k: v[split_idx:split_idx + SWEEP_VAL] if torch.is_tensor(v) else v
                 for k, v in train_ds.items() if k != "activity_ids"}
    print(f"  Sweep split: {SWEEP_TRAIN} train, {SWEEP_VAL} val events (subset of last {SWEEP_TRAIN+SWEEP_VAL})")

    # Sample configs
    print(f"\n[2/3] Sampling {n_configs} LHS configurations...")
    configs = lhs_sample(GRID, n_configs, seed=args.seed)

    # Evaluate
    print(f"\n[3/3] Evaluating configurations...")
    results = []
    best_mae = float("inf")
    best_cfg = None

    for i, cfg in enumerate(configs):
        print(f"  [{i+1:3d}/{n_configs}] d={cfg['d_model']} h={cfg['n_heads']} "
              f"lstm={cfg['n_lstm_layers']} drop={cfg['dropout']:.2f} "
              f"lr={cfg['lr']:.0e} bs={cfg['batch_size']} ... ", end="", flush=True)

        mae, val_loss = evaluate_config(cfg, sweep_train, sweep_val,
                                        args.device, max_epochs=50, patience=8)
        results.append({"config": cfg, "val_mae": mae, "val_loss": val_loss})
        flag = ""
        if not math.isnan(mae) and mae < best_mae:
            best_mae = mae
            best_cfg = cfg
            flag = " ★ NEW BEST"
        print(f"MAE={mae:.2f}h{flag}")

    # Sort by MAE
    valid_results = [r for r in results if not math.isnan(r["val_mae"])]
    valid_results.sort(key=lambda r: r["val_mae"])

    print(f"\n=== Top 10 Configurations ===")
    for rank, r in enumerate(valid_results[:10]):
        c = r["config"]
        print(f"  #{rank+1}: MAE={r['val_mae']:.2f}h  "
              f"d={c['d_model']} h={c['n_heads']} lstm={c['n_lstm_layers']} "
              f"drop={c['dropout']:.2f} lr={c['lr']:.0e} bs={c['batch_size']}")

    if best_cfg:
        print(f"\nBest config: {best_cfg}")
        print(f"Best sweep MAE: {best_mae:.2f}h")

    # Save
    sweep_out = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "n_configs": n_configs,
        "n_static": N_STATIC,
        "n_seq_channels": N_SEQ_CH,
        "n_existing_preds": N_EXISTING,
        "best_mae": best_mae,
        "best_config": best_cfg,
        "top10": valid_results[:10],
        "all_results": valid_results,
    }
    out_path = OUT_DIR / "sweep_results.json"
    out_path.write_text(json.dumps(sweep_out, indent=2))
    print(f"\nResults saved: {out_path}")

    # Save best config separately for train_tft_model.py to pick up
    if best_cfg:
        best_path = OUT_DIR / "sweep_best.json"
        best_path.write_text(json.dumps({"config": best_cfg, "val_mae": best_mae}, indent=2))
        print(f"Best config:   {best_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
