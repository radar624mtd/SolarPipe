"""Temporal Fusion Transformer for CME transit time prediction.

Full TFT architecture in pure PyTorch (no pytorch-forecasting dependency):
  - Variable selection networks for static covariates and time series
  - Gated residual networks (GRN) throughout
  - LSTM encoder for local processing
  - Multi-head temporal self-attention
  - Quantile output head (P10/P50/P90)

Ensemble head (Phase 4):
  - Concatenates TFT encoder output with scalar predictions from existing models
    (Phase 8, PINN V1, Phase 9 progressive, physics ODE)
  - Dense layers → three quantile outputs

Input contract:
  static_features:  (N, S)   float32  — scalar CME/SHARP/CDAW features
  pre_seq:          (N, 150, C) float32 — pre-launch OMNI (150h × 20ch)
  transit_seq:      (N, T_max, C) float32 — in-transit OMNI (≤72h × 20ch, right-padded)
  transit_mask:     (N, T_max) bool  — True for valid (non-padded) positions
  existing_preds:   (N, E)   float32  — E scalar predictions from existing models (may be NaN)

Output:
  quantiles:        (N, 3)   float32  — [P10, P50, P90] transit time in hours

Training:
  Quantile (pinball) loss: sum over q in [0.1, 0.5, 0.9]
  Temporal CV: 5 expanding-window folds, last fold calibration-only
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Gated Residual Network ────────────────────────────────────────────────────

class GRN(nn.Module):
    """Gated Residual Network.

    Applies: ELU(W1·x + W2·context + b1) → gate → skip connection → LayerNorm.
    context is optional (used in variable selection).
    """

    def __init__(self, d_in: int, d_hidden: int, d_out: int,
                 d_context: int = 0, dropout: float = 0.1) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_in + d_context, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_out)
        self.gate = nn.Linear(d_hidden, d_out)
        self.skip = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()
        self.norm = nn.LayerNorm(d_out)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,
                context: Optional[torch.Tensor] = None) -> torch.Tensor:
        inp = torch.cat([x, context], dim=-1) if context is not None else x
        h = F.elu(self.fc1(inp))
        h = self.drop(h)
        h2 = self.fc2(h)
        g = torch.sigmoid(self.gate(h))
        out = g * h2 + (1 - g) * self.skip(x)
        return self.norm(out)


# ── Variable Selection Network ────────────────────────────────────────────────

class VariableSelectionNetwork(nn.Module):
    """Learns soft feature importance weights; returns weighted sum of per-variable GRNs."""

    def __init__(self, n_vars: int, d_model: int, d_context: int = 0,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.n_vars = n_vars
        self.d_model = d_model
        # One GRN per variable to produce d_model embedding
        self.var_grns = nn.ModuleList([
            GRN(1, d_model, d_model, d_context=d_context, dropout=dropout)
            for _ in range(n_vars)
        ])
        # Selection weight softmax over all variables
        self.softmax_grn = GRN(n_vars * d_model, d_model, n_vars,
                               d_context=d_context, dropout=dropout)

    def forward(self, x: torch.Tensor,
                context: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x: (..., n_vars)  — can be (N, S) for static or (N, T, C) for temporal
        Returns:
            processed: (..., d_model)
            weights:   (..., n_vars)  — softmax importance per variable
        """
        original_shape = x.shape[:-1]
        x_flat = x.reshape(-1, self.n_vars)  # (*, n_vars)
        ctx_flat = context.reshape(-1, context.shape[-1]) if context is not None else None

        var_embeddings = []
        for i, grn in enumerate(self.var_grns):
            vi = x_flat[:, i:i+1]  # (*, 1)
            ve = grn(vi, ctx_flat)  # (*, d_model)
            var_embeddings.append(ve)

        stacked = torch.stack(var_embeddings, dim=1)  # (*, n_vars, d_model)
        flat_emb = stacked.reshape(x_flat.shape[0], -1)  # (*, n_vars*d_model)

        weights = F.softmax(self.softmax_grn(flat_emb, ctx_flat), dim=-1)  # (*, n_vars)
        processed = (stacked * weights.unsqueeze(-1)).sum(dim=1)  # (*, d_model)

        processed = processed.reshape(*original_shape, self.d_model)
        weights = weights.reshape(*original_shape, self.n_vars)
        return processed, weights


# ── Multi-Head Temporal Attention ─────────────────────────────────────────────

class TemporalAttention(nn.Module):
    """Multi-head scaled dot-product self-attention with optional causal mask."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x:    (N, T, d_model)
        mask: (N, T) bool — True for valid positions; False positions are masked out
        Returns: (N, T, d_model), attn_weights (N, n_heads, T, T)
        """
        N, T, _ = x.shape
        H, Dh = self.n_heads, self.d_head

        Q = self.q(x).view(N, T, H, Dh).transpose(1, 2)  # (N, H, T, Dh)
        K = self.k(x).view(N, T, H, Dh).transpose(1, 2)
        V = self.v(x).view(N, T, H, Dh).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Dh)  # (N, H, T, T)

        if mask is not None:
            # mask: (N, T) → broadcast to (N, 1, 1, T) to mask key positions
            key_mask = ~mask.unsqueeze(1).unsqueeze(2)  # (N, 1, 1, T)
            scores = scores.masked_fill(key_mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.drop(attn)

        out = torch.matmul(attn, V).transpose(1, 2).reshape(N, T, H * Dh)
        return self.out(out), attn


# ── Full TFT ──────────────────────────────────────────────────────────────────

class TransitTimeTFT(nn.Module):
    """Two-headed TFT for CME transit time prediction.

    Architecture:
      1. Static variable selection (S features → d_model context vector)
      2. Pre-launch sequence variable selection (C channels → d_model per timestep)
      3. In-transit sequence variable selection (C channels → d_model per timestep)
      4. LSTM encoder over combined [pre_launch || in_transit] sequence
      5. Multi-head temporal self-attention
      6. Static enrichment GRN (context injected at each timestep)
      7. Positional gate + skip
      8. Final GRN per timestep
      9. Aggregate last-valid timestep representation
      10. Ensemble head: concat with existing model predictions → quantile output
    """

    QUANTILES = [0.1, 0.5, 0.9]

    def __init__(
        self,
        n_static: int,
        n_seq_channels: int,
        n_existing_preds: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_lstm_layers: int = 2,
        dropout: float = 0.1,
        max_pre_len: int = 150,
        max_transit_len: int = 72,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_pre_len = max_pre_len
        self.max_transit_len = max_transit_len
        self.n_existing_preds = n_existing_preds

        # 1. Static variable selection
        self.static_vsn = VariableSelectionNetwork(
            n_vars=n_static, d_model=d_model, dropout=dropout
        )
        # Static context encoder: maps static embedding to context vector
        self.static_context_grn = GRN(d_model, d_model, d_model, dropout=dropout)

        # 2 & 3. Sequence variable selection (shared weights for pre + transit)
        self.seq_vsn = VariableSelectionNetwork(
            n_vars=n_seq_channels, d_model=d_model,
            d_context=d_model, dropout=dropout
        )

        # 4. LSTM encoder
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=n_lstm_layers,
            batch_first=True,
            dropout=dropout if n_lstm_layers > 1 else 0.0,
        )

        # Skip connection gate after LSTM
        self.lstm_gate = GRN(d_model, d_model, d_model, dropout=dropout)

        # 5. Multi-head temporal attention
        self.attention = TemporalAttention(d_model, n_heads, dropout=dropout)
        self.attn_gate = GRN(d_model, d_model, d_model, dropout=dropout)

        # 6. Static enrichment
        self.static_enrich_grn = GRN(d_model, d_model, d_model,
                                     d_context=d_model, dropout=dropout)

        # 8. Final GRN per timestep
        self.final_grn = GRN(d_model, d_model, d_model, dropout=dropout)

        # 10. Ensemble head
        # existing_preds may be NaN — use a learned NaN embedding per predictor
        if n_existing_preds > 0:
            self.pred_nan_embed = nn.Parameter(torch.zeros(n_existing_preds))
            self.pred_proj = nn.Linear(n_existing_preds, d_model // 2)
            head_in = d_model + d_model // 2
        else:
            head_in = d_model

        self.head = nn.Sequential(
            nn.Linear(head_in, d_model),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, len(self.QUANTILES)),
        )

    def _build_seq_mask(self, seq: torch.Tensor) -> torch.Tensor:
        """True where at least one channel is non-NaN (valid timestep)."""
        return (~torch.isnan(seq).all(dim=-1))  # (N, T)

    def _replace_nan(self, x: torch.Tensor, fill: float = 0.0) -> torch.Tensor:
        return torch.where(torch.isnan(x), torch.full_like(x, fill), x)

    def forward(
        self,
        static_features: torch.Tensor,          # (N, S)
        pre_seq: torch.Tensor,                   # (N, 150, C)
        transit_seq: torch.Tensor,               # (N, T_max, C)
        transit_mask: Optional[torch.Tensor],    # (N, T_max) True=valid
        existing_preds: Optional[torch.Tensor],  # (N, E) may contain NaN
    ) -> torch.Tensor:                           # (N, 3) quantile predictions
        N = static_features.shape[0]

        # Replace NaN with 0.0 for numeric computations (mask tracks validity)
        static_clean = self._replace_nan(static_features)
        pre_clean = self._replace_nan(pre_seq)
        transit_clean = self._replace_nan(transit_seq)

        # 1. Static variable selection → context vector (N, d_model)
        static_emb, _static_wts = self.static_vsn(static_clean)
        ctx = self.static_context_grn(static_emb)  # (N, d_model)

        # 2. Pre-launch variable selection: (N, 150, C) → (N, 150, d_model)
        pre_emb, _ = self.seq_vsn(pre_clean,
                                   context=ctx.unsqueeze(1).expand(-1, pre_clean.shape[1], -1))

        # 3. In-transit variable selection: (N, T_max, C) → (N, T_max, d_model)
        T_max = transit_clean.shape[1]
        if T_max > 0:
            transit_emb, _ = self.seq_vsn(
                transit_clean,
                context=ctx.unsqueeze(1).expand(-1, T_max, -1)
            )
        else:
            transit_emb = torch.zeros(N, 0, self.d_model, device=static_features.device)

        # Concatenate pre + transit along time axis
        combined = torch.cat([pre_emb, transit_emb], dim=1)  # (N, 150+T_max, d_model)
        T_total = combined.shape[1]

        # Build combined validity mask for attention
        pre_mask = self._build_seq_mask(pre_seq)   # (N, 150)
        if T_max > 0:
            tr_mask = transit_mask if transit_mask is not None else self._build_seq_mask(transit_seq)
        else:
            tr_mask = torch.zeros(N, 0, dtype=torch.bool, device=static_features.device)
        combined_mask = torch.cat([pre_mask, tr_mask], dim=1)  # (N, T_total)
        # Ensure at least one valid position per row (avoid all-inf softmax)
        any_valid = combined_mask.any(dim=1, keepdim=True)
        combined_mask = combined_mask | (~any_valid)  # if nothing valid, unmask all

        # 4. LSTM encoder
        lstm_out, _ = self.lstm(combined)  # (N, T_total, d_model)
        lstm_gated = self.lstm_gate(lstm_out) + combined  # gated skip

        # 6. Static enrichment: inject context at each timestep
        ctx_exp = ctx.unsqueeze(1).expand(-1, T_total, -1)
        enriched = self.static_enrich_grn(lstm_gated, context=ctx_exp)

        # 5. Multi-head temporal attention
        attn_out, _attn_wts = self.attention(enriched, mask=combined_mask)
        attn_gated = self.attn_gate(attn_out) + enriched  # gated skip

        # 8. Final GRN
        final = self.final_grn(attn_gated)  # (N, T_total, d_model)

        # 9. Aggregate: take last valid timestep per sequence
        # valid_idx: (N,) — index of last True in combined_mask
        valid_idx = combined_mask.long().cumsum(dim=1).argmax(dim=1)  # (N,)
        # clamp to avoid out-of-bounds on all-invalid (shouldn't happen after unmask)
        valid_idx = valid_idx.clamp(0, T_total - 1)
        seq_repr = final[torch.arange(N, device=final.device), valid_idx]  # (N, d_model)

        # 10. Ensemble head
        if self.n_existing_preds > 0 and existing_preds is not None:
            # Replace NaN positions with learned embedding
            ep_clean = existing_preds.clone()
            nan_mask = torch.isnan(ep_clean)
            ep_clean = torch.where(nan_mask,
                                   self.pred_nan_embed.unsqueeze(0).expand(N, -1),
                                   ep_clean)
            pred_emb = F.elu(self.pred_proj(ep_clean))  # (N, d_model//2)
            head_in = torch.cat([seq_repr, pred_emb], dim=-1)
        else:
            head_in = seq_repr

        return self.head(head_in)  # (N, 3)


# ── Quantile (Pinball) Loss ───────────────────────────────────────────────────

class QuantileLoss(nn.Module):
    """Pinball loss averaged over quantiles and batch."""

    def __init__(self, quantiles: list[float]) -> None:
        super().__init__()
        self.register_buffer("q", torch.tensor(quantiles, dtype=torch.float32))

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        preds:  (N, Q)  — one column per quantile
        target: (N,) or (N, 1)
        """
        if target.dim() == 1:
            target = target.unsqueeze(1)  # (N, 1)
        err = target - preds  # (N, Q)
        loss = torch.where(err >= 0,
                           self.q * err,
                           (self.q - 1.0) * err)
        return loss.mean()


# ── Trainer ───────────────────────────────────────────────────────────────────

class TFTTrainer:
    """Walk-forward temporal CV trainer for TransitTimeTFT.

    Handles:
      - NaN imputation (static mean fill at train time, applied at predict time)
      - 5-fold expanding-window CV with gap buffer
      - Last fold calibration-only (RULE-164)
      - Asymmetric loss: late predictions penalised 1.5× (matches CLAUDE.md spec)
      - Checkpoint saving per fold
    """

    LATE_PENALTY = 1.5  # P50 errors where pred > truth cost 1.5× more

    def __init__(
        self,
        model_kwargs: dict,
        epochs: int = 50,
        lr: float = 1e-3,
        batch_size: int = 64,
        n_folds: int = 5,
        gap_days: int = 14,
        device: str = "cpu",
        patience: int = 20,
    ) -> None:
        self.model_kwargs = model_kwargs
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.n_folds = n_folds
        self.gap_days = gap_days
        self.device = torch.device(device)
        self.patience = patience  # early stopping patience in epochs
        self.q_loss = QuantileLoss(TransitTimeTFT.QUANTILES).to(self.device)

    def _asymmetric_weight(self, preds_p50: torch.Tensor,
                           target: torch.Tensor) -> torch.Tensor:
        """Return per-sample weight: LATE_PENALTY if pred > truth, else 1.0."""
        late = (preds_p50 > target).float()
        return 1.0 + (self.LATE_PENALTY - 1.0) * late

    def _compute_loss(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        base_loss = self.q_loss(preds, target)
        p50 = preds[:, 1].detach()  # P50 column
        weights = self._asymmetric_weight(p50, target.squeeze())
        return (base_loss * weights.mean()).mean()

    def train_fold(
        self,
        train_data: dict,
        val_data: dict,
        progress_cb,
        ct_event,
    ) -> TransitTimeTFT:
        """Train one fold; return trained model."""
        model = TransitTimeTFT(**self.model_kwargs).to(self.device)
        opt = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)

        N_train = train_data["target"].shape[0]
        best_val = float("inf")
        best_state = None
        patience_counter = 0

        for epoch in range(1, self.epochs + 1):
            if ct_event.is_set():
                break

            model.train()
            perm = torch.randperm(N_train)
            epoch_losses = []

            for start in range(0, N_train, self.batch_size):
                idx = perm[start:start + self.batch_size]
                batch = {k: v[idx].to(self.device) if torch.is_tensor(v) else v
                         for k, v in train_data.items()}
                opt.zero_grad()
                preds = model(
                    batch["static"],
                    batch["pre_seq"],
                    batch["transit_seq"],
                    batch.get("transit_mask"),
                    batch.get("existing_preds"),
                )
                loss = self._compute_loss(preds, batch["target"])
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                epoch_losses.append(loss.item())

            scheduler.step()
            train_loss = sum(epoch_losses) / len(epoch_losses)

            # Validation — batched to avoid WDDM TDR timeout on large val sets
            model.eval()
            val_n = val_data["target"].shape[0]
            val_batch = max(self.batch_size, 128)
            val_pred_chunks = []
            val_tgt_chunks = []
            with torch.no_grad():
                for vb_start in range(0, val_n, val_batch):
                    vb_end = min(vb_start + val_batch, val_n)
                    vb_mask_default = torch.ones(vb_end - vb_start, 1, dtype=torch.bool)
                    vb_ep_default = torch.full(
                        (vb_end - vb_start,
                         self.model_kwargs.get("n_existing_preds", 0)),
                        float("nan"))
                    vb_preds = model(
                        val_data["static"][vb_start:vb_end].to(self.device),
                        val_data["pre_seq"][vb_start:vb_end].to(self.device),
                        val_data["transit_seq"][vb_start:vb_end].to(self.device),
                        val_data.get("transit_mask",
                                     vb_mask_default)[vb_start:vb_end].to(self.device),
                        val_data.get("existing_preds",
                                     vb_ep_default)[vb_start:vb_end].to(self.device),
                    )
                    val_pred_chunks.append(vb_preds.cpu())
                    val_tgt_chunks.append(val_data["target"][vb_start:vb_end])
            val_preds = torch.cat(val_pred_chunks, dim=0)   # (N, Q) on CPU
            val_tgt = torch.cat(val_tgt_chunks, dim=0)       # (N,) or (N,1) on CPU
            # Compute pinball loss on CPU (preds already moved off GPU)
            if val_tgt.dim() == 1:
                val_tgt = val_tgt.unsqueeze(1)
            q_cpu = self.q_loss.q.cpu()
            err = val_tgt - val_preds
            val_loss = torch.where(err >= 0, q_cpu * err, (q_cpu - 1.0) * err).mean().item()

            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            progress_cb(epoch, train_loss, val_loss)

            if patience_counter >= self.patience:
                progress_cb(epoch, train_loss, val_loss)  # ensure final state logged
                break

        if best_state is not None:
            model.load_state_dict(best_state)

        return model

    def predict(
        self,
        model: TransitTimeTFT,
        data: dict,
    ) -> torch.Tensor:
        """Run inference; returns (N, 3) quantile predictions."""
        model.eval()
        device = next(model.parameters()).device
        results = []
        N = data["target"].shape[0]
        for start in range(0, N, self.batch_size):
            batch = {k: v[start:start + self.batch_size].to(device) if torch.is_tensor(v) else v
                     for k, v in data.items()}
            with torch.no_grad():
                preds = model(
                    batch["static"],
                    batch["pre_seq"],
                    batch["transit_seq"],
                    batch.get("transit_mask"),
                    batch.get("existing_preds"),
                )
            results.append(preds.cpu())
        return torch.cat(results, dim=0)
