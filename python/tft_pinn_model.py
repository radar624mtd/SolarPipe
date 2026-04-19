"""TFT + PINN neural ensemble model for CME transit time prediction (G3 gate).

Architecture
------------
Two-headed encoder feeding a shared quantile head:

  FlatEncoder:
    Input: x_flat (B, N_FLAT) + m_flat (B, N_FLAT)
    - NullEmbedding: learnable (N_FLAT, EMB_DIM) scalar per sparse column
    - Concat(x_flat, m_flat, null_emb) -> (B, N_FLAT * 3)
    - MLP: Linear -> LayerNorm -> GELU -> 2x ResidualBlock(hidden_dim)
    Output: (B, flat_hidden_dim)

  SeqEncoder (TFT-style):
    Input: x_seq (B, T, C_SEQ) + m_seq (B, T, C_SEQ)
    - Input projection: (B, T, C_SEQ*2) -> (B, T, seq_hidden_dim)
    - N x TransformerBlock with additive -inf masking on m_seq==0 timesteps
    - Global mean pool over observed timesteps
    Output: (B, seq_hidden_dim)

  QuantileHead:
    Input: concat(flat_enc, seq_enc) -> (B, flat_hidden + seq_hidden)
    - 2-layer MLP -> (B, 3) = [P10, P50, P90] transit time (hours)

Null handling (ADR-NE-002):
  - Phantom cols dropped upstream by loader -- never reach this model.
  - For each flat input position i:
      input_i = x_flat[i]          (0.0 where NULL was imputed)
      mask_i  = m_flat[i]          (0.0 if was NULL)
      emb_i   = null_embed[i]      (learnable; same value regardless of x)
    Concat of all three gives the network three separate signals:
      (1) the imputed value, (2) whether it was observed, (3) learned NULL token.
  - For sequence inputs: additive -inf attention bias applied on timesteps
    where ALL channels are unobserved (m_seq.sum(-1) == 0).

Feature flag
------------
  use_tft_pinn=True  -> this model
  use_tft_pinn=False -> legacy _SimpleTftModel (LSTM stub) in solarpipe_server.py

ONNX export
-----------
  Named inputs:  x_flat (B,105), m_flat (B,105), x_seq (B,T,C), m_seq (B,T,C)
  Named outputs: p10 (B,1), p50 (B,1), p90 (B,1)
  Opset <= 20  (enforced at export time in solarpipe_server.py)
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from feature_schema import FLAT_COLS, FLAT_SPARSE_COLS, N_SEQ_CHANNELS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_FLAT: int = len(FLAT_COLS)          # 105
N_SPARSE: int = len(FLAT_SPARSE_COLS) # 58 -- only these cols get null embedding


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    """Pre-norm residual block: LayerNorm -> Linear -> GELU -> Linear + skip."""

    def __init__(self, dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.ff1  = nn.Linear(dim, dim * 2)
        self.ff2  = nn.Linear(dim * 2, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.drop(F.gelu(self.ff1(h)))
        h = self.ff2(h)
        return x + h


class FlatEncoder(nn.Module):
    """Encodes the 105-dim flat feature vector with explicit null masking.

    For each flat column position i, the input to the network is:
        [x_flat_i, m_flat_i, null_embed_i]   (shape: 3 per column)

    Dense columns (m_flat always 1.0) still receive null_embed, which the
    network learns to ignore.  Sparse columns benefit from the distinct
    null token when the mask is 0.
    """

    def __init__(
        self,
        n_flat: int = N_FLAT,
        null_embedding_dim: int = 16,
        hidden_dim: int = 256,
        n_residual_blocks: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_flat = n_flat
        self.hidden_dim = hidden_dim     # stored for test/introspection
        self.emb_dim = null_embedding_dim

        # One learnable scalar embedding per flat column position.
        # The same embedding fires regardless of observed/NULL status;
        # the mask m_flat provides the binary signal.
        self.null_embed = nn.Embedding(n_flat, null_embedding_dim)

        # Input projection: x_flat + m_flat + null_embed per column
        in_dim = n_flat * (2 + null_embedding_dim)
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, dropout) for _ in range(n_residual_blocks)]
        )

    def forward(
        self, x_flat: torch.Tensor, m_flat: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x_flat: (B, N_FLAT) float32, NaN already replaced with 0.0
            m_flat: (B, N_FLAT) float32, 1.0=observed 0.0=was NULL

        Returns:
            (B, hidden_dim) float32
        """
        B = x_flat.size(0)
        # Null embeddings for all positions: (B, N_FLAT, emb_dim)
        idx = torch.arange(self.n_flat, device=x_flat.device)
        emb = self.null_embed(idx).unsqueeze(0).expand(B, -1, -1)
        emb_flat = emb.reshape(B, -1)   # (B, N_FLAT * emb_dim)

        # Concatenate: values + masks + embeddings
        h = torch.cat([x_flat, m_flat, emb_flat], dim=-1)   # (B, N_FLAT*(2+emb))
        h = self.proj(h)
        for block in self.blocks:
            h = block(h)
        return h


class SeqEncoder(nn.Module):
    """TFT-style transformer encoder for the OMNI pre-launch sequence.

    Applies additive -inf bias on fully-unobserved timesteps (all channels
    masked) so attention cannot attend to padding timesteps.

    Architecture:
      input_proj: (B, T, C*2) -> (B, T, hidden_dim)
      N x MultiheadAttention + FFN with pre-norm
      global mean pool over observed timesteps -> (B, hidden_dim)
    """

    def __init__(
        self,
        n_channels: int = N_SEQ_CHANNELS,
        hidden_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

        # Project concatenated (x_seq, m_seq) at each timestep
        self.input_proj = nn.Sequential(
            nn.Linear(n_channels * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Sinusoidal positional encoding (fixed)
        self.register_buffer(
            "pos_enc", self._sinusoidal_pe(512, hidden_dim), persistent=False
        )

        # Transformer layers.
        # enable_nested_tensor=False: nested tensor requires norm_first=False on
        # Maxwell (sm_52); with norm_first=True PyTorch silently falls back to
        # the generic (CPU-speed) path even on CUDA. Disabling it explicitly keeps
        # us on the cuBLAS-backed dense-tensor path (correct for sm_52 + RULE-300).
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,      # pre-norm (more stable)
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            enable_nested_tensor=False,   # Maxwell sm_52: force dense cuBLAS path
        )

    @staticmethod
    def _sinusoidal_pe(max_len: int, dim: int) -> torch.Tensor:
        pe = torch.zeros(max_len, dim)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[:dim // 2])
        return pe.unsqueeze(0)   # (1, max_len, dim)

    def forward(
        self, x_seq: torch.Tensor, m_seq: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x_seq: (B, T, C) float32, NaN replaced with 0.0
            m_seq: (B, T, C) float32, 1.0=observed 0.0=was NULL

        Returns:
            (B, hidden_dim) float32
        """
        B, T, C = x_seq.shape

        # Build key_padding_mask: True = ignore this timestep.
        # A timestep is fully unobserved when all channels are masked (0).
        # shape: (B, T)  bool
        # Use .any(dim=-1) rather than .sum(dim=-1) > 0 — traces to ReduceMax
        # (not ReduceSum), which the ORT CUDA EP cuDNN path handles on sm_52.
        observed_ts = (m_seq != 0).any(dim=-1)   # (B, T) -- True = has data
        key_padding_mask = ~observed_ts           # True = pad = ignore

        # Guard: if ALL timesteps are masked for a row, the Transformer softmax
        # produces NaN (softmax of all -inf). Unmask at least timestep 0
        # for those rows so the forward pass stays finite. The mean-pool
        # later zeroes out such rows via (denom > 0) guard anyway.
        all_masked = key_padding_mask.all(dim=-1)   # (B,)
        if all_masked.any():
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[all_masked, 0] = False

        # Input projection including mask as separate channel
        h = torch.cat([x_seq, m_seq], dim=-1)          # (B, T, C*2)
        h = self.input_proj(h)                          # (B, T, hidden_dim)

        # Add positional encoding
        h = h + self.pos_enc[:, :T, :].to(h.device)

        # Transformer with padding mask
        h = self.transformer(h, src_key_padding_mask=key_padding_mask)

        # Mean-pool over observed timesteps only via bmm — avoids 3D ReduceSum
        # which fails on ORT CUDA EP cuDNN path for sm_52 (Maxwell).
        # obs_float: (B, T, 1)
        # pooled = bmm(h^T, obs_float): (B, d, T) @ (B, T, 1) → (B, d, 1) → squeeze
        # denom  = bmm(obs^T, ones_T): (B, 1, T) @ (B, T, 1) → (B, 1, 1) → squeeze
        obs_float = observed_ts.unsqueeze(-1).float()                         # (B, T, 1)
        ones_T = torch.ones(B, T, 1, device=x_seq.device, dtype=torch.float32)
        pooled = torch.bmm(h.transpose(1, 2), obs_float).squeeze(-1)         # (B, hidden_dim)
        denom  = torch.bmm(obs_float.transpose(1, 2), ones_T).squeeze(-1)    # (B, 1)
        safe_denom = denom.clamp(min=1.0)
        pooled = pooled / safe_denom
        pooled = pooled * (denom > 0).float()
        return pooled


class QuantileHead(nn.Module):
    """MLP head producing P10, P50, P90 transit-time predictions (hours)."""

    def __init__(self, in_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_dim // 2, 3),   # P10, P50, P90
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Return (B, 3) predictions [p10, p50, p90] in transit hours."""
        return self.net(h)


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class TftPinnModel(nn.Module):
    """Two-headed TFT + PINN neural ensemble for CME transit time.

    Args:
        flat_hidden_dim:       output dim of FlatEncoder
        flat_n_residual_blocks: depth of FlatEncoder MLP
        null_embedding_dim:    learnable null token size (per flat column)
        seq_hidden_dim:        Transformer d_model
        seq_n_heads:           number of attention heads
        seq_n_layers:          number of Transformer encoder layers
        dropout:               dropout probability (all sub-modules)
    """

    def __init__(
        self,
        n_flat: int = N_FLAT,
        n_seq_channels: int = N_SEQ_CHANNELS,
        flat_hidden_dim: int = 256,
        flat_n_residual_blocks: int = 2,
        null_embedding_dim: int = 16,
        seq_hidden_dim: int = 128,
        seq_n_heads: int = 4,
        seq_n_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.flat_encoder = FlatEncoder(
            n_flat=n_flat,
            null_embedding_dim=null_embedding_dim,
            hidden_dim=flat_hidden_dim,
            n_residual_blocks=flat_n_residual_blocks,
            dropout=dropout,
        )
        self.seq_encoder = SeqEncoder(
            n_channels=n_seq_channels,
            hidden_dim=seq_hidden_dim,
            n_heads=seq_n_heads,
            n_layers=seq_n_layers,
            dropout=dropout,
        )
        self.head = QuantileHead(
            in_dim=flat_hidden_dim + seq_hidden_dim,
            dropout=dropout,
        )

    def forward(
        self,
        x_flat: torch.Tensor,
        m_flat: torch.Tensor,
        x_seq: torch.Tensor,
        m_seq: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x_flat: (B, N_FLAT)    float32
            m_flat: (B, N_FLAT)    float32
            x_seq:  (B, T, C_SEQ) float32
            m_seq:  (B, T, C_SEQ) float32

        Returns:
            preds: (B, 3) float32 -- [P10, P50, P90] transit hours
        """
        flat_enc = self.flat_encoder(x_flat, m_flat)   # (B, flat_hidden)
        seq_enc  = self.seq_encoder(x_seq, m_seq)      # (B, seq_hidden)
        combined = torch.cat([flat_enc, seq_enc], dim=-1)
        return self.head(combined)                     # (B, 3)

    def predict_p50(
        self,
        x_flat: torch.Tensor,
        m_flat: torch.Tensor,
        x_seq: torch.Tensor,
        m_seq: torch.Tensor,
    ) -> torch.Tensor:
        """Convenience: return only P50 (median) as (B, 1)."""
        return self.forward(x_flat, m_flat, x_seq, m_seq)[:, 1:2]


# ---------------------------------------------------------------------------
# Factory: build from YAML hyperparameters dict
# ---------------------------------------------------------------------------

def build_tft_pinn_model(hp: Optional[dict] = None) -> TftPinnModel:
    """Instantiate TftPinnModel from a hyperparameters dict (YAML-sourced).

    All keys are optional; defaults match neural_ensemble_v1.yaml.

    Args:
        hp: dict of hyperparameters, e.g. from StageConfig.Hyperparameters.

    Returns:
        TftPinnModel instance (on CPU; move to device in training loop).
    """
    hp = hp or {}
    return TftPinnModel(
        n_flat=N_FLAT,
        n_seq_channels=N_SEQ_CHANNELS,
        flat_hidden_dim=int(hp.get("flat_hidden_dim", 256)),
        flat_n_residual_blocks=int(hp.get("flat_n_residual_blocks", 2)),
        null_embedding_dim=int(hp.get("null_embedding_dim", 16)),
        seq_hidden_dim=int(hp.get("seq_hidden_dim", 128)),
        seq_n_heads=int(hp.get("seq_n_heads", 4)),
        seq_n_layers=int(hp.get("seq_n_layers", 2)),
        dropout=float(hp.get("seq_dropout", 0.1)),
    )
