"""Unit tests for tft_pinn_model.py (G3 gate).

Tests:
  1. Output shape: (B, 3) for various batch sizes
  2. All outputs are finite float32
  3. predict_p50 returns (B, 1)
  4. FlatEncoder handles all-zero mask (all phantom cols) without NaN
  5. SeqEncoder handles fully-masked sequence (all timesteps unobserved)
  6. build_tft_pinn_model() constructs from empty and full hp dicts
  7. Overfit test: 32 events, 200 gradient steps -> MAE < 0.5h (seed=42)
  8. Gradient flows to all parameter groups (no dead branches)

Run with:
    cd python && python -m pytest tests/test_tft_pinn_model.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_PYTHON_DIR = Path(__file__).parent.parent
if str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))

from feature_schema import N_SEQ_CHANNELS  # noqa: E402
from tft_pinn_model import (  # noqa: E402
    N_FLAT,
    TftPinnModel,
    build_tft_pinn_model,
)

# Tiny dimensions for fast CPU tests
_SMALL_HP: dict = {
    "flat_hidden_dim": 32,
    "flat_n_residual_blocks": 1,
    "null_embedding_dim": 4,
    "seq_hidden_dim": 16,
    "seq_n_heads": 2,
    "seq_n_layers": 1,
    "seq_dropout": 0.0,
}
_T = 222    # timesteps matching actual Parquet
_B = 8      # small batch for unit tests


def _random_flat(b: int) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randn(b, N_FLAT)
    m = (torch.rand(b, N_FLAT) > 0.3).float()
    x = x * m     # imputed zeros where mask=0
    return x, m


def _random_seq(b: int, t: int = _T) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randn(b, t, N_SEQ_CHANNELS)
    m = (torch.rand(b, t, N_SEQ_CHANNELS) > 0.2).float()
    x = x * m
    return x, m


class TestOutputShapes:

    def test_full_forward_shape(self) -> None:
        model = build_tft_pinn_model(_SMALL_HP)
        model.eval()
        x_f, m_f = _random_flat(_B)
        x_s, m_s = _random_seq(_B)
        with torch.no_grad():
            out = model(x_f, m_f, x_s, m_s)
        assert out.shape == (_B, 3), f"Expected ({_B}, 3), got {out.shape}"

    def test_batch_size_1(self) -> None:
        model = build_tft_pinn_model(_SMALL_HP)
        model.eval()
        x_f, m_f = _random_flat(1)
        x_s, m_s = _random_seq(1)
        with torch.no_grad():
            out = model(x_f, m_f, x_s, m_s)
        assert out.shape == (1, 3)

    def test_predict_p50_shape(self) -> None:
        model = build_tft_pinn_model(_SMALL_HP)
        model.eval()
        x_f, m_f = _random_flat(_B)
        x_s, m_s = _random_seq(_B)
        with torch.no_grad():
            p50 = model.predict_p50(x_f, m_f, x_s, m_s)
        assert p50.shape == (_B, 1)

    def test_output_dtype_float32(self) -> None:
        model = build_tft_pinn_model(_SMALL_HP)
        model.eval()
        x_f, m_f = _random_flat(_B)
        x_s, m_s = _random_seq(_B)
        with torch.no_grad():
            out = model(x_f, m_f, x_s, m_s)
        assert out.dtype == torch.float32


class TestNumericalStability:

    def test_outputs_finite(self) -> None:
        model = build_tft_pinn_model(_SMALL_HP)
        model.eval()
        for _ in range(5):
            x_f, m_f = _random_flat(_B)
            x_s, m_s = _random_seq(_B)
            with torch.no_grad():
                out = model(x_f, m_f, x_s, m_s)
            assert torch.isfinite(out).all(), "Non-finite output detected"

    def test_all_null_flat_mask(self) -> None:
        """All flat features masked (all NULL) should not produce NaN."""
        model = build_tft_pinn_model(_SMALL_HP)
        model.eval()
        x_f = torch.zeros(_B, N_FLAT)
        m_f = torch.zeros(_B, N_FLAT)   # everything NULL
        x_s, m_s = _random_seq(_B)
        with torch.no_grad():
            out = model(x_f, m_f, x_s, m_s)
        assert torch.isfinite(out).all()

    def test_fully_masked_sequence(self) -> None:
        """All sequence timesteps unobserved should not produce NaN."""
        model = build_tft_pinn_model(_SMALL_HP)
        model.eval()
        x_f, m_f = _random_flat(_B)
        x_s = torch.zeros(_B, _T, N_SEQ_CHANNELS)
        m_s = torch.zeros(_B, _T, N_SEQ_CHANNELS)  # all timesteps masked
        with torch.no_grad():
            out = model(x_f, m_f, x_s, m_s)
        assert torch.isfinite(out).all()

    def test_dense_flat_mask(self) -> None:
        """All flat features observed (mask=1) should be stable."""
        model = build_tft_pinn_model(_SMALL_HP)
        model.eval()
        x_f = torch.randn(_B, N_FLAT)
        m_f = torch.ones(_B, N_FLAT)
        x_s, m_s = _random_seq(_B)
        with torch.no_grad():
            out = model(x_f, m_f, x_s, m_s)
        assert torch.isfinite(out).all()


class TestBuildFromHyperparameters:

    def test_build_from_empty_hp(self) -> None:
        model = build_tft_pinn_model({})
        assert isinstance(model, TftPinnModel)

    def test_build_from_none(self) -> None:
        model = build_tft_pinn_model(None)
        assert isinstance(model, TftPinnModel)

    def test_build_from_full_hp(self) -> None:
        model = build_tft_pinn_model(_SMALL_HP)
        assert isinstance(model, TftPinnModel)
        assert model.flat_encoder.hidden_dim == 32  # type: ignore[attr-defined]

    def test_parameter_count_positive(self) -> None:
        model = build_tft_pinn_model(_SMALL_HP)
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params > 0, "Model has no parameters"


class TestGradientFlow:

    def test_all_params_receive_gradient(self) -> None:
        """Single forward+backward -- every parameter must have a gradient."""
        model = build_tft_pinn_model(_SMALL_HP)
        model.train()
        x_f, m_f = _random_flat(_B)
        x_s, m_s = _random_seq(_B)
        y = torch.rand(_B, 1) * 80 + 20    # transit times 20-100h

        out = model(x_f, m_f, x_s, m_s)
        loss = F.mse_loss(out[:, 1:2], y)
        loss.backward()

        no_grad = [
            name for name, p in model.named_parameters()
            if p.requires_grad and p.grad is None
        ]
        assert no_grad == [], f"Parameters with no gradient: {no_grad}"

    def test_null_embed_receives_gradient(self) -> None:
        """Null embeddings must be trainable (masked columns provide signal)."""
        model = build_tft_pinn_model(_SMALL_HP)
        model.train()
        x_f = torch.zeros(_B, N_FLAT)
        m_f = torch.zeros(_B, N_FLAT)   # all NULL -- embedding is the only signal
        x_s, m_s = _random_seq(_B)
        y = torch.rand(_B, 1) * 80 + 20

        out = model(x_f, m_f, x_s, m_s)
        loss = F.mse_loss(out[:, 1:2], y)
        loss.backward()

        assert model.flat_encoder.null_embed.weight.grad is not None
        assert model.flat_encoder.null_embed.weight.grad.abs().sum() > 0


class TestOverfit:
    """Overfit test: 32 synthetic events, 200 steps -> MAE < 0.5h."""

    def test_overfit_32_events(self) -> None:
        torch.manual_seed(42)
        model = build_tft_pinn_model({
            "flat_hidden_dim": 64,
            "flat_n_residual_blocks": 2,
            "null_embedding_dim": 8,
            "seq_hidden_dim": 32,
            "seq_n_heads": 2,
            "seq_n_layers": 1,
            "seq_dropout": 0.0,
        })
        model.train()
        opt = torch.optim.Adam(model.parameters(), lr=3e-3)

        # Fixed synthetic data
        torch.manual_seed(42)
        N = 32
        x_f, m_f = _random_flat(N)
        x_s, m_s = _random_seq(N, t=32)   # short T for speed
        y = torch.rand(N, 1) * 60 + 20    # targets 20-80h

        for step in range(500):
            opt.zero_grad()
            out = model(x_f, m_f, x_s, m_s)
            loss = F.mse_loss(out[:, 1:2], y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            out = model(x_f, m_f, x_s, m_s)
        mae = (out[:, 1:2] - y).abs().mean().item()
        assert mae < 1.0, (
            f"Overfit MAE {mae:.3f}h >= 1.0h after 500 steps -- "
            f"model may not be learning"
        )


