"""Unit tests for physics_loss.py (G4 gate).

Coverage:
  1. Pinball loss: shape, positivity, zero at perfect prediction, gradient
  2. Quantile ordering loss: zero when ordered, positive when violated, grad
  3. Transit bound loss: zero inside range, positive outside, grad
  4. Drag ODE residual: unit sanity (C# parity), slow-dense CME arrives later
     than fast-sparse CME, gradient wrt pred_transit_hours
  5. Monotonic deceleration: zero when decelerating, positive on speedup, grad
  6. PinnLoss aggregation: total = pinball + weighted components; lambda=0 off
  7. build_pinn_loss factory: reads YAML dict, default + override paths
  8. Gradient finite-difference check on pinball + qorder + bound

Run with:
    cd python && python -m pytest tests/test_physics_loss.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

_PYTHON_DIR = Path(__file__).parent.parent
if str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))

from physics_loss import (  # noqa: E402
    PinnLoss,
    build_pinn_loss,
    drag_ode_residual_loss,
    monotonic_deceleration_loss,
    pinball_loss,
    quantile_ordering_loss,
    transit_bound_loss,
)

_B = 8
_T = 32
_C = 22  # matches N_SEQ_CHANNELS


def _make_preds(b: int = _B) -> torch.Tensor:
    """Monotone [P10, P50, P90] predictions in [20, 80] hours."""
    p50 = torch.rand(b) * 40 + 30
    return torch.stack([p50 - 5, p50, p50 + 5], dim=1).requires_grad_(True)


def _make_targets(b: int = _B) -> torch.Tensor:
    return (torch.rand(b, 1) * 40 + 30)


# ---------------------------------------------------------------------------
# Pinball loss
# ---------------------------------------------------------------------------
class TestPinballLoss:
    def test_shape_scalar(self) -> None:
        preds = _make_preds()
        y = _make_targets()
        loss = pinball_loss(preds, y)
        assert loss.shape == torch.Size([])

    def test_positive(self) -> None:
        preds = _make_preds()
        y = _make_targets()
        loss = pinball_loss(preds, y)
        assert loss.item() >= 0.0

    def test_zero_on_perfect_p50(self) -> None:
        """All three quantiles exactly at target -> loss collapses to zero."""
        y = torch.tensor([[50.0], [50.0]])
        preds = torch.tensor([[50.0, 50.0, 50.0], [50.0, 50.0, 50.0]])
        loss = pinball_loss(preds, y)
        assert loss.item() < 1e-6

    def test_gradient_flows(self) -> None:
        preds = _make_preds()
        y = _make_targets()
        loss = pinball_loss(preds, y)
        loss.backward()
        assert preds.grad is not None
        assert preds.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# Quantile ordering
# ---------------------------------------------------------------------------
class TestQuantileOrdering:
    def test_zero_when_ordered(self) -> None:
        preds = torch.tensor([[20.0, 40.0, 60.0], [30.0, 45.0, 70.0]])
        loss = quantile_ordering_loss(preds)
        assert loss.item() < 1e-9

    def test_positive_on_violation(self) -> None:
        # P10 > P50 and P50 > P90 -- violates ordering
        preds = torch.tensor([[50.0, 30.0, 10.0]])
        loss = quantile_ordering_loss(preds)
        assert loss.item() > 0.0

    def test_gradient_penalises_violation(self) -> None:
        preds = torch.tensor([[50.0, 30.0, 20.0]], requires_grad=True)
        loss = quantile_ordering_loss(preds)
        loss.backward()
        # P10 grad should be positive (pushing it down) since P10 > P50
        assert preds.grad is not None
        assert preds.grad[0, 0].item() > 0


# ---------------------------------------------------------------------------
# Transit bound
# ---------------------------------------------------------------------------
class TestTransitBound:
    def test_zero_inside_range(self) -> None:
        preds = torch.full((4, 3), 50.0)   # well within [12, 120]
        loss = transit_bound_loss(preds)
        assert loss.item() < 1e-9

    def test_positive_below_min(self) -> None:
        preds = torch.full((4, 3), 5.0)    # below T_MIN=12
        loss = transit_bound_loss(preds)
        assert loss.item() > 0.0

    def test_positive_above_max(self) -> None:
        preds = torch.full((4, 3), 200.0)  # above T_MAX=120
        loss = transit_bound_loss(preds)
        assert loss.item() > 0.0

    def test_gradient_pushes_inward(self) -> None:
        preds = torch.full((1, 3), 5.0, requires_grad=True)  # below min
        loss = transit_bound_loss(preds)
        loss.backward()
        # grad < 0 means pred must increase to decrease loss
        assert preds.grad is not None
        assert (preds.grad < 0).all()


# ---------------------------------------------------------------------------
# Drag ODE residual -- physics sanity
# ---------------------------------------------------------------------------
class TestDragOdeResidual:
    def test_shape_scalar(self) -> None:
        b = 4
        pred_t = torch.tensor([40.0, 50.0, 60.0, 70.0], requires_grad=True)
        v0 = torch.full((b,), 1000.0)
        v_sw = torch.full((b,), 400.0)
        dens = torch.full((b, _T), 5.0)
        mask = torch.ones(b, _T)
        loss = drag_ode_residual_loss(pred_t, v0, v_sw, dens, mask)
        assert loss.shape == torch.Size([])

    def test_unit_sanity_matches_cs_drag_range(self) -> None:
        """With gamma0=0.5e-7 km^-1, n=n_ref, v0=1000, v_sw=400:
        integrated arrival should be 50-80h (physical range, not 0 or inf).

        This catches the cm^-1 vs km^-1 bug: at gamma=2e-13 km^-1 the CME
        wouldn't decelerate and arrival would hit the 120h integration floor.
        """
        v0 = torch.tensor([1000.0])
        v_sw = torch.tensor([400.0])
        dens = torch.full((1, _T), 5.0)
        mask = torch.ones(1, _T)

        # Duplicate the integration to read arrival time directly:
        # the loss is MSE against pred_transit, so a pred that matches ODE gives 0.
        # Sweep pred_t over a wide range, find the minimum -> that's the ODE arrival.
        losses = []
        for t in range(20, 100, 2):
            pt = torch.tensor([float(t)])
            loss = drag_ode_residual_loss(pt, v0, v_sw, dens, mask)
            losses.append((t, loss.item()))
        best_t = min(losses, key=lambda x: x[1])[0]
        # Physical expectation: 1000 km/s CME in 5/cm3 ambient decelerates to
        # roughly 500-600 km/s by 1 AU, giving ~50-80h transit.
        assert 40 <= best_t <= 85, (
            f"ODE arrival time {best_t}h outside physical range "
            f"[40, 85] -- likely unit bug in gamma or density"
        )

    def test_dense_plasma_slower_than_sparse(self) -> None:
        """High density -> high gamma_eff -> more deceleration -> later arrival.

        If gamma_eff scaling is correct, dense-plasma ODE arrival time must
        be strictly greater than sparse-plasma for the same v0.
        """
        v0 = torch.tensor([1000.0])
        v_sw = torch.tensor([400.0])
        mask = torch.ones(1, _T)

        dens_low = torch.full((1, _T), 2.0)      # 2 cm^-3 (sparse)
        dens_high = torch.full((1, _T), 20.0)    # 20 cm^-3 (dense)

        # Find ODE arrival by minimising loss over pred_t sweep
        def arrival(dens: torch.Tensor) -> int:
            losses = []
            for t in range(20, 120, 2):
                pt = torch.tensor([float(t)])
                loss = drag_ode_residual_loss(pt, v0, v_sw, dens, mask)
                losses.append((t, loss.item()))
            return min(losses, key=lambda x: x[1])[0]

        t_sparse = arrival(dens_low)
        t_dense = arrival(dens_high)
        assert t_dense > t_sparse, (
            f"Expected dense plasma arrival ({t_dense}h) > "
            f"sparse arrival ({t_sparse}h); density scaling broken"
        )

    def test_gradient_flows_to_pred_transit(self) -> None:
        pred_t = torch.tensor([50.0, 60.0], requires_grad=True)
        v0 = torch.tensor([1000.0, 800.0])
        v_sw = torch.tensor([400.0, 400.0])
        dens = torch.full((2, _T), 5.0)
        mask = torch.ones(2, _T)
        loss = drag_ode_residual_loss(pred_t, v0, v_sw, dens, mask)
        loss.backward()
        assert pred_t.grad is not None
        assert pred_t.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# Monotonic deceleration
# ---------------------------------------------------------------------------
class TestMonotonicDeceleration:
    def test_zero_on_pure_deceleration(self) -> None:
        # Monotonically decreasing speed while above v_sw=400
        speed = torch.tensor([[800.0, 700.0, 600.0, 500.0, 450.0]])
        mask = torch.ones(1, 5)
        v_sw = torch.tensor([400.0])
        loss = monotonic_deceleration_loss(speed, mask, v_sw)
        assert loss.item() < 1e-9

    def test_positive_on_speedup_above_ambient(self) -> None:
        # Speed increases while above v_sw -> physics violation
        speed = torch.tensor([[500.0, 700.0, 900.0]])
        mask = torch.ones(1, 3)
        v_sw = torch.tensor([400.0])
        loss = monotonic_deceleration_loss(speed, mask, v_sw)
        assert loss.item() > 0.0

    def test_zero_when_below_ambient(self) -> None:
        """Speedup BELOW ambient wind should NOT trigger the penalty."""
        speed = torch.tensor([[200.0, 300.0, 350.0]])   # all < v_sw=400
        mask = torch.ones(1, 3)
        v_sw = torch.tensor([400.0])
        loss = monotonic_deceleration_loss(speed, mask, v_sw)
        assert loss.item() < 1e-9

    def test_mask_blocks_unobserved_pairs(self) -> None:
        # Speedup on a masked pair must not contribute to loss
        speed = torch.tensor([[500.0, 900.0]])
        mask = torch.tensor([[1.0, 0.0]])   # t=1 unobserved
        v_sw = torch.tensor([400.0])
        loss = monotonic_deceleration_loss(speed, mask, v_sw)
        assert loss.item() < 1e-9


# ---------------------------------------------------------------------------
# Combined PinnLoss
# ---------------------------------------------------------------------------
class TestPinnLoss:
    def _make_batch(self) -> dict:
        torch.manual_seed(0)
        preds = _make_preds(_B)
        y = _make_targets(_B)
        v0 = torch.full((_B,), 1000.0)
        v_sw = torch.full((_B,), 400.0)
        x_seq = torch.randn(_B, _T, _C)
        m_seq = torch.ones(_B, _T, _C)
        return dict(
            preds=preds, targets=y, v0_kms=v0, v_sw_kms=v_sw,
            x_seq=x_seq, m_seq=m_seq,
        )

    def test_returns_all_components(self) -> None:
        loss_fn = PinnLoss(lambda_ode=1.0, lambda_mono=1.0)
        out = loss_fn(**self._make_batch())
        for key in ["total", "pinball", "ode", "mono", "bound", "qorder"]:
            assert key in out, f"Missing {key} in loss dict"

    def test_total_is_sum_of_weighted_components(self) -> None:
        loss_fn = PinnLoss(
            lambda_ode=0.0, lambda_mono=0.0,
            lambda_bound=0.3, lambda_qorder=0.4,
        )
        batch = self._make_batch()
        out = loss_fn(**batch)
        expected = (
            out["pinball"] + 0.3 * out["bound"] + 0.4 * out["qorder"]
        )
        assert torch.allclose(out["total"], expected, atol=1e-5)

    def test_lambda_zero_skips_ode_path(self) -> None:
        """With lambda_ode=0 and lambda_mono=0, those losses stay at zero."""
        loss_fn = PinnLoss(lambda_ode=0.0, lambda_mono=0.0)
        out = loss_fn(**self._make_batch())
        assert out["ode"].item() == 0.0
        assert out["mono"].item() == 0.0

    def test_total_differentiable(self) -> None:
        loss_fn = PinnLoss(lambda_ode=0.0, lambda_mono=0.0)
        batch = self._make_batch()
        out = loss_fn(**batch)
        out["total"].backward()
        assert batch["preds"].grad is not None
        assert batch["preds"].grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# YAML factory
# ---------------------------------------------------------------------------
class TestBuildPinnLoss:
    def test_empty_dict_defaults(self) -> None:
        loss = build_pinn_loss({})
        assert loss.lambda_ode == 0.0
        assert loss.lambda_mono == 0.0
        assert loss.lambda_bound == 0.1
        assert loss.lambda_qorder == 0.5
        assert loss.gamma0_km_inv == 0.5e-7    # C# parity
        assert loss.n_ref_cm3 == 5.0

    def test_none_yields_defaults(self) -> None:
        loss = build_pinn_loss(None)
        assert isinstance(loss, PinnLoss)
        assert loss.gamma0_km_inv == 0.5e-7

    def test_override_from_yaml_dict(self) -> None:
        loss = build_pinn_loss({
            "lambda_ode": 0.5,
            "lambda_bound": 0.2,
            "drag_gamma_km_inv": 1.0e-7,
        })
        assert loss.lambda_ode == 0.5
        assert loss.lambda_bound == 0.2
        assert loss.gamma0_km_inv == 1.0e-7


# ---------------------------------------------------------------------------
# Finite-difference gradient check on differentiable components
# ---------------------------------------------------------------------------
class TestFiniteDifferenceGradient:
    """Compare autograd to numerical gradient at a random point."""

    @staticmethod
    def _fd_grad(fn, x: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
        # FD requires float64 to resolve O(eps^2) truncation without float32
        # rounding swamping the signal (autograd uses fp32 here — OK because
        # analytical grads are exact; FD is the noisy side).
        x = x.to(torch.float64)
        g = torch.zeros_like(x)
        flat = x.view(-1)
        gf = g.view(-1)
        for i in range(flat.numel()):
            orig = flat[i].item()
            flat[i] = orig + eps
            fp = fn(x).item()
            flat[i] = orig - eps
            fm = fn(x).item()
            flat[i] = orig
            gf[i] = (fp - fm) / (2 * eps)
        return g.to(torch.float32)

    def test_pinball_gradient_matches_fd(self) -> None:
        torch.manual_seed(1)
        preds = torch.randn(3, 3) * 10 + 50
        y = torch.randn(3, 1) * 10 + 50

        preds_auto = preds.clone().requires_grad_(True)
        pinball_loss(preds_auto, y).backward()
        grad_auto = preds_auto.grad

        grad_fd = self._fd_grad(
            lambda p: pinball_loss(p, y.to(torch.float64)), preds.clone()
        )
        # pinball has kinks (non-differentiable at err=0); allow generous tolerance
        assert torch.allclose(grad_auto, grad_fd, atol=5e-3)

    def test_transit_bound_gradient_matches_fd(self) -> None:
        torch.manual_seed(2)
        # Include points both inside and outside the bound
        preds = torch.tensor([[5.0, 50.0, 150.0], [15.0, 70.0, 200.0]])

        preds_auto = preds.clone().requires_grad_(True)
        transit_bound_loss(preds_auto).backward()
        grad_auto = preds_auto.grad

        grad_fd = self._fd_grad(
            lambda p: transit_bound_loss(p), preds.clone()
        )
        assert torch.allclose(grad_auto, grad_fd, rtol=1e-3, atol=1e-3)

    def test_qorder_gradient_matches_fd(self) -> None:
        # Construct violating preds so loss is non-zero
        preds = torch.tensor([[50.0, 30.0, 20.0], [40.0, 45.0, 35.0]])

        preds_auto = preds.clone().requires_grad_(True)
        quantile_ordering_loss(preds_auto).backward()
        grad_auto = preds_auto.grad

        grad_fd = self._fd_grad(
            lambda p: quantile_ordering_loss(p), preds.clone()
        )
        assert torch.allclose(grad_auto, grad_fd, rtol=1e-3, atol=1e-3)


# ---------------------------------------------------------------------------
# Smoke test: end-to-end loss+backward through a real forward signal
# ---------------------------------------------------------------------------
def test_end_to_end_training_step() -> None:
    """Simulate one training step: model preds -> PinnLoss -> backward."""
    torch.manual_seed(42)
    loss_fn = PinnLoss(
        lambda_ode=0.0, lambda_mono=0.0,
        lambda_bound=0.1, lambda_qorder=0.5,
    )
    # Synthetic "model output" -- create leaf tensor then enable grad
    preds = (torch.randn(_B, 3) * 10 + 50).requires_grad_(True)
    y = torch.rand(_B, 1) * 40 + 30
    v0 = torch.full((_B,), 1000.0)
    v_sw = torch.full((_B,), 400.0)
    x_seq = torch.randn(_B, _T, _C)
    m_seq = torch.ones(_B, _T, _C)

    out = loss_fn(preds, y, v0, v_sw, x_seq, m_seq)
    out["total"].backward()

    assert torch.isfinite(out["total"]).all()
    assert preds.grad is not None
    assert preds.grad.abs().sum() > 0
