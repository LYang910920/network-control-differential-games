import numpy as np
import pytest

from cybercontrol.io import require_outputs
from cybercontrol.models import MalwareParams, controlled_sir_rhs, isolation_jump
from cybercontrol.numerics import (
    as_time_function,
    project_box,
    project_simplex3,
    rk4_integrate,
    solve_ode_grid,
    trapezoid_integral,
)


def test_rk4_controlled_sir_preserves_simplex_after_projection():
    params = MalwareParams(beta=0.8, gamma=0.15, omega=0.02)
    x0 = np.array([0.95, 0.05, 0.0])
    rhs = lambda x, t: controlled_sir_rhs(x, u_patch=0.2, u_clean=0.1, p=params)

    xT, path = rk4_integrate(rhs, x0, t0=0.0, dt=1.0, substeps=20, project=project_simplex3)

    assert path.shape == (21, 3)
    assert np.all(xT >= 0.0)
    assert np.isclose(xT.sum(), 1.0)


def test_isolation_jump_is_impulsive_and_mass_preserving():
    x0 = np.array([0.75, 0.20, 0.05, 0.2])

    x_plus = isolation_jump(x0, isolation_rate=0.5)

    assert x_plus[1] < x0[1]
    assert x_plus[2] > x0[2]
    assert np.isclose(x_plus[:3].sum(), 1.0)
    assert np.isclose(x_plus[3], x0[3])


def test_grid_helpers_share_projection_quadrature_and_interpolation():
    t = np.linspace(0.0, 1.0, 11)
    clipped = project_box(np.array([-1.0, 0.25, 2.0]), 0.0, 1.0)
    assert np.allclose(clipped, [0.0, 0.25, 1.0])
    assert np.isclose(trapezoid_integral(t, t), 0.5)

    values = np.column_stack([t, t * t])
    value_at_half = as_time_function(t, values)(0.5)
    assert np.allclose(value_at_half, [0.5, 0.25])

    rhs_forward = lambda tau, y: np.array([1.0])
    forward = solve_ode_grid(rhs_forward, np.array([0.0]), t)
    backward = solve_ode_grid(rhs_forward, np.array([1.0]), t, backward=True)
    assert np.allclose(forward[:, 0], t)
    assert np.allclose(backward[:, 0], t)


def test_require_outputs_flags_missing_or_empty_files(tmp_path):
    good = tmp_path / "good.txt"
    empty = tmp_path / "empty.txt"
    good.write_text("ok\n", encoding="utf-8")
    empty.write_text("", encoding="utf-8")

    require_outputs([good])
    with pytest.raises(RuntimeError, match="empty"):
        require_outputs([empty])
    with pytest.raises(RuntimeError, match="missing"):
        require_outputs([tmp_path / "missing.txt"])


def test_torch_time_derivative_and_networks_keep_gradients():
    torch = pytest.importorskip("torch")
    from cybercontrol.torch_utils import BoundedControlNet, MLP, SimplexStateNet, time_derivative

    t = torch.linspace(0.0, 1.0, 8).view(-1, 1)
    t.requires_grad_(True)
    state = SimplexStateNet(width=8, depth=2)
    control = BoundedControlNet(width=8, depth=2, umax=0.7)
    costate = MLP(1, 3, width=8, depth=2)

    x = state(t)
    u = control(t)
    lam = costate(t)
    dxdt = time_derivative(x, t)
    loss = dxdt.pow(2).mean() + u.pow(2).mean() + lam.pow(2).mean()
    loss.backward()

    assert torch.allclose(x.sum(dim=1), torch.ones(8), atol=1e-6)
    assert torch.all((u >= 0.0) & (u <= 0.7))
    assert any(p.grad is not None for p in state.parameters())
    assert any(p.grad is not None for p in control.parameters())
