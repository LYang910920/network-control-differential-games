import numpy as np
import pytest

from cybercontrol.io import require_outputs
from cybercontrol.diagnostics import diagnostic_terms_for, rolling_mean, write_diagnostic_glossary
from cybercontrol.models import MalwareParams, controlled_sir_rhs, controlled_sir_rhs_torch, isolation_jump
from cybercontrol.network_models import (
    NodeSIPRSParams,
    graph_pressure_numpy,
    node_sips_rhs_numpy,
    node_siprs_rhs_numpy,
    node_siprs_rhs_torch,
    normalize_adjacency,
    sample_node_siprs_step,
)
from cybercontrol.numerics import (
    as_time_function,
    project_box,
    project_compartments,
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


def test_project_compartments_handles_node_and_batch_shapes():
    x = np.array([[0.2, -0.1, 0.4, 0.5], [0.0, 0.0, 0.0, 0.0]])
    projected = project_compartments(x)

    assert projected.shape == (2, 4)
    assert np.all(projected >= 0.0)
    assert np.allclose(projected.sum(axis=1), 1.0)
    assert np.allclose(projected[1], [1.0, 0.0, 0.0, 0.0])


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


def test_training_diagnostic_helpers_write_reader_glossary(tmp_path):
    assert np.allclose(rolling_mean([1.0, 3.0, float("nan"), 5.0], window=2), [1.0, 2.0, 3.0, 5.0], equal_nan=True)

    terms = diagnostic_terms_for(["iteration", "evaluation return", "stationarity loss"])
    assert [term.name for term in terms] == ["iteration", "evaluation return", "stationarity loss"]

    out = tmp_path / "diagnostics.md"
    write_diagnostic_glossary(out, terms, title="Test Diagnostics")
    text = out.read_text(encoding="utf-8")
    assert "evaluation return" in text
    assert "PMP consistency" in text


def test_torch_time_derivative_and_networks_keep_gradients():
    torch = pytest.importorskip("torch")
    from cybercontrol.torch_utils import BoundedControlNet, MLP, SimplexStateNet, project_compartments_torch, time_derivative

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

    projected = project_compartments_torch(torch.tensor([[0.2, -0.1, 0.4, 0.5], [0.0, 0.0, 0.0, 0.0]]))
    assert torch.allclose(projected.sum(dim=1), torch.ones(2))
    assert torch.allclose(projected[1], torch.tensor([1.0, 0.0, 0.0, 0.0]))


def test_controlled_sir_numpy_torch_parity():
    torch = pytest.importorskip("torch")
    params = MalwareParams(beta=0.7, gamma=0.11, omega=0.03)
    x = np.array([[0.8, 0.15, 0.05], [0.6, 0.30, 0.10]], dtype=np.float64)
    u_patch = np.array([[0.08], [0.02]], dtype=np.float64)
    u_clean = np.array([[0.03], [0.05]], dtype=np.float64)
    expected = np.vstack(
        [
            controlled_sir_rhs(row, float(up), float(uc), params)
            for row, up, uc in zip(x, u_patch[:, 0], u_clean[:, 0])
        ]
    )

    actual = controlled_sir_rhs_torch(
        torch.tensor(x),
        torch.tensor(u_patch),
        params.beta,
        params.gamma,
        u_clean=torch.tensor(u_clean),
        omega=params.omega,
    )

    assert np.allclose(actual.detach().numpy(), expected)


def test_node_sips_siprs_mass_and_sparse_pressure_parity():
    sp = pytest.importorskip("scipy.sparse")
    A_dense = normalize_adjacency(
        np.array(
            [
                [0.0, 1.0, 1.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
            ]
        )
    )
    A_sparse = normalize_adjacency(sp.csr_matrix(A_dense))
    x4 = np.array(
        [
            [0.80, 0.10, 0.05, 0.05],
            [0.70, 0.20, 0.05, 0.05],
            [0.65, 0.25, 0.05, 0.05],
        ]
    )
    params = NodeSIPRSParams(beta=0.9, gamma=0.2, omega_p=0.04, omega_r=0.03)
    rhs_dense = node_siprs_rhs_numpy(x4, A_dense, params, patch=0.05, clean=np.array([0.02, 0.03, 0.04]))
    rhs_sparse = node_siprs_rhs_numpy(x4, A_sparse, params, patch=0.05, clean=np.array([0.02, 0.03, 0.04]))

    assert np.allclose(graph_pressure_numpy(A_dense, x4[:, 1]), graph_pressure_numpy(A_sparse, x4[:, 1]))
    assert np.allclose(rhs_dense, rhs_sparse)
    assert np.allclose(rhs_dense.sum(axis=1), 0.0)

    x3 = project_compartments(x4[:, :3])
    sips_rhs = node_sips_rhs_numpy(x3, A_dense, params, patch=0.05, clean=0.02)
    assert np.allclose(sips_rhs.sum(axis=1), 0.0)


def test_node_siprs_torch_parity_and_stochastic_step():
    torch = pytest.importorskip("torch")
    A = normalize_adjacency(np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64))
    x = np.array([[0.8, 0.1, 0.05, 0.05], [0.6, 0.3, 0.05, 0.05]], dtype=np.float64)
    params = NodeSIPRSParams(beta=0.75, gamma=0.18, omega_p=0.02, omega_r=0.01)
    expected = node_siprs_rhs_numpy(x, A, params, patch=np.array([0.05, 0.02]), clean=0.03)
    actual = node_siprs_rhs_torch(
        torch.tensor(x),
        torch.tensor(A),
        params,
        patch=torch.tensor([0.05, 0.02]),
        clean=0.03,
    )
    assert np.allclose(actual.detach().numpy(), expected)

    sampled = sample_node_siprs_step(
        np.array([0, 1, 2, 3]),
        normalize_adjacency(np.ones((4, 4)) - np.eye(4)),
        params,
        dt=0.05,
        patch=0.1,
        clean=0.1,
        rng=np.random.default_rng(4),
    )
    assert sampled.shape == (4,)
    assert set(sampled.tolist()).issubset({0, 1, 2, 3})
