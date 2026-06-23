import numpy as np
import pytest

from cybercontrol.io import require_outputs
from cybercontrol.diagnostics import diagnostic_terms_for, rolling_mean, write_diagnostic_glossary
from cybercontrol.models import MalwareParams, controlled_sir_rhs, controlled_sir_rhs_torch, isolation_jump
from cybercontrol.network_models import (
    NodeSIPSParams,
    community_correlated_node_sips_params,
    contiguous_community_index,
    graph_pressure_numpy,
    node_sips_rhs_numpy,
    node_sips_rhs_torch,
    node_sips_transition_rates,
    normalize_adjacency,
    sample_node_sips_step,
)
from cybercontrol.heterogeneity import (
    DegreeSISParams,
    degree_correlated_sis_params,
    degree_sis_jacobian,
    degree_sis_rhs,
    finite_difference_jacobian,
    neutral_degree_mixing,
    node_heterogeneity_summary,
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
from cybercontrol.plotting import figure_size, publication_style, save_publication_figure


def test_rk4_controlled_sir_preserves_simplex_after_projection():
    params = MalwareParams(beta=0.8, gamma=0.15, omega=0.02)
    x0 = np.array([0.95, 0.05, 0.0])

    def rhs(x, t):
        return controlled_sir_rhs(x, u_patch=0.2, u_clean=0.1, p=params)

    xT, path = rk4_integrate(rhs, x0, t0=0.0, dt=1.0, substeps=20, project=project_simplex3)

    assert path.shape == (21, 3)
    assert np.all(xT >= 0.0)
    assert np.isclose(xT.sum(), 1.0)


def test_isolation_jump_is_impulsive_and_mass_preserving():
    x0 = np.array([0.75, 0.20, 0.05])

    x_plus = isolation_jump(x0, isolation_rate=0.5)

    assert x_plus[1] < x0[1]
    assert x_plus[2] > x0[2]
    assert np.isclose(x_plus.sum(), 1.0)


def test_grid_helpers_share_projection_quadrature_and_interpolation():
    t = np.linspace(0.0, 1.0, 11)
    clipped = project_box(np.array([-1.0, 0.25, 2.0]), 0.0, 1.0)
    assert np.allclose(clipped, [0.0, 0.25, 1.0])
    assert np.isclose(trapezoid_integral(t, t), 0.5)

    values = np.column_stack([t, t * t])
    value_at_half = as_time_function(t, values)(0.5)
    assert np.allclose(value_at_half, [0.5, 0.25])

    def rhs_forward(tau, y):
        return np.array([1.0])

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


def test_publication_figure_export_writes_vector_raster_and_manifest(tmp_path):
    import json

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with publication_style():
        fig, ax = plt.subplots(figsize=figure_size("single", ratio=0.6))
        ax.plot([0, 1], [0, 1], label="diagnostic")
        ax.set_xlabel("iteration")
        ax.set_ylabel("loss")
        ax.legend(frameon=False)
        written = save_publication_figure(
            fig,
            tmp_path / "diagnostic_panel",
            metadata={"source": "unit-test", "meaning": "publication export contract"},
        )
        plt.close(fig)

    assert (tmp_path / "diagnostic_panel.pdf") in written
    assert (tmp_path / "diagnostic_panel.png") in written
    manifest = tmp_path / "diagnostic_panel.figure.json"
    assert manifest in written

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["style"] == "publication"
    assert payload["dpi"] == 600
    assert payload["formats"] == ["pdf", "png"]
    assert payload["width_in"] > 0
    assert payload["height_in"] > 0
    assert payload["metadata"]["source"] == "unit-test"


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
    from cybercontrol.torch_utils import (
        BoundedControlNet,
        MLP,
        SimplexStateNet,
        configure_torch,
        project_compartments_torch,
        time_derivative,
    )

    _, resolved_device, resolved_dtype = configure_torch(seed=0, device="cpu", threads=1)
    assert resolved_device == "cpu"
    assert resolved_dtype == torch.float32

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


def test_node_sips_mass_and_sparse_pressure_parity():
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
    x = np.array(
        [
            [0.85, 0.10, 0.05],
            [0.75, 0.20, 0.05],
            [0.70, 0.25, 0.05],
        ]
    )
    params = NodeSIPSParams(beta=0.9, gamma=0.2, omega=0.04)
    rhs_dense = node_sips_rhs_numpy(x, A_dense, params, patch=0.05, clean=np.array([0.02, 0.03, 0.04]))
    rhs_sparse = node_sips_rhs_numpy(x, A_sparse, params, patch=0.05, clean=np.array([0.02, 0.03, 0.04]))

    assert np.allclose(graph_pressure_numpy(A_dense, x[:, 1]), graph_pressure_numpy(A_sparse, x[:, 1]))
    assert np.allclose(rhs_dense, rhs_sparse)
    assert np.allclose(rhs_dense.sum(axis=1), 0.0)
    rates = node_sips_transition_rates(x, A_dense, params, patch=0.05, clean=0.02)
    assert {"infection", "patch", "clean", "recovery", "waning"}.issubset(rates)


def test_degree_sis_heterogeneity_preserves_scalar_compatibility_and_jacobian():
    k = np.array([1.0, 3.0, 6.0])
    p = np.array([0.5, 0.3, 0.2])
    x = np.array([0.08, 0.16, 0.24])
    u = np.array([0.02, 0.04, 0.06])
    scalar = DegreeSISParams(susceptibility=0.7, recovery=0.18).resolve(k, p)

    theta = float((k * p) @ x / (k @ p))
    expected = 0.7 * k * (1.0 - x) * theta - (0.18 + u) * x
    assert np.allclose(degree_sis_rhs(x, u, k, scalar), expected)
    assert np.allclose(neutral_degree_mixing(k, p).sum(axis=1), 1.0)

    heterogeneous = degree_correlated_sis_params(k, p, strength=0.25)
    analytic = degree_sis_jacobian(x, u, k, heterogeneous)
    numeric = finite_difference_jacobian(lambda y: degree_sis_rhs(y, u, k, heterogeneous), x)
    assert np.allclose(analytic, numeric, rtol=1e-5, atol=1e-7)
    assert heterogeneous.summaries()[0]["name"] == "susceptibility"


def test_node_sips_heterogeneous_arrays_dense_sparse_and_summaries():
    sp = pytest.importorskip("scipy.sparse")
    A = normalize_adjacency(
        np.array(
            [
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
            ]
        )
    )
    x = np.array(
        [
            [0.85, 0.10, 0.05],
            [0.77, 0.18, 0.05],
            [0.73, 0.22, 0.05],
        ]
    )
    params = NodeSIPSParams(
        beta=np.array([0.7, 0.8, 0.9]),
        susceptibility=np.array([0.9, 1.1, 1.3]),
        infectivity=np.array([1.2, 0.8, 1.0]),
        gamma=np.array([0.12, 0.16, 0.20]),
        omega=np.array([0.02, 0.03, 0.04]),
        criticality=np.array([1.0, 1.5, 2.0]),
        patch_bound=np.array([0.03, 0.04, 0.05]),
        clean_bound=0.06,
    )
    rhs_dense = node_sips_rhs_numpy(x, A, params, patch=0.5, clean=np.array([0.02, 0.07, 0.08]))
    rhs_sparse = node_sips_rhs_numpy(x, sp.csr_matrix(A), params, patch=0.5, clean=np.array([0.02, 0.07, 0.08]))

    assert np.allclose(rhs_dense, rhs_sparse)
    assert np.allclose(rhs_dense.sum(axis=1), 0.0)
    resolved = params.resolve(3)
    summary = node_heterogeneity_summary(resolved)
    assert summary.shape == (4,)
    assert resolved.risk_score()[2] > resolved.risk_score()[0]


def test_node_sips_torch_parity_and_stochastic_step():
    torch = pytest.importorskip("torch")
    A = normalize_adjacency(np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64))
    x = np.array([[0.85, 0.1, 0.05], [0.65, 0.3, 0.05]], dtype=np.float64)
    params = NodeSIPSParams(beta=0.75, gamma=0.18, omega=0.02)
    expected = node_sips_rhs_numpy(x, A, params, patch=np.array([0.05, 0.02]), clean=0.03)
    actual = node_sips_rhs_torch(
        torch.tensor(x),
        torch.tensor(A),
        params,
        patch=torch.tensor([0.05, 0.02]),
        clean=0.03,
    )
    assert np.allclose(actual.detach().numpy(), expected)

    sampled = sample_node_sips_step(
        np.array([0, 1, 2]),
        normalize_adjacency(np.ones((3, 3)) - np.eye(3)),
        params,
        dt=0.05,
        patch=0.1,
        clean=0.1,
        rng=np.random.default_rng(4),
    )
    assert sampled.shape == (3,)
    assert set(sampled.tolist()).issubset({0, 1, 2})


def test_node_sips_heterogeneous_torch_parity():
    torch = pytest.importorskip("torch")
    community = np.array([0, 0, 1, 1])
    params = community_correlated_node_sips_params(community, strength=0.4)
    A = normalize_adjacency(np.ones((4, 4)) - np.eye(4))
    x = project_compartments(
        np.array(
            [
                [0.86, 0.10, 0.04],
                [0.75, 0.20, 0.05],
                [0.70, 0.25, 0.05],
                [0.81, 0.14, 0.05],
            ]
        )
    )
    patch = np.array([0.08, 0.02, 0.04, 0.01])
    clean = np.array([0.01, 0.03, 0.02, 0.04])

    expected = node_sips_rhs_numpy(x, A, params, patch=patch, clean=clean)
    actual = node_sips_rhs_torch(
        torch.tensor(x),
        torch.tensor(A),
        params,
        patch=torch.tensor(patch),
        clean=torch.tensor(clean),
    )
    assert np.allclose(actual.detach().numpy(), expected, rtol=1e-6, atol=1e-7)


def test_contiguous_community_index_is_validated_and_deterministic():
    labels = contiguous_community_index(8, 3)

    assert labels.tolist() == [0, 0, 0, 1, 1, 1, 2, 2]
    assert labels.dtype.kind in {"i", "u"}
    with pytest.raises(ValueError):
        contiguous_community_index(0, 3)
    with pytest.raises(ValueError):
        contiguous_community_index(8, 0)


def test_foundation_baseline_evaluators_use_same_heterogeneous_problem():
    from examples.foundations.code import network_control_examples as ex

    D = ex.DegreeData(
        k=np.array([1.0, 3.0, 6.0]),
        Nk=np.array([4, 3, 2]),
        pk=np.array([4 / 9, 3 / 9, 2 / 9], dtype=float),
        kbar=float(np.array([1.0, 3.0, 6.0]) @ np.array([4 / 9, 3 / 9, 2 / 9])),
        node_degree=np.array([1, 1, 1, 1, 3, 3, 3, 6, 6]),
    )
    t = np.linspace(0.0, 1.0, 6)
    u = np.full((len(t), len(D.k)), 0.05)
    x, cost = ex.evaluate_degree_control_policy(D, t, u)
    model = ex.degree_control_model(D)
    expected_cost = ex.integral((x * model.state_weight) @ D.pk + 0.5 * ((u * u * model.control_weight) @ D.pk), t)

    assert np.isclose(cost, expected_cost)
    assert not np.allclose(model.state_weight, model.state_weight.mean())

    A = ex.normalize_adjacency(
        np.array(
            [
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
            ]
        ),
        "row",
    )
    attack = np.full((len(t), A.shape[0]), 0.03)
    defense = np.full((len(t), A.shape[0]), 0.04)
    x_game, ja, jd = ex.evaluate_node_game_strategy(A, t, attack, defense)
    game_model = ex.node_game_model(A)
    expected_ja = ex.integral((game_model.attack_reward * x_game).sum(axis=1) - 0.5 * (attack * attack * game_model.attack_cost).sum(axis=1), t)
    expected_jd = ex.integral(-(game_model.defense_loss * x_game).sum(axis=1) - 0.5 * (defense * defense * game_model.defense_cost).sum(axis=1), t)

    assert np.isclose(ja, expected_ja)
    assert np.isclose(jd, expected_jd)
