# Copyright (c) 2026 Lu-Xing Yang.
# No project-wide open-source license is granted; see repository COPYRIGHT_AND_LICENSE.md.

"""Regression checks for the sparse node-level FBS scalability helper."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from examples.foundations.code import scalability_analysis


def test_sparse_node_adjoint_matches_finite_difference_jacobian() -> None:
    """The adjoint RHS should be -L_x - J_f(x,u)^T lambda for the scaled costate."""
    A = sp.csr_matrix(
        np.array(
            [
                [0.0, 0.40, 0.20],
                [0.40, 0.0, 0.10],
                [0.20, 0.10, 0.0],
            ]
        )
    )
    x = np.array([0.20, 0.35, 0.10])
    u = np.array([0.05, 0.12, 0.08])
    lam = np.array([0.30, -0.20, 0.15])
    beta = 0.55
    delta = 0.20
    state_weight = 1.0

    def rhs(x_value: np.ndarray) -> np.ndarray:
        return scalability_analysis.sparse_node_rhs(x_value, u, A, beta=beta, delta=delta)

    eps = 1e-6
    jacobian = np.zeros((len(x), len(x)))
    for col in range(len(x)):
        direction = np.zeros_like(x)
        direction[col] = eps
        jacobian[:, col] = (rhs(x + direction) - rhs(x - direction)) / (2.0 * eps)

    pressure = A @ x
    actual = scalability_analysis.sparse_node_adjoint_rhs(
        lam,
        x,
        u,
        pressure,
        A,
        beta=beta,
        delta=delta,
        state_weight=state_weight,
    )
    expected = -state_weight - jacobian.T @ lam

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-8)
