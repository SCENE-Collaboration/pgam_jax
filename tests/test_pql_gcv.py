"""
Checks of the analytic GCV gradient.

``gcv_compute_factory`` builds a custom-VJP objective whose forward pass uses
``PenaltyHandler.compute_sqrt`` and whose backward pass reconstructs the
gradient from the per-term matrices in ``penalty_tree``. Those are two
independent representations of the same penalty, so a mismatch (lambda
ordering, identifiability drop, block offset) only shows up as a gradient
error. The finite-difference test pins the forward/backward pairing in the
exact configuration ``GAM.fit`` uses: ``penalize_null_space=True`` with
per-component column drop, exercising SINGLE_WITH_NULL and KRONECKER_WITH_NULL.

The dense test compares ``_gcv_grad_compute_from_states`` with the previous
dense calculation on wide and tall designs, with a tight tolerance.
"""

import jax
import jax.numpy as jnp
import nemos as nmo
import numpy as np
import pytest
from jax.flatten_util import ravel_pytree

from pgam_jax import GAM
from pgam_jax._pql_gcv import _gcv_grad_compute_from_states, gcv_compute_factory
from pgam_jax.penalty_utils import (
    DROP_LAST_ROW_COL,
    IDENTITY,
    compute_penalty_blocks,
    identifiability_column_fns_per_component,
    identifiability_square_fns_per_component,
)

jax.config.update("jax_enable_x64", True)

GAMMA = 1.5


def _bsp(n):
    return nmo.basis.BSplineEval(n_basis_funcs=n, order=4, bounds=(-1.0, 1.0))


_BASIS_CASES = [
    pytest.param(lambda: _bsp(8), id="eval"),
    pytest.param(lambda: _bsp(8) + _bsp(7), id="eval+eval"),
    pytest.param(lambda: _bsp(6) * _bsp(5), id="eval*eval"),
    pytest.param(lambda: _bsp(7) + (_bsp(5) * _bsp(4)), id="eval+(eval*eval)"),
]


def _gcv_setup(make_basis, seed=0, n_obs=300):
    """
    Build the GCV objective the way ``GAM.fit`` does and synthesise consistent
    QR inputs sized to the penalty's column count.
    """
    rng = np.random.default_rng(seed)
    gam = GAM(make_basis(), method="pql_gcv")
    infos = gam._unmasked_component_infos()
    penalty_tree = gam._get_penalty_tree(infos)
    ph = gam._build_penalty_handler(penalty_tree, infos)
    compute_sqrt, _ = ph.build()

    rho = [jnp.array(rng.uniform(-1.0, 1.0, size=p.rho_len)) for p in ph._penalties]

    # n_params = intercept column + columns of the block-diagonal sqrt penalty
    n_params = compute_sqrt(rho).shape[1] + 1
    Xw = jnp.array(rng.standard_normal((n_obs, n_params)))
    Q, R = jnp.linalg.qr(Xw)
    y = jnp.array(rng.standard_normal(n_obs))

    gcv_fn = gcv_compute_factory(
        compute_sqrt,
        identifiability_column_fns_per_component(infos),
        identifiability_square_fns_per_component(infos),
        GAMMA,
    )
    return gcv_fn, rho, penalty_tree, Xw, Q, R, y


@pytest.mark.parametrize("make_basis", _BASIS_CASES)
def test_gcv_gradient_matches_finite_diff(make_basis):
    gcv_fn, rho, penalty_tree, X, Q, R, y = _gcv_setup(make_basis)

    _, g_tree = jax.value_and_grad(gcv_fn)(rho, penalty_tree, X, Q, R, y)
    g_analytic, _ = ravel_pytree(g_tree)

    def _from_flat(flat_rho):
        return float(gcv_fn(unravel_rho(flat_rho), penalty_tree, X, Q, R, y))

    flat, unravel_rho = ravel_pytree(rho)
    h = 1e-6
    g_fd = np.zeros_like(np.array(flat))
    for j in range(flat.size):
        up, down = flat.at[j].add(h), flat.at[j].add(-h)
        g_fd[j] = (_from_flat(up) - _from_flat(down)) / (2 * h)

    np.testing.assert_allclose(np.array(g_analytic), g_fd, atol=1e-6, rtol=1e-4)


@pytest.mark.parametrize("gamma", [0.0, 1.5])
@pytest.mark.parametrize("n_obs", [8, 32])
def test_gcv_gradient_matches_dense(n_obs, gamma):
    """Exercise both gradient terms on wide and tall weighted designs."""
    rng = np.random.default_rng(12)
    penalties = []
    for count, size in [(2, 4), (1, 2), (2, 4)]:
        factors = rng.normal(size=(count, size, size))
        penalties.append(jnp.asarray(factors.transpose(0, 2, 1) @ factors))
    transforms = (DROP_LAST_ROW_COL, IDENTITY, DROP_LAST_ROW_COL)
    rho = [jnp.asarray(rng.uniform(-1, 1, len(term))) for term in penalties]
    blocks = compute_penalty_blocks(
        penalties, shift_by=1, apply_identifiability=transforms
    )

    # Build one working fit, including the unpenalized intercept.
    penalty = sum(
        np.einsum("k,kij->ij", np.exp(log_lams), block)
        for log_lams, block in zip(rho, blocks, strict=True)
    )
    X = rng.normal(size=(n_obs, penalty.shape[0]))
    X[:, 0] = 1
    y = rng.normal(size=n_obs)
    Q, R = np.linalg.qr(X)
    sqrt_penalty = np.zeros_like(penalty)
    sqrt_penalty[1:, 1:] = np.linalg.cholesky(penalty[1:, 1:]).T
    U, s, V_T = np.linalg.svd(np.vstack((R, sqrt_penalty)), full_matrices=False)
    U1 = U[: R.shape[0]]
    y1 = U1.T @ (Q.T @ y)
    alpha = np.sum((Q @ U1 @ y1 - y) ** 2)
    delta = n_obs - gamma * np.sum(U1**2)

    grad = _gcv_grad_compute_from_states(
        rho, penalties, y, gamma, alpha, delta, n_obs, U1, V_T, Q, 1 / s, transforms
    )

    # The previous backward pass formed M and F for each full-size penalty.
    cov_factor = V_T.T / s
    assert len(grad) == len(blocks)
    for block, log_lams, grad_term in zip(blocks, rho, grad, strict=True):
        M = np.einsum("ia,kij,jb->kab", cov_factor, block, cov_factor)
        F = M @ (U1.T @ U1)
        lams = np.exp(log_lams)
        alpha_grad = lams * np.einsum(
            "i,kij,j->k", y1, 2 * M - F - F.transpose(0, 2, 1), y1
        )
        delta_grad = gamma * lams * np.trace(F, axis1=1, axis2=2)
        expected = (
            n_obs * alpha_grad / delta**2 - 2 * n_obs * alpha * delta_grad / delta**3
        )
        assert grad_term.shape == expected.shape
        np.testing.assert_allclose(grad_term, expected, rtol=2e-12, atol=1e-12)
