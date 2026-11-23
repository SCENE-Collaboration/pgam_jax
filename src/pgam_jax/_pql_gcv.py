from collections.abc import Callable
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from nemos.tree_utils import tree_scalar_mul
from numpy.typing import NDArray

from ._score_fn_grad_helper import _score_grad_blocks
from ._utils import singular_value_keep_mask
from .penalty_utils import prepend_zeros_for_intercept

_vmap_where = jax.vmap(jnp.where, (None, None, 0), out_axes=0)


@partial(jax.jit, static_argnames=("compute_sqrt", "gamma"))
def _compute_gcv_and_states(
    regularization_strength: Any,
    penalty_tree: Any,
    X: NDArray,
    Q: NDArray,
    R: NDArray,
    y: NDArray,
    compute_sqrt: Callable,
    gamma=1.5,
):
    sqrt_penalty = compute_sqrt(regularization_strength)
    sqrt_penalty = prepend_zeros_for_intercept(sqrt_penalty)

    n_obs = X.shape[0]
    A = jnp.vstack((R, sqrt_penalty))
    U, s, V_T = jnp.linalg.svd(A, full_matrices=False)

    # remove low val singular values (numerical stability)
    low_vals = ~singular_value_keep_mask(s, A.shape)
    s = jnp.where(low_vals, 0, s)
    U = _vmap_where(low_vals, 0, U)
    V_T = _vmap_where(low_vals, 0, V_T.T).T

    U1 = U[: R.shape[0]]

    # make sure it is 2D
    y = y[:n_obs].reshape(n_obs, -1)
    s_inv = jnp.where(low_vals, 0.0, 1.0 / s)
    square_s_inv = s_inv**2

    # compute A.dot(y) and trA without forming the n_obs x n_obs matrix.
    mat_vec = jnp.dot(V_T, jnp.dot(X.T, y[:n_obs]))
    Ay = X.dot(V_T.T.dot((mat_vec.T * square_s_inv).T))
    trA = (U1**2).sum()
    delta = n_obs - gamma * trA
    alpha = jnp.sum(jnp.power(Ay - y, 2))
    gcv = n_obs * alpha / (delta**2)
    return gcv, alpha, delta, n_obs, U1, V_T, Q, trA, Ay, s_inv, square_s_inv


@partial(jax.jit, static_argnames=("apply_identifiability", "gamma"))
def _gcv_grad_compute_from_states(
    regularization_strength,
    penalty_tree,
    y,
    gamma,
    alpha,
    delta,
    n_obs,
    U1,
    V_T,
    Q,
    s_inv,
    apply_identifiability,
):
    # compute useful vector
    qt_y = (Q.T @ y).ravel()
    y1 = U1.T @ qt_y

    cov_factor = V_T.T * s_inv
    D = cov_factor @ U1.T
    beta_hat = cov_factor @ y1  # estimate of beta for the current lambda, (P,)
    u = D @ (qt_y - U1 @ y1)

    lams = jtu.tree_map(jnp.exp, regularization_strength)
    bilinear_term, trace_term = _score_grad_blocks(
        penalty_tree,
        apply_identifiability,
        D,
        u,
        beta_hat,
        lams,
    )
    alpha_grad = tree_scalar_mul(2, bilinear_term)
    delta_grad = tree_scalar_mul(gamma, trace_term)
    gcv_grad = jtu.tree_map(
        lambda x, y: (n_obs / delta**2) * x - 2 * n_obs * alpha / delta**3 * y,
        alpha_grad,
        delta_grad,
    )
    return gcv_grad


def gcv_compute_factory(
    compute_sqrt,
    apply_identifiability_columns,
    apply_identifiability,
    gamma,
):
    @jax.custom_vjp
    def _gcv_compute(
        regularization_strength: Any,
        penalty_tree: Any,
        X: NDArray,
        Q: NDArray,
        R: NDArray,
        y: NDArray,
    ):
        gcv = _compute_gcv_and_states(
            regularization_strength,
            penalty_tree,
            X,
            Q,
            R,
            y,
            compute_sqrt=compute_sqrt,
            gamma=gamma,
        )[0]
        return gcv

    def _gcv_compute_fwd(
        regularization_strength,
        penalty_tree,
        X,
        Q,
        R,
        y,
    ):
        gcv, alpha, delta, n_obs, U1, V_T, Q_, trA, Ay, s_inv, square_s_inv = (
            _compute_gcv_and_states(
                regularization_strength,
                penalty_tree,
                X,
                Q,
                R,
                y,
                compute_sqrt=compute_sqrt,
                gamma=gamma,
            )
        )

        # Save inputs for backward (must be JAX types only)
        return gcv, (
            gcv,
            alpha,
            delta,
            n_obs,
            U1,
            V_T,
            Q_,
            trA,
            Ay,
            s_inv,
            square_s_inv,
            regularization_strength,
            penalty_tree,
            X,
            Q,
            R,
            y,
        )

    def _gcv_compute_bwd(res, gcv_bar):
        (
            gcv,
            alpha,
            delta,
            n_obs,
            U1,
            V_T,
            Q,
            trA,
            Ay,
            s_inv,
            square_s_inv,
            regularization_strength,
            penalty_tree,
            X,
            Q,
            R,
            y,
        ) = res

        # Compute full gradient PyTree using the pre-computed states.
        gcv_grad = _gcv_grad_compute_from_states(
            regularization_strength,
            penalty_tree,
            y,
            gamma,
            alpha,
            delta,
            n_obs,
            U1,
            V_T,
            Q,
            s_inv,
            apply_identifiability,
        )
        return (jtu.tree_map(lambda g: gcv_bar * g, gcv_grad),) + (None,) * 5

    _gcv_compute.defvjp(_gcv_compute_fwd, _gcv_compute_bwd)
    return _gcv_compute
