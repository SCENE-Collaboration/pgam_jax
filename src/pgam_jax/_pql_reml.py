from collections import defaultdict
from collections.abc import Callable
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from jaxtyping import PyTree
from numpy.typing import NDArray

from ._pql_gcv import _vmap_where
from ._utils import singular_value_keep_mask
from .penalty_utils import prepend_zeros_for_intercept


@partial(
    jax.jit,
    static_argnames=("compute_log_det_and_grad", "compute_sqrt"),
)
def _compute_reml_and_states(
    regularization_strength: Any,
    X: NDArray,
    Q: NDArray,
    R: NDArray,
    y: NDArray,
    compute_log_det_and_grad: Callable,
    compute_sqrt: Callable,
):
    """
    Forward pass for linearized REML.

    Mirrors _compute_gcv_and_states in structure: same SVD of [R; sqrt_penalty],
    same singular-value masking.  Computes the three REML terms:

        reml_val = 0.5 * RSS_reml
                 + 0.5 * log|X'X + S_lam|
                 - 0.5 * log|S_lam|_+

    and returns all intermediate quantities needed by the backward pass.

    Parameters
    ----------
    regularization_strength :
        Pytree of log-smoothing parameters (one leaf per penalty block).
    X, Q, R :
        Whitened design matrix and its QR factors (sqrt(W) X, Q, R).
    y :
        Augmented working response (length >= n_obs; only first n_obs rows used).
    compute_log_det_and_grad :
        Static callable (from PenaltyHandler.build()) that receives
        regularization_strength and returns (log_dets list, grads list).
    compute_sqrt :
        Static callable (from PenaltyHandler.build()) that receives
        regularization_strength and returns the block-diagonal sqrt penalty matrix.

    Returns
    -------
    reml_val :
        Scalar REML objective.
    RSS_reml :
        Scalar penalised RSS (stored for diagnostics).
    log_det_XtXpSl :
        Scalar log|X'X + S_lam| (stored for diagnostics).
    y1 :
        Shape (M, 1). U1.T @ Q.T @ y, needed for beta_hat in the gradient.
        M is the number of singular values of A = [R; sqrt_penalty].
    s_inv :
        Shape (M,). Masked reciprocal singular values, needed for cov_factor.
    V_T :
        Shape (M, P). Right singular vectors of A, needed for cov_factor.
        P is the number of coefficients, including the intercept.
    log_det_sl_grads :
        Pytree of per-block log|S_lam|_+ gradients (same structure as
        regularization_strength), computed alongside the log-det in the
        forward pass and stored to avoid recomputation in the backward.
    n_obs :
        Number of observations (scalar int).
    """
    sqrt_penalty = compute_sqrt(regularization_strength)
    sqrt_penalty = prepend_zeros_for_intercept(sqrt_penalty)

    n_obs = X.shape[0]
    A = jnp.vstack((R, sqrt_penalty))
    U, s, V_T = jnp.linalg.svd(A, full_matrices=False)

    # mask near-zero singular values (identical to GCV)
    low_vals = ~singular_value_keep_mask(s, A.shape)
    s = jnp.where(low_vals, 0.0, s)
    U = _vmap_where(low_vals, 0, U)
    V_T = _vmap_where(low_vals, 0, V_T.T).T

    U1 = U[: R.shape[0]]
    s_inv = jnp.where(low_vals, 0.0, 1.0 / jnp.where(low_vals, 1.0, s))

    # --- REML RSS = ||y||^2 - ||U1^T Q^T y||^2 ---
    y_obs = y[:n_obs].reshape(n_obs, -1)
    y1 = U1.T @ (Q.T @ y_obs)  # (M, 1)
    RSS_reml = jnp.sum(y_obs**2) - jnp.sum(y1**2)

    # --- log|X'X + S_lam| = 2 * sum log(s_k) over positive singular values ---
    log_s_safe = jnp.where(low_vals, 0.0, jnp.log(jnp.where(low_vals, 1.0, s)))
    log_det_XtXpSl = 2.0 * jnp.sum(log_s_safe)

    # --- log|S_lam|_+ and its gradient (shared eigh, no extra cost) ---
    log_dets, log_det_sl_grads = compute_log_det_and_grad(regularization_strength)
    log_det_Sl = jtu.tree_reduce(lambda a, b: a + b, log_dets)

    # TODO: Support fixed or estimated dispersion beyond the current Poisson case.
    # Update this objective and its custom gradient together. For unknown
    # dispersion, estimate it during smoothing selection, not only after fitting.
    reml_val = 0.5 * RSS_reml + 0.5 * log_det_XtXpSl - 0.5 * log_det_Sl

    return reml_val, RSS_reml, log_det_XtXpSl, y1, s_inv, V_T, log_det_sl_grads, n_obs


def _reml_grad_blocks(
    penalty_tree: PyTree[jax.Array],
    apply_identifiability: tuple[Callable, ...],
    cov_factor: jax.Array,
    beta_hat: jax.Array,
    lams: PyTree[jax.Array],
) -> tuple[PyTree[jax.Array], PyTree[jax.Array]]:
    """
    Compute the RSS and log|X.T @ W @ X + S_lambda| contributions to the gradient, using per-term penalties.

    Parameters
    ----------
    penalty_tree
        One leaf per smooth term, with shape (K_j, N_j, N_j). K_j is the
        number of penalties for term j. After applying identifiability, each
        leaf has shape (K_j, Q_j, Q_j). The term blocks are contiguous in
        cov_factor, in leaf order, and start at row 1. Row 0 is the intercept.
    apply_identifiability
        Transform applied to each penalty leaf, for example dropping a row
        and column to match a dropped design column. The tuple contains
        one callable per leaf in the flattened order of penalty_tree.
    cov_factor
        Shape (P, M). V_T.T * s_inv from _reml_grad_compute_from_states.
        The product cov_factor @ cov_factor.T is the numerically truncated pseudoinverse
        of X.T @ W @ X + S_lambda, with small singular directions dropped. With
        scale phi = 1, it is the Bayesian posterior covariance of beta.
        P is the total number of coefficients, including the intercept. M is
        the number of singular directions returned by the thin SVD of [R; sqrt_penalty].
    beta_hat
        Shape (P,). Fitted coefficient vector for the current weighted, penalized
        least-squares problem, computed as cov_factor @ y1.ravel().
    lams
        Regularizer strengths, exp(rho), with one leaf of shape (K_j,) per term.

    Returns
    -------
    rss_grad
        Same tree structure as penalty_tree, with leaves of shape (K_j,).
        Entry k for term j is lambda_jk * beta_hat_j.T @ S_jk @ beta_hat_j, where
        S_jk is penalty k of term j after apply_identifiability, with shape
        (Q_j, Q_j), and beta_hat_j is term j's slice of beta_hat.
    logdet_xtx_grad
        Same tree structure as penalty_tree, with leaves of shape (K_j,).
        Entry k for term j is lambda_jk * trace(C_j.T @ S_jk @ C_j),
        where C_j is the term's row slice of cov_factor. The caller applies the
        factors of 0.5 to both returned contributions.

    Notes
    -----
    Each penalty is nonzero only on its term's coefficients. So each product
    uses only the term's rows of beta_hat and cov_factor and the (Q_j, Q_j)
    penalty, not a zero-padded (P, P) penalty. Terms with equal
    penalty counts and block sizes are stacked and processed with vmap.
    C_j @ C_j.T is computed once per term and reused for all K_j penalties of
    the term. The trace then uses trace(C_j.T @ S_jk @ C_j) = sum(S_jk * (C_j @ C_j.T)),
    which holds because C_j @ C_j.T is symmetric. No (M, M) matrix is formed.
    """
    if not isinstance(apply_identifiability, tuple):
        raise TypeError("apply_identifiability has to be a tuple")

    penalty_flat, structure = jtu.tree_flatten(penalty_tree)
    S_per_term = [
        fn(p) for fn, p in zip(apply_identifiability, penalty_flat, strict=True)
    ]
    lams_per_term = jtu.tree_leaves(lams)

    # term j has coefficients term_boundaries[j] to term_boundaries[j + 1]. index 0 is the intercept.
    block_sizes = [p.shape[-1] for p in S_per_term]
    term_boundaries = np.cumsum([1, *block_sizes])
    term_coef_indices = [
        np.arange(start, stop)
        for start, stop in zip(term_boundaries[:-1], term_boundaries[1:])
    ]

    # group terms by penalty shape (K, Q, Q) -> one vmap per group
    groups = defaultdict(list)
    for j, S_j in enumerate(S_per_term):
        groups[S_j.shape].append(j)

    def _per_term_grad(S_j, C_j, beta_hat_j, lam_j):
        # rss_grad[k] = lam_j[k] * beta_hat_j.T @ S_j[k] @ beta_hat_j
        rss_grad = lam_j * jnp.einsum("kij,i,j->k", S_j, beta_hat_j, beta_hat_j)
        # reuse the term's covariance block across all its penalties
        covariance = C_j @ C_j.T
        # logdet_xtx_grad[k] = lam_j[k] * jnp.trace(C_j.T @ S_j[k] @ C_j)
        logdet_xtx_grad = lam_j * jnp.einsum("kij,ij->k", S_j, covariance)

        return rss_grad, logdet_xtx_grad

    _per_group_grad = jax.vmap(_per_term_grad)

    n_terms = len(S_per_term)
    rss_grad_leaves = [None] * n_terms
    logdet_xtx_grad_leaves = [None] * n_terms
    # shape glossary: G is the number of terms in the group
    for member_indices in groups.values():
        group_coef_indices = np.stack([term_coef_indices[j] for j in member_indices])
        group_C = cov_factor[group_coef_indices]  # (G, Q, M)
        group_beta_hat = beta_hat[group_coef_indices]  # (G, Q)
        group_S = jnp.stack([S_per_term[j] for j in member_indices])  # (G, K, Q, Q)
        group_lams = jnp.stack([lams_per_term[j] for j in member_indices])  # (G, K)

        group_rss_grads, group_logdet_xtx_grads = _per_group_grad(
            group_S, group_C, group_beta_hat, group_lams
        )

        # put the terms' gradients in the original order
        for j, rss_grad, logdet_xtx_grad in zip(
            member_indices, group_rss_grads, group_logdet_xtx_grads, strict=True
        ):
            rss_grad_leaves[j] = rss_grad
            logdet_xtx_grad_leaves[j] = logdet_xtx_grad

    return (
        jtu.tree_unflatten(structure, rss_grad_leaves),
        jtu.tree_unflatten(structure, logdet_xtx_grad_leaves),
    )


@partial(
    jax.jit,
    static_argnames=("apply_identifiability",),
)
def _reml_grad_compute_from_states(
    regularization_strength: Any,
    penalty_tree: Any,
    y1: jnp.ndarray,
    s_inv: jnp.ndarray,
    V_T: jnp.ndarray,
    log_det_sl_grads: Any,
    apply_identifiability: tuple[Callable, ...],
):
    """
    Gradient of the REML objective w.r.t. regularization_strength.

    Given states from _compute_reml_and_states, computes three gradient
    contributions per leaf j:

        grad_j = 0.5 * lam_j * beta_hat^T Sj beta_hat          [RSS]
               + 0.5 * lam_j * tr(cov_factor^T Sj cov_factor)  [log|X'X+Slam|]
               - 0.5 * d log|Slam|_+ / d rho_j                 [log|Slam|]

    where cov_factor = V_T^T * s_inv and beta_hat = cov_factor @ y1.
    """
    # cov_factor @ cov_factor.T is the pseudoinverse of X.T @ W @ X + S_lambda,
    # with small singular directions dropped. With scale phi = 1, it is the
    # Bayesian posterior covariance of beta at the current lambda and IRLS weights.
    cov_factor = V_T.T * s_inv  # (P, M)
    beta_hat = cov_factor @ y1.ravel()  # estimate of beta for the current lambda, (P,)

    lams = jtu.tree_map(jnp.exp, regularization_strength)
    rss_grad, logdet_xtx_grad = _reml_grad_blocks(
        penalty_tree, apply_identifiability, cov_factor, beta_hat, lams
    )

    return jtu.tree_map(
        lambda rss_g, ld_xtx_g, ld_sl_g: 0.5 * rss_g + 0.5 * ld_xtx_g - 0.5 * ld_sl_g,
        rss_grad,
        logdet_xtx_grad,
        log_det_sl_grads,
    )


# TODO: Remove apply_identifiability_columns. It is unused here and in
# gcv_compute_factory. Callers: gam.py and tests/test_pql_reml.py.
def reml_compute_factory(
    compute_sqrt: Callable,
    compute_log_det_and_grad: Callable,
    apply_identifiability_columns: tuple[Callable, ...],
    apply_identifiability: tuple[Callable, ...],
):
    """
    Build a differentiable REML objective with a custom VJP.

    Parameters
    ----------
    compute_sqrt :
        Static callable from ``PenaltyHandler.build()`` computing the block-diagonal
        sqrt penalty matrix from regularization_strength.
    compute_log_det_and_grad :
        Static callable from ``PenaltyHandler.build()`` computing log|S_lam|_+ and
        its gradient w.r.t. rho from regularization_strength.
        Same semantics as in gcv_compute_factory.

    Returns
    -------
    :
        A jax.custom_vjp function
        _reml_compute(regularization_strength, penalty_tree, X, Q, R, y)
        returning the scalar REML objective with analytic reverse-mode gradient.
    """

    @jax.custom_vjp
    def _reml_compute(
        regularization_strength: Any,
        penalty_tree: Any,
        X: NDArray,
        Q: NDArray,
        R: NDArray,
        y: NDArray,
    ):
        return _compute_reml_and_states(
            regularization_strength,
            X,
            Q,
            R,
            y,
            compute_log_det_and_grad=compute_log_det_and_grad,
            compute_sqrt=compute_sqrt,
        )[0]

    def _reml_compute_fwd(regularization_strength, penalty_tree, X, Q, R, y):
        reml_val, RSS_reml, log_det_XtXpSl, y1, s_inv, V_T, log_det_sl_grads, n_obs = (
            _compute_reml_and_states(
                regularization_strength,
                X,
                Q,
                R,
                y,
                compute_log_det_and_grad=compute_log_det_and_grad,
                compute_sqrt=compute_sqrt,
            )
        )
        return reml_val, (
            regularization_strength,
            penalty_tree,
            y1,
            s_inv,
            V_T,
            log_det_sl_grads,
        )

    def _reml_compute_bwd(res, reml_bar):
        regularization_strength, penalty_tree, y1, s_inv, V_T, log_det_sl_grads = res
        reml_grad = _reml_grad_compute_from_states(
            regularization_strength,
            penalty_tree,
            y1,
            s_inv,
            V_T,
            log_det_sl_grads,
            apply_identifiability=apply_identifiability,
        )
        return (jtu.tree_map(lambda g: reml_bar * g, reml_grad),) + (None,) * 5

    _reml_compute.defvjp(_reml_compute_fwd, _reml_compute_bwd)
    return _reml_compute
