from collections import defaultdict
from collections.abc import Callable
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from numpy.typing import NDArray

from ._pql_gcv import _vmap_symm_mult, _vmap_where
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
        Shape (k, 1) — U1ᵀ Qᵀ y, needed for the RSS gradient.
    s_inv :
        Shape (k,) — masked reciprocal singular values, needed for M_j.
    V_T :
        Shape (k, p) — right singular vectors, needed for M_j.
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
    y1 = U1.T @ (Q.T @ y_obs)  # (k, 1)
    RSS_reml = jnp.sum(y_obs**2) - jnp.sum(y1**2)

    # --- log|X'X + S_lam| = 2 * sum log(s_k) over positive singular values ---
    log_s_safe = jnp.where(low_vals, 0.0, jnp.log(jnp.where(low_vals, 1.0, s)))
    log_det_XtXpSl = 2.0 * jnp.sum(log_s_safe)

    # --- log|S_lam|_+ and its gradient (shared eigh, no extra cost) ---
    log_dets, log_det_sl_grads = compute_log_det_and_grad(regularization_strength)
    log_det_Sl = jtu.tree_reduce(lambda a, b: a + b, log_dets)

    reml_val = 0.5 * RSS_reml + 0.5 * log_det_XtXpSl - 0.5 * log_det_Sl

    return reml_val, RSS_reml, log_det_XtXpSl, y1, s_inv, V_T, log_det_sl_grads, n_obs


def _reml_grad_blocks_sliced(penalty_tree, apply_identifiability, comp, comp_y1, lams):
    """
    Compute gradient contributions using term-local penalties grouped by shape.

    A smooth term's penalty ``S_j`` is nonzero only on that term's own columns,
    so ``comp.T @ S_j @ comp`` depends only on the few rows of ``comp`` for that
    term. We slice ``comp`` to each term's columns, multiply by the SMALL
    ``(q_j, q_j)`` penalty, then stack all same-shaped terms and ``vmap`` -- one
    batched op per distinct penalty shape (e.g. one for all the 6x6 coupling
    terms) instead of one full-size matmul per term. The dropped off-block
    entries are structurally zero, so slicing preserves the algebra. Floating-point
    reduction order can differ. Grouping and offsets use only static shapes.
    """
    penalty_flat, structure = jtu.tree_flatten(penalty_tree)
    if isinstance(apply_identifiability, (list, tuple)):
        scaled = [
            fn(p) for fn, p in zip(apply_identifiability, penalty_flat, strict=True)
        ]
    else:
        scaled = [apply_identifiability(p) for p in penalty_flat]
    lams = jtu.tree_leaves(lams)

    n = len(scaled)
    # diagonal column offset of each block in the intercept-shifted coef space
    offsets = []
    acc = 1  # shift_by=1 skips the unpenalized intercept column
    for sp in scaled:
        offsets.append(acc)
        acc += int(sp.shape[-1])

    # group terms by (n_penalties_per_term, block_size) -> one vmap per group
    groups: dict = defaultdict(list)
    for j, sp in enumerate(scaled):
        groups[(int(sp.shape[0]), int(sp.shape[-1]))].append(j)

    def _per_term(pen, comp_slice, comp_y1_slice, lam):
        # pen (k,q,q)  comp_slice (q,m)  comp_y1_slice (q,)  lam (k,) -> (k,),(k,)
        rss = lam * _vmap_symm_mult(pen, comp_y1_slice)
        # Only the trace is needed. Avoid forming a (k, m, m) product.
        pen_comp = jnp.matmul(pen, comp_slice, precision=jax.lax.Precision.HIGHEST)
        ld = lam * jnp.sum(comp_slice[None] * pen_comp, axis=(-2, -1))
        return rss, ld

    rss_out: list = [None] * n
    ld_out: list = [None] * n
    for (_k, q), members in groups.items():
        rows = np.stack([np.arange(offsets[j], offsets[j] + q) for j in members])
        comp_g = comp[rows]  # (G, q, m)
        comp_y1_g = comp_y1[rows]  # (G, q)
        pen_g = jnp.stack([scaled[j] for j in members])  # (G, k, q, q)
        lam_g = jnp.stack([lams[j] for j in members])  # (G, k)
        rss_b, ld_b = jax.vmap(_per_term)(pen_g, comp_g, comp_y1_g, lam_g)
        for i, j in enumerate(members):
            rss_out[j] = rss_b[i]
            ld_out[j] = ld_b[i]
    return (
        jtu.tree_unflatten(structure, rss_out),
        jtu.tree_unflatten(structure, ld_out),
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
    apply_identifiability: Callable | None,
):
    """
    Gradient of the REML objective w.r.t. regularization_strength.

    Given states from _compute_reml_and_states, computes three gradient
    contributions per leaf j:

        grad_j = 0.5 * lam_j * (comp @ y1)^T Sj (comp @ y1)   [RSS]
               + 0.5 * lam_j * tr(comp^T Sj comp)              [log|X'X+Slam|]
               - 0.5 * d log|Slam|_+ / d rho_j                 [log|Slam|]

    where comp = V_T^T * s_inv  and  Mj = comp^T Sj comp.
    """
    comp = V_T.T * s_inv  # (n_features, k)
    comp_y1 = comp @ y1.ravel()  # (n_features,)

    lams = jtu.tree_map(jnp.exp, regularization_strength)
    rss_grad, logdet_xtx_grad = _reml_grad_blocks_sliced(
        penalty_tree, apply_identifiability, comp, comp_y1, lams
    )

    return jtu.tree_map(
        lambda rss_g, ld_xtx_g, ld_sl_g: 0.5 * rss_g + 0.5 * ld_xtx_g - 0.5 * ld_sl_g,
        rss_grad,
        logdet_xtx_grad,
        log_det_sl_grads,
    )


def reml_compute_factory(
    compute_sqrt: Callable,
    compute_log_det_and_grad: Callable,
    apply_identifiability_columns: Callable | None,
    apply_identifiability: Callable | None,
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
