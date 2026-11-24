from collections import defaultdict
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np


def _score_grad_blocks(
    penalty_tree: list[jax.Array],
    apply_identifiability: tuple[Callable, ...],
    trace_factor: jax.Array,
    left_vector: jax.Array,
    beta_hat: jax.Array,
    lams: list[jax.Array],
) -> tuple[list[jax.Array], list[jax.Array]]:
    """
    Compute the per-penalty bilinear and trace terms for PQL-REML and GCV.

    Parameters
    ----------
    penalty_tree
        One array per smooth. After applying identifiability, entry j has shape
        (K, Q, Q) for K penalties on the Q coefficients of smooth j.
        The coefficients of smooth j are a contiguous block of rows in
        trace_factor, left_vector and beta_hat. The blocks follow the list
        order and start at row 1, after the unpenalized intercept in row 0.
    apply_identifiability
        One transform per smooth that aligns its penalty matrices with the
        corresponding fitted coefficients.
    trace_factor
        Matrix with one row per coefficient, including the intercept. Xw is
        the weighted design and S_lambda is the total penalty. REML passes C,
        where H_pinv = C @ C.T is the pseudoinverse of Xw.T @ Xw + S_lambda
        on the retained directions. GCV passes D = C @ U1.T. With scale
        phi = 1, trace_factor @ trace_factor.T is a covariance matrix of
        beta_hat in both cases. For REML it is the Bayesian covariance
        H_pinv. For GCV it is the frequentist covariance
        H_pinv @ Xw.T @ Xw @ H_pinv.
    left_vector
        One value per coefficient. REML passes beta_hat. GCV passes
        H_pinv @ Xw.T @ (yw - Xw @ beta_hat), where yw is the weighted
        working response.
    beta_hat
        Coefficients from the weighted penalized fit at the trial smoothing
        parameters, including the intercept.
    lams
        One array of smoothing parameters exp(rho) per smooth, in penalty order.

    Returns
    -------
    bilinear_terms
        One array per smooth. For each penalty S with smoothing parameter lam,
        the entry is lam * (left_vector_j.T @ S @ beta_hat_j). The subscript j
        selects the coefficients of the smooth to which S belongs.
    trace_terms
        Same structure as bilinear_terms. Each entry is
        lam * trace(factor_j.T @ S @ factor_j), where factor_j contains the
        rows of trace_factor for smooth j.

    Notes
    -----
    All derivatives are with respect to log smoothing parameters at fixed
    working data. For REML the terms give the derivatives of the penalized
    residual sum of squares and the log determinant of the penalized
    information matrix. For GCV, twice the bilinear term gives the derivative
    of the residual sum of squares. Multiplying the trace term by gamma gives
    the derivative of n_obs - gamma * EDF.
    """
    if not isinstance(apply_identifiability, tuple):
        raise TypeError("apply_identifiability has to be a tuple")
    if not isinstance(penalty_tree, list):
        raise TypeError("penalty_tree has to be a list")
    if not isinstance(lams, list):
        raise TypeError("lams has to be a list")
    if len(lams) != len(penalty_tree):
        raise ValueError("lams has to have one entry per term in penalty_tree")

    S_per_term = [
        fn(p) for fn, p in zip(apply_identifiability, penalty_tree, strict=True)
    ]

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

    def _per_term_grad(S_j, factor_j, left_j, beta_hat_j, lam_j):
        # REML: rss_grad[k] = lam_j[k] * beta_hat_j.T @ S_j[k] @ beta_hat_j
        # GCV: alpha_grad[k] = 2 * lam_j[k] * left_j.T @ S_j[k] @ beta_hat_j
        bilinear = lam_j * jnp.einsum("kij,i,j->k", S_j, left_j, beta_hat_j)
        # reuse the term's covariance block across all its penalties
        trace_block = factor_j @ factor_j.T
        # REML: logdet_xtx_grad[k] = lam_j[k] * jnp.trace(C_j.T @ S_j[k] @ C_j)
        # GCV: delta_grad[k] = gamma * lam_j[k] * jnp.trace(D_j.T @ S_j[k] @ D_j)
        # factor_j is C_j for REML and D_j for GCV. GCV applies 2 and gamma in the caller.
        trace = lam_j * jnp.einsum("kij,ij->k", S_j, trace_block)
        return bilinear, trace

    _per_group_grad = jax.vmap(_per_term_grad)
    bilinear_terms = [None] * len(S_per_term)
    trace_terms = [None] * len(S_per_term)

    # shape glossary: G is the number of terms in the group.
    # n_cols is the number of columns of trace_factor.
    for member_indices in groups.values():
        group_coef_indices = np.stack([term_coef_indices[j] for j in member_indices])
        group_trace_factor = trace_factor[group_coef_indices]  # (G, Q, n_cols)
        group_left_vector = left_vector[group_coef_indices]  # (G, Q)
        group_beta_hat = beta_hat[group_coef_indices]  # (G, Q)
        group_S = jnp.stack([S_per_term[j] for j in member_indices])  # (G, K, Q, Q)
        group_lams = jnp.stack([lams[j] for j in member_indices])  # (G, K)

        group_bilinear, group_trace = _per_group_grad(
            group_S, group_trace_factor, group_left_vector, group_beta_hat, group_lams
        )

        # put the terms' gradients in the original order
        for j, bilinear, trace in zip(
            member_indices, group_bilinear, group_trace, strict=True
        ):
            bilinear_terms[j] = bilinear
            trace_terms[j] = trace

    return bilinear_terms, trace_terms
