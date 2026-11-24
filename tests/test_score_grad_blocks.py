"""Dense reference checks for the per-term gradient blocks of PQL-REML and GCV."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pgam_jax._score_fn_grad_helper import _score_grad_blocks
from pgam_jax.penalty_utils import DROP_LAST_ROW_COL, IDENTITY


@pytest.mark.parametrize("n_columns", [1, 3, 13])
@pytest.mark.parametrize("mixed_identifiability", [False, True])
def test_score_blocks_match_dense(n_columns, mixed_identifiability):
    """
    Check grouping, offsets, multiple penalties, separate left and right
    vectors, and the returned lists.
    """
    rng = np.random.default_rng(12)
    shapes = [(2, 4), (1, 2), (2, 4), (3, 3)]
    penalties = []
    for k, q in shapes:
        factors = rng.normal(size=(k, q, q))
        penalties.append(jnp.asarray(np.swapaxes(factors, -1, -2) @ factors))
    ids = (DROP_LAST_ROW_COL, IDENTITY, DROP_LAST_ROW_COL, IDENTITY)
    if not mixed_identifiability:
        ids = (IDENTITY,) * len(penalties)
    local = [fn(p) for fn, p in zip(ids, penalties, strict=True)]
    size = 1 + sum(p.shape[-1] for p in local)
    trace_factor = jnp.asarray(rng.normal(size=(size, n_columns)))
    left = jnp.asarray(rng.normal(size=size))
    beta = trace_factor @ jnp.asarray(rng.normal(size=n_columns))
    lams = [jnp.exp(jnp.asarray(rng.uniform(-4, 4, k))) for k, _ in shapes]

    bilinear, trace = jax.jit(_score_grad_blocks, static_argnums=1)(
        penalties, ids, trace_factor, left, beta, lams
    )
    assert jax.tree_util.tree_structure(bilinear) == jax.tree_util.tree_structure(lams)
    assert jax.tree_util.tree_structure(trace) == jax.tree_util.tree_structure(lams)
    offset = 1
    for s, lam, bilinear_j, trace_j in zip(local, lams, bilinear, trace, strict=True):
        q = s.shape[-1]
        full = np.zeros((s.shape[0], size, size))
        full[:, offset : offset + q, offset : offset + q] = s
        bilinear_ref = np.asarray(lam) * np.einsum("i,kij,j->k", left, full, beta)
        trace_ref = np.asarray(lam) * np.einsum(
            "ia,kij,ja->k", trace_factor, full, trace_factor
        )
        np.testing.assert_allclose(bilinear_j, bilinear_ref, rtol=2e-13, atol=1e-12)
        np.testing.assert_allclose(trace_j, trace_ref, rtol=2e-13, atol=1e-12)
        offset += q


@pytest.mark.parametrize(
    "penalties_type, lams_type, n_lams, error, match",
    [
        (tuple, list, 2, TypeError, "penalty_tree has to be a list"),
        (list, tuple, 2, TypeError, "lams has to be a list"),
        (list, list, 1, ValueError, "one entry per term"),
    ],
)
def test_score_blocks_reject_bad_containers(
    penalties_type, lams_type, n_lams, error, match
):
    """Check that non-list inputs and a lams length mismatch raise."""
    penalties = penalties_type(jnp.eye(3)[None] for _ in range(2))
    lams = lams_type(jnp.ones(1) for _ in range(n_lams))
    trace_factor = jnp.ones((7, 7))
    beta = jnp.ones(7)
    with pytest.raises(error, match=match):
        _score_grad_blocks(
            penalties, (IDENTITY, IDENTITY), trace_factor, beta, beta, lams
        )
