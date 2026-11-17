"""Independent dense checks for term-local REML gradient contractions."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pgam_jax._pql_reml import _reml_grad_blocks
from pgam_jax.penalty_utils import DROP_LAST_ROW_COL, IDENTITY


@pytest.mark.parametrize("container", [list, tuple])
@pytest.mark.parametrize("n_columns", [1, 3, 13])
@pytest.mark.parametrize("mixed_identifiability", [False, True])
def test_reml_blocks_match_dense(container, n_columns, mixed_identifiability):
    """Check grouping, offsets, multiple penalties, and the returned tree."""
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
    cov_factor = jnp.asarray(rng.normal(size=(size, n_columns)))
    beta = cov_factor @ jnp.asarray(rng.normal(size=n_columns))
    lams = container(jnp.exp(jnp.asarray(rng.uniform(-4, 4, k))) for k, _ in shapes)
    penalties = container(penalties)

    rss, ld = jax.jit(_reml_grad_blocks, static_argnums=1)(
        penalties, ids, cov_factor, beta, lams
    )
    assert jax.tree_util.tree_structure(rss) == jax.tree_util.tree_structure(lams)
    assert jax.tree_util.tree_structure(ld) == jax.tree_util.tree_structure(lams)
    offset = 1
    for s, lam, rss_j, ld_j in zip(local, lams, rss, ld, strict=True):
        q = s.shape[-1]
        full = np.zeros((s.shape[0], size, size))
        full[:, offset : offset + q, offset : offset + q] = s
        rss_ref = np.asarray(lam) * np.einsum("i,kij,j->k", beta, full, beta)
        ld_ref = np.asarray(lam) * np.einsum(
            "ia,kij,ja->k", cov_factor, full, cov_factor
        )
        np.testing.assert_allclose(rss_j, rss_ref, rtol=2e-13, atol=1e-12)
        np.testing.assert_allclose(ld_j, ld_ref, rtol=2e-13, atol=1e-12)
        offset += q
