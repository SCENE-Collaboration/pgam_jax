"""Compatibility shims for differences across supported nemos versions."""

import inspect

import jax.numpy as jnp
import nemos
from nemos.glm.initialize_parameters import (
    initialize_intercept_matching_mean_rate as _nemos_init_intercept,
)
from nemos.observation_models import Observations
from packaging.version import Version

_NEMOS_HAS_N_INPUTS = Version(nemos.__version__) >= Version("0.2.7")


def get_n_inputs(basis) -> int:
    """
    Return the number of inputs a basis component consumes.

    Wraps the attribute rename from ``_n_input_dimensionality`` (nemos < 0.2.7)
    to ``_n_inputs`` (nemos >= 0.2.7).
    """
    if _NEMOS_HAS_N_INPUTS:
        return basis._n_inputs
    return basis._n_input_dimensionality


def initialize_intercept_matching_mean_rate(
    observation_model: Observations, X: jnp.ndarray, y: jnp.ndarray
) -> jnp.ndarray:
    """
    Return the intercept for which the mean prediction equals the mean of ``y``.

    There was a signature change from ``(inverse_link_function, y)`` (nemos <= 0.2.9)
    to ``(inverse_link_function, X, y, ..., rate_range=None)`` (nemos >= 0.2.10).

    If nemos accepts ``rate_range``, the observation model's range is passed. Then
    an output with a mean on the boundary of the range (for example, a silent
    neuron with a Poisson model) gets a finite intercept and a warning. Older nemos
    versions raise a ``ValueError`` for such an output.
    """
    # Check the signature, not the version. nemos dev builds report "0.2.10.devN", which
    # compares lower than "0.2.10", but some of them have X and/or rate_range.
    params = inspect.signature(_nemos_init_intercept).parameters
    kwargs = {}
    if "X" in params:
        kwargs["X"] = X
    if "rate_range" in params:
        kwargs["rate_range"] = observation_model.rate_range
    return _nemos_init_intercept(
        observation_model.default_inverse_link_function, y=y, **kwargs
    )
