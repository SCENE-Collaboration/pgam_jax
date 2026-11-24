"""Compatibility shims for differences across supported nemos versions."""

import inspect
import os
import warnings

import jax.numpy as jnp
import nemos
from nemos.convolve import create_convolutional_predictor
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


def use_direct_convolution(basis) -> None:
    """
    Make every convolutional component of ``basis`` use direct convolution.

    nemos 0.2.10 added an FFT path that it selects on CPU when
    window_size > 5 * log2(n_samples). On that path, one NaN in the input makes
    every output row NaN, which differs from the direct convolution path.
    Additionally, exact zeros become round-off values, which can lead to
    column-masking not being applied columns with no data.

    This sets ``use_fft=False`` where ``use_fft`` is not set or is None.
    If the user set ``use_fft=True``, it is kept and a warning is given.
    Older nemos versions have no FFT path, and then this function does nothing.
    """
    if "use_fft" not in inspect.signature(create_convolutional_predictor).parameters:
        return

    for leaf in basis._iterate_over_components():
        if not hasattr(leaf, "conv_kwargs"):
            continue

        use_fft = leaf.conv_kwargs.get("use_fft")
        match use_fft:
            case None:
                leaf.conv_kwargs = {**leaf.conv_kwargs, "use_fft": False}
            case True:
                warnings.warn(
                    f"Basis {leaf.label!r} sets use_fft=True. On the FFT path, nemos "
                    "makes every output row NaN if the input contains a NaN, and "
                    "gives round-off values instead of exact zeros. Exact zeros "
                    "matter for drop_empty_columns. Remove use_fft or set "
                    "use_fft=False to use direct convolution.",
                    UserWarning,
                    # Point at the user's call. The depth differs for fit and concurvity.
                    skip_file_prefixes=(os.path.dirname(__file__),),
                )
            case False:
                pass
            case _:
                raise ValueError(
                    f"Basis {leaf.label!r} has use_fft={use_fft!r}. "
                    "Expected True, False, or None."
                )
