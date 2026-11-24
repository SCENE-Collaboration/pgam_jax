"""Tests for ``use_direct_convolution`` in ``pgam_jax._nemos_compat``."""

import inspect

import jax
import nemos as nmo
import numpy as np
import pytest
from nemos.convolve import create_convolutional_predictor

from pgam_jax import GAM
from pgam_jax._nemos_compat import use_direct_convolution

jax.config.update("jax_enable_x64", True)

# nemos 0.2.10 added the FFT convolution path and its ``use_fft`` argument (nemos #588).
# nemos 0.2.9 rejects ``use_fft`` in ``conv_kwargs``.
NEMOS_HAS_USE_FFT = (
    "use_fft" in inspect.signature(create_convolutional_predictor).parameters
)
pytestmark = pytest.mark.skipif(
    not NEMOS_HAS_USE_FFT, reason="nemos <= 0.2.9 has no FFT convolution path"
)

N_SAMPLES = 200
# 40 > 5 * log2(200) = 38.2, so nemos picks the FFT path for this data on CPU, unless use_fft=False.
WINDOW = 40


def _conv_basis(**conv_kwargs):
    return nmo.basis.BSplineConv(
        8, window_size=WINDOW, conv_kwargs=conv_kwargs, label="spikes"
    )


def _data(seed=0):
    rng = np.random.default_rng(seed)
    x = (rng.random(N_SAMPLES) < 0.1).astype(float)
    y = rng.poisson(0.5, N_SAMPLES).astype(float)
    return x, y


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("conv_kwargs", [{}, {"use_fft": None}])
def test_unset_use_fft_becomes_false(conv_kwargs):
    basis = _conv_basis(**conv_kwargs)
    use_direct_convolution(basis)
    assert basis.conv_kwargs["use_fft"] is False


@pytest.mark.filterwarnings("error")
def test_false_use_fft_is_kept():
    basis = _conv_basis(use_fft=False)
    use_direct_convolution(basis)
    assert basis.conv_kwargs["use_fft"] is False


def test_true_use_fft_is_kept_with_warning():
    basis = _conv_basis(use_fft=True)
    with pytest.warns(UserWarning, match="'spikes' sets use_fft=True"):
        use_direct_convolution(basis)
    assert basis.conv_kwargs["use_fft"] is True


def test_unknown_use_fft_raises():
    # nemos checks only the keys of conv_kwargs, so it accepts this value.
    basis = _conv_basis(use_fft="yes")
    with pytest.raises(ValueError, match="Expected True, False, or None"):
        use_direct_convolution(basis)


def test_other_conv_kwargs_are_kept():
    basis = _conv_basis(batch_size_samples=50)
    use_direct_convolution(basis)
    assert basis.conv_kwargs == {"batch_size_samples": 50, "use_fft": False}


def test_only_conv_components_change():
    basis = _conv_basis() + nmo.basis.BSplineEval(8, bounds=(0.0, 1.0))
    use_direct_convolution(basis)
    conv_component, eval_component = tuple(basis)
    assert conv_component.conv_kwargs["use_fft"] is False
    assert not hasattr(eval_component, "conv_kwargs")


def test_fit_sets_direct_convolution():
    x, y = _data()
    gam = GAM(_conv_basis(), maxiter=3)
    gam.fit((x,), y)
    assert gam.basis.conv_kwargs["use_fft"] is False


def test_use_fft_true_warning_points_at_fit_call():
    x, y = _data()
    gam = GAM(_conv_basis(use_fft=True), maxiter=3)
    with pytest.warns(UserWarning, match="use_fft=True") as record:
        gam.fit((x,), y)
    ours = [w for w in record if "use_fft=True" in str(w.message)]
    assert [w.filename for w in ours] == [__file__]


def test_nan_in_conv_input_stays_local():
    """
    One NaN in the input makes only the rows whose window contains it NaN.

    On the nemos FFT path, it makes every row NaN, and the default "drop" NaN
    policy then removes every row, so the fit raises.
    """
    x, y = _data()
    x[150] = np.nan
    gam = GAM(_conv_basis(), maxiter=3)
    gam.fit((x,), y)

    pred = np.asarray(gam.predict((x,)))
    # NaN rows: the first WINDOW rows (causal padding) and the WINDOW rows that
    # start at the NaN sample.
    assert np.isfinite(pred).sum() == N_SAMPLES - 2 * WINDOW
