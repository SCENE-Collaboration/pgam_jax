import jax
from jax._src.config import bool_env

# enable x64 by default
if bool_env("JAX_ENABLE_X64", True):
    jax.config.update("jax_enable_x64", True)

# Use highest matmul precision unless the user has configured a default.
if jax.config.jax_default_matmul_precision is None:
    jax.config.update("jax_default_matmul_precision", "highest")

# monkey-patch nemos.basis.BSplineEval and nemos.basis.BSPlineConv to have a derivative() method
from ._patch_nemos import *  # isort: skip

from .gam import GAM

__all__ = ["GAM"]
