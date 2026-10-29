"""Optional warp-tiled CUDA pair forces exposed to JAX through XLA FFI."""

from __future__ import annotations

import ctypes
import os
import threading
from numbers import Real
from pathlib import Path

import jax
import jax.numpy as jnp

_TARGET_NAME = "opendust_softened_yukawa_cuda"
_DEFAULT_LIBRARY = Path(__file__).with_name("_cuda") / "libopendust_jax_pair_forces.so"
_registration_lock = threading.Lock()
_library: ctypes.CDLL | None = None
_registration_error: Exception | None = None


def cuda_pair_forces_library_path() -> Path:
    """Return the configured shared-library path without loading it."""

    override = os.environ.get("OPENDUST_JAX_CUDA_LIBRARY")
    return Path(override).expanduser().resolve() if override else _DEFAULT_LIBRARY


def _register_cuda_target() -> None:
    global _library, _registration_error
    if _library is not None:
        return
    with _registration_lock:
        if _library is not None:
            return
        path = cuda_pair_forces_library_path()
        if not path.is_file():
            raise RuntimeError(
                f"OpenDust JAX CUDA library was not found at {path}. "
                "Build it with performance/JAX/build_cuda_pair_forces.py."
            )
        try:
            library = ctypes.CDLL(str(path))
            handler = library.OpenDustSoftenedYukawa
            jax.ffi.register_ffi_target(
                _TARGET_NAME,
                jax.ffi.pycapsule(handler),
                platform="CUDA",
            )
        except Exception as exc:
            _registration_error = exc
            raise RuntimeError(
                f"Failed to load or register the CUDA pair-force library {path}: {exc}"
            ) from exc
        _library = library
        _registration_error = None


def cuda_pair_forces_available() -> bool:
    """Return whether the library exists and can be registered in this process."""

    if jax.default_backend() != "gpu":
        return False
    try:
        _register_cuda_target()
    except RuntimeError:
        return False
    return True


def softened_yukawa_forces_cuda(
    positions: jax.Array,
    prefactor: float | jax.Array,
    screening_length: float | jax.Array,
    softening: float | jax.Array,
) -> jax.Array:
    """Evaluate exact pair forces with a triangular 32x32 CUDA warp tiling.

    Every unordered particle pair is evaluated once. Forces on both particles
    are accumulated with FP32 atomics. The operation supports ``jax.jit`` but
    intentionally does not define automatic-differentiation or batching rules.
    """

    positions = jnp.asarray(positions, dtype=jnp.float32)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape (N, 3).")
    if positions.shape[0] == 0:
        raise ValueError("positions must contain at least one particle.")
    if isinstance(screening_length, Real) and screening_length <= 0.0:
        raise ValueError("screening_length must be positive.")
    if isinstance(softening, Real) and softening < 0.0:
        raise ValueError("softening must be non-negative.")

    parameters = tuple(
        jnp.asarray(value, dtype=jnp.float32)
        for value in (prefactor, screening_length, softening)
    )
    for name, value in zip(
        ("prefactor", "screening_length", "softening"), parameters
    ):
        if value.ndim != 0:
            raise ValueError(f"{name} must be a scalar.")

    _register_cuda_target()
    result = jax.ShapeDtypeStruct(positions.shape, jnp.float32)
    return jax.ffi.ffi_call(_TARGET_NAME, result)(positions, *parameters)


__all__ = [
    "cuda_pair_forces_available",
    "cuda_pair_forces_library_path",
    "softened_yukawa_forces_cuda",
]
