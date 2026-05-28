import ctypes
from ctypes import util
import pickle
from pathlib import Path

__all__ = ["save_hierarchy", "load_hierarchy"]

def get_gpu_l2_cache_size(device_id=0):
    r"""
    Returns the L2 cache size in bytes for a given CUDA device ID.
    """
    libname = util.find_library('cuda')
    cuda = ctypes.CDLL(libname)
    device = ctypes.c_int()
    attribute_value = ctypes.c_int()
    L2_CACHE_ATTR = 38

    cuda.cuInit(0)
    cuda.cuDeviceGet(ctypes.byref(device), device_id)
    cuda.cuDeviceGetAttribute(ctypes.byref(attribute_value), L2_CACHE_ATTR, device)

    return attribute_value.value

def save_hierarchy(data, path, overwrite = False):
    r"""
    Save a file containing the full hierarchy information. Based on the implementation by Jakob-Unfried on https://github.com/jax-ml/jax/issues/2116.
    """
    path = Path(path)
    if path.suffix != '.hier':
        path = path.with_suffix('.hier')
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if overwrite:
            path.unlink()
        else:
            raise RuntimeError(f'File {path} already exists.')
    with open(path, 'wb') as file:
        pickle.dump(data, file)

def load_hierarchy(path):
    r"""
    Read a file containing the full hierarchy information. Based on the implementation by Jakob-Unfried on https://github.com/jax-ml/jax/issues/2116.
    """
    path = Path(path)
    if not path.is_file():
        raise ValueError(f'Not a file: {path}')
    if path.suffix != '.hier':
        raise ValueError(f'Not a .hier file: {path}')
    with open(path, 'rb') as file:
        data = pickle.load(file)
    return data