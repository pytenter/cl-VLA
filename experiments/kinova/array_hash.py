"""C-order array digest using bounded leading-axis transfers from a device."""
import hashlib
import math
import jax
import numpy as np


def hash_array(value, target_bytes=64*2**20):
    digest = hashlib.sha256()
    shape = tuple(value.shape)
    digest.update(str((shape, str(value.dtype))).encode())
    if not shape:
        host = np.asarray(jax.device_get(value))
        digest.update(np.ascontiguousarray(host).reshape(-1).view(np.uint8))
        return digest.hexdigest()
    row_bytes = max(1, math.prod(shape[1:]) * np.dtype(value.dtype).itemsize)
    rows = max(1, target_bytes//row_bytes)
    for start in range(0, shape[0], rows):
        # A complete leading-axis row may exceed target_bytes. For the largest
        # existing scan weight, one row is128MiB, not the entire2.25GiB leaf.
        host = np.asarray(jax.device_get(value[start:start+rows]))
        digest.update(np.ascontiguousarray(host).reshape(-1).view(np.uint8))
        del host
    return digest.hexdigest()
