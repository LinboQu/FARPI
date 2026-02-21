import hashlib
import os
from typing import Any

import numpy as np


def _ensure_contiguous_bytes(arr: np.ndarray) -> bytes:
    a = np.ascontiguousarray(arr)
    return a.tobytes(order="C")


def sha1_of_array(arr: np.ndarray) -> str:
    h = hashlib.sha1()
    h.update(_ensure_contiguous_bytes(np.asarray(arr)))
    return h.hexdigest()


def sha1_of_file(path: str, chunk_size: int = 1024 * 1024) -> str:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def stats_of_array(arr: np.ndarray) -> dict[str, Any]:
    a = np.asarray(arr, dtype=np.float64)
    return {
        "min": float(np.nanmin(a)),
        "max": float(np.nanmax(a)),
        "mean": float(np.nanmean(a)),
        "std": float(np.nanstd(a)),
    }
