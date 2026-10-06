"""Audio I/O: any file/bytes libsndfile can read -> mono float32 at 16 kHz."""
from __future__ import annotations

import io
from math import gcd

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

SR = 16_000


def load_audio(src) -> np.ndarray:
    """``src`` is a path, raw bytes or a file-like object."""
    if isinstance(src, (bytes, bytearray)):
        src = io.BytesIO(src)
    x, sr = sf.read(src, dtype="float32", always_2d=True)
    x = x.mean(1)
    if sr != SR:
        g = gcd(sr, SR)
        x = resample_poly(x, SR // g, sr // g).astype(np.float32)
    peak = np.abs(x).max()
    return x / peak * 0.95 if peak > 1.0 else x


def duration(x: np.ndarray) -> float:
    return len(x) / SR
