from __future__ import annotations

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class LogIntensityConverter:

    def __init__(self, height: int, width: int) -> None:
        self._tmp = np.empty((height, width), dtype=np.float32)
        self._out = np.empty((height, width), dtype=np.float32)
        self._eps = np.float32(config.LOG_EPSILON)

    def convert(self, frame_u8: np.ndarray) -> np.ndarray:
        # L = log(I/255 + eps)  —  log domain so threshold is contrast-relative
        np.multiply(frame_u8, np.float32(1.0 / 255.0), out=self._tmp)
        self._tmp += self._eps
        np.log(self._tmp, out=self._out)
        return self._out
