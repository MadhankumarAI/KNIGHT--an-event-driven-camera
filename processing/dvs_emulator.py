from __future__ import annotations

import numpy as np
from typing import Optional
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

EVENT_DTYPE = np.dtype([
    ("x",         np.int16),
    ("y",         np.int16),
    ("polarity",  np.int8),
    ("timestamp", np.float64),
])


class DVSEmulator:

    def __init__(self, height: int, width: int, contrast_threshold: Optional[float] = None) -> None:
        self._h = height
        self._w = width
        self._C = np.float32(contrast_threshold or config.DVS_CONTRAST_THRESHOLD)

        self._L_ref = np.full((height, width), np.nan, dtype=np.float32)
        self._delta = np.empty((height, width), dtype=np.float32)
        self._pos_mask = np.empty((height, width), dtype=bool)
        self._neg_mask = np.empty((height, width), dtype=bool)
        self._neigh = np.empty((height, width), dtype=bool)

        self._noise_filter = config.NOISE_FILTER_ENABLED
        self._seeded = False

        yy, xx = np.mgrid[0:height, 0:width]
        self._xx_flat = xx.ravel().astype(np.int16)
        self._yy_flat = yy.ravel().astype(np.int16)

        self.total_events: int = 0

    def process(self, log_frame: np.ndarray, timestamp_us: float) -> np.ndarray:
        if not self._seeded:
            np.copyto(self._L_ref, log_frame)
            self._seeded = True
            return np.empty(0, dtype=EVENT_DTYPE)

        np.subtract(log_frame, self._L_ref, out=self._delta)
        np.greater_equal(self._delta,  self._C, out=self._pos_mask)
        np.less_equal(   self._delta, -self._C, out=self._neg_mask)

        event_mask = self._pos_mask | self._neg_mask

        if self._noise_filter:
            # drop isolated pixels — keep only events with at least one neighbour
            np.logical_or(np.roll(event_mask,  1, axis=0), np.roll(event_mask, -1, axis=0), out=self._neigh)
            np.logical_or(self._neigh, np.roll(event_mask,  1, axis=1), out=self._neigh)
            np.logical_or(self._neigh, np.roll(event_mask, -1, axis=1), out=self._neigh)
            np.logical_and(event_mask, self._neigh, out=event_mask)

        self._L_ref[self._pos_mask] += self._C
        self._L_ref[self._neg_mask] -= self._C

        flat = np.flatnonzero(event_mask)
        if flat.size == 0:
            return np.empty(0, dtype=EVENT_DTYPE)

        events = np.empty(flat.size, dtype=EVENT_DTYPE)
        events["x"]         = self._xx_flat[flat]
        events["y"]         = self._yy_flat[flat]
        events["polarity"]  = np.where(self._pos_mask.ravel()[flat], np.int8(1), np.int8(-1)).astype(np.int8)
        events["timestamp"] = timestamp_us

        self.total_events += flat.size
        return events

    def reset_reference(self, log_frame: np.ndarray) -> None:
        np.copyto(self._L_ref, log_frame)

    @property
    def reference_frame(self) -> np.ndarray:
        return self._L_ref
