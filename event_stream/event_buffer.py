from __future__ import annotations

import time
import numpy as np
from typing import Optional, Tuple
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from processing.dvs_emulator import EVENT_DTYPE


class EventBuffer:

    def __init__(self, capacity: int, height: int, width: int) -> None:
        self._capacity = capacity
        self._h = height
        self._w = width
        self._buf = np.empty(capacity, dtype=EVENT_DTYPE)
        self._write_ptr = 0
        self._total_written = 0

        self._rate_t = time.monotonic()
        self._rate_count = 0
        self._event_rate: float = 0.0

        self._density = np.zeros((height, width), dtype=np.float32)

    def append(self, events: np.ndarray) -> None:
        n = events.size
        if n == 0:
            return

        start = self._write_ptr
        end = start + n
        if end <= self._capacity:
            self._buf[start:end] = events
        else:
            first = self._capacity - start
            self._buf[start:] = events[:first]
            self._buf[:n - first] = events[first:]

        self._write_ptr = end % self._capacity
        self._total_written += n

        self._rate_count += n
        now = time.monotonic()
        elapsed = now - self._rate_t
        if elapsed >= 0.5:
            self._event_rate = self._rate_count / elapsed
            self._rate_count = 0
            self._rate_t = now

        np.add.at(self._density, (events["y"], events["x"]), 1)

    def get_recent(self, window_us: float) -> np.ndarray:
        if self._total_written == 0:
            return np.empty(0, dtype=EVENT_DTYPE)

        cutoff = (time.monotonic_ns() / 1000.0) - window_us
        if self._total_written < self._capacity:
            live = self._buf[:self._total_written]
        else:
            live = np.concatenate((self._buf[self._write_ptr:], self._buf[:self._write_ptr]))

        return live[live["timestamp"] >= cutoff]

    def event_rate(self) -> float:
        return self._event_rate

    def density_map(self, reset: bool = True) -> np.ndarray:
        out = self._density.copy()
        if reset:
            self._density[:] = 0.0
        return out

    def event_count(self) -> int:
        return self._total_written
