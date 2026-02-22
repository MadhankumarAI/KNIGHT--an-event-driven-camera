from __future__ import annotations

import os
import time
import collections
from typing import Deque, Optional

import numpy as np
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

try:
    import psutil
    _PSUTIL = True
except ImportError:
    _PSUTIL = False


class PerformanceMonitor:

    def __init__(self, window: int = 120) -> None:
        self._durations: Deque[float] = collections.deque(maxlen=window)
        self._ev_counts: Deque[int] = collections.deque(maxlen=window)
        self._last_report = time.monotonic()

    def setup(self) -> None:
        self._pin_cpu()
        self._check_governor()

    @staticmethod
    def tick() -> float:
        return time.monotonic_ns() / 1_000_000.0

    def tock(self, t0: float, event_count: int = 0) -> None:
        self._durations.append(self.tick() - t0)
        self._ev_counts.append(event_count)
        now = time.monotonic()
        if now - self._last_report >= config.PERF_REPORT_INTERVAL_SEC:
            self.report()
            self._last_report = now

    def report(self) -> None:
        if not self._durations:
            return
        arr = np.asarray(self._durations, dtype=np.float32)
        mean = float(arr.mean())
        fps = 1000.0 / mean if mean > 0 else 0
        ev_rate = sum(self._ev_counts) / (mean * len(arr) / 1000.0) if mean > 0 else 0
        mem = self._mem_mb()
        mem_str = f" | mem={mem:.0f}MB" if mem else ""
        print(f"[Perf] fps={fps:5.1f} | mean={mean:.2f}ms | p95={float(np.percentile(arr, 95)):.2f}ms"
              f" | max={float(arr.max()):.2f}ms | events/s={ev_rate:,.0f}{mem_str}")

    @staticmethod
    def _pin_cpu() -> None:
        cores = config.CPU_AFFINITY_CORES
        if not cores:
            return
        try:
            os.sched_setaffinity(0, cores)
            print(f"[Perf] CPU pinned to cores {cores}")
        except (AttributeError, PermissionError, OSError):
            pass

    @staticmethod
    def _check_governor() -> None:
        path = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"
        if not os.path.exists(path):
            return
        try:
            gov = open(path).read().strip()
            if gov != "performance":
                print(f"[Perf] governor='{gov}' — run: sudo cpupower frequency-set -g performance")
            else:
                print(f"[Perf] governor: {gov} ✓")
        except OSError:
            pass

    @staticmethod
    def _mem_mb() -> Optional[float]:
        if not _PSUTIL:
            return None
        try:
            return psutil.Process().memory_info().rss / (1024 * 1024)
        except Exception:
            return None
