from __future__ import annotations

import numpy as np
import cv2
from typing import Optional
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from event_stream.event_buffer import EventBuffer
from utils.mjpeg_server import MJPEGServer

_GREY = 128


class EventRenderer:

    def __init__(self, height: int, width: int, buffer: EventBuffer) -> None:
        self._h = height
        self._w = width
        self._buffer = buffer
        self._canvas = np.full((height, width, 3), _GREY, dtype=np.uint8)
        self._window_us = config.VIZ_ACCUMULATION_WINDOW_MS * 1_000.0
        self._enabled = config.VISUALIZATION_ENABLED

        self._server: Optional[MJPEGServer] = None
        if self._enabled:
            self._server = MJPEGServer(port=8081)
            self._server.start()
            print("[EventRenderer] stream: http://localhost:8081")

    def render_once(self) -> Optional[np.ndarray]:
        if not self._enabled:
            return None
        events = self._buffer.get_recent(self._window_us)
        return self._build_frame(events)

    def show(self) -> bool:
        if not self._enabled:
            return False
        frame = self.render_once()
        if frame is None:
            return False
        if self._server:
            self._server.push_frame(frame)
        return True

    def destroy(self) -> None:
        if self._server:
            self._server.stop()

    def _build_frame(self, events: np.ndarray) -> np.ndarray:
        self._canvas[:] = _GREY
        if events.size == 0:
            return self._canvas
        xs, ys, pol = events["x"], events["y"], events["polarity"]
        self._canvas[ys[pol == 1],  xs[pol == 1]]  = (255, 255, 255)
        self._canvas[ys[pol == -1], xs[pol == -1]] = (0, 0, 0)
        return self._canvas
