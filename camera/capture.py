from __future__ import annotations

import threading
import time
from typing import Optional, Tuple

import numpy as np
import cv2

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

try:
    from picamera2 import Picamera2
    _PICAM = True
except ImportError:
    _PICAM = False
    print("[CameraCapture] picamera2 not found, falling back to OpenCV v4l2\n"
          "  sudo apt install -y python3-picamera2  then recreate venv with --system-site-packages")

Frame = np.ndarray
CaptureResult = Tuple[Frame, float, int]


class CameraCapture:

    def __init__(self) -> None:
        self._width, self._height = config.CAMERA_RESOLUTION
        self._target_fps = config.CAMERA_TARGET_FPS
        self._buf: Frame = np.empty((self._height, self._width), dtype=np.uint8)

        self._latest_frame: Optional[Frame] = None
        self._latest_ts_us: float = 0.0
        self._frame_index: int = 0
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._fps_counter = 0
        self._fps_last_time = time.monotonic()
        self._measured_fps: float = 0.0

        self._backend = "picamera2" if _PICAM else "opencv"
        self._cam_picam = None
        self._cam_cv = None

    def start(self) -> None:
        if self._backend == "picamera2":
            self._init_picamera2()
        else:
            self._init_opencv()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._capture_loop, name="CameraCapture", daemon=True)
        self._thread.start()
        print(f"[CameraCapture] started ({self._backend}) @ {self._width}x{self._height}")

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=3.0)
        self._release()
        print("[CameraCapture] stopped.")

    def read(self) -> Optional[CaptureResult]:
        with self._lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame, self._latest_ts_us, self._frame_index

    @property
    def measured_fps(self) -> float:
        return self._measured_fps

    def _init_picamera2(self) -> None:
        cam = Picamera2()
        cam.start()

        print("[CameraCapture] waiting 2s for AE to settle...")
        import time as _t
        _t.sleep(2.0)

        meta = cam.capture_metadata()
        exposure = int(meta.get("ExposureTime", config.CAMERA_EXPOSURE_TIME))
        gain = float(meta.get("AnalogueGain", config.CAMERA_ANALOGUE_GAIN))
        frame_us = int(1_000_000 / self._target_fps)

        cam.set_controls({
            "AeEnable": False,
            "AwbEnable": False,
            "ExposureTime": exposure,
            "AnalogueGain": gain,
            "FrameDurationLimits": (frame_us, frame_us),
        })
        print(f"[CameraCapture] locked at {self._target_fps} FPS  exp={exposure}µs  gain={gain:.2f}")
        self._cam_picam = cam

    def _init_opencv(self) -> None:
        cap = None
        for idx in range(5):
            c = cv2.VideoCapture(idx, cv2.CAP_V4L2)
            if c.isOpened():
                ret, frame = c.read()
                if ret and frame is not None:
                    cap = c
                    print(f"[CameraCapture] OpenCV using /dev/video{idx}")
                    break
            c.release()

        if cap is None:
            raise RuntimeError("[CameraCapture] no camera found on /dev/video0-4")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        cap.set(cv2.CAP_PROP_FPS, self._target_fps)
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        cap.set(cv2.CAP_PROP_EXPOSURE, config.CAMERA_EXPOSURE_TIME / 1e6)
        self._cam_cv = cap

    def _capture_loop(self) -> None:
        while not self._stop_event.is_set():
            frame = self._grab_picamera2() if self._backend == "picamera2" else self._grab_opencv()
            if frame is None:
                continue

            ts_us = time.monotonic_ns() / 1_000.0
            with self._lock:
                self._latest_frame = frame
                self._latest_ts_us = ts_us
                self._frame_index += 1

            self._fps_counter += 1
            now = time.monotonic()
            elapsed = now - self._fps_last_time
            if elapsed >= 1.0:
                self._measured_fps = self._fps_counter / elapsed
                self._fps_counter = 0
                self._fps_last_time = now

    def _grab_picamera2(self) -> Optional[Frame]:
        try:
            raw = self._cam_picam.capture_array()
            if raw.shape[1] != self._width or raw.shape[0] != self._height:
                raw = cv2.resize(raw, (self._width, self._height))
            # XBGR8888: ch0 is padding, actual BGR is ch1-3
            if raw.ndim == 3 and raw.shape[2] == 4:
                cv2.cvtColor(raw[..., 1:4], cv2.COLOR_BGR2GRAY, dst=self._buf)
            elif raw.ndim == 3:
                cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY, dst=self._buf)
            else:
                np.copyto(self._buf, raw)
            return self._buf
        except Exception as e:
            print(f"[CameraCapture] grab error: {e}")
            return None

    def _grab_opencv(self) -> Optional[Frame]:
        ret, frame = self._cam_cv.read()
        if not ret:
            return None
        if frame.ndim == 3:
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY, dst=self._buf)
        else:
            if frame.shape[:2] != (self._height, self._width):
                frame = cv2.resize(frame, (self._width, self._height))
            np.copyto(self._buf, frame)
        if not hasattr(self, '_black_warned') and self._buf.max() == 0:
            print("[CameraCapture] ⚠ all-black frame — check CSI cable or run libcamera-hello")
            self._black_warned = True
        return self._buf

    def _release(self) -> None:
        try:
            if self._cam_picam:
                self._cam_picam.stop()
        except Exception:
            pass
        if self._cam_cv:
            self._cam_cv.release()