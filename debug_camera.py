import time
import sys
import numpy as np
import cv2
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.mjpeg_server import MJPEGServer

try:
    from picamera2 import Picamera2
    PICAM = True
except ImportError:
    PICAM = False

WIDTH, HEIGHT, FPS = 320, 240, 60


def run_picamera2():
    cam = Picamera2()
    cam.start()

    print("waiting 2s for AE...")
    time.sleep(2)

    server = MJPEGServer(port=8080)
    server.start()
    print("open in browser: http://localhost:8080")

    fps_count, fps_t, save_t = 0, time.monotonic(), time.monotonic()

    while True:
        raw = cam.capture_array()

        if raw.ndim == 3 and raw.shape[2] == 4:
            bgr = raw[..., 1:4]   # XBGR8888 — ch0 is padding
        elif raw.ndim == 3:
            bgr = raw
        else:
            bgr = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        fps_count += 1
        now = time.monotonic()
        if now - fps_t >= 1.0:
            print(f"FPS={fps_count / (now - fps_t):.1f}  "
                  f"min={gray.min()} max={gray.max()} mean={gray.mean():.1f}")
            fps_count, fps_t = 0, now

        if now - save_t >= 5.0:
            cv2.imwrite("debug_frame.jpg", bgr)
            print("saved debug_frame.jpg")
            save_t = now

        server.push_frame(bgr)
        time.sleep(0.001)

    cam.stop()
    server.stop()


def run_opencv():
    cap = None
    for idx in range(4):
        c = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        if c.isOpened():
            ret, f = c.read()
            if ret and f is not None:
                cap = c
                print(f"opencv: /dev/video{idx}")
                break
        c.release()

    if cap is None:
        print("no camera found — check: ls /dev/video*")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)

    server = MJPEGServer(port=8080)
    server.start()
    print("open in browser: http://localhost:8080")

    fps_count, fps_t = 0, time.monotonic()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        fps_count += 1
        now = time.monotonic()
        if now - fps_t >= 1.0:
            print(f"FPS={fps_count / (now - fps_t):.1f}  mean={gray.mean():.1f}")
            fps_count, fps_t = 0, now
        server.push_frame(frame if frame.ndim == 3 else cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
        time.sleep(0.001)

    cap.release()
    server.stop()


if __name__ == "__main__":
    if PICAM:
        run_picamera2()
    else:
        run_opencv()