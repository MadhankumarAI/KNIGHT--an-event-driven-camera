"""
minimal_cam_test.py – Absolute minimum camera test.

Bypasses all cv2.imshow, all format gymnastics.
Just captures one frame and saves it as test_frame.jpg.

Run:
    python3 minimal_cam_test.py

Then open test_frame.jpg in the file manager.
If it's black → hardware/cable problem.
If it shows your scene → camera works, problem was in debug_camera.py.
"""

import time
from picamera2 import Picamera2

print("Starting Picamera2...")
cam = Picamera2()

# Use the absolute default config — let picamera2 decide format/size
cam.start()

# MUST wait for AE to settle — do NOT skip this sleep
print("Waiting 3 seconds for auto-exposure to stabilise...")
time.sleep(3)

# Capture one frame
frame = cam.capture_array()

print(f"Frame shape : {frame.shape}")
print(f"Frame dtype : {frame.dtype}")
print(f"Pixel range : min={frame.min()}  max={frame.max()}  mean={frame.mean():.1f}")

if frame.max() < 10:
    print("\n⚠  Pixel max < 10 — image is essentially black.")
    print("   Possible causes:")
    print("   1. Lens cap left on the camera module")
    print("   2. CSI cable inserted incorrectly (blue side facing away from contacts)")
    print("   3. Camera module not enabled — run: sudo raspi-config → Interface Options → Camera")
    print("   4. Wrong camera detected — run: libcamera-hello  (should show preview)")
else:
    print("\n✓  Frame looks valid — saving to test_frame.jpg")

# Save regardless so you can inspect it
import cv2
import numpy as np

# picamera2 default is XBGR8888 (4-channel) — handle both 3 and 4 channel
if frame.ndim == 3 and frame.shape[2] == 4:
    bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
elif frame.ndim == 3 and frame.shape[2] == 3:
    bgr = frame
else:
    bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

cv2.imwrite("test_frame.jpg", bgr)
print("Saved: test_frame.jpg  ← open this in the file manager to inspect")

cam.stop()
