from __future__ import annotations

import os
import signal
import time

if "DISPLAY" not in os.environ:
    os.environ["DISPLAY"] = ":0"

import config
from camera.capture import CameraCapture
from processing.log_converter import LogIntensityConverter
from processing.dvs_emulator import DVSEmulator
from event_stream.event_buffer import EventBuffer
from visualization.event_renderer import EventRenderer
from utils.performance import PerformanceMonitor


def main() -> None:
    width, height = config.CAMERA_RESOLUTION

    perf = PerformanceMonitor()
    perf.setup()

    cam = CameraCapture()
    log_cvt = LogIntensityConverter(height, width)
    dvs = DVSEmulator(height, width)
    buf = EventBuffer(config.EVENT_BUFFER_CAPACITY, height, width)
    viz = EventRenderer(height, width, buf)

    _running = [True]

    def _shutdown(sig, frame):
        print("\n[Main] stopping pipeline.")
        _running[0] = False

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    cam.start()
    time.sleep(0.5)

    print(f"[Main] running  {width}x{height}  C={config.DVS_CONTRAST_THRESHOLD}")

    prev_idx = -1
    frame_count = 0
    diag_t = time.monotonic()

    while _running[0]:
        t0 = perf.tick()

        capture = cam.read()
        if capture is None:
            time.sleep(0.001)
            continue

        gray, ts_us, idx = capture
        if idx == prev_idx:
            time.sleep(0.0005)
            continue
        prev_idx = idx

        log_frame = log_cvt.convert(gray)
        events = dvs.process(log_frame, ts_us)
        buf.append(events)

        if config.VISUALIZATION_ENABLED:
            if not viz.show():
                break

        perf.tock(t0, event_count=events.size)
        frame_count += 1

        now_t = time.monotonic()
        if now_t - diag_t >= 2.0:
            print(f"[Main] frames={frame_count} | events_frame={events.size} | "
                  f"total={dvs.total_events:,} | cam_fps={cam.measured_fps:.1f}")
            diag_t = now_t

    cam.stop()
    viz.destroy()
    print(f"[Main] done. total events: {dvs.total_events:,}")


if __name__ == "__main__":
    main()
