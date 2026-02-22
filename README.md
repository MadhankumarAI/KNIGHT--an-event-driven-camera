# KNIGHT — Event-Driven Vision Pipeline

A DVS (Dynamic Vision Sensor) emulator built on a Raspberry Pi 5 with a standard CSI camera. Instead of processing video frames, it outputs asynchronous per-pixel events whenever a scene changes beyond a contrast threshold — mirroring how neuromorphic cameras like the DAVIS346 or Prophesee Metavision work.

---

## What it does

Each output event carries `(x, y, polarity, timestamp_µs)`.  
Polarity is `+1` for a brightness increase, `-1` for a decrease.  
A static scene produces near-zero events. Motion produces tight clusters along edges.

The core model, derived from the DVS pixel circuit:

```
L(x,y) = log( I(x,y)/255 + ε )
ΔL     = L_current − L_ref
fire   if  ΔL ≥  C  →  +1
fire   if  ΔL ≤ −C  →  −1
update:    L_ref += C · sign(ΔL)
```

The reference update is accumulated rather than reset — this is what separates DVS emulation from frame differencing and why the events track real contrast change.

---

## Hardware

- Raspberry Pi 5 (8GB)
- Raspberry Pi Camera Module v2 (Sony IMX219, CSI interface)

---

## Quick start

```bash
# one-time setup
sudo apt install -y python3-picamera2 python3-opencv
sudo cpupower frequency-set -g performance
python3 -m venv venv1 --system-site-packages
source venv1/bin/activate
pip install numpy

# verify camera
python3 minimal_cam_test.py        # saves test_frame.jpg

# DVS pipeline
python3 main.py
```

Open `http://localhost:8081` in a browser on the Pi to see the event visualization.  
Grey background = no event. White = brightness increase. Black = brightness decrease.

---

## Structure

```
event_camera/
├── config.py                   tuning parameters
├── main.py                     pipeline entry point
├── camera/capture.py           picamera2 + OpenCV fallback, background thread
├── processing/
│   ├── log_converter.py        uint8 → float32 log intensity
│   └── dvs_emulator.py         threshold, noise filter, event output
├── event_stream/event_buffer.py  preallocated ring buffer (500K events)
├── visualization/event_renderer.py  event frame → MJPEG browser stream
└── utils/
    ├── performance.py          CPU affinity, governor check, rolling stats
    └── mjpeg_server.py         stdlib HTTP MJPEG server, no extra deps
```

---

## Tuning

All parameters live in `config.py`. The two most impactful:

| Parameter | Default | Effect |
|---|---|---|
| `DVS_CONTRAST_THRESHOLD` | `0.30` | raise → fewer, cleaner events |
| `NOISE_FILTER_ENABLED` | `True` | drops isolated single-pixel noise |

The camera startup sequence: default picamera2 config → 2s AE settle → read actual exposure/gain from metadata → lock `FrameDurationLimits` at 60 FPS. Forcing format and frame rate before AE converges causes black frames.

---

## Results

Measured on a live run before noise tuning (`C=0.15`):

- 29.28 million events in a single session  
- 53,490 events on a single frame during heavy motion  
- ~710 events per frame on a static scene  
- Pipeline memory: 128–129 MB RSS  
- Latency: 28–32 ms mean at 30–34 FPS (camera was the bottleneck at 15 FPS)

After raising `C=0.30` and enabling the spatial coherence filter, the noise floor drops to near-zero on static scenes.

---

## Notes

`technical.txt` in the project root has the full documentation — DVS math, every bug encountered, fixes, and design decisions.

cv2.imshow is broken under VNC and renders black regardless of image data. Visualization uses a browser MJPEG stream instead.
