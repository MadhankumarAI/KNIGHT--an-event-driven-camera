import io
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Optional

import cv2
import numpy as np


class MJPEGServer:

    def __init__(self, port: int = 8080, quality: int = 80) -> None:
        self._port = port
        self._quality = quality
        self._lock = threading.Lock()
        self._frame: Optional[bytes] = None
        self._server: Optional[HTTPServer] = None

    def push_frame(self, bgr: np.ndarray) -> None:
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, self._quality])
        if ok:
            with self._lock:
                self._frame = buf.tobytes()

    def start(self) -> None:
        ref = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a): pass

            def do_GET(self):
                if self.path == "/":
                    html = b"<html><body style='background:#111'><img src='/stream' style='width:100%'></body></html>"
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.send_header("Content-Length", str(len(html)))
                    self.end_headers()
                    self.wfile.write(html)
                elif self.path == "/stream":
                    self.send_response(200)
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                    self.end_headers()
                    try:
                        while True:
                            with ref._lock:
                                frame = ref._frame
                            if frame is None:
                                time.sleep(0.02)
                                continue
                            hdr = (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                   + str(len(frame)).encode() + b"\r\n\r\n")
                            self.wfile.write(hdr + frame + b"\r\n")
                            self.wfile.flush()
                            time.sleep(0.033)
                    except (BrokenPipeError, ConnectionResetError):
                        pass

        self._server = HTTPServer(("0.0.0.0", self._port), Handler)
        threading.Thread(target=self._server.serve_forever, daemon=True, name="MJPEGServer").start()
        print(f"[MJPEGServer] http://localhost:{self._port}")

    def stop(self) -> None:
        if self._server:
            self._server.shutdown()
