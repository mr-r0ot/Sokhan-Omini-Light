"""Images for multimodal turns (optional: Pillow; the webcam also needs OpenCV).

    omni.config.vision.enabled = True     # before omni.start(); needs a vision LLM + mmproj
    cam = Webcam(omni).start()            # every user turn now carries the latest frame
"""
from __future__ import annotations

import base64
import io
import logging
import threading
from typing import Any, Optional

log = logging.getLogger("sokhan.vision")


def to_data_url(img: Any, max_side: int = 448, quality: int = 80) -> str:
    """Path, bytes, data URL, PIL image or numpy array (RGB, or BGR from OpenCV with ``bgr``)
    -> small JPEG data URL. Downscaling keeps the vision tokens (and latency) low."""
    from PIL import Image  # type: ignore
    import numpy as np

    if isinstance(img, str):
        if img.startswith("data:image"):
            return img
        im = Image.open(img)
    elif isinstance(img, (bytes, bytearray)):
        im = Image.open(io.BytesIO(img))
    elif isinstance(img, np.ndarray):
        im = Image.fromarray(img)
    else:
        im = img
    im = im.convert("RGB")
    w, h = im.size
    s = max_side / max(w, h)
    if s < 1.0:
        im = im.resize((max(1, int(w * s)), max(1, int(h * s))), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


class Webcam:
    """Keeps the latest webcam frame and hands it to the engine at every user turn."""

    def __init__(self, omni, device: int = 0, fps: float = 5.0):
        self.omni, self.device, self.fps = omni, device, fps
        self.frame = None                   # latest RGB numpy frame
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._unsub = None

    def start(self) -> "Webcam":
        try:
            import cv2  # type: ignore
        except ImportError as e:
            raise RuntimeError("the webcam needs `pip install opencv-python`") from e
        cap = cv2.VideoCapture(self.device)
        if not cap.isOpened():
            raise RuntimeError(f"cannot open camera {self.device}")

        def loop():
            while not self._stop.wait(1.0 / self.fps):
                ok, bgr = cap.read()
                if ok:
                    self.frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            cap.release()

        self._thread = threading.Thread(target=loop, daemon=True, name="sokhan-webcam")
        self._thread.start()
        self._unsub = self.omni.on("speech_start", self._attach)
        return self

    def _attach(self) -> None:
        if self.frame is not None:
            self.omni.set_image(self.frame)

    def snapshot(self):
        return self.frame

    def stop(self) -> None:
        self._stop.set()
        if self._unsub:
            self._unsub()
