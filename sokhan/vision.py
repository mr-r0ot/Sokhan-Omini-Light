"""Image preparation for multimodal turns (optional: needs Pillow)."""
from __future__ import annotations

import base64
import io
from typing import Any


def to_data_url(img: Any, max_side: int = 448, quality: int = 80) -> str:
    """Accepts a path, bytes, PIL.Image or numpy array (RGB/BGR uint8); returns a
    small JPEG data URL.  Downscaling keeps visual tokens (and latency) low."""
    from PIL import Image  # type: ignore
    import numpy as np

    if isinstance(img, str):
        if img.startswith("data:image"):
            return img
        im = Image.open(img)
    elif isinstance(img, (bytes, bytearray)):
        im = Image.open(io.BytesIO(img))
    elif isinstance(img, np.ndarray):
        im = Image.fromarray(img[..., ::-1] if img.ndim == 3 and img.shape[2] == 3 else img)
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
