"""Image preprocessing for cost-efficient vision calls."""

from __future__ import annotations

import base64
import io
from pathlib import Path

from PIL import Image

from foodcheat.config import JPEG_QUALITY, MAX_IMAGE_SIDE


def load_and_preprocess(
    path: str | Path,
    max_side: int = MAX_IMAGE_SIDE,
    quality: int = JPEG_QUALITY,
) -> tuple[bytes, str, dict]:
    """Load image, strip alpha, resize, JPEG-encode.

    Returns (jpeg_bytes, mime, meta).
    """
    path = Path(path)
    with Image.open(path) as im:
        original = {
            "width": im.width,
            "height": im.height,
            "mode": im.mode,
            "format": im.format,
            "bytes": path.stat().st_size,
        }
        # Flatten alpha onto white
        if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
            rgba = im.convert("RGBA")
            background = Image.new("RGB", rgba.size, (255, 255, 255))
            background.paste(rgba, mask=rgba.split()[-1])
            im = background
        else:
            im = im.convert("RGB")

        w, h = im.size
        scale = min(1.0, max_side / max(w, h))
        if scale < 1.0:
            im = im.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)

        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=quality, optimize=True)
        data = buf.getvalue()

        meta = {
            **original,
            "out_width": im.width,
            "out_height": im.height,
            "out_bytes": len(data),
            "max_side": max_side,
            "quality": quality,
        }
        return data, "image/jpeg", meta


def to_data_url(path: str | Path, **kwargs) -> tuple[str, dict]:
    data, mime, meta = load_and_preprocess(path, **kwargs)
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{b64}", meta
