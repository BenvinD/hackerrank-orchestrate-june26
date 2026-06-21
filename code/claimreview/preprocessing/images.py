"""Image resolution, downscaling and base64 encoding.

Downscaling to a bounded long side is the main cost lever: image tokens dominate
VLM spend, and most claim damage is assessable at <=1024px. Each image is hashed
(on its encoded bytes) so the model-calling layer can cache by content and avoid
re-billing identical inputs across runs and across the two-model comparison.
"""

from __future__ import annotations

import base64
import hashlib
import io
from pathlib import Path

from PIL import Image, ImageOps

from ..config import ImageConfig
from ..schema import EncodedImage, image_id_from_path


def _downscale(img: Image.Image, max_long_side: int) -> Image.Image:
    """Shrink so the longer side <= max_long_side; never upscale."""
    w, h = img.size
    long_side = max(w, h)
    if long_side <= max_long_side:
        return img
    scale = max_long_side / float(long_side)
    new_size = (max(1, round(w * scale)), max(1, round(h * scale)))
    return img.resize(new_size, Image.LANCZOS)


def encode_image(
    rel_path: str,
    dataset_dir: Path,
    image_cfg: ImageConfig,
) -> EncodedImage:
    """Resolve, downscale and base64-encode one image into an ``EncodedImage``.

    ``rel_path`` is the dataset-relative path from a claim row, e.g.
    ``images/test/case_001/img_1.jpg``. Missing files or decode errors are
    captured on the record (``exists``/``error``) rather than raised, so one bad
    image never crashes a whole batch.
    """
    abs_path = (dataset_dir / rel_path).resolve()
    image_id = image_id_from_path(rel_path)

    if not abs_path.is_file():
        return EncodedImage(
            image_id=image_id,
            rel_path=rel_path,
            abs_path=str(abs_path),
            exists=False,
            error="file_not_found",
        )

    try:
        with Image.open(abs_path) as raw:
            raw = ImageOps.exif_transpose(raw)  # respect camera orientation
            orig_w, orig_h = raw.size
            rgb = raw.convert("RGB")
            scaled = _downscale(rgb, image_cfg.max_long_side)
            buf = io.BytesIO()
            scaled.save(buf, format=image_cfg.format, quality=image_cfg.jpeg_quality)
            data = buf.getvalue()
    except Exception as exc:  # noqa: BLE001 - record, don't crash the batch
        return EncodedImage(
            image_id=image_id,
            rel_path=rel_path,
            abs_path=str(abs_path),
            exists=True,
            error=f"decode_error: {type(exc).__name__}: {exc}",
            orig_width=None,
            orig_height=None,
        )

    media_type = "image/jpeg" if image_cfg.format.upper() in {"JPEG", "JPG"} else "image/png"
    b64 = base64.b64encode(data).decode("ascii")
    return EncodedImage(
        image_id=image_id,
        rel_path=rel_path,
        abs_path=str(abs_path),
        exists=True,
        media_type=media_type,
        width=scaled.width,
        height=scaled.height,
        orig_width=orig_w,
        orig_height=orig_h,
        content_hash=hashlib.sha256(data).hexdigest(),
        data_url=f"data:{media_type};base64,{b64}",
    )
