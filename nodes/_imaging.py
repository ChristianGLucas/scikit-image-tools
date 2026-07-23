"""Shared scikit-image helpers: canonical `Image` message <-> a decoded numpy
array, plus the structured-error contract every node in this package uses.

This package is deliberately OFFLINE and DETERMINISTIC: unlike image-tools/
opencv-tools, `Image.url` is never fetched (see messages.proto). Every node
catches `SkimgError` at its boundary and returns it as a structured `error`
field instead of raising — a malformed or unfetchable input never crashes the
node. Payload size, pixel-count, and memory/CPU/time bounds are the
platform's job (ingress/gateway/sidecar limits + sandboxed execution), not
this package's — a node is a pure input -> output function and does not
duplicate that enforcement.
"""
from __future__ import annotations

import io
import math

import numpy as np
from PIL import Image as PILImage

_ALIASES = {"JPG": "JPEG", "TIF": "TIFF"}


class SkimgError(ValueError):
    """Raised for any expected, caller-facing input problem. Every node
    catches this at its boundary and returns it as a structured `error`
    field rather than letting it propagate as a crash.
    """


def normalize_format(fmt: str) -> str:
    f = (fmt or "").strip().upper()
    return _ALIASES.get(f, f)


def load_array(image, allow_color: bool = True) -> np.ndarray:
    """Resolve a canonical `Image` message into a decoded numpy array.

    Raises `SkimgError` — never a raw exception — for any malformed/
    unsupported input. This package never fetches `url`; if `data` is empty
    and `url` is set, that is itself a structured error (offline-only, no
    network policy).
    """
    raw = bytes(image.data)
    if not raw:
        if image.url:
            raise SkimgError(
                "network fetch not supported: this package is offline-only and "
                "deterministic — supply the image inline via `data`, not `url`"
            )
        raise SkimgError("Image has no `data` (and no `url` to reject cleanly)")
    try:
        pil = PILImage.open(io.BytesIO(raw))
        pil.verify()
        # verify() invalidates the file pointer/object for further use; reopen.
        pil = PILImage.open(io.BytesIO(raw))
    except SkimgError:
        raise
    except Exception as exc:  # noqa: BLE001 — any decode failure is a structured error
        raise SkimgError(f"could not decode image: {exc}") from exc

    w, h = pil.size
    if w <= 0 or h <= 0:
        raise SkimgError(f"invalid image dimensions: {w}x{h}")

    try:
        if allow_color:
            if pil.mode not in ("RGB", "L", "RGBA"):
                pil = pil.convert("RGB")
        else:
            if pil.mode != "L":
                pil = pil.convert("L")
        arr = np.asarray(pil)
    except Exception as exc:  # noqa: BLE001
        raise SkimgError(f"could not decode image pixels: {exc}") from exc

    if arr.size == 0:
        raise SkimgError("decoded image has zero pixels")
    return arr


def to_gray(arr: np.ndarray) -> np.ndarray:
    """Convert an RGB/RGBA/grayscale uint8 array to a float64 grayscale array
    in [0, 1], via scikit-image's standard luminosity conversion.
    """
    from skimage.color import rgb2gray

    if arr.ndim == 2:
        return arr.astype(np.float64) / 255.0
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    return rgb2gray(arr)


def encode_gray_u8(arr01: np.ndarray) -> dict:
    """Encode a float array in [0, 1] (or already-uint8) as a single-channel
    PNG `Image` kwargs dict.
    """
    if arr01.dtype != np.uint8:
        clipped = np.clip(arr01, 0.0, 1.0)
        u8 = (clipped * 255.0).round().astype(np.uint8)
    else:
        u8 = arr01
    pil = PILImage.fromarray(u8, mode="L")
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return {"data": buf.getvalue(), "format": "PNG", "width": pil.width, "height": pil.height}


def encode_mask(bool_or_u8: np.ndarray) -> dict:
    """Encode a boolean/0-1/0-255 mask array as a binary (0/255) PNG."""
    arr = bool_or_u8
    if arr.dtype == bool:
        u8 = (arr.astype(np.uint8)) * 255
    elif arr.max(initial=0) <= 1:
        u8 = (arr.astype(np.uint8)) * 255
    else:
        u8 = arr.astype(np.uint8)
    pil = PILImage.fromarray(u8, mode="L")
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return {"data": buf.getvalue(), "format": "PNG", "width": pil.width, "height": pil.height}


def encode_rgb_u8(arr: np.ndarray) -> dict:
    """Encode a uint8 (or [0,1] float) HxWx3 array as an RGB PNG."""
    if arr.dtype != np.uint8:
        clipped = np.clip(arr, 0.0, 1.0)
        u8 = (clipped * 255.0).round().astype(np.uint8)
    else:
        u8 = arr
    pil = PILImage.fromarray(u8, mode="RGB")
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return {"data": buf.getvalue(), "format": "PNG", "width": pil.width, "height": pil.height}


def rescale_to_u8(arr: np.ndarray) -> np.ndarray:
    """Per-array min-max rescale to 0-255 uint8. Constant arrays map to 0."""
    arr = arr.astype(np.float64)
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-12:
        return np.zeros_like(arr, dtype=np.uint8)
    scaled = (arr - lo) / (hi - lo)
    return (scaled * 255.0).round().astype(np.uint8)


def shannon_entropy(gray01: np.ndarray, bins: int = 256) -> float:
    hist, _ = np.histogram(gray01, bins=bins, range=(0.0, 1.0))
    total = hist.sum()
    if total == 0:
        return 0.0
    p = hist.astype(np.float64) / total
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def segment_stats(labels: np.ndarray, source_arr: np.ndarray):
    """Shared per-segment summary-statistics computation for the three
    segmentation nodes (SLIC / Felzenszwalb / Watershed). `labels` is an
    int label array (0 or background excluded); `source_arr` is the
    original decoded image (grayscale or RGB/RGBA, any dtype) used for the
    mean-color stats. Returns (rows, total_segment_count) where `rows` is a
    list of dicts with label/area/centroid_row/centroid_col/mean_r/mean_g/
    mean_b for every segment found.
    """
    from skimage.measure import regionprops

    intensity = source_arr
    if intensity.ndim == 3 and intensity.shape[-1] == 4:
        intensity = intensity[..., :3]

    props = regionprops(labels, intensity_image=intensity)
    total = len(props)

    rows = []
    for p in props:
        im = p.intensity_mean
        if np.ndim(im) == 0:
            mean_r = mean_g = mean_b = float(im)
        else:
            im = np.atleast_1d(im)
            mean_r = float(im[0])
            mean_g = float(im[1]) if im.size > 1 else 0.0
            mean_b = float(im[2]) if im.size > 2 else 0.0
        rows.append(
            {
                "label": int(p.label),
                "area": float(p.area),
                "centroid_row": float(p.centroid[0]),
                "centroid_col": float(p.centroid[1]),
                "mean_r": mean_r,
                "mean_g": mean_g,
                "mean_b": mean_b,
            }
        )
    return rows, total
