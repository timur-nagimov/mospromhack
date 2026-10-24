"""Explicit model image preprocessing without torch or ultralytics."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Literal

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class LetterboxMeta:
    """Resize ratio and actual left/top padding, in model-input pixels."""

    scale: float
    pad_x: int
    pad_y: int
    orig_w: int
    orig_h: int


@dataclass(frozen=True, slots=True)
class DirectResizeMeta:
    """Original image size for normalized boxes from a direct resize."""

    orig_w: int
    orig_h: int


def resize_rgb(
    img_bgr: np.ndarray,
    size: tuple[int, int] = (640, 640),
) -> tuple[np.ndarray, DirectResizeMeta]:
    """D-FINE HF processor: PIL bilinear RGB resize, /255, float32 NCHW.

    Aspect ratio changes to the requested (height, width). There is no padding
    or mean/std normalization. PIL interpolation is intentional: OpenCV's
    INTER_LINEAR is not numerically equivalent to this source processor.
    """
    from PIL import Image

    if not isinstance(img_bgr, np.ndarray):
        raise TypeError("img_bgr must be a numpy array")
    if img_bgr.ndim != 3 or img_bgr.shape[2] != 3 or img_bgr.dtype != np.uint8:
        raise ValueError("img_bgr must be a uint8 array of shape (height, width, 3)")
    orig_h, orig_w = img_bgr.shape[:2]
    if orig_h <= 0 or orig_w <= 0:
        raise ValueError("img_bgr must not be empty")
    if len(size) != 2 or any(
        isinstance(value, bool) or not isinstance(value, Integral) or value <= 0
        for value in size
    ):
        raise ValueError("size must contain two positive integers (height, width)")
    rgb = Image.fromarray(np.ascontiguousarray(img_bgr[:, :, ::-1]))
    resized = np.asarray(rgb.resize((int(size[1]), int(size[0])), Image.Resampling.BILINEAR))
    tensor = np.ascontiguousarray(resized.transpose(2, 0, 1)[None], dtype=np.float32)
    tensor /= np.float32(255.)
    return tensor, DirectResizeMeta(orig_w=orig_w, orig_h=orig_h)


def letterbox(
    img_bgr: np.ndarray,
    size: tuple[int, int] = (640, 640),
    dtype: Literal["float32", "uint8"] = "float32",
) -> tuple[np.ndarray, LetterboxMeta]:
    """Convert a uint8 BGR image to contiguous RGB (1, 3, h, w).

    Resizing preserves aspect ratio, allows upscaling, and uses bilinear
    interpolation. Padding is centered, with any extra pixel on the
    bottom/right. Float32 output is normalized to [0, 1].
    """
    if not isinstance(img_bgr, np.ndarray):
        raise TypeError("img_bgr must be a numpy array")
    if img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
        raise ValueError("img_bgr must have shape (height, width, 3)")
    if img_bgr.dtype != np.uint8:
        raise ValueError("img_bgr must have dtype uint8")
    orig_h, orig_w = img_bgr.shape[:2]
    if orig_h == 0 or orig_w == 0:
        raise ValueError("img_bgr must not be empty")
    if len(size) != 2 or any(
        isinstance(value, bool) or not isinstance(value, Integral) or value <= 0
        for value in size
    ):
        raise ValueError("size must contain two positive integers (height, width)")
    if dtype not in ("float32", "uint8"):
        raise ValueError("dtype must be 'float32' or 'uint8'")

    target_h, target_w = map(int, size)
    scale = min(target_w / orig_w, target_h / orig_h)
    resized_w = max(1, min(target_w, round(orig_w * scale)))
    resized_h = max(1, min(target_h, round(orig_h * scale)))
    pad_x = (target_w - resized_w) // 2
    pad_y = (target_h - resized_h) // 2

    resized = cv2.resize(
        img_bgr, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR
    )
    padded = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
    padded[pad_y : pad_y + resized_h, pad_x : pad_x + resized_w] = resized
    tensor = np.ascontiguousarray(padded[:, :, ::-1].transpose(2, 0, 1)[None])
    if dtype == "float32":
        tensor = tensor.astype(np.float32)
        tensor /= np.float32(255.0)

    return tensor, LetterboxMeta(
        scale=scale,
        pad_x=pad_x,
        pad_y=pad_y,
        orig_w=orig_w,
        orig_h=orig_h,
    )


def make_batch(tensors: list[np.ndarray]) -> np.ndarray:
    """Join identically shaped and typed (1, 3, h, w) tensors along batch."""
    if not tensors:
        raise ValueError("cannot make a batch from an empty list")
    for tensor in tensors:
        if not isinstance(tensor, np.ndarray):
            raise TypeError("each tensor must be a numpy array")
        if (
            tensor.ndim != 4
            or tensor.shape[:2] != (1, 3)
            or tensor.shape[2] == 0
            or tensor.shape[3] == 0
        ):
            raise ValueError("each tensor must have shape (1, 3, h, w)")
        if tensor.dtype not in (np.dtype("float32"), np.dtype("uint8")):
            raise ValueError("tensor dtype must be float32 or uint8")
        if tensor.shape != tensors[0].shape or tensor.dtype != tensors[0].dtype:
            raise ValueError("all tensors must have the same shape and dtype")
    return np.ascontiguousarray(np.concatenate(tensors, axis=0))
