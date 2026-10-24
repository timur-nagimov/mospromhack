"""Помощники для тестов источников: скриптованный cv2.VideoCapture без камеры и сети."""

from __future__ import annotations

import cv2
import numpy as np

FOURCC_AVC1 = int.from_bytes(b"avc1", "little")


class FakeCapture:
    """Скриптованный cv2.VideoCapture: frames кадров, дальше read() возвращает (False, None)."""

    def __init__(self, opened=True, frames=0, size=(64, 48), fps=25.0, fourcc=FOURCC_AVC1, pos_step_ms=40.0):
        self.opened = opened
        self.frames = frames
        self.size = size
        self.fps = fps
        self.fourcc = fourcc
        self.pos_step_ms = pos_step_ms
        self.pos_ms = 0.0
        self.read_calls = 0
        self.released = False
        self.args = None

    def isOpened(self):  # noqa: N802 — интерфейс cv2
        return self.opened

    def read(self):
        self.read_calls += 1
        if self.frames > 0:
            self.frames -= 1
            self.pos_ms += self.pos_step_ms
            w, h = self.size
            return True, np.zeros((h, w, 3), dtype=np.uint8)
        return False, None

    def get(self, prop):
        if prop == cv2.CAP_PROP_FPS:
            return self.fps
        if prop == cv2.CAP_PROP_FOURCC:
            return float(self.fourcc)
        if prop == cv2.CAP_PROP_POS_MSEC:
            return self.pos_ms
        return 0.0

    def set(self, prop, value):
        return True

    def release(self):
        self.released = True
