"""Папка изображений как камера: по кадру раз в 1/fps, по кругу."""

from __future__ import annotations

import glob
import time
from pathlib import Path

import cv2

from guard.contracts import SourceConfig
from guard.sources.base import ERROR, OFFLINE, ONLINE, FrameCallback, FrameSource

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(uri: str) -> list[Path]:
    p = Path(uri)
    if p.is_dir():
        files = [f for f in p.iterdir() if f.suffix.lower() in IMAGE_EXTS]
    else:  # glob-шаблон: data/images/site1/*.jpg
        files = [Path(f) for f in glob.glob(uri) if Path(f).suffix.lower() in IMAGE_EXTS]
    return sorted(files)


class ImageFolderSource(FrameSource):
    kind = "images"

    def __init__(
        self, cfg: SourceConfig, uri: str, camera_id: str = "", on_frame: FrameCallback | None = None, fps: float = 1.0
    ) -> None:
        super().__init__(cfg, uri, camera_id, on_frame)
        self.native_fps = max(0.05, float(fps))

    def _run(self) -> None:
        period = 1.0 / (self.native_fps or 1.0)
        while not self._stop.is_set():
            files = list_images(self.uri)
            if not files:
                self._error(f"no images in {self.uri}")
                self.connected = False
                self.state = ERROR
                if not self._sleep(2.0):
                    return
                continue
            self.connected = True
            self.state = ONLINE
            self._new_capture()
            next_t = time.monotonic()
            for f in files:
                if self._stop.is_set():
                    return
                t_read = time.perf_counter()
                frame = cv2.imread(str(f), cv2.IMREAD_COLOR)
                decode_ms = (time.perf_counter() - t_read) * 1000.0
                if frame is None:
                    self._error(f"cannot read image {f}")
                    continue
                self._publish(frame, decode_ms)
                next_t += period
                if self.cfg.realtime and not self._sleep(next_t - time.monotonic()):
                    return
            if not self.cfg.loop:
                self.eof = True
                self.connected = False
                self.state = OFFLINE
                return
