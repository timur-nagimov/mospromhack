"""Источники кадров: file, images, rtsp, webcam. Каждый — свой поток и слот последнего кадра."""

from __future__ import annotations

from guard.contracts import SourceConfig
from guard.sources.base import FrameCallback, FramePacket, FrameSource
from guard.sources.images import ImageFolderSource
from guard.sources.video import FileSource, LiveSource, WebcamSource


def make_source(
    cfg: SourceConfig,
    uri: str | None = None,
    camera_id: str = "",
    on_frame: FrameCallback | None = None,
    fps: float = 1.0,
) -> FrameSource:
    """Источник по типу. uri — уже разрешённый путь (по умолчанию cfg.uri); fps — темп для images."""
    uri = uri or cfg.uri
    if cfg.type == "file":
        return FileSource(cfg, uri, camera_id, on_frame)
    if cfg.type == "images":
        return ImageFolderSource(cfg, uri, camera_id, on_frame, fps=fps)
    if cfg.type == "rtsp":
        return LiveSource(cfg, uri, camera_id, on_frame)
    if cfg.type == "webcam":
        return WebcamSource(cfg, uri, camera_id, on_frame)
    raise ValueError(f"unknown source type {cfg.type!r}")


__all__ = ["FrameCallback", "FramePacket", "FrameSource", "FileSource", "ImageFolderSource", "LiveSource", "WebcamSource", "make_source"]
