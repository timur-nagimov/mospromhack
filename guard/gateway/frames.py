"""A preview and its metadata are one immutable observation of one edge frame."""

from __future__ import annotations

import base64
import copy
import io
import math
import threading
import time
from typing import Any, Callable

from PIL import Image, ImageDraw, UnidentifiedImageError


class RemoteCameraView:
    def __init__(
        self, camera_id: str, stale_after_s: float = 10.0,
        clock: Callable[[], float] = time.monotonic, wall_clock: Callable[[], float] = time.time,
    ):
        self.camera_id = camera_id
        self.stale_after_s = stale_after_s
        self._clock = clock
        self._wall_clock = wall_clock
        self._lock = threading.Lock()
        self._sequence = -1
        self._run_id: str | None = None
        self._producer_id: str | None = None
        self._received_at: float | None = None
        self._expires_at: float | None = None
        self._capture_age = 0.0
        self._payload: dict[str, Any] | None = None
        self._raw: bytes | None = None
        self._overlay: bytes | None = None

    def update(self, payload: dict[str, Any], *, sequence: int, run_id: str, producer_id: str = "") -> bool:
        payload = copy.deepcopy(payload)
        if payload.get("camera_id") != self.camera_id:
            raise ValueError("Frame camera identity mismatch")
        captured_at = payload.get("capture_ts")
        if type(captured_at) not in (int, float) or not math.isfinite(captured_at):
            raise ValueError("Frame requires a finite capture timestamp")
        capture_age = self._wall_clock() - captured_at
        if capture_age >= self.stale_after_s or capture_age < -self.stale_after_s:
            return False
        capture_age = max(0.0, capture_age)
        received_at = self._clock()
        expires_at = received_at + self.stale_after_s - capture_age
        try:
            raw = base64.b64decode(payload["jpeg_b64"], validate=True)
            width, height = payload["width"], payload["height"]
            if (not isinstance(width, int) or not isinstance(height, int)
                    or not 1 <= width <= 4096 or not 1 <= height <= 4096 or width * height > 16_777_216
                    or not raw or len(raw) > 1_048_576):
                raise ValueError("Preview dimensions or size exceed limits")
            with Image.open(io.BytesIO(raw)) as source:
                if source.format != "JPEG" or source.size != (width, height):
                    raise ValueError("Preview content does not match its metadata")
                image = source.convert("RGB")
            overlay = self._draw(image, payload)
        except (KeyError, TypeError, OSError, UnidentifiedImageError) as exc:
            raise ValueError("Invalid JPEG preview") from exc
        with self._lock:
            if self._producer_id == producer_id and sequence <= self._sequence:
                return False
            if self._clock() >= expires_at:
                return False
            self._producer_id, self._run_id, self._sequence = producer_id, run_id, sequence
            self._raw, self._overlay = raw, overlay
            self._payload = payload
            self._received_at = received_at
            self._capture_age = capture_age
            # Queueing/reconnect time consumes the frame's TTL. A new sent_at
            # or a delayed receipt must never make an old capture live again.
            self._expires_at = expires_at
            return True

    @staticmethod
    def _draw(image: Image.Image, payload: dict[str, Any]) -> bytes:
        draw = ImageDraw.Draw(image)
        width, height = image.size

        def coordinates(values):
            if any(not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1 for v in values):
                raise ValueError("Frame geometry must be finite and normalized")
            return [v * (width if i % 2 == 0 else height) for i, v in enumerate(values)]

        for zone in payload.get("zones", []):
            polygon = zone.get("polygon", [])
            if len(polygon) < 3:
                raise ValueError("Zone needs at least three points")
            points = [tuple(coordinates(point)) for point in polygon]
            draw.line(points + [points[0]], fill="#f3b84d", width=2)
        highlights = set(payload.get("highlight", []))
        for track in [*payload.get("people", []), *payload.get("objects", [])]:
            if len(track.get("bbox", [])) != 4:
                raise ValueError("Track needs a normalized bounding box")
            box = coordinates(track["bbox"])
            if box[2] < box[0] or box[3] < box[1]:
                raise ValueError("Invalid track bounding box")
            color = "#ff5454" if track.get("track_id") in highlights else "#5edb9d"
            draw.rectangle(box, outline=color, width=2)
            draw.text((box[0] + 2, box[1] + 2), f"#{track.get('track_id', '?')}", fill=color)
        out = io.BytesIO()
        image.save(out, format="JPEG", quality=75)
        return out.getvalue()

    def _fresh(self) -> bool:
        return self._received_at is not None and self._expires_at is not None and self._clock() < self._expires_at

    @property
    def running(self) -> bool:
        with self._lock:
            return self._fresh()

    def frame_jpeg(self) -> bytes | None:
        with self._lock:
            return self._raw if self._fresh() else None

    def overlay_jpeg(self, mode: str = "analyzed", max_width: int = 1280, quality: int = 80):
        with self._lock:
            if not self._fresh():
                return None
            identity = (self._producer_id, self._run_id, self._sequence, mode)
            return identity, self._raw if mode == "live" else self._overlay

    def invalidate(self) -> None:
        with self._lock:
            self._received_at = None

    def status(self) -> dict[str, Any]:
        with self._lock:
            fresh = self._fresh()
            payload = self._payload or {}
            age = None if self._received_at is None else self._capture_age + max(0.0, self._clock() - self._received_at)
            return {
                "running": fresh, "stale": not fresh, "preview_age_s": age,
                "fps_in": 0.0, "fps_processed": 0.0, "lag_s": age or 0.0,
                "errors": {} if fresh else {"edge": "No fresh edge preview"},
                "frame_identity": {
                    "producer_id": self._producer_id, "run_id": self._run_id,
                    "capture_id": payload.get("capture_id"), "frame_seq": payload.get("frame_seq"),
                    "capture_ts": payload.get("capture_ts"), "source_pos_s": payload.get("source_pos_s"),
                },
            }
