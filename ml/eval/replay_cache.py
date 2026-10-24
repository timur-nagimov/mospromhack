"""Файловый кэш детекций для последовательной офлайн-оценки."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import tempfile
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from guard.contracts import Detection
from guard.pipeline.detector import Detector


CACHE_VERSION = 1


def _file_metadata(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _postprocess_hash() -> str:
    root = Path(__file__).resolve().parents[2]
    paths = [
        root / "guard" / "pipeline" / "detector.py",
        *sorted((root / "guard" / "vision").glob("*.py")),
    ]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _copy_detections(detections: list[Detection]) -> list[Detection]:
    # Остальные поля Detection — скаляры и неизменяемый bbox.
    return [
        replace(
            detection,
            keypoints=(
                np.copy(detection.keypoints)
                if detection.keypoints is not None
                else None
            ),
        )
        for detection in detections
    ]


class DetectionCache:
    def __init__(self, cache_dir: Path, mode: str) -> None:
        if mode not in {"readwrite", "read", "write"}:
            raise ValueError(f"Unknown cache mode: {mode}")
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        self.mode = mode

    def for_video(
        self,
        video_path: Path,
        detectors: Mapping[str, Detector],
        backend_name: str,
        models_dir: Path,
    ) -> VideoCache:
        return VideoCache(
            self.cache_dir,
            self.mode,
            Path(video_path),
            detectors,
            backend_name,
            Path(models_dir),
        )


class VideoCache:
    def __init__(
        self,
        cache_dir: Path,
        mode: str,
        video_path: Path,
        detectors: Mapping[str, Detector],
        backend_name: str,
        models_dir: Path,
    ) -> None:
        self.cache_dir = cache_dir
        self.mode = mode
        self.video_path = video_path
        self.detectors = dict(detectors)
        self.hits = {model_id: 0 for model_id in detectors}
        self.misses = {model_id: 0 for model_id in detectors}
        self._paths: dict[str, Path] = {}
        self._key_parts: dict[str, dict[str, Any]] = {}
        self._frames: dict[str, dict[int, list[Detection]]] = {}
        self._dirty: set[str] = set()

        video_metadata = {"name": video_path.name, **_file_metadata(video_path)}
        code_hash = _postprocess_hash()
        for model_id, detector in self.detectors.items():
            spec = detector.spec
            onnx_path = Path(spec.onnx_path)
            if not onnx_path.is_absolute():
                onnx_path = models_dir / onnx_path
            try:
                onnx_metadata = _file_metadata(onnx_path)
            except FileNotFoundError:
                onnx_metadata = None

            key_parts = {
                "video": video_metadata,
                "model": spec.model_dump(mode="json"),
                "onnx": onnx_metadata,
                "backend": backend_name,
                "postprocess_sha256": code_hash,
                "version": CACHE_VERSION,
            }
            key = hashlib.sha256(
                json.dumps(key_parts, sort_keys=True).encode("utf-8")
            ).hexdigest()[:16]
            path = cache_dir / f"{video_path.stem}__{model_id}__{key}.pkl"
            self._paths[model_id] = path
            self._key_parts[model_id] = key_parts
            self._frames[model_id] = {}

            if mode == "write":
                continue
            try:
                with path.open("rb") as stream:
                    payload = pickle.load(stream)
            except FileNotFoundError:
                continue
            except (pickle.UnpicklingError, EOFError, AttributeError, ImportError) as exc:
                raise ValueError(f"Cannot read detection cache: {path}") from exc

            if (
                not isinstance(payload, dict)
                or payload.get("version") != CACHE_VERSION
                or payload.get("key_parts") != key_parts
                or not isinstance(payload.get("frames"), dict)
            ):
                raise ValueError(f"Invalid detection cache: {path}")
            self._frames[model_id] = payload["frames"]

    def detect(
        self,
        model_id: str,
        frame_idx: int,
        frame: np.ndarray,
        timings: dict[str, float] | None,
    ) -> list[Detection]:
        frames = self._frames[model_id]
        if self.mode != "write" and frame_idx in frames:
            self.hits[model_id] += 1
            return _copy_detections(frames[frame_idx])

        self.misses[model_id] += 1
        if self.mode == "read":
            raise ValueError(
                f"Detection cache miss: video {self.video_path}, "
                f"model {model_id}, frame {frame_idx}"
            )

        detections = self.detectors[model_id].detect(frame, timings)
        frames[frame_idx] = _copy_detections(detections)
        self._dirty.add(model_id)
        return detections

    def commit(self) -> None:
        """Публиковать изменения только после успешной обработки всего видео."""
        if not self._dirty:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        for model_id in sorted(self._dirty):
            path = self._paths[model_id]
            temporary_path = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="wb",
                    dir=self.cache_dir,
                    prefix=f".{path.name}.",
                    suffix=".tmp",
                    delete=False,
                ) as stream:
                    temporary_path = Path(stream.name)
                    pickle.dump(
                        {
                            "version": CACHE_VERSION,
                            "key_parts": self._key_parts[model_id],
                            "frames": self._frames[model_id],
                        },
                        stream,
                        protocol=5,
                    )
                os.replace(temporary_path, path)
                self._dirty.remove(model_id)
            finally:
                if temporary_path is not None:
                    temporary_path.unlink(missing_ok=True)
