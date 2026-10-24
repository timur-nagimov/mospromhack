"""S3 «Падение и неподвижность».

История наблюдений проверяется во всём immobile_window_s. Подтверждение
и дедупликацию нарушений выполняет EventManager.

Плагин хранит снимки наблюдений, поскольку FrameContext содержит только
активные треки. Если первый пропуск произошёл менее чем через секунду
после начала лежания или быстрого изменения формы рамки, последняя поза
может удерживаться до track_buffer_s секунд от последнего наблюдения.
Этот параметр следует согласовывать с буфером трекера.

Удержание не добавляет вымышленные наблюдения в историю. Замеченное после
падения движение проверяется по снимку на момент потери и не забывается
на последующих пустых кадрах.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from math import hypot, isfinite
from typing import Any, ClassVar

from guard.contracts import (
    CameraConfig,
    FrameContext,
    RuleConfig,
    Severity,
    Track,
    Violation,
)
from guard.rules.base import RulePlugin, register_rule

Sample = tuple[float, float, float, float]
_EPS = 1e-6


@dataclass
class _Memory:
    track: Track
    upright_at: float | None
    fall_since: float | None
    lost_since: float | None = None


@register_rule
class FallImmobileRule(RulePlugin):
    type: ClassVar[str] = "fall_immobile"
    title: ClassVar[str] = "Падение и неподвижность"
    required_models: ClassVar[frozenset[str]] = frozenset({"det", "pose"})
    default_severity: ClassVar[Severity] = Severity.critical
    params_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "title": "Падение и неподвижность",
        "properties": {
            "min_lying_s": {
                "type": "number",
                "title": "Мин. длительность позы лёжа, с",
                "minimum": 0.0,
                "default": 1.5,
            },
            "immobile_window_s": {
                "type": "number",
                "title": "Окно проверки неподвижности, с",
                "exclusiveMinimum": 0.0,
                "default": 3.0,
            },
            "max_shift_bh": {
                "type": "number",
                "title": "Макс. смещение, высот тела",
                "exclusiveMinimum": 0.0,
                "default": 0.3,
            },
            "allow_bbox_fallback": {
                "type": "boolean",
                "title": "Использовать форму рамки при неизвестной позе",
                "default": True,
            },
            "count_lost_after_fall": {
                "type": "boolean",
                "title": "Учитывать потерю трека сразу после падения",
                "default": True,
            },
            "track_buffer_s": {
                "type": "number",
                "title": "Удержание потерянного трека от последнего наблюдения, с",
                "description": "Должно соответствовать track_buffer_s трекера.",
                "minimum": 0.0,
                "default": 3.0,
            },
        },
        "additionalProperties": False,
    }

    def __init__(self, cfg: RuleConfig, camera: CameraConfig) -> None:
        super().__init__(cfg, camera)
        params = cfg.params
        self.min_lying_s = float(params.get("min_lying_s", 1.5))
        self.immobile_window_s = float(params.get("immobile_window_s", 3.0))
        self.max_shift_bh = float(params.get("max_shift_bh", 0.3))
        self.allow_bbox_fallback = bool(params.get("allow_bbox_fallback", True))
        self.count_lost_after_fall = bool(params.get("count_lost_after_fall", True))
        self.track_buffer_s = float(params.get("track_buffer_s", 3.0))
        for name, value, positive in (
            ("min_lying_s", self.min_lying_s, False),
            ("immobile_window_s", self.immobile_window_s, True),
            ("max_shift_bh", self.max_shift_bh, True),
            ("track_buffer_s", self.track_buffer_s, False),
        ):
            if not isfinite(value) or value < 0 or (positive and value == 0):
                raise ValueError(f"Invalid {name}")

        # RuleConfig имеет общий default high; явно заданный high сохраняем.
        self._severity = (
            cfg.severity
            if "severity" in cfg.model_fields_set
            else self.default_severity
        )
        self._memory: dict[int, _Memory] = {}
        self._last_ts: float | None = None

    @staticmethod
    def _lying_since(track: Track, ts: float) -> float | None:
        value = track.attributes.get("lying_since")
        if value is None:
            return None
        try:
            since = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        return since if isfinite(since) and since <= ts else None

    @staticmethod
    def _samples_shift_bh(
        samples: list[Sample], *, allow_single: bool = False
    ) -> float | None:
        if not samples:
            return None
        if any(
            not all(isfinite(value) for value in sample) or sample[3] <= 0
            for sample in samples
        ):
            return None
        if len(samples) == 1:
            return 0.0 if allow_single else None
        if max(sample[0] for sample in samples) <= min(
            sample[0] for sample in samples
        ):
            return None

        mean_height = sum(sample[3] for sample in samples) / len(samples)
        displacement = max(
            hypot(right[1] - left[1], right[2] - left[2])
            for index, left in enumerate(samples)
            for right in samples[index + 1 :]
        )
        shift = displacement / mean_height
        return float(shift) if isfinite(shift) else None

    def _shift_bh(self, track: Track, ts: float) -> float | None:
        """Диаметр всех наблюдений в окне / средняя высота рамки.

        Движение с возвратом также считается движением. Для активного
        трека нужны хотя бы две разновременные точки. Как и раньше,
        используется доступная часть окна; длительность задают
        min_lying_s и окно подтверждения EventManager.
        """
        samples = [
            sample
            for sample in track.history
            if ts - self.immobile_window_s - _EPS <= sample[0] <= ts
        ]
        return self._samples_shift_bh(samples)

    def _remember(self, track: Track, ts: float) -> None:
        observed_at = track.last_seen
        if (
            not isfinite(observed_at)
            or not isfinite(track.first_seen)
            or observed_at > ts
            or observed_at < track.first_seen
        ):
            self._memory.pop(track.track_id, None)
            return

        previous = self._memory.get(track.track_id)
        if previous is not None and previous.track.first_seen != track.first_seen:
            previous = None

        x1, y1, x2, y2 = track.bbox
        width, height = x2 - x1, y2 - y1
        upright_at = previous.upright_at if previous is not None else None
        if upright_at is not None and observed_at - upright_at > 1.0:
            upright_at = None
        if height > 1.5 * width:
            upright_at = observed_at

        pose_state = track.attributes.get("pose_state", "unknown")
        fall_since: float | None = None
        if pose_state == "lying":
            fall_since = self._lying_since(track, observed_at)
            # A directly observed transition can establish onset even if
            # the producer did not provide lying_since.
            if (
                fall_since is None
                and track.attributes.get("lying_since") is None
                and previous is not None
            ):
                if previous.track.attributes.get("pose_state") == "lying":
                    fall_since = previous.fall_since
                else:
                    fall_since = observed_at
        elif (
            pose_state == "unknown"
            and self.allow_bbox_fallback
            and width >= height
        ):
            if previous is not None and previous.fall_since is not None:
                # A missing pose estimate must not restart the fall clock.
                fall_since = previous.fall_since
            elif (
                upright_at is not None
                and 0 <= observed_at - upright_at <= 1.0
            ):
                fall_since = observed_at

        # Snapshot mutable tracks: the tracker reuses them on subsequent frames.
        snapshot = replace(
            track,
            attributes=dict(track.attributes),
            history=deque(track.history),
            keypoints=None,
        )
        self._memory[track.track_id] = _Memory(
            track=snapshot,
            upright_at=upright_at,
            fall_since=fall_since,
        )

    def _lost_violation(
        self, ctx: FrameContext, memory: _Memory
    ) -> Violation | None:
        if memory.lost_since is None:
            memory.lost_since = ctx.ts

        since = memory.fall_since
        if since is None or not 0 <= memory.lost_since - since < 1.0:
            return None

        lying_s = ctx.ts - since
        if lying_s + _EPS < self.min_lying_s:
            return None

        track = memory.track
        # Check actual post-fall movement at the LAST observation, not at
        # ctx.ts: otherwise crawling would age out on empty frames.
        start = max(since, track.last_seen - self.immobile_window_s)
        samples = [
            sample
            for sample in track.history
            if start - _EPS <= sample[0] <= track.last_seen
        ]
        # One observed lying position is sufficient only for this explicit
        # lost-after-fall heuristic. Missing frames are not measurements.
        shift_bh = self._samples_shift_bh(samples, allow_single=True)
        if shift_bh is None or shift_bh >= self.max_shift_bh:
            return None

        return self.violation(
            ctx,
            severity=self._severity,
            track_id=track.track_id,
            zone_id=None,
            bbox=tuple(float(value) for value in track.bbox),
            score=float(track.conf),
            details={
                "lying_s": float(lying_s),
                "shift_bh": shift_bh,
                "lost_after_fall": True,
                "lost_since": float(memory.lost_since),
            },
        )

    def evaluate(self, ctx: FrameContext) -> list[Violation]:
        if not isfinite(ctx.ts):
            return []
        if self._last_ts is not None and ctx.ts < self._last_ts:
            # A restarted video must not inherit observations from its future.
            self._memory.clear()
        self._last_ts = ctx.ts

        if self.count_lost_after_fall:
            self._memory = {
                track_id: memory
                for track_id, memory in self._memory.items()
                if 0 <= ctx.ts - memory.track.last_seen
                <= self.track_buffer_s + _EPS
            }
        else:
            self._memory.clear()

        out: list[Violation] = []
        present = {track.track_id for track in ctx.people}
        for track in ctx.people:
            if not all(isfinite(value) for value in track.bbox):
                self._memory.pop(track.track_id, None)
                continue
            x1, y1, x2, y2 = track.bbox
            width, height = x2 - x1, y2 - y1
            if width <= 0 or height <= 0:
                self._memory.pop(track.track_id, None)
                continue

            if self.count_lost_after_fall:
                self._remember(track, ctx.ts)

            pose_state = track.attributes.get("pose_state", "unknown")
            lying_s: float | None = None
            if pose_state == "lying":
                since = self._lying_since(track, ctx.ts)
                if since is None:
                    continue
                lying_s = float(ctx.ts - since)
                if lying_s + _EPS < self.min_lying_s:
                    continue
            elif (
                pose_state == "unknown"
                and self.allow_bbox_fallback
                and width / height >= 1.3
            ):
                # Age alone does not establish how long a person has lain.
                lying_s = None
            else:
                continue

            shift_bh = self._shift_bh(track, ctx.ts)
            if shift_bh is None or shift_bh >= self.max_shift_bh:
                continue

            out.append(
                self.violation(
                    ctx,
                    severity=self._severity,
                    track_id=track.track_id,
                    zone_id=None,
                    bbox=tuple(float(value) for value in track.bbox),
                    score=float(track.conf),
                    details={"lying_s": lying_s, "shift_bh": shift_bh},
                )
            )

        for track_id, memory in self._memory.items():
            if track_id not in present:
                violation = self._lost_violation(ctx, memory)
                if violation is not None:
                    out.append(violation)
        return out
