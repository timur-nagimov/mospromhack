"""Confirm frame-level violations and suppress repeated notifications.

Timestamps must be finite and nondecreasing independently for each camera.
Trackless identities use stable IDs and one-to-one spatial matching against
the latest boxes within each camera, rule and zone.
Cooldown starts when closure is detected; Event.ended_at records the last
violating frame. Public methods return independent Event snapshots.
"""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import dataclass, field
from math import hypot, isfinite
from threading import Lock
from uuid import uuid4

from guard.contracts import BBox, CameraConfig, Event, RuleConfig, Violation


_EPS = 1e-9
_Key = tuple[str, str, int] | tuple[str, str, str | None, int]


@dataclass
class _Candidate:
    started_at: float
    last_violation_ts: float
    latest: Violation
    hits: deque[float] = field(default_factory=deque)


@dataclass
class _EventState:
    event: Event
    last_violation_ts: float
    closed_at: float | None = None


@dataclass
class _CameraState:
    frames: deque[float] = field(default_factory=deque)
    candidates: dict[_Key, _Candidate] = field(default_factory=dict)
    events: dict[str, _EventState] = field(default_factory=dict)
    aliases: dict[_Key, str] = field(default_factory=dict)
    next_spatial_id: int = 0


def _trim(timestamps: deque[float], cutoff: float) -> None:
    while timestamps and timestamps[0] < cutoff - _EPS:
        timestamps.popleft()


def _iou(first: BBox, second: BBox) -> float:
    width = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    height = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    intersection = width * height
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def _spatial_quality(first: BBox, second: BBox) -> tuple[float, float] | None:
    overlap = _iou(first, second)
    distance = hypot(
        (first[0] + first[2] - second[0] - second[2]) / 2,
        (first[1] + first[3] - second[1] - second[3]) / 2,
    )
    largest_dimension = max(
        0.0,
        first[2] - first[0],
        first[3] - first[1],
        second[2] - second[0],
        second[3] - second[1],
    )
    if overlap >= 0.3 or distance <= 0.5 * largest_dimension:
        return overlap, distance
    return None


def _match_trackless(
    state: _CameraState,
    camera_id: str,
    violations: list[Violation],
    reserved: set[str],
) -> dict[_Key, Violation]:
    """Assign stable keys greedily, using each observation and target once."""
    targets: dict[_Key, Violation | Event] = {
        key: candidate.latest
        for key, candidate in state.candidates.items()
        if candidate.latest.track_id is None
    }
    for key, event_id in state.aliases.items():
        # A four-part key is the persistent spatial alias created when a
        # trackless candidate confirms. Tracked aliases have three parts.
        if len(key) != 4 or event_id in reserved:
            continue
        event = state.events[event_id].event
        if event.track_id is None:
            # Recently closed events remain eligible until cooldown expires.
            targets[key] = event

    target_items = list(targets.items())
    matches: list[tuple[float, float, int, int]] = []
    for observation_index, violation in enumerate(violations):
        for target_index, (_, target) in enumerate(target_items):
            if (
                target.rule_id != violation.rule_id
                or target.zone_id != violation.zone_id
            ):
                continue
            quality = _spatial_quality(target.bbox, violation.bbox)
            if quality is not None:
                overlap, distance = quality
                matches.append(
                    (-overlap, distance, observation_index, target_index)
                )

    # Score every pair before updating any box. Prefer larger IoU, then
    # shorter center distance, independently of observation processing order.
    matches.sort()
    assigned: dict[int, _Key] = {}
    used_targets: set[int] = set()
    for _, _, observation_index, target_index in matches:
        if observation_index in assigned or target_index in used_targets:
            continue
        assigned[observation_index] = target_items[target_index][0]
        used_targets.add(target_index)

    current: dict[_Key, Violation] = {}
    for observation_index, violation in enumerate(violations):
        key = assigned.get(observation_index)
        if key is None:
            key = (
                camera_id,
                violation.rule_id,
                violation.zone_id,
                state.next_spatial_id,
            )
            state.next_spatial_id += 1
        current[key] = violation
    return current


def _overlapping_event(
    state: _CameraState,
    violation: Violation,
    reserved: set[str],
) -> _EventState | None:
    best = None
    best_iou = 0.3
    for event_id, record in state.events.items():
        if event_id in reserved or record.event.rule_id != violation.rule_id:
            continue
        overlap = _iou(record.event.bbox, violation.bbox)
        if overlap >= best_iou:
            best = record
            best_iou = overlap
    return best


def _touch(record: _EventState, violation: Violation, ts: float) -> None:
    if ts >= record.last_violation_ts:
        record.last_violation_ts = ts
        record.event.bbox = violation.bbox
    record.event.ended_at = None
    record.closed_at = None


class EventManager:
    """Maintain independent confirmation windows and event histories per camera."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._cameras: dict[str, _CameraState] = {}

    def process(
        self,
        camera: CameraConfig,
        ts: float,
        violations: list[Violation],
        timings: dict[str, float] | None = None,
    ) -> tuple[list[Event], list[Event]]:
        """Return newly confirmed and newly closed events for this frame.

        Empty frames count in confirmation ratios. Reopening an existing event
        preserves its ID and does not produce another confirmation notification.
        """
        if not isfinite(ts):
            raise ValueError("ts must be finite")

        with self._lock:
            state = self._cameras.setdefault(camera.camera_id, _CameraState())
            if state.frames and ts < state.frames[-1]:
                raise ValueError("ts must be nondecreasing for each camera")

            # One hit per tracked identity per frame; retain the last observation.
            current: dict[_Key, Violation] = {
                (camera.camera_id, v.rule_id, v.track_id): v
                for v in violations
                if v.track_id is not None
            }
            trackless = [v for v in violations if v.track_id is None]
            rules = {rule.rule_id: rule for rule in camera.rules}
            rule_types = {v.rule_id: v.type for v in current.values()}
            rule_types.update({v.rule_id: v.type for v in trackless})
            rule_types.update(
                {
                    candidate.latest.rule_id: candidate.latest.type
                    for candidate in state.candidates.values()
                }
            )
            rule_types.update(
                {
                    record.event.rule_id: record.event.type
                    for record in state.events.values()
                }
            )
            for rule_id, kind in rule_types.items():
                if rule_id not in rules:
                    rules[rule_id] = RuleConfig(rule_id=rule_id, type=kind)

            state.frames.append(ts)
            longest_window = max(
                (rule.min_duration_s for rule in rules.values()),
                default=0.0,
            )
            _trim(state.frames, ts - longest_window)

            expired = {
                event_id
                for event_id, record in state.events.items()
                if record.closed_at is not None
                and ts - record.closed_at
                > rules[record.event.rule_id].cooldown_s + _EPS
            }
            if expired:
                for event_id in expired:
                    del state.events[event_id]
                state.aliases = {
                    key: event_id
                    for key, event_id in state.aliases.items()
                    if event_id not in expired
                }

            for key, candidate in list(state.candidates.items()):
                rule = rules[candidate.latest.rule_id]
                if ts - candidate.last_violation_ts > rule.min_duration_s + _EPS:
                    del state.candidates[key]

            # Reserve known identities before considering spatial matches.
            # Two people visible together must not share an event merely because
            # their boxes overlap, regardless of their order in the input list.
            reserved = {
                state.aliases[key]
                for key in current
                if key in state.aliases
            }
            current.update(
                _match_trackless(state, camera.camera_id, trackless, reserved)
            )
            reserved.update(
                state.aliases[key]
                for key in current
                if key in state.aliases
            )

            for key, violation in current.items():
                event_id = state.aliases.get(key)
                record = (
                    state.events[event_id]
                    if event_id is not None
                    else None
                )
                if record is None and violation.track_id is not None:
                    record = _overlapping_event(state, violation, reserved)
                if record is not None:
                    state.aliases[key] = record.event.id
                    reserved.add(record.event.id)
                    _touch(record, violation, ts)
                    state.candidates.pop(key, None)
                    continue

                candidate = state.candidates.get(key)
                latest = deepcopy(violation)
                if candidate is None:
                    candidate = _Candidate(
                        started_at=ts,
                        last_violation_ts=ts,
                        latest=latest,
                    )
                    state.candidates[key] = candidate
                else:
                    candidate.last_violation_ts = ts
                    candidate.latest = latest
                candidate.hits.append(ts)

            created: list[Event] = []
            closed: list[Event] = []
            frame_counts: dict[float, int] = {}

            # Evaluate on every frame, including frames without a violation.
            for key, candidate in list(state.candidates.items()):
                violation = candidate.latest
                rule = rules[violation.rule_id]
                duration = rule.min_duration_s
                cutoff = ts - duration
                _trim(candidate.hits, cutoff)
                if ts - candidate.started_at + _EPS < duration:
                    continue

                if duration not in frame_counts:
                    frame_counts[duration] = sum(
                        frame_ts >= cutoff - _EPS
                        for frame_ts in state.frames
                    )
                total = frame_counts[duration]
                if total == 0 or len(candidate.hits) / total + _EPS < rule.min_ratio:
                    continue

                # Trackless identities were assigned together before any box
                # updates. A second spatial pass could merge separate objects.
                record = (
                    _overlapping_event(state, violation, reserved)
                    if violation.track_id is not None
                    else None
                )
                if record is not None:
                    state.aliases[key] = record.event.id
                    reserved.add(record.event.id)
                    _touch(record, violation, candidate.last_violation_ts)
                else:
                    event = Event(
                        id=uuid4().hex,
                        camera_id=camera.camera_id,
                        site=camera.site,
                        zone_id=violation.zone_id,
                        rule_id=violation.rule_id,
                        type=violation.type,
                        severity=violation.severity,
                        track_id=violation.track_id,
                        started_at=candidate.started_at,
                        confirmed_at=ts,
                        bbox=violation.bbox,
                        details=deepcopy(violation.details),
                        latency_ms=dict(timings) if timings is not None else {},
                    )
                    state.events[event.id] = _EventState(
                        event=event,
                        last_violation_ts=candidate.last_violation_ts,
                    )
                    state.aliases[key] = event.id
                    reserved.add(event.id)
                    created.append(event.model_copy(deep=True))
                del state.candidates[key]

            for record in state.events.values():
                rule = rules[record.event.rule_id]
                if (
                    record.closed_at is None
                    and ts - record.last_violation_ts > rule.close_after_s + _EPS
                ):
                    record.closed_at = ts
                    record.event.ended_at = record.last_violation_ts
                    closed.append(record.event.model_copy(deep=True))

            return created, closed

    def open_events(self, camera_id: str | None = None) -> list[Event]:
        """Return independent snapshots of currently open events."""
        with self._lock:
            return [
                record.event.model_copy(deep=True)
                for current_camera_id, state in self._cameras.items()
                if camera_id is None or current_camera_id == camera_id
                for record in state.events.values()
                if record.closed_at is None
            ]

    def reset(self, camera_id: str) -> None:
        """Forget candidates, frame history, events and aliases for one camera."""
        with self._lock:
            self._cameras.pop(camera_id, None)
