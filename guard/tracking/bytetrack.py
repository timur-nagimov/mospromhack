"""ByteTrack with a constant-velocity Kalman filter in physical time."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from guard.contracts import BBox, Detection, Track

from ._geometry import match_boxes, valid_detection


def _measurement(box: BBox) -> np.ndarray:
    x1, y1, x2, y2 = box
    return np.array(
        [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1],
        dtype=float,
    )


def _trim_history(history: deque, ts: float, window_s: float) -> None:
    cutoff = ts - window_s - 1e-9
    while history and history[0][0] < cutoff:
        history.popleft()


def _record(
    track: Track,
    ts: float,
    history_s: float,
    speed_history: deque,
) -> None:
    x1, y1, x2, y2 = track.bbox
    sample = (ts, (x1 + x2) / 2, y2, y2 - y1)
    for history, window_s in (
        (track.history, history_s),
        (speed_history, 2.0),
    ):
        if history and history[-1][0] == ts:
            history[-1] = sample
        else:
            history.append(sample)
        _trim_history(history, ts, window_s)

    # Kinematics always use the same two-second window, independently of
    # public history retention, including when history_s is less than two.
    recent = np.asarray(speed_history, dtype=float)
    track.velocity = (0.0, 0.0)
    track.speed_bh = 0.0
    if len(recent) < 2:
        return

    # Center relative times first to preserve precision with epoch timestamps.
    times = recent[:, 0] - recent[-1, 0]
    times -= times.mean()
    denominator = float(times @ times)
    if denominator <= 1e-12:
        return
    positions = recent[:, 1:3]
    positions = positions - positions.mean(axis=0)
    velocity = (times @ positions) / denominator
    track.velocity = (float(velocity[0]), float(velocity[1]))
    track.speed_bh = float(np.linalg.norm(velocity) / recent[:, 3].mean())


@dataclass
class _State:
    track: Track
    mean: np.ndarray
    covariance: np.ndarray
    active: bool = True
    history_s: float = 8.0
    speed_history: deque = field(default_factory=deque)

    @classmethod
    def create(
        cls,
        track_id: int,
        det: Detection,
        ts: float,
        history_s: float = 8.0,
    ) -> _State:
        measurement = _measurement(det.bbox)
        height = max(float(measurement[3]), 1.0)
        mean = np.zeros(8, dtype=float)
        mean[:4] = measurement
        covariance = np.diag(
            [max(0.1 * height, 1.0) ** 2] * 4
            + [(2.0 * height) ** 2] * 4
        )
        track = Track(
            track_id=track_id,
            cls=det.cls,
            bbox=tuple(float(value) for value in det.bbox),
            conf=float(det.conf),
            first_seen=ts,
            last_seen=ts,
            keypoints=None if det.keypoints is None else det.keypoints.copy(),
            # The contract's default maxlen=64 would truncate eight seconds
            # at higher frame rates. Retention here is bounded by time.
            history=deque(),
        )
        state = cls(track, mean, covariance, history_s=history_s)
        _record(track, ts, history_s, state.speed_history)
        return state

    @property
    def predicted_box(self) -> BBox:
        cx, cy, width, height = self.mean[:4]
        return (
            float(cx - width / 2),
            float(cy - height / 2),
            float(cx + width / 2),
            float(cy + height / 2),
        )

    def predict(self, dt: float) -> None:
        transition = np.eye(8)
        transition[:4, 4:] = np.eye(4) * dt

        # Piecewise constant random acceleration, expressed in px/s².
        acceleration = max(float(self.mean[3]) * 0.5, 1.0)
        gain = np.zeros((8, 4))
        gain[:4] = np.eye(4) * (dt * dt / 2.0)
        gain[4:] = np.eye(4) * dt
        process_noise = acceleration**2 * (gain @ gain.T)

        self.mean = transition @ self.mean
        self.mean[2:4] = np.maximum(self.mean[2:4], 1e-3)
        self.covariance = (
            transition @ self.covariance @ transition.T + process_noise
        )

    def observe(self, det: Detection, ts: float) -> None:
        measurement = _measurement(det.bbox)
        sigma = max(float(measurement[3]) * 0.05, 1.0)
        noise = np.eye(4) * sigma**2
        innovation_covariance = self.covariance[:4, :4] + noise
        kalman_gain = np.linalg.solve(
            innovation_covariance, self.covariance[:4, :]
        ).T
        self.mean += kalman_gain @ (measurement - self.mean[:4])
        self.mean[2:4] = np.maximum(self.mean[2:4], 1e-3)

        # Joseph form retains a positive semidefinite covariance.
        residual = np.eye(8)
        residual[:, :4] -= kalman_gain
        self.covariance = (
            residual @ self.covariance @ residual.T
            + kalman_gain @ noise @ kalman_gain.T
        )
        self.covariance = (self.covariance + self.covariance.T) / 2.0

        if ts > self.track.last_seen:
            self.track.hits += 1
        self.track.last_seen = ts
        # Public geometry/history describe observations; predictions are internal.
        self.track.bbox = tuple(float(value) for value in det.bbox)
        self.track.conf = float(det.conf)
        self.track.keypoints = (
            None if det.keypoints is None else det.keypoints.copy()
        )
        self.track.attributes.pop("lost_since", None)
        self.active = True
        _record(self.track, ts, self.history_s, self.speed_history)


class Tracker:
    """One tracker per camera and class group.

    High-confidence observations match confirmed active/lost tracks first.
    Low-confidence observations can maintain active confirmed tracks, but cannot
    initialize tracks or revive lost ones. Remaining high-confidence detections
    confirm tentative tracks and initialize new ones.

    After IoU association, unmatched high-confidence detections can recover
    unmatched confirmed tracks last observed at most one second ago. This
    fallback greedily matches bottom-center distances normalized by the larger
    dimension of the last observed box, with a maximum distance of 0.6.

    Timestamps must be finite and nondecreasing. Repeated timestamps do not
    increase hits or duplicate history samples. Returned Track objects are
    mutable and retain their attributes across updates.

    Public history retains history_s seconds; velocity uses two seconds.
    Lost tracks retain their last observed box, keypoints and pose attributes.
    attributes["lost_since"] is the timestamp of the first unmatched update
    and is removed on recovery. Only confirmed active tracks are returned.
    """

    def __init__(
        self,
        classes: set[str],
        high_thresh: float = 0.5,
        low_thresh: float = 0.1,
        match_iou: float = 0.3,
        new_track_thresh: float = 0.6,
        track_buffer_s: float = 3.0,
        min_hits: int = 2,
        history_s: float = 8.0,
    ) -> None:
        thresholds = (high_thresh, low_thresh, match_iou, new_track_thresh)
        if any(not np.isfinite(value) or not 0 <= value <= 1 for value in thresholds):
            raise ValueError("Thresholds must be finite values in [0, 1]")
        if low_thresh > high_thresh:
            raise ValueError("low_thresh must not exceed high_thresh")
        if not np.isfinite(track_buffer_s) or track_buffer_s < 0:
            raise ValueError("track_buffer_s must be finite and nonnegative")
        if isinstance(min_hits, bool) or not isinstance(min_hits, int) or min_hits < 1:
            raise ValueError("min_hits must be a positive integer")
        if not np.isfinite(history_s) or history_s <= 0:
            raise ValueError("history_s must be finite and positive")

        self.classes = set(classes)
        self.high_thresh = float(high_thresh)
        self.low_thresh = float(low_thresh)
        self.match_iou = float(match_iou)
        self.new_track_thresh = float(new_track_thresh)
        self.track_buffer_s = float(track_buffer_s)
        self.min_hits = min_hits
        self.history_s = float(history_s)
        self._states: list[_State] = []
        self._next_id = 1
        self._last_ts: float | None = None

    def _match(
        self, states: list[_State], dets: list[Detection]
    ) -> list[tuple[int, int]]:
        if not states or not dets:
            return []
        allowed = np.array(
            [[state.track.cls == det.cls for det in dets] for state in states],
            dtype=bool,
        )
        return match_boxes(
            [state.predicted_box for state in states],
            [det.bbox for det in dets],
            self.match_iou,
            allowed,
        )

    def _match_footpoints(
        self, states: list[_State], dets: list[Detection]
    ) -> list[tuple[int, int]]:
        candidates = []
        for state_index, state in enumerate(states):
            # Use the last observation: a posture change can invalidate the
            # Kalman prediction of the box's center and dimensions.
            x1, y1, x2, y2 = state.track.bbox
            foot_x = (x1 + x2) / 2.0
            scale = max(x2 - x1, y2 - y1)
            for det_index, det in enumerate(dets):
                if state.track.cls != det.cls:
                    continue
                dx1, _, dx2, dy2 = det.bbox
                distance = float(
                    np.hypot((dx1 + dx2) / 2.0 - foot_x, dy2 - y2) / scale
                )
                if distance <= 0.6:
                    candidates.append((distance, state_index, det_index))

        matches = []
        used_states: set[int] = set()
        used_dets: set[int] = set()
        for _, state_index, det_index in sorted(candidates):
            if state_index in used_states or det_index in used_dets:
                continue
            matches.append((state_index, det_index))
            used_states.add(state_index)
            used_dets.add(det_index)
        return matches

    def update(self, dets: list[Detection], ts: float) -> list[Track]:
        ts = float(ts)
        if not np.isfinite(ts):
            raise ValueError("ts must be finite")
        if self._last_ts is not None and ts < self._last_ts:
            raise ValueError("Timestamps must be nondecreasing")
        dt = 0.0 if self._last_ts is None else ts - self._last_ts

        # Expire before association, including when no intervening empty frames
        # were submitted.
        self._states = [
            state
            for state in self._states
            if ts - state.track.last_seen <= self.track_buffer_s + 1e-9
        ]
        was_active = {
            state.track.track_id for state in self._states if state.active
        }
        for state in self._states:
            state.predict(dt)
            state.active = False

        accepted = [
            det
            for det in dets
            if det.cls in self.classes and valid_detection(det)
        ]
        high = [det for det in accepted if det.conf >= self.high_thresh]
        low = [
            det
            for det in accepted
            if self.low_thresh <= det.conf < self.high_thresh
        ]
        confirmed = [
            state for state in self._states if state.track.hits >= self.min_hits
        ]
        tentative = [
            state for state in self._states if state.track.hits < self.min_hits
        ]

        used_high: set[int] = set()
        for state_index, det_index in self._match(confirmed, high):
            confirmed[state_index].observe(high[det_index], ts)
            used_high.add(det_index)

        low_candidates = [
            state
            for state in confirmed
            if not state.active and state.track.track_id in was_active
        ]
        for state_index, det_index in self._match(low_candidates, low):
            low_candidates[state_index].observe(low[det_index], ts)

        remaining_indices = [
            index for index in range(len(high)) if index not in used_high
        ]
        remaining_high = [high[index] for index in remaining_indices]
        for state_index, det_index in self._match(tentative, remaining_high):
            tentative[state_index].observe(remaining_high[det_index], ts)
            used_high.add(remaining_indices[det_index])

        # Third association pass, after all existing IoU matches have priority.
        # Low-confidence detections must not revive lost tracks.
        footpoint_candidates = [
            state
            for state in confirmed
            if not state.active and ts - state.track.last_seen <= 1.0 + 1e-9
        ]
        remaining_indices = [
            index for index in range(len(high)) if index not in used_high
        ]
        remaining_high = [high[index] for index in remaining_indices]
        for state_index, det_index in self._match_footpoints(
            footpoint_candidates, remaining_high
        ):
            footpoint_candidates[state_index].observe(remaining_high[det_index], ts)
            used_high.add(remaining_indices[det_index])

        # Unmatched tentative tracks are discarded; confirmed tracks can be lost.
        self._states = [
            state
            for state in self._states
            if state.active or state.track.hits >= self.min_hits
        ]
        for state in self._states:
            if not state.active:
                state.track.attributes.setdefault("lost_since", ts)
                _trim_history(state.track.history, ts, self.history_s)
                _trim_history(state.speed_history, ts, 2.0)

        for index, det in enumerate(high):
            if index not in used_high and det.conf >= self.new_track_thresh:
                self._states.append(
                    _State.create(self._next_id, det, ts, self.history_s)
                )
                self._next_id += 1

        self._last_ts = ts
        return [
            state.track
            for state in self._states
            if state.active and state.track.hits >= self.min_hits
        ]
