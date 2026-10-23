"""Assign PPE, COCO-17 pose and phone observations to people."""

from __future__ import annotations

import math

import numpy as np

from guard.contracts import BBox, Detection, Track

from ._geometry import match_boxes, valid_detection

_PPE = {
    "helmet": ("helmet", True, 0.0, 0.35, 0.12),
    "no_helmet": ("helmet", False, 0.0, 0.35, 0.12),
    "vest": ("vest", True, 0.20, 0.75, 0.475),
    "no_vest": ("vest", False, 0.20, 0.75, 0.475),
    "respirator": ("respirator", True, 0.0, 0.35, 0.22),
    "no_respirator": ("respirator", False, 0.0, 0.35, 0.22),
}
_ATTRIBUTE_NAMES = ("helmet", "vest", "respirator")
_STATE_KEY = "_tracking_attribute_state"
_POINT_CONFIDENCE = 0.5


def _keypoints(value: np.ndarray | None) -> np.ndarray | None:
    if value is None:
        return None
    points = np.asarray(value, dtype=float)
    if points.shape != (17, 3):
        return None
    return points.copy()


def _confident(points: np.ndarray, indices: list[int]) -> list[int]:
    return [
        index
        for index in indices
        if np.all(np.isfinite(points[index]))
        and points[index, 2] >= _POINT_CONFIDENCE
    ]


def _pose_state(box: BBox, points: np.ndarray | None) -> str:
    x1, y1, x2, y2 = box
    if (x2 - x1) / (y2 - y1) >= 1.3:
        return "lying"
    if points is None or len(_confident(points, [5, 6, 11, 12])) != 4:
        return "unknown"
    shoulders = points[[5, 6], :2].mean(axis=0)
    hips = points[[11, 12], :2].mean(axis=0)
    torso = hips - shoulders
    if np.linalg.norm(torso) < 1e-6:
        return "unknown"
    angle = math.degrees(math.atan2(abs(float(torso[0])), abs(float(torso[1]))))
    return "lying" if angle > 55.0 else "standing"


def _phone_near_person(person: Track, phone: Detection) -> bool:
    x1, y1, x2, y2 = person.bbox
    px1, py1, px2, py2 = phone.bbox
    if min(x2, px2) <= max(x1, px1) or min(y2, py2) <= max(y1, py1):
        return False

    points = person.keypoints
    if points is None:
        return True

    targets = [
        points[index, :2] for index in _confident(points, [9, 10])
    ]
    head = _confident(points, [0, 1, 2, 3, 4])
    if head:
        targets.append(points[head, :2].mean(axis=0))
    if not targets:
        return False

    center = np.array([(px1 + px2) / 2.0, (py1 + py2) / 2.0])
    # COCO-17 has no head bounding box: use a documented body-height estimate.
    head_height = 0.20 * (y2 - y1)
    return any(
        float(np.linalg.norm(center - target)) < 0.6 * head_height
        for target in targets
    )


def assign_attributes(
    people: list[Track], dets: dict[str, list[Detection]], ts: float
) -> None:
    """Update attributes in place using nondecreasing timestamps.

    PPE uses one vote per attribute/person/timestamp over the last two seconds.
    The strongest compatible observation supplies that vote; ties across time
    produce None. Missing observations do not vote and old votes expire.

    Private, JSON-compatible smoothing state resides in Track.attributes,
    keeping it local to the Track instance instead of a global track-ID cache.
    Pose observations are matched one-to-one. Unmatched/stale pose data becomes
    unknown; current keypoints supplied by the tracker may also be used when
    the pose model is absent from dets.
    """
    ts = float(ts)
    if not math.isfinite(ts):
        raise ValueError("ts must be finite")
    for person in people:
        state = person.attributes.get(_STATE_KEY)
        if state is not None and ts < state["last_ts"]:
            raise ValueError("Attribute timestamps must be nondecreasing")

    votes: list[dict[str, tuple[float, bool]]] = [{} for _ in people]
    for det in dets.get("ppe", []):
        if det.cls not in _PPE or not valid_detection(det):
            continue
        name, value, top, bottom, anchor = _PPE[det.cls]
        cx = (det.bbox[0] + det.bbox[2]) / 2.0
        cy = (det.bbox[1] + det.bbox[3]) / 2.0
        candidates: list[tuple[float, int, int]] = []
        for index, person in enumerate(people):
            x1, y1, x2, y2 = person.bbox
            width, height = x2 - x1, y2 - y1
            if width <= 0 or height <= 0:
                continue
            if (
                x1 - 0.1 * width <= cx <= x2 + 0.1 * width
                and y1 + top * height <= cy <= y1 + bottom * height
            ):
                distance = (
                    ((cx - (x1 + x2) / 2.0) / width) ** 2
                    + ((cy - (y1 + anchor * height)) / height) ** 2
                )
                candidates.append((distance, person.track_id, index))
        if not candidates:
            continue
        _, _, index = min(candidates)
        previous = votes[index].get(name)
        if previous is None or det.conf > previous[0]:
            votes[index][name] = (float(det.conf), value)

    poses = [det for det in dets.get("pose", []) if valid_detection(det)]
    pose_matches = {
        person_index: poses[pose_index]
        for person_index, pose_index in match_boxes(
            [person.bbox for person in people],
            [det.bbox for det in poses],
            0.5,
        )
    }
    phones = [
        det
        for det in dets.get("det", [])
        if det.cls == "cell phone" and valid_detection(det)
    ]

    for index, person in enumerate(people):
        attributes = person.attributes
        state = attributes.setdefault(
            _STATE_KEY, {"last_ts": ts, "ppe": {}}
        )
        state["last_ts"] = ts
        histories = state["ppe"]
        for name in _ATTRIBUTE_NAMES:
            samples = [
                sample
                for sample in histories.get(name, [])
                if ts - 2.0 <= sample[0] < ts
            ]
            if name in votes[index]:
                samples.append((ts, votes[index][name][1]))
            histories[name] = samples
            positives = sum(value for _, value in samples)
            if not samples or positives * 2 == len(samples):
                attributes[name] = None
            else:
                attributes[name] = bool(positives * 2 > len(samples))

        previous_pose = attributes.get("pose_state")
        previous_since = attributes.get("lying_since")
        pose = pose_matches.get(index)
        pose_state = "unknown"
        current_points = (
            _keypoints(person.keypoints)
            if "pose" not in dets and person.last_seen == ts
            else None
        )
        person.keypoints = None
        if pose is not None:
            person.keypoints = _keypoints(pose.keypoints)
            pose_state = _pose_state(pose.bbox, person.keypoints)
        elif current_points is not None:
            person.keypoints = current_points
            pose_state = _pose_state(person.bbox, current_points)

        attributes["pose_state"] = pose_state
        if pose_state == "lying":
            attributes["lying_since"] = (
                previous_since
                if previous_pose == "lying" and previous_since is not None
                else ts
            )
        else:
            attributes["lying_since"] = None
        attributes["phone"] = any(
            _phone_near_person(person, phone) for phone in phones
        )
