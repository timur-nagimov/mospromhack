"""Shared, opt-in owner-bound PPE crops; never constructs an inference backend."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, replace

import numpy as np

from guard.contracts import Detection, Track
from guard.tracking.attributes import ppe_attribute, ppe_candidates

ATTRIBUTES = frozenset({"helmet", "vest", "respirator"})


class PpeFrameError(RuntimeError):
    def __init__(self, result):
        super().__init__("Incomplete PPE inference/ROI coverage")
        self.audit = result.audit_record()


@dataclass(frozen=True)
class PpePolicy:
    mode: str = "full_frame"
    audit: bool = False

    def __post_init__(self):
        if self.mode not in {"full_frame", "person_crop"}:
            raise ValueError("PPE mode must be full_frame or person_crop")


@dataclass
class PpeFrameResult:
    mode: str
    detections: list[Detection] = field(default_factory=list)
    by_track: dict[int, list[Detection]] = field(default_factory=dict)
    unknown: dict[int, set[str]] = field(default_factory=dict)
    rois: list[dict] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=lambda: dict(eligible=0, attempted=0, succeeded=0, errors=0, calls=0))

    @property
    def complete(self) -> bool:
        return self.counts["errors"] == 0 and self.counts["attempted"] == self.counts["succeeded"]

    def audit_record(self) -> dict:
        return {"mode": self.mode, "complete": self.complete, "counts": dict(self.counts),
                "rois": self.rois, "timings_ms": dict(self.timings),
                "unknown": {str(key): sorted(value) for key, value in self.unknown.items()}}


def crop_rect(bbox, width: int, height: int) -> tuple[int, int, int, int] | None:
    """10% per side; floor/ceil then clamp to actual pixels, never recenter."""
    if len(bbox) != 4 or not all(math.isfinite(float(value)) for value in bbox):
        return None
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return None
    rect = (max(0, min(width, math.floor(x1 - .1 * w))), max(0, min(height, math.floor(y1 - .1 * h))),
            max(0, min(width, math.ceil(x2 + .1 * w))), max(0, min(height, math.ceil(y2 + .1 * h))))
    return rect if rect[2] > rect[0] and rect[3] > rect[1] else None


def _detection(det: Detection) -> dict:
    return {"cls": det.cls, "cls_id": det.cls_id, "conf": float(det.conf), "bbox": list(det.bbox)}


def _add(target: dict, values: dict):
    for key, value in values.items():
        target[key] = target.get(key, 0) + value


def infer_ppe(frame: np.ndarray, people: list[Track], detector, ts: float,
              policy: PpePolicy = PpePolicy()) -> PpeFrameResult:
    """Sequential requests on the existing detector/scheduler; no absence inference.

    Successful owner ROIs contribute one model-run observation per timestamp.
    Failed/stale/invalid ROIs add no history vote and mask current attributes.
    Ambiguous/foreign boxes are never reassigned to another ROI's person.
    """
    result = PpeFrameResult(policy.mode)
    if policy.mode == "full_frame":
        call = {}
        result.counts.update(attempted=1, calls=1)
        try:
            kwargs = {"audit": call} if policy.audit else {}
            result.detections = detector.detect(frame, result.timings, **kwargs)
            result.counts["succeeded"] = 1
            result.rois.append({"status": "ok", "call": call})
        except Exception as exc:
            result.counts["errors"] = 1
            result.rois.append({"status": "error", "error": f"{type(exc).__name__}: {exc}"[:300], "call": call})
        return result

    for person in people:
        row = {"track_id": person.track_id,
               "original_bbox": [float(v) if math.isfinite(float(v)) else None for v in person.bbox],
               "actual_rect": None, "crop_detections": [], "global_detections": [], "observations": []}
        result.rois.append(row)
        if person.last_seen != ts:
            row["status"] = "stale"
            result.unknown.setdefault(person.track_id, set()).update(ATTRIBUTES)
            continue
        result.counts["eligible"] += 1
        rect = crop_rect(person.bbox, frame.shape[1], frame.shape[0])
        if rect is None:
            row["status"] = "invalid"
            result.counts["errors"] += 1
            result.unknown.setdefault(person.track_id, set()).update(ATTRIBUTES)
            continue
        row["actual_rect"] = list(rect)
        x0, y0, x1, y1 = rect
        call, timings = {}, {}
        row.update(call=call, timings_ms=timings)
        result.counts["attempted"] += 1
        result.counts["calls"] += 1
        started = time.perf_counter()
        try:
            local = detector.detect(frame[y0:y1, x0:x1], timings, audit=call, force_batch_one=True)
            if call and (not call.get("input", {}).get("finite", True)
                         or any(not value.get("finite", True) for value in call.get("outputs", []))):
                raise ValueError("Non-finite PPE tensor/output")
            if any(not all(math.isfinite(float(v)) for v in (*det.bbox, det.conf)) for det in local):
                raise ValueError("Non-finite decoded PPE observation")
            row["crop_detections"] = [_detection(det) for det in local]
            result.by_track[person.track_id] = []
            row["status"] = "ok"
            for det in local:
                name = ppe_attribute(det.cls)
                if name is None:
                    continue
                if not all(math.isfinite(float(value)) for value in (*det.bbox, det.conf)):
                    raise ValueError("Non-finite decoded PPE observation")
                box = det.bbox
                global_det = replace(det, bbox=(box[0] + x0, box[1] + y0, box[2] + x0, box[3] + y0))
                row["global_detections"].append(_detection(global_det))
                candidates = ppe_candidates(global_det, people)
                ids = [candidate[1] for candidate in candidates]
                accepted = len(candidates) == 1 and ids[0] == person.track_id
                reason = "accepted" if accepted else "ambiguous" if len(candidates) > 1 else "foreign" if ids else "incompatible"
                row["observations"].append({"attribute": name, "candidate_track_ids": ids, "decision": reason})
                if accepted:
                    result.by_track[person.track_id].append(global_det)
                else:
                    row["status"] = "rejected_observation"
                    result.unknown.setdefault(person.track_id, set()).add(name)
                    if len(candidates) > 1:
                        for track_id in ids:
                            result.unknown.setdefault(track_id, set()).add(name)
            result.counts["succeeded"] += 1
        except Exception as exc:
            row.update(status="error", error=f"{type(exc).__name__}: {exc}"[:300])
            result.counts["errors"] += 1
            result.by_track.pop(person.track_id, None)
            result.unknown.setdefault(person.track_id, set()).update(ATTRIBUTES)
        finally:
            row["wall_ms"] = (time.perf_counter() - started) * 1000
            _add(result.timings, timings)
    result.detections = [det for values in result.by_track.values() for det in values]
    return result


def attribute_coverage(people: list[Track], ts: float) -> dict:
    """Explicit person/attribute denominators, including unknown and stale tracks."""
    counts = {name: {"true": 0, "false": 0, "unknown": 0} for name in sorted(ATTRIBUTES)}
    for person in people:
        if person.last_seen == ts:
            for name in counts:
                value = person.attributes.get(name)
                counts[name]["true" if value is True else "false" if value is False else "unknown"] += 1
    return {"observed_person_samples": sum(person.last_seen == ts for person in people),
            "stale_person_samples": sum(person.last_seen != ts for person in people), "attributes": counts}
