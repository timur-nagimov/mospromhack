"""Обработка одной камеры: выборка с sample_fps → модели → треки → правила → EventManager."""

from __future__ import annotations

import dataclasses
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from guard.contracts import TIMING_KEYS, CameraConfig, Detection, Event, FrameContext, Track
from guard.pipeline import compat
from guard.pipeline.config import resolve_source_uri
from guard.pipeline.ppe import PpePolicy, infer_ppe
from guard.rules.base import RulePlugin, get_rule
from guard.rules.geometry import is_zone_active
from guard.sources import FramePacket, make_source

if TYPE_CHECKING:
    from guard.pipeline.detector import Detector
    from guard.pipeline.engine import PipelineEngine

log = logging.getLogger(__name__)

PERSON_CLASSES = frozenset({"person"})
# Короткие метки нарушений для оверлея (латиница: cv2.putText без кириллицы)
SHORT_LABELS = {
    "zone_intrusion": "ZONE",
    "ppe_missing": "PPE",
    "fall_immobile": "FALL",
    "vehicle_proximity": "VEHICLE",
    "running": "RUN",
    "phone_usage": "PHONE",
    "smoking_fire": "FIRE",
    "fence_climb": "FENCE",
    "height_no_harness": "HEIGHT",
    "unsafe_carry": "CARRY",
}
OBJECT_CLASSES = frozenset(
    {"car", "truck", "bus", "forklift", "motorcycle", "bicycle", "train", "tractor", "excavator", "loader", "crane"}
)


def object_detections(dets: dict[str, list[Detection]]) -> list[Detection]:
    """Детекции техники для трекера объектов: классы OBJECT_CLASSES из всех моделей кадра.

    Грузовики и автобусы даёт det, погрузчик — отдельная модель (model_id forklift), её подключает
    правило vehicle_proximity параметром models. Модели перебираются по model_id, поэтому порядок не зависит
    от порядка правил. На камере только с det результат прежний.
    """
    return [d for model_id in sorted(dets) for d in dets[model_id] if d.cls in OBJECT_CLASSES]


@dataclass(slots=True)
class AnalysisResult:
    seq: int
    ts: float
    frame: np.ndarray
    people: list[Track]
    objects: list[Track]
    highlight: dict[int, str]
    active_zones: set[str]
    done_mono: float = field(default_factory=time.monotonic)
    capture_id: str = ""
    source_pos_s: float | None = None


def _scaled_track(t: Track, s: float) -> Track:
    kp = None
    if t.keypoints is not None:
        kp = np.asarray(t.keypoints, dtype=np.float32).copy()
        kp[:, :2] *= s
    return dataclasses.replace(t, bbox=tuple(v * s for v in t.bbox), keypoints=kp)


class CameraWorker:
    def __init__(self, cfg: CameraConfig, engine: PipelineEngine) -> None:
        self.cfg = cfg
        self.engine = engine
        st = engine.settings
        self.metrics = engine.metrics.reset(cfg.camera_id)
        self.metrics.set_period(cfg.sample_fps)
        self.errors: dict[str, str] = {}
        self.rules: list[RulePlugin] = []
        self.detectors: dict[str, Detector] = {}
        self._build()
        tracker_kwargs: dict[str, Any] = dict(
            high_thresh=st.tracker_high_thresh,
            low_thresh=st.tracker_low_thresh,
            match_iou=st.tracker_match_iou,
            new_track_thresh=st.tracker_new_track_thresh,
            track_buffer_s=st.tracker_buffer_s,
            min_hits=st.tracker_min_hits,
        )
        self._tracker_kwargs = tracker_kwargs
        self.person_tracker = compat.Tracker(classes=set(PERSON_CLASSES), **tracker_kwargs)
        self.object_tracker = compat.Tracker(classes=set(OBJECT_CLASSES), **tracker_kwargs)
        uri = cfg.source.uri
        if cfg.source.type in ("file", "images"):
            uri = resolve_source_uri(uri, st.data_dir)
        self.source = make_source(cfg.source, uri, cfg.camera_id, on_frame=self._on_frame, fps=cfg.sample_fps)
        self.last: AnalysisResult | None = None
        self.last_context: FrameContext | None = None
        self.last_ppe = None
        self.frame_idx = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._rule_fail_logged: set[str] = set()
        self._jpeg_cache: dict[str, tuple[Any, bytes]] = {}
        self._jpeg_lock = threading.Lock()
        self._capture_id = ""
        self._pending_capture_closures: list[Event] | None = None
        self._pending_capture_id = ""

    # ---- сборка правил и моделей ----

    def _build(self) -> None:
        needed: dict[str, list[RulePlugin]] = {}
        for rcfg in self.cfg.rules:
            if not rcfg.enabled:
                continue
            try:
                rule = get_rule(rcfg.type)(rcfg, self.cfg)
            except KeyError:
                self.errors[f"rule:{rcfg.rule_id}"] = f"unknown rule type {rcfg.type!r}"
                continue
            except Exception as exc:  # noqa: BLE001
                self.errors[f"rule:{rcfg.rule_id}"] = f"init failed: {exc}"
                continue
            for model_id in rule.required_models:
                needed.setdefault(model_id, []).append(rule)
            self.rules.append(rule)
        unavailable: set[int] = set()
        for model_id, rules in needed.items():
            try:
                self.detectors[model_id] = self.engine.get_detector(model_id)
            except Exception as exc:  # noqa: BLE001 — нет модели: правила без неё отключаются
                self.errors[f"model:{model_id}"] = str(exc)
                unavailable.update(id(r) for r in rules)
        if unavailable:
            for r in self.rules:
                if id(r) in unavailable:
                    self.errors[f"rule:{r.cfg.rule_id}"] = "disabled: required model unavailable"
            self.rules = [r for r in self.rules if id(r) not in unavailable]

    @property
    def model_ids(self) -> list[str]:
        return sorted(self.detectors)

    # ---- жизненный цикл ----

    def start(self) -> None:
        self.source.start()
        self._thread = threading.Thread(target=self._run, name=f"cam-{self.cfg.camera_id}", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        self.source.stop(timeout=timeout)
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                raise RuntimeError("Camera worker did not stop")
        if self.capture_boundary_pending:
            self._finish_capture_boundary()

    @property
    def capture_boundary_pending(self) -> bool:
        return self._pending_capture_closures is not None

    def _finish_capture_boundary(self, capture_id: str = "") -> None:
        pending = self._pending_capture_closures
        if pending is None:
            return
        while pending:
            # A successfully announced closure is removed before another write
            # can fail, so partial batch recovery does not repeat its update.
            self.engine.close_capture(self, [pending[0]])
            pending.pop(0)
        rules = [type(rule)(rule.cfg, self.cfg) for rule in self.rules]
        people = compat.Tracker(classes=set(PERSON_CLASSES), **self._tracker_kwargs)
        objects = compat.Tracker(classes=set(OBJECT_CLASSES), **self._tracker_kwargs)
        self.engine.events.reset(self.cfg.camera_id)
        self.rules, self.person_tracker, self.object_tracker = rules, people, objects
        self.last = None
        self.last_context = None
        with self._jpeg_lock:
            self._jpeg_cache.clear()
        self._rule_fail_logged.clear()
        self._capture_id = capture_id or self._pending_capture_id
        self._pending_capture_closures = None
        self._pending_capture_id = ""
        self.errors.pop("capture", None)

    def _advance_capture(self, pkt: FramePacket) -> None:
        if self.capture_boundary_pending:
            self._finish_capture_boundary(pkt.capture_id)
        elif pkt.capture_id and self._capture_id and pkt.capture_id != self._capture_id:
            self._pending_capture_id = pkt.capture_id
            self._pending_capture_closures = self.engine.events.capture_closures(self.cfg.camera_id)
            self._finish_capture_boundary(pkt.capture_id)
        elif pkt.capture_id and not self._capture_id:
            self._capture_id = pkt.capture_id  # First known identity is bootstrap.

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive() and not self._stop.is_set()

    def _on_frame(self, pkt: FramePacket) -> None:
        """Поток источника: каждый кадр — в fps_in и кольцевой буфер клипов."""
        self.metrics.on_source_frame()
        if self.engine.settings.clips_enabled:
            self.engine.media.push_frame(self.cfg.camera_id, pkt.frame, pkt.ts, capture_id=pkt.capture_id)

    def _run(self) -> None:
        period = 1.0 / max(self.cfg.sample_fps, 0.1)
        last_seq = 0
        next_t = time.monotonic()
        while not self._stop.is_set():
            pkt = self.source.wait_newer(last_seq, timeout=1.0)
            if pkt is None:
                continue
            last_seq = pkt.seq
            try:
                self.process(pkt)
                self.errors.pop("process", None)
            except Exception as exc:  # noqa: BLE001 — камера продолжает работу
                self.metrics.errors += 1
                self.errors["process"] = f"{type(exc).__name__}: {exc}"[:300]
                log.exception("camera %s: frame processing failed", self.cfg.camera_id)
            next_t += period
            now = time.monotonic()
            if next_t <= now:
                next_t = now  # не копим долг: отстали — берём следующий свежий кадр сразу
            elif self._stop.wait(next_t - now):
                break

    # ---- обработка кадра ----

    def process(self, pkt: FramePacket) -> FrameContext:
        try:
            self._advance_capture(pkt)
        except Exception as exc:  # noqa: BLE001 — keep the frozen batch for the next frame/stop retry
            self.errors["capture"] = f"retryable boundary: {type(exc).__name__}: {exc}"[:300]
            raise
        t_start = time.perf_counter()
        timings = {k: 0.0 for k in TIMING_KEYS}
        timings["decode"] = pkt.decode_ms
        frame, ts = pkt.frame, pkt.ts
        ppe_mode = getattr(self.engine.settings, "ppe_input_mode", "full_frame")
        self.last_ppe = None
        dets: dict[str, list[Detection]] = {}
        model_observations: dict[str, str] = {}
        for model_id, det in self.detectors.items():
            if model_id == "ppe" and ppe_mode == "person_crop":
                continue
            try:
                if model_id == "ppe":
                    self.last_ppe = infer_ppe(frame, [], det, ts)
                    dets[model_id] = self.last_ppe.detections
                    for key, value in self.last_ppe.timings.items():
                        timings[key] = timings.get(key, 0) + value
                    if not self.last_ppe.complete:
                        raise RuntimeError(self.last_ppe.rois[0]["error"])
                else:
                    dets[model_id] = det.detect(frame, timings)
                model_observations[model_id] = "observed"
                self.errors.pop(f"infer:{model_id}", None)
            except Exception as exc:  # noqa: BLE001
                dets[model_id] = []
                model_observations[model_id] = "error"
                self.errors[f"infer:{model_id}"] = f"{type(exc).__name__}: {exc}"[:300]
                self.metrics.errors += 1
        t0 = time.perf_counter()
        if "det" in dets:
            person_dets = [d for d in dets["det"] if d.cls in PERSON_CLASSES]
        else:
            person_dets = list(dets.get("pose", []))
        people = self.person_tracker.update(person_dets, ts)
        objects = self.object_tracker.update(object_detections(dets), ts)
        tracking_ms = (time.perf_counter() - t0) * 1000
        if ppe_mode == "person_crop" and "ppe" in self.detectors:
            self.last_ppe = infer_ppe(frame, people, self.detectors["ppe"], ts, PpePolicy(ppe_mode))
            dets["ppe"] = self.last_ppe.detections
            for key, value in self.last_ppe.timings.items():
                timings[key] = timings.get(key, 0) + value
            if self.last_ppe.complete:
                model_observations["ppe"] = "observed" if self.last_ppe.counts["eligible"] else "not_observable"
                self.errors.pop("infer:ppe", None)
            else:
                model_observations["ppe"] = "error"
                self.errors["infer:ppe"] = "incomplete person PPE crops; affected attributes unknown"
                self.metrics.errors += self.last_ppe.counts["errors"]
        attribute_started = time.perf_counter()
        if ppe_mode == "person_crop" and self.last_ppe is not None:
            compat.assign_attributes(people, dets, ts, ppe_by_track=self.last_ppe.by_track,
                                     ppe_unknown=self.last_ppe.unknown)
        else:
            compat.assign_attributes(people, dets, ts)
        t1 = time.perf_counter()
        timings["tracking"] = tracking_ms + (t1 - attribute_started) * 1000.0
        ctx = FrameContext(
            camera=self.cfg, ts=ts, frame_idx=self.frame_idx, frame=frame,
            detections=dets, people=people, objects=objects, timings=timings,
            model_observations=model_observations,
        )
        violations = []
        for rule in self.rules:
            try:
                violations.extend(rule.evaluate(ctx))
            except Exception as exc:  # noqa: BLE001 — одно правило не роняет остальные
                key = f"rule:{rule.cfg.rule_id}"
                ctx.rule_observations[rule.cfg.rule_id] = "error"
                self.errors[key] = f"evaluate failed: {type(exc).__name__}: {exc}"[:300]
                if key not in self._rule_fail_logged:
                    self._rule_fail_logged.add(key)
                    log.exception("camera %s: rule %s failed", self.cfg.camera_id, rule.cfg.rule_id)
        t2 = time.perf_counter()
        timings["rules"] = (t2 - t1) * 1000.0
        process_updates = getattr(self.engine.events, "process_with_updates", None)
        if process_updates is None:  # compatibility with the minimal fallback manager
            new, closed = self.engine.events.process(self.cfg, ts, violations, dict(timings))
            reopened = []
        else:
            new, closed, reopened = process_updates(self.cfg, ts, violations, dict(timings))
        for event in new:
            event.details = {**event.details, "origin": {
                "edge_id": self.engine.settings.edge_id,
                "run_id": self.engine.run_id,
                "capture_id": pkt.capture_id, "frame_seq": pkt.seq,
                "capture_ts": pkt.ts, "source_pos_s": pkt.pos_s,
            }}
        if new or closed or reopened:
            self.engine.handle_events(self, ctx, new, closed, t_start, reopened=reopened)
        t3 = time.perf_counter()
        timings["alert"] = (t3 - t2) * 1000.0
        timings["total"] = pkt.decode_ms + (t3 - t_start) * 1000.0
        self.metrics.record_frame(timings, capture_ts=ts)
        highlight = {
            int(v.track_id): SHORT_LABELS.get(v.type, v.type[:8].upper())
            for v in violations
            if v.track_id is not None
        }
        # снимки треков: трекер меняет свои объекты на следующих кадрах, а оверлей рисует этот кадр
        self.last = AnalysisResult(
            seq=pkt.seq, ts=ts, frame=frame,
            capture_id=pkt.capture_id, source_pos_s=pkt.pos_s,
            people=[dataclasses.replace(t, attributes=dict(t.attributes)) for t in people],
            objects=[dataclasses.replace(t, attributes=dict(t.attributes)) for t in objects],
            highlight=highlight,
            active_zones={z.zone_id for z in self.cfg.zones if is_zone_active(z, ts)},
        )
        self.last_context = ctx
        self.frame_idx += 1
        self.engine.on_analysis(self, pkt, self.last)
        flywheel = getattr(self.engine, "flywheel", None)
        if flywheel is not None:  # маховик: мигание треков и неуверенные рамки; запись — в фоне
            flywheel.observe(self, ctx)
        return ctx

    # ---- картинки для API ----

    def _cached_jpeg(self, slot: str, key: Any, render) -> tuple[Any, bytes]:
        with self._jpeg_lock:
            cached = self._jpeg_cache.get(slot)
            if cached is not None and cached[0] == key:
                return cached
        data = render()
        with self._jpeg_lock:
            self._jpeg_cache[slot] = (key, data)
        return key, data

    def frame_jpeg(self, quality: int = 85) -> bytes | None:
        pkt = self.source.latest()
        if pkt is None:
            return None

        def render() -> bytes:
            return cv2.imencode(".jpg", pkt.frame, [cv2.IMWRITE_JPEG_QUALITY, quality])[1].tobytes()

        return self._cached_jpeg("raw", (pkt.capture_id, pkt.seq), render)[1]

    def overlay_jpeg(self, mode: str = "analyzed", max_width: int = 1280, quality: int = 80) -> tuple[Any, bytes] | None:
        """Кадр с оверлеями. analyzed — кадр анализа (рамки точно на месте);
        live — свежий кадр источника с последними рамками (плавнее, рамки чуть отстают)."""
        res = self.last
        raw = self.source.latest()
        same_capture = not (res is not None and raw is not None and res.capture_id and raw.capture_id
                            and res.capture_id != raw.capture_id)
        fresh = res is not None and same_capture and time.monotonic() - res.done_mono < 2.0
        if mode != "live" and fresh:
            frame, key = res.frame, ("a", res.capture_id, res.seq)
        elif raw is not None:
            frame, key = raw.frame, ("r", raw.capture_id, raw.seq, res.seq if res else 0)
        else:
            return None
        people = res.people if fresh else []
        objects = res.objects if fresh else []
        highlight = res.highlight if fresh else {}

        def render() -> bytes:
            h, w = frame.shape[:2]
            s = min(1.0, max_width / w)
            img = cv2.resize(frame, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA) if s < 1.0 else frame.copy()
            ppl = [_scaled_track(t, s) for t in people] if s < 1.0 else people
            objs = [_scaled_track(t, s) for t in objects] if s < 1.0 else objects
            img = compat.safe_draw_overlay(img, ppl, objs, self.cfg.zones, highlight)
            self._status_bar(img)
            return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])[1].tobytes()

        return self._cached_jpeg(f"overlay-{mode}", key, render)

    def _status_bar(self, img: np.ndarray) -> None:
        snap = self.metrics.snapshot()
        text = (
            f"{self.cfg.camera_id}  in {snap['fps_in']:.1f} fps  proc {snap['fps_processed']:.1f} fps  "
            f"lag {snap['lag_s'] * 1000:.0f} ms  total p50 {snap['timings_ms']['total']['p50']:.0f} ms"
        )
        scale = max(0.45, img.shape[1] / 1800)
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        cv2.rectangle(img, (0, img.shape[0] - th - base - 8), (tw + 12, img.shape[0]), (0, 0, 0), -1)
        cv2.putText(img, text, (6, img.shape[0] - base - 4), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)

    # ---- состояние ----

    def status(self) -> dict[str, Any]:
        snap = self.metrics.snapshot()
        result = {
            "running": self.running,
            "source": self.source.status(),
            "models": self.model_ids,
            "rules": [r.cfg.rule_id for r in self.rules],
            "capture_id": self._capture_id,
            "capture_boundary_pending": self.capture_boundary_pending,
            "fps_in": snap["fps_in"],
            "fps_processed": snap["fps_processed"],
            "lag_s": snap["lag_s"],
            "errors": dict(self.errors),
        }
        context = self.last_context
        if context is not None:
            result["model_observations"] = dict(context.model_observations)
            result["rule_observations"] = dict(context.rule_observations)
        return result
