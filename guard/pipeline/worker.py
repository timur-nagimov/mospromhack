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

from guard.contracts import TIMING_KEYS, CameraConfig, Detection, FrameContext, Track
from guard.pipeline import compat
from guard.pipeline.config import resolve_source_uri
from guard.rules.base import RulePlugin, get_rule
from guard.rules.geometry import is_zone_active
from guard.sources import FramePacket, make_source

if TYPE_CHECKING:
    from guard.pipeline.detector import Detector
    from guard.pipeline.engine import PipelineEngine

log = logging.getLogger(__name__)

PERSON_CLASSES = frozenset({"person"})
OBJECT_CLASSES = frozenset(
    {"car", "truck", "bus", "forklift", "motorcycle", "bicycle", "train", "tractor", "excavator", "loader", "crane"}
)


@dataclass(slots=True)
class AnalysisResult:
    seq: int
    ts: float
    frame: np.ndarray
    people: list[Track]
    objects: list[Track]
    highlight: set[int]
    active_zones: set[str]
    done_mono: float = field(default_factory=time.monotonic)


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
        self.person_tracker = compat.Tracker(classes=set(PERSON_CLASSES), **tracker_kwargs)
        self.object_tracker = compat.Tracker(classes=set(OBJECT_CLASSES), **tracker_kwargs)
        uri = cfg.source.uri
        if cfg.source.type in ("file", "images"):
            uri = resolve_source_uri(uri, st.data_dir)
        self.source = make_source(cfg.source, uri, cfg.camera_id, on_frame=self._on_frame, fps=cfg.sample_fps)
        self.last: AnalysisResult | None = None
        self.frame_idx = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._rule_fail_logged: set[str] = set()
        self._jpeg_cache: dict[str, tuple[Any, bytes]] = {}
        self._jpeg_lock = threading.Lock()

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

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive() and not self._stop.is_set()

    def _on_frame(self, pkt: FramePacket) -> None:
        """Поток источника: каждый кадр — в fps_in и кольцевой буфер клипов."""
        self.metrics.on_source_frame()
        self.engine.media.push_frame(self.cfg.camera_id, pkt.frame, pkt.ts)

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
        t_start = time.perf_counter()
        timings = {k: 0.0 for k in TIMING_KEYS}
        timings["decode"] = pkt.decode_ms
        frame, ts = pkt.frame, pkt.ts
        dets: dict[str, list[Detection]] = {}
        for model_id, det in self.detectors.items():
            try:
                dets[model_id] = det.detect(frame, timings)
                self.errors.pop(f"infer:{model_id}", None)
            except Exception as exc:  # noqa: BLE001
                dets[model_id] = []
                self.errors[f"infer:{model_id}"] = f"{type(exc).__name__}: {exc}"[:300]
                self.metrics.errors += 1
        t0 = time.perf_counter()
        if "det" in dets:
            person_dets = [d for d in dets["det"] if d.cls in PERSON_CLASSES]
        else:
            person_dets = list(dets.get("pose", []))
        people = self.person_tracker.update(person_dets, ts)
        objects = self.object_tracker.update([d for d in dets.get("det", []) if d.cls in OBJECT_CLASSES], ts)
        compat.assign_attributes(people, dets, ts)
        t1 = time.perf_counter()
        timings["tracking"] = (t1 - t0) * 1000.0
        ctx = FrameContext(
            camera=self.cfg, ts=ts, frame_idx=self.frame_idx, frame=frame,
            detections=dets, people=people, objects=objects, timings=timings,
        )
        violations = []
        for rule in self.rules:
            try:
                violations.extend(rule.evaluate(ctx))
            except Exception as exc:  # noqa: BLE001 — одно правило не роняет остальные
                key = f"rule:{rule.cfg.rule_id}"
                self.errors[key] = f"evaluate failed: {type(exc).__name__}: {exc}"[:300]
                if key not in self._rule_fail_logged:
                    self._rule_fail_logged.add(key)
                    log.exception("camera %s: rule %s failed", self.cfg.camera_id, rule.cfg.rule_id)
        t2 = time.perf_counter()
        timings["rules"] = (t2 - t1) * 1000.0
        new, closed = self.engine.events.process(self.cfg, ts, violations, dict(timings))
        if new or closed:
            self.engine.handle_events(self, ctx, new, closed, t_start)
        t3 = time.perf_counter()
        timings["alert"] = (t3 - t2) * 1000.0
        timings["total"] = pkt.decode_ms + (t3 - t_start) * 1000.0
        self.metrics.record_frame(timings, capture_ts=ts)
        highlight = {v.track_id for v in violations if v.track_id is not None}
        for ev in self._open_events():
            if ev.track_id is not None:
                highlight.add(ev.track_id)
        # снимки треков: трекер меняет свои объекты на следующих кадрах, а оверлей рисует этот кадр
        self.last = AnalysisResult(
            seq=pkt.seq, ts=ts, frame=frame,
            people=[dataclasses.replace(t, attributes=dict(t.attributes)) for t in people],
            objects=[dataclasses.replace(t, attributes=dict(t.attributes)) for t in objects],
            highlight=highlight,
            active_zones={z.zone_id for z in self.cfg.zones if is_zone_active(z, ts)},
        )
        self.frame_idx += 1
        return ctx

    def _open_events(self) -> list:
        fn = getattr(self.engine.events, "open_events", None)
        try:
            return list(fn(self.cfg.camera_id)) if fn else []
        except Exception:  # noqa: BLE001
            return []

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

        return self._cached_jpeg("raw", pkt.seq, render)[1]

    def overlay_jpeg(self, mode: str = "analyzed", max_width: int = 1280, quality: int = 80) -> tuple[Any, bytes] | None:
        """Кадр с оверлеями. analyzed — кадр анализа (рамки точно на месте);
        live — свежий кадр источника с последними рамками (плавнее, рамки чуть отстают)."""
        res = self.last
        raw = self.source.latest()
        fresh = res is not None and time.monotonic() - res.done_mono < 2.0
        if mode != "live" and fresh:
            frame, key = res.frame, ("a", res.seq)
        elif raw is not None:
            frame, key = raw.frame, ("r", raw.seq, res.seq if res else 0)
        else:
            return None
        people = res.people if fresh else []
        objects = res.objects if fresh else []
        highlight = res.highlight if fresh else set()

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
        return {
            "running": self.running,
            "source": self.source.status(),
            "models": self.model_ids,
            "rules": [r.cfg.rule_id for r in self.rules],
            "fps_in": snap["fps_in"],
            "fps_processed": snap["fps_processed"],
            "lag_s": snap["lag_s"],
            "errors": dict(self.errors),
        }
