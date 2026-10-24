"""Сборщик маховика в конвейере (docs/FLYWHEEL.md, звено «Сбор»).

Горячий путь кадра (handle_events, CameraWorker.process) только отбирает кадр и ставит задачу в
ограниченную очередь: ссылки на кадр и детекции, без копий и без I/O. Если очередь полна, сэмпл
отбрасывается и растёт счётчик dropped. JPEG, файл, строка SQLite и сообщение WS — в фоновом потоке
писателя (LocalWriter) или отправителя edge (guard/flywheel/edge.py).

Источники (source): event — каждое новое событие; uncertain — рамка целевого класса у порога решения;
flicker — смена ID трека человека рядом за ≤ 1 с; disagreement — расхождение TPU и CPU (shadow.py);
operator — ручной захват кадра камеры (capture).
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np

from guard.contracts import Detection, Event, FrameContext
from guard.flywheel.frames import TrackRow, detections_to_dicts, encode_jpeg, scale_box, snapshot_tracks
from guard.flywheel.holdout import holdout_names, is_holdout_uri
from guard.flywheel.select import FlickerState, find_uncertain
from guard.flywheel.store import OPERATOR_LABELS, UNCAPPED_SOURCES, FlywheelStore, frame_rel_path, new_sample_id
from guard.rules.geometry import LOCAL_TZ

log = logging.getLogger(__name__)

DEFAULT_PERIOD_S = 10.0
EDGE_PERIOD_S = 30.0
QUEUE_SIZE = 16


def sample_url(sample_id: str) -> str:
    return f"/api/flywheel/samples/{sample_id}/frame.jpg"


def short_sample(sample: Mapping[str, Any]) -> dict[str, Any]:
    """Короткая форма для WS: без детекций и меток, только счётчики и классы."""
    dets = sample.get("detections") or []
    return {
        "id": sample["id"],
        "created_at": sample["created_at"],
        "camera_id": sample["camera_id"],
        "source": sample["source"],
        "event_id": sample.get("event_id"),
        "status": sample.get("status", "new"),
        "operator_label": sample.get("operator_label"),
        "reason": sample.get("reason", ""),
        "holdout": int(sample.get("holdout") or 0),
        "width": sample.get("width"),
        "height": sample.get("height"),
        "n_detections": len(dets),
        "classes": sorted({d.get("cls") for d in dets if d.get("cls")}),
        "frame_url": sample_url(sample["id"]),
    }


def sample_ws_message(sample: Mapping[str, Any]) -> dict[str, Any]:
    """Сообщение шины WS. kind — как у остальных сообщений шины, type — как в контракте."""
    return {"kind": "flywheel_sample", "type": "flywheel_sample", "sample": short_sample(sample)}


def _scale_meta(meta: Mapping[str, Any], scale: float) -> dict[str, Any]:
    """Рамки в meta (ключи *bbox) — в пиксели сохранённого кадра."""
    out = dict(meta)
    if scale != 1.0:
        for key, value in meta.items():
            if key.endswith("bbox") and isinstance(value, (list, tuple)) and len(value) == 4:
                out[key] = scale_box(value, scale)
    return out


@dataclass
class SampleJob:
    """Задача записи: поля сэмпла без кадра; кадр — сырой (кодирует фон) или готовый JPEG (шлюз)."""

    sample: dict[str, Any]
    frame: np.ndarray | None = None
    detections: Mapping[str, list[Detection]] | None = None
    tracks: list[TrackRow] = field(default_factory=list)
    jpeg: bytes | None = None
    det_dicts: list[dict[str, Any]] | None = None


class LocalWriter:
    """Фоновый писатель (режимы local и gateway): JPEG → frames/, строка → flywheel.db, сообщение → WS."""

    def __init__(
        self,
        store: FlywheelStore,
        *,
        publish: Callable[[dict[str, Any]], None] | None = None,
        label_lookup: Callable[[str], str | None] | None = None,
        max_width: int = 1280,
        quality: int = 90,
        queue_size: int = QUEUE_SIZE,
    ) -> None:
        self.store = store
        self.publish = publish
        self.label_lookup = label_lookup
        self.max_width = max_width
        self.quality = quality
        self._q: queue.Queue[SampleJob | None] = queue.Queue(maxsize=max(1, queue_size))
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.written = 0
        self.dropped = 0
        self.errors = 0
        self.write_ms: list[float] = []

    # ---- жизненный цикл ----

    def start(self) -> None:
        if self._thread is None or not self._thread.is_alive():
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, name="flywheel-writer", daemon=True)
            self._thread.start()

    def stop(self, timeout: float = 3.0) -> None:
        """Дописать очередь (не дольше timeout) и остановиться."""
        thread = self._thread
        if thread is None:
            return
        self._stop.set()
        try:
            self._q.put_nowait(None)
        except queue.Full:
            pass
        thread.join(timeout)
        self._thread = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def put(self, job: SampleJob) -> bool:
        """Без ожидания: очередь полна — сэмпл отброшен (счётчик dropped)."""
        try:
            self._q.put_nowait(job)
            return True
        except queue.Full:
            self.dropped += 1
            return False

    def _run(self) -> None:
        while True:
            try:
                job = self._q.get(timeout=0.5)
            except queue.Empty:
                if self._stop.is_set():
                    return
                continue
            if job is None:
                if self._stop.is_set() and self._q.empty():
                    return
                continue
            try:
                self.write(job)
            except Exception:  # noqa: BLE001 — сбой записи сэмпла не влияет на конвейер
                self.errors += 1
                if self.errors in (1, 10, 100) or self.errors % 1000 == 0:
                    log.exception("flywheel: sample write failed (%d errors)", self.errors)

    # ---- запись ----

    def _operator_label(self, event_id: str | None) -> str | None:
        if not event_id or self.label_lookup is None:
            return None
        try:
            return OPERATOR_LABELS.get(self.label_lookup(event_id) or "")
        except Exception:  # noqa: BLE001
            return None

    def write(self, job: SampleJob) -> dict[str, Any]:
        """Синхронная запись сэмпла (фоновый поток или ручной захват из API)."""
        t0 = time.perf_counter()
        sample = dict(job.sample)
        if job.jpeg is None:
            if job.frame is None:
                raise ValueError("sample job has neither frame nor jpeg")
            jpeg, width, height, scale = encode_jpeg(job.frame, self.max_width, self.quality)
            sample["width"], sample["height"] = width, height
            sample["detections"] = detections_to_dicts(job.detections or {}, job.tracks, scale)
            sample["meta"] = _scale_meta(sample.get("meta") or {}, scale)
        else:
            jpeg = job.jpeg
            sample["detections"] = list(job.det_dicts or sample.get("detections") or [])
        sample.setdefault("id", new_sample_id())
        rel = frame_rel_path(sample["camera_id"], sample["id"])
        sample["frame_path"] = rel
        if sample.get("source") == "event" and not sample.get("operator_label"):
            sample["operator_label"] = self._operator_label(sample.get("event_id"))
        self.store.write_frame(rel, jpeg)
        stored = self.store.add(sample)
        if stored.get("source") == "event" and stored.get("event_id"):
            # решение оператора могло прийти между проверкой и вставкой строки
            label = self._operator_label(stored["event_id"])
            if label and label != stored.get("operator_label"):
                self.store.set_operator_label(stored["event_id"], label)
                stored = self.store.get(stored["id"]) or stored
        self.written += 1
        self.write_ms.append((time.perf_counter() - t0) * 1000.0)
        if len(self.write_ms) > 200:
            del self.write_ms[:100]
        if self.publish is not None:
            try:
                self.publish(sample_ws_message(stored))
            except Exception:  # noqa: BLE001
                log.exception("flywheel: WS publish failed")
        return stored

    def stats(self) -> dict[str, Any]:
        ms = sorted(self.write_ms)
        return {
            "written": self.written,
            "dropped": self.dropped,
            "errors": self.errors,
            "queue": self._q.qsize(),
            "write_ms_p50": round(ms[len(ms) // 2], 1) if ms else None,
        }


@dataclass
class _CameraState:
    camera_id: str
    uri: str
    holdout: int
    day: str = ""
    capped: int = 0
    last_uncertain: float = -1e18
    last_flicker: float = -1e18
    last_shadow: float = -1e18
    flicker: FlickerState = field(default_factory=FlickerState)
    last_frame: np.ndarray | None = None
    last_detections: Mapping[str, list[Detection]] | None = None
    last_ts: float = 0.0


def _day_key(ts: float) -> str:
    return datetime.fromtimestamp(ts, LOCAL_TZ).strftime("%Y-%m-%d")


class FlywheelCollector:
    """Отбор кадров камеры в выборку. Методы горячего пути не бросают исключений и не делают I/O."""

    def __init__(
        self,
        output: Any,
        *,
        period_s: float = DEFAULT_PERIOD_S,
        daily_cap: int = 500,
        tracker_high: float = 0.4,
        enabled: bool = True,
        holdout: frozenset[str] | None = None,
        model_thresholds: Mapping[str, float] | None = None,
        shadow: Any | None = None,
        shadow_period_s: float = 30.0,
    ) -> None:
        self.output = output
        self.enabled = enabled
        self.period_s = float(period_s)
        self.daily_cap = int(daily_cap)
        self.tracker_high = float(tracker_high)
        # имена отложенных роликов читаются здесь, не в горячем пути
        self.holdout = holdout if holdout is not None else holdout_names()
        self.model_thresholds = dict(model_thresholds or {})
        self.shadow = shadow
        self.shadow_period_s = float(shadow_period_s)
        self._cams: dict[str, _CameraState] = {}
        self._lock = threading.Lock()
        self.submitted: dict[str, int] = defaultdict(int)
        self.dropped: dict[str, int] = defaultdict(int)  # queue — очередь полна, cap — суточный лимит
        self.errors = 0
        self._caps_today: dict[str, int] = {}
        self._caps_day = ""
        self.started = False

    # ---- жизненный цикл ----

    def start(self, store: FlywheelStore | None = None) -> None:
        if store is not None:
            try:
                self._caps_today = store.capped_today()
                self._caps_day = _day_key(time.time())
            except Exception:  # noqa: BLE001
                log.exception("flywheel: cannot read today's counters")
        start = getattr(self.output, "start", None)
        if callable(start):
            start()
        if self.shadow is not None:
            self.shadow.start()
        self.started = True

    def stop(self, timeout: float = 3.0) -> None:
        if self.shadow is not None:
            self.shadow.stop()
        stop = getattr(self.output, "stop", None)
        if callable(stop):
            stop(timeout)
        self.started = False

    @property
    def collecting(self) -> bool:
        return self.enabled and self.started

    # ---- состояние камеры ----

    def _state(self, cfg: Any) -> _CameraState:
        camera_id = cfg.camera_id
        uri = getattr(getattr(cfg, "source", None), "uri", "") or ""
        st = self._cams.get(camera_id)
        if st is None or st.uri != uri:
            st = _CameraState(camera_id=camera_id, uri=uri, holdout=int(is_holdout_uri(uri, self.holdout)))
            if self._caps_day:
                st.day = self._caps_day
                st.capped = self._caps_today.get(camera_id, 0)
            with self._lock:
                self._cams[camera_id] = st
        return st

    def _cap_ok(self, st: _CameraState, ts: float) -> bool:
        day = _day_key(ts)
        if day != st.day:
            st.day, st.capped = day, 0
        if st.capped >= self.daily_cap:
            self.dropped["cap"] += 1
            return False
        return True

    def is_holdout(self, cfg: Any) -> bool:
        return bool(self._state(cfg).holdout)

    # ---- постановка в очередь ----

    def _submit(
        self,
        st: _CameraState,
        source: str,
        ctx: FrameContext,
        reason: str,
        *,
        event_id: str | None = None,
        meta: dict[str, Any] | None = None,
        detections: Mapping[str, list[Detection]] | None = None,
    ) -> bool:
        sample = {
            "id": new_sample_id(),
            "created_at": float(ctx.ts),
            "camera_id": st.camera_id,
            "source": source,
            "event_id": event_id,
            "reason": reason,
            "holdout": st.holdout,
            "meta": meta or {},
        }
        dets = detections if detections is not None else ctx.detections
        job = SampleJob(
            sample=sample,
            frame=ctx.frame,
            detections={m: list(d) for m, d in dets.items()},
            tracks=snapshot_tracks(ctx.people, ctx.objects),
        )
        if not self.output.put(job):
            self.dropped["queue"] += 1
            return False
        self.submitted[source] += 1
        if source not in UNCAPPED_SOURCES:
            st.capped += 1
        return True

    def on_event(self, ctx: FrameContext, event: Event) -> None:
        """Новое событие → сэмпл event: сырой кадр подтверждения и детекции всех моделей кадра."""
        if not self.enabled:
            return
        try:
            st = self._state(ctx.camera)
            meta = {
                "event_type": event.type,
                "rule_id": event.rule_id,
                "zone_id": event.zone_id,
                "track_id": event.track_id,
                "event_bbox": [float(v) for v in event.bbox],
                "camera_name": ctx.camera.name,
            }
            reason = f"событие {event.type}" + (f", трек #{event.track_id}" if event.track_id is not None else "")
            self._submit(st, "event", ctx, reason, event_id=event.id, meta=meta)
        except Exception:  # noqa: BLE001 — сбор не мешает оповещению
            self.errors += 1
            if self.errors in (1, 10, 100):
                log.exception("flywheel: on_event failed")

    def observe(self, worker: Any, ctx: FrameContext) -> None:
        """Каждый обработанный кадр камеры: мигание треков, неуверенные рамки, выборочная сверка TPU и CPU."""
        if not self.enabled:
            return
        try:
            st = self._state(worker.cfg)
            st.last_frame, st.last_detections, st.last_ts = ctx.frame, ctx.detections, ctx.ts
            ts = ctx.ts
            # период считается от попытки: и при отказе по лимиту или по очереди следующая — через period_s
            hit = st.flicker.update(((t.track_id, t.bbox) for t in ctx.people), ts)
            if hit is not None and ts - st.last_flicker >= self.period_s:
                st.last_flicker = ts
                if self._cap_ok(st, ts):
                    meta = {"old_track_id": hit.old_id, "new_track_id": hit.new_id, "gap_s": round(hit.gap_s, 2),
                            "old_bbox": list(hit.old_bbox), "focus_bbox": list(hit.new_bbox)}
                    self._submit(st, "flicker", ctx, hit.reason, meta=meta)
            if ts - st.last_uncertain >= self.period_s:
                found = find_uncertain(ctx.detections, getattr(worker, "rules", []), self._thresholds(worker),
                                       self.tracker_high)
                if found is not None:
                    st.last_uncertain = ts
                    if self._cap_ok(st, ts):
                        meta = {"model_id": found.model_id, "cls": found.cls, "conf": round(found.conf, 4),
                                "band": [found.lo, found.hi], "rule_id": found.rule_id,
                                "focus_bbox": list(found.bbox)}
                        self._submit(st, "uncertain", ctx, found.reason, meta=meta)
            if self.shadow is not None and ts - st.last_shadow >= self.shadow_period_s:
                st.last_shadow = ts
                self.shadow.submit(self, worker, ctx)
        except Exception:  # noqa: BLE001
            self.errors += 1
            if self.errors in (1, 10, 100):
                log.exception("flywheel: observe failed")

    def submit_disagreement(self, worker: Any, ctx: FrameContext, reason: str, meta: dict[str, Any],
                            detections: Mapping[str, list[Detection]]) -> bool:
        """Вызывает сверка TPU и CPU (свой поток): сэмпл disagreement с детекциями обеих сторон."""
        st = self._state(worker.cfg)
        if not self._cap_ok(st, ctx.ts):
            return False
        return self._submit(st, "disagreement", ctx, reason, meta=meta, detections=detections)

    def _thresholds(self, worker: Any) -> dict[str, float]:
        out = dict(self.model_thresholds)
        for model_id, det in (getattr(worker, "detectors", None) or {}).items():
            spec = getattr(det, "spec", None)
            if spec is not None:
                out[model_id] = float(spec.conf_threshold)
        return out

    # ---- ручной захват (API) ----

    def capture(self, camera_id: str, writer: LocalWriter, camera_name: str | None = None) -> dict[str, Any] | None:
        """Текущий кадр камеры → сэмпл operator (синхронно, из потока API). None — кадров ещё не было."""
        st = self._cams.get(camera_id)
        if st is None or st.last_frame is None:
            return None
        sample = {
            "id": new_sample_id(),
            "created_at": float(st.last_ts),
            "camera_id": camera_id,
            "source": "operator",
            "reason": "оператор отправил кадр в выборку",
            "holdout": st.holdout,
            "meta": {"camera_name": camera_name} if camera_name else {},
        }
        dets = st.last_detections or {}
        job = SampleJob(sample=sample, frame=st.last_frame, detections={m: list(d) for m, d in dets.items()})
        stored = writer.write(job)
        self.submitted["operator"] += 1
        return stored

    # ---- сводка ----

    def stats(self) -> dict[str, Any]:
        out_stats = getattr(self.output, "stats", None)
        output = out_stats() if callable(out_stats) else {}
        dropped = dict(self.dropped)
        if isinstance(output, dict):
            for key in ("dropped_budget", "dropped_channel"):
                if output.get(key):
                    dropped[key.removeprefix("dropped_")] = output[key]
        return {
            "enabled": self.enabled,
            "collecting": self.collecting,
            "period_s": self.period_s,
            "daily_cap": self.daily_cap,
            "submitted": dict(self.submitted),
            "dropped": dropped,
            "dropped_total": int(sum(dropped.values())),
            "errors": self.errors,
            "cameras": {cid: {"holdout": st.holdout, "capped_today": st.capped} for cid, st in self._cams.items()},
            "output": output,
            "shadow": self.shadow.stats() if self.shadow is not None and hasattr(self.shadow, "stats") else None,
        }
