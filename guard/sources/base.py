"""Базовый источник кадров: свой поток, слот «последний кадр» без очереди.

Потребитель (CameraWorker) берёт самый свежий кадр; старые кадры перезаписываются,
поэтому лаг не накапливается, даже если анализ медленнее источника.
Колбэк on_frame вызывается на КАЖДЫЙ кадр (кольцевой буфер клипов, fps_in).

Состояние источника (status()["state"], docs/STREAMS.md):
connecting — первое подключение, кадров ещё не было; online — кадры идут;
reconnecting — поток оборвался, идут повторные попытки; offline — нет сигнала
(попытки не удаются дольше OFFLINE_AFTER_ATTEMPTS раз подряд, файл закончился, источник остановлен);
error — ошибка конфигурации или адреса (файл не найден, ссылку не удалось разрешить).
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

import numpy as np

from guard.contracts import SourceConfig
from guard.sources.stream_url import mask_uri

log = logging.getLogger(__name__)

CONNECTING, ONLINE, RECONNECTING, OFFLINE, ERROR = "connecting", "online", "reconnecting", "offline", "error"
SOURCE_STATES = (CONNECTING, ONLINE, RECONNECTING, OFFLINE, ERROR)
FPS_WINDOW_S = 10.0  # окно расчёта fps_in источника


@dataclass(slots=True)
class FramePacket:
    frame: np.ndarray  # BGR
    ts: float  # epoch-время захвата
    seq: int  # номер кадра источника, растёт монотонно
    decode_ms: float  # время чтения и декодирования
    pos_s: float | None = None  # позиция в файле (для file)
    capture_id: str = field(default_factory=lambda: str(uuid4()))


FrameCallback = Callable[[FramePacket], None]


class FrameSource:
    kind = "base"

    def __init__(self, cfg: SourceConfig, uri: str, camera_id: str = "", on_frame: FrameCallback | None = None) -> None:
        self.cfg = cfg
        self.uri = uri
        self.camera_id = camera_id
        self.on_frame = on_frame
        self._cond = threading.Condition()
        self._latest: FramePacket | None = None
        self._seq = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.connected = False
        self.eof = False
        self.last_error: str | None = None
        self.errors = 0
        self.native_fps: float | None = None
        self.frame_size: tuple[int, int] | None = None  # (w, h)
        self.capture_id = str(uuid4())
        # состояние и счётчики подключения (status())
        self.state = CONNECTING
        self.codec: str | None = None
        self.reconnects = 0  # успешных переподключений после обрыва
        self.drops = 0  # обрывов уже работавшего потока
        self.attempts = 0  # неудачных попыток подключения подряд
        self.connect_ms: float | None = None  # от начала попытки до первого кадра (последнее подключение)
        self.last_outage_s: float | None = None  # длительность последнего обрыва: от потери до первого кадра
        self.retry_at: float | None = None  # monotonic следующей попытки, пока ждём паузу
        self.online_since: float | None = None  # epoch первого кадра текущего подключения
        self._frame_times: deque[float] = deque(maxlen=600)
        self._last_frame_mono: float | None = None

    def _new_capture(self) -> None:
        """A reconnect/file loop is a new capture, even if frame positions repeat."""
        self.capture_id = str(uuid4())

    # ---- жизненный цикл ----

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._safe_run, name=f"src-{self.camera_id or self.kind}", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 3.0) -> None:
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        self.retry_at = None
        if self.state != ERROR:
            self.state = OFFLINE

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive() and not self._stop.is_set()

    def _safe_run(self) -> None:
        try:
            self._run()
        except Exception as exc:  # noqa: BLE001 — поток источника не должен падать молча
            self._error(exc)
            self.state = ERROR
            log.exception("source %s crashed", self.camera_id)
        finally:
            self.connected = False

    def _run(self) -> None:  # pragma: no cover - реализуют наследники
        raise NotImplementedError

    def _sleep(self, seconds: float) -> bool:
        """Сон с ранним выходом при остановке. False — источник остановлен."""
        return not self._stop.wait(max(0.0, seconds))

    def _error(self, exc: BaseException | str) -> None:
        self.errors += 1
        self.last_error = mask_uri(str(exc))[:300]  # пароль и токены из адреса не попадают в API и логи

    # ---- выдача кадров ----

    def _publish(self, frame: np.ndarray, decode_ms: float, ts: float | None = None, pos_s: float | None = None) -> None:
        self._seq += 1
        pkt = FramePacket(frame=frame, ts=time.time() if ts is None else ts, seq=self._seq, decode_ms=decode_ms, pos_s=pos_s, capture_id=self.capture_id)
        self.frame_size = (frame.shape[1], frame.shape[0])
        now = time.monotonic()
        with self._cond:
            self._latest = pkt
            self._frame_times.append(now)
            self._last_frame_mono = now
            self._cond.notify_all()
        if self.on_frame is not None:
            try:
                self.on_frame(pkt)
            except Exception:  # noqa: BLE001
                log.exception("on_frame callback failed for %s", self.camera_id)

    def latest(self) -> FramePacket | None:
        with self._cond:
            return self._latest

    def wait_newer(self, seq: int, timeout: float) -> FramePacket | None:
        """Самый свежий кадр с номером > seq; None, если не дождались."""
        deadline = time.monotonic() + timeout
        with self._cond:
            while not self._stop.is_set():
                if self._latest is not None and self._latest.seq > seq:
                    return self._latest
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._cond.wait(remaining)
        return None

    # ---- состояние ----

    def fps_in(self, now: float | None = None) -> float:
        """Кадров в секунду на входе за последние FPS_WINDOW_S; поток встал — значение падает к нулю."""
        now = time.monotonic() if now is None else now
        with self._cond:
            times = [t for t in self._frame_times if now - t <= FPS_WINDOW_S]
        if len(times) < 2:
            return 0.0
        span = now - times[0]
        return (len(times) - 1) / span if span > 0 else 0.0

    def last_frame_age_s(self, now: float | None = None) -> float | None:
        if self._last_frame_mono is None:
            return None
        return max(0.0, (time.monotonic() if now is None else now) - self._last_frame_mono)

    def brief_status(self) -> dict[str, Any]:
        """Компактный статус для метрик (WS metrics каждые 2 с и транспорт edge)."""
        now = time.monotonic()
        w, h = self.frame_size or (None, None)
        age = self.last_frame_age_s(now)
        return {
            "state": self.state,
            "fps_in": round(self.fps_in(now), 2),
            "last_frame_age_s": None if age is None else round(age, 2),
            "width": w,
            "height": h,
            "reconnects": self.reconnects,
            "drops": self.drops,
            "last_error": self.last_error,
        }

    def status(self) -> dict[str, Any]:
        now = time.monotonic()
        retry_in = None if self.retry_at is None else round(max(0.0, self.retry_at - now), 2)
        return {
            "kind": self.kind,
            **self.brief_status(),
            "connected": self.connected,
            "eof": self.eof,
            "frames": self._seq,
            "errors": self.errors,
            "native_fps": self.native_fps,
            "frame_size": self.frame_size,
            "codec": self.codec,
            "attempts": self.attempts,
            "connect_ms": None if self.connect_ms is None else round(self.connect_ms, 1),
            "last_outage_s": None if self.last_outage_s is None else round(self.last_outage_s, 2),
            "next_retry_s": retry_in,
            "online_since": self.online_since,
        }
