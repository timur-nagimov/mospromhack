"""Базовый источник кадров: свой поток, слот «последний кадр» без очереди.

Потребитель (CameraWorker) берёт самый свежий кадр; старые кадры перезаписываются,
поэтому лаг не накапливается, даже если анализ медленнее источника.
Колбэк on_frame вызывается на КАЖДЫЙ кадр (кольцевой буфер клипов, fps_in).
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from guard.contracts import SourceConfig

log = logging.getLogger(__name__)


@dataclass(slots=True)
class FramePacket:
    frame: np.ndarray  # BGR
    ts: float  # epoch-время захвата
    seq: int  # номер кадра источника, растёт монотонно
    decode_ms: float  # время чтения и декодирования
    pos_s: float | None = None  # позиция в файле (для file)


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

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive() and not self._stop.is_set()

    def _safe_run(self) -> None:
        try:
            self._run()
        except Exception as exc:  # noqa: BLE001 — поток источника не должен падать молча
            self._error(exc)
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
        self.last_error = str(exc)[:300]

    # ---- выдача кадров ----

    def _publish(self, frame: np.ndarray, decode_ms: float, ts: float | None = None, pos_s: float | None = None) -> None:
        self._seq += 1
        pkt = FramePacket(frame=frame, ts=time.time() if ts is None else ts, seq=self._seq, decode_ms=decode_ms, pos_s=pos_s)
        self.frame_size = (frame.shape[1], frame.shape[0])
        with self._cond:
            self._latest = pkt
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

    def status(self) -> dict:
        return {
            "kind": self.kind,
            "connected": self.connected,
            "eof": self.eof,
            "frames": self._seq,
            "errors": self.errors,
            "last_error": self.last_error,
            "native_fps": self.native_fps,
            "frame_size": self.frame_size,
        }
