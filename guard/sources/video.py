"""Источники на cv2.VideoCapture: файл (реальный темп, по кругу), RTSP/HTTP-поток, веб-камера."""

from __future__ import annotations

import logging
import os
import time

import cv2

from guard.sources.base import FrameSource

log = logging.getLogger(__name__)

# RTSP по TCP надёжнее UDP в сетях с потерями; пользовательская настройка имеет приоритет
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")


class FileSource(FrameSource):
    """Видеофайл. realtime=True — темп по FPS файла (отстаём — пропускаем кадры через grab);
    loop=True — по кругу. realtime=False — без пауз (последний кадр всё равно перезаписывается)."""

    kind = "file"

    def _open(self) -> cv2.VideoCapture | None:
        cap = cv2.VideoCapture(self.uri)
        if not cap.isOpened():
            self._error(f"cannot open video file {self.uri}")
            cap.release()
            return None
        fps = cap.get(cv2.CAP_PROP_FPS)
        self.native_fps = fps if fps and 1.0 <= fps <= 240.0 else 25.0
        return cap

    def _run(self) -> None:
        while not self._stop.is_set():
            cap = self._open()
            if cap is None:
                if not self._sleep(2.0):
                    return
                continue
            self.connected, self.eof = True, False
            fps = self.native_fps or 25.0
            t0 = time.monotonic()
            idx = 0  # номер следующего кадра от начала прохода
            try:
                while not self._stop.is_set():
                    if self.cfg.realtime:
                        due = t0 + idx / fps
                        now = time.monotonic()
                        if now < due:
                            if not self._sleep(due - now):
                                return
                        elif now - due > 2.0 / fps:
                            # отстаём больше чем на 2 кадра — догоняем без декодирования в BGR
                            behind = int((now - due) * fps)
                            for _ in range(min(behind, int(fps))):
                                if not cap.grab():
                                    break
                                idx += 1
                    t_read = time.perf_counter()
                    ok, frame = cap.read()
                    decode_ms = (time.perf_counter() - t_read) * 1000.0
                    if not ok or frame is None:
                        break
                    self._publish(frame, decode_ms, pos_s=idx / fps)
                    idx += 1
            finally:
                cap.release()
            if not self.cfg.loop:
                self.eof = True
                self.connected = False
                return
            if idx == 0 and not self._sleep(1.0):  # пустой/битый файл — не крутимся вхолостую
                return


class LiveSource(FrameSource):
    """RTSP/HTTP-поток или веб-камера: читаем непрерывно (буфер декодера не копится),
    переподключение с экспоненциальной паузой."""

    kind = "rtsp"

    def _open(self) -> cv2.VideoCapture | None:
        if self.cfg.type == "webcam":
            index = int(self.uri) if str(self.uri).isdigit() else self.uri
            cap = cv2.VideoCapture(index)
        else:
            params = [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000]
            try:
                cap = cv2.VideoCapture(self.uri, cv2.CAP_FFMPEG, params)
            except cv2.error:
                cap = cv2.VideoCapture(self.uri)
        if not cap.isOpened():
            self._error(f"cannot open stream {self.uri}")
            cap.release()
            return None
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        fps = cap.get(cv2.CAP_PROP_FPS)
        self.native_fps = fps if fps and 1.0 <= fps <= 240.0 else None
        return cap

    def _run(self) -> None:
        backoff = 1.0
        while not self._stop.is_set():
            cap = self._open()
            if cap is None:
                if not self._sleep(backoff):
                    return
                backoff = min(backoff * 2, 15.0)
                continue
            self.connected = True
            backoff = 1.0
            failures = 0
            try:
                while not self._stop.is_set():
                    t_read = time.perf_counter()
                    ok, frame = cap.read()
                    decode_ms = (time.perf_counter() - t_read) * 1000.0
                    if not ok or frame is None:
                        failures += 1
                        if failures >= 25:
                            self._error("stream read failed, reconnecting")
                            break
                        if not self._sleep(0.02):
                            return
                        continue
                    failures = 0
                    self._publish(frame, decode_ms)
            finally:
                cap.release()
                self.connected = False
            if not self._sleep(backoff):
                return


class WebcamSource(LiveSource):
    kind = "webcam"
