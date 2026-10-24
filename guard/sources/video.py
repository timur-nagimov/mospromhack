"""Источники на cv2.VideoCapture: файл (реальный темп, по кругу), сетевой поток
(RTSP, HTTP MJPEG, HLS, YouTube Live), веб-камера. Форматы и ограничения — docs/STREAMS.md."""

from __future__ import annotations

import logging
import math
import os
import time
from collections.abc import Callable
from typing import Any

import cv2

from guard.contracts import SourceConfig
from guard.sources.base import CONNECTING, ERROR, OFFLINE, ONLINE, RECONNECTING, FrameCallback, FrameSource
from guard.sources.stream_url import StreamResolveError, paced_kind, resolve_stream_uri, stream_kind

log = logging.getLogger(__name__)

# RTSP по TCP надёжнее UDP в сетях с потерями; пользовательская настройка имеет приоритет
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")


def _env_seconds(name: str, default: float) -> float:
    try:
        value = float(os.environ.get(name) or default)
    except ValueError:
        return default
    return value if value > 0 else default


OPEN_TIMEOUT_S = _env_seconds("GUARD_STREAM_OPEN_TIMEOUT_S", 10.0)  # открытие потока (CAP_PROP_OPEN_TIMEOUT_MSEC)
READ_TIMEOUT_S = _env_seconds("GUARD_STREAM_READ_TIMEOUT_S", 5.0)  # чтение кадра (CAP_PROP_READ_TIMEOUT_MSEC)
# Потоки декодера H.264 для сетевых потоков (CAP_PROP_N_THREADS). По умолчанию FFmpeg берёт по потоку на ядро,
# и каждый поток кадрового распараллеливания держит кадр: 12 потоков — +480 мс задержки при 25 к/с,
# 1 поток — 45 мс при той же загрузке CPU (замер на 720p, streams-*.json). 4K можно поднять до 2–4.
DECODE_THREADS = int(_env_seconds("GUARD_STREAM_DECODE_THREADS", 1))
# Живой HLS: FFmpeg начинает за 3 сегмента до края эфира. Этот запас пропускаем без выдачи, пока чтение
# не встанет у края (ожидание следующего сегмента ≥ CATCHUP_EDGE_RATIO его длительности), но не дольше лимитов.
CATCHUP_BLOCK_S = 0.3
CATCHUP_EDGE_RATIO = 0.5
CATCHUP_MAX_STREAM_S = 60.0  # сегменты бывают по 15 с: запас из 3 сегментов — 45 с
CATCHUP_MAX_WALL_S = 30.0
# Сегмент HLS длится 2–15 с: у края эфира чтение ждёт следующий сегмент, открытие качает первый
HLS_MIN_TIMEOUT_S = 30.0
BACKOFF_S = (1.0, 2.0, 5.0, 10.0)  # паузы перед повторными попытками: 1 → 2 → 5 → 10 с, дальше по 10
OFFLINE_AFTER_ATTEMPTS = 4  # столько неудачных попыток подряд — «нет сигнала» (около 18 с)
READ_FAILS_MAX = 5  # подряд неудачных чтений — обрыв (сервер закрыл поток, EOF)

CODEC_NAMES = {"avc1": "h264", "h264": "h264", "x264": "h264", "hev1": "h265", "hvc1": "h265", "hevc": "h265",
               "h265": "h265", "mjpg": "mjpeg", "jpeg": "mjpeg", "mp4v": "mpeg4", "fmp4": "mpeg4", "av01": "av1",
               "vp80": "vp8", "vp90": "vp9"}

CaptureFactory = Callable[..., Any]


def fourcc_name(cap: Any) -> str | None:
    """Кодек по CAP_PROP_FOURCC: h264, h265, mjpeg… или сырой fourcc."""
    try:
        value = int(cap.get(cv2.CAP_PROP_FOURCC))
    except (TypeError, ValueError, cv2.error):
        return None
    if value <= 0:
        return None
    raw = "".join(chr((value >> (8 * i)) & 0xFF) for i in range(4)).strip("\x00 ").strip()
    if not raw or not raw.isprintable():
        return None
    return CODEC_NAMES.get(raw.lower(), raw.lower())


def sane_fps(value: float | None) -> float | None:
    """CAP_PROP_FPS у потоков бывает 0, 90000 или 100 вместо 25: берём только 1..60."""
    try:
        fps = float(value or 0.0)
    except (TypeError, ValueError):
        return None
    return fps if 1.0 <= fps <= 60.0 else None


def stream_timeouts(kind: str, open_s: float | None = None, read_s: float | None = None) -> tuple[float, float]:
    open_s = float(open_s or OPEN_TIMEOUT_S)
    read_s = float(read_s or READ_TIMEOUT_S)
    if paced_kind(kind):
        open_s, read_s = max(open_s, HLS_MIN_TIMEOUT_S), max(read_s, HLS_MIN_TIMEOUT_S)
    return open_s, read_s


def open_capture(
    uri: str, open_timeout_s: float, read_timeout_s: float, factory: CaptureFactory | None = None, threads: int | None = None,
) -> Any:
    """VideoCapture через FFmpeg с таймаутами открытия и чтения (OpenCV ≥ 4.5.2) и числом потоков декодера."""
    factory = factory or cv2.VideoCapture
    params = [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, int(open_timeout_s * 1000), cv2.CAP_PROP_READ_TIMEOUT_MSEC, int(read_timeout_s * 1000)]
    threads = DECODE_THREADS if threads is None else threads
    if threads > 0 and hasattr(cv2, "CAP_PROP_N_THREADS"):
        try:
            return factory(uri, cv2.CAP_FFMPEG, [*params, cv2.CAP_PROP_N_THREADS, int(threads)])
        except cv2.error:
            pass
    try:
        return factory(uri, cv2.CAP_FFMPEG, params)
    except cv2.error:
        return factory(uri)


class Pacer:
    """Темп выдачи по меткам времени кадров (CAP_PROP_POS_MSEC) для потоков, приходящих пачками.

    HLS отдаёт сегмент целиком: без выравнивания 6 с видео декодируются за полсекунды, затем поток стоит,
    и анализ видит рывки, а правила — неверные длительности и скорости. Здесь кадр ждёт своего момента;
    опоздавшие кадры выдаются сразу (догоняем). Отстали больше MAX_LAG_S или метки прыгнули — отсчёт заново.
    """

    MAX_LAG_S = 10.0
    MAX_LEAD_S = 30.0

    def __init__(self, wall: Callable[[], float] = time.time) -> None:
        self._wall = wall
        self._anchor: tuple[float, float] | None = None  # (monotonic, pts_s)
        self._last_pts: float | None = None
        self._last_ts = 0.0

    def schedule(self, pos_ms: Any, now: float) -> tuple[float, float | None]:
        """(сколько подождать, epoch-время кадра в темпе потока); (0, None) — меток нет, темп не выравниваем."""
        try:
            pts = float(pos_ms) / 1000.0
        except (TypeError, ValueError):
            return 0.0, None
        if not math.isfinite(pts) or pts < 0:
            return 0.0, None
        if self._last_pts is not None and pts == self._last_pts:
            return 0.0, None
        if self._anchor is None or self._last_pts is None or pts < self._last_pts:
            self._anchor = (now, pts)
        due = self._anchor[0] + (pts - self._anchor[1])
        lead = due - now
        if lead > self.MAX_LEAD_S or lead < -self.MAX_LAG_S:
            self._anchor, due, lead = (now, pts), now, 0.0
        self._last_pts = pts
        ts = max(self._wall() + (due - now), self._last_ts + 1e-4)
        self._last_ts = ts
        return max(0.0, lead), ts


class _CatchUp:
    """Живой HLS: пропустить запас сегментов, с которого FFmpeg начинает (по умолчанию 3 до края эфира).

    Запас читается почти мгновенно: долгое чтение бывает только на загрузке сегмента. У края эфира чтение ждёт
    появления следующего сегмента — ожидание сравнимо с длительностью сегмента (span по меткам времени).
    Без меток времени или дольше лимитов — выдаём как есть. Замер: задержка HLS 7.6 → около 3 с (streams-*.json).
    """

    def __init__(self) -> None:
        self.t0: float | None = None
        self.first_pts: float | None = None
        self.seg_pts: float | None = None

    def backlog(self, pos_ms: Any, read_s: float, now: float) -> bool:
        """True — кадр из запаса (пропустить), False — край эфира достигнут, дальше выдавать."""
        try:
            pts = float(pos_ms) / 1000.0
        except (TypeError, ValueError):
            return False
        if not math.isfinite(pts) or pts < 0:
            return False
        if self.t0 is None or self.first_pts is None or self.seg_pts is None:
            self.t0, self.first_pts, self.seg_pts = now, pts, pts
            return True
        span = pts - self.seg_pts
        if read_s >= CATCHUP_BLOCK_S:
            if span > 0 and read_s >= CATCHUP_EDGE_RATIO * span:
                return False  # ждали сегмент почти его длительность — это край эфира
            self.seg_pts = pts  # загрузка очередного сегмента запаса
        return pts - self.first_pts < CATCHUP_MAX_STREAM_S and now - self.t0 < CATCHUP_MAX_WALL_S



class FileSource(FrameSource):
    """Видеофайл. realtime=True — темп по FPS файла (отстаём — пропускаем кадры через grab);
    loop=True — по кругу. realtime=False — без пауз (последний кадр всё равно перезаписывается)."""

    kind = "file"
    frame_count: int | None = None
    _sync_target: tuple[int, float] | None = None

    def sync(self, frame_idx: int, at_mono: float | None = None) -> None:
        """Режим gateway: подтянуть воспроизведение к позиции frame_idx (кадр файла у edge),
        которую edge обработал в момент at_mono (локальные часы). Без перемотки при дрейфе < 0.5 с."""
        self._sync_target = (int(frame_idx), time.monotonic() if at_mono is None else at_mono)

    def _open(self) -> cv2.VideoCapture | None:
        cap = cv2.VideoCapture(self.uri)
        if not cap.isOpened():
            self._error(f"cannot open video file {self.uri}")
            self.state = ERROR
            cap.release()
            return None
        fps = cap.get(cv2.CAP_PROP_FPS)
        self.native_fps = fps if fps and 1.0 <= fps <= 240.0 else 25.0
        n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self.frame_count = int(n) if n and n > 0 else None
        self.codec = fourcc_name(cap)
        self.state = ONLINE
        return cap

    def _run(self) -> None:
        while not self._stop.is_set():
            cap = self._open()
            if cap is None:
                if not self._sleep(2.0):
                    return
                continue
            self.connected, self.eof = True, False
            self._new_capture()
            fps = self.native_fps or 25.0
            t0 = time.monotonic()
            idx = 0  # номер следующего кадра от начала прохода
            try:
                while not self._stop.is_set():
                    target = self._sync_target
                    if target is not None:
                        self._sync_target = None
                        desired = target[0] + (time.monotonic() - target[1]) * fps
                        if self.frame_count:
                            desired %= self.frame_count
                        if abs(desired - idx) > max(0.5 * fps, 3):
                            cap.set(cv2.CAP_PROP_POS_FRAMES, int(desired))
                            idx = int(desired)
                        t0 = time.monotonic() - desired / fps  # дальше темп ведёт часы edge
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
                self.state = OFFLINE
                return
            if idx == 0 and not self._sleep(1.0):  # пустой/битый файл — не крутимся вхолостую
                return




class LiveSource(FrameSource):
    """Сетевой поток (RTSP, HTTP MJPEG, HLS, YouTube Live) или веб-камера.

    Читаем непрерывно (буфер декодера не копится). Таймауты открытия и чтения — CAP_PROP_OPEN_TIMEOUT_MSEC
    и CAP_PROP_READ_TIMEOUT_MSEC. Обрыв (нет кадров дольше таймаута чтения или READ_FAILS_MAX неудач подряд) —
    переподключение с паузами 1 → 2 → 5 → 10 с. Ссылка YouTube переразрешается через yt-dlp при каждой попытке.
    HLS и файлы по HTTP выдаются в темпе меток времени (cfg.realtime, Pacer). Состояние — status()["state"].
    """

    kind = "rtsp"

    def __init__(
        self,
        cfg: SourceConfig,
        uri: str,
        camera_id: str = "",
        on_frame: FrameCallback | None = None,
        *,
        open_timeout_s: float | None = None,
        read_timeout_s: float | None = None,
        capture_factory: CaptureFactory | None = None,
        resolver: Callable[[str], tuple[str, str | None]] | None = None,
        diagnose: Callable[[str], str | None] | None = None,
    ) -> None:
        super().__init__(cfg, uri, camera_id, on_frame)
        self.stream_kind = "webcam" if cfg.type == "webcam" else stream_kind(uri)
        self.open_timeout_s, self.read_timeout_s = stream_timeouts(self.stream_kind, open_timeout_s, read_timeout_s)
        self.stall_s = self.read_timeout_s  # столько без кадров — обрыв
        self.paced = bool(cfg.realtime) and paced_kind(self.stream_kind)
        self.resolved_via: str | None = None
        self._capture_factory = capture_factory or cv2.VideoCapture
        self._resolver = resolver or resolve_stream_uri
        self._diagnose = diagnose
        self._config_error = False
        self._ever_online = False
        self._lost_at: float | None = None  # monotonic последнего кадра перед обрывом

    # ---- подключение ----

    def _connect(self) -> Any | None:
        """Одна попытка: разрешить адрес, открыть. None — не удалось (last_error заполнен)."""
        self._config_error = False
        if self.cfg.type == "webcam":
            index = int(self.uri) if str(self.uri).strip().isdigit() else self.uri
            cap = self._capture_factory(index)
        else:
            try:
                target, self.resolved_via = self._resolver(self.uri)
            except StreamResolveError as exc:
                self._config_error = True
                self._error(f"адрес не разрешён: {exc}")
                return None
            cap = open_capture(target, self.open_timeout_s, self.read_timeout_s, self._capture_factory)
        if not cap.isOpened():
            cap.release()
            reason = self._diagnose_failure()
            self._error(f"не удалось открыть поток {self.uri}" + (f": {reason}" if reason else ""))
            return None
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except cv2.error:
            pass
        self.native_fps = sane_fps(cap.get(cv2.CAP_PROP_FPS))
        self.codec = fourcc_name(cap)
        return cap

    def _diagnose_failure(self) -> str | None:
        """Почему не открылось: DNS, порт, 401/404 — проверка за пару секунд (guard.sources.probe)."""
        if self.cfg.type == "webcam":
            return None
        diagnose = self._diagnose
        if diagnose is None:
            from guard.sources.probe import diagnose_stream

            diagnose = diagnose_stream
        try:
            return diagnose(self.uri)
        except Exception:  # noqa: BLE001 — диагностика необязательна
            log.debug("diagnose failed for %s", self.camera_id, exc_info=True)
            return None

    # ---- цикл источника ----

    def _run(self) -> None:
        pauses = 0  # пауз в текущей серии попыток — индекс в BACKOFF_S
        while not self._stop.is_set():
            t_attempt = time.monotonic()
            cap = self._connect()
            got = False
            if cap is not None:
                got = self._stream(cap, t_attempt)
            if self._stop.is_set():
                return
            if got:
                pauses = 0
            else:
                self._attempt_failed()
            pause = BACKOFF_S[min(pauses, len(BACKOFF_S) - 1)]
            pauses += 1
            self.retry_at = time.monotonic() + pause
            slept = self._sleep(pause)
            self.retry_at = None
            if not slept:
                return

    def _attempt_failed(self) -> None:
        self.attempts += 1
        if self._config_error:
            self.state = ERROR
        elif self.attempts >= OFFLINE_AFTER_ATTEMPTS:
            self.state = OFFLINE
        else:
            self.state = RECONNECTING if self._ever_online else CONNECTING

    def _stream(self, cap: Any, t_attempt: float) -> bool:
        """Чтение до обрыва или остановки. True — получен хотя бы один кадр."""
        self.connected = True
        self._new_capture()
        pacer = Pacer() if self.paced else None
        catchup = _CatchUp() if self.paced and self.stream_kind in ("hls", "youtube") else None
        got = False
        fails = 0
        last_ok = time.monotonic()
        try:
            while not self._stop.is_set():
                t_read = time.perf_counter()
                ok, frame = cap.read()
                decode_ms = (time.perf_counter() - t_read) * 1000.0
                now = time.monotonic()
                if not ok or frame is None:
                    fails += 1
                    stalled = now - last_ok
                    if fails >= READ_FAILS_MAX or stalled >= self.stall_s:
                        what = "поток оборвался" if got else "поток открылся, но кадров нет"
                        self._error(f"{what}: {stalled:.1f} с без кадров")
                        break
                    if not self._sleep(0.05):
                        break
                    continue
                fails, last_ok = 0, now
                if catchup is not None:
                    if catchup.backlog(cap.get(cv2.CAP_PROP_POS_MSEC), decode_ms / 1000.0, now):
                        continue  # кадр из запаса за краем эфира — не выдаём
                    catchup = None
                ts = None
                if pacer is not None:
                    delay, ts = pacer.schedule(cap.get(cv2.CAP_PROP_POS_MSEC), now)
                    if delay > 0 and not self._sleep(delay):
                        break
                if not got:
                    got = True
                    self._on_online(t_attempt)
                self._publish(frame, decode_ms, ts=ts)
        finally:
            cap.release()
            self.connected = False
        if got and not self._stop.is_set():
            self._on_lost()
        return got

    def _on_online(self, t_attempt: float) -> None:
        now = time.monotonic()
        self.connect_ms = (now - t_attempt) * 1000.0
        if self._lost_at is not None:
            self.last_outage_s = now - self._lost_at
            self.reconnects += 1
            self._lost_at = None
        self.attempts = 0
        self.state = ONLINE
        self.online_since = time.time()
        self._ever_online = True
        if self.reconnects:
            log.info("source %s online again after %.1f s (reconnects %d)", self.camera_id, self.last_outage_s or 0.0, self.reconnects)

    def _on_lost(self) -> None:
        self.drops += 1
        self._lost_at = self._last_frame_mono or time.monotonic()
        self.state = RECONNECTING
        self.online_since = None
        log.warning("source %s lost: %s", self.camera_id, self.last_error)

    def status(self) -> dict[str, Any]:
        return {**super().status(), "stream": self.stream_kind, "resolved_via": self.resolved_via, "paced": self.paced}


class WebcamSource(LiveSource):
    kind = "webcam"
