"""Проверка источника до добавления камеры (POST /api/cameras/probe) и диагностика сбоев подключения.

probe_source открывает источник так же, как камера (FFmpeg, таймауты, разрешение ссылки YouTube),
ждёт первый кадр, измеряет разрешение, к/с и кодек и возвращает уменьшенный кадр-превью.
diagnose_stream объясняет, почему поток не открылся: DNS, закрытый порт, ответ сервера RTSP или HTTP
(401 — нужен пароль, 404 — неверный путь, HTML-страница вместо потока). Форматы — docs/STREAMS.md.
"""

from __future__ import annotations

import base64
import hashlib
import socket
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit, urlunsplit

import cv2

from guard.contracts import SourceConfig
from guard.sources.images import list_images
from guard.sources.stream_url import StreamResolveError, is_network, mask_uri, resolve_stream_uri, stream_kind
from guard.sources.video import CaptureFactory, fourcc_name, open_capture, sane_fps

DEFAULT_PORTS = {"rtsp": 554, "rtsps": 322, "http": 80, "https": 443}
PROBE_TIMEOUT_S = 15.0
PREVIEW_WIDTH = 640
FPS_SAMPLE_S = 1.5  # сколько читать после первого кадра, чтобы измерить к/с
FPS_SAMPLE_FRAMES = 40
DIAG_TIMEOUT_S = 3.0
BIG_FRAME_PX = 1920 * 1088


# ---------------------------------------------------------------- диагностика


def _without_userinfo(uri: str) -> tuple[str, str | None, str | None]:
    parts = urlsplit(uri)
    user = unquote(parts.username) if parts.username is not None else None
    password = unquote(parts.password) if parts.password is not None else None
    netloc = parts.hostname or ""
    if ":" in netloc:  # IPv6
        netloc = f"[{netloc}]"
    if parts.port:
        netloc = f"{netloc}:{parts.port}"
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, "")), user, password


def _resolve_host(host: str, port: int, timeout_s: float) -> None:
    """getaddrinfo без таймаута может висеть — ждём его в отдельном потоке."""
    box: dict[str, BaseException | None] = {}

    def work() -> None:
        try:
            socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
            box["err"] = None
        except BaseException as exc:  # noqa: BLE001
            box["err"] = exc

    t = threading.Thread(target=work, daemon=True, name="probe-dns")
    t.start()
    t.join(timeout_s)
    if "err" not in box:
        raise TimeoutError
    if box["err"] is not None:
        raise box["err"]


def _read_response(sock: socket.socket, limit: int = 65536) -> tuple[int | None, str, dict[str, list[str]]]:
    data = b""
    while b"\r\n\r\n" not in data and len(data) < limit:
        chunk = sock.recv(4096)
        if not chunk:
            break
        data += chunk
    head = data.split(b"\r\n\r\n", 1)[0].decode("latin-1", "replace")
    lines = head.split("\r\n")
    parts = lines[0].split(" ", 2) if lines else []
    code = int(parts[1]) if len(parts) >= 2 and parts[0].startswith("RTSP/") and parts[1].isdigit() else None
    reason = parts[2] if len(parts) > 2 else ""
    headers: dict[str, list[str]] = {}
    for line in lines[1:]:
        if ":" in line:
            k, v = line.split(":", 1)
            headers.setdefault(k.strip().lower(), []).append(v.strip())
    return code, reason, headers


def _digest_header(challenge: str, user: str, password: str, method: str, uri: str) -> str:
    fields: dict[str, str] = {}
    for item in challenge[len("Digest"):].split(","):
        if "=" in item:
            k, v = item.split("=", 1)
            fields[k.strip().lower()] = v.strip().strip('"')
    realm, nonce = fields.get("realm", ""), fields.get("nonce", "")
    ha1 = hashlib.md5(f"{user}:{realm}:{password}".encode()).hexdigest()  # noqa: S324 — так устроен RTSP Digest
    ha2 = hashlib.md5(f"{method}:{uri}".encode()).hexdigest()  # noqa: S324
    response = hashlib.md5(f"{ha1}:{nonce}:{ha2}".encode()).hexdigest()  # noqa: S324
    return f'Digest username="{user}", realm="{realm}", nonce="{nonce}", uri="{uri}", response="{response}"'


def _rtsp_describe(sock: socket.socket, url: str, user: str | None, password: str | None) -> tuple[int | None, str, bool]:
    """(код ответа DESCRIBE, текст, пробовали ли логин и пароль)."""
    request = f"DESCRIBE {url} RTSP/1.0\r\nCSeq: 2\r\nUser-Agent: guard-probe\r\nAccept: application/sdp\r\n\r\n"
    sock.sendall(request.encode())
    code, reason, headers = _read_response(sock)
    if code != 401 or user is None:
        return code, reason, False
    challenges = headers.get("www-authenticate", [])
    digest = next((c for c in challenges if c.lower().startswith("digest")), None)
    if digest:
        auth = _digest_header(digest, user, password or "", "DESCRIBE", url)
    else:
        auth = "Basic " + base64.b64encode(f"{user}:{password or ''}".encode()).decode()
    request = f"DESCRIBE {url} RTSP/1.0\r\nCSeq: 3\r\nUser-Agent: guard-probe\r\nAccept: application/sdp\r\nAuthorization: {auth}\r\n\r\n"
    sock.sendall(request.encode())
    code, reason, _ = _read_response(sock)
    return code, reason, True


def _rtsp_message(code: int | None, reason: str, tried_auth: bool) -> str:
    if code is None:
        return "по адресу отвечает не RTSP-сервер: проверьте порт и протокол (rtsp://, http://)"
    if code == 200:
        return ("RTSP-сервер отвечает (200), но поток не декодируется: проверьте кодек (H.265 декодируется "
                "медленнее и не всеми сборками) или увеличьте GUARD_STREAM_OPEN_TIMEOUT_S")
    if code == 401:
        if tried_auth:
            return "неверный логин или пароль (RTSP 401)"
        return "камера требует логин и пароль (RTSP 401): rtsp://логин:пароль@адрес/…"
    if code == 403:
        return "доступ запрещён (RTSP 403): у пользователя нет права смотреть поток"
    if code == 404:
        return ("поток не найден (RTSP 404): проверьте путь после адреса камеры — у Hikvision /Streaming/Channels/101, "
                "у Dahua /cam/realmonitor?channel=1&subtype=0; у ретранслятора — что публикация идёт")
    if code >= 500:
        return f"ошибка RTSP-сервера {code} {reason}".strip()
    return f"RTSP-сервер ответил {code} {reason}".strip()


def _http_message(url: str, user: str | None, password: str | None, timeout_s: float) -> str | None:
    headers = {"User-Agent": "guard-probe"}
    if user is not None:
        headers["Authorization"] = "Basic " + base64.b64encode(f"{user}:{password or ''}".encode()).decode()
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:  # noqa: S310 — адрес задал администратор
            ctype = (resp.headers.get("Content-Type") or "").lower()
    except urllib.error.HTTPError as exc:
        code = exc.code
        if code == 401:
            if user is not None:
                return "неверный логин или пароль (HTTP 401)"
            return "нужна авторизация (HTTP 401): http://логин:пароль@адрес/…"
        if code == 403:
            return ("доступ запрещён (HTTP 403): ссылки публичных камер часто содержат временный токен — "
                    "получите свежую ссылку на странице камеры")
        if code == 404:
            return "не найдено (HTTP 404): проверьте путь; для HLS — что трансляция идёт"
        return f"сервер ответил HTTP {code}"
    except urllib.error.URLError as exc:
        reason = getattr(exc, "reason", exc)
        return f"HTTP-запрос не прошёл: {reason}"
    except (TimeoutError, socket.timeout):
        return f"сервер не ответил на HTTP-запрос за {timeout_s:.0f} с"
    except OSError as exc:
        return f"HTTP-запрос не прошёл: {exc}"
    if "text/html" in ctype:
        return "по ссылке веб-страница, а не видеопоток: нужна прямая ссылка на поток (.m3u8, MJPEG, RTSP или страница YouTube)"
    if "mpegurl" in ctype:
        return "плейлист HLS доступен, но видео не декодируется: возможно, сегменты закрыты токеном или кодек не поддерживается"
    if "multipart" in ctype:
        return "MJPEG-поток доступен, но кадры не декодируются"
    return None


def diagnose_stream(uri: str, timeout_s: float = DIAG_TIMEOUT_S) -> str | None:
    """Короткое объяснение по-русски, почему сетевой поток не открывается; None — причина не найдена."""
    uri = (uri or "").strip()
    parts = urlsplit(uri)
    scheme = parts.scheme.lower()
    if scheme not in DEFAULT_PORTS:
        return None
    host = parts.hostname
    if not host:
        return "в адресе нет хоста камеры"
    try:
        port = parts.port or DEFAULT_PORTS[scheme]
    except ValueError:
        return "в адресе неверный порт"
    try:
        _resolve_host(host, port, timeout_s)
    except TimeoutError:
        return f"имя {host} не разрешилось за {timeout_s:.0f} с (DNS)"
    except OSError:
        return f"адрес {host} не найден (DNS)"
    try:
        sock = socket.create_connection((host, port), timeout=timeout_s)
    except ConnectionRefusedError:
        return f"порт {port} на {host} закрыт: камера или сервер не принимает подключения"
    except (TimeoutError, socket.timeout):
        return f"{host}:{port} не отвечает {timeout_s:.0f} с: проверьте сеть, маршрут и брандмауэр"
    except OSError as exc:
        return f"{host}:{port} недоступен: {exc.strerror or exc}"
    url, user, password = _without_userinfo(uri)
    try:
        if scheme == "rtsp":
            sock.settimeout(timeout_s)
            try:
                code, reason, tried = _rtsp_describe(sock, url, user, password)
            except (TimeoutError, socket.timeout):
                return f"{host}:{port} принимает подключение, но не отвечает по RTSP {timeout_s:.0f} с"
            except OSError as exc:
                return f"RTSP-запрос не прошёл: {exc}"
            return _rtsp_message(code, reason, tried)
    finally:
        sock.close()
    if scheme in ("http", "https"):
        return _http_message(url, user, password, timeout_s)
    return None


# ---------------------------------------------------------------- проверка источника


def _empty_result(cfg: SourceConfig, uri: str) -> dict[str, Any]:
    return {
        "ok": False,
        "type": cfg.type,
        "stream": stream_kind(uri) if cfg.type == "rtsp" else cfg.type,
        "uri": mask_uri(cfg.uri),
        "width": None,
        "height": None,
        "fps": None,
        "fps_nominal": None,
        "fps_measured": None,
        "codec": None,
        "first_frame_ms": None,
        "open_ms": None,
        "frames": 0,
        "resolved_via": None,
        "error": None,
        "error_code": None,
        "hints": [],
        "preview_jpeg_b64": None,
    }


def _fail(result: dict[str, Any], code: str, message: str) -> dict[str, Any]:
    result["ok"] = False
    result["error_code"] = code
    result["error"] = mask_uri(message)
    return result


def _preview(frame: Any, width: int = PREVIEW_WIDTH, quality: int = 75) -> str | None:
    h, w = frame.shape[:2]
    if w > width:
        frame = cv2.resize(frame, (width, max(1, round(h * width / w))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf.tobytes()).decode("ascii") if ok else None


def _hints(result: dict[str, Any]) -> list[str]:
    hints = []
    w, h = result.get("width") or 0, result.get("height") or 0
    if w * h > BIG_FRAME_PX:
        hints.append(f"Разрешение {w}×{h}: декодирование тяжёлое, для анализа достаточно субпотока 640–1280 px "
                     "(у Hikvision — канал 102, у Dahua — subtype=1)")
    if result.get("codec") == "h265":
        hints.append("H.265 декодируется дороже H.264; если камера умеет, переключите поток на H.264")
    if result.get("stream") in ("hls", "youtube"):
        hints.append("HLS идёт с задержкой в 1–3 сегмента (обычно 5–30 с): события придут позже, чем по RTSP")
    if result.get("stream") == "youtube":
        hints.append("Адрес видео YouTube временный: сервис получает его заново при каждом подключении")
    return hints


def _probe_images(cfg: SourceConfig, uri: str, result: dict[str, Any]) -> dict[str, Any]:
    t0 = time.perf_counter()
    files = list_images(uri)
    if not files:
        return _fail(result, "not_found", f"в {uri} нет изображений (jpg, png, bmp, webp)")
    frame = cv2.imread(str(files[0]), cv2.IMREAD_COLOR)
    if frame is None:
        return _fail(result, "no_frames", f"не удалось прочитать {files[0].name}")
    result.update(ok=True, frames=len(files), first_frame_ms=round((time.perf_counter() - t0) * 1000.0, 1),
                  width=frame.shape[1], height=frame.shape[0], codec="images", preview_jpeg_b64=_preview(frame))
    return result


def _measure(cap: Any, first: Any, deadline: float) -> tuple[Any, int, float | None]:
    """Дочитать до FPS_SAMPLE_FRAMES кадров: (последний кадр, сколько прочитано, к/с по меткам или по часам)."""
    pos0 = cap.get(cv2.CAP_PROP_POS_MSEC)
    last, n = first, 1
    pos_last = pos0
    t_start = time.perf_counter()
    stop_at = min(deadline, t_start + FPS_SAMPLE_S)
    while n < FPS_SAMPLE_FRAMES and time.perf_counter() < stop_at:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        last, n = frame, n + 1
        pos_last = cap.get(cv2.CAP_PROP_POS_MSEC)
    elapsed = time.perf_counter() - t_start
    if n < 2:
        return last, n, None
    try:
        span = (float(pos_last) - float(pos0)) / 1000.0
    except (TypeError, ValueError):
        span = 0.0
    if span > 0:  # по меткам времени: HLS приходит пачкой, по часам вышло бы 100+ к/с
        return last, n, (n - 1) / span
    return last, n, ((n - 1) / elapsed if elapsed > 0 else None)


def _probe_capture(
    cfg: SourceConfig, uri: str, timeout_s: float, result: dict[str, Any],
    factory: CaptureFactory | None, resolver: Any, diagnose: Any,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    factory = factory or cv2.VideoCapture
    if cfg.type == "webcam":
        cap = factory(int(uri) if uri.isdigit() else uri)
    elif cfg.type == "file":
        cap = factory(uri)
    else:
        try:
            target, result["resolved_via"] = resolver(uri)
        except StreamResolveError as exc:
            return _fail(result, "resolve_failed", f"адрес не разрешён: {exc}")
        cap = open_capture(target, timeout_s, timeout_s, factory)
    try:
        if not cap.isOpened():
            reason = diagnose(uri) if cfg.type == "rtsp" else None
            what = {"file": "файл не открылся", "webcam": "веб-камера не открылась"}.get(cfg.type, "поток не открылся")
            if cfg.type == "webcam":
                reason = "проверьте индекс устройства и разрешение на доступ к камере для процесса сервера"
            return _fail(result, "open_failed", f"{what}: {reason}" if reason else f"{what} за {timeout_s:.0f} с")
        result["open_ms"] = round((time.perf_counter() - t0) * 1000.0, 1)
        ok, frame = cap.read()
        if not ok or frame is None:
            reason = diagnose(uri) if cfg.type == "rtsp" else None
            msg = f"источник открылся, но кадр не пришёл за {timeout_s:.0f} с"
            return _fail(result, "no_frames", f"{msg}: {reason}" if reason else msg)
        result["first_frame_ms"] = round((time.perf_counter() - t0) * 1000.0, 1)
        result["codec"] = fourcc_name(cap)
        result["fps_nominal"] = sane_fps(cap.get(cv2.CAP_PROP_FPS))
        last, n, measured = _measure(cap, frame, t0 + timeout_s)
        result["frames"] = n
        result["fps_measured"] = round(measured, 2) if measured else None
        # измеренные к/с точнее: у MJPEG FFmpeg подставляет 25, у HLS бывает 100 вместо 25
        fps = measured if measured and n >= 5 else result["fps_nominal"] or measured
        result.update(ok=True, width=int(frame.shape[1]), height=int(frame.shape[0]),
                      fps=round(fps, 2) if fps else None, preview_jpeg_b64=_preview(last))
        result["hints"] = _hints(result)
        return result
    finally:
        cap.release()


def probe_source(
    cfg: SourceConfig,
    uri: str | None = None,
    timeout_s: float = PROBE_TIMEOUT_S,
    *,
    capture_factory: CaptureFactory | None = None,
    resolver: Any = None,
    diagnose: Any = None,
) -> dict[str, Any]:
    """Открыть источник и дождаться первого кадра. uri — уже разрешённый путь (для file/images
    относительно GUARD_DATA_DIR), по умолчанию cfg.uri. Ответ ограничен по времени: timeout_s плюс запас
    на разрешение ссылки YouTube и диагностику; зависший cv2 дорабатывает в фоне и закрывается сам."""
    uri = (uri or cfg.uri or "").strip()
    result = _empty_result(cfg, uri)
    timeout_s = float(timeout_s)
    if not uri:
        return _fail(result, "bad_uri", "адрес источника пуст")
    if cfg.type == "images":
        return _probe_images(cfg, uri, result)
    if cfg.type == "file" and not Path(uri).is_file():
        return _fail(result, "not_found", f"файл не найден: {cfg.uri}")
    if cfg.type == "rtsp" and not is_network(uri):
        return _fail(result, "bad_uri", "адрес сетевого потока начинается с rtsp://, http:// или https://")
    resolver = resolver or resolve_stream_uri
    diagnose = diagnose or diagnose_stream
    box: dict[str, Any] = {}

    def work() -> None:
        try:
            box["result"] = _probe_capture(cfg, uri, timeout_s, result, capture_factory, resolver, diagnose)
        except Exception as exc:  # noqa: BLE001 — ответ API вместо 500
            box["result"] = _fail(result, "error", f"{type(exc).__name__}: {exc}")

    budget = timeout_s * 2 + DIAG_TIMEOUT_S * 2 + (45.0 if result["stream"] == "youtube" else 0.0)
    thread = threading.Thread(target=work, daemon=True, name="probe")
    thread.start()
    thread.join(budget)
    if "result" not in box:
        return _fail(dict(result), "timeout", f"источник не ответил за {budget:.0f} с")
    return box["result"]
