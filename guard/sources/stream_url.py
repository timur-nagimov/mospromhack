"""Адреса сетевых потоков: разбор, маскировка секретов, YouTube Live через yt-dlp.

Описание форматов и ограничений — docs/STREAMS.md.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

# Значения этих параметров запроса — секреты (временные токены публичных камер, ключи API)
SECRET_QUERY_KEYS = frozenset({"token", "key", "apikey", "api_key", "password", "pass", "pwd", "auth", "sig", "signature", "secret"})
YOUTUBE_HOSTS = frozenset({"youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be", "www.youtu.be"})
VIDEO_FILE_EXTS = (".mp4", ".mkv", ".mov", ".avi", ".webm", ".ts", ".flv")
# Формат для yt-dlp: прямой HLS не выше 720p (живые трансляции отдают только HLS)
YTDLP_FORMAT = "best[height<=720][protocol^=m3u8]/best[height<=720]/best"
YTDLP_TIMEOUT_S = 40.0

_USERINFO_RE = re.compile(r"(?P<scheme>[a-zA-Z][a-zA-Z0-9+.-]*://)(?P<user>[^/@:\s]*):(?P<pw>[^/@\s]*)@")


class StreamResolveError(RuntimeError):
    """Адрес не удалось превратить в прямой поток (yt-dlp нет, трансляция не идёт и т. п.)."""


def mask_uri(text: str | None) -> str | None:
    """Пароль из rtsp://логин:пароль@… и значения секретных параметров запроса → ***.

    Работает и для адреса, и для произвольного текста с адресом внутри (сообщения об ошибках)."""
    if not text:
        return text
    masked = _USERINFO_RE.sub(lambda m: f"{m['scheme']}{m['user']}:***@", text)
    return re.sub(
        r"(?i)([?&](?:" + "|".join(sorted(SECRET_QUERY_KEYS)) + r")=)[^&\s#]+",
        r"\1***",
        masked,
    )


def scheme_of(uri: str) -> str:
    return urlsplit(uri.strip()).scheme.lower()


def is_network(uri: str) -> bool:
    return scheme_of(uri) in {"rtsp", "rtsps", "rtmp", "rtmps", "http", "https", "srt", "udp", "tcp"}


def is_youtube(uri: str) -> bool:
    parts = urlsplit(uri.strip())
    return parts.scheme.lower() in {"http", "https"} and (parts.hostname or "").lower() in YOUTUBE_HOSTS


def is_hls(uri: str) -> bool:
    parts = urlsplit(uri.strip())
    return parts.scheme.lower() in {"http", "https"} and (".m3u8" in parts.path.lower() or "m3u8" in parts.query.lower())


def is_http_file(uri: str) -> bool:
    parts = urlsplit(uri.strip())
    return parts.scheme.lower() in {"http", "https"} and parts.path.lower().endswith(VIDEO_FILE_EXTS)


def stream_kind(uri: str) -> str:
    """rtsp | hls | youtube | http-file | mjpeg | rtmp | srt | other — для статуса и подсказок."""
    scheme = scheme_of(uri)
    if scheme in {"rtsp", "rtsps"}:
        return "rtsp"
    if scheme in {"rtmp", "rtmps"}:
        return "rtmp"
    if scheme == "srt":
        return "srt"
    if is_youtube(uri):
        return "youtube"
    if is_hls(uri):
        return "hls"
    if is_http_file(uri):
        return "http-file"
    if scheme in {"http", "https"}:
        return "mjpeg"  # HTTP без расширения видеофайла и без .m3u8 — обычно MJPEG камеры или телефона
    return "other"


def paced_kind(kind: str) -> bool:
    """Потоки, которые приходят пачками (сегмент HLS, файл по HTTP): выдавать в темпе меток времени."""
    return kind in {"hls", "youtube", "http-file"}


def ytdlp_command() -> list[str] | None:
    exe = os.environ.get("GUARD_YTDLP") or shutil.which("yt-dlp")
    if exe:
        return [exe]
    try:
        import yt_dlp  # noqa: F401
    except ImportError:
        return None
    import sys

    return [sys.executable, "-m", "yt_dlp"]


def resolve_youtube(uri: str, timeout_s: float = YTDLP_TIMEOUT_S, run=subprocess.run) -> str:
    """Временный HLS-адрес трансляции YouTube (yt-dlp -g). Адрес живёт часы, поэтому его
    получают заново при каждом (пере)подключении, а в конфигурации камеры хранится ссылка на страницу."""
    cmd = ytdlp_command()
    if cmd is None:
        raise StreamResolveError("для ссылок YouTube нужен yt-dlp: brew install yt-dlp или pip install yt-dlp")
    try:
        proc = run(
            [*cmd, "-g", "--no-warnings", "--no-playlist", "-f", YTDLP_FORMAT, uri],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        raise StreamResolveError(f"yt-dlp не ответил за {timeout_s:.0f} с") from exc
    lines = [ln.strip() for ln in (proc.stdout or "").splitlines() if ln.strip().startswith("http")]
    if proc.returncode != 0 or not lines:
        err = (proc.stderr or "").strip().splitlines()
        reason = err[-1] if err else f"код {proc.returncode}"
        raise StreamResolveError(f"yt-dlp не получил адрес потока: {reason[:200]}")
    return lines[0]


def resolve_stream_uri(uri: str, timeout_s: float = YTDLP_TIMEOUT_S, run=subprocess.run) -> tuple[str, str | None]:
    """(адрес для cv2, чем разрешён). Ссылка YouTube → HLS через yt-dlp; остальные — как есть."""
    uri = uri.strip()
    if is_youtube(uri):
        return resolve_youtube(uri, timeout_s=timeout_s, run=run), "yt-dlp"
    return uri, None


def strip_secrets(uri: str) -> str:
    """Адрес без логина, пароля и секретных параметров — для логов и отчётов."""
    parts = urlsplit(uri)
    netloc = parts.hostname or ""
    if parts.port:
        netloc = f"{netloc}:{parts.port}"
    query = urlencode([(k, "***" if k.lower() in SECRET_QUERY_KEYS else v) for k, v in parse_qsl(parts.query, keep_blank_values=True)])
    return urlunsplit((parts.scheme, netloc, parts.path, query, ""))
