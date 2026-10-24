"""Какие адреса источников принимает API: POST и PUT /api/cameras, POST /api/cameras/probe.

Описание — docs/ROLES.md, «Безопасность: что проверено».

Сервер сам открывает адрес камеры (FFmpeg через OpenCV), поэтому адрес должен соответствовать типу:

- rtsp (сетевой поток): только сетевые схемы — rtsp, rtsps, rtmp, rtmps, http, https, srt, udp, tcp.
  Не file://, не concat: и другие протоколы FFmpeg, не путь к файлу или устройству. Адреса облачных
  метаданных (169.254.169.254 и т. п.) запрещены: через них из облака утекают ключи;
- webcam: только номер устройства (0, 1, …) или /dev/videoN — строку OpenCV открыл бы как файл или URL;
- file и images: локальный путь без схемы URL, не в /dev, /proc и /sys.

Камеры в локальной сети и тестовый RTSP-сервер на 127.0.0.1 разрешены: это обычный случай. Имя хоста
не разрешается в адрес заранее (DNS может зависнуть), поэтому имя, указывающее на адрес метаданных, не ловится.

Модуль не импортирует guard.sources (там cv2): шлюз (GUARD_MODE=gateway) его не загружает. Список схем
совпадает с guard.sources.stream_url.is_network — это проверяет тест.
"""

from __future__ import annotations

import os
import re
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

from guard.contracts import SourceConfig

NETWORK_SCHEMES = frozenset({"rtsp", "rtsps", "rtmp", "rtmps", "http", "https", "srt", "udp", "tcp"})
METADATA_HOSTS = frozenset({
    "169.254.169.254",  # AWS, GCP, Azure, Яндекс Облако
    "169.254.170.2",  # AWS ECS
    "100.100.100.200",  # Alibaba
    "fd00:ec2::254",
    "metadata.google.internal",
    "metadata",
})
DEVICE_DIRS = ("/dev", "/proc", "/sys")
_WEBCAM_RE = re.compile(r"^(\d{1,3}|/dev/video\d{1,3})$")
_URL_SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]+:")  # одна буква — диск Windows (C:\)


def _under_device_dir(path: str) -> bool:
    parts = PurePosixPath(path).parts
    return len(parts) > 1 and parts[0] == "/" and f"/{parts[1]}" in DEVICE_DIRS


def _local_paths(uri: str, data_dir: Path | None) -> list[str]:
    """Путь как записан и после разрешения ссылок и «..» (относительный — от GUARD_DATA_DIR)."""
    path = Path(uri).expanduser()
    if not path.is_absolute():
        if data_dir is None:
            return [uri]
        path = Path(data_dir) / path
    return [str(path), os.path.normpath(path), os.path.realpath(path)]


def source_problem(source: SourceConfig, data_dir: Path | None = None) -> str | None:
    """Почему адрес нельзя открыть с сервера, или None. data_dir — база относительных путей file и images."""
    uri = (source.uri or "").strip()
    if not uri:
        return "адрес источника пуст"
    if source.type == "rtsp":
        try:
            parts = urlsplit(uri)
            host = (parts.hostname or "").lower().rstrip(".")
        except ValueError:
            return "адрес сетевого потока не разобран"
        scheme = parts.scheme.lower()
        if scheme not in NETWORK_SCHEMES:
            return (
                "адрес сетевого потока начинается с rtsp://, rtsps://, rtmp://, http://, https://, srt://, "
                "udp:// или tcp://; для файла выберите тип file, для веб-камеры — webcam"
            )
        if not host and scheme not in ("udp", "srt"):
            return "в адресе сетевого потока нет хоста"
        if host in METADATA_HOSTS:
            return "адрес служебных метаданных облака запрещён"
        return None
    if source.type == "webcam":
        if not _WEBCAM_RE.match(uri):
            return "для веб-камеры укажите номер устройства на сервере: 0, 1, … (или /dev/videoN)"
        return None
    # file, images
    if _URL_SCHEME_RE.match(uri):
        return "для сетевого адреса выберите тип rtsp; file и images — путь к файлу или папке на сервере"
    if any(_under_device_dir(p) for p in _local_paths(uri, data_dir)):
        return "путь к устройству или системному каталогу запрещён"
    return None
