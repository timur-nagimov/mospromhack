"""Какие адреса источников принимает API: POST и PUT /api/cameras, POST /api/cameras/probe.

Описание — docs/ROLES.md, «Безопасность: что проверено».

Сервер сам открывает адрес камеры (FFmpeg через OpenCV), поэтому адрес должен соответствовать типу:

- rtsp (сетевой поток): только сетевые схемы — rtsp, rtsps, rtmp, rtmps, http, https, srt, udp, tcp.
  Не file://, не concat: и другие протоколы FFmpeg, не путь к файлу или устройству. Адреса облачных
  метаданных (вся link-local сеть 169.254.0.0/16 в любой записи адреса) запрещены: через них утекают ключи облака;
- webcam: только номер устройства (0, 1, …) или /dev/videoN — строку OpenCV открыл бы как файл или URL;
- file и images: локальный путь без схемы URL, не в /dev, /proc и /sys.

Камеры в локальной сети и тестовый RTSP-сервер на 127.0.0.1 разрешены: это обычный случай. Имя хоста
не разрешается в адрес заранее (DNS может зависнуть), поэтому имя, указывающее на адрес метаданных, не ловится.

Модуль не импортирует guard.sources (там cv2): шлюз (GUARD_MODE=gateway) его не загружает. Список схем
совпадает с guard.sources.stream_url.is_network — это проверяет тест.
"""

from __future__ import annotations

import ipaddress
import os
import re
import socket
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

from guard.contracts import SourceConfig

NETWORK_SCHEMES = frozenset({"rtsp", "rtsps", "rtmp", "rtmps", "http", "https", "srt", "udp", "tcp"})
METADATA_HOSTS = frozenset({"metadata.google.internal", "metadata.goog", "metadata", "instance-data"})
METADATA_IPS = (
    ipaddress.ip_network("169.254.0.0/16"),  # link-local: метаданные AWS, GCP, Azure, Яндекс Облака, ECS
    ipaddress.ip_network("fe80::/10"),
    ipaddress.ip_network("fd00:ec2::254/128"),
    ipaddress.ip_network("100.100.100.200/32"),  # Alibaba
)
DEVICE_DIRS = ("/dev", "/proc", "/sys")
_WEBCAM_RE = re.compile(r"^(\d{1,3}|/dev/video\d{1,3})$")
_URL_SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]+:")  # одна буква — диск Windows (C:\)


def _host_ip(host: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    """IP-адрес в любой записи, которую понимает FFmpeg: 169.254.169.254, 2852039166, 0xa9fea9fe, ::ffff:…"""
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        try:
            ip = ipaddress.ip_address(socket.inet_aton(host))  # десятичная, шестнадцатеричная, сокращённая запись
        except (OSError, ValueError):
            return None
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        return ip.ipv4_mapped
    return ip


def _metadata_host(host: str) -> bool:
    if host in METADATA_HOSTS:
        return True
    ip = _host_ip(host)
    return ip is not None and any(ip in net for net in METADATA_IPS if ip.version == net.version)


def _under_device_dir(path: str) -> bool:
    parts = PurePosixPath(path).parts
    return len(parts) > 1 and parts[0] == "/" and f"/{parts[1]}" in DEVICE_DIRS


def _local_paths(uri: str, data_dir: Path | None) -> list[str]:
    """Путь как записан и после разрешения ссылок и «..». Относительный — от тех же баз, что resolve_source_uri:
    GUARD_DATA_DIR, его родитель, текущий каталог."""
    path = Path(uri).expanduser()
    if path.is_absolute():
        candidates = [path]
    else:
        bases = ([Path(data_dir), Path(data_dir).parent] if data_dir is not None else []) + [Path.cwd()]
        candidates = [base / path for base in bases]
    out: list[str] = []
    for cand in candidates:
        out += [str(cand), os.path.normpath(cand), os.path.realpath(cand)]
    return out


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
        if _metadata_host(host):
            return "адрес служебных метаданных облака (link-local 169.254.0.0/16) запрещён"
        return None
    if source.type == "webcam":
        if not _WEBCAM_RE.match(uri):
            return "для веб-камеры укажите номер устройства на сервере: 0, 1, … (или /dev/videoN)"
        return None
    # file, images
    if _URL_SCHEME_RE.match(uri):
        return "для сетевого адреса выберите тип rtsp; file и images — путь к файлу или папке на сервере"
    try:
        paths = _local_paths(uri, data_dir)
    except (RuntimeError, ValueError, OSError):  # ~нет_такого, NUL, непарный суррогат
        return "путь к файлу не разобран"
    if any(_under_device_dir(p) for p in paths):
        return "путь к устройству или системному каталогу запрещён"
    return None
