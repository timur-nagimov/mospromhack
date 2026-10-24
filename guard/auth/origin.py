"""Защита от подделки запросов с чужого сайта (CSRF) и чтения WebSocket чужой страницей (docs/ROLES.md).

Cookie с SameSite=Lax браузер не шлёт в запросах с другого сайта, но шлёт со страницы того же хоста
на другом порту, а при выключенном входе (GUARD_AUTH=0) cookie не нужна вовсе. Поэтому изменяющие
запросы (POST, PUT, PATCH, DELETE) и WebSocket с заголовком Origin пропускаются, только если источник:

- сам сервер: Origin совпадает с Host или X-Forwarded-Host (интерфейс с того же адреса, прокси);
- указан в GUARD_CORS_ORIGINS («*» там отключает эту проверку);
- локальный (localhost, 127.0.0.1, ::1), и запрос тоже пришёл на локальный адрес: dev-сервер интерфейса
  на любом порту с прокси.

Запросы без Origin (curl, скрипты, edge) проходят как раньше: браузер ставит Origin в каждый POST,
PUT, PATCH, DELETE и в рукопожатие WebSocket. Отказ — 403 {"detail": "cross-origin request blocked"},
WebSocket закрывается кодом 4403.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any
from urllib.parse import urlsplit

from starlette.datastructures import Headers
from starlette.responses import JSONResponse

SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
LOOPBACK = frozenset({"localhost", "127.0.0.1", "::1"})
WS_FORBIDDEN = 4403
DETAIL = "cross-origin request blocked"
_DEFAULT_PORTS = {"http": 80, "https": 443, "ws": 80, "wss": 443}


def _host_port(value: str, scheme: str) -> tuple[str, int | None]:
    """«host[:port]» → (host в нижнем регистре, порт или порт схемы по умолчанию)."""
    try:
        parts = urlsplit(f"//{value.strip()}")
        port = parts.port
    except ValueError:
        return "", None
    return (parts.hostname or "").lower(), port if port is not None else _DEFAULT_PORTS.get(scheme)


def _origin(value: str) -> tuple[str, str, int | None] | None:
    try:
        parts = urlsplit(value.strip())
        port = parts.port
    except ValueError:
        return None
    scheme = parts.scheme.lower()
    if scheme not in ("http", "https") or not parts.hostname:
        return None
    return scheme, parts.hostname.lower(), port if port is not None else _DEFAULT_PORTS[scheme]


def _normalize(origin: str) -> str:
    return origin.strip().rstrip("/").lower()


class OriginGuard:
    """ASGI-прослойка: изменяющие запросы и WebSocket с чужим Origin не доходят до маршрутов."""

    def __init__(self, app: Any, allowed_origins: Iterable[str] = ()) -> None:
        self.app = app
        origins = list(allowed_origins)
        self.allow_all = "*" in origins
        self.allowed = {_normalize(o) for o in origins if o != "*"}

    def allowed_origin(self, origin: str, headers: Headers) -> bool:
        if self.allow_all or _normalize(origin) in self.allowed:
            return True
        parsed = _origin(origin)  # «null» (песочница, file://) и мусор — чужие
        if parsed is None:
            return False
        scheme, host, port = parsed
        request_hosts = [headers.get("host", "")]
        forwarded = headers.get("x-forwarded-host")
        if forwarded:
            request_hosts.append(forwarded.split(",")[0])
        for value in request_hosts:
            if not value:
                continue
            req_host, req_port = _host_port(value, scheme)
            if req_host == host and req_port == port:
                return True
            if host in LOOPBACK and req_host in LOOPBACK:
                return True
        return False

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        kind = scope["type"]
        if kind == "websocket" or (kind == "http" and scope["method"] not in SAFE_METHODS):
            headers = Headers(scope=scope)
            origin = headers.get("origin")
            if origin is not None and not self.allowed_origin(origin, headers):
                if kind == "websocket":
                    message = await receive()  # websocket.connect
                    if message["type"] == "websocket.connect":
                        await send({"type": "websocket.close", "code": WS_FORBIDDEN, "reason": DETAIL})
                    return
                await JSONResponse({"detail": DETAIL}, status_code=403)(scope, receive, send)
                return
        await self.app(scope, receive, send)
