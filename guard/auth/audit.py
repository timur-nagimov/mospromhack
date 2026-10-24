"""Журнал действий: кто, роль, когда, что, с какого адреса (docs/ROLES.md).

Запись делает обработчик после успешного действия. Сбой записи только логируется: действие, которое
уже произошло, не откатывается из-за журнала. Пароли, хэши и учётные данные в адресах камер
в журнал не попадают.
"""

from __future__ import annotations

import logging
import re
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from starlette.requests import HTTPConnection

from guard.auth.deps import request_user
from guard.auth.service import CurrentUser

log = logging.getLogger(__name__)

AUTH_LOGIN = "auth.login"
AUTH_LOGIN_FAILED = "auth.login_failed"
AUTH_LOGOUT = "auth.logout"
CAMERA_CREATE = "camera.create"
CAMERA_UPDATE = "camera.update"
CAMERA_DELETE = "camera.delete"
CAMERA_PROBE = "camera.probe"
ZONES_UPDATE = "zones.update"
RULES_UPDATE = "rules.update"
EVENT_UPDATE = "event.update"
EVENT_ASSIGN = "event.assign"
EMPLOYEE_CREATE = "employee.create"
USER_CREATE = "user.create"
USER_UPDATE = "user.update"
FLYWHEEL_REVIEW = "flywheel.review"
FLYWHEEL_CAPTURE = "flywheel.capture"
FLYWHEEL_JOB = "flywheel.job"

ACTIONS = (
    AUTH_LOGIN, AUTH_LOGIN_FAILED, AUTH_LOGOUT, CAMERA_CREATE, CAMERA_UPDATE, CAMERA_DELETE, CAMERA_PROBE,
    ZONES_UPDATE, RULES_UPDATE, EVENT_UPDATE, EVENT_ASSIGN, EMPLOYEE_CREATE, USER_CREATE, USER_UPDATE,
    FLYWHEEL_REVIEW, FLYWHEEL_CAPTURE, FLYWHEEL_JOB,
)
NOTE_LIMIT = 500
# Значения этих параметров адреса — секреты; список как guard.sources.stream_url.SECRET_QUERY_KEYS
# (модуль не импортируется: шлюз не загружает guard.sources и cv2; совпадение проверяет тест)
SECRET_QUERY_KEYS = frozenset({
    "token", "key", "apikey", "api_key", "password", "pass", "pwd", "auth", "sig", "signature", "secret",
})
_SECRET_QUERY_RE = re.compile(r"(?i)([?&](?:" + "|".join(sorted(SECRET_QUERY_KEYS)) + r")=)[^&\s#]+")


def client_ip(conn: HTTPConnection) -> str | None:
    return conn.client.host if conn.client else None


def record(
    conn: HTTPConnection,
    action: str,
    target: str | None = None,
    details: dict[str, Any] | None = None,
    *,
    user: CurrentUser | None = None,
    username: str | None = None,
    role: str | None = None,
) -> None:
    """Записать действие. Кто — явный user, иначе пользователь запроса; username и role перекрывают."""
    service = getattr(conn.app.state, "auth", None)
    if service is None:  # приложение собрано без create_app: журнала действий нет
        return
    actor = user or request_user(conn)
    try:
        service.store.add_audit(
            action,
            username=username or (actor.username if actor else None),
            role=role or (actor.role if actor else None),
            target=target,
            details=details,
            ip=client_ip(conn),
        )
    except Exception:  # noqa: BLE001 — журнал не должен ломать действие
        log.exception("audit write failed: %s %s", action, target)


def redact_uri(uri: str) -> str:
    """rtsp://user:pass@host/x → rtsp://user:***@host/x, ?token=abc → ?token=***; прочие адреса не меняются."""
    try:
        parts = urlsplit(uri)
    except ValueError:
        return "***"
    if "@" in parts.netloc:
        userinfo, _, host = parts.netloc.rpartition("@")
        if ":" in userinfo:
            netloc = f"{userinfo.split(':', 1)[0]}:***@{host}"
            uri = urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))
    return _SECRET_QUERY_RE.sub(r"\1***", uri)


def source_details(source: Any) -> dict[str, Any]:
    """Источник камеры для журнала: тип и адрес без пароля."""
    return {"type": source.type, "uri": redact_uri(source.uri)}


def note(value: str | None) -> str | None:
    if value is None:
        return None
    return value if len(value) <= NOTE_LIMIT else value[:NOTE_LIMIT] + "…"
