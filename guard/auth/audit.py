"""Журнал действий: кто, роль, когда, что, с какого адреса (docs/ROLES.md).

Запись делает обработчик после успешного действия. Сбой записи только логируется: действие, которое
уже произошло, не откатывается из-за журнала. Пароли, хэши и учётные данные в адресах камер
в журнал не попадают.
"""

from __future__ import annotations

import logging
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
ZONES_UPDATE = "zones.update"
RULES_UPDATE = "rules.update"
EVENT_UPDATE = "event.update"
USER_CREATE = "user.create"
USER_UPDATE = "user.update"
FLYWHEEL_REVIEW = "flywheel.review"
FLYWHEEL_JOB = "flywheel.job"

ACTIONS = (
    AUTH_LOGIN, AUTH_LOGIN_FAILED, AUTH_LOGOUT, CAMERA_CREATE, CAMERA_UPDATE, CAMERA_DELETE, ZONES_UPDATE,
    RULES_UPDATE, EVENT_UPDATE, USER_CREATE, USER_UPDATE, FLYWHEEL_REVIEW, FLYWHEEL_JOB,
)
NOTE_LIMIT = 500


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
    """rtsp://user:pass@host/x → rtsp://user:***@host/x; адреса без пароля не меняются."""
    try:
        parts = urlsplit(uri)
    except ValueError:
        return "***"
    if parts.password is None:
        return uri
    host = parts.hostname or ""
    if parts.port is not None:
        host = f"{host}:{parts.port}"
    netloc = f"{parts.username or ''}:***@{host}"
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


def source_details(source: Any) -> dict[str, Any]:
    """Источник камеры для журнала: тип и адрес без пароля."""
    return {"type": source.type, "uri": redact_uri(source.uri)}


def note(value: str | None) -> str | None:
    if value is None:
        return None
    return value if len(value) <= NOTE_LIMIT else value[:NOTE_LIMIT] + "…"
