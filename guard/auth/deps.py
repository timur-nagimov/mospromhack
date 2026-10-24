"""Проверка входа и прав для FastAPI (docs/ROLES.md, «Как устроено»).

Все маршруты закрыты по умолчанию: create_app ставит gate зависимостью уровня приложения, и она
работает для каждого HTTP- и WS-маршрута, включая роутеры, подключённые позже. Маршрут объявляет доступ
одной из зависимостей:

- public()            — без входа: /health, /api/auth/config, /api/auth/login;
- authenticated()     — любой вошедший пользователь;
- require(*perms)     — все перечисленные права;
- require_any(*perms) — хотя бы одно из прав (PUT камеры: точное право обработчик выбирает сам).

Маршрут без объявления доступен только администратору (403 с permission "admin"); исключение —
FALLBACK_PERMISSIONS для известных маршрутов из других модулей. Тест tests/test_auth_matrix.py требует
объявления у каждого маршрута. Ответы: 401 {"detail": "auth required"},
403 {"detail": "forbidden", "permission": "<право>"}. WebSocket сначала принимается, потом закрывается
кодом 4401 или 4403 — иначе браузер не увидит код закрытия.
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.params import Depends as DependsParam
from fastapi.responses import JSONResponse
from starlette.requests import HTTPConnection
from starlette.websockets import WebSocket
from starlette.exceptions import WebSocketException

from guard.auth.roles import ADMIN, CAMERAS_MANAGE
from guard.auth.service import CurrentUser, auth_off_user

SESSION_COOKIE = "guard_session"
SESSION_RECHECK_S = 60.0  # долгие соединения (MJPEG, WebSocket) перепроверяют сессию с этим периодом
WS_AUTH_REQUIRED = 4401
WS_FORBIDDEN = 4403
ADMIN_ONLY = ADMIN  # permission в ответе 403 для маршрута без объявленного доступа
USER_SCOPE_KEY = "guard.user"

# Маршруты из модулей, которые объявлены без зависимостей guard.auth: (метод, шаблон пути) → право.
FALLBACK_PERMISSIONS: dict[tuple[str, str], str] = {
    ("POST", "/api/cameras/probe"): CAMERAS_MANAGE,  # проверка потока до добавления камеры (docs/ROLES.md)
    # справочник сотрудников и разбор инцидентов (guard/api/employees.py, docs/EMPLOYEES.md)
    ("GET", "/api/employees"): "employees.view",
    ("POST", "/api/employees"): "employees.manage",
    ("GET", "/api/employees/{employee_id}/report"): "employees.view",
    ("GET", "/api/events/{event_id}/employees"): "employees.view",
    ("PUT", "/api/events/{event_id}/employees"): "events.assign",
}


class AuthRequired(HTTPException):
    def __init__(self) -> None:
        super().__init__(401, "auth required")


class PermissionDenied(HTTPException):
    def __init__(self, permission: str) -> None:
        super().__init__(403, "forbidden")
        self.permission = permission


@dataclass(frozen=True)
class Policy:
    kind: str  # public | login | permissions | undeclared
    permissions: tuple[str, ...] = ()
    any_of: bool = False

    def describe(self) -> str:
        if self.kind == "public":
            return "без входа"
        if self.kind == "login":
            return "вход"
        if self.kind == "undeclared":
            return "только admin (не объявлено)"
        return (" или " if self.any_of else " и ").join(self.permissions)


PUBLIC = Policy("public")
LOGIN = Policy("login")
UNDECLARED = Policy("undeclared")


# ---------------------------------------------------------------- кто делает запрос


class _NoAuth:
    """Приложение собрано без create_app (тесты отдельных роутеров): вход выключен, как у Settings из кода."""

    enabled = False

    def resolve(self, token: str | None) -> CurrentUser:
        return auth_off_user()

    def session_alive(self, user: CurrentUser) -> bool:
        return True


NO_AUTH = _NoAuth()


def auth_service(conn: HTTPConnection) -> Any:
    return getattr(conn.app.state, "auth", None) or NO_AUTH


async def resolve_user(conn: HTTPConnection) -> CurrentUser | None:
    """Пользователь запроса по cookie guard_session (кэшируется в scope); None — нет живой сессии."""
    if USER_SCOPE_KEY in conn.scope:
        return conn.scope[USER_SCOPE_KEY]
    service = auth_service(conn)
    if not service.enabled:
        user = service.resolve(None)
    else:
        token = conn.cookies.get(SESSION_COOKIE)
        user = await run_in_threadpool(service.resolve, token) if token else None
    conn.scope[USER_SCOPE_KEY] = user
    return user


def request_user(conn: HTTPConnection) -> CurrentUser | None:
    """Уже определённый gate пользователь (для журнала действий)."""
    return conn.scope.get(USER_SCOPE_KEY)


async def _deny(conn: HTTPConnection, permission: str | None) -> None:
    """401 (permission None) или 403. Для WebSocket — принять и закрыть кодом 4401 или 4403."""
    if conn.scope["type"] == "websocket":
        assert isinstance(conn, WebSocket)
        await conn.accept()
        if permission is None:
            raise WebSocketException(WS_AUTH_REQUIRED, "auth required")
        raise WebSocketException(WS_FORBIDDEN, f"forbidden: {permission}")
    if permission is None:
        raise AuthRequired()
    raise PermissionDenied(permission)


async def _enforce(conn: HTTPConnection, policy: Policy) -> CurrentUser | None:
    if policy.kind == "public":
        return await resolve_user(conn)
    user = await resolve_user(conn)
    if user is None:
        await _deny(conn, None)
    assert user is not None
    if policy.kind == "permissions":
        if policy.any_of:
            if not any(user.can(p) for p in policy.permissions):
                await _deny(conn, policy.permissions[0])
        else:
            missing = next((p for p in policy.permissions if not user.can(p)), None)
            if missing is not None:
                await _deny(conn, missing)
    elif policy.kind == "undeclared" and user.role != ADMIN:
        await _deny(conn, ADMIN_ONLY)
    return user


# ---------------------------------------------------------------- объявления доступа


def _declaration(policy: Policy, name: str) -> Any:
    async def dependency(conn: HTTPConnection) -> CurrentUser | None:
        return await _enforce(conn, policy)

    dependency.__name__ = name
    dependency.__guard_policy__ = policy  # type: ignore[attr-defined]
    return dependency


_PUBLIC_DEP = _declaration(PUBLIC, "public_route")
_LOGIN_DEP = _declaration(LOGIN, "login_required")


def public() -> DependsParam:
    """Маршрут открыт без входа."""
    return Depends(_PUBLIC_DEP)


def authenticated() -> DependsParam:
    """Нужен вход, права не важны. Как значение параметра отдаёт CurrentUser."""
    return Depends(_LOGIN_DEP)


def require(*permissions: str) -> DependsParam:
    """Нужны все права. Как значение параметра отдаёт CurrentUser."""
    if not permissions:
        raise ValueError("require() needs at least one permission")
    return Depends(_declaration(Policy("permissions", tuple(permissions)), "require_" + "_".join(permissions)))


def require_any(*permissions: str) -> DependsParam:
    """Нужно хотя бы одно право; 403 называет первое. Как значение параметра отдаёт CurrentUser."""
    if not permissions:
        raise ValueError("require_any() needs at least one permission")
    policy = Policy("permissions", tuple(permissions), any_of=True)
    return Depends(_declaration(policy, "require_any_" + "_".join(permissions)))


def ensure_permission(user: CurrentUser, permission: str) -> None:
    """Проверка внутри обработчика (когда право зависит от тела запроса)."""
    if not user.can(permission):
        raise PermissionDenied(permission)


def _walk(dependant: Any) -> Iterable[Policy]:
    for sub in getattr(dependant, "dependencies", None) or ():
        policy = getattr(sub.call, "__guard_policy__", None)
        if policy is not None:
            yield policy
        yield from _walk(sub)


_UNSET = object()


def declared_policy(route: Any) -> Policy | None:
    """Доступ, объявленный зависимостями маршрута (кэш на объекте маршрута); None — не объявлен."""
    if route is None:
        return None
    cached = getattr(route, "_guard_declared", _UNSET)
    if cached is not _UNSET:
        return cached
    policies = list(_walk(getattr(route, "dependant", None)))
    restrictive = [p for p in policies if p.kind != "public"]
    found = restrictive[0] if restrictive else (PUBLIC if policies else None)
    try:
        route._guard_declared = found
    except AttributeError:
        pass
    return found


def route_policy(route: Any, method: str) -> Policy:
    """Доступ к маршруту: объявленный, из FALLBACK_PERMISSIONS или UNDECLARED (только admin)."""
    declared = declared_policy(route)
    if declared is not None:
        return declared
    permission = FALLBACK_PERMISSIONS.get((method, getattr(route, "path_format", "")))
    return Policy("permissions", (permission,)) if permission else UNDECLARED


# ---------------------------------------------------------------- точки подключения


async def gate(conn: HTTPConnection) -> None:
    """Зависимость уровня приложения: открытые маршруты пропускает, остальным нужен вход.

    Объявленные права проверяет зависимость самого маршрута. Маршрут без объявления gate пускает
    по FALLBACK_PERMISSIONS, иначе только администратора.
    """
    route = conn.scope.get("route")
    declared = declared_policy(route)
    if declared is not None:
        if declared.kind != "public":
            await _enforce(conn, LOGIN)
        return
    method = conn.scope.get("method", "GET") if conn.scope["type"] == "http" else "WS"
    await _enforce(conn, route_policy(route, method))


async def current_user(conn: HTTPConnection) -> CurrentUser:
    user = await resolve_user(conn)
    if user is None:
        await _deny(conn, None)
    assert user is not None
    return user


async def _auth_required(_: Request, exc: AuthRequired) -> JSONResponse:
    return JSONResponse({"detail": "auth required"}, status_code=401)


async def _forbidden(_: Request, exc: PermissionDenied) -> JSONResponse:
    return JSONResponse({"detail": "forbidden", "permission": exc.permission}, status_code=403)


def install(app: FastAPI) -> None:
    """Ответы 401 и 403 в формате контракта."""
    app.add_exception_handler(AuthRequired, _auth_required)  # type: ignore[arg-type]
    app.add_exception_handler(PermissionDenied, _forbidden)  # type: ignore[arg-type]


class SessionWatch:
    """Долгое соединение раз в SESSION_RECHECK_S проверяет, что сессия не закрыта (выход, отключение)."""

    def __init__(self, conn: HTTPConnection, user: CurrentUser) -> None:
        self.service = auth_service(conn)
        self.user = user
        self._next = time.monotonic() + SESSION_RECHECK_S

    async def alive(self) -> bool:
        if self.user.session_hash is None:
            return True
        now = time.monotonic()
        if now < self._next:
            return True
        self._next = now + SESSION_RECHECK_S
        return await run_in_threadpool(self.service.session_alive, self.user)
