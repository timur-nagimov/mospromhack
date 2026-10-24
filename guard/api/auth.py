"""Вход и роли: /api/auth/config, login, logout, me, roles (docs/ROLES.md, «API»)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from guard.auth import audit
from guard.auth.deps import SESSION_COOKIE, authenticated, public
from guard.auth.roles import PERMISSIONS, ROLES
from guard.auth.service import AuthService, CurrentUser, InvalidCredentials, LoginThrottled
from guard.auth.store import normalize_username

router = APIRouter(prefix="/api/auth", tags=["auth"])


class LoginIn(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=256)


def _service(request: Request) -> AuthService:
    return request.app.state.auth


@router.get("/config", dependencies=[public()])
def auth_config(request: Request) -> dict[str, Any]:
    """Для страницы входа: включён ли вход и демо-учётки для кнопок «Войти как …»."""
    return _service(request).config()


@router.post("/login", dependencies=[public()])
def login(body: LoginIn, request: Request, response: Response) -> Any:
    """Проверить пароль, поставить cookie guard_session и вернуть то же, что /me."""
    service = _service(request)
    name = normalize_username(body.username)
    try:
        user, token = service.login(
            name, body.password, ip=audit.client_ip(request), user_agent=request.headers.get("user-agent"),
        )
    except LoginThrottled as exc:
        return JSONResponse(
            {"detail": "too many attempts", "retry_after": exc.retry_after},
            status_code=429,
            headers={"Retry-After": str(exc.retry_after)},
        )
    except InvalidCredentials as exc:
        details: dict[str, Any] = {"reason": "invalid_credentials"}
        if exc.locked_for_s:
            details["locked_for_s"] = int(exc.locked_for_s)
        audit.record(request, audit.AUTH_LOGIN_FAILED, name, details, username=name or "-")
        raise HTTPException(401, "invalid credentials") from None
    audit.record(request, audit.AUTH_LOGIN, user.username, user=user)
    response.set_cookie(
        SESSION_COOKIE, token, httponly=True, samesite="lax", path="/", secure=request.url.scheme == "https",
    )
    return user.me()


@router.post("/logout")
def logout(request: Request, response: Response, user: CurrentUser = authenticated()) -> dict[str, Any]:
    _service(request).logout(user)
    if user.session_hash:
        audit.record(request, audit.AUTH_LOGOUT, user.username, user=user)
    response.delete_cookie(SESSION_COOKIE, path="/", httponly=True, samesite="lax")
    return {"ok": True}


@router.get("/me")
def me(user: CurrentUser = authenticated()) -> dict[str, Any]:
    return user.me()


@router.get("/roles")
def roles_catalog(_: CurrentUser = authenticated()) -> dict[str, Any]:
    """Роли и права с названиями и описаниями — для страницы «Пользователи» и матрицы ролей."""
    return {
        "roles": [role.to_dict() for role in ROLES.values()],
        "permissions": [perm.to_dict() for perm in PERMISSIONS.values()],
    }
