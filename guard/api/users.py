"""Учётные записи: /api/users (право users.manage, docs/ROLES.md).

Последнего активного администратора нельзя понизить или отключить (409). Смена пароля и отключение
закрывают сессии пользователя; при смене своего пароля текущая сессия остаётся.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, field_validator, model_validator

from guard.auth import audit
from guard.auth.deps import require
from guard.auth.passwords import password_problem
from guard.auth.roles import ROLES, USERS_MANAGE
from guard.auth.service import CurrentUser
from guard.auth.store import USERNAME_RE, AuthStore, LastAdminError, UserExists, UserNotFound, normalize_username

router = APIRouter(prefix="/api/users", tags=["users"])

DISPLAY_NAME_MAX = 80


def _display_name(value: str) -> str:
    value = value.strip()
    if not value or len(value) > DISPLAY_NAME_MAX:
        raise ValueError(f"display_name: 1-{DISPLAY_NAME_MAX} symbols")
    return value


def _role(value: str) -> str:
    if value not in ROLES:
        raise ValueError(f"role must be one of {list(ROLES)}")
    return value


def _password(value: str) -> str:
    problem = password_problem(value)
    if problem:
        raise ValueError(problem)
    return value


class UserCreate(BaseModel):
    username: str
    display_name: str
    role: str
    password: str

    @field_validator("username")
    @classmethod
    def check_username(cls, value: str) -> str:
        name = normalize_username(value)
        if not USERNAME_RE.match(name):
            raise ValueError("username: 2-32 symbols a-z, 0-9, '_', '.', '-'")
        return name

    @field_validator("display_name")
    @classmethod
    def check_display_name(cls, value: str) -> str:
        return _display_name(value)

    @field_validator("role")
    @classmethod
    def check_role(cls, value: str) -> str:
        return _role(value)

    @field_validator("password")
    @classmethod
    def check_password(cls, value: str) -> str:
        return _password(value)


class UserPatch(BaseModel):
    display_name: str | None = None
    role: str | None = None
    password: str | None = None
    active: bool | None = None

    @field_validator("display_name")
    @classmethod
    def check_display_name(cls, value: str | None) -> str | None:
        return None if value is None else _display_name(value)

    @field_validator("role")
    @classmethod
    def check_role(cls, value: str | None) -> str | None:
        return None if value is None else _role(value)

    @field_validator("password")
    @classmethod
    def check_password(cls, value: str | None) -> str | None:
        return None if value is None else _password(value)

    @model_validator(mode="after")
    def check_not_empty(self) -> UserPatch:
        if all(getattr(self, name) is None for name in ("display_name", "role", "password", "active")):
            raise ValueError("nothing to update: display_name, role, password or active")
        return self


def _store(request: Request) -> AuthStore:
    return request.app.state.auth.store


@router.get("")
def list_users(request: Request, _: CurrentUser = require(USERS_MANAGE)) -> list[dict[str, Any]]:
    return [u.public() for u in _store(request).list_users()]


@router.post("", status_code=201)
def create_user(body: UserCreate, request: Request, user: CurrentUser = require(USERS_MANAGE)) -> dict[str, Any]:
    try:
        created = _store(request).create_user(body.username, body.display_name, body.role, body.password)
    except UserExists as exc:
        raise HTTPException(409, f"user {body.username} already exists") from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    audit.record(request, audit.USER_CREATE, created.username,
                 {"role": created.role, "display_name": created.display_name}, user=user)
    return created.public()


@router.patch("/{username}")
def update_user(
    username: str, body: UserPatch, request: Request, user: CurrentUser = require(USERS_MANAGE)
) -> dict[str, Any]:
    store = _store(request)
    before = store.get_user(username)
    if before is None:
        raise HTTPException(404, f"user {username} not found")
    try:
        updated = store.update_user(
            before.username, display_name=body.display_name, role=body.role, password=body.password,
            active=body.active,
        )
    except UserNotFound as exc:
        raise HTTPException(404, f"user {username} not found") from exc
    except LastAdminError as exc:
        raise HTTPException(409, "last active admin") from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    if body.password is not None or not updated.active:
        own = updated.username == user.username and updated.active
        store.delete_user_sessions(updated.username, keep_hash=user.session_hash if own else None)
    audit.record(request, audit.USER_UPDATE, updated.username, _changes(before, updated, body), user=user)
    return updated.public()


def _changes(before: Any, after: Any, body: UserPatch) -> dict[str, Any]:
    details: dict[str, Any] = {}
    for name in ("display_name", "role", "active"):
        old, new = getattr(before, name), getattr(after, name)
        if old != new:
            details[name] = {"from": old, "to": new}
    if body.password is not None:
        details["password_changed"] = True
    return details
