"""Вход и сессии поверх AuthStore: кто делает запрос, вход с ограничением подбора, выход, конфиг для UI.

Один экземпляр на приложение (app.state.auth). База открывается лениво или при старте приложения;
при включённом входе пустая база получает демо-учётки.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from guard.auth.ratelimit import LoginLimiter
from guard.auth.roles import ADMIN, permissions_for, role_title, sorted_permissions
from guard.auth.store import DEMO_USERS, SESSION_TTL_S, AuthStore, UserRecord, normalize_username, token_hash

log = logging.getLogger(__name__)

LOGIN_MAX_FAILURES = 10
LOGIN_BLOCK_S = 60.0


@dataclass(frozen=True)
class CurrentUser:
    """Пользователь запроса. session_hash = None — вход выключен (GUARD_AUTH=0)."""

    username: str
    display_name: str
    role: str
    permissions: frozenset[str]
    session_hash: str | None = None

    def can(self, permission: str) -> bool:
        return permission in self.permissions

    def me(self) -> dict[str, Any]:
        return {
            "username": self.username,
            "display_name": self.display_name,
            "role": self.role,
            "role_title": role_title(self.role),
            "permissions": sorted_permissions(self.permissions),
        }


def auth_off_user() -> CurrentUser:
    """При выключенном входе все запросы идут как администратор."""
    return CurrentUser(ADMIN, role_title(ADMIN), ADMIN, permissions_for(ADMIN))


def _current(user: UserRecord, session_hash: str) -> CurrentUser:
    return CurrentUser(user.username, user.display_name, user.role, permissions_for(user.role), session_hash)


class InvalidCredentials(Exception):
    def __init__(self, locked_for_s: float = 0.0) -> None:
        super().__init__("invalid credentials")
        self.locked_for_s = locked_for_s


class LoginThrottled(Exception):
    def __init__(self, retry_after: int) -> None:
        super().__init__("too many attempts")
        self.retry_after = retry_after


class AuthService:
    def __init__(
        self,
        *,
        enabled: bool,
        demo_login: bool,
        db_path: Path,
        session_ttl_s: float = SESSION_TTL_S,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.enabled = enabled
        self.demo_login = demo_login
        self.db_path = Path(db_path)
        self.session_ttl_s = session_ttl_s
        self.clock = clock  # тесты подменяют часы; хранилище и ограничитель читают их через self
        self.limiter = LoginLimiter(LOGIN_MAX_FAILURES, LOGIN_BLOCK_S, clock=lambda: self.clock())
        self._store: AuthStore | None = None
        self._lock = threading.Lock()

    @classmethod
    def from_settings(cls, settings: Any) -> AuthService:
        return cls(
            enabled=bool(settings.auth),
            demo_login=bool(settings.demo_login),
            db_path=settings.auth_db_path,
            session_ttl_s=float(settings.session_ttl_s),
        )

    # ------------------------------------------------------------ база

    def open(self) -> list[str]:
        """Открыть базу; при включённом входе в пустую базу — демо-учётки. Вернуть созданные логины."""
        with self._lock:
            if self._store is not None:
                return []
            store = AuthStore(self.db_path, session_ttl_s=self.session_ttl_s, clock=lambda: self.clock())
            try:
                seeded = store.ensure_demo_users() if self.enabled else []
            except BaseException:
                store.close()
                raise
            self._store = store
        if seeded:
            log.warning(
                "auth: созданы демо-учётки %s с паролем, равным логину. Для реального объекта выключите "
                "GUARD_DEMO_LOGIN и смените пароли", ", ".join(seeded),
            )
        return seeded

    @property
    def store(self) -> AuthStore:
        if self._store is None:
            self.open()
        assert self._store is not None
        return self._store

    def close(self) -> None:
        with self._lock:
            if self._store is not None:
                self._store.close()
                self._store = None

    # ------------------------------------------------------------ сессии

    def resolve(self, token: str | None) -> CurrentUser | None:
        """Пользователь по токену cookie; None — нет живой сессии."""
        if not self.enabled:
            return auth_off_user()
        found = self.store.resolve_session(token)
        return None if found is None else _current(*found)

    def login(
        self, username: str, password: str, *, ip: str | None = None, user_agent: str | None = None
    ) -> tuple[CurrentUser, str]:
        """Проверить пароль и открыть сессию; вернуть пользователя и токен для cookie."""
        name = normalize_username(username)
        wait = self.limiter.retry_after(name, ip)
        if wait > 0:
            raise LoginThrottled(math.ceil(wait))
        user = self.store.authenticate(name, password)
        if user is None:
            raise InvalidCredentials(self.limiter.failure(name, ip))
        self.limiter.success(name, ip)
        token = self.store.create_session(user.username, ip=ip, user_agent=user_agent)
        self.store.mark_login(user.username)
        return _current(user, token_hash(token)), token

    def logout(self, user: CurrentUser) -> None:
        if user.session_hash:
            self.store.delete_session(user.session_hash)

    def session_alive(self, user: CurrentUser) -> bool:
        if user.session_hash is None:
            return True
        return self.store.session_alive(user.session_hash)

    # ------------------------------------------------------------ для интерфейса

    def config(self) -> dict[str, Any]:
        """GET /api/auth/config: demo_users — демо-учётки, у которых пароль ещё равен логину."""
        demo: list[dict[str, str]] = []
        if self.demo_login:
            users = {u.username: u for u in self.store.list_users() if u.demo and u.active}
            demo = [
                {"username": name, "display_name": users[name].display_name, "role": users[name].role}
                for name in DEMO_USERS
                if name in users
            ]
        return {"auth_enabled": self.enabled, "demo_login": self.demo_login, "demo_users": demo}
