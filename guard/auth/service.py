"""Вход и сессии поверх AuthStore: кто делает запрос, вход с ограничением подбора, выход, конфиг для UI.

Один экземпляр на приложение (app.state.auth). База открывается лениво или при старте приложения.
Пустая база при включённом входе получает демо-учётки (GUARD_DEMO_LOGIN=1) или одного admin со
случайным паролем в файле INITIAL_ADMIN_FILE рядом с базой (GUARD_DEMO_LOGIN=0). При GUARD_DEMO_LOGIN=0
демо-учётки, у которых пароль ещё равен логину, не входят, и их сессии не действуют.
"""

from __future__ import annotations

import logging
import math
import os
import secrets
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from guard.auth.ratelimit import LoginLimiter
from guard.auth.roles import ADMIN, permissions_for, role_title, sorted_permissions
from guard.auth.store import (
    DEMO_USERS, SESSION_TTL_S, AuthStore, UserRecord, normalize_username, token_hash,
)

log = logging.getLogger(__name__)

LOGIN_MAX_FAILURES = 10
LOGIN_IP_MAX_FAILURES = 30  # неудачи с одного адреса по всем логинам (перебор логинов)
LOGIN_BLOCK_S = 60.0
LOGIN_WINDOW_S = 900.0  # неудачи старше 15 минут без новых не считаются
VERIFY_CONCURRENCY = 4  # одновременных проверок пароля: scrypt берёт 16 МБ, поток входов не съест память
INITIAL_ADMIN_FILE = "auth-initial-admin.txt"


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
        self.limiter = LoginLimiter(
            LOGIN_MAX_FAILURES, LOGIN_BLOCK_S, clock=lambda: self.clock(), ip_max_failures=LOGIN_IP_MAX_FAILURES,
            window_s=LOGIN_WINDOW_S,
        )
        self._store: AuthStore | None = None
        self._lock = threading.Lock()
        self._verify_slots = threading.BoundedSemaphore(VERIFY_CONCURRENCY)

    @classmethod
    def from_settings(cls, settings: Any) -> AuthService:
        return cls(
            enabled=bool(settings.auth),
            demo_login=bool(settings.demo_login),
            db_path=settings.auth_db_path,
            session_ttl_s=float(settings.session_ttl_s),
        )

    # ------------------------------------------------------------ база

    @property
    def initial_admin_file(self) -> Path:
        return self.db_path.parent / INITIAL_ADMIN_FILE

    def open(self) -> list[str]:
        """Открыть базу; при включённом входе заполнить пустую базу. Вернуть созданные логины.

        GUARD_DEMO_LOGIN=1 — демо-учётки с паролем, равным логину; GUARD_DEMO_LOGIN=0 — только admin
        со случайным паролем в файле initial_admin_file (права 600; в лог пишется путь, не пароль)."""
        with self._lock:
            if self._store is not None:
                return []
            store = AuthStore(self.db_path, session_ttl_s=self.session_ttl_s, clock=lambda: self.clock())
            try:
                if not self.enabled:
                    seeded = []
                elif self.demo_login:
                    seeded = store.ensure_demo_users()
                else:
                    seeded = self._seed_initial_admin(store)
            except BaseException:
                store.close()
                raise
            self._store = store
        if seeded and self.demo_login:
            log.warning(
                "auth: созданы демо-учётки %s с паролем, равным логину. Для реального объекта выключите "
                "GUARD_DEMO_LOGIN и смените пароли", ", ".join(seeded),
            )
        elif seeded:
            log.warning(
                "auth: база входа была пустой — создан admin, пароль в файле %s. Войдите, смените пароль "
                "(python -m guard.auth set-password admin) и удалите файл", self.initial_admin_file,
            )
        return seeded

    def _seed_initial_admin(self, store: AuthStore) -> list[str]:
        if store.count_users():
            return []
        password = secrets.token_urlsafe(12)
        path = self.initial_admin_file
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(
                "# Первый вход при GUARD_DEMO_LOGIN=0: логин и пароль администратора.\n"
                "# Смените пароль (страница «Пользователи» или python -m guard.auth set-password admin)"
                " и удалите этот файл.\n"
                f"{ADMIN} {password}\n"
            )
        os.chmod(path, 0o600)  # файл мог существовать с другими правами
        if not store.ensure_initial_admin(password):  # базу успел заполнить другой процесс
            path.unlink(missing_ok=True)
            return []
        return [ADMIN]

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
        if found is None or self._demo_locked(found[0]):
            return None
        return _current(*found)

    def _demo_locked(self, user: UserRecord) -> bool:
        """Демо-учётка с паролем, равным логину, при выключенном демо-входе не действует."""
        return user.demo and not self.demo_login

    def login(
        self, username: str, password: str, *, ip: str | None = None, user_agent: str | None = None
    ) -> tuple[CurrentUser, str]:
        """Проверить пароль и открыть сессию; вернуть пользователя и токен для cookie."""
        name = normalize_username(username)
        wait = self.limiter.retry_after(name, ip)
        if wait > 0:
            raise LoginThrottled(math.ceil(wait))
        with self._verify_slots:
            user = self.store.authenticate(name, password)
        if user is not None and self._demo_locked(user):
            log.warning(
                "auth: вход %s отклонён — пароль демо-учётки равен логину, а GUARD_DEMO_LOGIN=0. "
                "Смените пароль: python -m guard.auth set-password %s", user.username, user.username,
            )
            user = None
        if user is None:
            raise InvalidCredentials(self.limiter.failure(name, ip))
        self.limiter.success(name, ip)
        token = self.store.create_session(user.username, ip=ip, user_agent=user_agent)
        self.store.mark_login(user.username)
        return _current(user, token_hash(token)), token

    def logout(self, user: CurrentUser) -> None:
        if user.session_hash:
            self.store.delete_session(user.session_hash)

    def drop_token(self, token: str | None) -> None:
        """Закрыть сессию по токену cookie (повторный вход в том же браузере), если она есть."""
        if self.enabled and token and len(token) <= 256:
            self.store.delete_session(token_hash(token))

    def session_alive(self, user: CurrentUser) -> bool:
        """Для долгих соединений: сессия жива, и роль не менялась с момента открытия."""
        if user.session_hash is None:
            return True
        return self.store.session_alive(user.session_hash, role=user.role, allow_demo=self.demo_login)

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
