"""SQLite входа (GUARD_AUTH_DB, по умолчанию <data_dir>/auth.db): users, sessions, audit_log.

Одно соединение на экземпляр под блокировкой, WAL. Изменения с проверкой «прочитал — записал»
идут в BEGIN IMMEDIATE: это защищает и от второго процесса (например, python -m guard.auth set-password).
В sessions лежит только SHA-256 токена: утечка базы не даёт войти по чужой сессии.
"""

from __future__ import annotations

import hashlib
import json
import re
import secrets
import sqlite3
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from guard.auth.passwords import burn, hash_password, verify_password
from guard.auth.roles import ADMIN, ROLES, role_title

SESSION_TTL_S = 12 * 3600.0  # смена
TOUCH_INTERVAL_S = 60.0  # сдвиг срока сессии пишем не чаще раза в минуту
DEMO_USERS = ("worker", "operator", "safety", "admin")  # логин = роль = пароль
USERNAME_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]{1,31}$")
_MAX_TOKEN_LEN = 256

_SCHEMA_VERSION = 1
_SCHEMA = (
    """CREATE TABLE IF NOT EXISTS users (
        username TEXT PRIMARY KEY NOT NULL,
        display_name TEXT NOT NULL,
        role TEXT NOT NULL,
        password_hash TEXT NOT NULL,
        active INTEGER NOT NULL DEFAULT 1,
        demo INTEGER NOT NULL DEFAULT 0,
        created_at REAL NOT NULL,
        updated_at REAL NOT NULL,
        last_login_at REAL
    )""",
    """CREATE TABLE IF NOT EXISTS sessions (
        token_hash TEXT PRIMARY KEY NOT NULL,
        username TEXT NOT NULL REFERENCES users(username) ON DELETE CASCADE,
        created_at REAL NOT NULL,
        last_seen_at REAL NOT NULL,
        expires_at REAL NOT NULL,
        ip TEXT,
        user_agent TEXT
    )""",
    "CREATE INDEX IF NOT EXISTS idx_sessions_username ON sessions(username)",
    "CREATE INDEX IF NOT EXISTS idx_sessions_expires ON sessions(expires_at)",
    """CREATE TABLE IF NOT EXISTS audit_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts REAL NOT NULL,
        username TEXT,
        role TEXT,
        action TEXT NOT NULL,
        target TEXT,
        details TEXT NOT NULL DEFAULT '{}',
        ip TEXT
    )""",
    "CREATE INDEX IF NOT EXISTS idx_audit_ts ON audit_log(ts)",
    "CREATE INDEX IF NOT EXISTS idx_audit_username_ts ON audit_log(username, ts)",
    "CREATE INDEX IF NOT EXISTS idx_audit_action_ts ON audit_log(action, ts)",
)


class AuthStoreError(Exception):
    pass


class UserExists(AuthStoreError):
    pass


class UserNotFound(AuthStoreError):
    pass


class LastAdminError(AuthStoreError):
    """Нельзя понизить или отключить последнего активного администратора."""


@dataclass(frozen=True)
class UserRecord:
    username: str
    display_name: str
    role: str
    active: bool
    demo: bool  # демо-учётка с паролем, равным логину (сбрасывается при смене пароля)
    created_at: float
    updated_at: float
    last_login_at: float | None
    password_hash: str = field(repr=False)

    def public(self) -> dict[str, Any]:
        """Поля для API: без хэша пароля."""
        return {
            "username": self.username,
            "display_name": self.display_name,
            "role": self.role,
            "active": self.active,
            "created_at": self.created_at,
            "last_login_at": self.last_login_at,
        }


def normalize_username(username: str) -> str:
    return (username or "").strip().lower()


def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _user(row: sqlite3.Row) -> UserRecord:
    return UserRecord(
        username=row["username"],
        display_name=row["display_name"],
        role=row["role"],
        active=bool(row["active"]),
        demo=bool(row["demo"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        last_login_at=row["last_login_at"],
        password_hash=row["password_hash"],
    )


def _audit_item(row: sqlite3.Row) -> dict[str, Any]:
    item = dict(row)
    try:
        item["details"] = json.loads(item["details"] or "{}")
    except ValueError:
        item["details"] = {}
    return item


def _like_prefix(prefix: str) -> str:
    return prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"


class AuthStore:
    def __init__(
        self,
        path: str | Path,
        *,
        session_ttl_s: float = SESSION_TTL_S,
        touch_interval_s: float = TOUCH_INTERVAL_S,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.session_ttl_s = session_ttl_s
        self.touch_interval_s = touch_interval_s
        self.clock = clock
        self._lock = threading.RLock()
        self._closed = False
        self._conn = sqlite3.connect(str(self.path), timeout=30.0, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        try:
            self._conn.execute("PRAGMA journal_mode = WAL")
            self._conn.execute("PRAGMA synchronous = NORMAL")
            self._conn.execute("PRAGMA foreign_keys = ON")
            with self._conn:
                self._conn.execute("BEGIN IMMEDIATE")
                version = self._conn.execute("PRAGMA user_version").fetchone()[0]
                if version not in (0, _SCHEMA_VERSION):
                    raise ValueError(f"Unsupported auth schema version: {version}")
                if version == 0:
                    for statement in _SCHEMA:
                        self._conn.execute(statement)
                    self._conn.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
        except BaseException:
            self._conn.close()
            raise

    def __enter__(self) -> AuthStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._conn.close()
                self._closed = True

    # ------------------------------------------------------------ пользователи

    def count_users(self) -> int:
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]

    def count_active_admins(self) -> int:
        with self._lock:
            return self._count_active_admins()

    def _count_active_admins(self) -> int:
        return self._conn.execute(
            "SELECT COUNT(*) FROM users WHERE role = ? AND active = 1", (ADMIN,)
        ).fetchone()[0]

    def ensure_demo_users(self) -> list[str]:
        """В пустую базу — демо-учётки worker, operator, safety, admin с паролем, равным логину."""
        if self.count_users():
            return []
        hashes = {name: hash_password(name) for name in DEMO_USERS}  # scrypt — вне блокировки
        now = self.clock()
        with self._lock, self._conn:
            self._conn.execute("BEGIN IMMEDIATE")
            if self._conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]:
                return []
            for name in DEMO_USERS:
                self._conn.execute(
                    "INSERT INTO users (username, display_name, role, password_hash, active, demo, created_at, updated_at)"
                    " VALUES (?, ?, ?, ?, 1, 1, ?, ?)",
                    (name, role_title(name), name, hashes[name], now, now),
                )
        return list(DEMO_USERS)

    def create_user(
        self, username: str, display_name: str, role: str, password: str, *, demo: bool = False
    ) -> UserRecord:
        name = normalize_username(username)
        if not USERNAME_RE.match(name):
            raise ValueError("username: 2-32 symbols a-z, 0-9, '_', '.', '-'")
        if role not in ROLES:
            raise ValueError(f"unknown role: {role}")
        display = (display_name or "").strip() or name
        password_hash = hash_password(password)
        now = self.clock()
        try:
            with self._lock, self._conn:
                self._conn.execute(
                    "INSERT INTO users (username, display_name, role, password_hash, active, demo, created_at, updated_at)"
                    " VALUES (?, ?, ?, ?, 1, ?, ?, ?)",
                    (name, display, role, password_hash, int(demo), now, now),
                )
        except sqlite3.IntegrityError as exc:
            raise UserExists(name) from exc
        user = self.get_user(name)
        assert user is not None
        return user

    def get_user(self, username: str) -> UserRecord | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM users WHERE username = ?", (normalize_username(username),)
            ).fetchone()
        return _user(row) if row else None

    def list_users(self) -> list[UserRecord]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM users ORDER BY username").fetchall()
        return [_user(r) for r in rows]

    def update_user(
        self,
        username: str,
        *,
        display_name: str | None = None,
        role: str | None = None,
        password: str | None = None,
        active: bool | None = None,
    ) -> UserRecord:
        """Изменить поля; None — не менять. Последнего активного администратора не понизить и не отключить."""
        name = normalize_username(username)
        if role is not None and role not in ROLES:
            raise ValueError(f"unknown role: {role}")
        password_hash = hash_password(password) if password is not None else None
        now = self.clock()
        with self._lock, self._conn:
            self._conn.execute("BEGIN IMMEDIATE")
            row = self._conn.execute("SELECT * FROM users WHERE username = ?", (name,)).fetchone()
            if row is None:
                raise UserNotFound(name)
            current = _user(row)
            new_role = role if role is not None else current.role
            new_active = active if active is not None else current.active
            loses_admin = current.role == ADMIN and current.active and (new_role != ADMIN or not new_active)
            if loses_admin and self._count_active_admins() <= 1:
                raise LastAdminError(name)
            self._conn.execute(
                "UPDATE users SET display_name = ?, role = ?, active = ?, password_hash = ?, demo = ?, updated_at = ?"
                " WHERE username = ?",
                (
                    (display_name.strip() or current.display_name) if display_name is not None else current.display_name,
                    new_role,
                    int(new_active),
                    password_hash or current.password_hash,
                    0 if password_hash else int(current.demo),
                    now,
                    name,
                ),
            )
            row = self._conn.execute("SELECT * FROM users WHERE username = ?", (name,)).fetchone()
        return _user(row)

    def mark_login(self, username: str) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE users SET last_login_at = ? WHERE username = ?", (self.clock(), normalize_username(username))
            )

    def authenticate(self, username: str, password: str) -> UserRecord | None:
        """Пользователь при верном пароле и активной учётке, иначе None. Время ответа не выдаёт, есть ли логин."""
        user = self.get_user(username)
        if user is None:
            burn(password)
            return None
        if not verify_password(password, user.password_hash) or not user.active:
            return None
        return user

    # ------------------------------------------------------------ сессии

    def create_session(self, username: str, *, ip: str | None = None, user_agent: str | None = None) -> str:
        """Новая сессия; вернуть токен для cookie. Попутно удаляются просроченные сессии."""
        token = secrets.token_urlsafe(32)
        now = self.clock()
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM sessions WHERE expires_at <= ?", (now,))
            self._conn.execute(
                "INSERT INTO sessions (token_hash, username, created_at, last_seen_at, expires_at, ip, user_agent)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    token_hash(token),
                    normalize_username(username),
                    now,
                    now,
                    now + self.session_ttl_s,
                    ip,
                    (user_agent or "")[:200] or None,
                ),
            )
        return token

    def resolve_session(self, token: str | None) -> tuple[UserRecord, str] | None:
        """(пользователь, SHA-256 токена) для живой сессии, иначе None. Активность сдвигает срок на 12 ч."""
        if not token or len(token) > _MAX_TOKEN_LEN:
            return None
        digest = token_hash(token)
        now = self.clock()
        with self._lock, self._conn:
            row = self._conn.execute(
                "SELECT s.expires_at AS s_expires_at, s.last_seen_at AS s_last_seen_at, u.*"
                " FROM sessions s JOIN users u ON u.username = s.username WHERE s.token_hash = ?",
                (digest,),
            ).fetchone()
            if row is None:
                return None
            if row["s_expires_at"] <= now or not row["active"]:
                self._conn.execute("DELETE FROM sessions WHERE token_hash = ?", (digest,))
                return None
            if now - row["s_last_seen_at"] >= self.touch_interval_s:
                self._conn.execute(
                    "UPDATE sessions SET last_seen_at = ?, expires_at = ? WHERE token_hash = ?",
                    (now, now + self.session_ttl_s, digest),
                )
        return _user(row), digest

    def session_alive(self, session_hash: str) -> bool:
        """Сессия ещё действует (для долгих соединений: MJPEG, WebSocket)."""
        with self._lock:
            row = self._conn.execute(
                "SELECT s.expires_at, u.active FROM sessions s JOIN users u ON u.username = s.username"
                " WHERE s.token_hash = ?",
                (session_hash,),
            ).fetchone()
        return bool(row) and row[0] > self.clock() and bool(row[1])

    def delete_session(self, session_hash: str) -> None:
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM sessions WHERE token_hash = ?", (session_hash,))

    def delete_user_sessions(self, username: str, *, keep_hash: str | None = None) -> int:
        """Закрыть сессии пользователя, кроме keep_hash; вернуть число закрытых."""
        with self._lock, self._conn:
            cur = self._conn.execute(
                "DELETE FROM sessions WHERE username = ? AND token_hash != ?",
                (normalize_username(username), keep_hash or ""),
            )
            return cur.rowcount

    # ------------------------------------------------------------ журнал действий

    def add_audit(
        self,
        action: str,
        *,
        username: str | None = None,
        role: str | None = None,
        target: str | None = None,
        details: dict[str, Any] | None = None,
        ip: str | None = None,
        ts: float | None = None,
    ) -> int:
        payload = json.dumps(details or {}, ensure_ascii=False, default=str)
        with self._lock, self._conn:
            cur = self._conn.execute(
                "INSERT INTO audit_log (ts, username, role, action, target, details, ip) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (self.clock() if ts is None else ts, username, role, action, target, payload, ip),
            )
            return int(cur.lastrowid)

    def query_audit(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
        username: str | None = None,
        action: str | None = None,
        ts_from: float | None = None,
        ts_to: float | None = None,
    ) -> tuple[list[dict[str, Any]], int]:
        """Записи от новых к старым и общее число. action — через запятую, «auth.*» — по префиксу."""
        clauses: list[str] = []
        params: list[Any] = []
        if username:
            names = [normalize_username(n) for n in username.split(",") if n.strip()]
            clauses.append(f"username IN ({', '.join('?' for _ in names)})")
            params.extend(names)
        if action:
            parts = []
            for item in (a.strip() for a in action.split(",")):
                if not item:
                    continue
                if item.endswith(".*"):
                    parts.append("action LIKE ? ESCAPE '\\'")
                    params.append(_like_prefix(item[:-1]))
                else:
                    parts.append("action = ?")
                    params.append(item)
            if parts:
                clauses.append("(" + " OR ".join(parts) + ")")
        if ts_from is not None:
            clauses.append("ts >= ?")
            params.append(ts_from)
        if ts_to is not None:
            clauses.append("ts <= ?")
            params.append(ts_to)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        with self._lock, self._conn:
            self._conn.execute("BEGIN")
            total = self._conn.execute("SELECT COUNT(*) FROM audit_log" + where, params).fetchone()[0]
            rows = self._conn.execute(
                "SELECT id, ts, username, role, action, target, details, ip FROM audit_log"
                + where
                + " ORDER BY ts DESC, id DESC LIMIT ? OFFSET ?",
                [*params, limit, offset],
            ).fetchall()
        return [_audit_item(r) for r in rows], total
