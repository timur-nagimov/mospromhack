"""Помощники тестов входа: управляемые часы и заголовок cookie сессии."""

from __future__ import annotations

from typing import Any

SESSION_COOKIE = "guard_session"


class FakeClock:
    """Часы для тестов: время стоит, пока его не сдвинут."""

    def __init__(self, start: float = 1_000_000.0) -> None:
        self.now = float(start)

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def login_headers(client: Any, username: str, password: str | None = None) -> dict[str, str]:
    """Войти через API и вернуть заголовок Cookie; банку cookie клиента оставляем пустой."""
    r = client.post("/api/auth/login", json={"username": username, "password": password or username})
    assert r.status_code == 200, r.text
    token = r.cookies.get(SESSION_COOKIE)
    assert token, "login did not set the session cookie"
    client.cookies.clear()
    return {"Cookie": f"{SESSION_COOKIE}={token}"}
