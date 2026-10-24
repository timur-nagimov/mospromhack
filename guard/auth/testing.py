"""Помощники тестов входа: управляемые часы, приложение с включённым входом, cookie сессии."""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
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


@dataclass
class AuthEnv:
    client: Any
    app: Any
    engine: Any
    settings: Any
    clock: FakeClock

    @property
    def auth(self) -> Any:
        return self.app.state.auth

    def login(self, username: str, password: str | None = None) -> dict[str, str]:
        return login_headers(self.client, username, password)


@contextmanager
def auth_env(tmp_path: Path, *, engine: Any | None = None, **overrides: Any) -> Iterator[AuthEnv]:
    """Приложение с FakeBackend без автозапуска; вход включён, если не передано auth=False."""
    from fastapi.testclient import TestClient

    from guard.api.app import create_app
    from guard.pipeline.engine import PipelineEngine
    from guard.pipeline.testing import FakeBackend, make_settings

    params: dict[str, Any] = {"auth": True, "autostart": False, **overrides}
    settings = engine.settings if engine is not None else make_settings(tmp_path, **params)
    if engine is None:
        engine = PipelineEngine(settings, backend=FakeBackend())
    app = create_app(settings, engine=engine)
    clock = FakeClock(time.time())
    app.state.auth.clock = clock
    with TestClient(app) as client:
        yield AuthEnv(client, app, engine, settings, clock)


def camera_dict(folder: Path, camera_id: str = "cam1", **extra: Any) -> dict[str, Any]:
    """Камера из папки кадров, выключенная: без воркера и потоков."""
    body: dict[str, Any] = {
        "camera_id": camera_id,
        "name": "Цех",
        "site": "Цех 1",
        "source": {"type": "images", "uri": str(folder)},
        "sample_fps": 5,
        "enabled": False,
        "zones": [
            {"zone_id": "z1", "name": "Опасная", "kind": "danger", "polygon": [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9]]}
        ],
        "rules": [{"rule_id": "s1", "type": "zone_intrusion", "zones": ["z1"]}],
    }
    body.update(extra)
    return body
