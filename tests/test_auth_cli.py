"""python -m guard.auth: баннер для scripts/run_*.sh, смена пароля из консоли, таблица «маршрут → право»."""

import io

import pytest

from guard.auth import __main__ as cli
from guard.auth import passwords
from guard.auth.store import AuthStore
from guard.pipeline import settings as settings_mod


@pytest.fixture(autouse=True)
def clean_env(monkeypatch, tmp_path):
    monkeypatch.setattr(passwords, "SCRYPT_N", 2**10)
    monkeypatch.setattr(settings_mod, "load_dotenv_guard", lambda *a, **k: None)
    for name in ("GUARD_AUTH", "GUARD_DEMO_LOGIN", "GUARD_AUTH_DB", "GUARD_MODE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("GUARD_DATA_DIR", str(tmp_path / "data"))


def test_banner_lists_demo_accounts_before_first_start(capsys, tmp_path):
    assert cli.main(["banner"]) == 0
    out = capsys.readouterr().out
    assert "Вход включён" in out
    for login, title in (("worker", "Сотрудник"), ("operator", "Диспетчер смены"), ("safety", "Инженер по ОТ"),
                         ("admin", "Администратор")):
        assert f"{login} / {login}" in out and title in out
    assert "GUARD_DEMO_LOGIN=0" in out and "смените пароли" in out
    assert not (tmp_path / "data" / "auth.db").exists()  # баннер базу не создаёт


def test_banner_skips_accounts_with_changed_password(capsys, tmp_path):
    with AuthStore(tmp_path / "data" / "auth.db") as store:
        store.ensure_demo_users()
        store.update_user("admin", password="secret-1")
    cli.main(["banner"])
    out = capsys.readouterr().out
    assert "worker / worker" in out and "admin / admin" not in out


def test_banner_without_demo_login(capsys, monkeypatch):
    monkeypatch.setenv("GUARD_DEMO_LOGIN", "0")
    cli.main(["banner"])
    out = capsys.readouterr().out
    assert "Вход включён" in out and "worker / worker" not in out


def test_banner_with_auth_off(capsys, monkeypatch):
    monkeypatch.setenv("GUARD_AUTH", "0")
    cli.main(["banner"])
    out = capsys.readouterr().out
    assert "Вход выключен" in out and "admin" in out


def test_set_password_from_stdin_closes_sessions(tmp_path, monkeypatch, capsys):
    db = tmp_path / "data" / "auth.db"
    with AuthStore(db) as store:
        store.ensure_demo_users()
        token = store.create_session("worker")
    monkeypatch.setattr("sys.stdin", io.StringIO("new-pass-9\n"))
    assert cli.main(["set-password", "worker", "--password-stdin"]) == 0
    with AuthStore(db) as store:
        assert store.authenticate("worker", "new-pass-9") is not None
        assert store.authenticate("worker", "worker") is None
        assert store.resolve_session(token) is None
    assert "new-pass-9" not in capsys.readouterr().out


def test_set_password_errors(tmp_path, monkeypatch):
    with AuthStore(tmp_path / "data" / "auth.db") as store:
        store.ensure_demo_users()
    monkeypatch.setattr("sys.stdin", io.StringIO("x\n"))
    assert cli.main(["set-password", "worker", "--password-stdin"]) == 2  # короткий пароль
    monkeypatch.setattr("sys.stdin", io.StringIO("long-enough\n"))
    assert cli.main(["set-password", "ghost", "--password-stdin"]) == 1


def test_routes_table(capsys):
    assert cli.main(["routes"]) == 0
    out = capsys.readouterr().out
    assert "| GET | /api/stats | analytics.view |" in out
    assert "| PUT | /api/cameras/{camera_id} | zones.edit или cameras.manage |" in out
    assert "| GET | /health | без входа |" in out
    assert "| WS | /ws/events | events.view |" in out
    assert "не объявлено" not in out
