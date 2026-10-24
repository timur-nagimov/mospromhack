"""SQLite входа: пользователи, сессии (SHA-256 токена, 12 ч со сдвигом) и журнал действий."""

import hashlib
import sqlite3
import threading
from contextlib import closing

import pytest

from guard.auth import passwords
from guard.auth.store import AuthStore, LastAdminError, UserExists, UserNotFound
from guard.auth.testing import FakeClock

HOUR = 3600.0


@pytest.fixture(autouse=True)
def fast_scrypt(monkeypatch):
    monkeypatch.setattr(passwords, "SCRYPT_N", 2**10)


@pytest.fixture()
def clock():
    return FakeClock(1_000_000.0)


@pytest.fixture()
def store(tmp_path, clock):
    s = AuthStore(tmp_path / "auth.db", clock=clock)
    yield s
    s.close()


def test_wal_and_tables(tmp_path, store):
    with closing(sqlite3.connect(tmp_path / "auth.db")) as conn:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert {"users", "sessions", "audit_log"} <= tables


def test_demo_users_are_seeded_only_into_empty_db(store):
    created = store.ensure_demo_users()
    assert created == ["worker", "operator", "safety", "admin"]
    assert store.ensure_demo_users() == []
    for name in created:
        user = store.get_user(name)
        assert user.role == name and user.active and user.demo
        assert store.authenticate(name, name) is not None
    assert [u.username for u in store.list_users()] == ["admin", "operator", "safety", "worker"]


def test_create_get_and_duplicate(store):
    user = store.create_user("Ivanov", "Иванов И.", "operator", "pass1234")
    assert user.username == "ivanov"  # логин без учёта регистра
    assert store.get_user("IVANOV").display_name == "Иванов И."
    assert not user.demo
    with pytest.raises(UserExists):
        store.create_user("ivanov", "Другой", "worker", "pass1234")
    with pytest.raises(ValueError):
        store.create_user("petrov", "Петров", "ghost", "pass1234")
    public = user.public()
    assert set(public) == {"username", "display_name", "role", "active", "created_at", "last_login_at"}


def test_authenticate(store):
    store.create_user("ivanov", "Иванов", "operator", "pass1234")
    assert store.authenticate("ivanov", "pass1234").username == "ivanov"
    assert store.authenticate("IVANOV", "pass1234") is not None
    assert store.authenticate("ivanov", "wrong") is None
    assert store.authenticate("nobody", "pass1234") is None
    store.update_user("ivanov", active=False)
    assert store.authenticate("ivanov", "pass1234") is None


def test_session_stores_only_token_hash(tmp_path, store):
    store.create_user("ivanov", "Иванов", "operator", "pass1234")
    token = store.create_session("ivanov", ip="10.0.0.5", user_agent="pytest")
    assert len(token) >= 40
    with closing(sqlite3.connect(tmp_path / "auth.db")) as conn:
        rows = conn.execute("SELECT token_hash, ip FROM sessions").fetchall()
    assert rows == [(hashlib.sha256(token.encode()).hexdigest(), "10.0.0.5")]
    user, token_hash = store.resolve_session(token)
    assert user.username == "ivanov" and token_hash == rows[0][0]
    assert store.resolve_session("bogus") is None
    assert store.resolve_session("") is None


def test_session_expires_after_12h_and_slides_with_activity(store, clock):
    store.create_user("ivanov", "Иванов", "operator", "pass1234")
    token = store.create_session("ivanov")
    clock.advance(11 * HOUR)
    assert store.resolve_session(token) is not None  # активность сдвигает срок
    clock.advance(11 * HOUR)
    assert store.resolve_session(token) is not None
    clock.advance(12 * HOUR + 1)
    assert store.resolve_session(token) is None
    assert store.resolve_session(token) is None  # просроченная удалена


def test_idle_session_expires(store, clock):
    store.create_user("ivanov", "Иванов", "operator", "pass1234")
    token = store.create_session("ivanov")
    clock.advance(12 * HOUR + 1)
    assert store.resolve_session(token) is None


def test_inactive_user_session_is_rejected(store):
    store.create_user("ivanov", "Иванов", "operator", "pass1234")
    token = store.create_session("ivanov")
    store.update_user("ivanov", active=False)
    assert store.resolve_session(token) is None


def test_session_alive_and_delete(store):
    store.create_user("ivanov", "Иванов", "operator", "pass1234")
    first = store.create_session("ivanov")
    second = store.create_session("ivanov")
    _, first_hash = store.resolve_session(first)
    _, second_hash = store.resolve_session(second)
    assert store.session_alive(first_hash)
    assert store.delete_user_sessions("ivanov", keep_hash=second_hash) == 1
    assert not store.session_alive(first_hash)
    assert store.session_alive(second_hash)
    store.delete_session(second_hash)
    assert store.resolve_session(second) is None


def test_password_change_clears_demo_flag(store):
    store.ensure_demo_users()
    store.update_user("worker", password="new-secret")
    user = store.get_user("worker")
    assert not user.demo
    assert store.authenticate("worker", "worker") is None
    assert store.authenticate("worker", "new-secret") is not None
    assert store.get_user("admin").demo


def test_last_active_admin_cannot_be_demoted_or_disabled(store):
    store.ensure_demo_users()
    with pytest.raises(LastAdminError):
        store.update_user("admin", role="operator")
    with pytest.raises(LastAdminError):
        store.update_user("admin", active=False)
    store.create_user("root2", "Второй админ", "admin", "pass1234")
    store.update_user("admin", role="safety")
    assert store.get_user("admin").role == "safety"
    with pytest.raises(LastAdminError):
        store.update_user("root2", active=False)
    store.update_user("root2", display_name="Главный")  # прочие поля менять можно
    with pytest.raises(UserNotFound):
        store.update_user("ghost", display_name="x")


def test_concurrent_demotion_keeps_one_admin(store):
    store.ensure_demo_users()
    store.create_user("root2", "Второй", "admin", "pass1234")
    errors = []

    def demote(name):
        try:
            store.update_user(name, role="worker")
        except LastAdminError as exc:
            errors.append(exc)

    threads = [threading.Thread(target=demote, args=(n,)) for n in ("admin", "root2")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(errors) == 1
    assert store.count_active_admins() == 1


def test_mark_login(store, clock):
    store.create_user("ivanov", "Иванов", "operator", "pass1234")
    assert store.get_user("ivanov").last_login_at is None
    store.mark_login("ivanov")
    assert store.get_user("ivanov").last_login_at == clock()


def test_audit_query_filters_and_pagination(store, clock):
    store.add_audit("auth.login", username="admin", role="admin", target="admin", ip="10.0.0.1")
    clock.advance(10)
    store.add_audit("camera.create", username="admin", role="admin", target="cam1", details={"name": "Цех"})
    clock.advance(10)
    store.add_audit("zones.update", username="safety", role="safety", target="cam1", details={"zones": 2})
    clock.advance(10)
    store.add_audit("auth.login_failed", username="ghost", role=None, details={"reason": "bad_credentials"})

    items, total = store.query_audit()
    assert total == 4
    assert [i["action"] for i in items] == ["auth.login_failed", "zones.update", "camera.create", "auth.login"]
    assert items[2]["details"] == {"name": "Цех"} and items[2]["target"] == "cam1"
    assert set(items[0]) == {"id", "ts", "username", "role", "action", "target", "details", "ip"}

    assert store.query_audit(username="admin")[1] == 2
    assert store.query_audit(action="camera.create,zones.update")[1] == 2
    assert store.query_audit(action="auth.*")[1] == 2
    start = clock() - 30
    assert store.query_audit(ts_from=start + 10, ts_to=start + 20)[1] == 2
    page, total = store.query_audit(limit=1, offset=1)
    assert total == 4 and [i["action"] for i in page] == ["zones.update"]


def test_store_is_thread_safe(store):
    store.create_user("ivanov", "Иванов", "operator", "pass1234")

    def work():
        for _ in range(20):
            token = store.create_session("ivanov")
            assert store.resolve_session(token) is not None
            store.add_audit("auth.login", username="ivanov", role="operator")

    threads = [threading.Thread(target=work) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert store.query_audit(limit=1)[1] == 120


def test_reopen_keeps_data(tmp_path, clock):
    with AuthStore(tmp_path / "x.db", clock=clock) as s:
        s.ensure_demo_users()
        token = s.create_session("admin")
    with AuthStore(tmp_path / "x.db", clock=clock) as s:
        assert s.resolve_session(token)[0].username == "admin"
        assert s.ensure_demo_users() == []
