"""Мягкое ограничение входа: 10 неудач подряд на пару «логин, адрес» — 429 на минуту."""

from guard.auth.ratelimit import LoginLimiter
from guard.auth.testing import FakeClock


def test_blocks_after_ten_failures_for_a_minute():
    clock = FakeClock(100.0)
    lim = LoginLimiter(max_failures=10, block_s=60.0, clock=clock)
    for _ in range(9):
        assert lim.failure("admin", "10.0.0.1") == 0
        assert lim.retry_after("admin", "10.0.0.1") == 0
    assert lim.failure("admin", "10.0.0.1") == 60.0
    assert lim.retry_after("admin", "10.0.0.1") == 60.0
    clock.advance(30)
    assert lim.retry_after("admin", "10.0.0.1") == 30.0
    # другой адрес и другой логин не затронуты
    assert lim.retry_after("admin", "10.0.0.2") == 0
    assert lim.retry_after("worker", "10.0.0.1") == 0


def test_block_ends_after_a_minute_and_counter_restarts():
    clock = FakeClock(100.0)
    lim = LoginLimiter(max_failures=10, block_s=60.0, clock=clock)
    for _ in range(10):
        lim.failure("admin", "ip")
    clock.advance(60.5)
    assert lim.retry_after("admin", "ip") == 0
    assert lim.failure("admin", "ip") == 0  # снова с нуля, а не блок с первой ошибки


def test_success_resets_the_counter():
    clock = FakeClock(100.0)
    lim = LoginLimiter(max_failures=10, block_s=60.0, clock=clock)
    for _ in range(9):
        lim.failure("admin", "ip")
    lim.success("admin", "ip")
    for _ in range(9):
        assert lim.failure("admin", "ip") == 0


def test_login_is_case_insensitive():
    lim = LoginLimiter(max_failures=2, block_s=60.0, clock=FakeClock(0.0))
    lim.failure("Admin", "ip")
    lim.failure("ADMIN", "ip")
    assert lim.retry_after("admin", "ip") == 60.0


def test_memory_is_bounded():
    clock = FakeClock(0.0)
    lim = LoginLimiter(max_failures=10, block_s=60.0, clock=clock, max_keys=100)
    for i in range(1000):
        lim.failure(f"user{i}", "ip")
    assert len(lim) <= 100


def test_active_block_survives_eviction():
    lim = LoginLimiter(max_failures=2, block_s=60.0, clock=FakeClock(0.0), max_keys=10)
    lim.failure("admin", "ip")
    lim.failure("admin", "ip")
    for i in range(100):
        lim.failure(f"user{i}", "ip")
    assert lim.retry_after("admin", "ip") == 60.0


def test_api_answers_429_after_ten_failed_logins(tmp_path, monkeypatch):
    from guard.auth import passwords
    from guard.auth.testing import auth_env

    monkeypatch.setattr(passwords, "SCRYPT_N", 2**10)
    with auth_env(tmp_path) as env:
        bad = {"username": "admin", "password": "wrong"}
        for _ in range(10):
            assert env.client.post("/api/auth/login", json=bad).status_code == 401
        r = env.client.post("/api/auth/login", json={"username": "admin", "password": "admin"})
        assert r.status_code == 429  # даже верный пароль во время паузы
        assert r.headers["retry-after"] == "60"
        assert r.json() == {"detail": "too many attempts", "retry_after": 60}
        # другой логин с того же адреса не затронут
        env.login("worker")
        env.clock.advance(61)
        headers = env.login("admin")
        failed = env.client.get("/api/audit", params={"action": "auth.login_failed"}, headers=headers).json()
        assert failed["total"] == 10
        assert failed["items"][0]["details"]["locked_for_s"] == 60
