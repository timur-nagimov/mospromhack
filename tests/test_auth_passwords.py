"""Пароли: hashlib.scrypt с солью на пользователя, сравнение за постоянное время (docs/ROLES.md)."""

import pytest

from guard.auth import passwords
from guard.auth.passwords import hash_password, verify_password


def test_hash_is_salted_scrypt_without_plaintext():
    first = hash_password("секрет-1")
    second = hash_password("секрет-1")
    assert first.startswith("scrypt$")
    assert first != second  # своя соль у каждого хэша
    assert "секрет-1" not in first


def test_verify_accepts_only_the_right_password():
    stored = hash_password("worker")
    assert verify_password("worker", stored)
    assert not verify_password("Worker", stored)
    assert not verify_password("worker ", stored)
    assert not verify_password("", stored)


@pytest.mark.parametrize(
    "stored",
    ["", "plain", "scrypt$1$2", "scrypt$x$8$1$AAAA$BBBB", "scrypt$16384$8$1$!!!$???", "md5$abc"],
)
def test_verify_rejects_malformed_hash(stored):
    assert not verify_password("worker", stored)


def test_verify_compares_in_constant_time(monkeypatch):
    calls = []
    real = passwords.hmac.compare_digest

    def spy(a, b):
        calls.append((a, b))
        return real(a, b)

    monkeypatch.setattr(passwords.hmac, "compare_digest", spy)
    assert verify_password("x", hash_password("x"))
    assert calls


def test_hash_keeps_its_parameters():
    """Параметры scrypt записаны в строке: хэш с другим N проверяется и после смены умолчаний."""
    stored = hash_password("pw", n=2**10)
    assert stored.split("$")[1] == str(2**10)
    assert verify_password("pw", stored)
    assert not verify_password("pw2", stored)


def test_burn_takes_a_verify_without_a_user():
    """Для несуществующего логина тратим то же время, что на проверку пароля."""
    assert passwords.burn("что угодно") is False
