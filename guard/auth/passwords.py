"""Хэши паролей: hashlib.scrypt с солью на каждый хэш, проверка через hmac.compare_digest.

Формат строки: scrypt$<n>$<r>$<p>$<соль base64>$<хэш base64>. Параметры хранятся в строке,
поэтому хэши со старыми параметрами проверяются и после смены умолчаний.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import os

SCRYPT_N = 2**14  # 16 МБ памяти, ~30 мс на ноутбуке
SCRYPT_R = 8
SCRYPT_P = 1
DKLEN = 32
SALT_BYTES = 16
_MAXMEM = 64 * 1024 * 1024
_PREFIX = "scrypt"

PASSWORD_MIN = 4  # мягко: демо-показ; для объекта задайте длинные пароли
PASSWORD_MAX = 128

_burn_hash: str | None = None


def password_problem(password: str) -> str | None:
    """Почему пароль не подходит, или None."""
    if not PASSWORD_MIN <= len(password) <= PASSWORD_MAX:
        return f"password: {PASSWORD_MIN}-{PASSWORD_MAX} symbols"
    return None


def _derive(password: str, salt: bytes, n: int, r: int, p: int, dklen: int) -> bytes:
    return hashlib.scrypt(password.encode("utf-8"), salt=salt, n=n, r=r, p=p, dklen=dklen, maxmem=_MAXMEM)


def hash_password(password: str, *, n: int | None = None) -> str:
    """Новый хэш со случайной солью."""
    n = n or SCRYPT_N
    salt = os.urandom(SALT_BYTES)
    digest = _derive(password, salt, n, SCRYPT_R, SCRYPT_P, DKLEN)
    b64 = base64.b64encode
    return f"{_PREFIX}${n}${SCRYPT_R}${SCRYPT_P}${b64(salt).decode()}${b64(digest).decode()}"


def verify_password(password: str, stored: str) -> bool:
    """True, если пароль подходит к хэшу. Испорченный или чужой формат — False, без исключений."""
    try:
        prefix, n, r, p, salt_b64, digest_b64 = stored.split("$")
        if prefix != _PREFIX:
            return False
        salt = base64.b64decode(salt_b64, validate=True)
        expected = base64.b64decode(digest_b64, validate=True)
        n_i, r_i, p_i = int(n), int(r), int(p)
        if not salt or not expected or n_i < 2 or n_i & (n_i - 1):
            return False
        actual = _derive(password, salt, n_i, r_i, p_i, len(expected))
    except (ValueError, TypeError, binascii.Error, MemoryError):
        return False
    return hmac.compare_digest(actual, expected)


def burn(password: str) -> bool:
    """Проверка против фиктивного хэша: вход с несуществующим логином длится столько же. Всегда False."""
    global _burn_hash
    if _burn_hash is None:
        _burn_hash = hash_password(os.urandom(16).hex())
    verify_password(password, _burn_hash)
    return False
