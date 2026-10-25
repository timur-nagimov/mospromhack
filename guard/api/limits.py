"""Предел тела запроса. API принимает только небольшие JSON: камеры с зонами, разметку, проверку потока.

FastAPI читает тело целиком до проверки входа, поэтому без предела запрос без сессии мог бы занять память
сервера гигабайтами (ноутбук демо уже падал по нехватке памяти). Больше MAX_BODY_BYTES — 413, в том числе
для тела без Content-Length (chunked): счёт идёт по мере чтения.
"""

from __future__ import annotations

from typing import Any

from starlette.datastructures import Headers
from starlette.exceptions import HTTPException
from starlette.responses import JSONResponse

MAX_BODY_BYTES = 8 * 1024 * 1024
DETAIL = "request body too large"


class BodyLimit:
    def __init__(self, app: Any, max_bytes: int = MAX_BODY_BYTES) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        length = Headers(scope=scope).get("content-length")
        if length is not None and (not (length.isascii() and length.isdigit()) or int(length) > self.max_bytes):
            await JSONResponse({"detail": DETAIL}, status_code=413)(scope, receive, send)
            return
        received = 0

        async def limited() -> Any:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    raise HTTPException(413, DETAIL)
            return message

        await self.app(scope, limited, send)
