#!/usr/bin/env python3
"""CLI и модуль для Jupyter / JupyterHub стенда LinQ по REST и kernel websocket.

Команды:
    login                      войти и закешировать сессию (~/.cache/zazerkalye/jhub_session.json)
    ls [path]                  содержимое каталога на сервере
    upload <local> <remote>    загрузить файл (чанками, base64 через /api/contents)
    download <remote> [local]  скачать файл (/files/, потоково)
    exec <code|file.py>        выполнить код в ядре (kernel API по websocket), вывести stdout/stderr
    run-bg <cmd>               запустить команду в фоне (nohup, лог в ~/jhub_bg/<name>.log)
    ps                         процессы пользователя на сервере + ядра + фоновые задачи
    kernels                    список ядер; --stop ID|all — остановить
    tail <remote-file>         последние строки файла на сервере

Логин (порядок попыток):
    1) токен: --token или JUPYTER_TOKEN из .env — заголовок `Authorization: token ...`;
    2) JupyterHub: GET /hub/login -> _xsrf (cookie или hidden-поле) -> POST /hub/login (username, password, _xsrf),
       затем запуск single-user сервера и выпуск API-токена (POST /hub/api/users/<u>/tokens);
       если выпуск токена запрещён — OAuth-cookie single-user сервера;
    3) одиночный Jupyter Server: GET /login -> POST /login (password, _xsrf).

Доступы: <корень>/.env — JUPYTER_URL, JUPYTER_LOGIN, JUPYTER_PASSWORD, опционально JUPYTER_TOKEN.
Статус: логин в hub не проверен на стенде (нужен VPN); contents/kernels/websocket проверены
на локальном jupyter_server (см. docs/LINQ_NOTES.md, раздел «Инструменты»).

Пример использования модулем:
    from tools.jhub import JHub
    jh = JHub.from_env(); jh.login()
    jh.upload("models/tpu/det/program.tpu", "work/tpu/det.tpu")
    res = jh.execute("import os; print(os.getcwd())"); print(res.stdout)
"""

from __future__ import annotations

import argparse
import base64
import datetime as _dt
import json
import os
import re
import shlex
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlencode, urlsplit

import requests

ROOT = Path(os.environ.get("GUARD_ROOT", "/Users/nagimov-timur/Desktop/mosprom"))
DEFAULT_ENV = ROOT / ".env"
SESSION_FILE = Path(os.environ.get("JHUB_SESSION_FILE", Path.home() / ".cache" / "zazerkalye" / "jhub_session.json"))
DEFAULT_TIMEOUT = 30.0
UPLOAD_CHUNK = 4 * 1024 * 1024  # сырые байты на чанк; при HTTP 413 уменьшаем


class JHubError(RuntimeError):
    pass


def load_env(path: Path | str = DEFAULT_ENV) -> dict[str, str]:
    """Простой парсер .env (KEY=VALUE, кавычки снимаются). Значения окружения имеют приоритет."""
    env: dict[str, str] = {}
    p = Path(path)
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip('"').strip("'")
    for k in ("JUPYTER_URL", "JUPYTER_LOGIN", "JUPYTER_PASSWORD", "JUPYTER_TOKEN"):
        if os.environ.get(k):
            env[k] = os.environ[k]
    return env


_HIDDEN_XSRF = re.compile(r'name=["\']_xsrf["\'][^>]*value=["\']([^"\']+)["\']|value=["\']([^"\']+)["\'][^>]*name=["\']_xsrf["\']')


def _parse_xsrf(html: str) -> str | None:
    m = _HIDDEN_XSRF.search(html or "")
    if not m:
        return None
    return m.group(1) or m.group(2)


@dataclass
class ExecResult:
    stdout: str = ""
    stderr: str = ""
    results: list[str] = field(default_factory=list)  # text/plain из execute_result / display_data
    error: dict[str, Any] | None = None  # {ename, evalue, traceback}
    status: str = ""  # ok | error | timeout
    elapsed_s: float = 0.0

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def text(self) -> str:
        parts = [self.stdout]
        parts += [r + ("\n" if not r.endswith("\n") else "") for r in self.results]
        return "".join(parts)


class JHub:
    """Клиент Jupyter/JupyterHub. Потокобезопасность не гарантируется (одна сессия requests)."""

    def __init__(
        self,
        url: str,
        user: str | None = None,
        password: str | None = None,
        token: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        session_file: Path | None = SESSION_FILE,
        verbose: bool = False,
    ) -> None:
        self.base = url.rstrip("/")
        self.user = user
        self.password = password
        self.token = token
        self.timeout = timeout
        self.session_file = session_file
        self.verbose = verbose
        self.s = requests.Session()
        self.s.headers["User-Agent"] = "zazerkalye-jhub/1.0"
        self.mode: str | None = None  # hub | server
        self.api_base: str | None = None  # URL single-user сервера (…/user/<u>) или сервера
        self.hub_api_token: str | None = None  # токен, выпущенный hub для этого клиента

    # ---------- конструирование ----------

    @classmethod
    def from_env(cls, env_path: Path | str = DEFAULT_ENV, **kw: Any) -> "JHub":
        env = load_env(env_path)
        url = kw.pop("url", None) or env.get("JUPYTER_URL")
        if not url:
            raise JHubError(f"JUPYTER_URL не задан ({env_path})")
        return cls(
            url,
            user=kw.pop("user", None) or env.get("JUPYTER_LOGIN"),
            password=kw.pop("password", None) or env.get("JUPYTER_PASSWORD"),
            token=kw.pop("token", None) or env.get("JUPYTER_TOKEN") or None,
            **kw,
        )

    def _log(self, *a: Any) -> None:
        if self.verbose:
            print("[jhub]", *a, file=sys.stderr)

    # ---------- сессия ----------

    def _xsrf(self) -> str | None:
        for c in self.s.cookies:
            if c.name == "_xsrf":
                return c.value
        return None

    def _headers(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        h: dict[str, str] = {}
        tok = self.token or self.hub_api_token
        if tok:
            h["Authorization"] = f"token {tok}"
        x = self._xsrf()
        if x:
            h["X-XSRFToken"] = x
        if extra:
            h.update(extra)
        return h

    def save_session(self) -> None:
        if not self.session_file:
            return
        self.session_file.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "base": self.base,
            "user": self.user,
            "mode": self.mode,
            "api_base": self.api_base,
            "hub_api_token": self.hub_api_token,
            "cookies": [
                {"name": c.name, "value": c.value, "domain": c.domain, "path": c.path} for c in self.s.cookies
            ],
            "saved_at": time.time(),
        }
        self.session_file.write_text(json.dumps(data, indent=1))
        os.chmod(self.session_file, 0o600)

    def load_session(self) -> bool:
        if not self.session_file or not self.session_file.exists():
            return False
        try:
            data = json.loads(self.session_file.read_text())
        except (OSError, ValueError):
            return False
        if data.get("base") != self.base:
            return False
        self.mode = data.get("mode")
        self.api_base = data.get("api_base")
        self.hub_api_token = data.get("hub_api_token")
        if data.get("user"):
            self.user = self.user or data["user"]
        for c in data.get("cookies", []):
            self.s.cookies.set(c["name"], c["value"], domain=c.get("domain"), path=c.get("path") or "/")
        return bool(self.api_base)

    def ensure(self) -> "JHub":
        """Восстановить сессию из кеша и проверить её; при неудаче — войти заново."""
        if self.api_base is None and self.load_session():
            try:
                r = self.s.get(self.api_base + "/api/status", headers=self._headers(), timeout=self.timeout,
                               allow_redirects=True)
                if r.status_code == 200 and r.headers.get("content-type", "").startswith("application/json"):
                    return self
            except requests.RequestException:
                pass
            self._log("кеш сессии недействителен, логинимся заново")
            self.s.cookies.clear()
            self.hub_api_token = None
            self.api_base = None
        if self.api_base is None:
            self.login()
        return self

    # ---------- логин ----------

    def login(self, spawn_timeout: float = 120.0) -> dict[str, Any]:
        info: dict[str, Any] = {"base": self.base}
        if self.token:
            info.update(self._login_token())
        else:
            r = self.s.get(self.base + "/hub/login", timeout=self.timeout, allow_redirects=True)
            self._log("GET /hub/login ->", r.status_code, r.url)
            if r.status_code == 200 and ("/hub/" in r.url) and ("<form" in r.text or "password" in r.text):
                info.update(self._login_hub(r, spawn_timeout))
            else:
                info.update(self._login_server())
        self.save_session()
        info["mode"] = self.mode
        info["api_base"] = self.api_base
        return info

    def _login_token(self) -> dict[str, Any]:
        # Пробуем как hub-токен, потом как токен одиночного сервера.
        r = self.s.get(self.base + "/hub/api/user", headers=self._headers(), timeout=self.timeout)
        if r.status_code == 200:
            u = r.json()
            self.mode, self.user = "hub", u.get("name", self.user)
            self.api_base = self._user_server_url(u)
            return {"auth": "token", "user": self.user, "server": u.get("server")}
        r = self.s.get(self.base + "/api/status", headers=self._headers(), timeout=self.timeout)
        if r.status_code == 200:
            self.mode, self.api_base = "server", self.base
            return {"auth": "token", "status": r.json()}
        raise JHubError(f"токен не принят: /hub/api/user и /api/status -> {r.status_code}")

    def _login_hub(self, login_page: requests.Response, spawn_timeout: float) -> dict[str, Any]:
        if not (self.user and self.password):
            raise JHubError("нужны JUPYTER_LOGIN и JUPYTER_PASSWORD")
        xsrf = _parse_xsrf(login_page.text) or self._xsrf()
        data = {"username": self.user, "password": self.password}
        if xsrf:
            data["_xsrf"] = xsrf
        r = self.s.post(self.base + "/hub/login?next=", data=data, timeout=self.timeout, allow_redirects=True)
        self._log("POST /hub/login ->", r.status_code, r.url)
        if r.status_code >= 400 or r.url.rstrip("/").endswith("/hub/login"):
            raise JHubError(f"логин в hub не удался: HTTP {r.status_code}, url={r.url}")
        self.mode = "hub"
        u = self._hub_get_user()
        if u is None:
            raise JHubError("после логина /hub/api/user недоступен")
        self.user = u.get("name", self.user)
        u = self._hub_ensure_server(u, spawn_timeout)
        self.api_base = self._user_server_url(u)
        # Пробуем выпустить API-токен (удобно для websocket и запросов к single-user серверу).
        try:
            r = self.s.post(
                f"{self.base}/hub/api/users/{quote(self.user)}/tokens",
                json={"note": "zazerkalye-jhub", "expires_in": 3 * 24 * 3600},
                headers=self._headers(),
                timeout=self.timeout,
            )
            if r.status_code in (200, 201):
                self.hub_api_token = r.json().get("token")
            else:
                self._log("выпуск токена ->", r.status_code, r.text[:200])
        except requests.RequestException as e:  # pragma: no cover - сеть
            self._log("выпуск токена: ошибка", e)
        if not self.hub_api_token:
            # OAuth-cookie single-user сервера: первый запрос пройдёт через /hub/api/oauth2/authorize
            r = self.s.get(self.api_base + "/api/status", timeout=self.timeout, allow_redirects=True)
            self._log("oauth single-user ->", r.status_code, r.url)
        return {"auth": "hub-form", "user": self.user, "token_issued": bool(self.hub_api_token)}

    def _hub_get_user(self) -> dict[str, Any] | None:
        r = self.s.get(self.base + "/hub/api/user", headers=self._headers(), timeout=self.timeout)
        if r.status_code == 200:
            return r.json()
        if self.user:
            r = self.s.get(f"{self.base}/hub/api/users/{quote(self.user)}", headers=self._headers(),
                           timeout=self.timeout)
            if r.status_code == 200:
                return r.json()
        self._log("/hub/api/user ->", r.status_code, r.text[:200])
        return None

    def _hub_ensure_server(self, u: dict[str, Any], spawn_timeout: float) -> dict[str, Any]:
        def ready(uu: dict[str, Any]) -> bool:
            srv = (uu.get("servers") or {}).get("") or {}
            return bool(srv.get("ready")) or (bool(uu.get("server")) and not uu.get("pending"))

        if ready(u):
            return u
        self._log("single-user сервер не запущен, запускаем")
        r = self.s.post(f"{self.base}/hub/api/users/{quote(self.user)}/server", headers=self._headers(),
                        timeout=self.timeout)
        if r.status_code not in (201, 202, 400):  # 400 = уже запущен/запускается
            # запасной путь: /hub/spawn через браузерный поток
            self.s.get(self.base + "/hub/spawn", timeout=self.timeout, allow_redirects=True)
        t0 = time.time()
        while time.time() - t0 < spawn_timeout:
            uu = self._hub_get_user()
            if uu and ready(uu):
                return uu
            time.sleep(2)
        raise JHubError(f"single-user сервер не поднялся за {spawn_timeout} с")

    def _user_server_url(self, u: dict[str, Any]) -> str:
        srv = (u.get("servers") or {}).get("") or {}
        path = srv.get("url") or u.get("server") or f"/user/{quote(u.get('name') or self.user or '')}/"
        return self.base + "/" + path.strip("/")

    def _login_server(self) -> dict[str, Any]:
        r = self.s.get(self.base + "/login", timeout=self.timeout)
        xsrf = _parse_xsrf(r.text) or self._xsrf()
        data = {"password": self.password or ""}
        if xsrf:
            data["_xsrf"] = xsrf
        r = self.s.post(self.base + "/login", data=data, timeout=self.timeout, allow_redirects=True)
        self._log("POST /login ->", r.status_code, r.url)
        self.mode, self.api_base = "server", self.base
        st = self.s.get(self.base + "/api/status", headers=self._headers(), timeout=self.timeout)
        if st.status_code != 200:
            raise JHubError(f"логин в Jupyter Server не удался: /api/status -> {st.status_code}")
        return {"auth": "server-form"}

    # ---------- REST ----------

    def _api(self, method: str, path: str, **kw: Any) -> requests.Response:
        self.ensure()
        extra = kw.pop("headers", None)
        kw.setdefault("timeout", self.timeout)
        r = self.s.request(method, self.api_base + path, headers=self._headers(extra), **kw)
        expired = r.status_code == 401 or (r.status_code == 403 and "xsrf" in r.text.lower())
        if expired and (self.password or self.token):
            self._log(method, path, "->", r.status_code, "релогин")
            self.s.cookies.clear()
            self.hub_api_token = None
            self.api_base = None
            self.login()
            r = self.s.request(method, self.api_base + path, headers=self._headers(extra), **kw)
        return r

    @staticmethod
    def _cpath(remote: str) -> str:
        return quote(remote.strip("/"))

    def status(self) -> dict[str, Any]:
        r = self._api("GET", "/api/status")
        r.raise_for_status()
        return r.json()

    def ls(self, remote: str = "") -> list[dict[str, Any]]:
        r = self._api("GET", f"/api/contents/{self._cpath(remote)}", params={"content": 1})
        if r.status_code == 404:
            raise JHubError(f"нет такого пути: {remote}")
        r.raise_for_status()
        d = r.json()
        if d.get("type") != "directory":
            return [d]
        return sorted(d.get("content") or [], key=lambda x: (x["type"] != "directory", x["name"]))

    def mkdirs(self, remote_dir: str) -> None:
        parts = [p for p in remote_dir.strip("/").split("/") if p]
        cur = ""
        for p in parts:
            cur = f"{cur}/{p}" if cur else p
            r = self._api("GET", f"/api/contents/{self._cpath(cur)}", params={"content": 0})
            if r.status_code == 200:
                continue
            r = self._api("PUT", f"/api/contents/{self._cpath(cur)}", json={"type": "directory"})
            if r.status_code not in (200, 201):
                raise JHubError(f"mkdir {cur}: HTTP {r.status_code} {r.text[:200]}")

    def upload(self, local: str | Path, remote: str, chunk: int = UPLOAD_CHUNK, progress: bool = False) -> dict:
        local = Path(local)
        size = local.stat().st_size
        parent = remote.strip("/").rsplit("/", 1)[0] if "/" in remote.strip("/") else ""
        if parent:
            self.mkdirs(parent)
        t0 = time.time()
        with local.open("rb") as f:
            if size <= chunk:
                body = {"type": "file", "format": "base64", "name": Path(remote).name,
                        "content": base64.b64encode(f.read()).decode()}
                r = self._api("PUT", f"/api/contents/{self._cpath(remote)}", json=body)
                if r.status_code == 413 and chunk > 256 * 1024:
                    return self.upload(local, remote, chunk=max(256 * 1024, chunk // 4), progress=progress)
                if r.status_code not in (200, 201):
                    raise JHubError(f"upload: HTTP {r.status_code} {r.text[:300]}")
            else:
                n = 0
                sent = 0
                while True:
                    data = f.read(chunk)
                    if not data:
                        break
                    n += 1
                    sent += len(data)
                    last = sent >= size
                    body = {"type": "file", "format": "base64", "name": Path(remote).name,
                            "content": base64.b64encode(data).decode(), "chunk": -1 if last else n}
                    r = self._api("PUT", f"/api/contents/{self._cpath(remote)}", json=body)
                    if r.status_code == 413 and chunk > 256 * 1024:
                        return self.upload(local, remote, chunk=max(256 * 1024, chunk // 4), progress=progress)
                    if r.status_code not in (200, 201):
                        raise JHubError(f"upload chunk {n}: HTTP {r.status_code} {r.text[:300]}")
                    if progress:
                        print(f"  {sent / 1e6:.1f}/{size / 1e6:.1f} MB", file=sys.stderr)
        dt = time.time() - t0
        return {"remote": remote, "bytes": size, "seconds": round(dt, 3), "MBps": round(size / 1e6 / max(dt, 1e-9), 3)}

    def download(self, remote: str, local: str | Path | None = None) -> dict:
        local = Path(local) if local else Path(Path(remote).name)
        if local.is_dir():
            local = local / Path(remote).name
        t0 = time.time()
        r = self._api("GET", f"/files/{self._cpath(remote)}", stream=True, timeout=max(self.timeout, 300))
        if r.status_code != 200:
            # запасной путь через contents API
            r2 = self._api("GET", f"/api/contents/{self._cpath(remote)}", params={"content": 1, "format": "base64"})
            if r2.status_code != 200:
                raise JHubError(f"download: HTTP {r.status_code}/{r2.status_code}")
            d = r2.json()
            data = base64.b64decode(d["content"]) if d.get("format") == "base64" else d["content"].encode()
            local.write_bytes(data)
            n = len(data)
        else:
            n = 0
            with local.open("wb") as f:
                for part in r.iter_content(1 << 20):
                    f.write(part)
                    n += len(part)
        dt = time.time() - t0
        return {"local": str(local), "bytes": n, "seconds": round(dt, 3), "MBps": round(n / 1e6 / max(dt, 1e-9), 3)}

    def read_text(self, remote: str) -> str:
        r = self._api("GET", f"/api/contents/{self._cpath(remote)}", params={"content": 1, "format": "text"})
        r.raise_for_status()
        return r.json()["content"]

    # ---------- ядра ----------

    def kernels(self) -> list[dict[str, Any]]:
        r = self._api("GET", "/api/kernels")
        r.raise_for_status()
        return r.json()

    def start_kernel(self, name: str = "python3") -> str:
        r = self._api("POST", "/api/kernels", json={"name": name})
        if r.status_code not in (200, 201):
            raise JHubError(f"start kernel: HTTP {r.status_code} {r.text[:300]}")
        return r.json()["id"]

    def stop_kernel(self, kernel_id: str) -> None:
        self._api("DELETE", f"/api/kernels/{kernel_id}")

    def _ws_url(self, kernel_id: str, session_id: str) -> str:
        parts = urlsplit(self.api_base)
        scheme = "wss" if parts.scheme == "https" else "ws"
        q = {"session_id": session_id}
        tok = self.token or self.hub_api_token
        if tok:
            q["token"] = tok
        return f"{scheme}://{parts.netloc}{parts.path}/api/kernels/{kernel_id}/channels?{urlencode(q)}"

    def execute(self, code: str, kernel_id: str | None = None, timeout: float = 600.0,
                stream_to: Any = None, kernel_name: str = "python3") -> ExecResult:
        """Выполнить код в ядре. Если kernel_id не задан — временное ядро (останавливается после)."""
        import websocket  # websocket-client

        self.ensure()
        own = kernel_id is None
        kid = kernel_id or self.start_kernel(kernel_name)
        session_id = uuid.uuid4().hex
        hdrs = [f"{k}: {v}" for k, v in self._headers().items()]
        cookie = "; ".join(f"{c.name}={c.value}" for c in self.s.cookies)
        if cookie:
            hdrs.append(f"Cookie: {cookie}")
        res = ExecResult()
        t0 = time.time()
        ws = websocket.create_connection(self._ws_url(kid, session_id), header=hdrs, timeout=min(timeout, 60))
        try:
            msg_id = uuid.uuid4().hex
            msg = {
                "header": {"msg_id": msg_id, "username": self.user or "jhub", "session": session_id,
                           "msg_type": "execute_request", "version": "5.3",
                           "date": _dt.datetime.now(_dt.timezone.utc).isoformat()},
                "parent_header": {}, "metadata": {}, "buffers": [], "channel": "shell",
                "content": {"code": code, "silent": False, "store_history": False, "user_expressions": {},
                            "allow_stdin": False, "stop_on_error": True},
            }
            ws.send(json.dumps(msg))
            got_idle = got_reply = False
            while not (got_idle and got_reply):
                left = timeout - (time.time() - t0)
                if left <= 0:
                    res.status = "timeout"
                    break
                ws.settimeout(min(left, 60))
                try:
                    raw = ws.recv()
                except websocket.WebSocketTimeoutException:
                    continue
                if isinstance(raw, bytes):  # бинарный протокол v1 — не запрашиваем, игнорируем
                    continue
                m = json.loads(raw)
                if (m.get("parent_header") or {}).get("msg_id") != msg_id:
                    continue
                mt = m.get("msg_type") or m.get("header", {}).get("msg_type")
                c = m.get("content", {})
                if mt == "stream":
                    if c.get("name") == "stderr":
                        res.stderr += c.get("text", "")
                    else:
                        res.stdout += c.get("text", "")
                    if stream_to is not None:
                        stream_to.write(c.get("text", ""))
                        stream_to.flush()
                elif mt in ("execute_result", "display_data"):
                    t = (c.get("data") or {}).get("text/plain")
                    if t:
                        res.results.append(t)
                        if stream_to is not None:
                            stream_to.write(t + "\n")
                elif mt == "error":
                    res.error = {"ename": c.get("ename"), "evalue": c.get("evalue"),
                                 "traceback": c.get("traceback", [])}
                elif mt == "execute_reply":
                    got_reply = True
                    res.status = c.get("status", "ok")
                elif mt == "status" and c.get("execution_state") == "idle":
                    got_idle = True
        finally:
            ws.close()
            if own:
                try:
                    self.stop_kernel(kid)
                except Exception:  # noqa: BLE001
                    pass
        if not res.status:
            res.status = "error" if res.error else "ok"
        res.elapsed_s = round(time.time() - t0, 3)
        return res

    def sh(self, cmd: str, timeout: float = 600.0, kernel_id: str | None = None) -> ExecResult:
        """Выполнить shell-команду на сервере через ядро; stdout+stderr в res.stdout, код в res.results[-1]."""
        code = (
            "import subprocess as _sp\n"
            f"_p = _sp.run(['bash', '-lc', {cmd!r}], stdout=_sp.PIPE, stderr=_sp.STDOUT, text=True)\n"
            "print(_p.stdout, end='')\n"
            "print('__RC__', _p.returncode)\n"
        )
        res = self.execute(code, kernel_id=kernel_id, timeout=timeout)
        m = re.search(r"__RC__ (-?\d+)\s*$", res.stdout)
        if m:
            res.stdout = res.stdout[: m.start()]
            res.results.append(m.group(1))
            if m.group(1) != "0" and res.status == "ok":
                res.status = "error"
        return res

    def run_bg(self, cmd: str, name: str | None = None, workdir: str = "~") -> dict[str, Any]:
        """Запустить команду в фоне (nohup + setsid), лог ~/jhub_bg/<name>.log, pid ~/jhub_bg/<name>.pid."""
        name = name or time.strftime("bg_%Y%m%d_%H%M%S")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
            raise JHubError("name: только буквы, цифры, _.-")
        script = (
            f"mkdir -p ~/jhub_bg && cd {workdir} && "
            f"nohup setsid bash -lc {shlex.quote(cmd)} > ~/jhub_bg/{name}.log 2>&1 < /dev/null & "
            f"echo $! > ~/jhub_bg/{name}.pid; "
            f"printf '%s\\n' {shlex.quote(cmd)} > ~/jhub_bg/{name}.cmd; cat ~/jhub_bg/{name}.pid"
        )
        res = self.sh(script, timeout=60)
        if not res.ok:
            raise JHubError(f"run-bg: {res.stdout} {res.error}")
        pid = int(res.stdout.strip().splitlines()[-1])
        return {"name": name, "pid": pid, "log": f"~/jhub_bg/{name}.log"}

    def ps(self) -> str:
        script = (
            "echo '## процессы пользователя'; ps -u $(id -u) -o pid,ppid,etime,%cpu,%mem,args --sort=start_time "
            "| grep -v -E 'ps -u|grep -v' | cut -c1-200; "
            "echo; echo '## фоновые задачи (~/jhub_bg)'; "
            "for f in ~/jhub_bg/*.pid; do [ -e \"$f\" ] || continue; n=$(basename \"$f\" .pid); p=$(cat \"$f\"); "
            "if kill -0 \"$p\" 2>/dev/null; then s=RUNNING; else s=EXITED; fi; "
            "echo \"$n pid=$p $s cmd=$(cat ~/jhub_bg/$n.cmd 2>/dev/null)\"; done"
        )
        return self.sh(script, timeout=60).stdout


# ---------------- CLI ----------------


def _print_json(obj: Any) -> None:
    print(json.dumps(obj, ensure_ascii=False, indent=1, default=str))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="jhub.py",
        description="Jupyter/JupyterHub стенда LinQ: логин, файлы, выполнение кода, фоновые процессы.",
    )
    ap.add_argument("--env", default=str(DEFAULT_ENV), help="путь к .env (по умолчанию %(default)s)")
    ap.add_argument("--url", help="переопределить JUPYTER_URL")
    ap.add_argument("--token", help="API-токен (иначе JUPYTER_TOKEN или логин/пароль)")
    ap.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help="HTTP таймаут, с")
    ap.add_argument("--no-cache", action="store_true", help="не использовать кеш сессии")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("login", help="войти и сохранить сессию")
    p = sub.add_parser("ls", help="список файлов")
    p.add_argument("path", nargs="?", default="")
    p = sub.add_parser("upload", help="загрузить файл")
    p.add_argument("local")
    p.add_argument("remote")
    p.add_argument("--chunk-mb", type=float, default=UPLOAD_CHUNK / 1024 / 1024)
    p = sub.add_parser("download", help="скачать файл")
    p.add_argument("remote")
    p.add_argument("local", nargs="?")
    p = sub.add_parser("exec", help="выполнить python-код (строка или путь к .py) в ядре")
    p.add_argument("code")
    p.add_argument("--kernel", help="id существующего ядра (иначе временное)")
    p.add_argument("--exec-timeout", type=float, default=600.0)
    p = sub.add_parser("sh", help="выполнить shell-команду через ядро")
    p.add_argument("command")
    p.add_argument("--exec-timeout", type=float, default=600.0)
    p = sub.add_parser("run-bg", help="запустить команду в фоне (nohup, лог в ~/jhub_bg)")
    p.add_argument("command")
    p.add_argument("--name")
    p.add_argument("--workdir", default="~")
    sub.add_parser("ps", help="процессы пользователя и фоновые задачи")
    p = sub.add_parser("kernels", help="список ядер")
    p.add_argument("--stop", help="id ядра или all")
    p = sub.add_parser("tail", help="хвост файла на сервере")
    p.add_argument("remote")
    p.add_argument("-n", type=int, default=50)
    sub.add_parser("status", help="GET /api/status")

    a = ap.parse_args(argv)
    jh = JHub.from_env(a.env, url=a.url, token=a.token, timeout=a.timeout, verbose=a.verbose,
                       session_file=None if a.no_cache else SESSION_FILE)
    try:
        if a.cmd == "login":
            _print_json(jh.login())
        elif a.cmd == "status":
            _print_json(jh.status())
        elif a.cmd == "ls":
            for it in jh.ls(a.path):
                kind = "d" if it["type"] == "directory" else "-"
                size = it.get("size")
                print(f"{kind} {size if size is not None else '':>12} {it.get('last_modified', '')[:19]}  {it['path']}")
        elif a.cmd == "upload":
            _print_json(jh.upload(a.local, a.remote, chunk=int(a.chunk_mb * 1024 * 1024), progress=a.verbose))
        elif a.cmd == "download":
            _print_json(jh.download(a.remote, a.local))
        elif a.cmd in ("exec", "sh"):
            if a.cmd == "exec":
                src = a.code
                if src.endswith(".py") and Path(src).exists():
                    src = Path(src).read_text(encoding="utf-8")
                res = jh.execute(src, kernel_id=a.kernel, timeout=a.exec_timeout, stream_to=sys.stdout)
            else:
                res = jh.sh(a.command, timeout=a.exec_timeout)
                print(res.stdout, end="")
            if res.stderr:
                print(res.stderr, file=sys.stderr, end="")
            if res.error:
                print(f"{res.error['ename']}: {res.error['evalue']}", file=sys.stderr)
                tb = "\n".join(res.error.get("traceback") or [])
                print(re.sub(r"\x1b\[[0-9;]*m", "", tb), file=sys.stderr)
            return 0 if res.ok else 1
        elif a.cmd == "run-bg":
            _print_json(jh.run_bg(a.command, name=a.name, workdir=a.workdir))
        elif a.cmd == "ps":
            print(jh.ps(), end="")
        elif a.cmd == "kernels":
            ks = jh.kernels()
            if a.stop:
                for k in ks:
                    if a.stop in ("all", k["id"]):
                        jh.stop_kernel(k["id"])
                        print("stopped", k["id"])
            else:
                _print_json(ks)
        elif a.cmd == "tail":
            res = jh.sh(f"tail -n {int(a.n)} {a.remote}", timeout=60)
            print(res.stdout, end="")
    except (JHubError, requests.RequestException) as e:
        print(f"ошибка: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
