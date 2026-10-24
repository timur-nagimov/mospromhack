#!/usr/bin/env python3
"""Управление edge на LinQ через существующий клиент jhub.

Contents API стенда имеет корень ~/work; проект находится в ~/work/mosprom.
Окружение ~/guard311 должно быть установлено заранее.

Коды возврата: 0 — успешно, 1 — процесс остановлен / health не пройден,
2 — ошибка операции или аргументов, 130 — прерывание пользователем.

Passive declare показывает число готовых сообщений, но не их возраст.
health проверяет наличие сообщений и явно оставляет свежесть неподтверждённой.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shlex
import sys
import tarfile
import tempfile
from typing import Any
from urllib.parse import quote
import uuid

TOOLS = Path(__file__).resolve().parent
REPO = TOOLS.parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
import jhub  # noqa: E402

DEFAULT_URL = "http://127.0.0.1:8888/"
DEFAULT_ENV = Path(getattr(jhub, "DEFAULT_ENV", REPO / ".env"))
CODE_PATHS = ("guard", "configs", "ml/eval", "pyproject.toml")
MANIFEST = ".stand-manifest.json"
QUEUE = "zazerkalye.gateway"
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".mpeg", ".mpg", ".ts"}
IGNORED_DIRS = {"__pycache__", "node_modules"}
BLOCK = 1024 * 1024


class StandError(RuntimeError):
    """Сообщение об ошибке, пригодное для показа пользователю."""


def fingerprint(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(BLOCK):
            digest.update(chunk)
            size += len(chunk)
    return {
        "size": size,
        "sha256": digest.hexdigest(),
        "mode": path.stat().st_mode & 0o777,
    }


def manifest_digest(manifest: dict[str, Any]) -> str:
    raw = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def _code_name(name: str) -> bool:
    return name == "pyproject.toml" or any(
        name.startswith(prefix + "/") for prefix in CODE_PATHS[:-1]
    )


def validate_manifest(value: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(value, dict):
        raise StandError("Некорректный манифест синхронизации.")
    for name, entry in value.items():
        if (
            not isinstance(name, str)
            or "\\" in name
            or name.startswith("/")
            or any(part in ("", ".", "..") for part in name.split("/"))
            or not _code_name(name)
            or not isinstance(entry, dict)
            or type(entry.get("size")) is not int
            or entry["size"] < 0
            or not isinstance(entry.get("sha256"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"])
            or type(entry.get("mode")) is not int
            or not 0 <= entry["mode"] <= 0o777
        ):
            raise StandError("Некорректная запись в манифесте синхронизации.")
    return value


def build_manifest(root: Path) -> dict[str, dict[str, Any]]:
    root = Path(root)
    result: dict[str, dict[str, Any]] = {}
    for relative in CODE_PATHS:
        source = root / relative
        if source.is_symlink() or not source.exists():
            raise StandError(f"Отсутствует обычный путь проекта: {relative}.")
        paths = sorted(source.rglob("*")) if source.is_dir() else [source]
        for path in paths:
            name = path.relative_to(root).as_posix()
            parts = Path(name).parts
            if any(part.startswith(".") or part in IGNORED_DIRS for part in parts):
                continue
            if path.suffix in {".pyc", ".pyo"}:
                continue
            if path.is_symlink():
                raise StandError(f"Символьная ссылка не поддерживается: {name}.")
            if path.is_dir():
                continue
            if not path.is_file():
                raise StandError(f"Необычный тип файла: {name}.")
            result[name] = fingerprint(path)
    return validate_manifest(result)


def changed_files(
    current: dict[str, Any], previous: dict[str, Any]
) -> list[str]:
    return sorted(name for name, entry in current.items() if previous.get(name) != entry)


def build_archive(
    root: Path,
    current: dict[str, Any],
    previous: dict[str, Any],
    destination: Path,
) -> list[str]:
    validate_manifest(current)
    validate_manifest(previous)
    changed = changed_files(current, previous)
    with tarfile.open(destination, "w:gz") as archive:
        for name in changed:
            path = root / name
            if path.is_symlink() or fingerprint(path) != current[name]:
                raise StandError("Файлы изменились во время сборки архива; повторите sync.")
            info = archive.gettarinfo(str(path), arcname=name)
            if not info.isfile():
                raise StandError("В архив можно включать только обычные файлы.")
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            with path.open("rb") as stream:
                archive.addfile(info, stream)
        content = json.dumps(current, sort_keys=True, separators=(",", ":")).encode()
        info = tarfile.TarInfo(MANIFEST)
        info.size = len(content)
        info.mode = 0o600
        archive.addfile(info, io.BytesIO(content))
    return changed


def select_videos(root: Path, selection: str | None) -> list[Path]:
    directory = root / "data" / "videos"
    if directory.is_symlink() or not directory.is_dir():
        raise StandError("Нет обычного каталога data/videos.")
    files: list[Path] = []
    for path in sorted(directory.rglob("*")):
        relative = path.relative_to(directory)
        if any(part.startswith(".") for part in relative.parts):
            continue
        if path.is_symlink():
            raise StandError("Символьные ссылки в data/videos не поддерживаются.")
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES:
            files.append(path)
    if selection is None:
        return files
    identifiers = [item.strip() for item in selection.split(",")]
    if not identifiers or any(
        not item or "/" in item or "\\" in item or item in {".", ".."}
        for item in identifiers
    ):
        raise StandError("--videos: нужны имена или идентификаторы через запятую.")
    selected: set[Path] = set()
    for identifier in identifiers:
        matches = [p for p in files if identifier in (p.name, p.stem)]
        if len(matches) != 1:
            raise StandError(f"Видео не найдено или имя неоднозначно: {identifier}.")
        selected.add(matches[0])
    return sorted(selected)


# Только стандартная библиотека: код выполняется в ядре сервера и в guard311.
# Не используем extractall: проверяем состав архива и копируем обычные файлы.
REMOTE_SOURCE = r'''
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import signal
import subprocess
import tarfile
import tempfile
import time
import uuid

BASE = Path.home() / "work" / "mosprom"
MANIFEST = ".stand-manifest.json"


class Failure(RuntimeError):
    pass


def target(name):
    if (not isinstance(name, str) or not name or "\\" in name
            or PurePosixPath(name).is_absolute()
            or any(p in ("", ".", "..") for p in name.split("/"))):
        raise Failure("unsafe-path")
    p = BASE
    if p.is_symlink():
        raise Failure("symlink-destination")
    for part in name.split("/"):
        p = p / part
        if p.is_symlink():
            raise Failure("symlink-destination")
    return p


def digest(path):
    h = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
            size += len(chunk)
    return {"size": size, "sha256": h.hexdigest()}


def same_content(left, right):
    return all(left.get(k) == right.get(k) for k in ("size", "sha256"))


def validate(value):
    if not isinstance(value, dict):
        raise Failure("invalid-manifest")
    for name, entry in value.items():
        target(name)
        if not (name == "pyproject.toml"
                or any(name.startswith(p) for p in ("guard/", "configs/", "ml/eval/"))):
            raise Failure("invalid-manifest-path")
        if (not isinstance(entry, dict)
                or type(entry.get("size")) is not int or entry["size"] < 0
                or not isinstance(entry.get("sha256"), str)
                or not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"])
                or type(entry.get("mode")) is not int
                or not 0 <= entry["mode"] <= 0o777):
            raise Failure("invalid-manifest-entry")
    return value


def read_manifest():
    path = target(MANIFEST)
    return validate(json.loads(path.read_text())) if path.exists() else {}


def manifest_hash(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def prefix():
    root = Path(os.path.expanduser(
        os.environ.get("JUPYTER_SERVER_ROOT", "~/work")
    )).resolve()
    try:
        return BASE.resolve().relative_to(root).as_posix()
    except ValueError:
        raise Failure("project-outside-jupyter-root")


def stage(name):
    if not re.fullmatch(r"\.stand-upload-[0-9a-f]{32}\.(tar\.gz|part)", name):
        raise Failure("invalid-stage-name")
    return target(name)


def atomic_text(path, text):
    temporary = path.with_name("." + path.name + "-" + uuid.uuid4().hex)
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextlib.contextmanager
def locked(name):
    target(name)
    BASE.mkdir(parents=True, exist_ok=True)
    with target(name).open("a+b") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def apply_archive(request):
    source = stage(request["stage"])
    if digest(source)["sha256"] != request["sha256"]:
        raise Failure("archive-checksum-mismatch")
    with locked(".stand-sync.lock"):
        previous = read_manifest()
        if manifest_hash(previous) != request["expected"]:
            raise Failure("manifest-changed-retry-sync")
        with tarfile.open(source, "r:gz") as archive:
            members = archive.getmembers()
            names = [m.name for m in members]
            if len(set(names)) != len(names) or any(not m.isfile() for m in members):
                raise Failure("archive-must-contain-unique-regular-files")
            if MANIFEST not in names:
                raise Failure("archive-missing-manifest")
            with archive.extractfile(MANIFEST) as stream:
                current = validate(json.load(stream))
            changed = sorted(k for k, v in current.items() if previous.get(k) != v)
            if set(names) != set(changed) | {MANIFEST}:
                raise Failure("archive-members-mismatch")
            removed = sorted(set(previous) - set(current))
            for name in changed + removed:
                path = target(name)
                if path.exists() and not path.is_file():
                    raise Failure("destination-is-not-a-file")
            with tempfile.TemporaryDirectory(prefix=".stand-unpack-", dir=BASE) as tmp:
                prepared = {}
                for index, name in enumerate(changed):
                    path = Path(tmp) / str(index)
                    with archive.extractfile(name) as src, path.open("wb") as dst:
                        shutil.copyfileobj(src, dst, 1024 * 1024)
                    if not same_content(digest(path), current[name]):
                        raise Failure("file-checksum-mismatch")
                    path.chmod(current[name]["mode"])
                    prepared[name] = path
                for name in changed:
                    path = target(name)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(prepared[name], path)
                for name in removed:
                    target(name).unlink(missing_ok=True)
                atomic_text(target(MANIFEST), json.dumps(
                    current, sort_keys=True, separators=(",", ":")
                ))
    return {"changed": len(changed), "removed": len(removed)}


def video_target(name):
    if not isinstance(name, str) or not name.startswith("data/videos/"):
        raise Failure("invalid-video-path")
    return target(name)


def data_state(request):
    result = {}
    for name, expected in request["files"].items():
        path = video_target(name)
        if not path.exists():
            result[name] = None
        elif not path.is_file():
            raise Failure("video-is-not-a-file")
        elif path.stat().st_size != expected["size"]:
            result[name] = {"size": path.stat().st_size, "sha256": None}
        else:
            result[name] = digest(path)
    return {"prefix": prefix(), "files": result}


def put_video(request):
    source = stage(request["stage"])
    if not same_content(digest(source), request["entry"]):
        raise Failure("video-checksum-mismatch")
    with locked(".stand-sync.lock"):
        destination = video_target(request["name"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(source, destination)
    return {"uploaded": True}


def boot_id():
    return Path("/proc/sys/kernel/random/boot_id").read_text().strip()


def identity(pid):
    fields = Path("/proc/{}/stat".format(pid)).read_text().rsplit(")", 1)[1].split()
    return {"state": fields[0], "pgid": int(fields[2]), "start": fields[19]}


def process(metrics=False):
    result = {"alive": False, "state": "stopped", "pid": None,
              "rss_kib": None, "cpu_percent": None}
    pidfile = target("edge.pid")
    if not pidfile.exists():
        return result
    result["state"] = "stale"
    try:
        pid = int(pidfile.read_text().strip())
        if pid <= 1:
            return result
        result["pid"] = pid
        saved = json.loads(target(".edge-process.json").read_text())
        current = identity(pid)
        if (saved["pid"] != pid or saved["boot"] != boot_id()
                or saved["start"] != current["start"]
                or current["state"] in ("Z", "X")):
            return result
        result.update(alive=True, state="running", pgid=current["pgid"],
                      launch_id=saved["launch_id"])
    except (OSError, ValueError, KeyError, IndexError, TypeError):
        return result
    if metrics:
        try:
            output = subprocess.run(
                ["ps", "-p", str(pid), "-o", "rss=", "-o", "%cpu="],
                capture_output=True, text=True, timeout=5,
                env=dict(os.environ, LC_ALL="C"),
            )
            fields = output.stdout.split()
            if output.returncode == 0 and len(fields) == 2:
                result.update(rss_kib=int(fields[0]), cpu_percent=float(fields[1]))
        except (OSError, ValueError, subprocess.TimeoutExpired):
            pass
    return result


def tail(n):
    if not n:
        return ""
    path = target("edge.log")
    if not path.exists():
        return ""
    result = subprocess.run(
        ["tail", "-n", str(n), "--", str(path)],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=10,
    )
    if result.returncode:
        raise Failure("cannot-read-edge-log")
    return result.stdout


def launch(request):
    # run_bg создаёт отдельную сессию. exec сохраняет её PID и группу процессов.
    with locked(".edge-control.lock"):
        runtime_lock = target(".edge-runtime.lock").open("a+b")
        try:
            fcntl.flock(runtime_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise Failure("edge-already-running")
        if process()["alive"]:
            raise Failure("edge-already-running")
        pid = os.getpid()
        if os.getpgrp() != pid:
            raise Failure("edge-needs-own-process-group")
        os.set_inheritable(runtime_lock.fileno(), True)
        os.chdir(BASE)
        os.environ["OMP_NUM_THREADS"] = "2"
        os.environ["GUARD_BACKEND"] = "linq-tpu"
        with target("edge.log").open("ab", buffering=0) as log:
            os.dup2(log.fileno(), 1)
            os.dup2(log.fileno(), 2)
        saved = dict(pid=pid, boot=boot_id(), start=identity(pid)["start"],
                     launch_id=request["launch_id"])
        atomic_text(target(".edge-process.json"), json.dumps(saved))
        atomic_text(target("edge.pid"), str(pid) + "\n")
    os.execvpe("bash", ["bash", "-lc", request["command"]], os.environ)


def wait_started(request):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        result = process()
        if result.get("launch_id") == request["launch_id"]:
            time.sleep(0.5)
            result = process(metrics=True)
            result["started"] = (
                result["alive"] and result.get("launch_id") == request["launch_id"]
            )
            result["log"] = tail(20) if not result["started"] else ""
            return result
        time.sleep(0.1)
    result = process(metrics=True)
    result.update(started=False, log=tail(20))
    return result


def group_exists(pgid):
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False


def stop():
    with locked(".edge-control.lock"):
        current = process()
        if not current["alive"]:
            return current
        pid = current["pid"]
        if current["pgid"] != pid:
            raise Failure("refusing-to-signal-unmanaged-process-group")
        # Перепроверяем identity непосредственно перед отправкой сигнала.
        checked = process()
        if not checked["alive"] or checked.get("launch_id") != current["launch_id"]:
            return checked
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + 10
        while group_exists(pid) and time.monotonic() < deadline:
            time.sleep(0.1)
        if group_exists(pid):
            try:
                os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            time.sleep(0.2)
        result = process()
        if result["alive"]:
            raise Failure("edge-did-not-stop")
        target("edge.pid").unlink(missing_ok=True)
        target(".edge-process.json").unlink(missing_ok=True)
        return process()


def dispatch(request):
    op = request["op"]
    if op == "state":
        return {"prefix": prefix(), "manifest": read_manifest()}
    if op == "apply":
        return apply_archive(request)
    if op == "data-state":
        return data_state(request)
    if op == "put-video":
        return put_video(request)
    if op == "discard":
        stage(request["stage"]).unlink(missing_ok=True)
        return {}
    if op == "status":
        result = process(metrics=True)
        result["log"] = tail(request.get("n", 50))
        return result
    if op == "logs":
        return {"log": tail(request["n"])}
    if op == "launch":
        return launch(request)
    if op == "wait":
        return wait_started(request)
    if op == "stop":
        return stop()
    raise Failure("unknown-operation")
'''


class Stand:
    def __init__(self, client: Any, root: Path = REPO, parallel: int = 4):
        self.client = client
        self.root = Path(root)
        self.parallel = parallel

    def remote(self, op: str, **arguments: Any) -> Any:
        request = {"op": op, **arguments}
        code = REMOTE_SOURCE + "\n_request = " + repr(request) + """
try:
    _reply = {"ok": True, "value": dispatch(_request)}
except Failure as _error:
    _reply = {"ok": False, "reason": str(_error)}
except Exception:
    _reply = {"ok": False, "reason": "remote-operation-failed"}
print(json.dumps(_reply))
"""
        result = self.client.execute(code, timeout=600)
        if not result.ok:
            raise StandError(f"Удалённая операция {op} завершилась ошибкой или таймаутом.")
        try:
            reply = json.loads(result.stdout)
            if reply["ok"]:
                return reply["value"]
        except (ValueError, KeyError, TypeError):
            raise StandError("Некорректный ответ ядра Jupyter.") from None
        raise StandError(f"Сервер: {reply.get('reason', 'remote-operation-failed')}.")

    def upload(self, path: Path, remote: str) -> None:
        if self.parallel > 1:
            self.client.upload_parallel(path, remote, workers=self.parallel, part_mb=4)
        else:
            self.client.upload(path, remote)

    def discard(self, stage: str) -> None:
        try:
            self.remote("discard", stage=stage)
        except Exception:
            # Ошибка уборки не должна скрывать исходную ошибку загрузки.
            pass

    def sync(self) -> dict[str, Any]:
        current = build_manifest(self.root)
        state = self.remote("state")
        previous = validate_manifest(state["manifest"])
        if current == previous:
            return {"changed": 0, "removed": 0, "total": len(current)}
        stage = f".stand-upload-{uuid.uuid4().hex}.tar.gz"
        with tempfile.TemporaryDirectory(prefix="stand-sync-") as temporary:
            archive = Path(temporary) / "code.tar.gz"
            build_archive(self.root, current, previous, archive)
            try:
                self.upload(archive, f"{state['prefix']}/{stage}")
                result = self.remote(
                    "apply",
                    stage=stage,
                    sha256=fingerprint(archive)["sha256"],
                    expected=manifest_digest(previous),
                )
            finally:
                self.discard(stage)
        return {**result, "total": len(current)}

    def sync_data(self, videos: str | None = None) -> dict[str, Any]:
        paths = select_videos(self.root, videos)
        local = {p.relative_to(self.root).as_posix(): p for p in paths}
        entries = {name: fingerprint(path) for name, path in local.items()}
        if not entries:
            return {"uploaded": 0, "skipped": 0}
        state = self.remote("data-state", files=entries)
        uploaded = skipped = 0
        for name, path in local.items():
            entry = entries[name]
            existing = state["files"].get(name)
            if existing and all(existing.get(k) == entry[k] for k in ("size", "sha256")):
                skipped += 1
                continue
            stage = f".stand-upload-{uuid.uuid4().hex}.part"
            try:
                self.upload(path, f"{state['prefix']}/{stage}")
                self.remote("put-video", stage=stage, name=name, entry=entry)
                uploaded += 1
            finally:
                self.discard(stage)
        return {"uploaded": uploaded, "skipped": skipped}

    def status(self, n: int = 50) -> dict[str, Any]:
        return self.remote("status", n=n)

    def logs(self, n: int = 50) -> str:
        return self.remote("logs", n=n)["log"]

    def start(
        self,
        mode: str = "edge",
        config: str = "configs/cameras.stand.yaml",
        cmd: str | None = None,
    ) -> dict[str, Any]:
        if mode != "edge":
            raise StandError("Поддерживается только --mode edge.")
        if cmd is not None and not cmd.strip():
            raise StandError("--cmd не может быть пустой строкой.")
        current = self.status(n=0)
        if current["alive"]:
            return {**current, "already_running": True}
        command = cmd if cmd is not None else (
            "exec ~/guard311/bin/python -m guard.edge --config " + shlex.quote(config)
        )
        launch_id = uuid.uuid4().hex
        payload = {"op": "launch", "command": command, "launch_id": launch_id}
        source = REMOTE_SOURCE + "\ndispatch(" + repr(payload) + ")\n"
        wrapper = "exec ~/guard311/bin/python -c " + shlex.quote(source)
        self.client.run_bg(wrapper, name="stand-edge", workdir="~")
        result = self.remote("wait", launch_id=launch_id)
        if not result.get("started"):
            result["error"] = (
                "Запуск не подтверждён; проверьте edge.log "
                "и ~/jhub_bg/stand-edge.log."
            )
        return result

    def stop(self) -> dict[str, Any]:
        return self.remote("stop")

    def restart(self, **arguments: Any) -> dict[str, Any]:
        result = self.stop()
        if result["alive"]:
            raise StandError("Не удалось остановить edge перед restart.")
        return self.start(**arguments)

    def health(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "hub": {"ok": False},
            "server": {"ok": False},
            "edge": {"alive": False},
            "queue": {
                "ok": False,
                "name": QUEUE,
                "message_count": None,
                "freshness": "unknown",
            },
        }
        try:
            response = self.client.s.get(
                self.client.base + "/",
                timeout=self.client.timeout,
                allow_redirects=False,
            )
            result["hub"]["ok"] = 200 <= response.status_code < 400
        except Exception:
            pass
        try:
            self.client.status()
            result["server"]["ok"] = True
        except Exception:
            pass
        if result["server"]["ok"]:
            try:
                result["edge"] = self.status(n=0)
            except Exception:
                pass
        try:
            count = queue_message_count()
            result["queue"].update(ok=count > 0, message_count=count)
        except Exception:
            result["queue"]["error"] = "Брокер или очередь недоступны."
        result["ok"] = bool(
            result["hub"]["ok"]
            and result["server"]["ok"]
            and result["edge"]["alive"]
            and result["queue"]["ok"]
        )
        result["note"] = (
            "Passive declare не определяет возраст сообщений. "
            "Пустая очередь также возможна при работающем потребителе."
        )
        return result


def queue_message_count() -> int:
    import pika

    connection = pika.BlockingConnection(
        pika.ConnectionParameters(
            host="127.0.0.1",
            port=5672,
            connection_attempts=1,
            retry_delay=0,
            socket_timeout=5,
            stack_timeout=8,
            blocked_connection_timeout=5,
            heartbeat=10,
        )
    )
    try:
        reply = connection.channel().queue_declare(queue=QUEUE, passive=True)
        return int(reply.method.message_count)
    finally:
        try:
            connection.close()
        except Exception:
            pass


def redact(text: str, secrets: list[str]) -> str:
    text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text)
    variants = set()
    for secret in secrets:
        if secret:
            variants.update((secret, quote(secret, safe="")))
    for secret in sorted(variants, key=len, reverse=True):
        text = text.replace(secret, "[REDACTED]")
    text = re.sub(
        r"(?i)([a-z][a-z0-9+.-]*://[^/\s:@]+:)[^@\s/]+@",
        r"\1[REDACTED]@",
        text,
    )
    text = re.sub(
        r"(?i)([?&](?:token|password|secret|api_key)=)[^&\s]+",
        r"\1[REDACTED]",
        text,
    )
    text = re.sub(
        r"""(?i)(\b(?:password|passwd|pwd|token|secret|api[_-]?key)\b\s*[:=]\s*)("[^"]*"|'[^']*'|[^\s,;]+)""",
        r"\1[REDACTED]",
        text,
    )
    return text


def clean(value: Any, secrets: list[str]) -> Any:
    if isinstance(value, str):
        return redact(value, secrets)
    if isinstance(value, dict):
        return {key: clean(item, secrets) for key, item in value.items()}
    if isinstance(value, list):
        return [clean(item, secrets) for item in value]
    return value


def positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("нужно положительное целое число")
    return number


def nonnegative(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("число строк не может быть отрицательным")
    return number


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Синхронизация и управление edge на стенде LinQ.",
        epilog="Коды: 0 — успешно; 1 — остановлен/нездоров; 2 — ошибка; 130 — прерывание.",
    )

    def common(target: argparse.ArgumentParser, suppress: bool = False) -> None:
        def default(value: Any) -> Any:
            return argparse.SUPPRESS if suppress else value

        target.add_argument("--url", default=default(DEFAULT_URL), help="адрес JupyterHub")
        target.add_argument("--env", default=default(str(DEFAULT_ENV)), help="файл доступа .env")
        target.add_argument("--root", type=Path, default=default(REPO), help="корень проекта")
        target.add_argument(
            "--parallel", type=positive, default=default(4), help="потоки загрузки (4)"
        )

    common(parser)
    subparsers = parser.add_subparsers(dest="command", required=True)
    helps = {
        "sync": "код архивом, только изменения",
        "sync-data": "видео с проверкой размера и SHA-256",
        "start": "запустить edge",
        "stop": "остановить группу процессов edge",
        "status": "процесс, RSS, CPU и хвост лога",
        "logs": "хвост лога",
        "restart": "остановить и запустить edge",
        "health": "hub, сервер, edge и passive declare очереди",
    }
    for name, help_text in helps.items():
        subparser = subparsers.add_parser(name, help=help_text)
        common(subparser, suppress=True)
        if name == "sync-data":
            subparser.add_argument("--videos", help="имена или ID видео через запятую")
        if name in {"start", "restart"}:
            subparser.add_argument("--mode", choices=["edge"], default="edge")
            subparser.add_argument("--config", default="configs/cameras.stand.yaml")
            subparser.add_argument("--cmd", help="shell-команда вместо guard.edge")
        if name in {"logs", "status"}:
            subparser.add_argument("-n", type=nonnegative, default=50)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    client = None
    secrets: list[str] = []

    def known_secrets() -> list[str]:
        return secrets + [
            getattr(client, name, None) or ""
            for name in ("password", "token", "hub_api_token")
        ]

    try:
        env = jhub.load_env(args.env)
        secrets = [
            env.get("JUPYTER_PASSWORD", ""),
            env.get("JUPYTER_TOKEN", ""),
        ]
        client = jhub.JHub.from_env(args.env, url=args.url, verbose=False)
        stand = Stand(client, root=args.root, parallel=args.parallel)
        command = args.command
        if command == "sync":
            result = stand.sync()
        elif command == "sync-data":
            result = stand.sync_data(args.videos)
        elif command in {"start", "restart"}:
            result = getattr(stand, command)(
                mode=args.mode, config=args.config, cmd=args.cmd
            )
        elif command == "stop":
            result = stand.stop()
        elif command == "status":
            result = stand.status(args.n)
        elif command == "logs":
            print(redact(stand.logs(args.n), known_secrets()), end="")
            return 0
        else:
            result = stand.health()
        print(json.dumps(clean(result, known_secrets()), ensure_ascii=False, indent=2))
        if command == "health":
            return 0 if result["ok"] else 1
        if command in {"start", "restart"}:
            return 0 if result["alive"] and (
                result.get("started") or result.get("already_running")
            ) else 1
        if command == "status":
            return 0 if result["alive"] else 1
        return 0
    except KeyboardInterrupt:
        print("Прервано пользователем.", file=sys.stderr)
        return 130
    except StandError as error:
        print("Ошибка: " + redact(str(error), known_secrets()), file=sys.stderr)
        return 2
    except Exception:
        # HTTP/WebSocket/AMQP исключения могут содержать URL и секреты.
        print(
            "Ошибка операции: проверьте доступ к стенду, настройки и локальные файлы.",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
