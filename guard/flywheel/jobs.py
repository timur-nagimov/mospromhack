"""Задачи маховика (POST /api/flywheel/jobs): CLI ml/flywheel/<kind>.py подпроцессом с логом.

- лог — data/flywheel/jobs/<id>.log, описание — <id>.json рядом (переживает перезапуск сервиса);
- тяжёлые задачи идут через tools/heavy.sh: autolabel, gate и deploy — полоса default, train — полоса train;
- train и deploy разрешены только при GUARD_FLYWHEEL_ALLOW_TRAIN=1, иначе JobForbidden (HTTP 403);
- если модуля ml/flywheel/<kind>.py ещё нет, задача сразу падает с понятной ошибкой;
- args → флаги CLI: {"limit": 10, "include_holdout": true, "classes": ["a", "b"]} →
  --limit 10 --include-holdout --classes a,b. Хранилище CLI получает через GUARD_FLYWHEEL_DIR.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from guard.pipeline.settings import CODE_ROOT

log = logging.getLogger(__name__)

JOB_KINDS = ("autolabel", "export", "train", "gate", "deploy")
TRAIN_KINDS = frozenset({"train", "deploy"})
# Полоса tools/heavy.sh; None — без очереди (лёгкая задача)
HEAVY_LANES: dict[str, str | None] = {"autolabel": "default", "export": None, "train": "train", "gate": "default",
                                      "deploy": "default"}
OWNERS = {"train": "ds-forklift", "gate": "ds-forklift", "deploy": "ds-forklift"}
_ARG_NAME = re.compile(r"^[a-z][a-z0-9_]*$")


class JobError(ValueError):
    """Некорректная задача (HTTP 422)."""


class JobForbidden(PermissionError):
    """Задача запрещена настройками (HTTP 403)."""


@dataclass
class Job:
    id: str
    kind: str
    args: dict[str, Any]
    status: str = "queued"  # queued | running | done | failed
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    returncode: int | None = None
    error: str | None = None
    cmd: list[str] = field(default_factory=list)
    pid: int | None = None

    @property
    def log_url(self) -> str:
        return f"/api/flywheel/jobs/{self.id}/log"


def args_to_cli(args: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for key, value in args.items():
        if not _ARG_NAME.match(str(key)):
            raise JobError(f"bad argument name {key!r}")
        flag = "--" + str(key).replace("_", "-")
        if value is None or value is False:
            continue
        if value is True:
            out.append(flag)
        elif isinstance(value, (list, tuple)):
            if not all(isinstance(v, (str, int, float)) and not isinstance(v, bool) for v in value):
                raise JobError(f"argument {key}: list items must be strings or numbers")
            out += [flag, ",".join(str(v) for v in value)]
        elif isinstance(value, (str, int, float)):
            out += [flag, str(value)]
        else:
            raise JobError(f"argument {key}: unsupported value type {type(value).__name__}")
    return out


class JobManager:
    def __init__(
        self,
        root: Path,
        *,
        allow_train: bool = False,
        python: str | None = None,
        code_root: Path = CODE_ROOT,
        heavy_script: Path | None = None,
    ) -> None:
        self.root = Path(root)
        self.jobs_dir = self.root / "jobs"
        self.allow_train = allow_train
        self.python = python or sys.executable
        self.code_root = Path(code_root)
        self.heavy_script = heavy_script or self.code_root / "tools" / "heavy.sh"
        self._jobs: dict[str, Job] = {}
        self._procs: dict[str, subprocess.Popen] = {}
        self._lock = threading.Lock()
        self._load()

    # ---- хранение ----

    def _load(self) -> None:
        if not self.jobs_dir.is_dir():
            return
        for path in self.jobs_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                job = Job(**{k: v for k, v in data.items() if k in Job.__dataclass_fields__})
            except (OSError, ValueError, TypeError):
                continue
            if job.status in ("queued", "running"):
                # процесс в своей сессии мог пережить перезапуск сервиса: итог — в логе
                job.status, job.error = "failed", "сервис перезапущен, пока задача работала; итог смотрите в логе"
            self._jobs[job.id] = job

    def _save(self, job: Job) -> None:
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.jobs_dir / f"{job.id}.json.tmp"
        tmp.write_text(json.dumps(asdict(job), ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, self.jobs_dir / f"{job.id}.json")

    def log_path(self, job_id: str) -> Path:
        return self.jobs_dir / f"{job_id}.log"

    # ---- запуск ----

    def module_path(self, kind: str) -> Path:
        return self.code_root / "ml" / "flywheel" / f"{kind}.py"

    def command(self, kind: str, args: dict[str, Any]) -> list[str]:
        cmd = [self.python, "-m", f"ml.flywheel.{kind}", *args_to_cli(args)]
        if HEAVY_LANES.get(kind):
            cmd = [str(self.heavy_script), *cmd]
        return cmd

    def start(self, kind: str, args: dict[str, Any] | None = None) -> Job:
        if kind not in JOB_KINDS:
            raise JobError(f"kind must be one of {JOB_KINDS}")
        if kind in TRAIN_KINDS and not self.allow_train:
            raise JobForbidden(
                f"задача {kind} запрещена: обучение и выкатка из UI включаются переменной GUARD_FLYWHEEL_ALLOW_TRAIN=1"
            )
        args = dict(args or {})
        job = Job(id=uuid.uuid4().hex[:12], kind=kind, args=args)
        job.cmd = self.command(kind, args)  # проверка аргументов до регистрации задачи
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        with self._lock:
            self._jobs[job.id] = job
        if not self.module_path(kind).is_file():
            owner = OWNERS.get(kind)
            job.status, job.finished_at = "failed", time.time()
            job.error = f"ml/flywheel/{kind}.py ещё нет в main" + (f" (модуль пишет {owner})" if owner else "")
            self.log_path(job.id).write_text(job.error + "\n", encoding="utf-8")
            self._save(job)
            return job
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        env["PYTHONPATH"] = str(self.code_root)
        env["GUARD_FLYWHEEL_DIR"] = str(self.root)
        env["PYTHONUNBUFFERED"] = "1"
        lane = HEAVY_LANES.get(kind)
        if lane:
            env["HEAVY_LANE"] = lane
        log_file = open(self.log_path(job.id), "ab")  # noqa: SIM115 — закрывает поток ожидания
        log_file.write(f"$ {' '.join(job.cmd)}\n".encode())
        log_file.flush()
        try:
            proc = subprocess.Popen(
                job.cmd, cwd=str(self.code_root), env=env, stdout=log_file, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, start_new_session=True,
            )
        except OSError as exc:
            log_file.close()
            job.status, job.finished_at, job.error = "failed", time.time(), f"не запустилась: {exc}"
            self._save(job)
            return job
        job.status, job.started_at, job.pid = "running", time.time(), proc.pid
        self._procs[job.id] = proc
        self._save(job)
        threading.Thread(target=self._wait, args=(job, proc, log_file), name=f"flywheel-job-{job.id}",
                         daemon=True).start()
        return job

    def _wait(self, job: Job, proc: subprocess.Popen, log_file: Any) -> None:
        try:
            code = proc.wait()
        finally:
            log_file.close()
        job.returncode = code
        job.finished_at = time.time()
        job.status = "done" if code == 0 else "failed"
        if code != 0 and not job.error:
            job.error = f"код выхода {code}, подробности в логе"
        with self._lock:
            self._procs.pop(job.id, None)
        self._save(job)

    # ---- чтение ----

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def list(self) -> list[Job]:
        return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)

    def tail(self, job_id: str, max_bytes: int = 4000) -> str:
        path = self.log_path(job_id)
        if not path.is_file():
            return ""
        with path.open("rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            fh.seek(max(0, size - max_bytes))
            return fh.read().decode("utf-8", errors="replace")

    def to_dict(self, job: Job, tail: bool = True) -> dict[str, Any]:
        out = {**asdict(job), "log_url": job.log_url}
        if tail:
            out["log_tail"] = self.tail(job.id)
        return out

    def stop_all(self) -> None:
        for proc in list(self._procs.values()):
            if proc.poll() is None:
                try:
                    proc.terminate()
                except OSError:
                    pass
