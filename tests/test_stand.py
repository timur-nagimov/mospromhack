from __future__ import annotations

import ast
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import shlex
import shutil
import sys
import tarfile
import types

import pytest


@pytest.fixture
def stand_module(monkeypatch):
    fake_jhub = types.ModuleType("jhub")
    fake_jhub.JHubError = RuntimeError
    fake_jhub.load_env = lambda path: {}

    def unexpected_client(*args, **kwargs):
        raise AssertionError("Тест должен подменить клиент JHub")

    fake_jhub.JHub = types.SimpleNamespace(from_env=unexpected_client)
    monkeypatch.setitem(sys.modules, "jhub", fake_jhub)
    path = Path(__file__).resolve().parents[1] / "tools" / "stand.py"
    spec = importlib.util.spec_from_file_location("_cx015_stand_test_module", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "project"
    for name in ("guard", "configs", "ml/eval", "data/videos"):
        (root / name).mkdir(parents=True)
    (root / "guard/app.py").write_text("VALUE = 1\n")
    (root / "configs/cameras.stand.yaml").write_text("cameras: []\n")
    (root / "ml/eval/check.py").write_text("CHECK = True\n")
    (root / "pyproject.toml").write_text('[project]\nname = "guard"\n')
    return root


class FilesystemJHub:
    """Подмена транспорта: исполняет серверный код в изолированном HOME."""

    def __init__(self, home):
        self.home = home
        self.uploads = []
        self.after_upload = None

    def execute(self, code, timeout=600):
        output = io.StringIO()
        try:
            with contextlib.redirect_stdout(output):
                exec(compile(code, "<fake-jupyter-kernel>", "exec"), {})
        except Exception:
            return types.SimpleNamespace(ok=False, stdout=output.getvalue())
        return types.SimpleNamespace(ok=True, stdout=output.getvalue())

    def upload(self, local, remote):
        destination = self.home / "work" / remote
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local, destination)
        self.uploads.append((remote, destination.read_bytes()))
        if self.after_upload:
            self.after_upload(destination)
        return {}

    def upload_parallel(self, local, remote, workers=4, part_mb=4):
        assert workers >= 2
        assert part_mb == 4
        return self.upload(local, remote)


@pytest.fixture
def filesystem_client(tmp_path, monkeypatch):
    home = tmp_path / "server-home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("JUPYTER_SERVER_ROOT", raising=False)
    return FilesystemJHub(home)


def test_manifest_uses_content_and_ignores_caches(stand_module, project):
    cached = project / "guard/__pycache__"
    cached.mkdir()
    (cached / "app.pyc").write_bytes(b"cached")
    (project / "guard/.env").write_text("PASSWORD=private")
    first = stand_module.build_manifest(project)
    assert set(first) == {
        "guard/app.py",
        "configs/cameras.stand.yaml",
        "ml/eval/check.py",
        "pyproject.toml",
    }
    original = project / "guard/app.py"
    stat = original.stat()
    original.write_text("VALUE = 2\n")
    import os

    os.utime(original, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    second = stand_module.build_manifest(project)
    assert first["guard/app.py"]["size"] == second["guard/app.py"]["size"]
    assert stand_module.changed_files(second, first) == ["guard/app.py"]


def test_manifest_rejects_symlinks(stand_module, project, tmp_path):
    outside = tmp_path / "private.txt"
    outside.write_text("private")
    (project / "guard/link.py").symlink_to(outside)
    with pytest.raises(stand_module.StandError, match="ссылка"):
        stand_module.build_manifest(project)


def test_archive_contains_only_delta_and_manifest(stand_module, project, tmp_path):
    old = stand_module.build_manifest(project)
    (project / "guard/app.py").write_text("VALUE = 42\n")
    current = stand_module.build_manifest(project)
    archive = tmp_path / "delta.tar.gz"
    assert stand_module.build_archive(project, current, old, archive) == ["guard/app.py"]
    with tarfile.open(archive, "r:gz") as stream:
        assert stream.getnames() == ["guard/app.py", stand_module.MANIFEST]
        assert stream.extractfile("guard/app.py").read() == b"VALUE = 42\n"
        assert json.load(stream.extractfile(stand_module.MANIFEST)) == current


def test_archive_detects_local_change(stand_module, project, tmp_path):
    manifest = stand_module.build_manifest(project)
    (project / "guard/app.py").write_text("changed after hashing")
    with pytest.raises(stand_module.StandError, match="изменились"):
        stand_module.build_archive(project, manifest, {}, tmp_path / "delta.tar.gz")


def test_sync_roundtrip_noop_delta_and_deletion(
    stand_module, project, filesystem_client
):
    controller = stand_module.Stand(filesystem_client, project)
    remote_root = filesystem_client.home / "work/mosprom"
    first = controller.sync()
    assert first == {"changed": 4, "removed": 0, "total": 4}
    assert len(filesystem_client.uploads) == 1
    assert filesystem_client.uploads[0][0].startswith("mosprom/stand-upload-")
    assert (remote_root / "guard/app.py").read_text() == "VALUE = 1\n"
    assert controller.sync() == {"changed": 0, "removed": 0, "total": 4}
    assert len(filesystem_client.uploads) == 1

    (project / "guard/app.py").write_text("VALUE = 3\n")
    (project / "ml/eval/check.py").unlink()
    assert controller.sync() == {"changed": 1, "removed": 1, "total": 3}
    assert not (remote_root / "ml/eval/check.py").exists()
    assert (remote_root / "guard/app.py").read_text() == "VALUE = 3\n"
    assert json.loads((remote_root / stand_module.MANIFEST).read_text()) == (
        stand_module.build_manifest(project)
    )
    assert not list(remote_root.glob("stand-upload-*"))


def test_sync_rejects_corrupted_archive_without_committing(
    stand_module, project, filesystem_client
):
    controller = stand_module.Stand(filesystem_client, project)
    controller.sync()
    remote_root = filesystem_client.home / "work/mosprom"
    old_manifest = (remote_root / stand_module.MANIFEST).read_bytes()
    (project / "guard/app.py").write_text("new version\n")
    filesystem_client.after_upload = lambda path: path.write_bytes(b"broken")
    with pytest.raises(stand_module.StandError, match="archive-checksum-mismatch"):
        controller.sync()
    assert (remote_root / stand_module.MANIFEST).read_bytes() == old_manifest
    assert (remote_root / "guard/app.py").read_text() == "VALUE = 1\n"
    assert not list(remote_root.glob("stand-upload-*"))


def test_sync_detects_concurrent_manifest_change(
    stand_module, project, filesystem_client
):
    controller = stand_module.Stand(filesystem_client, project)
    controller.sync()
    remote_root = filesystem_client.home / "work/mosprom"
    (project / "guard/app.py").write_text("new version\n")

    def concurrent_change(path):
        (remote_root / stand_module.MANIFEST).write_text("{}")

    filesystem_client.after_upload = concurrent_change
    with pytest.raises(stand_module.StandError, match="manifest-changed"):
        controller.sync()
    assert (remote_root / "guard/app.py").read_text() == "VALUE = 1\n"


def test_remote_rejects_archive_symlinks(
    stand_module, project, filesystem_client, tmp_path
):
    manifest = {"guard/app.py": stand_module.fingerprint(project / "guard/app.py")}
    archive = tmp_path / "unsafe.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        member = tarfile.TarInfo("guard/app.py")
        member.type = tarfile.SYMTYPE
        member.linkname = "../../outside"
        stream.addfile(member)
        data = json.dumps(manifest).encode()
        member = tarfile.TarInfo(stand_module.MANIFEST)
        member.size = len(data)
        stream.addfile(member, io.BytesIO(data))
    stage = "stand-upload-" + "a" * 32 + ".tar.gz"
    filesystem_client.upload(archive, "mosprom/" + stage)
    controller = stand_module.Stand(filesystem_client, project)
    with pytest.raises(stand_module.StandError, match="regular-files"):
        controller.remote(
            "apply",
            stage=stage,
            sha256=stand_module.fingerprint(archive)["sha256"],
            expected=stand_module.manifest_digest({}),
        )
    assert not (filesystem_client.home / "work/mosprom/guard/app.py").exists()


def test_video_selection(stand_module, project):
    directory = project / "data/videos"
    (directory / "s1_01.mp4").write_bytes(b"one")
    (directory / "s2_01.mkv").write_bytes(b"two")
    (directory / "README.txt").write_text("not a video")
    selected = stand_module.select_videos(project, "s1_01,s2_01.mkv,s1_01")
    assert [p.name for p in selected] == ["s1_01.mp4", "s2_01.mkv"]
    assert len(stand_module.select_videos(project, None)) == 2
    for invalid in ("missing", "../s1_01", "", "s1_01,"):
        with pytest.raises(stand_module.StandError):
            stand_module.select_videos(project, invalid)
    (directory / "s1_01.mov").write_bytes(b"duplicate")
    with pytest.raises(stand_module.StandError, match="неоднозначно"):
        stand_module.select_videos(project, "s1_01")


def test_sync_data_checks_actual_remote_hash(
    stand_module, project, filesystem_client
):
    (project / "data/videos/s1_01.mp4").write_bytes(b"video")
    (project / "data/videos/s2_01.mp4").write_bytes(b"other")
    controller = stand_module.Stand(filesystem_client, project)
    assert controller.sync_data("s1_01") == {"uploaded": 1, "skipped": 0}
    assert controller.sync_data("s1_01") == {"uploaded": 0, "skipped": 1}
    remote = filesystem_client.home / "work/mosprom/data/videos/s1_01.mp4"
    remote.write_bytes(b"wrong")
    assert controller.sync_data("s1_01") == {"uploaded": 1, "skipped": 0}
    assert remote.read_bytes() == b"video"
    assert len(filesystem_client.uploads) == 2
    assert not remote.with_name("s2_01.mp4").exists()


def test_failed_video_upload_preserves_destination(
    stand_module, project, filesystem_client
):
    local = project / "data/videos/s1_01.mp4"
    local.write_bytes(b"first")
    controller = stand_module.Stand(filesystem_client, project, parallel=1)
    controller.sync_data()
    local.write_bytes(b"second")
    filesystem_client.after_upload = lambda path: path.write_bytes(b"broken")
    with pytest.raises(stand_module.StandError, match="video-checksum"):
        controller.sync_data()
    remote = filesystem_client.home / "work/mosprom/data/videos/s1_01.mp4"
    assert remote.read_bytes() == b"first"


class ScriptedJHub:
    def __init__(self, alive=False):
        self.base = "http://127.0.0.1:8888"
        self.timeout = 30
        self.password = "private-password"
        self.token = None
        self.hub_api_token = None
        self.s = types.SimpleNamespace(
            get=lambda *args, **kwargs: types.SimpleNamespace(status_code=200)
        )
        self.alive = alive
        self.calls = []
        self.background = []
        self.log = "line one\nline two\n"
        self.fail_start = False

    def status(self):
        return {}

    def run_bg(self, command, name=None, workdir="~"):
        self.background.append((command, name, workdir))
        return {"pid": 123, "name": name, "log": "~/jhub_bg/stand-edge.log"}

    def execute(self, code, timeout=600):
        request_line = next(
            line for line in code.splitlines() if line.startswith("_request = ")
        )
        request = ast.literal_eval(request_line[len("_request = "):])
        self.calls.append(request)
        op = request["op"]
        state = {
            "alive": self.alive,
            "pid": 123 if self.alive else None,
            "rss_kib": 1024 if self.alive else None,
            "cpu_percent": 2.5 if self.alive else None,
            "log": "",
        }
        if op == "status":
            value = state
        elif op == "logs":
            value = {"log": self.log}
        elif op == "stop":
            self.alive = False
            value = {**state, "alive": False}
        elif op == "wait":
            self.alive = not self.fail_start
            value = {**state, "alive": self.alive, "started": self.alive}
        else:
            raise AssertionError(f"Неожиданная операция: {op}")
        return types.SimpleNamespace(
            ok=True, stdout=json.dumps({"ok": True, "value": value})
        )


@pytest.mark.parametrize(
    "argv,command",
    [
        (["sync"], "sync"),
        (["sync-data", "--videos", "s1_01,s2_01"], "sync-data"),
        (["start"], "start"),
        (["restart", "--mode", "edge", "--cmd", "sleep 10"], "restart"),
        (["stop"], "stop"),
        (["status"], "status"),
        (["logs", "-n", "0"], "logs"),
        (["health"], "health"),
    ],
)
def test_parser_commands(stand_module, argv, command):
    arguments = stand_module.make_parser().parse_args(argv)
    assert arguments.command == command
    assert arguments.url == "http://127.0.0.1:8888/"
    assert arguments.parallel == 4


def test_parser_global_options_before_or_after_command(stand_module):
    parser = stand_module.make_parser()
    assert parser.parse_args(["--url", "http://example/", "status"]).url == (
        "http://example/"
    )
    assert parser.parse_args(["status", "--url", "http://example/"]).url == (
        "http://example/"
    )
    for argv in (["logs", "-n", "-1"], ["sync", "--parallel", "0"], ["start", "--mode", "x"]):
        with pytest.raises(SystemExit) as error:
            parser.parse_args(argv)
        assert error.value.code == 2


def test_start_quotes_config_and_custom_command(stand_module):
    client = ScriptedJHub()
    controller = stand_module.Stand(client)
    config = "configs/camera 'north'; echo unsafe.yaml"
    assert controller.start(config=config)["started"]
    wrapper = shlex.split(client.background[0][0])
    assert wrapper[:3] == ["exec", "~/guard311/bin/python", "-c"]
    tree = ast.parse(wrapper[3])
    request = ast.literal_eval(tree.body[-1].value.args[0])
    command = shlex.split(request["command"])
    assert command == [
        "exec", "~/guard311/bin/python", "-m", "guard.edge", "--config", config,
        "--backend", "linq-tpu", "--transport", "rabbit",
    ]
    assert client.background[0][1:] == ("stand-edge", "~")

    client.alive = False
    custom = "printf '%s\\n' 'hello world'; sleep 30"
    controller.start(cmd=custom)
    tree = ast.parse(shlex.split(client.background[-1][0])[3])
    assert ast.literal_eval(tree.body[-1].value.args[0])["command"] == custom


def test_start_is_idempotent_and_restart_stops_first(stand_module):
    client = ScriptedJHub(alive=True)
    controller = stand_module.Stand(client)
    assert controller.start()["already_running"]
    assert client.background == []
    assert controller.restart(cmd="sleep 30")["started"]
    assert [request["op"] for request in client.calls] == [
        "status", "stop", "status", "wait"
    ]


def test_remote_launch_forces_external_tpu_without_cpu_fallback(stand_module, tmp_path):
    import contextlib

    namespace = {}
    exec(compile(stand_module.REMOTE_SOURCE, "<remote-library>", "exec"), namespace)
    executed = []
    fake_os = types.SimpleNamespace(
        environ={"GUARD_BACKEND": "onnx-cpu", "GUARD_BACKEND_FALLBACK": "onnx-cpu"},
        getpid=lambda: 111, getpgrp=lambda: 111, chdir=lambda path: None,
        set_inheritable=lambda *args: None, dup2=lambda *args: None,
        execvpe=lambda *args: executed.append(args),
    )
    namespace.update(
        os=fake_os, target=lambda name: tmp_path / name,
        locked=lambda name: contextlib.nullcontext(),
        fcntl=types.SimpleNamespace(flock=lambda *args: None, LOCK_EX=1, LOCK_NB=2),
        process=lambda: {"alive": False}, boot_id=lambda: "boot", identity=lambda pid: {"start": "start"},
        atomic_text=lambda *args: None,
    )
    namespace["launch"]({"launch_id": "new", "command": "exec python -m guard.edge"})
    assert len(executed) == 1
    environment = executed[0][2]
    assert environment["GUARD_MODE"] == "edge"
    assert environment["GUARD_BACKEND"] == "linq-tpu"
    assert environment["GUARD_BACKEND_FALLBACK"] == "none"
    assert environment["LINQ_RAW"] == "1" and environment["LINQ_TIMEOUT_S"] == "5"


def test_stale_pid_does_not_receive_signal(
    stand_module, filesystem_client, monkeypatch
):
    namespace = {}
    exec(compile(stand_module.REMOTE_SOURCE, "<remote-library>", "exec"), namespace)
    remote_root = filesystem_client.home / "work/mosprom"
    remote_root.mkdir(parents=True)
    (remote_root / "edge.pid").write_text("123\n")
    (remote_root / ".edge-process.json").write_text(json.dumps({
        "pid": 123, "boot": "same-boot", "start": "old", "launch_id": "old-launch"
    }))
    namespace["boot_id"] = lambda: "same-boot"
    namespace["identity"] = lambda pid: {"state": "S", "start": "new", "pgid": pid}

    def forbidden_signal(*args):
        raise AssertionError("Нельзя сигналить переиспользованному PID")

    monkeypatch.setattr(namespace["os"], "killpg", forbidden_signal)
    assert namespace["stop"]()["alive"] is False


def test_queue_check_only_passively_declares_and_closes(stand_module, monkeypatch):
    events = []

    class Channel:
        def queue_declare(self, **kwargs):
            events.append(("declare", kwargs))
            return types.SimpleNamespace(
                method=types.SimpleNamespace(message_count=7)
            )

    class Connection:
        def channel(self):
            return Channel()

        def close(self):
            events.append(("close",))

    parameters = {}

    def make_parameters(**kwargs):
        parameters.update(kwargs)
        return kwargs

    fake_pika = types.SimpleNamespace(
        ConnectionParameters=make_parameters,
        BlockingConnection=lambda params: Connection(),
    )
    monkeypatch.setitem(sys.modules, "pika", fake_pika)
    assert stand_module.queue_message_count() == 7
    assert parameters["host"] == "127.0.0.1"
    assert parameters["port"] == 5672
    assert events == [
        ("declare", {"queue": "guard.gateway.stand-linq.events", "passive": True}),
        ("close",),
    ]


def test_queue_connection_closed_on_declare_failure(stand_module, monkeypatch):
    closed = []

    def fail(**kwargs):
        raise RuntimeError("missing queue")

    connection = types.SimpleNamespace(
        channel=lambda: types.SimpleNamespace(queue_declare=fail),
        close=lambda: closed.append(True),
    )
    monkeypatch.setitem(sys.modules, "pika", types.SimpleNamespace(
        ConnectionParameters=lambda **kwargs: kwargs,
        BlockingConnection=lambda params: connection,
    ))
    with pytest.raises(RuntimeError):
        stand_module.queue_message_count()
    assert closed == [True]


def test_health_distinguishes_queue_count_from_freshness(stand_module, monkeypatch):
    controller = stand_module.Stand(ScriptedJHub(alive=True))
    monkeypatch.setattr(stand_module, "queue_message_count", lambda: 3)
    result = controller.health()
    assert result["ok"]
    assert result["queue"]["message_count"] == 3
    assert result["queue"]["freshness"] == "unknown"
    monkeypatch.setattr(stand_module, "queue_message_count", lambda: 0)
    assert not controller.health()["ok"]


@pytest.mark.parametrize(
    "argv,alive,fail_start,expected",
    [
        (["status"], False, False, 1),
        (["status"], True, False, 0),
        (["stop"], False, False, 0),
        (["start"], False, False, 0),
        (["start"], False, True, 1),
        (["restart"], True, False, 0),
    ],
)
def test_main_return_codes(
    stand_module, monkeypatch, capsys, argv, alive, fail_start, expected
):
    client = ScriptedJHub(alive=alive)
    client.fail_start = fail_start
    constructed = []

    def from_env(path, **kwargs):
        constructed.append(kwargs)
        return client

    monkeypatch.setattr(stand_module.jhub.JHub, "from_env", from_env)
    assert stand_module.main(argv) == expected
    assert constructed == [{"url": stand_module.DEFAULT_URL, "timeout": 300.0, "verbose": False}]
    output = capsys.readouterr()
    assert "private-password" not in output.out + output.err


def test_logs_redact_credentials_and_errors_do_not_echo_them(
    stand_module, monkeypatch, capsys
):
    client = ScriptedJHub()
    client.log = (
        "known private-password\n"
        "password=another-password\n"
        "amqp://guest:broker-password@localhost/\n"
    )
    monkeypatch.setattr(stand_module.jhub.JHub, "from_env", lambda *a, **kw: client)
    assert stand_module.main(["logs", "-n", "3"]) == 0
    output = capsys.readouterr().out
    for secret in ("private-password", "another-password", "broker-password"):
        assert secret not in output
    assert "[REDACTED]" in output
    assert client.calls[-1] == {"op": "logs", "n": 3}

    def fail(*args, **kwargs):
        raise RuntimeError("HTTP error password=do-not-print-me")

    monkeypatch.setattr(stand_module.jhub.JHub, "from_env", fail)
    assert stand_module.main(["status"]) == 2
    output = capsys.readouterr()
    assert "do-not-print-me" not in output.out + output.err
    assert "Ошибка" in output.err


def test_help_does_not_construct_client(stand_module, capsys):
    with pytest.raises(SystemExit) as error:
        stand_module.main(["--help"])
    assert error.value.code == 0
    output = capsys.readouterr().out
    assert "sync-data" in output
    assert "health" in output


@pytest.mark.parametrize(
    "argv,expected_timeout",
    [
        (["status"], 300.0),
        (["--timeout", "45", "status"], 45.0),
        (["status", "--timeout", "45"], 45.0),
        (["status", "--timeout", "0.5"], 0.5),
    ],
)
def test_main_http_timeout(
    stand_module, monkeypatch, argv, expected_timeout
):
    arguments = stand_module.make_parser().parse_args(argv)
    assert arguments.timeout == expected_timeout
    assert isinstance(arguments.timeout, float)
    client = ScriptedJHub(alive=True)
    constructed = []

    def from_env(path, **kwargs):
        constructed.append((path, kwargs))
        return client

    monkeypatch.setattr(stand_module.jhub.JHub, "from_env", from_env)
    assert stand_module.main(argv) == 0
    assert constructed == [
        (
            str(stand_module.DEFAULT_ENV),
            {
                "url": stand_module.DEFAULT_URL,
                "timeout": expected_timeout,
                "verbose": False,
            },
        )
    ]


@pytest.mark.parametrize("value", ["0", "-1", "nan", "abc", "inf", "1e309"])
@pytest.mark.parametrize("before_command", [True, False])
def test_main_rejects_invalid_http_timeout(
    stand_module, monkeypatch, capsys, value, before_command
):
    constructed = []

    def from_env(*args, **kwargs):
        constructed.append((args, kwargs))
        return ScriptedJHub(alive=True)

    monkeypatch.setattr(stand_module.jhub.JHub, "from_env", from_env)
    argv = (
        ["--timeout", value, "status"]
        if before_command
        else ["status", "--timeout", value]
    )
    with pytest.raises(SystemExit) as error:
        stand_module.main(argv)
    assert error.value.code == 2
    assert constructed == []
    assert "нужно положительное число секунд" in capsys.readouterr().err


def test_main_sync_reports_exception_class_without_details(
    stand_module, project, filesystem_client, monkeypatch, capsys
):
    class ReadTimeout(Exception):
        pass

    uploads = []

    def fail_upload(local, remote):
        uploads.append((local, remote))
        raise ReadTimeout(
            "HTTP upload timed out: "
            "http://secret-host:8888/api/contents?token=leak-token"
        )

    monkeypatch.setattr(filesystem_client, "upload", fail_upload)
    monkeypatch.setattr(
        stand_module.jhub.JHub, "from_env",
        lambda *args, **kwargs: filesystem_client,
    )
    assert stand_module.main([
        "sync", "--root", str(project), "--parallel", "1",
    ]) == 2
    assert len(uploads) == 1
    output = capsys.readouterr()
    assert output.err == (
        "Ошибка операции (ReadTimeout): "
        "проверьте доступ к стенду, настройки и локальные файлы.\n"
    )
    for secret in ("secret-host", "leak-token", "http://"):
        assert secret not in output.out + output.err
