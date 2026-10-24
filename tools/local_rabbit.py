"""Own a temporary loopback RabbitMQ node without a system service (POSIX)."""
from __future__ import annotations

import argparse
import contextlib
import json
import os
from pathlib import Path
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import time
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]


def _port(value: str) -> int:
    number = int(value)
    if not 1024 <= number <= 65535:
        raise argparse.ArgumentTypeError("port must be in 1024..65535")
    return number


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("start", "stop", "status"))
    p.add_argument("--state-dir", type=Path, required=True, help="new directory, or an existing node owned by this tool")
    p.add_argument("--port", type=_port, default=56783)
    p.add_argument("--dist-port", type=_port, default=25683)
    p.add_argument("--epmd-port", type=_port, default=43690)
    p.add_argument("--server", type=Path, help="rabbitmq-server executable (otherwise PATH/Homebrew)")
    p.add_argument("--timeout", type=float, default=120)
    return p


def _save(path: Path, value: dict, private: bool = False) -> None:
    temporary = path.with_suffix(".part")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600 if private else 0o644)
    with os.fdopen(fd, "w") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
    temporary.replace(path)


def _binary(given: Path | None) -> Path:
    found = shutil.which("rabbitmq-server")
    candidates = [given] if given else [Path(found) if found else None,
        Path("/opt/homebrew/opt/rabbitmq/sbin/rabbitmq-server"),
        Path("/usr/local/opt/rabbitmq/sbin/rabbitmq-server")]
    for candidate in candidates:
        if candidate and candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate.resolve()
    raise ValueError("Install RabbitMQ/Erlang first, or supply --server")


def _open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.2):
            return True
    except OSError:
        return False


def _load(base: Path) -> dict:
    state = json.loads((base / "state.json").read_text())
    if (state.get("schema") != "guard-local-rabbit-v1" or state.get("state_dir") != str(base)
            or state.get("host") != "127.0.0.1"
            or not str(state.get("node", "")).startswith("guard_local_")
            or not str(state.get("node", "")).endswith("@localhost")):
        raise ValueError("Refusing a directory not created by this tool")
    ports = [state.get(key) for key in ("port", "dist_port", "epmd_port")]
    if any(type(value) is not int or not 1024 <= value <= 65535 for value in ports) or len(set(ports)) != 3:
        raise ValueError("Invalid owned node ports")
    return state


def _prepare(args, base: Path) -> dict:
    ports = (args.port, args.dist_port, args.epmd_port)
    if len(set(ports)) != 3:
        raise ValueError("AMQP, distribution and epmd ports must differ")
    server = _binary(args.server)
    if base.exists():
        raise ValueError("Existing directory must contain state.json created by this tool")
    if any(_open(port) for port in ports):
        raise ValueError("A requested port is already occupied; choose distinct free ports")
    base.mkdir(parents=True, mode=0o700)
    node = "guard_local_" + uuid4().hex[:12] + "@localhost"
    state = {"schema": "guard-local-rabbit-v1", "state_dir": str(base), "host": "127.0.0.1",
             "node": node, "port": args.port, "dist_port": args.dist_port,
             "epmd_port": args.epmd_port, "server": str(server)}
    (base / "rabbitmq.conf").write_text(
        f"listeners.tcp.1 = 127.0.0.1:{args.port}\n"
        "distribution.listener.interface = 127.0.0.1\n"
        f"distribution.listener.port_range.min = {args.dist_port}\n"
        f"distribution.listener.port_range.max = {args.dist_port}\n"
        "vm_memory_high_watermark.absolute = 256MiB\ndisk_free_limit.absolute = 1GB\n")
    (base / "enabled_plugins").write_text("[].\n")
    (base / "rabbitmq-env.conf").write_text("")
    cookie = secrets.token_hex(24)
    erlang = shutil.which("erl")
    erl_dir = str(Path(erlang).parent) if erlang else "/opt/homebrew/opt/erlang/bin"
    if not Path(erl_dir, "erl").is_file():
        erl_dir = "/usr/local/opt/erlang/bin"
    env = {
        "PATH": erl_dir + os.pathsep + os.environ.get("PATH", ""), "HOSTNAME": "localhost",
        "RABBITMQ_NODENAME": node, "RABBITMQ_NODE_IP_ADDRESS": "127.0.0.1",
        "RABBITMQ_DIST_PORT": str(args.dist_port), "ERL_EPMD_ADDRESS": "127.0.0.1",
        "ERL_EPMD_PORT": str(args.epmd_port),
        "RABBITMQ_CONFIG_FILE": str(base / "rabbitmq.conf"),
        "RABBITMQ_CONF_ENV_FILE": str(base / "rabbitmq-env.conf"),
        "CONF_ENV_FILE": str(base / "rabbitmq-env.conf"),
        "RABBITMQ_MNESIA_DIR": str(base / "mnesia"), "RABBITMQ_LOG_BASE": str(base / "logs"),
        "RABBITMQ_PID_FILE": str(base / "rabbit.pid"),
        "RABBITMQ_ENABLED_PLUGINS_FILE": str(base / "enabled_plugins"),
        "RABBITMQ_PLUGINS_EXPAND_DIR": str(base / "plugins-expand"),
        "RABBITMQ_SERVER_ADDITIONAL_ERL_ARGS": f"+S 2:2 +A 4 -setcookie {cookie}",
        "RABBITMQ_CTL_ERL_ARGS": f"+S 2:2 -setcookie {cookie}",
    }
    _save(base / "private-env.json", env, private=True)
    _save(base / "state.json", state)
    command = [sys.executable, str(Path(__file__).resolve())]
    _save(base / "control.json", {"scope": "owned_loopback_broker", "host": "127.0.0.1", "port": args.port,
        "start": [*command, "start", "--state-dir", str(base)],
        "stop": [*command, "stop", "--state-dir", str(base)]})
    return state


def _handshake(port: int) -> str:
    import pika
    params = pika.ConnectionParameters(host="127.0.0.1", port=port, socket_timeout=1,
        stack_timeout=2, connection_attempts=1, credentials=pika.PlainCredentials("guest", "guest"))
    connection = pika.BlockingConnection(params)
    try:
        return str(connection._impl.server_properties.get("version", "unknown"))
    finally:
        connection.close()


def _stop_epmd(state: dict, env: dict) -> None:
    epmd = shutil.which("epmd", path=env.get("PATH"))
    if epmd and str(state.get("epmd_port")) == env.get("ERL_EPMD_PORT"):
        subprocess.run([epmd, "-port", str(state["epmd_port"]), "-kill"], env=env,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5)


def _start(base: Path, state: dict, env: dict, timeout: float) -> dict:
    if _open(state["port"]):
        raise ValueError("AMQP port is already listening; use status or stop the owned node first")
    if _open(state["dist_port"]):
        raise ValueError("Distribution port is already listening")
    with (base / "console.log").open("ab") as log:
        child = subprocess.Popen([state["server"]], env=env, stdout=log, stderr=log, start_new_session=True)
    deadline = time.monotonic() + timeout
    try:
        while time.monotonic() < deadline:
            if child.poll() is not None:
                raise RuntimeError("RabbitMQ exited; inspect the owned console.log")
            if _open(state["port"]):
                try:
                    version = _handshake(state["port"])
                    _save(base / "process.json", {"pid": child.pid, "rabbitmq_version": version})
                    return {"status": "started", "port": state["port"], "version": version}
                except Exception:
                    pass
            time.sleep(0.2)
        raise TimeoutError("RabbitMQ did not complete AMQP handshake before timeout")
    except BaseException:
        if child.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(child.pid, signal.SIGKILL)
                with contextlib.suppress(subprocess.TimeoutExpired):
                    child.wait(timeout=2)
        with contextlib.suppress(Exception):
            _stop_epmd(state, env)
        raise


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    base = args.state_dir.expanduser().resolve()
    try:
        if not 0 < args.timeout <= 120:
            raise ValueError("timeout must be finite and in (0, 120]")
        state = _load(base) if (base / "state.json").is_file() else _prepare(args, base) if args.action == "start" else _load(base)
        if args.action == "status":
            result = {"status": "listening" if _open(state["port"]) else "stopped", "port": state["port"]}
        else:
            env = {**os.environ, **json.loads((base / "private-env.json").read_text())}
            if args.action == "start":
                result = _start(base, state, env, args.timeout)
            else:
                ctl = Path(state["server"]).with_name("rabbitmqctl")
                outcome = subprocess.run([str(ctl), "--timeout", "20", "shutdown"], env=env,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=25)
                if outcome.returncode:
                    raise RuntimeError("Owned node did not confirm shutdown")
                _stop_epmd(state, env)
                if any(_open(state[key]) for key in ("port", "dist_port", "epmd_port")):
                    raise RuntimeError("Owned node ports are still listening")
                result = {"status": "stopped", "port": state["port"]}
        print(json.dumps({**result, "state_dir": str(base)}))
        return 0
    except Exception as exc:
        # No AMQP URL, cookie, environment or subprocess output in diagnostics.
        print(json.dumps({"status": "failed", "error": type(exc).__name__, "message": str(exc) if isinstance(exc, ValueError) else "Inspect the owned node logs", "state_dir": str(base)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
