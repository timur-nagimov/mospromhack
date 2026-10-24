"""Бэкенд LinQ TPU: инференс через RabbitMQ с быстрым отказом и circuit breaker.

Модель на TPU может быть «головой» сети; ONNX-хвост выполняется на CPU.
После трёх неуспешных вызовов сеть не используется в течение 10 секунд.
По истечении интервала допускается один пробный вызов.

Окружение: LINQ_RABBIT_HOST/PORT, LINQ_RABBIT_USER/PASSWORD,
LINQ_PROGRAM_DIR, LINQ_TIMEOUT_S. Таймаут ответа по умолчанию — 2 секунды.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from guard.contracts import ModelSpec
from guard.inference.base import BackendStats, InferenceBackend, resolve_model_path
from guard.pipeline.settings import find_dotenv, project_root

log = logging.getLogger(__name__)

EXCHANGE = "tpu.direct"
INPUT_ROUTING_KEY = "input"
URI_HEADER = "uri"
ROUTING_KEYS_HEADER = "response_routing_keys"
MESSAGE_ID_HEADER = "tpu_pycloud_message_id"

ROOT = project_root()


def _dotenv() -> dict[str, str]:
    env: dict[str, str] = {}
    p = find_dotenv()
    if p is None:
        return env
    try:
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                k = k.strip().removeprefix("export ").strip()
                if k.startswith("LINQ_") or k == "JUPYTER_LOGIN":
                    env[k] = v.strip().strip('"').strip("'")
    except OSError:
        pass
    return env


def _default_host() -> str:
    """На стенде подключаемся напрямую, с ноутбука — через локальный туннель."""
    if os.environ.get("JUPYTERHUB_USER") or Path("/auto/cache").is_dir():
        return "10.0.114.30"
    return "127.0.0.1"


def _cfg(*names: str, default: str | None = None) -> str | None:
    dot = None
    for n in names:
        if os.environ.get(n):
            return os.environ[n]
    for n in names:
        if dot is None:
            dot = _dotenv()
        if dot.get(n):
            return dot[n]
    return default


# ---------------------------------------------------------------- wire format

_PROTO_DESCRIPTOR = (
    b'\n\rmessage.proto\x1a\x1fgoogle/protobuf/timestamp.proto\"8\n\x05\x41rray\x12\r\n\x05\x64type\x18\x01 '
    b'\x01(\t\x12\r\n\x05shape\x18\x02 \x03(\x05\x12\x11\n\tbyte_data\x18\x03 \x01(\x0c\"m\n\x0cMetadataDict'
    b'\x12\'\n\x05items\x18\x01 \x03(\x0b\x32\x18.MetadataDict.ItemsEntry\x1a\x34\n\nItemsEntry\x12\x0b\n\x03key'
    b'\x18\x01 \x01(\t\x12\x15\n\x05value\x18\x02 \x01(\x0b\x32\x06.Array:\x02\x38\x01\"\xce\x02\n\x04Task\x12&'
    b'\n\tdict_data\x18\x01 \x03(\x0b\x32\x13.Task.DictDataEntry\x12$\n\rlist_metadata\x18\x02 \x03(\x0b\x32\r'
    b'.MetadataDict\x12\x11\n\tdata_path\x18\x03 \x01(\t\x12\x12\n\nworker_pid\x18\x04 \x01(\t\x12\x13\n\x0b'
    b'thread_name\x18\x05 \x01(\t\x12\x16\n\x0einference_time\x18\x06 \x01(\x02\x12\x13\n\x0bstatus_code\x18'
    b'\x07 \x01(\x05\x12\x15\n\rerror_message\x18\x08 \x01(\t\x12?\n\x1binference_opening_timestamp\x18\t '
    b'\x01(\x0b\x32\x1a.google.protobuf.Timestamp\x1a\x37\n\rDictDataEntry\x12\x0b\n\x03key\x18\x01 \x01(\t'
    b'\x12\x15\n\x05value\x18\x02 \x01(\x0b\x32\x06.Array:\x02\x38\x01\x62\x06proto3'
)
_TASK_CLS: Any = None
_TASK_LOCK = threading.Lock()


def _task_cls() -> Any:
    global _TASK_CLS
    if _TASK_CLS is not None:
        return _TASK_CLS
    with _TASK_LOCK:
        if _TASK_CLS is None:
            try:
                from tpu_pycloud_format import message_pb2  # type: ignore

                _TASK_CLS = message_pb2.Task
            except ImportError:
                from google.protobuf import descriptor_pool, timestamp_pb2
                from google.protobuf.internal import builder

                pool = descriptor_pool.DescriptorPool()
                pool.AddSerializedFile(timestamp_pb2.DESCRIPTOR.serialized_pb)
                desc = pool.AddSerializedFile(_PROTO_DESCRIPTOR)
                ns: dict[str, Any] = {}
                builder.BuildMessageAndEnumDescriptors(desc, ns)
                builder.BuildTopDescriptorsAndMessages(desc, "zz_linq_message_pb2", ns)
                _TASK_CLS = ns["Task"]
    return _TASK_CLS


def serialize_task(tensors: dict[str, np.ndarray]) -> bytes:
    msg = _task_cls()()
    for name, arr in tensors.items():
        a = msg.dict_data[name]
        a.dtype = arr.dtype.name
        a.shape.extend(arr.shape)
        a.byte_data = np.ascontiguousarray(arr).tobytes()
    return msg.SerializeToString()


def deserialize_task(body: bytes) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    msg = _task_cls()()
    msg.ParseFromString(body)
    data = {
        name: np.frombuffer(a.byte_data, dtype=a.dtype).reshape(tuple(a.shape))
        for name, a in msg.dict_data.items()
    }
    worker = {
        "worker_pid": msg.worker_pid,
        "thread_name": msg.thread_name,
        "inference_time": float(msg.inference_time),
        "status_code": int(msg.status_code),
        "error_message": msg.error_message,
        "data_path": msg.data_path,
    }
    return data, worker


# ---------------------------------------------------------------- raw-кодек


_RAW_OUTPUT_DTYPES = {"int8": np.dtype(np.int8), "float16": np.dtype("<f2")}


class RawCodec:
    """Упаковка user-тензоров в формат TPU и обратно по metadata.json."""

    def __init__(self, inputs: dict[str, dict], outputs: dict[str, dict]) -> None:
        self.inputs, self.outputs = inputs, outputs

    @staticmethod
    def _scale(t: dict, ndim_order: list[str]) -> np.ndarray:
        s = np.asarray(t["scales"], np.float32)
        shp = t.get("scale_shape") or [1]
        if s.size == 1:
            return s.reshape(())
        return s.reshape([1] * (len(ndim_order) - len(shp)) + list(shp))

    def pack(self, name: str, x: np.ndarray) -> np.ndarray:
        t = self.inputs[name]
        if t.get("tpu_dtype") != "int8":
            raise NotImplementedError(f"raw: tpu_dtype {t.get('tpu_dtype')}")
        uo, to = list(t["user_order"]), list(t["tpu_order"])
        q = np.rint(np.asarray(x, np.float32) * self._scale(t, uo))
        q = np.clip(q, -128, 127).astype(np.int8)
        pads = [tuple(p) for p in t.get("padding") or [[0, 0]] * q.ndim]
        if any(a < 0 or b < 0 for a, b in pads):
            raise NotImplementedError("raw: отрицательный паддинг на входе")
        q = np.pad(q, pads)
        B = t["tpu_shape"][to.index("B")]
        ci = uo.index("C")
        shp = list(q.shape)
        q = q.reshape(shp[:ci] + [shp[ci] // B, B] + shp[ci + 1:])
        axes_named = uo[:ci] + ["C", "B"] + uo[ci + 1:]
        q = q.transpose([axes_named.index(a) for a in to])
        return np.ascontiguousarray(q).reshape(-1).view(np.uint8)

    def unpack(self, name: str, raw: np.ndarray) -> np.ndarray:
        t = self.outputs[name]
        uo, to = list(t["user_order"]), list(t["tpu_order"])
        # Тип элемента выхода — из metadata: у pose видимость точек (Sigmoid_1) приходит в float16.
        dtype = _RAW_OUTPUT_DTYPES.get(t.get("tpu_dtype", "int8"))
        if dtype is None:
            raise NotImplementedError(f"raw: tpu_dtype {t.get('tpu_dtype')}")
        count = int(np.prod(t["tpu_shape"]))
        data = np.ascontiguousarray(raw).reshape(-1).view(np.uint8)
        q = data[: count * dtype.itemsize].view(dtype).reshape(t["tpu_shape"])
        perm = [to.index(a) if a != "C" else to.index("C") for a in uo]
        ci = uo.index("C")
        perm = perm[:ci] + [to.index("C"), to.index("B")] + perm[ci + 1:]
        q = q.transpose(perm)
        shp = list(q.shape)
        q = q.reshape(shp[:ci] + [shp[ci] * shp[ci + 1]] + shp[ci + 2:])
        sl = []
        for ax, (a, b) in enumerate(t.get("padding") or [[0, 0]] * q.ndim):
            n = q.shape[ax]
            start = -a if a < 0 else 0
            stop = n + b if b < 0 else n
            sl.append(slice(start, stop))
        q = q[tuple(sl)]
        return q.astype(np.float32) * self._scale(t, uo)


# ---------------------------------------------------------------- транспорт


class BackendUnavailable(RuntimeError):
    """Бэкенд не смог выполнить инференс либо временно отключён."""


class TpuWorkerError(BackendUnavailable):
    """Воркер вернул ошибку (status_code != 0 или пустой ответ)."""


@dataclass
class Reply:
    data: dict[str, np.ndarray]
    worker: dict[str, Any]
    t_sent: float
    t_recv: float
    req_bytes: int
    resp_bytes: int


class RabbitClient:
    """Одно pika-соединение на поток; submit/collect поддерживают конвейер."""

    def __init__(self, host: str, port: int, user: str = "guest", password: str = "guest",
                 connect_timeout: float = 2.0, heartbeat: int = 60) -> None:
        self.host, self.port, self.user, self.password = host, port, user, password
        self.connect_timeout = max(0.01, min(2.0, float(connect_timeout)))
        self.heartbeat = heartbeat
        self.conn: Any = None
        self.ch: Any = None
        self.reply_queue = ""
        self._replies: dict[str, Reply] = {}
        self._pending: dict[str, tuple[float, int]] = {}
        self._stop = threading.Event()
        self.connect_ms = 0.0

    def connect(self) -> None:
        import pika

        if self._stop.is_set():
            raise BackendUnavailable("linq-tpu: соединение закрывается")
        self.close()
        t0 = time.perf_counter()
        params = pika.ConnectionParameters(
            self.host, self.port, credentials=pika.PlainCredentials(self.user, self.password),
            heartbeat=self.heartbeat,
            blocked_connection_timeout=self.connect_timeout,
            socket_timeout=self.connect_timeout / 2.0,
            stack_timeout=self.connect_timeout,
            connection_attempts=1,
            retry_delay=0,
        )
        try:
            self.conn = pika.BlockingConnection(params)
            if self._stop.is_set():
                raise BackendUnavailable("linq-tpu: соединение закрывается")
            self.ch = self.conn.channel()
            self.reply_queue = f"zz_guard_{uuid.uuid4().hex[:12]}"
            self.ch.queue_declare(queue=self.reply_queue, exclusive=True)
            self.ch.queue_bind(exchange=EXCHANGE, queue=self.reply_queue, routing_key=self.reply_queue)
            self.ch.basic_consume(self.reply_queue, self._on_reply, auto_ack=True)
        except Exception:
            self.close()
            raise
        self._replies.clear()
        self._pending.clear()
        self.connect_ms = (time.perf_counter() - t0) * 1000

    @property
    def is_open(self) -> bool:
        return bool(self.conn is not None and self.conn.is_open and self.ch is not None and self.ch.is_open)

    def request_close(self) -> None:
        """Сигнал владельцу соединения; pika не вызывается из чужого потока."""
        self._stop.set()

    def close(self) -> None:
        try:
            if self.conn is not None and self.conn.is_open:
                self.conn.close()
        except Exception:  # noqa: BLE001
            pass
        finally:
            self.conn = self.ch = None
            self._pending.clear()
            self._replies.clear()

    def _on_reply(self, ch: Any, method: Any, props: Any, body: bytes) -> None:
        t = time.monotonic()
        mid = (props.headers or {}).get(MESSAGE_ID_HEADER)
        if mid not in self._pending:
            log.warning("linq-tpu: ответ на чужую/просроченную задачу %s", mid)
            return
        t_sent, req_bytes = self._pending.pop(mid)
        data, worker = deserialize_task(body)
        self._replies[mid] = Reply(data, worker, t_sent, t, req_bytes, len(body))

    def submit(self, tensors: dict[str, np.ndarray], uri: str, body: bytes | None = None) -> str:
        import pika

        if self._stop.is_set():
            raise BackendUnavailable("linq-tpu: соединение закрывается")
        if not self.is_open:
            self.connect()
        mid = uuid.uuid4().hex
        body = body if body is not None else serialize_task(tensors)
        self.ch.basic_publish(
            exchange=EXCHANGE, routing_key=INPUT_ROUTING_KEY, body=body,
            properties=pika.BasicProperties(headers={
                URI_HEADER: uri,
                ROUTING_KEYS_HEADER: json.dumps([self.reply_queue]),
                MESSAGE_ID_HEADER: mid,
            }),
        )
        self._pending[mid] = (time.monotonic(), len(body))
        return mid

    def collect(self, ids: list[str], timeout: float) -> dict[str, Reply]:
        deadline = time.monotonic() + timeout
        while any(i not in self._replies for i in ids):
            if self._stop.is_set():
                raise BackendUnavailable("linq-tpu: соединение закрывается")
            left = deadline - time.monotonic()
            if left <= 0:
                for i in ids:
                    self._pending.pop(i, None)
                    self._replies.pop(i, None)
                raise TimeoutError(f"linq-tpu: нет ответа за {timeout:g} с")
            self.conn.process_data_events(time_limit=min(left, 0.1))
        return {i: self._replies.pop(i) for i in ids}


# ---------------------------------------------------------------- модель


@dataclass
class _TpuModel:
    spec: ModelSpec
    uri: str
    input_name: str
    program_batch: int
    tpu_outputs: list[str]
    out_names: list[str]
    tail: Any = None
    tail_inputs: list[str] = field(default_factory=list)
    manifest: dict[str, Any] = field(default_factory=dict)
    tail_batch1: bool = False
    codec: Any = None
    ddr_word_len: int = 32


def _find_manifest(spec: ModelSpec, models_dir: Path | None, local_program: Path | None) -> dict[str, Any]:
    cands: list[Path] = []
    if local_program is not None:
        cands.append(local_program.parent / "manifest.json")
    base = models_dir or Path(os.environ.get("GUARD_MODELS_DIR") or project_root() / "models").expanduser()
    cands.append(base / "tpu" / spec.model_id / "manifest.json")
    if spec.tpu_program:
        stem = Path(spec.tpu_program.split("?")[0]).name
        cands += sorted((base / "tpu").glob(f"*/{stem}"))
        cands = [c.parent / "manifest.json" if c.suffix == ".tpu" else c for c in cands]
    for c in cands:
        if c.is_file():
            try:
                m = json.loads(c.read_text())
                m["_path"] = str(c)
                return m
            except ValueError:
                continue
    return {}


class LinqTpuBackend(InferenceBackend):
    name = "linq-tpu"
    FAILURE_THRESHOLD = 3
    CIRCUIT_RESET_S = 10.0

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        user: str | None = None,
        password: str | None = None,
        program_dir: str | None = None,
        models_dir: str | Path | None = None,
        timeout_s: float | None = None,
        retries: int = 0,
        raw: bool | None = None,
        window: int | None = None,
        connect_timeout_s: float = 2.0,
    ) -> None:
        self.host = host or _cfg("LINQ_RABBIT_HOST", default=_default_host())
        self.port = int(port or _cfg("LINQ_RABBIT_PORT", default="5672"))
        self.user = user or _cfg("LINQ_RABBIT_USER", default="guest")
        self.password = password or _cfg("LINQ_RABBIT_PASSWORD", default="guest")
        login = _cfg("LINQ_LOGIN", "JUPYTER_LOGIN", default="hackathon10864")
        self.program_dir = (program_dir or _cfg("LINQ_PROGRAM_DIR", default=f"/auto/cache/{login}")).rstrip("/")
        self.models_dir = Path(models_dir).expanduser() if models_dir else None
        self.timeout_s = max(0.01, float(
            timeout_s if timeout_s is not None else _cfg("LINQ_TIMEOUT_S", default="2")
        ))
        self.retries = max(0, min(1, int(retries)))
        self.connect_timeout_s = max(0.01, min(2.0, float(connect_timeout_s)))
        self.raw = bool(int(_cfg("LINQ_RAW", default="0"))) if raw is None else bool(raw)
        self.window = max(1, int(window or _cfg("LINQ_WINDOW", default="4")))
        self._pool: Any = None
        self._models: dict[str, _TpuModel] = {}
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._close_lock = threading.Lock()
        self._local = threading.local()
        self._clients: list[RabbitClient] = []
        self._stats = BackendStats()
        self._bytes = {"sent": 0, "received": 0}
        self._closed = False
        self._inflight = 0
        self._consecutive_failures = 0
        self._open_until = 0.0
        self._half_open = False
        self._last_error: str | None = None
        self._availability: dict[str, bool] = {}

    # ---- соединения и circuit breaker ----

    def _client(self) -> RabbitClient:
        c = getattr(self._local, "client", None)
        if c is None:
            with self._lock:
                if self._closed:
                    raise BackendUnavailable("linq-tpu: бэкенд закрыт")
                c = RabbitClient(
                    self.host, self.port, self.user, self.password,
                    connect_timeout=self.connect_timeout_s,
                )
                self._clients.append(c)
                self._local.client = c
        if not c.is_open:
            c.connect()
        return c

    def _begin_call(self) -> None:
        with self._condition:
            if self._closed:
                raise BackendUnavailable("linq-tpu: бэкенд закрыт")
            now = time.monotonic()
            if now < self._open_until or self._half_open:
                raise BackendUnavailable(f"linq-tpu: circuit open; {self._last_error}")
            if self._open_until:
                self._half_open = True
            self._inflight += 1

    def _finish_call(self) -> None:
        with self._condition:
            self._inflight -= 1
            self._condition.notify_all()

    def _failed(self, model_id: str, exc: Exception) -> None:
        with self._lock:
            self._availability[model_id] = False
            self._consecutive_failures += 1
            self._last_error = f"{type(exc).__name__}: {exc}"[:300]
            if self._consecutive_failures >= self.FAILURE_THRESHOLD:
                self._open_until = time.monotonic() + self.CIRCUIT_RESET_S
            self._half_open = False

    def _succeeded(self, model_id: str) -> None:
        with self._lock:
            self._availability[model_id] = True
            self._consecutive_failures = 0
            self._open_until = 0.0
            self._half_open = False
            self._last_error = None

    # ---- загрузка ----

    def _resolve_uri(self, ref: str) -> tuple[str, Path | None]:
        if ref.startswith("file://"):
            uri = ref
            local = None
        else:
            p = resolve_model_path(ref, self.models_dir)
            local = p if p.exists() else None
            uri = f"file://{self.program_dir}/{p.name}"
        uri = uri.split("?")[0] + ("?raw=True" if self.raw else "?raw=False")
        return uri, local

    def load(self, spec: ModelSpec) -> None:
        with self._lock:
            if self._closed:
                raise BackendUnavailable("linq-tpu: бэкенд закрыт")
            if spec.model_id in self._models:
                return
        if not spec.tpu_program:
            raise ValueError(f"{spec.model_id}: tpu_program не задан (нужна программа TPU, tpu/compile.py)")
        uri, local = self._resolve_uri(spec.tpu_program)
        man = _find_manifest(spec, self.models_dir, local)
        input_name = next(iter(man.get("inputs", {})), None)
        batch = int(man.get("batch", 1))
        tpu_outputs = list(man.get("tpu_outputs") or [])
        tail = None
        tail_inputs: list[str] = []
        out_names = list(man.get("orig_outputs") or tpu_outputs)
        if man.get("tail_onnx"):
            import onnxruntime as ort

            tail_path = Path(man["tail_onnx"])
            if not tail_path.exists() and man.get("_path"):
                tail_path = Path(man["_path"]).parent / tail_path.name
            so = ort.SessionOptions()
            so.intra_op_num_threads = 2
            tail = ort.InferenceSession(str(tail_path), so, providers=["CPUExecutionProvider"])
            tail_inputs = [i.name for i in tail.get_inputs()]
            out_names = [o.name for o in tail.get_outputs()]
        meta: dict[str, Any] = {}
        prog = local or Path(uri[len("file://"):].split("?")[0])
        if prog.is_file():
            try:
                import zipfile

                with zipfile.ZipFile(prog) as z:
                    meta = json.loads(z.read("metadata.json"))
            except Exception:  # noqa: BLE001
                meta = {}
        if input_name is None and meta:
            try:
                ins = [n for d in meta.get("inputs", {}).values() for n in d]
                outs = [n for d in meta.get("outputs", {}).values() for n in d]
                input_name = ins[0] if ins else None
                out_names = out_names or outs
                tpu_outputs = tpu_outputs or outs
                batch = int(next(t for d in meta["inputs"].values() for t in d.values())["user_shape"][0])
            except Exception:  # noqa: BLE001
                pass
        if input_name is None:
            try:
                import onnx

                m = onnx.load(str(resolve_model_path(spec.onnx_path, self.models_dir)), load_external_data=False)
                inits = {i.name for i in m.graph.initializer}
                input_name = next(i.name for i in m.graph.input if i.name not in inits)
                out_names = out_names or [o.name for o in m.graph.output]
            except Exception as e:  # noqa: BLE001
                raise ValueError(f"{spec.model_id}: нет manifest.json и не удалось прочитать вход из ONNX: {e}") from e
        model = _TpuModel(spec, uri, input_name, batch, tpu_outputs, out_names, tail, tail_inputs, man)
        if self.raw:
            pm = man.get("program_meta") or {}
            ins = pm.get("inputs") or {n: t for d in meta.get("inputs", {}).values() for n, t in d.items()}
            outs = pm.get("outputs") or {n: t for d in meta.get("outputs", {}).values() for n, t in d.items()}
            if not ins or not outs:
                raise ValueError(f"{spec.model_id}: raw=True требует metadata программы (manifest.json или локальный .tpu)")
            model.codec = RawCodec(ins, outs)
            model.ddr_word_len = int((meta.get("hardware_parameters") or {}).get("ddr_word_len", 32))
            model.tpu_outputs = model.tpu_outputs or list(outs)
        if tail is not None:
            shp = tail.get_inputs()[0].shape
            model.tail_batch1 = bool(shp) and shp[0] == 1
        with self._lock:
            if self._closed:
                raise BackendUnavailable("linq-tpu: бэкенд закрыт")
            self._models[spec.model_id] = model
        log.info("linq-tpu: %s -> %s (input %s, tail=%s)", spec.model_id, uri, input_name, bool(tail))

    def loaded_models(self) -> list[str]:
        with self._lock:
            return list(self._models)

    def max_batch_size(self, model_id: str) -> int:
        return 8

    def class_names(self, model_id: str) -> list[str] | None:
        with self._lock:
            m = self._models.get(model_id)
            return list(m.spec.classes) if m and m.spec.classes else None

    # ---- инференс ----

    def _prepare(self, m: _TpuModel, batch: np.ndarray) -> np.ndarray:
        x = np.asarray(batch)
        if x.ndim == 3:
            x = x[None]
        if x.dtype == np.uint8:
            x = x.astype(np.float32) / 255.0
        elif x.dtype != np.float32:
            x = x.astype(np.float32)
        return np.ascontiguousarray(x)

    def infer_timed(self, model_id: str, batch: np.ndarray) -> tuple[list[np.ndarray], dict[str, float]]:
        self._begin_call()
        try:
            result = self._infer_timed(model_id, batch)
        except Exception as exc:
            self._failed(model_id, exc)
            if isinstance(exc, BackendUnavailable):
                raise
            raise BackendUnavailable(f"linq-tpu: {type(exc).__name__}: {exc}") from exc
        else:
            self._succeeded(model_id)
            return result
        finally:
            self._finish_call()

    def _infer_timed(self, model_id: str, batch: np.ndarray) -> tuple[list[np.ndarray], dict[str, float]]:
        m = self._models.get(model_id)
        if m is None:
            raise KeyError(f"модель {model_id} не загружена")
        x = self._prepare(m, batch)
        if x.shape[0] == 0:
            raise ValueError("linq-tpu: пустой батч")
        pb = max(1, m.program_batch)
        chunks = [x[i:i + pb] for i in range(0, x.shape[0], pb)]
        if chunks[-1].shape[0] != pb:
            pad = np.zeros((pb - chunks[-1].shape[0],) + x.shape[1:], np.float32)
            chunks[-1] = np.concatenate([chunks[-1], pad])
        last_exc: Exception | None = None
        for attempt in range(self.retries + 1):
            with self._lock:
                if self._closed:
                    raise BackendUnavailable("linq-tpu: бэкенд закрыт")
            try:
                c = self._client()
                t0 = time.perf_counter()
                if m.codec is not None:
                    bodies = [serialize_task({m.input_name: m.codec.pack(m.input_name, ch).reshape(-1, m.ddr_word_len)})
                              for ch in chunks]
                else:
                    bodies = [serialize_task({m.input_name: ch}) for ch in chunks]
                t_ser = time.perf_counter()
                ids = [c.submit({}, m.uri, body=b) for b in bodies]
                t_pub = time.perf_counter()
                replies = c.collect(ids, self.timeout_s)
                t_rcv = time.perf_counter()
                outs_per_chunk = []
                worker_ms = []
                for i in ids:
                    r = replies[i]
                    if r.worker["status_code"] != 0 or not r.data:
                        raise TpuWorkerError(f"воркер: status={r.worker['status_code']} {r.worker['error_message']}")
                    worker_ms.append(r.worker["inference_time"] * 1000.0)
                    if m.codec is not None:
                        outs_per_chunk.append({n: m.codec.unpack(n, a) for n, a in r.data.items()})
                    else:
                        outs_per_chunk.append(r.data)
                    with self._lock:
                        self._bytes["sent"] += r.req_bytes
                        self._bytes["received"] += r.resp_bytes
                names = m.tail_inputs or m.tpu_outputs or sorted(outs_per_chunk[0])
                merged = {n: np.concatenate([o[n] for o in outs_per_chunk])[: x.shape[0]] for n in names}
                t_tail0 = time.perf_counter()
                if m.tail is not None:
                    feeds = {n: merged[n].astype(np.float32) for n in m.tail_inputs}
                    if m.tail_batch1 and x.shape[0] > 1:
                        parts = [m.tail.run(m.out_names, {n: v[i:i + 1] for n, v in feeds.items()})
                                 for i in range(x.shape[0])]
                        outputs = [np.concatenate([p[k] for p in parts]) for k in range(len(m.out_names))]
                    else:
                        outputs = [np.asarray(r) for r in m.tail.run(m.out_names, feeds)]
                else:
                    outputs = [merged[n] for n in (m.out_names if set(m.out_names) <= set(merged) else names)]
                t_end = time.perf_counter()
                rtt_ms = (t_rcv - t_ser) * 1000.0
                infer_ms = float(np.max(worker_ms)) if worker_ms else 0.0
                timings = {
                    "serialize": (t_ser - t0) * 1000.0,
                    "publish": (t_pub - t_ser) * 1000.0,
                    "rtt": rtt_ms,
                    "transfer": max(0.0, rtt_ms - infer_ms),
                    "infer": infer_ms,
                    "tail": (t_end - t_tail0) * 1000.0,
                    "total": (t_end - t0) * 1000.0,
                }
                self._stats.record(model_id, x.shape[0], **timings)
                return outputs, timings
            except TpuWorkerError as e:
                last_exc = e
                self._stats.error(model_id, e)
                break
            except Exception as e:  # noqa: BLE001
                last_exc = e
                self._stats.error(model_id, e)
                log.warning("linq-tpu: %s попытка %d: %r", model_id, attempt + 1, e)
                c = getattr(self._local, "client", None)
                if c is not None:
                    c.close()
                # Повтор только явный; никаких вложенных повторов pika и пауз.
        assert last_exc is not None
        raise last_exc

    def infer(self, model_id: str, batch: np.ndarray) -> list[np.ndarray]:
        return self.infer_timed(model_id, batch)[0]

    def submit(self, model_id: str, batch: np.ndarray) -> Any:
        """Future[(outputs, timings)], до window параллельных задач."""
        import concurrent.futures as cf

        with self._lock:
            if self._closed:
                future = cf.Future()
                future.set_exception(BackendUnavailable("linq-tpu: бэкенд закрыт"))
                return future
            if self._pool is None:
                self._pool = cf.ThreadPoolExecutor(self.window, thread_name_prefix="linq-tpu")
            return self._pool.submit(self.infer_timed, model_id, batch)

    def stats(self) -> dict[str, Any]:
        with self._lock:
            state = {
                "backend": self.name,
                "active": self.name,
                "requested_backend": self.name,
                "active_backend": self.name,
                "fallback_enabled": False,
                "fallback": False,
                "cpu_tail_models": sorted(model_id for model_id, model in self._models.items() if model.tail is not None),
                "broker": f"{self.host}:{self.port}",
                "bytes": dict(self._bytes),
                "consecutive_failures": self._consecutive_failures,
                "circuit_open": self._half_open or time.monotonic() < self._open_until,
                "last_error": self._last_error,
                "closed": self._closed,
                "available": not self._closed and (
                    not self._availability or any(self._availability.values())
                ),
            }
        state["models"] = self._stats.snapshot()
        return state

    def close(self) -> None:
        """Отменить очередь, прервать collect, дождаться вызовов и закрыть соединения."""
        with self._close_lock:
            with self._condition:
                self._closed = True
                clients = list(self._clients)
                pool = self._pool
            for client in clients:
                request_close = getattr(client, "request_close", None)
                if callable(request_close):
                    request_close()
            if pool is not None:
                pool.shutdown(wait=True, cancel_futures=True)
            with self._condition:
                while self._inflight:
                    self._condition.wait()
                clients = list(self._clients)
                self._clients.clear()
                self._pool = None
                self._models.clear()
            # Теперь ни один поток не использует pika.
            for client in clients:
                client.close()


def create_backend(**kwargs: Any) -> LinqTpuBackend:
    return LinqTpuBackend(**kwargs)


__all__ = [
    "BackendUnavailable",
    "LinqTpuBackend",
    "RabbitClient",
    "TpuWorkerError",
    "create_backend",
    "deserialize_task",
    "serialize_task",
]
