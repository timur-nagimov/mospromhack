"""Бэкенд LinQ TPU: инференс через облачный воркер по RabbitMQ (формат tpu_pycloud_format).

Протокол (vendor/linq_examples/example.ipynb + tpu_pycloud_format 0.0.1, проверено на стенде 24.10):
  * брокер RABBIT_HOST:5672, guest/guest; exchange ``tpu.direct``; задачи — routing key ``input``;
  * тело — protobuf ``Task``: dict_data {имя входа: Array(dtype, shape, byte_data)};
  * AMQP-заголовки: ``uri`` = ``file:///auto/cache/<login>/<prog>.tpu?raw=False`` (программа лежит на
    общей ФС стенда), ``response_routing_keys`` = JSON-список ключей ответа (последний — первый хоп),
    ``tpu_pycloud_message_id`` — наш id, воркер возвращает его в заголовках ответа;
  * ответ — тот же ``Task``: dict_data {имя выхода: float32}, worker_pid, inference_time (с),
    status_code (0 = ок), error_message.

Модель на TPU может быть «головой» сети: tpu/compile.py режет граф (--cut-preset yolov8), а хвост
(decode DFL + concat -> output0) лежит в *_tail.onnx и выполняется здесь на CPU (onnxruntime), поэтому
infer() возвращает выходы в том же формате, что исходная ONNX (контракт CONTRACTS.md).

Привязка программы (ModelSpec.tpu_program):
  * ``file:///auto/cache/...tpu[?raw=False]`` — URI как есть; manifest ищется в models/tpu/<model_id>/;
  * локальный путь к .tpu (абсолютный или от GUARD_MODELS_DIR) — URI = file://<LINQ_PROGRAM_DIR>/<имя>,
    LINQ_PROGRAM_DIR по умолчанию /auto/cache/<JUPYTER_LOGIN>; manifest.json — рядом с программой.

Окружение: LINQ_RABBIT_HOST/PORT (иначе RABBIT_HOST/RABBIT_PORT из .env; с ноутбука через туннель —
127.0.0.1:5672), LINQ_RABBIT_USER/PASSWORD (guest/guest), LINQ_PROGRAM_DIR, LINQ_TIMEOUT_S.
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

log = logging.getLogger(__name__)

EXCHANGE = "tpu.direct"
INPUT_ROUTING_KEY = "input"
URI_HEADER = "uri"
ROUTING_KEYS_HEADER = "response_routing_keys"
MESSAGE_ID_HEADER = "tpu_pycloud_message_id"

ROOT = Path(os.environ.get("GUARD_ROOT", "/Users/nagimov-timur/Desktop/mosprom"))


def _dotenv() -> dict[str, str]:
    env: dict[str, str] = {}
    p = ROOT / ".env"
    try:
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip('"').strip("'")
    except OSError:
        pass
    return env


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


# ---------------------------------------------------------------- wire format (tpu_pycloud_format 0.0.1)

# Сериализованный дескриптор message.proto из tpu_pycloud_format-0.0.1 (packages/ на стенде): Array, MetadataDict, Task.
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
    """Класс protobuf Task: из установленного tpu_pycloud_format или из встроенного дескриптора."""
    global _TASK_CLS
    if _TASK_CLS is not None:
        return _TASK_CLS
    with _TASK_LOCK:
        if _TASK_CLS is None:
            try:
                from tpu_pycloud_format import message_pb2  # type: ignore

                _TASK_CLS = message_pb2.Task
            except ImportError:
                from google.protobuf import descriptor_pool, timestamp_pb2  # noqa: F401
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
        "inference_time": float(msg.inference_time),  # секунды
        "status_code": int(msg.status_code),
        "error_message": msg.error_message,
        "data_path": msg.data_path,
    }
    return data, worker


# ---------------------------------------------------------------- транспорт


class TpuWorkerError(RuntimeError):
    """Воркер вернул ошибку (status_code != 0 или пустой ответ)."""


@dataclass
class Reply:
    data: dict[str, np.ndarray]
    worker: dict[str, Any]
    t_sent: float  # monotonic, после publish
    t_recv: float  # monotonic, ответ получен
    req_bytes: int
    resp_bytes: int


class RabbitClient:
    """Одно соединение pika + своя exclusive-очередь ответов. НЕ потокобезопасен — по экземпляру на поток.

    submit() публикует задачу и сразу возвращает id; collect() ждёт ответы — так кадры батча
    идут конвейером (на стенде воркер обслуживает несколько задач параллельно: Worker Thread #0/#1)."""

    def __init__(self, host: str, port: int, user: str = "guest", password: str = "guest",
                 connect_timeout: float = 15.0, heartbeat: int = 60) -> None:
        self.host, self.port, self.user, self.password = host, port, user, password
        self.connect_timeout, self.heartbeat = connect_timeout, heartbeat
        self.conn: Any = None
        self.ch: Any = None
        self.reply_queue = ""
        self._replies: dict[str, Reply] = {}
        self._pending: dict[str, tuple[float, int]] = {}
        self.connect_ms = 0.0

    def connect(self) -> None:
        import pika

        self.close()
        t0 = time.perf_counter()
        params = pika.ConnectionParameters(
            self.host, self.port, credentials=pika.PlainCredentials(self.user, self.password),
            heartbeat=self.heartbeat, blocked_connection_timeout=60, socket_timeout=self.connect_timeout,
            connection_attempts=2, retry_delay=1,
        )
        self.conn = pika.BlockingConnection(params)
        self.ch = self.conn.channel()
        self.reply_queue = f"zz_guard_{uuid.uuid4().hex[:12]}"
        self.ch.queue_declare(queue=self.reply_queue, exclusive=True)
        self.ch.queue_bind(exchange=EXCHANGE, queue=self.reply_queue, routing_key=self.reply_queue)
        self.ch.basic_consume(self.reply_queue, self._on_reply, auto_ack=True)
        self._replies.clear()
        self._pending.clear()
        self.connect_ms = (time.perf_counter() - t0) * 1000

    @property
    def is_open(self) -> bool:
        return bool(self.conn is not None and self.conn.is_open and self.ch is not None and self.ch.is_open)

    def close(self) -> None:
        try:
            if self.conn is not None and self.conn.is_open:
                self.conn.close()
        except Exception:  # noqa: BLE001
            pass
        self.conn = self.ch = None

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
            left = deadline - time.monotonic()
            if left <= 0:
                for i in ids:
                    self._pending.pop(i, None)
                raise TimeoutError(f"linq-tpu: нет ответа за {timeout:.0f} с ({sum(i not in self._replies for i in ids)} шт.)")
            self.conn.process_data_events(time_limit=min(left, 0.5))
        return {i: self._replies.pop(i) for i in ids}


# ---------------------------------------------------------------- модель


@dataclass
class _TpuModel:
    spec: ModelSpec
    uri: str
    input_name: str
    program_batch: int
    tpu_outputs: list[str]
    out_names: list[str]  # порядок итоговых выходов (как у исходной ONNX)
    tail: Any = None  # onnxruntime.InferenceSession хвоста или None
    tail_inputs: list[str] = field(default_factory=list)
    manifest: dict[str, Any] = field(default_factory=dict)


def _find_manifest(spec: ModelSpec, models_dir: Path | None, local_program: Path | None) -> dict[str, Any]:
    cands: list[Path] = []
    if local_program is not None:
        cands.append(local_program.parent / "manifest.json")
    base = models_dir or Path(os.environ.get("GUARD_MODELS_DIR", ROOT / "models"))
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

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        user: str | None = None,
        password: str | None = None,
        program_dir: str | None = None,
        models_dir: str | Path | None = None,
        timeout_s: float | None = None,
        retries: int = 2,
        raw: bool = False,
    ) -> None:
        self.host = host or _cfg("LINQ_RABBIT_HOST", "RABBIT_HOST", default="127.0.0.1")
        self.port = int(port or _cfg("LINQ_RABBIT_PORT", "RABBIT_PORT", default="5672"))
        self.user = user or _cfg("LINQ_RABBIT_USER", default="guest")
        self.password = password or _cfg("LINQ_RABBIT_PASSWORD", default="guest")
        login = _cfg("LINQ_LOGIN", "JUPYTER_LOGIN", default="hackathon10864")
        self.program_dir = (program_dir or _cfg("LINQ_PROGRAM_DIR", default=f"/auto/cache/{login}")).rstrip("/")
        self.models_dir = Path(models_dir).expanduser() if models_dir else None
        self.timeout_s = float(timeout_s or _cfg("LINQ_TIMEOUT_S", default="30"))
        self.retries = int(retries)
        self.raw = raw
        if raw:
            raise NotImplementedError("raw=True: упаковка tpu_io на клиенте не реализована, используйте raw=False")
        self._models: dict[str, _TpuModel] = {}
        self._lock = threading.Lock()
        self._local = threading.local()
        self._clients: list[RabbitClient] = []
        self._stats = BackendStats()
        self._bytes = {"sent": 0, "received": 0}

    # ---- соединения ----

    def _client(self) -> RabbitClient:
        c = getattr(self._local, "client", None)
        if c is None:
            c = RabbitClient(self.host, self.port, self.user, self.password)
            self._local.client = c
            with self._lock:
                self._clients.append(c)
        if not c.is_open:
            c.connect()
        return c

    # ---- загрузка ----

    def _resolve_uri(self, ref: str) -> tuple[str, Path | None]:
        if ref.startswith("file://"):
            uri = ref
            local = None
        else:
            p = resolve_model_path(ref, self.models_dir)
            local = p if p.exists() else None
            uri = f"file://{self.program_dir}/{p.name}"
        if "?" not in uri:
            uri += "?raw=True" if self.raw else "?raw=False"
        return uri, local

    def load(self, spec: ModelSpec) -> None:
        with self._lock:
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
        if input_name is None:  # без manifest: имена из metadata.json программы, если файл доступен
            prog = local or Path(uri[len("file://"):].split("?")[0])
            try:
                import zipfile

                with zipfile.ZipFile(prog) as z:
                    meta = json.loads(z.read("metadata.json"))
                ins = [n for d in meta.get("inputs", {}).values() for n in d]
                outs = [n for d in meta.get("outputs", {}).values() for n in d]
                input_name = ins[0] if ins else None
                out_names = out_names or outs
                tpu_outputs = tpu_outputs or outs
                batch = int(next(t for d in meta["inputs"].values() for t in d.values())["user_shape"][0])
            except Exception:  # noqa: BLE001
                pass
        if input_name is None:  # иначе — имя входа из исходной ONNX
            try:
                import onnx

                m = onnx.load(str(resolve_model_path(spec.onnx_path, self.models_dir)), load_external_data=False)
                inits = {i.name for i in m.graph.initializer}
                input_name = next(i.name for i in m.graph.input if i.name not in inits)
                out_names = out_names or [o.name for o in m.graph.output]
            except Exception as e:  # noqa: BLE001
                raise ValueError(f"{spec.model_id}: нет manifest.json и не удалось прочитать вход из ONNX: {e}") from e
        model = _TpuModel(spec, uri, input_name, batch, tpu_outputs, out_names, tail, tail_inputs, man)
        with self._lock:
            self._models[spec.model_id] = model
        log.info("linq-tpu: %s -> %s (input %s, tail=%s)", spec.model_id, uri, input_name, bool(tail))

    def loaded_models(self) -> list[str]:
        return list(self._models)

    def max_batch_size(self, model_id: str) -> int:
        return 8  # кадры батча уходят конвейером отдельными задачами

    def class_names(self, model_id: str) -> list[str] | None:
        m = self._models.get(model_id)
        return list(m.spec.classes) if m and m.spec.classes else None

    # ---- инференс ----

    def _prepare(self, m: _TpuModel, batch: np.ndarray) -> np.ndarray:
        x = np.asarray(batch)
        if x.ndim == 3:
            x = x[None]
        if x.dtype == np.uint8:  # программа ждёт float32 0..1 (letterbox-препроцессинг YOLO)
            x = x.astype(np.float32) / 255.0
        elif x.dtype != np.float32:
            x = x.astype(np.float32)
        return np.ascontiguousarray(x)

    def infer_timed(self, model_id: str, batch: np.ndarray) -> tuple[list[np.ndarray], dict[str, float]]:
        m = self._models.get(model_id)
        if m is None:
            raise KeyError(f"модель {model_id} не загружена")
        x = self._prepare(m, batch)
        pb = max(1, m.program_batch)
        chunks = [x[i:i + pb] for i in range(0, x.shape[0], pb)]
        if chunks[-1].shape[0] != pb:  # программа со статическим batch — добиваем нулями
            pad = np.zeros((pb - chunks[-1].shape[0],) + x.shape[1:], np.float32)
            chunks[-1] = np.concatenate([chunks[-1], pad])
        last_exc: BaseException | None = None
        for attempt in range(self.retries + 1):
            try:
                c = self._client()  # подключение (если нужно) не входит в замеры этапов
                t0 = time.perf_counter()
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
                    outs_per_chunk.append(r.data)
                    self._bytes["sent"] += r.req_bytes
                    self._bytes["received"] += r.resp_bytes
                names = m.tail_inputs or m.tpu_outputs or sorted(outs_per_chunk[0])
                merged = {n: np.concatenate([o[n] for o in outs_per_chunk])[: x.shape[0]] for n in names}
                t_tail0 = time.perf_counter()
                if m.tail is not None:
                    res = m.tail.run(m.out_names, {n: merged[n].astype(np.float32) for n in m.tail_inputs})
                    outputs = [np.asarray(r) for r in res]
                else:
                    outputs = [merged[n] for n in (m.out_names if set(m.out_names) <= set(merged) else names)]
                t_end = time.perf_counter()
                rtt_ms = (t_rcv - t_ser) * 1000.0
                infer_ms = float(np.max(worker_ms)) if worker_ms else 0.0
                timings = {
                    "serialize": (t_ser - t0) * 1000.0,
                    "publish": (t_pub - t_ser) * 1000.0,
                    "rtt": rtt_ms,
                    "transfer": max(0.0, rtt_ms - infer_ms),  # сеть туда-обратно + очередь воркера
                    "infer": infer_ms,  # время инференса по данным воркера
                    "tail": (t_end - t_tail0) * 1000.0,
                    "total": (t_end - t0) * 1000.0,
                }
                self._stats.record(model_id, x.shape[0], **timings)
                return outputs, timings
            except TpuWorkerError as e:
                last_exc = e
                self._stats.error(model_id, e)
                break  # ошибка программы/воркера — повтор не поможет
            except Exception as e:  # noqa: BLE001 — сеть/таймаут: переподключаемся и повторяем
                last_exc = e
                self._stats.error(model_id, e)
                log.warning("linq-tpu: %s попытка %d: %r", model_id, attempt + 1, e)
                c = getattr(self._local, "client", None)
                if c is not None:
                    c.close()
                time.sleep(min(2.0, 0.3 * (attempt + 1)))
        assert last_exc is not None
        raise last_exc

    def infer(self, model_id: str, batch: np.ndarray) -> list[np.ndarray]:
        return self.infer_timed(model_id, batch)[0]

    def stats(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "broker": f"{self.host}:{self.port}",
            "bytes": dict(self._bytes),
            "models": self._stats.snapshot(),
        }

    def close(self) -> None:
        with self._lock:
            for c in self._clients:
                c.close()
            self._clients.clear()


def create_backend(**kwargs: Any) -> LinqTpuBackend:
    return LinqTpuBackend(**kwargs)


__all__ = ["LinqTpuBackend", "RabbitClient", "TpuWorkerError", "create_backend", "deserialize_task", "serialize_task"]
