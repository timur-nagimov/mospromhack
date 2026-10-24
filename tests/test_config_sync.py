"""Канал конфигурации камер gateway → edge (docs/EDGE.md §8) на тестовом транспорте.

Настоящие ConfigSyncEdgeEngine и ConfigSyncGatewayEngine, настоящие SQLite, оверлей, API и WS;
события и телеметрия — через FakeBroker, управляющий канал — настоящий RabbitControlLink поверх
FakeAmqp (direct exchange в памяти, durable-очереди, возврат неподтверждённых при обрыве).
Кадры подаются явно, без потоков камер и без сна.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict, deque
from types import SimpleNamespace
from uuid import uuid4

import anyio
import cv2
import numpy as np
import pytest
import yaml
from fastapi.testclient import TestClient

from guard.api.app import create_app
from guard.contracts import CameraConfig
from guard.edge.config_sync import ConfigSyncEdgeEngine
from guard.gateway.config_sync import ConfigSyncGatewayEngine
from guard.inference.base import InferenceBackend
from guard.pipeline.config_sync import GatewayConfigSync, overlay_path
from guard.pipeline.engine import PipelineEngine
from guard.pipeline.settings import Settings
from guard.sources.base import FrameSource
from guard.transport.fake import FakeBroker
from guard.transport.protocol import Envelope, decode, encode
from guard.transport.rabbit import RabbitConfig

EDGE_ID = "edge-1"
URL = "amqp://guest:hidden-password@localhost/"
FRAME = 128
LEFT = (10, 24, 50, 96)    # точка опоры x = 30 / 128 — левая половина кадра
RIGHT = (78, 24, 118, 96)  # x = 98 / 128 — правая половина
LEFT_HALF = [[0.0, 0.0], [0.5, 0.0], [0.5, 1.0], [0.0, 1.0]]
RIGHT_HALF = [[0.5, 0.0], [1.0, 0.0], [1.0, 1.0], [0.5, 1.0]]


# ---------------------------------------------------------------- AMQP в памяти


class FakeAmqp:
    """Достаточно BlockingConnection/BlockingChannel для RabbitControlLink: direct exchange,
    durable-очереди, basic_get/ack/nack/reject, обрыв соединений с возвратом неподтверждённого."""

    def __init__(self):
        self.online = True
        self.bindings: dict[tuple[str, str], set[str]] = defaultdict(set)
        self.queues: dict[str, deque[bytes]] = defaultdict(deque)
        self.unacked: dict[int, tuple[str, bytes]] = {}
        self.connections: list[FakeConnection] = []
        self.declared_exchanges: set[str] = set()
        self._tag = 0

    def connect(self, config):
        if not self.online:
            raise ConnectionError("broker offline")
        connection = FakeConnection(self)
        self.connections.append(connection)
        return connection

    def drop(self):
        """Обрыв: соединения закрыты, неподтверждённые сообщения возвращаются в голову очереди."""
        for connection in self.connections:
            connection.is_open = False
        for tag in sorted(self.unacked, reverse=True):
            queue, body = self.unacked.pop(tag)
            self.queues[queue].appendleft(body)

    def next_tag(self):
        self._tag += 1
        return self._tag


class FakeConnection:
    def __init__(self, broker):
        self.broker = broker
        self.is_open = True

    def channel(self):
        return FakeChannel(self)

    def close(self):
        self.is_open = False

    def process_data_events(self, time_limit=0):
        self._check()

    def _check(self):
        if not self.is_open or not self.broker.online:
            self.is_open = False
            raise ConnectionError("connection lost")


class FakeChannel:
    def __init__(self, connection):
        self.connection = connection
        self.broker = connection.broker

    def exchange_declare(self, *, exchange, exchange_type, durable):
        assert exchange_type == "direct" and durable is True
        self.broker.declared_exchanges.add(exchange)

    def queue_declare(self, *, queue, durable, **_):
        assert durable is True
        self.broker.queues.setdefault(queue, deque())

    def queue_bind(self, *, queue, exchange, routing_key):
        self.broker.bindings[(exchange, routing_key)].add(queue)

    def confirm_delivery(self):
        pass

    def basic_qos(self, *, prefetch_count):
        assert prefetch_count == 1

    def basic_publish(self, *, exchange, routing_key, body, properties, mandatory):
        self.connection._check()
        assert properties.delivery_mode == 2 and mandatory is True
        queues = self.broker.bindings.get((exchange, routing_key))
        if not queues:
            raise ConnectionError("unroutable")
        for queue in queues:
            self.broker.queues[queue].append(body)

    def basic_get(self, *, queue, auto_ack):
        self.connection._check()
        assert auto_ack is False
        if not self.broker.queues[queue]:
            return None, None, None
        body = self.broker.queues[queue].popleft()
        tag = self.broker.next_tag()
        self.broker.unacked[tag] = (queue, body)
        return SimpleNamespace(delivery_tag=tag), None, body

    def basic_ack(self, *, delivery_tag):
        self.connection._check()
        self.broker.unacked.pop(delivery_tag)

    def basic_nack(self, *, delivery_tag, requeue):
        queue, body = self.broker.unacked.pop(delivery_tag)
        if requeue:
            self.broker.queues[queue].appendleft(body)

    def basic_reject(self, *, delivery_tag, requeue):
        self.basic_nack(delivery_tag=delivery_tag, requeue=requeue)


# ---------------------------------------------------------------- конвейер без моделей и потоков


class SyntheticBackend(InferenceBackend):
    """Только инференс подменён: Detector выполняет настоящие letterbox и декод YOLO."""

    name = "synthetic-cpu"

    def __init__(self):
        self.boxes = []

    def load(self, spec):
        assert spec.model_id == "det"

    def infer(self, model_id, batch):
        raw = np.empty((batch.shape[0], 5, len(self.boxes)), dtype=np.float32)
        for index, (left, top, right, bottom) in enumerate(self.boxes):
            raw[:, :, index] = [(left + right) / 2, (top + bottom) / 2, right - left, bottom - top, 0.95]
        return [raw]

    def stats(self):
        return {"backend": self.name, "models": {}}


class ManualSource(FrameSource):
    """Кадры подаёт тест; потока источника нет."""

    kind = "synthetic"

    def start(self):
        pass

    def emit(self, frame, ts):
        self.connected = True
        self._publish(frame, decode_ms=0.25, ts=ts)
        return self.latest()


def camera(zone_id="danger", polygon=LEFT_HALF, *, camera_id="cam-1", uri="frames/cam-1", rules=None, **extra):
    return {
        "camera_id": camera_id, "name": "Камера 1", "site": "Цех 1",
        "source": {"type": "images", "uri": uri, "loop": True, "realtime": True},
        "sample_fps": 1.0, "enabled": True,
        "zones": [{"zone_id": zone_id, "name": "Опасная зона", "kind": "danger", "polygon": polygon, "active_hours": None}],
        "rules": rules if rules is not None else [{
            "rule_id": "intrusion", "type": "zone_intrusion", "enabled": True, "zones": [zone_id],
            "params": {}, "min_duration_s": 2.0, "min_ratio": 1.0, "close_after_s": 1.0, "cooldown_s": 0.0,
            "severity": "high",
        }],
        **extra,
    }


def settings(root, mode):
    return Settings(
        data_dir=root / "data", models_dir=root / "models", cameras_config=root / "cameras.yaml",
        models_config=root / "models.yaml", db_path=root / "data" / "guard.db", media_dir=root / "data" / "media",
        frontend_dist=root / "no-frontend", mode=mode, edge_id=EDGE_ID, rabbit_url=URL, autostart=False,
        clips_enabled=False, flywheel=False, batching=False, tracker_min_hits=2, tracker_buffer_s=1.0,
    )


class Stand:
    """edge и шлюз, связанные FakeBroker (события, телеметрия) и FakeAmqp (управляющий канал)."""

    def __init__(self, root, monkeypatch, base_cameras):
        self.root = root
        self.monkeypatch = monkeypatch
        self.amqp = FakeAmqp()
        self.broker = FakeBroker()
        self.base_ts = time.time() - 600
        monkeypatch.setattr("guard.pipeline.worker.make_source",
                            lambda config, uri, camera_id, *, on_frame, fps: ManualSource(config, uri, camera_id, on_frame=on_frame))
        monkeypatch.setattr("guard.pipeline.worker.CameraWorker.start", lambda worker: worker.source.start())
        edge_root = root / "edge"
        edge_root.mkdir(parents=True)
        (edge_root / "models.yaml").write_text(
            "models:\n  det:\n    onnx_path: unused\n    input_size: [128, 128]\n    classes: [person]\n", encoding="utf-8")
        self.base_text = yaml.safe_dump({"cameras": base_cameras}, allow_unicode=True, sort_keys=False)
        (edge_root / "cameras.yaml").write_text(self.base_text, encoding="utf-8")
        self.edge_settings = settings(edge_root, "edge")
        self.backend = SyntheticBackend()
        self.edge = self.start_edge()
        self.gateway = ConfigSyncGatewayEngine(settings(root / "gateway", "gateway"), consumer=self.broker,
                                               control_connection_factory=self.amqp.connect)

    def start_edge(self):
        # Новый процесс edge — новый run_id конвертов (в тестах процесс один, подменяем явно).
        self.monkeypatch.setattr("guard.transport.store.RUN_ID", str(uuid4()))
        edge = ConfigSyncEdgeEngine(self.edge_settings, backend=self.backend, publisher=self.broker,
                                    telemetry_publisher=self.broker, control_connection_factory=self.amqp.connect)
        PipelineEngine.start(edge)  # камеры и оверлей; потоки репликации и канала тест крутит сам
        return edge

    def restart_edge(self):
        self.edge.stop()
        self.edge = self.start_edge()
        return self.edge

    def pump(self, rounds=10):
        """Отправить команды шлюза, применить их на edge, вернуть подтверждения шлюзу."""
        for _ in range(rounds):
            moved = self.gateway.control.poll_once() + self.edge.control.poll_once()
            if not moved and not self.gateway.control.status()["outgoing"]:
                return
        raise AssertionError("control channel did not settle")

    def telemetry(self):
        self.edge.replication._next_metrics = float("-inf")
        self.edge.replication.publish_telemetry_once()
        while self.broker.telemetry:
            self.broker.deliver(self.gateway.receive, durable=False)

    def events(self):
        self.edge.replication.publish_pending_once()
        while self.broker.events:
            self.broker.deliver(self.gateway.receive)

    def frame(self, seconds, boxes, camera_id="cam-1"):
        self.backend.boxes = list(boxes)
        image = np.full((FRAME, FRAME, 3), (12, 34, 56), dtype=np.uint8)
        for left, top, right, bottom in boxes:
            cv2.rectangle(image, (left, top), (right, bottom), (90, 120, 150), -1)
        worker = self.edge.workers[camera_id]
        return worker.process(worker.source.emit(image, self.base_ts + seconds))

    def stand_for(self, start, seconds, box):
        for second in range(start, start + seconds):
            self.frame(second, [box])

    def close(self):
        for engine in (self.edge, self.gateway):
            try:
                engine.stop()
            except Exception:  # noqa: BLE001
                pass


@pytest.fixture
def stand(tmp_path, monkeypatch):
    system = Stand(tmp_path, monkeypatch, [camera()])
    yield system
    system.close()


def drain(engine, ws):
    marker = {"kind": "test_barrier", "id": str(uuid4())}
    engine.bus.publish(marker)
    result = []
    while True:
        async def receive():
            with anyio.fail_after(3):
                return await ws._send_rx.receive()
        item = json.loads(ws.portal.call(receive)["text"])
        if item == marker:
            return result
        if item["kind"] != "metrics":
            result.append(item)


# ---------------------------------------------------------------- требования брифа


def test_zone_change_in_gateway_ui_applies_on_edge_old_zone_silent_new_zone_alerts(stand):
    """Правка зоны в UI шлюза → edge: по старой зоне событий больше нет, по новой — есть."""
    stand.stand_for(0, 4, LEFT)
    stand.events()
    gateway_events = stand.gateway.journal.query(limit=10)[0]
    assert [event.zone_id for event in gateway_events] == ["danger"]  # исходная зона работает
    stand.telemetry()
    assert stand.edge.replication.telemetry_error is None  # каталог и метрики уходят и после кадров
    worker_before = stand.edge.workers["cam-1"]

    with TestClient(create_app(stand.gateway.settings, stand.gateway)) as client:
        with client.websocket_connect("/ws/events") as ws:
            listed = client.get("/api/cameras").json()
            assert listed[0]["source"]["uri"] == f"edge://{EDGE_ID}/cam-1"  # настоящий URI наружу не уходит
            assert listed[0]["status"]["config_sync"]["state"] == "synced"
            edited = {**listed[0], "zones": [{**listed[0]["zones"][0], "zone_id": "dock", "polygon": RIGHT_HALF}],
                      "rules": [{**listed[0]["rules"][0], "zones": ["dock"]}]}
            edited.pop("status")
            response = client.put("/api/cameras/cam-1", json=edited)
            assert response.status_code == 200, response.text
            assert response.json()["status"]["config_sync"]["state"] == "pending"
            pending = client.get("/api/cameras/cam-1").json()
            assert pending["zones"][0]["zone_id"] == "dock" and pending["status"]["config_sync"]["state"] == "pending"

            stand.pump()
            sync = client.get("/api/cameras/cam-1").json()["status"]["config_sync"]
            assert sync["state"] == "synced" and sync["error"] is None
            messages = [m for m in drain(stand.gateway, ws) if m["kind"] == "camera_sync"]
            assert [m["state"] for m in messages] == ["pending", "synced"]
            assert {m["camera_id"] for m in messages} == {"cam-1"}

        # edge перезапустил только эту камеру с новой зоной; базовый файл не тронут, правка — в оверлее
        worker = stand.edge.workers["cam-1"]
        assert worker is not worker_before and [z.zone_id for z in worker.cfg.zones] == ["dock"]
        assert worker.cfg.source.uri == "frames/cam-1"
        assert stand.edge_settings.cameras_config.read_text(encoding="utf-8") == stand.base_text
        overlay = yaml.safe_load(overlay_path(stand.edge_settings.cameras_config).read_text(encoding="utf-8"))
        assert overlay["cameras"]["cam-1"]["op"] == "upsert" and overlay["cameras"]["cam-1"]["ok"] is True

        created = stand.edge.events_created
        stand.stand_for(10, 5, LEFT)  # старая зона: тишина
        assert stand.edge.events_created == created
        stand.stand_for(20, 4, RIGHT)  # новая зона: тревога
        assert stand.edge.events_created == created + 1
        stand.events()
        zones = [item["zone_id"] for item in client.get("/api/events", params={"order": "asc"}).json()["items"]]
        assert zones == ["danger", "dock"]

        stand.telemetry()  # каталог edge догнал шлюз: та же конфигурация, версия совпадает
        shown = client.get("/api/cameras/cam-1").json()
        assert shown["zones"][0]["zone_id"] == "dock"
        assert shown["status"]["config_sync"]["edge_version"] == shown["status"]["config_sync"]["version"]


def test_older_version_is_ignored_and_duplicate_is_idempotent(stand):
    stand.telemetry()
    sync = stand.gateway.config_sync
    older = sync.submit("upsert", "cam-1", CameraConfig.model_validate(camera("old", RIGHT_HALF)))
    newer = sync.submit("upsert", "cam-1", CameraConfig.model_validate(camera("new", RIGHT_HALF)))
    assert newer["version"] > older["version"]
    stand.gateway.control.poll_once()  # обе команды ушли в очередь edge
    commands = stand.amqp.queues[RabbitConfig(URL, EDGE_ID).config_queue]
    assert len(commands) == 2
    commands.reverse()  # доставка не по порядку: сначала новая, потом старая
    stand.pump()
    worker = stand.edge.workers["cam-1"]
    assert [z.zone_id for z in worker.cfg.zones] == ["new"]
    assert stand.edge.config_applier.counters == {"applied": 1, "failed": 0, "duplicate": 0, "stale": 1}
    assert sync.status("cam-1")["state"] == "synced" and sync.status("cam-1")["version"] == newer["version"]

    # повтор доставки той же версии: прежний ответ, без перезапуска камеры
    wire = stand.gateway.config_sync.pending_wires()  # синхронизировано — повторять нечего
    assert wire == []
    row_wire = stand.gateway.config_sync._row("cam-1")["wire"]
    stand.amqp.queues[RabbitConfig(URL, EDGE_ID).config_queue].append(bytes(row_wire))
    stand.pump()
    assert stand.edge.workers["cam-1"] is worker
    assert stand.edge.config_applier.counters["duplicate"] == 1
    assert sync.status("cam-1")["state"] == "synced"


def test_apply_error_is_reported_in_status_and_running_config_is_kept(stand):
    stand.telemetry()
    worker = stand.edge.workers["cam-1"]
    rules = [{"rule_id": "ppe", "type": "ppe_missing", "enabled": True, "zones": None, "params": {},
              "min_duration_s": 2.0, "min_ratio": 0.7, "close_after_s": 3.0, "cooldown_s": 30.0, "severity": "high"}]
    with TestClient(create_app(stand.gateway.settings, stand.gateway)) as client:
        rejected = client.put("/api/cameras/cam-1", json=camera("danger", LEFT_HALF, uri=f"edge://{EDGE_ID}/cam-1",
                                                                 rules=[{**camera()["rules"][0], "zones": ["missing"]}]))
        assert rejected.status_code == 422  # ошибка, видная шлюзу, не уходит на edge
        assert stand.amqp.queues[RabbitConfig(URL, EDGE_ID).config_queue] == deque()
        response = client.put("/api/cameras/cam-1", json=camera("danger", LEFT_HALF, uri=f"edge://{EDGE_ID}/cam-1", rules=rules))
        assert response.status_code == 200  # модели ppe нет только в реестре edge — решает edge
        stand.pump()
        shown = client.get("/api/cameras/cam-1").json()
    sync = shown["status"]["config_sync"]
    assert sync["state"] == "error" and "ppe" in sync["error"] and "registry" in sync["error"]
    assert shown["status"]["errors"]["config_sync"] == sync["error"]
    assert [rule["type"] for rule in shown["rules"]] == ["zone_intrusion"]  # показана действующая конфигурация
    assert stand.edge.workers["cam-1"] is worker  # камера на edge не перезапускалась
    assert stand.edge.config_applier.counters["failed"] == 1
    record = stand.edge.config_overlay.record("cam-1")
    assert record["ok"] is False and record["applied_version"] == 0  # ошибка записана, правки нет
    assert stand.edge_settings.cameras_config.read_text(encoding="utf-8") == stand.base_text


def test_gateway_resends_unconfirmed_commands_after_reconnect(stand):
    stand.telemetry()
    sync = stand.gateway.config_sync
    stand.amqp.online = False  # брокер недоступен в момент правки
    state = sync.submit("upsert", "cam-1", CameraConfig.model_validate(camera("dock", RIGHT_HALF, uri=f"edge://{EDGE_ID}/cam-1")))
    with pytest.raises(ConnectionError):
        stand.gateway.control.poll_once()
    assert sync.status("cam-1")["state"] == "pending"
    stand.amqp.online = True
    stand.pump()
    assert sync.status("cam-1")["state"] == "synced"
    assert [z.zone_id for z in stand.edge.workers["cam-1"].cfg.zones] == ["dock"]

    # команда опубликована, но потеряна брокером (очередь очищена), соединение шлюза оборвалось
    second = sync.submit("upsert", "cam-1", CameraConfig.model_validate(camera("gate", LEFT_HALF, uri=f"edge://{EDGE_ID}/cam-1")))
    stand.gateway.control.poll_once()
    stand.amqp.queues[RabbitConfig(URL, EDGE_ID).config_queue].clear()
    stand.amqp.drop()
    connects = stand.gateway.control.connects
    stand.pump()  # переподключение шлюза → повторная отправка неподтверждённого
    assert stand.gateway.control.connects == connects + 1
    assert sync.status("cam-1") == {**sync.status("cam-1"), "state": "synced", "version": second["version"]}
    assert [z.zone_id for z in stand.edge.workers["cam-1"].cfg.zones] == ["gate"]
    assert state["version"] < second["version"]

    # обрыв после применения, но до ACK команды: повторная доставка не перезапускает камеру
    third = sync.submit("upsert", "cam-1", CameraConfig.model_validate(camera("yard", RIGHT_HALF, uri=f"edge://{EDGE_ID}/cam-1")))
    stand.gateway.control.poll_once()
    original = stand.edge.control._handler

    def apply_then_drop(envelope):
        replies = original(envelope)
        stand.amqp.drop()
        return replies

    stand.edge.control._handler = apply_then_drop
    with pytest.raises(ConnectionError):
        stand.edge.control.poll_once()
    stand.edge.control._handler = original
    worker = stand.edge.workers["cam-1"]
    stand.pump()
    assert stand.edge.workers["cam-1"] is worker
    # две повторных доставки: брокер вернул неподтверждённую команду, шлюз после обрыва переслал свою
    assert stand.edge.config_applier.counters["duplicate"] == 2
    assert stand.edge.config_applier.counters["applied"] == 3
    assert sync.status("cam-1")["state"] == "synced" and sync.status("cam-1")["version"] == third["version"]


def test_overlay_survives_edge_restart_and_gateway_repairs_lost_overlay(stand):
    stand.telemetry()
    sync = stand.gateway.config_sync
    sync.submit("upsert", "cam-1", CameraConfig.model_validate(camera("dock", RIGHT_HALF, uri=f"edge://{EDGE_ID}/cam-1")))
    sync.submit("upsert", "cam-2", CameraConfig.model_validate(camera("bay", LEFT_HALF, camera_id="cam-2", uri="frames/cam-2")))
    stand.pump()
    assert sync.summary() == {"pending": 0, "synced": 2, "error": 0}
    version = sync.status("cam-1")["version"]

    edge = stand.restart_edge()
    assert stand.edge_settings.cameras_config.read_text(encoding="utf-8") == stand.base_text
    assert [z.zone_id for z in edge.cameras["cam-1"].zones] == ["dock"]
    assert edge.cameras["cam-2"].source.uri == "frames/cam-2"
    assert set(edge.workers) == {"cam-1", "cam-2"}
    assert edge.camera_status("cam-1")["config_version"] == version

    sync.submit("delete", "cam-2")
    stand.pump()
    edge = stand.restart_edge()
    assert set(edge.cameras) == {"cam-1"}

    # повторная доставка старой версии после рестарта не применяется
    payload = {"camera_id": "cam-1", "version": version, "op": "upsert", "config": camera("other", LEFT_HALF)}
    replay = Envelope(edge_id=EDGE_ID, producer_id=sync.producer_id, run_id=str(uuid4()), message_id=f"{sync.producer_id}:999",
                      kind="camera_config", sequence=999, sent_at=time.time(), payload=payload)
    ack = decode(edge.handle_control(replay)[0][1])
    assert ack.payload["ok"] is True and [z.zone_id for z in edge.cameras["cam-1"].zones] == ["dock"]

    # оверлей потерян (удалён руками): новый процесс edge сообщает версию 0, шлюз повторяет свои правки
    edge.stop()
    overlay_path(stand.edge_settings.cameras_config).unlink()
    stand.edge = edge = stand.start_edge()
    assert [z.zone_id for z in edge.cameras["cam-1"].zones] == ["danger"]
    stand.telemetry()
    assert sync.status("cam-1")["state"] == "pending"
    stand.pump()
    assert [z.zone_id for z in edge.cameras["cam-1"].zones] == ["dock"]
    assert sync.status("cam-1")["state"] == "synced"


# ---------------------------------------------------------------- граничные случаи


def test_gateway_api_validation_conflicts_and_delete(stand):
    stand.telemetry()
    with TestClient(create_app(stand.gateway.settings, stand.gateway)) as client:
        assert client.post("/api/cameras", json=camera()).status_code == 409
        assert client.put("/api/cameras/nope", json=camera(camera_id="nope")).status_code == 404
        assert client.put("/api/cameras/cam-1", json=camera(camera_id="other")).status_code == 422
        new = camera(camera_id="cam-9", uri=f"edge://{EDGE_ID}/cam-9")
        assert client.post("/api/cameras", json=new).status_code == 422  # у новой камеры нужен настоящий источник
        assert client.put("/api/cameras/cam-1", json=camera(uri=f"edge://{EDGE_ID}/cam-2")).status_code == 422
        assert client.post("/api/cameras", json=camera(camera_id="cam-9", uri="frames/cam-9")).status_code == 201
        assert {c["camera_id"] for c in client.get("/api/cameras").json()} == {"cam-1", "cam-9"}
        assert client.get("/api/cameras/cam-9").json()["source"]["uri"] == f"edge://{EDGE_ID}/cam-9"
        stand.pump()
        assert stand.edge.cameras["cam-9"].source.uri == "frames/cam-9"
        assert client.delete("/api/cameras/cam-1").status_code == 204
        assert [c["camera_id"] for c in client.get("/api/cameras").json()] == ["cam-9"]  # удаление ждёт edge
        stand.pump()
        assert "cam-1" not in stand.edge.cameras and "cam-1" not in stand.edge.workers
        health = client.get("/health").json()
    assert health["config_sync"]["pending"] == 0 and health["config_sync"]["synced"] == 2


def test_pending_source_change_is_kept_when_next_edit_uses_edge_reference(tmp_path):
    sync = GatewayConfigSync(tmp_path / "sync.db", EDGE_ID)
    sync.submit("upsert", "cam-1", CameraConfig.model_validate(camera(uri="rtsp://user:secret@cam/1")))
    sync.submit("upsert", "cam-1", CameraConfig.model_validate(camera("dock", uri=f"edge://{EDGE_ID}/cam-1")))
    wire = sync.pending_wires()
    assert len(wire) == 1
    payload = decode(wire[0][1]).payload
    assert payload["config"]["source"]["uri"] == "rtsp://user:secret@cam/1"
    shown = sync.view({})["cam-1"]
    assert shown.source.uri == f"edge://{EDGE_ID}/cam-1" and shown.zones[0].zone_id == "dock"
    sync.close()


def test_versions_are_monotonic_across_gateway_store_loss(tmp_path):
    clock = [1000.0]
    first = GatewayConfigSync(tmp_path / "a.db", EDGE_ID, clock=lambda: clock[0])
    versions = [first.submit("upsert", "cam-1", CameraConfig.model_validate(camera()))["version"] for _ in range(3)]
    assert versions == sorted(set(versions)) and versions[0] >= 1_000_000
    first.close()
    clock[0] += 1
    second = GatewayConfigSync(tmp_path / "b.db", EDGE_ID, clock=lambda: clock[0])  # БД шлюза потеряна
    assert second.submit("upsert", "cam-1", CameraConfig.model_validate(camera()))["version"] > versions[-1]
    second.close()
    with pytest.raises(ValueError):
        GatewayConfigSync(tmp_path / "b.db", "edge-2")


def test_overlay_file_is_hidden_from_stand_sync_and_corrupt_overlay_is_set_aside(tmp_path):
    from guard.pipeline.config_sync import CameraOverlay

    base = tmp_path / "configs" / "cameras.stand.yaml"
    path = overlay_path(base)
    assert path == tmp_path / "configs" / ".cameras.stand.overlay.yaml"
    assert overlay_path(base, tmp_path / "custom.yaml") == tmp_path / "custom.yaml"
    path.parent.mkdir()
    path.write_text("cameras: [unclosed", encoding="utf-8")
    overlay = CameraOverlay(path)
    assert overlay.entries == {} and not path.exists()
    assert any(item.name.startswith(".cameras.stand.overlay.yaml.corrupt-") for item in path.parent.iterdir())


def test_control_messages_are_validated_like_other_envelopes():
    producer = str(uuid4())

    def message(kind, payload, sequence=1):
        return Envelope(edge_id=EDGE_ID, producer_id=producer, run_id=str(uuid4()), message_id=f"{producer}:{sequence}",
                        kind=kind, sequence=sequence, sent_at=1.0, payload=payload)

    good = message("camera_config", {"camera_id": "cam-1", "version": 5, "op": "upsert", "config": camera()})
    assert decode(encode(good)).payload["version"] == 5
    message("camera_config", {"camera_id": "cam-1", "version": 6, "op": "delete", "config": None})
    message("camera_config_ack", {"camera_id": "cam-1", "version": 6, "ok": True, "error": None, "current_version": 6})
    bad = [
        ("camera_config", {"camera_id": "cam-1", "version": 0, "op": "upsert", "config": camera()}),
        ("camera_config", {"camera_id": "cam-1", "version": 1, "op": "upsert", "config": camera(camera_id="cam-2")}),
        ("camera_config", {"camera_id": "cam-1", "version": 1, "op": "delete", "config": camera()}),
        ("camera_config", {"camera_id": "cam-1", "version": 1, "op": "move", "config": None}),
        ("camera_config", {"camera_id": "cam-1", "version": 1, "op": "upsert", "config": camera("зона")}),
        ("camera_config", {"camera_id": "cam-1", "version": 1, "op": "upsert", "config": camera(), "extra": 1}),
        ("camera_config_ack", {"camera_id": "cam-1", "version": 1, "ok": False, "error": None, "current_version": 0}),
        ("camera_config_ack", {"camera_id": "cam-1", "version": 1, "ok": True, "error": "x", "current_version": 0}),
        ("camera_config_ack", {"camera_id": "cam-1", "version": 1, "ok": 1, "error": None, "current_version": 0}),
    ]
    for kind, payload in bad:
        with pytest.raises(ValueError):
            message(kind, payload)


def test_topology_suffix_isolates_every_exchange_and_queue():
    config = RabbitConfig(URL, EDGE_ID, suffix="-c6test")
    names = [config.exchange_name, config.event_queue, config.telemetry_queue, config.config_queue, config.config_ack_queue]
    assert all(name.endswith("-c6test") for name in names)
    assert config.config_queue == "guard.edge.edge-1.config-c6test"
    plain = RabbitConfig(URL, EDGE_ID)
    assert (plain.exchange_name, plain.event_queue) == ("guard.edge.v1", "guard.gateway.edge-1.events")
    with pytest.raises(ValueError):
        RabbitConfig(URL, EDGE_ID, suffix="bad suffix")


def test_settings_read_config_sync_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("GUARD_RABBIT_SUFFIX", "-c6test")
    monkeypatch.setenv("GUARD_CONFIG_SYNC", "0")
    monkeypatch.setenv("GUARD_CAMERAS_OVERLAY", str(tmp_path / "overlay.yaml"))
    monkeypatch.setenv("GUARD_DATA_DIR", str(tmp_path / "data"))
    loaded = Settings.from_env()
    assert loaded.rabbit_suffix == "-c6test" and loaded.config_sync is False
    assert loaded.cameras_overlay == tmp_path / "overlay.yaml"


def test_gateway_without_config_sync_keeps_read_only_cameras(tmp_path):
    from guard.api.app import _create_engine
    from guard.gateway.engine import GatewayEngine

    cfg = settings(tmp_path, "gateway")
    cfg.config_sync = False
    engine = _create_engine(cfg)
    try:
        assert type(engine) is GatewayEngine
        with pytest.raises(Exception, match="Configure cameras on the edge"):
            engine.update_camera("cam-1", CameraConfig.model_validate(camera()))
    finally:
        engine.stop()


def test_unchanged_put_is_confirmed_without_restarting_the_edge_camera(stand):
    """Повторный PUT той же конфигурации (проверка канала после выкатки) не трогает камеру на edge."""
    stand.telemetry()
    worker = stand.edge.workers["cam-1"]
    with TestClient(create_app(stand.gateway.settings, stand.gateway)) as client:
        current = client.get("/api/cameras/cam-1").json()
        current.pop("status")
        assert client.put("/api/cameras/cam-1", json=current).status_code == 200
        stand.pump()
        sync = client.get("/api/cameras/cam-1").json()["status"]["config_sync"]
    assert sync["state"] == "synced"
    assert stand.edge.workers["cam-1"] is worker
    assert stand.edge.cameras["cam-1"].source.uri == "frames/cam-1"
    assert stand.edge.config_overlay.applied_version("cam-1") == sync["version"]
    assert stand.edge.config_applier.counters["applied"] == 1
