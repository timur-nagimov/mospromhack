"""Шлюз, на котором камеры можно менять: правка уходит на edge по каналу camera_config (docs/EDGE.md §8).

GatewayEngine без канала отвечает 409 на POST/PUT/DELETE камер. Здесь та же операция проверяется,
сохраняется как желаемая конфигурация с версией и отправляется на edge; состояние синхронизации
(pending, synced, error) видно в status.config_sync камеры в /api/cameras и в WS-сообщении camera_sync.
Модуль лёгкий: без inference, OpenCV и источников.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from guard.contracts import CameraConfig
from guard.gateway.engine import GatewayEngine
from guard.pipeline.config_sync import GatewayConfigSync, edge_source_ref, validate_camera_light
from guard.pipeline.errors import CameraError, CameraExists
from guard.rules.base import with_default_severity
from guard.transport.control import RabbitControlLink, Wire
from guard.transport.protocol import Envelope
from guard.transport.rabbit import RabbitConfig, _connection

log = logging.getLogger(__name__)


class ConfigSyncGatewayEngine(GatewayEngine):
    """GatewayEngine с управляющим каналом. control_connection_factory — фабрика AMQP (тесты)."""

    def __init__(self, settings, *, control_connection_factory: Callable | None = None, **kwargs: Any) -> None:
        super().__init__(settings, **kwargs)
        self.control: RabbitControlLink | None = None
        try:
            self.config_sync = GatewayConfigSync(
                settings.db_path.with_name("config_sync.db"), settings.edge_id, notify=self.bus.publish,
            )
            if settings.rabbit_url:
                self.control = RabbitControlLink(
                    RabbitConfig(settings.rabbit_url, settings.edge_id, suffix=settings.rabbit_suffix),
                    role="gateway", handler=self.handle_control, on_connect=self.config_sync.pending_wires,
                    connection_factory=control_connection_factory or _connection,
                )
                self.config_sync.attach(self.control.send)
        except BaseException:
            for close in (self.consumer.close, self.journal.close):
                try:
                    close()
                except Exception:  # noqa: BLE001
                    pass
            raise

    # ---- канал ----

    def handle_control(self, envelope: Envelope) -> list[Wire]:
        self.config_sync.on_ack(envelope)
        return []

    def _catalogue(self, envelope: Envelope) -> None:
        super()._catalogue(envelope)
        if (self._producer, self._run, self._catalogue_sequence) == (
                envelope.producer_id, envelope.run_id, envelope.sequence):  # каталог принят
            self.config_sync.observe_catalogue(envelope.producer_id, envelope.run_id, dict(self.cameras),
                                               envelope.payload["statuses"])

    # ---- камеры: желаемая конфигурация поверх каталога edge ----

    def _view(self) -> dict[str, CameraConfig]:
        with self._lock:
            catalogue = dict(self.cameras)
        return self.config_sync.view(catalogue)

    def list_cameras(self):
        return list(self._view().values())

    def get_camera(self, camera_id):
        return self._view().get(camera_id)

    def camera_status(self, camera_id):
        status = dict(super().camera_status(camera_id))
        sync = self.config_sync.status(camera_id)
        status["config_sync"] = sync
        if sync["state"] == "error":
            status["errors"] = {**(status.get("errors") or {}), "config_sync": sync["error"]}
        return status

    def _checked(self, camera: CameraConfig, *, new: bool) -> CameraConfig:
        camera = with_default_severity(camera)
        validate_camera_light(camera)
        ref = edge_source_ref(self.settings.edge_id, camera.camera_id)
        if camera.source.uri.startswith("edge://") and (new or camera.source.uri != ref):
            raise CameraError("source.uri: a new camera needs a real source; edge:// refers to an existing one"
                              if new else "source.uri refers to another edge camera")
        return camera

    def add_camera(self, camera):
        camera = self._checked(camera, new=True)
        if camera.camera_id in self._view():
            raise CameraExists(camera.camera_id)
        self.config_sync.submit("upsert", camera.camera_id, camera)
        return camera

    def update_camera(self, camera_id, camera):
        if camera.camera_id != camera_id:
            raise CameraError("camera_id in body must match the path")
        if camera_id not in self._view():
            raise KeyError(camera_id)
        camera = self._checked(camera, new=False)
        self.config_sync.submit("upsert", camera_id, camera)
        return camera

    def delete_camera(self, camera_id):
        if camera_id not in self._view():
            raise KeyError(camera_id)
        self.config_sync.submit("delete", camera_id)

    # ---- жизненный цикл ----

    def start(self):
        super().start()
        if self.control is not None:
            self.control.start()

    def stop(self):
        if self._closed:
            return
        if self.control is not None:
            self.control.stop()
        super().stop()
        self.config_sync.close()

    def health(self):
        return {**super().health(), "config_sync": self.config_sync_status()}

    def config_sync_status(self) -> dict[str, Any]:
        return {**self.config_sync.summary(), "link": self.control.status() if self.control else None}
