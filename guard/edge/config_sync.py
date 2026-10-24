"""Edge с каналом конфигурации камер от шлюза (docs/EDGE.md §8).

Поверх EdgeEngine: команды camera_config применяются атомарной конфигурацией движка,
изменения пишутся в оверлей рядом с базовым файлом камер (базовый файл не переписывается),
при старте читаются базовый файл и оверлей. Ответ camera_config_ack — конверт edge v1.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from guard.contracts import CameraConfig
from guard.edge.runner import EdgeEngine
from guard.pipeline.config_sync import ACK_KIND, CameraOverlay, EdgeConfigApplier, overlay_path
from guard.transport.control import RabbitControlLink, Wire
from guard.transport.protocol import Envelope, encode
from guard.transport.rabbit import RabbitConfig, _connection

log = logging.getLogger(__name__)


class ConfigSyncEdgeEngine(EdgeEngine):
    """EdgeEngine, который принимает конфигурацию камер от шлюза.

    control_connection_factory — фабрика соединений AMQP для канала (тесты подставляют свою)."""

    def __init__(self, settings, *, control_connection_factory: Callable | None = None, **kwargs: Any) -> None:
        self._config_change: tuple[str, int] | None = None  # (camera_id, версия) для _persist во время команды
        self.control: RabbitControlLink | None = None
        super().__init__(settings, **kwargs)
        try:
            self.config_overlay = CameraOverlay(overlay_path(settings.cameras_config, settings.cameras_overlay))
            self.config_applier = EdgeConfigApplier(self, self.config_overlay, settings.edge_id)
            if settings.config_sync and settings.rabbit_url and (
                    settings.transport == "rabbit" or control_connection_factory is not None):
                self.control = RabbitControlLink(
                    RabbitConfig(settings.rabbit_url, settings.edge_id, suffix=settings.rabbit_suffix),
                    role="edge", handler=self.handle_control,
                    connection_factory=control_connection_factory or _connection,
                )
        except BaseException:
            for resource in (self.journal, self.backend):
                try:
                    resource.close()
                except Exception:  # noqa: BLE001
                    pass
            raise

    # ---- конфигурация: базовый файл + оверлей ----

    def _load_camera_configs(self) -> list[CameraConfig]:
        return self.config_overlay.merge(super()._load_camera_configs())

    def _persist(self, cameras: dict[str, CameraConfig]) -> None:
        # Базовый файл (configs/cameras.stand.yaml) не переписываем: только оверлей.
        self.config_overlay.commit(cameras, self._config_change)

    def handle_control(self, envelope: Envelope) -> list[Wire]:
        payload = self.config_applier.handle(envelope)
        ack = self.journal.make_message(ACK_KIND, payload)
        # Каталог камер — сразу, а не через telemetry_interval_s: шлюз увидит новую конфигурацию быстрее.
        self.replication._next_metrics = float("-inf")
        return [(ack.message_id, encode(ack))]

    def camera_status(self, camera_id: str) -> dict[str, Any]:
        status = super().camera_status(camera_id)
        return {**status, "config_version": self.config_overlay.applied_version(camera_id)}

    # ---- жизненный цикл ----

    def start(self) -> None:
        super().start()
        if self.control is not None:
            self.control.start()

    def stop(self) -> None:
        if self.control is not None:
            self.control.stop()  # команды больше не принимаются до остановки камер и журнала
        super().stop()

    def health(self) -> dict[str, Any]:
        return {**super().health(), "config_sync": self.config_sync_status()}

    def metrics_snapshot(self) -> dict[str, Any]:
        return {**super().metrics_snapshot(), "config_sync": self.config_sync_status()}

    def config_sync_status(self) -> dict[str, Any]:
        return {**self.config_applier.status(), "link": self.control.status() if self.control else None}
