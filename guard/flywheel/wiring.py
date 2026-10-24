"""Подключение маховика к движкам конвейера: одна строка в PipelineEngine и в GatewayEngine.

У движка появляются атрибуты:
- flywheel_store — FlywheelStore (local и gateway; на edge журнала нет — None);
- flywheel_writer — LocalWriter: фоновая запись в хранилище и сообщение flywheel_sample в WS;
- flywheel — FlywheelCollector (local и edge; на gateway кадры приходят с edge — None);
- flywheel_assembler — сборка сэмплов edge из частей (только gateway).
Сбой хранилища не останавливает сервис: маховик выключается, оповещения работают.
"""

from __future__ import annotations

import logging
from typing import Any

from guard.flywheel.collector import DEFAULT_PERIOD_S, EDGE_PERIOD_S, FlywheelCollector, LocalWriter
from guard.flywheel.store import FlywheelStore

log = logging.getLogger(__name__)

LOCAL_MAX_WIDTH = 1280
LOCAL_QUALITY = 90


def _event_status(engine: Any, event_id: str) -> str | None:
    """Статус события в журнале (фоновый поток писателя): для operator_label сэмпла event."""
    ev = engine.journal.get(event_id)
    if ev is None:
        return None
    status = ev.status
    return getattr(status, "value", status)


def _open_store(engine: Any) -> FlywheelStore | None:
    root = engine.settings.flywheel_root
    try:
        return FlywheelStore(root)
    except Exception:  # noqa: BLE001
        log.exception("flywheel store at %s unavailable: collection disabled", root)
        return None


def period_for(settings: Any) -> float:
    if settings.flywheel_period_s:
        return float(settings.flywheel_period_s)
    return EDGE_PERIOD_S if settings.mode == "edge" else DEFAULT_PERIOD_S


def attach_flywheel(engine: Any) -> None:
    """PipelineEngine: local — хранилище и писатель; edge — отправка сэмплов на шлюз (протокол edge → шлюз)."""
    st = engine.settings
    engine.flywheel_store = None
    engine.flywheel_writer = None
    engine.flywheel = None
    engine.flywheel_assembler = None
    if st.mode == "edge":
        # Конверты выдаёт журнал edge с репликацией (EdgeEngine). Старый путь через sink сэмплы не возит.
        journal = getattr(engine, "journal", None)
        if engine.sink is not None or getattr(journal, "producer_id", None) is None:
            log.info("flywheel: edge without replication journal, collection disabled")
            return
        from guard.flywheel.amqp import edge_publisher
        from guard.flywheel.edge import EdgeSender

        output: Any = EdgeSender(lambda: edge_publisher(engine), journal.make_message,
                                 kbps_per_camera=st.flywheel_edge_kbps,
                                 cameras=lambda: max(1, len(engine.workers)))
    else:
        store = _open_store(engine)
        if store is None:
            return
        engine.flywheel_store = store
        engine.flywheel_writer = LocalWriter(
            store,
            publish=lambda message: engine.bus.publish(message),
            label_lookup=lambda event_id: _event_status(engine, event_id),
            max_width=LOCAL_MAX_WIDTH,
            quality=LOCAL_QUALITY,
        )
        output = engine.flywheel_writer
    shadow = None
    if st.flywheel and st.flywheel_shadow:
        try:
            from guard.flywheel.shadow import make_shadow

            shadow = make_shadow(engine)
        except Exception:  # noqa: BLE001 — без сверки сбор работает
            log.exception("flywheel: shadow check unavailable")
    engine.flywheel = FlywheelCollector(
        output,
        period_s=period_for(st),
        daily_cap=st.flywheel_daily_cap,
        tracker_high=st.tracker_high_thresh,
        enabled=st.flywheel,
        model_thresholds={mid: float(spec.conf_threshold) for mid, spec in (engine.models or {}).items()},
        shadow=shadow,
        shadow_period_s=st.flywheel_shadow_period_s,
    )


def attach_gateway_flywheel(engine: Any) -> None:
    """GatewayEngine: хранилище, писатель и сборка сэмплов, пришедших с edge частями.

    С настоящим RabbitConsumer шлюз читает части из своей очереди (guard/flywheel/amqp.py) отдельным
    соединением. Части, пришедшие через GatewayEngine.receive (тесты, другой транспорт), идут в тот же сборщик."""
    engine.flywheel = None
    engine.flywheel_writer = None
    engine.flywheel_assembler = None
    engine.flywheel_receiver = None
    engine.flywheel_store = _open_store(engine)
    if engine.flywheel_store is None:
        return
    engine.flywheel_writer = LocalWriter(
        engine.flywheel_store,
        publish=lambda message: engine.bus.publish(message),
        label_lookup=lambda event_id: _event_status(engine, event_id),
        max_width=LOCAL_MAX_WIDTH,
        quality=LOCAL_QUALITY,
    )
    from guard.flywheel.amqp import FlywheelConsumer, FlywheelReceiver
    from guard.flywheel.edge import SampleAssembler
    from guard.transport.rabbit import RabbitConsumer

    engine.flywheel_assembler = SampleAssembler(engine.flywheel_writer)
    consumer = getattr(engine, "consumer", None)
    if isinstance(consumer, RabbitConsumer):
        engine.flywheel_receiver = FlywheelReceiver(FlywheelConsumer(consumer.config))


def start_flywheel(engine: Any) -> None:
    writer = getattr(engine, "flywheel_writer", None)
    collector = getattr(engine, "flywheel", None)
    if collector is not None:
        collector.start(getattr(engine, "flywheel_store", None))
    elif writer is not None:
        writer.start()
    receiver = getattr(engine, "flywheel_receiver", None)
    assembler = getattr(engine, "flywheel_assembler", None)
    if receiver is not None and assembler is not None:
        try:
            receiver.start(assembler.add_envelope)
        except Exception:  # noqa: BLE001 — без приёма сэмплов шлюз оповещает как обычно
            log.exception("flywheel: gateway sample receiver failed to start")


def stop_flywheel(engine: Any) -> None:
    receiver = getattr(engine, "flywheel_receiver", None)
    if receiver is not None:
        receiver.stop()  # поток приёма не держит SQLite: зависший не мешает закрыть хранилище
    collector = getattr(engine, "flywheel", None)
    writer = getattr(engine, "flywheel_writer", None)
    if collector is not None:
        collector.stop()
    elif writer is not None:
        writer.stop()
    store = getattr(engine, "flywheel_store", None)
    if store is not None:
        store.close()


def operator_decision(engine: Any, event_id: str, status: Any) -> None:
    """PATCH /api/events/{id} со status → operator_label сэмплов event этого события."""
    store = getattr(engine, "flywheel_store", None)
    if store is None:
        return
    from guard.flywheel.store import OPERATOR_LABELS

    value = getattr(status, "value", status)
    try:
        store.set_operator_label(event_id, OPERATOR_LABELS.get(str(value)))
    except Exception:  # noqa: BLE001
        log.exception("flywheel: operator label for %s failed", event_id)
