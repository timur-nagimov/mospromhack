# Сквозная проверка нового нарушения через HTTP и WebSocket

```bash
python -m pytest -q tests/test_violation_api_integration.py
```

Один синтетический сценарий `zone_dwell` проходит через настоящий
`CameraWorker.process`, preprocessing/YOLO decoder, Tracker, плагин, EventManager,
callbacks PipelineEngine, SQLite Journal, HTTP API и TestClient WebSocket
`/ws/events`. Подменены только inference и media; результат модели — фиксированный
raw YOLO tensor либо корректный пустой выход. Кадры публикуются через настоящий
источник и его callback вручную, без запуска source/analysis threads.

Проверяется отсутствие раннего события: сначала подтверждение трека, затем dwell,
затем окно EventManager. Уведомление `event` полностью совпадает с HTTP и независимым
SQLite readback, включая EventID, детали и capture origin. Текущие model/rule
observation statuses сверяются между FrameContext и HTTP camera status; они не
объявляются частью Event, в котором этих полей сейчас нет.

Операторский PATCH даёт ровно `event_update`; повторные нарушения не создают новый
alarm. Смена capture закрывает прежний инцидент на последнем violation, сохраняя
operator status/note, и создаёт свежие tracker/rule state. Повторно использованный
числовой track ID на том же месте получает отдельный EventID после новых dwell и
confirmation окон. Второй инцидент закрывается по обычному `close_after_s`.
FIFO bus markers и общий трёхсекундный deadline каждого WS drain проверяют число
уведомлений без sleeps и неограниченного ожидания.

Это проверка проводки одного нового правила. Она не измеряет качество модели,
реальные камеры/RTSP, scheduler throughput, media encoding, RabbitMQ или внешний
TPU. Алгоритмы и production code не изменяются.

## Следующий небольшой пакет СИЗ

Текущая PPE модель объявляет `goggles/no_goggles` и `gloves/no_gloves`
(`configs/models.yaml:89`), но `guard/tracking/attributes.py:13` и
`guard/pipeline/ppe.py:14` пока потребляют только helmet/vest/respirator.
Следующий кандидат для очков/перчаток должен иметь явные positive **и** negative
классы, проверенную привязку наблюдения к человеку, правила видимости и отдельную
оценку качества. Наличие имени класса само по себе этого не доказывает.

Минимальный будущий capability gate: optional декларации модели в
`guard/contracts.py:52`, pure разрешение capabilities при загрузке реестра
`guard/pipeline/config.py:105`, opt-in требования плагина в
`guard/rules/base.py:19`, проверка до persistence в
`guard/pipeline/engine.py:259` и повторная проверка при startup в
`guard/pipeline/worker.py:111`. Старые конфиги и required_models сохраняют прежнюю
семантику; новые отсутствие СИЗ нельзя включать при unknown/unsupported capability.
CPU и LinQ используют один статический контракт; эта проверка не должна создавать
ORT session, обращаться к TPU или объявлять TPU parity/quality подтверждёнными.

У обуви основного PPE реестра нет explicit negative класса. `no_mask` не означает
`no_respirator`; модели привязи тоже нет. Такие состояния остаются unknown,
отсутствие bbox не превращается в `False`. Существующий strict height mode —
контроль доступа, а не распознавание привязи
(`guard/rules/plugins/height_no_harness.py:6`). Capability-пакет здесь не реализован.
