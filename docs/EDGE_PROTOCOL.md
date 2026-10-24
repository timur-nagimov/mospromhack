# Edge → gateway: протокол и гарантии хранения

Версия wire-протокола: **1**. Код контракта — `guard/transport/protocol.py`,
SQLite-состояния — `guard/transport/store.py`, публичный API хранения —
`guard/events/journal.py`. Протокол не зависит от RabbitMQ и не открывает
пути из принятых сообщений.

## Конверт

Сообщение — UTF-8 JSON-объект:

```json
{
  "version": 1,
  "edge_id": "edge-1",
  "producer_id": "a2be81d4-a5b0-4ab7-9faf-e6f908c42310",
  "run_id": "e459d705-b5e5-4a91-b266-c1eac82fd64f",
  "message_id": "a2be81d4-a5b0-4ab7-9faf-e6f908c42310:12",
  "kind": "event_update",
  "sequence": 12,
  "sent_at": 1792782000.0,
  "payload": {"event": "полный JSON Event вместо этой строки"}
}
```

`producer_id` — UUID, сохраняемый в той же БД, что события и outbox.
`run_id` — новый UUID при старте процесса Python. `sequence` монотонно
растёт в пределах producer, включая временные сообщения; после перезапуска
продолжается из БД. Пропуски последовательности допустимы.
`message_id` строго равен `producer_id:sequence`. При потере/замене БД новый
producer не может присвоить существующие события старого producer.

`edge_id`, camera/event/rule/type/zone ID — от 1 до 128 символов:
первая буква/цифра ASCII, затем ASCII-буквы, цифры, `_`, `-`, `.`.
UUID передаются в канонической форме с дефисами и строчными буквами.
Неизвестная версия, неизвестный kind, лишние поля конверта, повторные ключи
JSON, NaN/Infinity и неверный `message_id` отклоняются до записи событий.
Получатель передаёт настроенный `expected_edge_id` в `decode` или
`apply_remote_event`; сетевой слой отвечает за аутентификацию отправителя.

## Payload

- **event / event_update**: `{"event": <Event JSON>}` с необязательным
  `snapshot_jpeg` — JPEG в base64. Обе формы содержат полное состояние
  события, поэтому update может быть принят до первоначального event.
- **frame_meta**: `camera_id`, `capture_id` (UUID конкретного открытого
  источника), `frame_seq` (неотрицательный номер захваченного кадра),
  `capture_ts` (epoch), `source_pos_s` (позиция видео либо null),
  `width`, `height`, `jpeg_b64`, `people`, `objects`, `highlight`, `zones`.
  Кадр и его метаданные передаются вместе. Треки содержат `track_id`,
  нормированный `bbox: [x1,y1,x2,y2]`, `cls`, `conf: 0..1`; `highlight` —
  список ID. Зоны — JSON Zone с нормированными полигонами.
- **metrics**: `{"data": <объект metrics_snapshot>}`. Допускает вложенные
  группы, списки, null и конечные числовые значения.
- **cameras**: `{"cameras": [<CameraConfig JSON>], "statuses": {<camera_id>: <статус>}}`.
  `source.uri` обязательно равен `edge://<edge_id>/<camera_id>`.
  Перед формированием сообщения отправитель удаляет исходный URI и
  секреты из диагностических строк.

Общие пределы: **2 MiB** на полное сериализованное сообщение, **1 MiB**
на декодированный JPEG, глубина JSON до **32**, до **100 000** узлов.
Размеры frame_meta — **1..4096** по каждой стороне, до **16 777 216**
пикселей; до 4096 треков каждого типа, до 8192 выделенных ID, до 256 зон
и до 4096 точек полигона. Список камер — до 256.
Проверка JPEG в протоколе ограничивает размер/base64 и маркеры SOI/EOI;
полный разбор изображения и согласованность размеров проверяет runtime
перед показом/сохранением.

## Durable outbox на edge

```python
with Journal(path, replication_edge_id="edge-1") as journal:
    journal.add(event)  # events + sequence + outbox: одна транзакция
    journal.update(event.id, ended_at=event_end)
    for pending in journal.pending_messages(limit=100):
        # Runtime читает локальный снимок через собственный safe media API.
        frozen = journal.prepare_message(pending.message_id, snapshot_jpeg=jpeg_or_none)
        transport.publish(encode(frozen))  # дождаться publisher confirmation
        journal.mark_published(frozen.message_id)
```

`prepare_message` один раз атомарно сохраняет окончательный конверт и
признак подготовки, включая случай `snapshot_jpeg=None`. Все повторные
вызовы возвращают прежний wire body независимо от новых JPEG-аргументов,
изменения/удаления файла или перезапуска процесса. Это сохраняет identity
повторной доставки. До подготовки событие также уже находится в outbox;
ошибка чтения медиа не должна приводить к `mark_published`.
Неизвестный message_id в `prepare_message` вызывает `KeyError`.

Отказ записи события, sequence, outbox или подготовки целиком откатывает
транзакцию. `mark_published` вызывается только после подтверждённой
публикации; повторное подтверждение безопасно. Сбой между публикацией и
удалением из outbox приводит к повторной доставке. Если confirmation
потерян, сообщение остаётся для повторной отправки.

`make_message(kind, payload)` выделяет sequence из той же БД без записи в
outbox. Runtime использует его для кадров, метрик и камер; такие данные
временны. `replication_stats()` возвращает `pending`, `producer_id`,
`run_id`, `next_sequence`. `producer_id` и `run_id` также доступны как
свойства Journal. `pending_messages` принимает limit 0..10000.

## Идемпотентное применение на gateway

```python
envelope = decode(wire, expected_edge_id="edge-1")
# Сначала runtime сохраняет JPEG под локальным именем, зависящим от revision.
result = journal.apply_remote_event(
    envelope,
    expected_edge_id="edge-1",
    snapshot_path="snapshots/local-revision.jpg",  # или None
)
if result.changed:
    bus.publish(result.event)
# Consumer ACK — после commit, включая безопасный replay/stale result.
```

`ApplyResult` содержит `event: Event`, `created: bool`, `changed: bool`.
Под `BEGIN IMMEDIATE` gateway сохраняет событие, receipt с SHA-256
wire body и последнюю sequence этого события в одной SQLite-транзакции.
Точный replay не меняет событие. Повторный message_id с другим wire body
отклоняется. Более старая sequence не заменяет новое состояние; stale
receipt также сохраняется для обнаружения повторного использования ID.
События разных ID могут приходить в любом порядке.

Происхождение события (`edge_id`, `producer_id`) неизменно. Совпадение
ID с локальным событием или другим producer отклоняется. Неизменяемые
поля: ID, камера, участок, зона, правило, тип, критичность, трек,
время начала/подтверждения, исходный bbox. Более новая revision обновляет
свидетельства, задержки и окончание события.

При первом приёме статус всегда `new`, заметка — null. Последующие
revision сохраняют статус и заметку из текущей БД gateway, включая
конкурентный PATCH через другое соединение. Edge не может изменить
операторское решение. Ошибка записи откатывает также receipt и revision,
поэтому повторная доставка остаётся применимой.

`event.snapshot_path` и `event.clip_path` отправителя никогда не
используются как пути gateway. `clip_path` всегда null. `snapshot_path`
может задать только локальный media слой через отдельный аргумент:
относительный POSIX-путь до 512 символов без `..`, `.`, пустых сегментов,
обратного слеша, двоеточия и NUL. При аргументе None сохраняется уже
принятый локальный снимок; у нового события снимка нет.

Файловая система и SQLite не составляют общую транзакцию: runtime
сохраняет снимок до применения, использует уникальный путь для revision
и удаляет неиспользуемые файлы согласно своей политике. Путь не должен
перезаписываться поздним старым сообщением.

## Границы гарантии

Гарантия — локальная durable outbox + at-least-once delivery +
идемпотентная запись gateway; она требует persistent дисков SQLite,
publisher confirmations и ACK после commit. Факт отправки в сокет сам
по себе не разрешает удаление из outbox. Полный live Rabbit/TPU прогон
проверяется отдельно от тестов протокола и fake transport.

Outbox освобождается после подтверждения публикации. Gateway receipts и
origins сохраняются вместе с журналом; автоматического retention/GC в
этом этапе нет. Их удаление потребует отдельной политики с явно
ограниченным окном допустимой повторной доставки.

## AMQP topology v2 и запись до уведомления

Wire version остаётся 1. Exchange — `guard.edge.v1`, durable события —
`guard.gateway.<edge_id>.events`. Телеметрия использует новую очередь
`guard.gateway.<edge_id>.telemetry.v2`: durable metadata очереди,
TTL/max-length и сообщения с delivery_mode=1. Временные кадры/метрики
не приобретают гарантию доставки событий. RabbitMQ 4.4 отклоняет прежнюю
non-durable/non-exclusive очередь: это подтверждено первым живым прогоном.
Объявление новой очереди устраняет конфликт свойств с прежним именем.
Старую `.telemetry` приложение автоматически не удаляет; при обновлении
оба процесса должны использовать новую версию. Очереди событий и wire
version сохраняются.

Edge удерживает полный подготовленный пакет EventManager до успешной
фиксации каждого нового события и закрытия. Временная ошибка SQLite
повторяется с ограниченной экспоненциальной задержкой и прерываемым
ожиданием. После неопределённого исхода записи readback проверяет тот же
ID и identity; повтор не создаёт вторую outbox revision. Закрытие меняет
только ended_at, сохраняет операторские поля и bbox подтверждения.

Уведомление WS, events_created и планирование клипа нового события
доступны только после commit/readback. Если stop прервал незаписанный
пакет, shutdown сообщает ошибку и сохраняет SQLite/backend до полной
очистки камер. После восстановления можно повторить stop. Это не
гарантирует сохранение при crash до первой успешной durable записи:
недоступный диск и память процесса не составляют дополнительный spool.

Повторяемая проверка с настоящим RabbitMQ и аварийными границами:
`docs/RABBIT_ACCEPTANCE.md`; CPU-инференс проверяется отдельно.
