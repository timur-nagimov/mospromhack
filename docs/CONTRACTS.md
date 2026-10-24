# Контракты системы

Код контрактов: `guard/contracts.py` (типы) и `guard/rules/base.py` (API правил). Изменения этих интерфейсов согласуются с зависимыми компонентами и отражаются в документации.

## Общие пути

- Корень кода: checkout (`CODE_ROOT`), Python 3.11 — `<checkout>/.venv/bin/python`. Общие данные/модели и поиск окружения можно направить через `GUARD_ROOT`; явные `GUARD_DATA_DIR`/`GUARD_MODELS_DIR` имеют приоритет. Пакеты ставятся командой `uv pip install --python <venv-python> <pkg>`, новая зависимость дописывается в `pyproject.toml`.
- Данные: `$GUARD_DATA_DIR` = `<корень>/data`. Внутри `videos/`, `images/`, `media/` (снапшоты и клипы событий) и `guard.db` (SQLite). В git не попадают.
- Модели: `$GUARD_MODELS_DIR` = `<корень>/models`. ONNX, программы TPU, `<model_id>.json` с классами. В git не попадают. Реестр моделей хранится в `configs/models.yaml`.
- Доступы лежат в `<корень>/.env`, его не коммитить.

## Поток данных

```
Source (file/rtsp/images) → sampler (sample_fps) → batcher → Detector(model_id)
  → [preprocess → InferenceBackend.infer → postprocess] → list[Detection] по моделям
  → Tracker (люди и объекты, атрибуты из ppe/pose/phone) → FrameContext
  → RulePlugin.evaluate() × правила камеры → list[Violation]
  → EventManager (окно подтверждения, дедуп, закрытие) → Event
  → Journal (SQLite) + Media (снапшот, клип ±5 с) + WebSocket /ws/events → UI
```

## Модели и инференс

- `ModelSpec` описан в `configs/models.yaml`: model_id, task, onnx_path, tpu_program, input_size, input_dtype, classes, output_format, пороги.
- `InferenceBackend` (`guard/inference/base.py`):
  - `name`: `"onnx-cpu"` или `"linq-tpu"`;
  - `load(spec: ModelSpec) -> None`;
  - `infer(model_id: str, batch: np.ndarray) -> list[np.ndarray]` — синхронный и потокобезопасный. На вход батч NCHW в `input_dtype`, на выход сырые выходы модели в том же формате, что у ONNX;
  - `stats() -> dict` — p50 и p95 по этапам, fps, ошибки.
- Реализации: `guard/inference/onnx_cpu.py` и `guard/inference/linq_tpu.py`. `guard.inference.get_backend(name)` импортирует реализацию лениво. Production entrypoints `scripts/run_tpu.sh`/`tools/stand.py start` выбирают внешний `linq-tpu` и `GUARD_BACKEND_FALLBACK=none`; ошибка TPU не превращается в успешный CPU inference. CPU используется явно для диагностики. Developer defaults `local`/`onnx-cpu` сохраняются, но не служат production-приёмкой.
- Параметры TPU inference `LINQ_RABBIT_HOST/PORT/USER/PASSWORD` независимы от `GUARD_RABBIT_URL` event bus. Raw-упаковка и timeout задаются отдельно (`LINQ_RAW`, `LINQ_TIMEOUT_S`); stand launcher выбирает raw и 5 с.
- Пре- и постпроцессинг (`guard/vision/`, CX-001):
  - `letterbox(img, size) -> (tensor, meta)`;
  - `decode_yolov8(outputs, meta, spec) -> list[Detection]`;
  - `decode_yolov8_pose(...)`;
  - NMS на numpy.
- Решение «где делать препроцессинг» (ноутбук или рядом с TPU) записывается в `docs/ARCHITECTURE.md` по замерам.

## Треки (`guard/tracking/`, CX-002)

- ByteTrack отдельно для людей (`person`) и для техники (`car`, `truck`, `bus`, `forklift`, …).
- Атрибуты человека переносятся из других моделей по пересечению рамок:
  - `helmet`, `vest`, `respirator` — `True`, `False` или `None` (не видно);
  - `phone`;
  - `pose_state` — `standing`, `lying` или `unknown`;
  - `keypoints`.
- `Track.history` — `(ts, foot_x, foot_y, h)`. `speed_bh` — скорость точки опоры в высотах тела в секунду.

## Правила-плагины (`guard/rules/plugins/`)

- Класс наследует `RulePlugin`, помечен `@register_rule`, задаёт `type`, `title`, `required_models` и `params_schema` (JSON Schema для формы в UI). Метод `evaluate(ctx) -> list[Violation]` работает без памяти о подтверждении.
- Зоны задаются нормированными полигонами. Нарушение «в зоне» проверяется по точке опоры (низ-центр рамки). Если текущее время вне `active_hours`, зона неактивна.
- Сценарии и имена типов:

| Сценарий | type |
|---|---|
| S1 | `zone_intrusion` |
| S2 | `ppe_missing` |
| S3 | `fall_immobile` |
| S4 | `vehicle_proximity` |
| S5 | `running` |
| S6 | `phone_usage` |
| S7 | `smoking_fire` |
| S8 | `fence_climb` |
| S9 | `height_no_harness` |
| S10 | `unsafe_carry` |

## События (`guard/events/`)

- `EventManager` (CX-003):
  - копит нарушения по ключу `(camera_id, rule_id, track_id | spatial cell)`;
  - событие подтверждается, когда нарушение длится не меньше `min_duration_s` и занимает не меньше `min_ratio` кадров окна;
  - строго одно событие на нарушение; пока трек жив, повторов нет;
  - при смене ID трека — дедуп по IoU ≥ 0.3 с открытым или недавно закрытым событием того же правила в пределах `cooldown_s`;
  - событие закрывается через `close_after_s` без нарушения.
- `Journal` (CX-004):
  - SQLite `data/guard.db`;
  - `add`, `update_status`, `close`, `get`;
  - `query(type, camera_id, zone_id, status, ts_from, ts_to, limit, offset)`;
  - `stats(group_by=type|camera|zone|hour|day, ts_from, ts_to)`.
- Медиа: кольцевой буфер кадров на камеру. Снапшот JPEG с выделенным человеком сохраняется сразу при подтверждении; клип H.264 mp4 ±5 с — после события (ffmpeg из imageio-ffmpeg). Всё в `data/media/`.
- Задержка оповещения = `confirmed_at − started_at` (окно подтверждения) + время обработки. Время обработки пишется в `Event.latency_ms`.

## HTTP API (FastAPI, `guard/api/`, порт 8000)

- `GET /health` → `{status, backend, cameras_running}`.
- Камеры:
  - `GET /api/cameras`, `POST /api/cameras`, `PUT /api/cameras/{id}`, `DELETE /api/cameras/{id}`;
  - `GET /api/cameras/{id}/frame.jpg` — последний кадр без разметки, нужен редактору зон;
  - `GET /api/cameras/{id}/stream.mjpg` — MJPEG с оверлеями: рамки, треки, зоны.
- `GET /api/rules/types` → `available_rules()`.
- События:
  - `GET /api/events?type=&camera_id=&zone_id=&status=&from=&to=&limit=&offset=` → `{items: Event[], total}`;
  - `GET /api/events/{id}`;
  - `PATCH /api/events/{id}` с телом `{status, operator_note}`;
  - `GET /api/events/{id}/snapshot.jpg` и `GET /api/events/{id}/clip.mp4`.
- `GET /api/stats?group_by=type|camera|zone|site|hour|day|weekday_hour|severity|status&from=&to=` → `{group_by, items: [{key, count}]}`.
- `GET /api/analytics?from=&to=` → согласованный набор агрегатов периода (формат ниже).
- `GET /api/metrics` → `{backend, cameras: {id: {fps_in, fps_processed, lag_s, timings_ms: {key: {p50, p95}}}}, inference: backend.stats()}`.
- `WS /ws/events` — сообщения:
  - `{"kind": "event", "event": Event}` — новое подтверждённое событие;
  - `{"kind": "event_update", "event": Event}`;
  - `{"kind": "metrics", "data": <как /api/metrics>}` раз в 2 с.
- Конфиг камер хранится в `configs/cameras.yaml`. UI меняет его через API, перезапуск процесса не нужен.

## Карта компонентов

| Путь | Компонент |
|---|---|
| `guard/contracts.py`, `guard/rules/base.py`, `docs/CONTRACTS.md` | Общие контракты |
| `guard/sources/`, `guard/inference/{base,onnx_cpu,__init__}.py`, `guard/pipeline/`, `guard/api/`, `guard/metrics/`, `guard/events/media.py`, `guard/rules/geometry.py`, `guard/rules/plugins/zone_intrusion.py` | Backend и конвейер |
| `guard/inference/linq_tpu.py`, `tpu/`, `tools/jhub.py`, `docs/LINQ_NOTES.md` | TPU-интеграция и инфраструктура |
| `ml/`, `configs/models.yaml`, `docs/MODELS.md` | Модели и оценка |
| `frontend/` | Пользовательский интерфейс |
| `data/` (вне git), `docs/DATA.md`, `tools/fetch_data.*` | Данные и демонстрационные сценарии |
| `guard/vision/`, `guard/tracking/`, `guard/events/{manager,journal}.py` и их тесты | Зрение, трекинг и события |

## Уточнения API (24.10, по итогам интеграции фронтенда)

- `GET /api/events`:
  - дополнительный фильтр `severity=`;
  - `from` и `to` — epoch-секунды, фильтр по `confirmed_at`;
  - фильтры принимают значения через запятую.
- `GET /api/stats?group_by=`: допустимы `type`, `camera`, `zone`, `severity`, `status`, `site`, `hour`, `day`, `weekday_hour` (ключ `"<0-6>:<0-23>"`, понедельник = 0, время МСК).
- `GET /api/cameras`: поле `status` (running, stopped, error плюс fps) — только для чтения. В PUT и POST оно игнорируется.
- `GET /api/metrics`: `inference` = `{backend, device, utilization (0..1, если известно), models: {id: {fps, calls, max_batch_seen, timings_ms: {transfer, queue, infer}, errors, last_error}}}`.
- `PATCH /api/events/{id}`: возвращает полный Event и рассылает `event_update` по WS. Когда клип готов, тоже рассылается `event_update` с `clip_path`.
- Ключи `Event.details`, которые понимает UI: `missing`, `vehicle`, `distance_bh`, `distance_m`, `immobile_s`, `speed_bh`, `pose_state`, `source`, `cls`, `conf`.
- Оверлеи MJPEG компактные: ID трека, а у нарушителя — короткая метка нарушения. Неизвестные атрибуты («?») не выводить.
- Режимы запуска: `local`, `edge` и `gateway`, выбираются `GUARD_MODE`. Сообщения транспорта edge → gateway: `event`, `event_update`, `frame_meta`, `metrics`, `cameras`.

## Локальная реализация после проверки (24.10)

- `/api/stats` принимает `type`, `camera_id`, `zone_id`, `status`, `severity` со списками через запятую, аналогично `/api/events`. Даты `from`/`to` включительны, относятся к `confirmed_at`, принимают epoch-секунды и ISO 8601 (без timezone — МСК). Неконечные epoch-значения отклоняются с 422.
- Группа `site` использует сохранённое `Event.site`: изменение или удаление камеры не переписывает историю. `weekday_hour` имеет ключ `"<0-6>:<0-23>"` в `Europe/Moscow`.
- `/api/analytics` принимает `from`/`to` с теми же правилами и возвращает один SQLite-снимок для всех графиков, без пагинации и выгрузки событий:

  ```json
  {
    "types": [], "cameras": [], "sites": [], "days": [],
    "weekdayHours": [], "criticalTypes": [],
    "statusTypes": {"new": [], "confirmed": [], "false": [], "forwarded": []}
  }
  ```

  Каждый массив содержит `{key, count}`. `criticalTypes` группирует по типам события с `severity=critical`; UI объединяет их с критичными типами нарушений без двойного счёта. Все группы и статусы читаются внутри одной read-транзакции, поэтому конкурентный PATCH не удваивает и не теряет событие между частями ответа. Лимит `/api/events` остаётся 1000 на страницу.
- CRUD камер сначала атомарно сохраняет кандидат YAML и затем меняет runtime. Отказ сохранения не меняет работающие камеры и не закрывает их события. Если worker не запускается уже после успешного сохранения, конфигурация остаётся сохранённой, API сообщает ошибку, а GET камер показывает `running=false` и `errors.start`. Полный rollback внешнего источника не гарантируется.
- `python -m guard.doctor [--json]` проверяет только локальную конфигурацию и предпосылки файловой системы. `scope=offline`; коды выхода 0/1/2 означают отсутствие локальных ошибок / найденные ошибки / неверный вызов. Команда не создаёт данные/БД/медиа и не запускает источники, inference или сетевую проверку. Доступность TPU и качество распознавания этой проверкой не подтверждаются.
- Режимы edge/gateway и транспорт v1 реализованы; локальный режим остаётся значением по умолчанию. Гарантии хранения и wire-формат: `docs/EDGE_PROTOCOL.md`; команды и настройки запуска: README.

## Edge/gateway v1 (24.10, локальная реализация)

- Для пары процессов обязательны `GUARD_EDGE_ID` и `GUARD_RABBIT_URL`; SQLite и media-каталоги разделены. Exchange `guard.edge.v1` отделён от канала inference TPU. На edge событие и outbox фиксируются в одной транзакции; publisher confirmation предшествует удалению outbox. Получатель подтверждает доставку после commit события и receipt.
- Повторная доставка идемпотентна. Edge не меняет `status` и `operator_note` gateway. При сбое локального SQLite edge не переходит к непостоянному журналу. Snapshot/body замораживаются до первой отправки и сохраняются при restart.
- Gateway не загружает модели, inference или источники. Каталог камер получает от edge; POST/PUT/DELETE камер возвращают 409, PATCH событий работает локально с WS-уведомлением. URI источника заменён на `edge://<edge_id>/<camera_id>`.
- JPEG события сохраняется под проверенным локальным путём gateway. Пути edge не используются; `clip_path=null`, доставка клипов ещё не реализована. Повреждённый/слишком большой снимок не блокирует durable событие, ошибка видна в состоянии репликации.
- Превью — JPEG конкретного analysis-кадра вместе с bbox/зонами, `capture_id`, `frame_seq`, `capture_ts`, `source_pos_s`. Наложения относятся к этому же изображению. По умолчанию до 1 кадра/с, ширина до 480 px, срок свежести 10 с. Возраст считается от захвата, задержка передачи/обработки уменьшает остаточный срок; старые кадры не оживают после reconnect. При истечении срока frame API возвращает 503.
- Камеры/кадры/метрики — временная телеметрия в отдельной ограниченной очереди с TTL. События durable. Полные гарантии требуют persistent SQLite, работающих AMQP confirms/ACK и отдельного живого прогона; fake-broker тесты не подтверждают доступность реального брокера или TPU.
- AMQP topology v2: очередь `.telemetry.v2` имеет durable metadata и TTL/max-length, сообщения телеметрии остаются delivery_mode=1. Это устраняет запрет non-durable/non-exclusive очередей в RabbitMQ 4.4 без включения deprecated feature; wire version остаётся 1. Старую очередь приложение не удаляет.
- В edge новые/закрытые события удерживаются до commit, transient SQLite failure повторяется; success-уведомления нового события идут после commit/readback. При прерывании незаписанного пакета stop явно завершается ошибкой и сохраняет ресурсы до повторной очистки. Crash при недоступном диске до первой durable записи остаётся вне гарантии.

## Совмещённые local/edge/gateway и отчёты

- `GUARD_MODE=local|edge|gateway` задаёт размещение компонентов. Headless `python -m guard.edge` использует durable `EdgeEngine`, SQLite outbox и wire v1; это не upstream `MemoryJournal`/транзиентный binary codec. Совместимые `RabbitSink`/`RabbitSource` и `guard.transport.codec` оставлены для старых callers/tests и не являются production-каналом.
- `GUARD_TRANSPORT=rabbit` — event bus с обязательными identity и URL; явные `memory`/`log` CLI edge предназначены для локальных тестов/диагностики. `guard.edge --url` задаёт event bus, `tools/stand.py --url` — JupyterHub. `GUARD_TRANSPORT_URL` остаётся совместимым alias только в edge CLI; для пары процессов задают `GUARD_RABBIT_URL`.
- `GET /health` возвращает `status=ok|degraded|error`, `mode`, `backend`, число работающих/всех камер и uptime, а также доступные сведения backend/репликации. Edge не заменяет родительский `degraded/error` на `ok`; незаписанные revisions дают `error`. В gateway health описывает связь/доставку и не является проверкой TPU inference.
- В `/api/metrics` могут присутствовать `mode`, `media`, состояние edge/transport и backend counters. PID, успешный doctor, накопленные сообщения и старые счётчики не доказывают активный TPU: приёмка требует свежих успешных вызовов каждой нужной модели и новых обработанных кадров при выключенном fallback.
- Отсутствующая модель для включённого правила — ошибка конфигурации/запуска, а не основание тихо выключить правило. Важность по умолчанию берётся из `default_severity` плагина через `with_default_severity`.
- На gateway камеры остаются только для чтения (409 для POST/PUT/DELETE); upstream-семантика изменения локального YAML gateway не используется. Конфиг и правила меняют на local/edge.
- `GET /api/reports/shift` формирует JSON/HTML/Markdown (`format=json|html|md`) по тому же журналу. Его границы полуоткрытые `[from,to)` по `confirmed_at`, epoch или ISO 8601; даты без timezone — МСК. Без границ выбирается смена 08–20 или 20–08. `GET /api/reports/presets` возвращает current_shift/previous_shift/day/week. Эти полуоткрытые границы не меняют включительные фильтры `/api/events`, `/api/stats` и `/api/analytics`.
- `scripts/run_tpu.sh` не удаляет очередь, receipt или outbox. `GUARD_VPN_MODE=auto` использует готовые AMQP/HTTP-пробросы, иначе пробует контейнер проекта; `external` запрещает автоматический VPN up, `container` выбирает контейнер явно. Эти проверки доступа и готовность процесса не заменяют активную приёмку новой версии. Исторические результаты отдельных TPU/CPU/channel-прогонов сохраняются как датированные измерения со своими ограничениями.
