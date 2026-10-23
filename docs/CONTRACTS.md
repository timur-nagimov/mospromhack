# Контракты системы

Код контрактов: `guard/contracts.py` (типы) и `guard/rules/base.py` (API правил). Меняет их только оркестратор. Если агенту нужно изменение, он описывает его в отчёте.

## Общие пути

- Корень: `/Users/nagimov-timur/Desktop/mosprom`. Python: `/Users/nagimov-timur/Desktop/mosprom/.venv/bin/python` (3.11). Агенты в worktree используют этот же venv по абсолютному пути. Пакеты ставятся командой `uv pip install --python <venv-python> <pkg>`, новая зависимость дописывается в `pyproject.toml`.
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
- `InferenceBackend` (`guard/inference/base.py`, владелец A3):
  - `name`: `"onnx-cpu"` или `"linq-tpu"`;
  - `load(spec: ModelSpec) -> None`;
  - `infer(model_id: str, batch: np.ndarray) -> list[np.ndarray]` — синхронный и потокобезопасный. На вход батч NCHW в `input_dtype`, на выход сырые выходы модели в том же формате, что у ONNX;
  - `stats() -> dict` — p50 и p95 по этапам, fps, ошибки.
- Реализации: `guard/inference/onnx_cpu.py` (A3) и `guard/inference/linq_tpu.py` (A1). Бэкенд выбирается по имени через `guard.inference.get_backend(name)`, который импортирует модули лениво.
- Пре- и постпроцессинг (`guard/vision/`, Codex CX-001):
  - `letterbox(img, size) -> (tensor, meta)`;
  - `decode_yolov8(outputs, meta, spec) -> list[Detection]`;
  - `decode_yolov8_pose(...)`;
  - NMS на numpy.
- Решение «где делать препроцессинг» (ноутбук или рядом с TPU) записывается в `.orchestra/DECISIONS.md` по замерам.

## Треки (`guard/tracking/`, Codex CX-002)

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

- `EventManager` (Codex CX-003):
  - копит нарушения по ключу `(camera_id, rule_id, track_id | spatial cell)`;
  - событие подтверждается, когда нарушение длится не меньше `min_duration_s` и занимает не меньше `min_ratio` кадров окна;
  - строго одно событие на нарушение; пока трек жив, повторов нет;
  - при смене ID трека — дедуп по IoU ≥ 0.3 с открытым или недавно закрытым событием того же правила в пределах `cooldown_s`;
  - событие закрывается через `close_after_s` без нарушения.
- `Journal` (Codex CX-004):
  - SQLite `data/guard.db`;
  - `add`, `update_status`, `close`, `get`;
  - `query(type, camera_id, zone_id, status, ts_from, ts_to, limit, offset)`;
  - `stats(group_by=type|camera|zone|hour|day, ts_from, ts_to)`.
- Медиа (A3): кольцевой буфер кадров на камеру. Снапшот JPEG с выделенным человеком сохраняется сразу при подтверждении; клип H.264 mp4 ±5 с — после события (ffmpeg из imageio-ffmpeg). Всё в `data/media/`.
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
- `GET /api/stats?group_by=type|camera|zone|hour|day&from=&to=` → `{group_by, items: [{key, count}]}`.
- `GET /api/metrics` → `{backend, cameras: {id: {fps_in, fps_processed, lag_s, timings_ms: {key: {p50, p95}}}}, inference: backend.stats()}`.
- `WS /ws/events` — сообщения:
  - `{"kind": "event", "event": Event}` — новое подтверждённое событие;
  - `{"kind": "event_update", "event": Event}`;
  - `{"kind": "metrics", "data": <как /api/metrics>}` раз в 2 с.
- Конфиг камер хранится в `configs/cameras.yaml`. UI меняет его через API, перезапуск процесса не нужен.

## Владение путями

| Путь | Владелец |
|---|---|
| `guard/contracts.py`, `guard/rules/base.py`, `docs/CONTRACTS.md` | оркестратор |
| `guard/sources/`, `guard/inference/{base,onnx_cpu,__init__}.py`, `guard/pipeline/`, `guard/api/`, `guard/metrics/`, `guard/events/media.py`, `guard/rules/geometry.py`, `guard/rules/plugins/zone_intrusion.py` | A3 backend |
| `guard/inference/linq_tpu.py`, `tpu/`, `tools/jhub.py`, `docs/LINQ_NOTES.md` | A1 infra-tpu |
| `ml/`, `configs/models.yaml`, `docs/MODELS.md` | A2 ml |
| `frontend/` | A4 frontend |
| `data/` (вне git), `docs/DATA.md`, `tools/fetch_data.*` | A5 data-demo |
| `guard/vision/`, `guard/tracking/`, `guard/events/{manager,journal}.py` и их тесты | Codex |
