# Маховик данных: система сама собирает обучающую выборку

Контракт реализации от 24.10. Требования, пути и команды приведены в соответствующих
разделах. Решение внутреннее, для демонстрации: лицензии открытых наборов не проверяем, но источник
каждого набора записываем.

## Зачем

Замкнутый цикл качества:
1. Оператор нажал «ложное».
2. Кадр попал в выборку.
3. Учитель разметил его, человек проверил.
4. Модель дообучилась и пересобрана под LinQ.
5. Новая модель выкатывается, только если на отложенных данных она не хуже прежней.

Принципы:
- **Данные маховика — только для обучения.** Качество меряем на отложенных данных, которые в обучение не попадали:
  - ролики из `ml/eval/scenarios.yaml`;
  - тестовые части открытых наборов.
- **Кадры отложенных роликов в обучающую выборку не попадают.** Сэмпл с камеры, чей источник — ролик из
  `scenarios.yaml`, получает `holdout = 1`. Экспорт пропускает такие сэмплы, если не указан явный флаг
  `--include-holdout`. С этим флагом отчёт помечает, что оценка на этих роликах загрязнена.
- **Без человека метки в обучение не идут.** Исключение — открытые наборы с готовой разметкой. Путь меток:
  1. учитель размечает — статус `auto`;
  2. человек принимает (`accepted`) или отклоняет (`rejected`).

  Модель, которая учится на своих же метках, закрепляет свои ошибки.

## Звенья

| # | Звено | Где | Что делает |
|---|---|---|---|
| 1 | Сбор | `guard/flywheel/` (в конвейере) | Сохраняет сырой кадр без оверлеев, детекции всех моделей этого кадра и причину отбора |
| 2 | Хранилище | `data/flywheel/` | `flywheel.db` (SQLite) и `frames/<camera_id>/<sample_id>.jpg` |
| 3 | Авторазметка учителем | `ml/flywheel/autolabel.py` | Тяжёлая модель предлагает рамки: YOLOv8x COCO и/или YOLO-World с текстовыми классами (`forklift`, `cigarette`, `hard hat` …) |
| 4 | Проверка человеком | UI «Данные» | Принять или отклонить сэмпл, поправить класс, удалить рамку |
| 5 | Версия датасета | `ml/flywheel/export.py`, API | YOLO-формат, разбиение по камере и времени, манифест |
| 6 | Дообучение и выкатка | `ml/flywheel/{train,gate,deploy}.py` | Цепочка шагов, описана под таблицей |

Шаги звена 6:
1. Дообучение (ultralytics, MPS).
2. Экспорт в ONNX: `ml/export_onnx.py`.
3. Компиляция под TPU: `tpu/compile.py`.
4. Сверка TPU с CPU.
5. Гейт на отложенных данных.
6. Обновление реестра моделей.

### Когда сохраняется сэмпл (`source`)

| source | Условие | Лимит |
|---|---|---|
| `event` | Каждое новое событие: сырой кадр подтверждения, детекции, `event_id` | Без лимита |
| `uncertain` | У целевого класса правила камеры уверенность в полосе `[conf_threshold − 0.10, conf_threshold + 0.05]` | 1 кадр в `GUARD_FLYWHEEL_PERIOD_S` (10 с) на камеру |
| `flicker` | Трек человека пропал и через ≤ 1 с рядом родился новый ID (рамки рядом: IoU > 0 или центры ≤ 0.75·max(h, w)) | Тот же |
| `disagreement` | На выборочном кадре детекции TPU сверены с CPU fp32 той же модели. Есть рамка без пары (IoU ≥ 0.5) с уверенностью выше порога. Работает только при бэкенде linq-tpu и доступном ONNX; включается `GUARD_FLYWHEEL_SHADOW=1` | 1 сверка в `GUARD_FLYWHEEL_SHADOW_PERIOD_S` (30 с) на камеру |
| `operator` | Оператор вручную отправил кадр камеры в выборку (кнопка в UI) | — |
| `import` | Импорт из открытого набора или ролика (CLI) | — |

Общий лимит — `GUARD_FLYWHEEL_DAILY_CAP` (500) сэмплов в сутки на камеру, кроме `event` и `operator`. Сбор
не должен задерживать оповещение:
- запись идёт в фоне через очередь; при переполнении сэмпл отбрасывается и увеличивается счётчик;
- в горячем пути кадра нет синхронного I/O.

Решение оператора по событию (`PATCH /api/events/{id}`, поле `status`) переносится в сэмпл `event` этого
события:

| status события | operator_label | Пояснение |
|---|---|---|
| `confirmed` | `positive` | — |
| `false` | `hard_negative` | Рамка нарушителя — ложное срабатывание |
| `forwarded` | `forwarded` | — |

### Edge-режим

На edge-узле нет журнала, поэтому сэмплы уходят на шлюз:
- сырой кадр (JPEG, ширина ≤ 960) и детекции — вместе с событием или отдельным сообщением `flywheel_sample`;
- для `uncertain` и `flicker` на edge период по умолчанию 30 с;
- прибавка к трафику со стенда — не больше 10 КБ/с на камеру. Сейчас весь трафик со стенда — 11 КБ/с, канал от
  стенда 0.1–0.4 МБ/с (`docs/EDGE.md`).

## Модель данных (`data/flywheel/flywheel.db`)

Таблица `samples`:

| поле | тип | смысл |
|---|---|---|
| id | TEXT PK | uuid4 hex |
| created_at | REAL | epoch, время кадра |
| camera_id | TEXT | — |
| source | TEXT | event, uncertain, flicker, disagreement, operator, import |
| event_id | TEXT NULL | для `event` |
| frame_path | TEXT | относительно `data/flywheel/` |
| width, height | INTEGER | размер сохранённого кадра |
| detections | TEXT (JSON) | `[{model_id, cls, conf, bbox: [x1, y1, x2, y2], track_id}]`, пиксели сохранённого кадра |
| labels | TEXT (JSON) | `[{cls, bbox, origin: teacher, operator, model или dataset, conf}]` — предложенная или итоговая разметка |
| status | TEXT | new → auto (учитель разметил) → accepted / rejected |
| operator_label | TEXT NULL | positive, hard_negative, forwarded |
| reason | TEXT | почему сохранён, например `no_helmet conf 0.31 ∈ [0.25, 0.40]` |
| holdout | INTEGER | 1 — кадр отложенного ролика, в обучение не экспортируется |
| reviewed_at | REAL NULL | — |
| dataset_version | TEXT NULL | последняя версия датасета, куда вошёл |

Таблица `datasets`:

| поле | смысл |
|---|---|
| version | PK, `fw-YYYYMMDD-HHMM` |
| created_at | — |
| path | — |
| classes | JSON |
| n_train, n_val | — |
| n_hard_negative | — |
| manifest | JSON: источники и смеси открытых наборов, sha256 списков файлов |

Доступ к БД — только через `guard/flywheel/store.py` (`FlywheelStore`). Этим модулем пользуются и API, и
офлайн-скрипты `ml/flywheel/*`.

## Карточки моделей (`ml/model_cards/<version>.json`, в git)

Любое дообучение — на открытых наборах или на данных маховика — заканчивается карточкой. По карточкам UI
показывает историю версий, а гейт принимает решение.

```json
{
  "model_id": "forklift",
  "version": "forklift-v1",
  "parent": "yolov8n-coco",
  "created_at": "2026-10-24T15:30:00+03:00",
  "task": "detect",
  "classes": ["person", "forklift"],
  "train": {"datasets": [{"name": "forklift-rf", "images": 1234, "url": "https://..."}],
            "epochs": 30, "imgsz": 480, "device": "mps", "minutes": 25},
  "weights": "models/weights/forklift_v1.pt",
  "onnx": "forklift_v1-480.onnx",
  "tpu_program": "tpu/forklift_v1_480/forklift_v1_480_b1_o5_h1m.tpu",
  "holdout": {
    "dataset_test": {"name": "forklift-rf test", "images": 120, "map50": 0.91, "per_class": {"forklift": 0.93}},
    "events": {"videos": ["s4_01", "s4_02", "s4_03"], "tp": 4, "fp": 0, "fn": 0, "duplicates": 0, "latency_p50_s": 2.1}
  },
  "baseline": {"version": "det (yolov8n COCO)", "holdout": {"events": {"tp": 2, "fp": 0, "fn": 2}}},
  "tpu_check": {"frames": 40, "recall": 0.95, "precision": 0.97},
  "gate": {"promote": true, "reasons": ["S4: 4/4 против 2/4", "FP 0 против 0"]},
  "status": "candidate"
}
```

Пути `onnx` и `tpu_program` даны относительно `GUARD_MODELS_DIR`, как в `configs/models.yaml`.

Поле `status`:

| Значение | Когда ставится |
|---|---|
| `candidate` | Модель обучена |
| `deployed` | Модель включена в `configs/models*.yaml` |
| `rejected` | Модель не прошла гейт |
| `archived` | Модель заменена новой |

Правило гейта: новая версия выкатывается, если на отложенных событиях одновременно:
- recall не ниже;
- precision не ниже;
- ложных тревог в час не больше;
- mAP50 на тесте набора ниже не больше чем на 0.01.

Причины решения записываются в `gate.reasons`.

## API (`guard/api/flywheel.py`, подключается в `guard/api/app.py`)

Все ответы — JSON. `Sample` — строка таблицы `samples`, JSON-поля раскрыты, плюс `frame_url`.

| Метод | Путь | Тело / параметры | Ответ |
|---|---|---|---|
| GET | `/api/flywheel/summary` | — | Сводка, поля ниже |
| GET | `/api/flywheel/samples` | `status, source, camera_id, cls, holdout, event_id, limit (50), offset, order (-created_at)` | `{items: [Sample], total}`; `event_id` — сэмплы события, для карточки события в UI |
| GET | `/api/flywheel/samples/{id}` | — | `Sample` |
| GET | `/api/flywheel/samples/{id}/frame.jpg` | — | сырой JPEG |
| PATCH | `/api/flywheel/samples/{id}` | `{status?: accepted\|rejected, labels?: [...]}` | `Sample`; ставит `reviewed_at`, у изменённых меток `origin = operator` |
| POST | `/api/flywheel/capture/{camera_id}` | — | `Sample` с `source = operator`: текущий кадр камеры |
| POST | `/api/flywheel/datasets` | `{include_holdout?: false, classes?: [...]}` | Описание версии; экспорт синхронный |
| GET | `/api/flywheel/datasets` | — | список версий |
| GET | `/api/flywheel/models` | — | карточки из `ml/model_cards/`, новые сверху |
| POST | `/api/flywheel/jobs` | `{kind: autolabel\|export\|train\|gate\|deploy, args: {...}}` | `{id, kind, status, log_url}` |
| GET | `/api/flywheel/jobs`, `/api/flywheel/jobs/{id}` | — | статус и хвост лога |
| GET | `/api/flywheel/jobs/{id}/log` | — | лог задачи текстом (`log_url`) |

Поля ответа `/api/flywheel/summary`:
- `total`;
- `today`;
- `by_source`, `by_status`, `by_class`;
- `reviewed`;
- `holdout`;
- `dropped` — счётчик отброшенных сэмплов;
- `last_dataset`;
- `models` — `{deployed: [...], candidate: [...]}`;
- `collecting` — идёт ли сбор.

Задачи (`jobs`) запускают CLI `ml/flywheel/*` подпроцессом, лог пишется в `data/flywheel/jobs/<id>.log`. Задачи `train` и
`deploy` разрешены, только если стоит `GUARD_FLYWHEEL_ALLOW_TRAIN=1`; иначе ответ 403 с объяснением. Обучение идёт
через `HEAVY_LANE=train tools/heavy.sh`.

WebSocket `/ws/events`: сообщение `{"kind": "flywheel_sample", "type": "flywheel_sample", "sample": {...короткая форма...}}`
при каждом новом сэмпле. `kind` — как у остальных сообщений шины, `type` — для совместимости с первой версией контракта.
UI обновляет по нему счётчики. Если UI его не обрабатывает, ничего не ломается.

## Настройки

| Переменная | По умолчанию | Смысл |
|---|---|---|
| `GUARD_FLYWHEEL` | 1 | Сбор включён |
| `GUARD_FLYWHEEL_DIR` | `<GUARD_DATA_DIR>/flywheel` | Хранилище |
| `GUARD_FLYWHEEL_PERIOD_S` | 10 (на edge 30) | Период `uncertain` и `flicker` на камеру |
| `GUARD_FLYWHEEL_DAILY_CAP` | 500 | Лимит в сутки на камеру |
| `GUARD_FLYWHEEL_SHADOW` | 0 | Сверка TPU с CPU на выборочных кадрах |
| `GUARD_FLYWHEEL_SHADOW_PERIOD_S` | 30 | Период сверки |
| `GUARD_FLYWHEEL_ALLOW_TRAIN` | 0 | Разрешить задачи `train` и `deploy` из UI |

## UI: страница «Данные» (`frontend/src/features/flywheel/`, маршрут `/data`)

Реализация: клиент API — `frontend/src/api/http.ts` (`flywheel`), запросы — `frontend/src/api/flywheelQueries.ts`, мок —
`frontend/src/api/mock/flywheel.ts`, проверка на моках — `frontend/scripts/flywheel.mjs` (см. `frontend/README.md`).

1. **Полоса цикла.** Шесть шагов: сбор → авторазметка → проверка → датасет → дообучение и сборка под LinQ → гейт
   и выкатка. На каждом шаге — число или статус последнего запуска.
2. **Счётчики:**
   - собрано сегодня и всего;
   - по источникам: событие, неуверенность, мигание трека, расхождение TPU и CPU, оператор;
   - проверено;
   - в последней версии датасета;
   - отброшено.
3. **Очередь проверки.** Карточки сэмплов с рамками поверх кадра, фильтры (источник, статус, камера, класс) и
   клавиши: `1` — принять, `2` — отклонить, `←`/`→` — соседний сэмпл. Это те же клавиши 1/2, что в карточке
   события. Сэмплы отложенных роликов помечены «отложенная выборка».
4. **Карточка сэмпла:**
   - большой кадр с рамками меток и детекций, их можно переключать;
   - список рамок: смена класса, удаление;
   - причина отбора;
   - ссылка на событие.
5. **Датасеты:** версии, состав, кнопка «Собрать версию».
6. **Модели:** версии из карточек — метрики на отложенных данных против базовой версии, решение гейта с причинами,
   статус.

   Кнопка «Дообучить» неактивна без `GUARD_FLYWHEEL_ALLOW_TRAIN` и показывает подсказку.

Для разработки без бэкенда нужны моки в `frontend/src/api/mock` по образцу существующих.

## Как устроено

Состояние бэкенда маховика на 24.10. Ниже — фактическое поведение кода; отступления от контракта отмечены
словом «отступление».

### Модули

| Что | Где |
|---|---|
| Хранилище: SQLite `flywheel.db` (WAL), кадры `frames/<camera_id>/<sample_id>.jpg`, версии датасетов | `guard/flywheel/store.py` (`FlywheelStore`) |
| Отбор кадров: полоса уверенности, мигание трека, сверка TPU и CPU | `guard/flywheel/select.py` |
| Сборщик в конвейере и фоновый писатель | `guard/flywheel/collector.py` (`FlywheelCollector`, `LocalWriter`) |
| Подключение к движкам (одна строка в `PipelineEngine` и `GatewayEngine`) | `guard/flywheel/wiring.py` |
| Отложенная выборка | `guard/flywheel/holdout.py` |
| Карточки моделей, задачи | `guard/flywheel/cards.py`, `guard/flywheel/jobs.py` |
| API | `guard/api/flywheel.py`, подключён в `guard/api/app.py` двумя строками |
| Экспорт версии датасета | `ml/flywheel/export.py` |

Хранилище по умолчанию — `<GUARD_DATA_DIR>/flywheel` (`data/flywheel`), меняется `GUARD_FLYWHEEL_DIR`.

### Горячий путь

- `PipelineEngine.handle_events`: после того как событие ушло в журнал, WS и транспорт, вызывается
  `flywheel.on_event(ctx, ev)`. Он только кладёт в очередь ссылки на сырой кадр `ctx.frame`, детекции всех
  моделей кадра и снимок треков. Снапшот события рисуется на копии кадра, поэтому `ctx.frame` остаётся без
  оверлеев.
- `CameraWorker.process`: в конце обработки кадра вызывается `flywheel.observe(worker, ctx)`, после
  `handle_events` и `frame_meta`. В нём обновляется жизненный цикл треков (мигание) и, раз в период, ищется
  неуверенная рамка.
- Очередь писателя — 16 задач. Если она полна, сэмпл отбрасывается и растёт `dropped.queue`.
- JPEG (≤ 1280 px, q90), файл, строка SQLite и сообщение WS делаются в потоке `flywheel-writer`.
- `PATCH /api/events/{id}` со `status` сразу пишет `operator_label` в сэмпл `event`. Если оператор успел
  раньше, чем фон записал сэмпл, писатель берёт статус события из журнала при вставке.

Замер на синтетике (1280×720, 9 детекций на кадр, 600 кадров; `data/media/checks/flywheel_bench/hotpath-20261024.json`):

| | без маховика | с маховиком |
|---|---|---|
| `CameraWorker.process`, p50 / p95 | 0.571 / 0.760 мс | 0.583 / 0.919 мс |
| `flywheel.on_event` (в `handle_events`, после отправки события), p50 | — | 4.7 мкс |
| `flywheel.observe` в кадре без отбора, p50 | — | 1.2 мкс |
| запись сэмпла в фоне, p50 | — | 4.7 мс |

Тест `tests/test_flywheel_latency.py`: при «зависшем» на 0.5 с диске `handle_events` возвращается сразу,
событие уже в журнале, сэмпл дописывается позже.

### Отбор

- **Полоса уверенности — отступление.** Декодер отбрасывает детекции ниже `conf_threshold` модели, поэтому
  полоса `[порог − 0.10, порог + 0.05]` считается от порога, который реально решает судьбу рамки:
  - `min_conf` правила, если он есть (`zone_intrusion` 0.5, `smoking_fire` 0.5);
  - для классов, которые ведёт трекер (человек и техника), — `tracker_high_thresh` (0.4);
  - иначе — `conf_threshold` модели (ppe 0.4, fire 0.4, телефон 0.25).

  Часть полосы ниже `conf_threshold` модели не видна. Пример: для ppe видна только часть [0.40, 0.45].
- Целевые классы правил — `RULE_TARGETS` в `select.py`. Для `ppe_missing` берутся `required` и `no_*`,
  для `vehicle_proximity` — `vehicle_classes`. Новый плагин может задать свои классы атрибутом
  `flywheel_classes`.
- Из рамок в полосе берётся ближайшая к порогу, причина — например `no_helmet conf 0.42 ∈ [0.30, 0.45], правило s2_ppe`.
- Период `uncertain` и `flicker` считается от попытки и отдельно для каждого источника: не больше
  1 сэмпла каждого вида за `GUARD_FLYWHEEL_PERIOD_S` на камеру.
- Суточный лимит `GUARD_FLYWHEEL_DAILY_CAP` — по местным суткам. Он общий для всех источников, кроме `event`
  и `operator`, а при старте счётчик читается из хранилища.
- **Holdout — расширение контракта.** Отложенными считаются ролики всех `ml/eval/scenarios*.yaml`, а не только
  `scenarios.yaml`: наборы `scenarios_<имя>.yaml` тоже оценочные. Совпадение ищется по имени ролика в пути
  источника: `videos/s1_01.mp4`, субпоток стенда `data_360/videos/s1_01.mp4`, каталоги кадров `images/s2`.
  **Все четыре демо-камеры `configs/cameras.yaml` — отложенные ролики**, их сэмплы имеют `holdout = 1` и без
  `--include-holdout` в датасет не попадают.

### API — уточнения

- `GET /api/flywheel/samples`: фильтры через запятую (`source=event,uncertain`). `cls` ищет класс и в метках,
  и в детекциях сэмпла. `holdout` — `0/1` или `true/false`. `order` — `-created_at`, `created_at`,
  `-reviewed_at` или `reviewed_at`. `limit` — до 500.
- `PATCH /api/flywheel/samples/{id}`: метка, совпавшая с прежней по классу и рамке, сохраняет `origin` и `conf`;
  изменённые и новые получают `origin = operator`.
- `GET /api/flywheel/summary`: поля контракта и ещё:
  - `by_class` — классы итоговых меток, `by_detected_class` — классы детекций;
  - `dropped` — число; `dropped_detail` — по причинам: `queue` (очередь полна), `cap` (суточный лимит);
  - `collector` — счётчики сборщика, `writer` — счётчики писателя;
  - `allow_train`, `mode`.
- `POST /api/flywheel/capture/{camera_id}` (local) берёт последний обработанный кадр камеры и его детекции.
- `POST /api/flywheel/datasets` → 201 с описанием версии. Если принятых сэмплов нет — 422.
- `POST /api/flywheel/jobs`: `args` превращаются во флаги CLI (`{"limit": 10, "include_holdout": true}` →
  `--limit 10 --include-holdout`). Хранилище CLI получает через `GUARD_FLYWHEEL_DIR`. Очереди tools/heavy.sh:
  autolabel, gate и deploy — `default`, train — `train`; export идёт без очереди. Если модуля
  `ml/flywheel/<kind>.py` нет, задача сразу получает `status = failed` и понятную ошибку. Описание задачи
  лежит в `jobs/<id>.json` рядом с логом и переживает перезапуск сервиса.

### Экспорт

```bash
python -m ml.flywheel.export [--include-holdout] [--classes person,forklift] [--val-ratio 0.2] \
    [--mix forklift-rf=data/datasets/forklift-rf] [--flywheel-dir data/flywheel]
```

- Берутся только `accepted`. Принятые без рамок — hard negatives (пустой файл меток).
- Разбиение: кадры камеры группируются окнами по 10 минут, окно целиком уходит в train или val по хэшу.
  Если окно одно, его последние 20 % кадров идут в val.
- Изображения — жёсткие ссылки на кадры хранилища. Результат — `data/flywheel/datasets/<версия>/`:
  `images/`, `labels/`, `data.yaml`, `manifest.json`. В манифесте — классы, состав, sha256 списков файлов,
  `holdout_contaminated` и открытые наборы смеси (`--mix`).

### Проверено

- Тесты: `tests/test_flywheel_{store,select,holdout,operator,latency,export,api}.py`.
- Живой прогон 24.10 14:01, `uvicorn guard.api.app:app --port 8010`, демо-камеры `configs/cameras.yaml`,
  onnx-cpu, 75 с. Журнал, медиа и хранилище маховика — отдельные, в `data/media/checks/flywheel_live_20261024-1401/`:
  - 12 событий → 12 сэмплов `event`, 17 `uncertain`, 2 `flicker`, 0 отброшено;
  - ручной захват `cam_s1` — 201;
  - кадры `frame_event.jpg` (падение) и `frame_uncertain.jpg` — сырые, без рамок и зон;
  - все сэмплы — `holdout = 1`: демо-камеры смотрят ролики оценки.

### Звено 6: дообучение, гейт, выкатка (ds-forklift)

Первый оборот сделан на открытых данных: погрузчик для S4 (`docs/DATASETS.md`, `docs/MODELS.md`). Те же CLI принимают
версию датасета маховика (`--dataset fw-…`, её собирает `ml/flywheel/export.py`).

| Шаг | Команда | Что пишет |
|---|---|---|
| Дообучение | `HEAVY_LANE=train tools/heavy.sh .venv/bin/python ml/flywheel/train.py --data <data.yaml> \| --dataset <fw-версия> [--mix <data.yaml>] --base <.pt или версия> --model-id <id> --version <версия>` | `models/weights/<версия>.pt`, карточка `ml/model_cards/<версия>.json` (`candidate`), прогон в `data/flywheel/runs/<версия>/` |
| Гейт | `ml/flywheel/gate.py --candidate <версия> --data <data.yaml> --events-baseline <json> --events-candidate <json> --write` | `holdout`, `baseline`, `gate` в карточке; при провале — `status: rejected` |
| Выкатка | `HEAVY_LANE=stand tools/heavy.sh .venv/bin/python ml/flywheel/deploy.py --version <версия> --data <data.yaml> [--steps export,calib,compile,check,registry] [--dry-run]` | ONNX, программа TPU на стенде, `tpu_check` и `status: deployed` в карточке, записи под якорем в `configs/models*.yaml` |

`train.py`:
- лимиты памяти проверяются до старта: batch ≤ 16 для n и ≤ 8 для s, workers ≤ 4, `cache=False`, `--time-min` ≤ 40;
- `--mix` добавляет train и val других наборов с теми же классами, test остаётся от основного;
- `--dry-run` проверяет входы и печатает параметры.

`gate.py`:
- mAP50 считает на CPU по ONNX тем же декодером, что сверка ONNX (`ml/yolo_decode.py`). Поэтому меряется именно
  та модель, которая уйдёт в конвейер;
- `--subset имя=список.txt` — mAP50 на части теста, `--reference имя=onnx:классы` — справочная модель
  в карточку, на решение не влияет;
- события — готовые отчёты `run_offline` (`--events-*`) или прогон здесь же (`--scenarios-*`). Лучше считать обе
  стороны здесь же: тогда база и кандидат посчитаны одним кодом. Коммит кода пишется в карточку (`events.code`),
  при разном коде гейт предупреждает. `--cache-dir` — общий кэш детекций: det, pose, ppe и fire считаются один раз;
- код выхода: 0 — promote, 1 — отказ, 2 — ошибка входа;
- кроме правила контракта проверяет, что дублей не больше.

`deploy.py` (шаги по порядку, каждый можно запустить отдельно, `--dry-run` только печатает команды и правки):
- `export` — `ml/export_onnx.py` через `tools/heavy.sh`: `models/<версия>-<imgsz>.onnx`;
- `calib` — `models/calib_<имя>/`: половина кадров — train набора с классом модели (иначе int8 срезает
  уверенность этого класса, `docs/LINQ_NOTES.md` §17), половина — `data/calib` без кадров роликов и без копий кадров
  теста;
- `compile` — на стенде: `tpu/compile.py`, затем `tpu/fix_thresholds.py` и вторая программа `<имя>s1` с потолком
  Sigmoid 1.0. Обе программы копируются в `/auto/cache/<логин>/` (`cp -n`, чужие файлы не трогаются);
- `check` — `tpu/yolo_check.py` (raw) для обеих программ: TPU против CPU fp32 на отложенных кадрах — все кадры теста
  с классом модели, случайные до `--check-n` и `--check-images` (кадры роликов оценки). Класс модели — при пороге
  реестра и при пороге рождения трека 0.45, все классы — с замером скорости. Сводка — `tpu_check` карточки,
  сырые ответы — `data/flywheel/runs/<версия>/`;
- `registry` — только после `gate.promote`: запись под якорем модели в `configs/models.yaml` (порог CPU прежний) и
  `configs/models.stand.yaml` (порог TPU — `--tpu-conf` или подсказка по сверке: самый высокий порог с recall ≥ 0.95
  и precision ≥ 0.9 относительно CPU). Комментарии неизменённых строк сохраняются. Карточка → `deployed`, прежняя
  `deployed` того же `model_id` → `archived`;
- камеры и edge не трогает: в конце печатает, что включить и что перезапустить оркестратору.
