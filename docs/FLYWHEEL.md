# Маховик данных: система сама собирает обучающую выборку

Контракт для реализации (24.10, оркестратор). Реализация дописывает сюда раздел «Как устроено» с фактическими
путями и командами. Решение внутреннее, для демонстрации: лицензии открытых наборов не проверяем, но источник
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
| GET | `/api/flywheel/samples` | `status, source, camera_id, cls, holdout, limit (50), offset, order (-created_at)` | `{items: [Sample], total}` |
| GET | `/api/flywheel/samples/{id}` | — | `Sample` |
| GET | `/api/flywheel/samples/{id}/frame.jpg` | — | сырой JPEG |
| PATCH | `/api/flywheel/samples/{id}` | `{status?: accepted\|rejected, labels?: [...]}` | `Sample`; ставит `reviewed_at`, у изменённых меток `origin = operator` |
| POST | `/api/flywheel/capture/{camera_id}` | — | `Sample` с `source = operator`: текущий кадр камеры |
| POST | `/api/flywheel/datasets` | `{include_holdout?: false, classes?: [...]}` | Описание версии; экспорт синхронный |
| GET | `/api/flywheel/datasets` | — | список версий |
| GET | `/api/flywheel/models` | — | карточки из `ml/model_cards/`, новые сверху |
| POST | `/api/flywheel/jobs` | `{kind: autolabel\|export\|train\|gate\|deploy, args: {...}}` | `{id, kind, status, log_url}` |
| GET | `/api/flywheel/jobs`, `/api/flywheel/jobs/{id}` | — | статус и хвост лога |

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

WebSocket `/ws/events`: сообщение `{"type": "flywheel_sample", "sample": {...короткая форма...}}` при каждом новом сэмпле.
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

## Как устроено: звено 6 — дообучение, гейт, выкатка (ds-forklift, 24.10)

Первый оборот сделан на открытых данных: погрузчик для S4 (`docs/DATASETS.md`, `docs/MODELS.md`). Те же CLI принимают
версию датасета маховика (`--dataset fw-…`), как только появится `ml/flywheel/export.py`.

| Шаг | Команда | Что пишет |
|---|---|---|
| Дообучение | `HEAVY_LANE=train tools/heavy.sh .venv/bin/python ml/flywheel/train.py --data <data.yaml> \| --dataset <fw-версия> [--mix <data.yaml>] --base <.pt или версия> --model-id <id> --version <версия>` | `models/weights/<версия>.pt`, карточка `ml/model_cards/<версия>.json` (`candidate`), прогон в `data/flywheel/runs/<версия>/` |
| Гейт | `ml/flywheel/gate.py --candidate <версия> --data <data.yaml> --events-baseline <json> --events-candidate <json> --write` | `holdout`, `baseline`, `gate` в карточке; при провале — `status: rejected` |
| Выкатка | `ml/flywheel/deploy.py --version <версия> [--dry-run]` — в работе, следующее слияние ds-forklift | ONNX, программа TPU, сверка, записи в `configs/models*.yaml` |

`train.py`:
- лимиты памяти проверяются до старта: batch ≤ 16 для n и ≤ 8 для s, workers ≤ 4, `cache=False`, `--time-min` ≤ 40;
- `--mix` добавляет train и val других наборов с теми же классами, test остаётся от основного;
- `--dry-run` проверяет входы и печатает параметры.

`gate.py`:
- mAP50 считает на CPU по ONNX тем же декодером, что сверка ONNX (`ml/yolo_decode.py`). Поэтому меряется именно
  та модель, которая уйдёт в конвейер;
- `--subset имя=список.txt` — mAP50 на части теста, `--reference имя=onnx:классы` — справочная модель
  в карточку, на решение не влияет;
- события — готовые отчёты `run_offline` (`--events-*`) или прогон здесь же (`--scenarios-*`);
- код выхода: 0 — promote, 1 — отказ, 2 — ошибка входа;
- кроме правила контракта проверяет, что дублей не больше.
