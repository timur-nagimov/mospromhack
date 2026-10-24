# Экспериментальный Vyra PPE-кандидат

Второй профиль `vyra` добавлен в безопасный загрузчик и существующий production
Detector runner. Профиль `buildvision` остаётся по умолчанию: прежние команды,
каталог модели, результаты и формат метрик сохраняются. Отдельный opt-in реестр
`configs/models.ppe-vyra.yaml` задаёт роли `det` и `ppe`; основной
`configs/models.yaml` не меняется.

## Проверенное происхождение

Первичный источник — [Hexmon/vyra-yolo-ppe-detection, закреплённая ревизия](https://huggingface.co/Hexmon/vyra-yolo-ppe-detection/tree/08895b33d95d2587423ebe4f7c1b9c41beebd642).
Ревизия: `08895b33d95d2587423ebe4f7c1b9c41beebd642`.
Карточка автора называет модель YOLOv8m и связывает обучение с PPE Combined
Model v4; исходный `args.yaml` также содержит `model: yolov8m.pt`.

| Файл | Байт | SHA-256 |
|---|---:|---|
| `best.onnx` | 103 655 282 | `b99fed37bae1d111ddb622a0ef9077d42918a4daebd4a0bffbd8faaba273f246` |
| `README.md` | 1 578 | `ec237e6af27e6a279f6e81afc7d387e6364e6c06739fe52143e2609662263458` |
| `args.yaml` | 1 645 | `3117297c8c15c0e0b49dc9e9fb0c29113956ae8cedae917d8ee9b2094d9c5f82` |

Файлы проверены локально 24 октября 2026 года. Скачиваются только закреплённый
ONNX, текстовые README/args и Hub API metadata. `.pt`, pickle, сторонние скрипты
и remote code не используются. `args.yaml` сохраняется как текст, его команды
и параметры не выполняются. Загрузчик проверяет размер, SHA-256, commit, repo ID
и лицензию карточки, затем выполняет ONNX checker.

**Расхождение лицензий сохранено явно:** Hub API и front matter карточки
указывают `cc-by-4.0`; встроенная ONNX metadata содержит
`AGPL-3.0 License (https://ultralytics.com/license)`. В `source.json` и
`ppe.json` записаны оба значения, `license=null`,
`license_provenance.status=conflicting_metadata` и `review_required=true`.
Подготовка для экспериментальной оценки не разрешает это противоречие и не
является основанием для включения модели по умолчанию или её распространения.

## Фактический ONNX-контракт

`onnx.checker.check_model(full_check=True)` прошёл. Вход `images` — **FLOAT**
`[1,3,640,640]`; выход `output0` — **FLOAT** `[1,18,8400]`; opset 20.
Модель имеет один raw YOLOv8 detection head: 4 координаты и 14 class scores,
без встроенного NMS. Custom operator domains, внешних tensor data и локальных
ONNX functions не найдено. Изменять production decoder не потребовалось.

Export metadata содержит `half=True`, но фактические input/output tensor types
и FLOAT initializers проверены как float32; профиль определяется реальными
тензорами. Численный ORT smoke и инференс в этой фиче **не запускались**.
Checker не доказывает качество модели или успешный запуск ORT.

Порядок 14 классов ONNX совпал с таблицей карточки:

| ID | Исходный класс | Runtime-класс |
|---:|---|---|
| 0 | `Fall-Detected` | `Fall-Detected` |
| 1 | `Gloves` | `Gloves` |
| 2 | `Goggles` | `Goggles` |
| 3 | `Hardhat` | `helmet` |
| 4 | `Ladder` | `Ladder` |
| 5 | `Mask` | `Mask` |
| 6 | `NO-Gloves` | `NO-Gloves` |
| 7 | `NO-Goggles` | `NO-Goggles` |
| 8 | `NO-Hardhat` | `no_helmet` |
| 9 | `NO-Mask` | `NO-Mask` |
| 10 | `NO-Safety Vest` | `no_vest` |
| 11 | `Person` | `person` |
| 12 | `Safety Cone` | `Safety Cone` |
| 13 | `Safety Vest` | `vest` |

`Mask`/`NO-Mask` не преобразуются в respirator: их семантика этого не доказывает.
`respirator` остаётся unknown. `Fall-Detected` также не подключается автоматически
к правилу падения. Наблюдения `no_helmet`/`no_vest` проходят штатный
`assign_attributes`; отсутствие наблюдения не становится `False`.
Runner сверяет SHA, profile, revision, class order и sidecar с реестром до
создания ORT-сессии, чтобы два разных 14-class head не могли смешаться.

## Подготовка и последующий запуск

Из корня checkout с установленными зависимостями проекта:

```bash
# Только загрузка и checker; без ORT/inference.
.venv/bin/python -m tools.prepare_ppe_candidate \
  --candidate-profile vyra --models-dir models

# Выполнить отдельно в согласованной CPU-очереди после объединения фичи.
.venv/bin/python -m ml.eval.run_ppe_onnx \
  --candidate-profile vyra --model-id ppe \
  --models-config configs/models.ppe-vyra.yaml --models-dir models \
  --data data/model_research/construction-ppe-test/test-only \
  --output-root data/ppe_evaluation/ppe-vyra
```

Без `--models-config` новый профиль выбирает `configs/models.ppe-vyra.yaml`;
без `--candidate-profile` сохраняется прежний BuildVision. При работе из
worktree укажите абсолютные пути к общим models/data.

Модель, оригинальные текстовые источники, `hub-metadata.json`, `source.json`
и канонический sidecar `ppe.json` находятся отдельно в
`models/candidates/vyra-yolo-ppe-detection/`. Существующие BuildVision assets
не переписываются при подготовке Vyra.

Протокол оценки не меняется: все 141 test image, data fingerprint
`800a03725b2c6fc37b8c80794735a03666fd424cc63b4eeecb54b548e37b6791`,
CPU ORT два потока, production Detector/letterbox/NMS, confidence для AP 0.001,
NMS IoU 0.7, рабочий confidence 0.35, COCO 101-point AP50 и дополнительный
IoU 0.3 для `no_helmet`. Метрики считаются общим `ml.eval_ppe`, с фиксированным
core GT-знаменателем и явным coverage. Ошибка любого изображения даёт
`failed`, `metrics=null` и ненулевой exit code.

Каждый запуск создаёт отдельный timestamp+UUID output с `report.json`,
`inputs.json`, всеми `predictions.jsonl` и semantic S2 smoke по реальным
предсказаниям. Последний проверяет mapping/association/rule; мгновенные
срабатывания не доказывают качество событий или временную устойчивость.

## Ограничения

Training overlap с официальными 141 test images **неизвестен**. `no_vest` не
имеет GT в Construction-PPE, а геометрия `no_helmet` в исходном наборе
неоднородна. Оценка по исходной разметке сохраняется; GT и пороги не меняются.
Численные свойства, скорость и качество Vyra должны быть измерены отдельным
полным запуском. Ещё нужны надёжные целевые S2 видео с person-level
present/absent/unknown и временными интервалами. До этого профиль остаётся
экспериментальным; поддержки респиратора и TPU не заявляется.
