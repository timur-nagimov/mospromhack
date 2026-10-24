# Экспериментальный CPU PPE-кандидат

`configs/models.ppe-candidate.yaml` добавляет логическую роль `ppe` с публичным
BuildVision v0.3 ONNX и сохраняет текущий COCO `det` отдельной ролью. Основной
`configs/models.yaml` и runtime-алгоритмы не изменены. Это opt-in кандидат,
который прошёл локальную рамочную оценку на 141 публичном изображении и не
заменяет рабочую модель. Событийное качество на целевых видео не подтверждено.

## Происхождение и фактический интерфейс

Первичный источник — [репозиторий автора](https://huggingface.co/constructelligence/construction-site-safety-hazards/tree/79673cff5a8bb249fc8ec9cca3daa19e2ef95191).
Ревизия закреплена: `79673cff5a8bb249fc8ec9cca3daa19e2ef95191`.
Лицензия в Hub metadata, карточке и ONNX: AGPL-3.0.

| Файл | Байт | SHA-256 |
|---|---:|---|
| `buildvision-hazards-v0.3.onnx` | 44 766 819 | `55f4dd2f53158eac4ff093d938b9fbda80c99454a03672145bbc36f539988249` |
| `config.json` | 560 | `1c5ca99d2b5f6257668d13ab53761a82d4a4d8872488b4fadede63a57a058224` |
| `README.md` | 13 065 | `7091675890c21fa3be69bafa9526b606fb4edd570e1167d5b0d50a68ee4ea10a` |

Проверено локально 24 октября 2026: `onnx.checker.check_model(full_check=True)`
прошёл; вход — float32 RGB `[1,3,640,640]`, выход — float32 `[1,18,8400]`.
Ровно 14 классов в ONNX metadata и `config.json` совпадают по порядку.
Закреплённая карточка также указывает 14 классов. Однократный ORT CPU smoke
на нулевом тензоре с двумя потоками дал ожидаемую форму и конечные числа.
Это проверка численного интерфейса; она не измеряет качество распознавания.

Загрузчик скачивает только ONNX и текстовые метаданные. `.pt`, pickle, Hub
скрипты и `trust_remote_code` не используются. Файлы публикуются после проверки
размера и SHA-256; ONNX с внешними весами, custom operators или локальными
функциями отклоняется. Исходная карточка, API metadata, `config.json`, результат
проверки и канонический export sidecar `ppe.json` сохраняются рядом с ONNX в
`models/candidates/buildvision-hazards-v0.3/`.

## Семантика классов

| Исходный класс | Runtime-класс | Атрибут |
|---|---|---|
| `hardhat` | `helmet` | `helmet=True` |
| `no-hardhat` | `no_helmet` | `helmet=False` |
| `safety vest` | `vest` | `vest=True` |
| `no-safety vest` | `no_vest` | `vest=False` |
| `person` | `person` | рамка человека |
| `no-mask` | `no-mask` | не используется для атрибута respirator |

Остальные восемь классов сохраняют исходные имена. Наличие маски и пригодного
респиратора не эквивалентно, поэтому кандидат покрывает только `helmet` и `vest`.
При отсутствии наблюдения штатный `assign_attributes` сохраняет `None`;
правило S2 реагирует только на явное `False`. Отсутствие детекции каски или
жилета само по себе не является отрицательным наблюдением. Список классов
в реестре и sidecar обязан совпадать; runner проверяет это до инференса.

## Данные и закреплённый протокол

Первичный источник — [Ultralytics Construction-PPE](https://docs.ultralytics.com/datasets/detect/construction-ppe/),
[закреплённый официальный YAML](https://github.com/ultralytics/ultralytics/blob/42044baf5eda3d52aa4bd7d8a098375d2c7fdad7/ultralytics/cfg/datasets/construction-ppe.yaml)
и указанный в нём [официальный assets ZIP](https://github.com/ultralytics/assets/releases/download/v0.0.0/construction-ppe.zip).
Источник указывает AGPL-3.0. Размер проверенного архива — **178 415 813** байт,
SHA-256 — `bef8dcb599aa4e9d9f5e602cb6fa7143d3c84d7f6a0ff40463d7f2a4c2632ccc`.
Загрузчик проверяет весь список ZIP на traversal, symlink, дубликаты, шифрование,
размер и коэффициент сжатия и извлекает только 141 test image/label пару.
`train` и `val` не извлекаются и не используются для настройки порогов.

В test разметке: `person=236`, `helmet=192`, `vest=178`, `no_helmet=40`.
`no_vest` отсутствует в наборе. Совокупный SHA-256 YAML и всех test image/label
байтов: `800a03725b2c6fc37b8c80794735a03666fd424cc63b4eeecb54b548e37b6791`.
Runner проверяет этот отпечаток перед запуском.

Пересечение с обучающими данными автора **не проверено**. Набор имеет официальный
test split, но независимость от обучения модели этим не доказана. Заявления и
метрики автора не заменяют локальную оценку.

Для кандидата и COCO baseline одинаковы: все 141 test images, production
`Detector` и letterbox/NMS, float32 640 px, CPU ORT два потока, AP confidence
0.001, NMS IoU 0.7, рабочая confidence 0.35. Production-декодер выбирает один
класс на anchor и не ограничивает число результатов `max_det=300`, поэтому
это отдельный воспроизводимый протокол, а не побитовое воспроизведение
авторского Ultralytics benchmark.

Метрики считаются общими `ml.eval_ppe.evaluate_predictions`, включая COCO
101-point AP50, P/R в рабочей точке и дополнительный IoU 0.3 для `no_helmet`.
Core знаменатель фиксирован по классам с GT: `person`, `helmet`, `vest`.
Неподдерживаемый моделью класс с GT даёт core-вклад 0; класс без GT остаётся
`null` и имеет явный статус coverage. Поэтому COCO person-only модель не
получает преимущество из-за меньшего знаменателя. Подробности —
[PPE_EVALUATION.md](PPE_EVALUATION.md).

## Воспроизведение

Команды выполняются из корня checkout с установленными зависимостями проекта.
При использовании worktree укажите общий каталог моделей/данных явно.

```bash
python -m tools.prepare_ppe_candidate --models-dir models \
  --data-dir data/model_research/construction-ppe-test --dataset --numerical-smoke

# Запускать последовательно: без параллельной тяжёлой CPU-нагрузки.
python -m ml.eval.run_ppe_onnx \
  --data data/model_research/construction-ppe-test/test-only \
  --models-dir models --models-config configs/models.yaml --model-id det \
  --output-root data/ppe_evaluation/baseline

python -m ml.eval.run_ppe_onnx \
  --data data/model_research/construction-ppe-test/test-only \
  --models-dir models --models-config configs/models.ppe-candidate.yaml --model-id ppe \
  --output-root data/ppe_evaluation/candidate
```

Каждый запуск создаёт отдельный каталог timestamp+UUID. `report.json` содержит
статус, модель/конфиг/sidecar SHA-256, data fingerprint, версии, метрики coverage
и latency; `inputs.json` — SHA каждого входа; `predictions.jsonl` — все
предсказания и ошибки по каждому изображению. Любая ошибка изображения делает
оценку `failed`, возвращает ненулевой exit code и оставляет `metrics=null`.
Ошибочный инференс не превращается в успешный пустой результат.

`semantic-smoke.json` для кандидата использует **его реальные предсказания** в
рабочей точке: рамки `person` → новые track-объекты для каждого отдельного
изображения → production `assign_attributes` → production `PpeMissingRule`.
Записываются все атрибуты и мгновенные срабатывания S2; respirator обязан
оставаться unknown. В still images не предполагается временная непрерывность,
а число срабатываний не является метрикой качества или подтверждённых событий.
Для COCO baseline этот smoke помечается `not_applicable`.

## Состояние проверки и границы

Проверены downloader/checker, численный однокадровый ORT интерфейс, полный
состав и хеши test split, а также узкие unit-тесты. После объединения три
модели (COCO n, COCO s и PPE) прошли по 141/141 изображению без ошибок.
Повторный readback всех raw predictions и GT точно воспроизвёл метрики.
Результаты и разбор ошибок — [MODEL_RESEARCH_20261024.md](MODEL_RESEARCH_20261024.md).
PPE AP50: person 0.791, helmet 0.895, vest 0.847; `no_helmet` 0.136,
P/R 0.25/0.20 относительно исходного GT при confidence 0.35. Визуальный
разбор выявил неоднородную геометрию и возможные пробелы GT: эти показатели
не следует выдавать за измеренное качество тревог на производстве.
Синтетические unit-тесты доказывают mapping отрицательных классов и сохранение
unknown; они не являются доказательством качества модели.

До включения по умолчанию нужны надёжная целевая разметка, уточнение границ
отрицательных рамок и отдельная размеченная S2 event-оценка на
целевых видео (включая отрицательные/unknown случаи и временные пороги).
Качество `no_vest` на Construction-PPE не измеряется из-за отсутствия GT.
Респиратор, TPU и обучение не проверялись. Кандидат остаётся экспериментальным.
