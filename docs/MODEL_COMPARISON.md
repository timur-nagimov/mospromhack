# Сравнение YOLOv8n с YOLOv8s/YOLOv8x на фиксированном S1

`python -m ml.eval.compare_s1_models` запускает **два новых последовательных**
`ml.eval.run_s1`: сначала YOLOv8n-640, затем выбранный candidate: YOLOv8s-640
по умолчанию или YOLOv8x-640 с `--candidate-variant x`. Старые результаты
baseline не переиспользуются. На каждом варианте обрабатываются оба исходных
Intel-видео из `tools/data_manifest.json`: первые 60 секунд, 5 FPS, 300 + 300
семплов. Метод сопоставления событий остаётся в `ml.eval.events_eval`.

Это исследовательское сравнение на четырёх certain и трёх uncertain интервалах.
120 секунд — экспозиция **положительных видео**, включая uncertain интервалы;
независимых чистых видео с S1-полигонами здесь нет. Полученный FP/hour не является
оценкой частоты ложных тревог на обычной эксплуатации. Результат не переключает
основной реестр и не выбирает модель автоматически.

## Подготовленные локальные модели

Для воспроизводимости нужны два отдельных каталога с canonical sidecar `det.json`:

| Вариант | Реестр | Каталог относительно корня checkout с артефактами | ONNX SHA256 |
| --- | --- | --- | --- |
| baseline n | `configs/models.yaml`, logical ID `det` | `models` | `aa9699693737742033bc899670016413e1c3f9c2725b5fd0b302cfda44ce3a99` |
| candidate s | `configs/models.yolov8s-640.yaml`, logical ID `det` | `models/candidates/yolov8s-640` | `ddac89c188b95f774e40c0f76b40a48e3ea3752059d95f87acaccedc2009a0dd` |
| candidate x | `configs/models.yolov8x-640.yaml`, logical ID `det` | `models/candidates/yolov8x-640` | Фиксируется в metadata/report после локального экспорта; registry сам не содержит весов |

Исходные официальные pretrained weights:

- [Ultralytics YOLOv8n, assets v8.3.0](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt), SHA256 `f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36`.
- [Ultralytics YOLOv8s, assets v8.3.0](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s.pt), SHA256 `1f47a78bf100391c2a140b7ac73a1caae18c32779be7d310658112f7ac9aa78a`.
- [Ultralytics YOLOv8x, assets v8.3.0](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8x.pt); SHA256 actual weights и ONNX записывает локальный экспорт и проверяет wrapper.

Экспорт выполняется отдельно существующим `ml/export_onnx.py`; пример после
скачивания доверенных официальных весов в `models/weights/yolov8s.pt`:

```bash
python ml/export_onnx.py \
  --weights "$PWD/models/weights/yolov8s.pt" \
  --model-id det --out yolov8s-640.onnx --imgsz 640 --threads 2 \
  --models-dir "$PWD/models/candidates/yolov8s-640" \
  --source https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s.pt \
  --license AGPL-3.0
```

Для x после загрузки официального `yolov8x.pt` используется тот же exporter:

```bash
python ml/export_onnx.py \
  --weights "$PWD/models/weights/yolov8x.pt" \
  --model-id det --out yolov8x-640.onnx --imgsz 640 --threads 2 \
  --models-dir "$PWD/models/candidates/yolov8x-640" \
  --source https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8x.pt \
  --license AGPL-3.0
```

Не использовать preset `det_s640` для этого сравнения: он создаёт другой logical
ID и sidecar, тогда как runtime здесь загружает `det.json`. Подробности экспорта,
лицензии и закреплённых optional ML dependencies — в `docs/MODELS.md`.

До inference wrapper проверяет совпадение metadata с реальными ONNX bytes и
исходными weights: имя, SHA256, размер ONNX, роль `det`, 80 классов, официальный
source URL с соответствующим именем weights. Сам source URL является записанной
provenance; проверка локальных хешей подтверждает согласованность файлов и
metadata, но не служит независимой подписью издателя. Pickle/torch при этой
проверке не загружаются.

## Запуск из корня проекта

На общем checkout, где находятся подготовленные `data` и `models`:

```bash
python -m ml.eval.compare_s1_models \
  --data-dir "$PWD/data" \
  --baseline-models-dir "$PWD/models" \
  --candidate-models-dir "$PWD/models/candidates/yolov8s-640" \
  --timeout 420 --run-timeout 180 --json
```

Тяжёлое сравнение n→x выполняется отдельно, последовательно; дополнительный
локальный CPU wall сам по себе не является запретом выбора модели по качеству:

```bash
python -m ml.eval.compare_s1_models \
  --candidate-variant x \
  --data-dir "$PWD/data" \
  --baseline-models-dir "$PWD/models" \
  --candidate-models-dir "$PWD/models/candidates/yolov8x-640" \
  --timeout 1800 --run-timeout 1500 --json
```

Каждая команда создаёт новый baseline и candidate по 600 семплов, включая
новый baseline n для сравнения с x. Без подготовленного canonical `det.json`
и соответствующих x ONNX/weights preflight завершится ошибкой. Включение x в
CLI не является свидетельством выполненного heavy прогона или его качества.

В отдельном worktree передайте абсолютные пути к общим `data/models`; пути к
runner, recipe и registry берутся из исполняемого checkout. По умолчанию CLI
также учитывает `GUARD_DATA_DIR` и `GUARD_MODELS_DIR`. Candidate по умолчанию
лежит в `baseline-models-dir/candidates/yolov8s-640`; для `--candidate-variant x`
автоматический путь — `baseline-models-dir/candidates/yolov8x-640`.
Явный `--candidate-models-dir` переопределяет путь, но не проверяемый variant:
подмена s metadata/weights под x отклоняется. `candidate_variant` записывается
в report, а registry и source filename проверяются для выбранного варианта.

Без `--output-dir` создаётся новый `data/model_comparison/<UTC>-<uuid>`; можно
задать **новый** путь внутри ignored `data`, существующий каталог отклоняется.
Результаты, DB, traces и веса не добавляются в git. Wrapper работает локально:
сетевых загрузок, RTSP, TPU и компиляции нет. ORT telemetry отключает `run_s1`,
OpenCV/ORT используют два потока; варианты запускаются последовательно.

Общий watchdog охватывает preflight, оба дочерних run и сравнение; максимум
`--timeout` — 1800 секунд, по умолчанию прежние 420. Каждый run дополнительно
ограничен `--run-timeout` (максимум 1800, по умолчанию прежние 180 секунд).
Heavy пример явно задаёт общий лимит 1800 и per-run 1500 секунд; standalone
CLI `run_s1` не меняется, расширенный deadline передаёт wrapper своему child.
При timeout завершается вся группа
процессов; на TERM/KILL зарезервировано до двух секунд завершения. Wrapper
использует внутренний `run_s1 --_run`, чтобы ORT child оставался в его группе.
Очистка группы выполняется и после обычного выхода worker, в том числе после
локального per-run timeout: оставшиеся потомки получают TERM, затем KILL по
истечении grace-периода независимо от returncode leader. Завершение одного
leader не принимается за завершение всего дерева. В report сохраняется
`process_group_cleanup` с отправленными сигналами и результатом reaping leader.

`--json` печатает один JSON report. Exit 0 означает два полных сопоставимых
прогона, а не достижение порога качества; 1 — ошибка, timeout или нарушение
сопоставимости; 2 — неверные аргументы/уже существующий output. Если baseline
не проходит preflight или выполнение, candidate не запускается, а отчёт явно
остаётся failed; частичное сравнение не публикуется как успешное.

## Что проверяется и сохраняется

Фиксированы input 640×640, float32, conf 0.25, IoU 0.5, классы/их порядок,
preprocessing, CPU export и версии exporter, tracker, rules, tolerance,
аннотации и source bytes. Допустимое вычислительное отличие — ONNX/weights и
соответствующие пути registry. Текстовые license/source/provenance обоих
артефактов сохраняются отдельно. Строки license могут содержать разное
пояснение одной AGPL-3.0 лицензии и не меняют inference semantics.

Wrapper проверяет 2/2 видео, 600/600 кадров, 600 ORT calls/items, batch 1,
нулевое число ошибок, CPU provider и два ORT потока. Оба run должны иметь один
git snapshot и версии библиотек. Проверяются SHA256 общих runner/evaluator,
manifest/video inputs, GT, camera configs, mapping, durations и всех выходных
артефактов дочерних run. Сохраняются bytes/mtime/hash recipe, registry, metadata,
weights и wrapper; финальная проверка отклоняет изменившиеся модельные файлы.

Дополнительно перед обоими прогонами фиксируется content fingerprint всех
`guard/**/*.py` и `ml/**/*.py`: относительный путь → SHA256 bytes, включая
untracked модули. Он проверяется до и после каждого run и в финале. Изменение
содержимого файла при том же dirty-имени, добавление или удаление модуля делает
сравнение failed. Полный исходный список файлов и общий SHA сохраняются в
`provenance.executable_code`, стадии проверок и изменившиеся пути — в
`provenance.code_checks`; одного совпадения HEAD/списка dirty-файлов недостаточно.

Для каждой из 600 пар проверяется фиксированная сетка sample index → source
frame index → source timestamp, decoder timestamp и capture ID; canonical
fingerprint должен совпасть. Recipe должен совпасть с исходным фиксированным
`configs/eval-s1.json`, за исключением model registry. `effective_model.json`
должен совпасть с preflight и с другим вариантом, за исключением ONNX path.
Fingerprint относится к координатам/времени семплов, не к пикселям: отдельного
pixel hash текущий `run_s1` не пишет; проверяются одинаковые source bytes,
индексы кадров и версии decoder.

В каталоге результата:

- `baseline/`, `candidate/`: отдельные полные reports, DB, исходные predictions,
  JSONL samples/lifecycle, GT и effective configs от `run_s1`.
- `config_baseline.json`, `config_candidate.json`: точные переданные recipes.
- `paired_s1_01.jsonl`, `paired_s1_02.jsonl`: все 300 пар каждого видео;
  detection/class counts, сырые изменения, run-local треки/boxes, нарушения и
  созданные event IDs. Полные raw detections остаются в исходных trace files.
- `report.json`: отдельные per-video/global evaluator metrics и strict metrics,
  denominators, duplicates/uncertain IDs и reasons, полные timing distributions,
  paired fingerprints, hashes и ограниченные примеры около границ аннотаций.

Original precision исключает duplicates и uncertain predictions. Дополнительно
без изменения evaluator приводятся strict precision = TP/(TP+FP+duplicates),
raw counts и (FP+duplicates)/positive-exposure-hour. FN, duplicates и uncertain
не исчезают из отчёта. Latency событий остаётся `confirmed − label start` в
source time; wall performance считается отдельно.

Timing p50/p95/mean/max строятся из **всех 600** raw samples для каждого варианта,
а не из конечного rolling ORT окна в 300 измерений. Для стадий приводится
собственный `n`; отсутствующие tracker/rules/journal/decode measurements не
подменяются нулями. Video-loop wall/600 — амортизированная сквозная стоимость,
включая decode, tracking, rules, journal и trace IO; это не покадровая latency
распределения всего processor. RAM текущий runner не измеряет. Фиксированный
порядок n→candidate (s/x), отсутствие отдельного warmup и системная нагрузка ограничивают
строгость сравнения производительности.

Треки имеют локальные IDs каждого run. Разница counts, boxes или rule outputs
сама по себе не доказывает, что одна модель правильно нашла/потеряла человека:
actor-level GT отсутствует. Boundary examples — первые 12 изменившихся кадров
в пределах двух секунд от границ аннотаций, без нового matching. Фиксированные
landmarks sample indices `0,50,100,150,200,250,299` для каждого видео выбраны
независимо от outputs и могут использоваться для следующей ручной разметки.

Даже улучшение всех четырёх certain событий остаётся exploratory результатом.
Решение о production promotion требует независимой actor-aware разметки,
чистых S1 видео с зонами и отдельного realtime/multicamera performance прогона.
Порогами под эти семь интервалов runner не управляет. Выполнение реального A/B
не входит в unit tests; фактические метрики нужно читать из нового report.
