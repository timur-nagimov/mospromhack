# Оценка разрешения 640 / 1280

Generic COCO `det` режим `ml.eval.run_ppe_onnx` теперь принимает квадратный
вход 640 или 1280 из `ModelSpec`. Новый opt-in реестр
`configs/models.yolov8x-1280.yaml` указывает на отдельный экспорт YOLOv8x 1280.
Основной реестр, GT, метрики и PPE-профили не изменяются. BuildVision и Vyra
по-прежнему требуют 640 и свои закреплённые SHA/revision/class order.

Это **отдельный exploratory эксперимент разрешения**. Сравнение S1 при
фиксированном разрешении, где менялись только веса, имеет другой дизайн.
Результат 1280 нельзя объявлять победителем на независимой held-out выборке:
одни и те же 141 test images уже используются для нескольких исследовательских
сравнений, training overlap неизвестен. Пороги и исходная разметка сохраняются.

## Контракт и effective recipe

| `ModelSpec.input_size` | Статический вход float32 | Сырой выход float32 | Anchors |
|---|---|---|---:|
| `[640,640]` | `[1,3,640,640]` | `[1,4+nc,8400]` | 8 400 |
| `[1280,1280]` | `[1,3,1280,1280]` | `[1,4+nc,33600]` | 33 600 |

Для стандартных YOLOv8 detection grids со stride 8/16/32 это соответственно
6400+1600+400 и 25600+6400+1600 позиций. Прямоугольный вход, 960 или другой
размер не принимаются. Встроенный NMS, внешние tensor data, custom operators
и локальные ONNX functions не поддерживаются.

Runner создаёт **новую глубокую копию** базового `RECIPE`, не меняя глобальное
значение. В `report.json` записываются:

- фактический `recipe.input_size`, `grid_count` и `strides`;
- `resolution_experiment` и явный `overrides_from_640` с reference/effective
  размером и числом anchors; для 640 overrides пуст;
- `registry_input_size`, источник имён классов и фактическая ONNX inspection;
- SHA модели, реестра, sidecar, runner/metrics, входных файлов и data fingerprint.

ONNX checker проверяет статические формы входа/выхода, batch=1, число каналов
и float32 dtype. Если metadata `imgsz`/`names` присутствует, она обязана
совпадать с размером/классами. При отсутствии `imgsz` используются явный реестр
и статический граф, и этот fallback отражается в отчёте. Пустые registry classes
заполняются из ONNX metadata через production `backend.class_names`, как в
PipelineEngine; затем исходный порядок проверяется по ONNX `names`, если поле
присутствует. Отсутствие классов и в реестре, и в metadata — ошибка.

Каждый реальный raw output проверяется на конечные числа и точную форму с
8 400 или 33 600 anchors. Ошибка любого изображения сохраняется в trace и
делает весь запуск `failed`, `metrics=null`, exit code ненулевой. Ошибочные
результаты не считаются успешным пустым предсказанием. Латентность по-прежнему
включает Detector preprocess+infer+postprocess.

## Неизменные условия

Оба разрешения используют все 141 исходное test image, одинаковую разметку и
data fingerprint
`800a03725b2c6fc37b8c80794735a03666fd424cc63b4eeecb54b548e37b6791`.
Production Detector берёт input size из реестра и выполняет прежний letterbox,
single-label argmax и class-aware NMS. AP confidence **0.001**, NMS IoU **0.7**,
рабочий confidence **0.35**, CPU ORT **два потока** остаются одинаковыми.
Registry thresholds 0.25/0.5 совпадают с x640 control; evaluation меняет их
только в локальной копии spec, как и ранее.

COCO 101-point AP, matching и coverage остаются в общем `ml.eval_ppe`.
Core GT-знаменатель одинаков: person/helmet/vest. COCO-модель покрывает person;
отсутствующие PPE-классы не исчезают из знаменателя. Увеличение разрешения
не добавляет модели классы `no_helmet`, `no_vest` или поддержку респиратора.
Semantic S2 smoke для generic COCO остаётся `not_applicable`.

## Запуск после отдельного экспорта

Новый реестр ожидает:

```text
models/candidates/yolov8x-1280/yolov8x-1280.onnx
models/candidates/yolov8x-1280/det.json
```

Root подготавливает этот экспорт отдельно из тех же официальных
`models/weights/yolov8x.pt`, что использованы для x640 control:
SHA-256 `3df4ada6b4dad6d657868f2fdf7faecfb34dcfccf3a25c4b82079064718524c8`.
При сравнении необходимо подтвердить одинаковый source-weights SHA в sidecar
и записать разные ONNX SHA, поскольку input grid у экспортов различается.

Из корня общего checkout, после завершения текущей тяжёлой задачи:

```bash
.venv/bin/python -m ml.eval.run_ppe_onnx \
  --model-id det --models-config configs/models.yolov8x-1280.yaml \
  --models-dir models \
  --data data/model_research/construction-ppe-test/test-only \
  --output-root data/ppe_evaluation/coco-x-1280
```

Старые CLI и выбор BuildVision по умолчанию сохраняются. Каждый запуск создаёт
отдельный timestamp+UUID output; предыдущие результаты не перезаписываются.
Запуски 640/1280 должны выполняться последовательно с одинаковым CPU-режимом.

## Состояние и трактовка

В этой фиче проверены чистое построение рецепта, оба synthetic raw grid,
ONNX shape/dtype/metadata guards, metadata fallback и запрет PPE1280.
ORT-сессии, настоящий инференс и экспорт здесь **не выполнялись**.
Локальные численные графы в unit-тестах доказывают контракт, а не качество.

После реального запуска следует сравнить person AP/P/R, полный fixed-denominator
coverage и ошибки на одинаковых входах. Изменение разрешения является отдельным
фактором; оценка повторно используемого test split остаётся исследовательской.
Для решения о включении по умолчанию нужны отдельные надёжные целевые данные
и подтверждение на зафиксированном наборе, который не использовался для выбора
модели или разрешения. Заявлять заранее прирост качества или качество S2 нельзя.
