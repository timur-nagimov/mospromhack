# YOLO11x: эксперимент качества и подготовка к серверу

Это opt-in кандидат. Production TPU реестры не меняются. Работа — личный
эксперимент; ограничения хакатона не используются как требования к системе.
CPU-время служит диагностикой, выбор делается по качеству событий.

## Закреплённые артефакты

- Официальный источник: [Ultralytics/YOLO11](https://huggingface.co/Ultralytics/YOLO11),
  revision `5e0da476eb5def45e8080bd4b92ea63f0b16974c`, `yolo11x.pt`.
- Веса: 114636239 байт, SHA256
  `7bc158aa95c0ebfdd87f70f01653c1131b93e92522dbe15c228bcd742e773a24`.
- Проверенный ONNX: 228057790 байт, SHA256
  `6ed1d257152c284ec7674417fa5d6e29dcfeb68677d40a67257af3323c2a94ea`.
- Файлы вне git: `models/candidates/yolo11x-640/yolo11x-640.onnx` и `det.json`.
- Реестр: `configs/models.yolo11x-640.yaml`, canonical роль `det`, COCO-80.

Вход FP32 `[1,3,640,640]`, выход `[1,84,8400]`, opset 12, batch 1,
без встроенного NMS. Препроцессинг — RGB /255, letterbox с padding 114.
`output_format: yolov8` здесь описывает совместимый layout, не архитектуру.
ONNX checker и настоящий zero-input ORT inference успешны, `problems=[]`.

Экспорт выполнен существующим `ml/export_onnx.py` с двумя CPU потоками,
Ultralytics 8.3.221, onnxslim 0.1.98, onnx 1.23.2, ORT 1.31.0.
Рецепт после скачивания и проверки SHA256 официальных весов:

```bash
tools/heavy.sh .venv/bin/python ml/export_onnx.py \
  --weights "$PWD/models/weights/yolo11x.pt" --model-id det \
  --out yolo11x-640.onnx --imgsz 640 \
  --models-dir "$PWD/models/candidates/yolo11x-640" --threads 2 \
  --source https://huggingface.co/Ultralytics/YOLO11/resolve/5e0da476eb5def45e8080bd4b92ea63f0b16974c/yolo11x.pt \
  --license AGPL-3.0
```

## Одинаковый image benchmark

141 Construction-PPE test image, person GT 236, AP50 COCO 101 point,
conf 0.35 / IoU 0.5. Image split, sampling и порог не менялись между моделями.

| Модель | Person AP50 | Precision | Recall |
|---|---:|---:|---:|
| YOLOv8n-640 | 0.757 | 0.736 | 0.780 |
| YOLO11x-640 | 0.772 | 0.675 | 0.826 |

Рост AP/recall сопровождается снижением precision в фиксированной рабочей
точке. Это не основание автоматически менять production. Порог на test не
подбирается. Этот COCO детектор не поддерживает helmet/no_helmet/vest.

Raw predictions, GT, provenance и backend counters:
`data/yolo11_evaluation_20261024/images-{n,yolo11x}/<run>/report.json`.
Полный S1 n/YOLO11x выполнен отдельно, после объединения main, на 600
observations каждой модели. Исходная попытка S1 остановилась при разборе CLI:
1800 секунд превышали прежний cap 600. Inference в той попытке не начинался;
image-результаты сохранены. Новый cap 1800 оставляет default 180 и общий watchdog.

## Перенос на LinQ

Граф содержит 618 узлов, 4 MatMul и 3 Softmax. В этом экспорте head prefix —
`/model.23`, поэтому старый hardcoded `/model.22` preset неприменим.
Точки разреза: `/model.23/dfl/conv/Conv_output_0` и
`/model.23/Sigmoid_output_0`. Их нужно подтверждать графом и FP32 parity,
а не выбирать только по имени архитектуры.

Компиляция LinQ, per-class качество квантования и внешний inference пока не
проверены. Отдельный SDK-free preflight проверяет структурный разрез и равенство
FP32 full vs head→tail; он не доказывает поддержку операторов TPU compiler.
Для promotion нужны event A/B, независимые clean negatives и серверная приёмка.
CPU baseline здесь 640; текущий stand profile использует det 480, поэтому
результат также не является сравнением с квантованной production программой.

## Завершённый S1

Свежие 600+600 observations на коде `3b6560a`: обе модели дали 3 TP / 0 FP /
1 FN / 1 duplicate / 1 uncertain-excluded; strict precision/recall 0.75/0.75.
Состав source-time samples совпал полностью. Независимый CLI evaluator по
SQLite дал точное совпадение, readback содержит по 5 событий. Улучшения S1
на этом наборе не выявлено. Полное время worker в этом CPU окружении — диагностическое:
37.30 с у n против 624.87 с у x (точное поле — в JSON report).

Результат: `.orchestra/metrics/yolo11-quality-20261024.json`. Новые production
defaults не выбраны. Следующий приоритет — другие семейства, специализированные
события и независимый целевой набор, затем проверка квантования на сервере.

Повтор S1 кандидата с новым output directory:

```bash
tools/heavy.sh .venv/bin/python -m ml.eval.run_s1 \
  --config configs/eval-s1-yolo11x.json --data-dir data \
  --models-dir models/candidates/yolo11x-640 --timeout 1800 \
  --output-dir data/s1_evaluation/yolo11x-new-run
```
