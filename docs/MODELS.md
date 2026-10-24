# Модели (A2 ml)

Реестр — `configs/models.yaml` (ключ `models` — записи `ModelSpec`, ключ `meta` — источник, лицензия, маппинг классов).
Файлы лежат в `models/` (`$GUARD_MODELS_DIR`, вне git): `<файл>.onnx` и `<model_id>.json` (классы, вход/выход, операторы, sha256, источник).

Общее для всех ONNX:
- opset 12, batch = 1, статический shape, упрощено onnxslim, NMS внутри нет;
- вход `images` NCHW float32, RGB 0..1, letterbox с паддингом 114 по центру;
- выход detect `(1, 4 + nc, N)`: cx, cy, w, h в пикселях входа и sigmoid-скоры классов; выход pose `(1, 56, N)` — плюс 17 точек COCO (x, y, conf);
- N = 8400 (640), 4725 (480), 3549 (416).

## Таблица моделей

CPU p50 — onnxruntime 1.31 CPU, Apple M3 Pro, потоки ORT по умолчанию, медиана p50 по 11 кадрам (каждый — 20 прогонов, только `session.run`; препроцесс около 2 мс, декод и NMS меньше 1 мс).

| model_id | файл | архитектура | классы | размер | CPU p50 | источник | лицензия |
|---|---|---|---|---|---|---|---|
| `det` | yolov8n-640.onnx | YOLOv8n | COCO-80 | 12.9 МБ | 24.0 мс | Ultralytics assets | AGPL-3.0 |
| `det_s640` | yolov8s-640.onnx | YOLOv8s | COCO-80 | 44.9 МБ | 67.4 мс | Ultralytics assets | AGPL-3.0 |
| `det_n480` | yolov8n-480.onnx | YOLOv8n | COCO-80 | 12.8 МБ | 14.9 мс | Ultralytics assets | AGPL-3.0 |
| `det_n416` | yolov8n-416.onnx | YOLOv8n | COCO-80 | 12.8 МБ | 11.2 мс | Ultralytics assets | AGPL-3.0 |
| `pose` | yolov8n-pose-640.onnx | YOLOv8n-pose | person + 17 точек | 13.5 МБ | 25.5 мс | Ultralytics assets | AGPL-3.0 |
| `pose_n480` | yolov8n-pose-480.onnx | YOLOv8n-pose | person + 17 точек | 13.4 МБ | 15.8 мс | Ultralytics assets | AGPL-3.0 |
| `ppe` | ppe-640.onnx | YOLOv8s | 14 (см. ниже) | 44.8 МБ | 66.8 мс | [constructelligence/construction-site-safety-hazards](https://huggingface.co/constructelligence/construction-site-safety-hazards) v0.3 | AGPL-3.0 (веса), CC BY 4.0 (данные) |
| `fire` | fire-640.onnx | YOLOv8n | smoke, fire | 12.3 МБ | 22.7 мс | [rabahdev/fire-smoke-yolov8n](https://huggingface.co/rabahdev/fire-smoke-yolov8n), D-Fire | AGPL-3.0 (веса) |

Телефон (S6) — класс COCO `cell phone` в `det`. Техника (S4) — COCO `car/truck/bus` в `det`, плюс `excavator/wheel_loader/dump_truck` в `ppe`. Погрузчика в COCO нет: `det` называет его `truck` или `bus`.

## СИЗ: выбор модели

Кандидаты: все YOLOv8-детекторы СИЗ с HF, у которых скачиваются веса. Оценка — `ml/eval_ppe.py` на **Ultralytics Construction-PPE** (val + test, 284 кадра). На этих кадрах, судя по карточкам, не обучался ни один кандидат. Метрика — AP50 (COCO, 101 точка) по нормализованным классам. Для `no_helmet` AP считается и при IoU 0.3: в этом наборе рамка `no_helmet` нарисована по лицу, а модели рисуют голову. Прогон на CPU.

| модель | арх. | person | helmet | vest | no_helmet IoU 0.5 / 0.3 | мс/кадр (torch CPU) |
|---|---|---|---|---|---|---|
| **constructelligence v0.3 (выбрана)** | v8s | **0.824** | **0.826** | **0.835** | 0.135 / 0.353 | 100 |
| killuminati1/construction-ppe-yolov8 | v8s | 0.821 | 0.651 | 0.708 | 0.054 / 0.141 | 71 |
| ayushgupta7777/safetyvision-yolov8 v2 | v8s | 0.341 | 0.751 | 0.800 | 0.071 / 0.354 | 102 |
| Hansung-Cho/yolov8-ppe-detection | v8n | 0.698 | 0.696 | 0.675 | 0.053 / 0.146 | 25 |
| badmuriss/ppe-detection-yolov8s (P2) | v8s-P2 | — | 0.615 | 0.592 | 0.083 / 0.386 | 210 |
| keremberke/yolov8m-hard-hat-detection | v8m | — | 0.648 | — | 0.085 / **0.451** | 193 |
| COCO yolov8s (для сравнения по person) | v8s | 0.775 | — | — | — | 98 |

В Construction-PPE нет разметки «нет жилета», поэтому `no_vest` здесь не оценён. Данные авторов модели на их тесте (573 кадра из 4 источников): `no-safety vest` AP50 0.52, `no-hardhat` 0.38, `person` 0.91, `hardhat` 0.90, `safety vest` 0.86.

Почему выбрана constructelligence:
- лучшая по person, helmet и vest на отложенных данных;
- YOLOv8s без нестандартных блоков: граф по операторам совпадает с COCO yolov8s, поэтому компиляция предсказуема;
- один проход даёт и человека, и наличие либо отсутствие каски и жилета.

`badmuriss` с головой P2 тяжелее в 2 раза. Специалист по каскам keremberke лучше по `no_helmet` при IoU 0.3, но это YOLOv8m только с двумя классами. Его можно держать как второй этап для голов: веса лежат в `models/weights/hardhat_keremberke_v8m.pt`, в ONNX не экспортированы.

План Б (дообучение) не понадобился: годные веса нашлись.

### Нормализация классов `ppe` (порядок = каналы ONNX)

| id | исходный | нормализованный | использовать |
|---|---|---|---|
| 0 | person | person | да |
| 1 | hardhat | helmet | да |
| 2 | no-hardhat | no_helmet | да, с подтверждением по треку |
| 3 | safety vest | vest | да |
| 4 | no-safety vest | no_vest | да, с подтверждением по треку |
| 5 | no-mask | no_mask | **нет** — по словам авторов, не обучен |
| 6 | gloves | gloves | осторожно |
| 7 | safety shoes | safety_shoes | осторожно |
| 8 | excavator | excavator | да (S4) |
| 9 | wheel loader | wheel_loader | да (S4) |
| 10 | dump truck | dump_truck | да (S4) |
| 11 | goggles | goggles | осторожно |
| 12 | no-goggles | no_goggles | нет (AP50 0.21) |
| 13 | no-gloves | no_gloves | нет (AP50 0.07) |

Респиратора нет: класса `respirator` у модели нет, `no-mask` не обучен.

## Качество на тестовых кадрах (просмотрено глазами)

Кадры: `data/images/samples/` (13 шт., источники и лицензии — в `SOURCES.md` там же). Визуализации: `data/media/ml_checks/<model_id>__<кадр>.jpg`, сводка — `summary.json`.

- **ppe**:
  - Демо-кадр камеры склада (09): обе каски (0.75 и 0.77), оба жилета и оба человека, ложных нет.
  - Кадры камер в цехе (07, 08): верно `no_helmet` у человека без каски, `helmet` у человека в каске, `no_vest` у человека в каске без жилета.
  - Вход на стройку (02) и плакат (01): каски и жилеты почти все, `no_vest` на людях без жилетов — верно.
  - Ошибки:
    - сеточка для волос в пищевом цехе (04) один раз принята за `helmet`;
    - в толпе пропущены мелкие и дальние люди;
    - в тесной электрощитовой (03) есть сомнительный `no_vest` 0.46;
    - голова, отвёрнутая или закрытая (10, наклонившаяся женщина), не размечается никак.
  - Вывод: `no_helmet` и `no_vest` нельзя подтверждать по одному кадру — нужно голосование по треку (EventManager) и проверка, что рамка СИЗ лежит в верхней части рамки человека.
- **det** (yolov8n):
  - Люди находятся уверенно (0.8–0.9 на кадрах камер).
  - Ошибки:
    - водитель погрузчика найден, второй человек за ним пропущен — `det_s640` и `pose` находят обоих;
    - погрузчик подписан как `bus`/`truck`;
    - на складе-магазине много `bowl` — это ожидаемо для COCO, правила должны фильтровать классы.
  - `det_n416` и `det_n480` на кадре цеха (04) находят почти тех же людей, что и 640, но теряют дальних и мелких.
- **pose**: скелеты корректны на кадре камеры (09), на стройке (01) и у водителя погрузчика (05); точки ног у сидящего — с низкой уверенностью.
- **fire**:
  - На 10 открытых кадрах с огнём (`Vertex-Test/FireSmokeDataset`, test) огонь или дым найден на 9; промах — на ч/б кадре.
  - Тлеющая сигарета крупным планом детектируется как `fire` 0.65 — полезно для S7, но это и источник ложных.
  - На 10 кадрах без огня (01–10) — ни одного ложного срабатывания при пороге 0.4.
  - Ночной уличный кадр с дымом (11): скоры 0.13–0.31, ниже порога.

Сверка ONNX с эталоном: на кадре 02 `ml/check_onnx.py` (ONNX + `ml/yolo_decode.py`) и ultralytics на `.pt` дают одинаковые 17 детекций с теми же рамками и скорами. На кадре 07 1280×720 есть мелкие расхождения: ultralytics на `.pt` работает с прямоугольным входом 640×384, а ONNX — со статическим квадратом 640×640.

## Калибровочный набор для квантования

`data/calib/` — 200 JPEG, `manifest.csv` (файл, тематика, источник, исходное имя, лицензия). Сборка: `python ml/build_calib.py --work <каталог со скачанными наборами>`, отбор детерминирован (seed 12), тестовые кадры исключены.

| тематика | кадров | источник | лицензия |
|---|---|---|---|
| стройка | 60 | Roboflow Construction Site Safety, [HF-зеркало](https://huggingface.co/datasets/keremberke/construction-safety-object-detection) (test + valid) | CC BY 4.0 |
| склад, погрузчики | 50 | Roboflow forklift (images.cv), [HF-зеркало](https://huggingface.co/datasets/keremberke/forklift-object-detection) | CC BY 4.0 |
| цех, производство | 60 | SH17 (фото Pexels), [HF-зеркало Roboflow](https://huggingface.co/datasets/fathansanum/SH-17-Dataset), test | Pexels License; зеркало помечено CC BY 4.0 |
| демо-камеры | 30 | кадры `data/videos/*` (worker-zone 15, store-aisle 8, people-detection 7) | см. docs/DATA.md |

Минусы:
- В SH17 и forklift попадаются офисы, лаборатории и рисунки погрузчиков. Для диапазонов активаций это терпимо, но доменных кадров камер лучше добавить больше.
- Для `fire` стоит отдельный калибровочный поднабор с огнём и дымом: на заводских кадрах ветка огня почти не активируется.

## Риски компиляции под LinQ TPU

Операторы во всех графах одинаковые (detect: 232–234 узла, pose: 268): Conv, Mul + Sigmoid (SiLU), Add, Concat, Split, Slice, MaxPool (SPPF), Resize (nearest, ×2), Reshape, Transpose, Softmax (DFL), Sub, Div.

- **Голова декодирования** (Reshape → Transpose → Softmax по 16 бинам DFL → Conv 1×1 → Sub/Add/Div с якорями → Concat) плохо квантуется в int8: координаты в пикселях до 640 и скоры 0..1 оказываются в одном тензоре. Рекомендация — резать граф до головы (у A1 это `tpu/compile.py --cut-preset yolov8`), DFL и декод делать на CPU.
- **Resize** (nearest, scale 2) и **Split** — проверить поддержку в SDK. Если не поддерживаются, Split заменяется на Slice, Resize — на ConvTranspose или ручной upsample.
- **SiLU** в виде Sigmoid + Mul — 57–66 пар на граф. Если в SDK нет слияния, это узкое место по скорости.
- **ppe и det_s640** (YOLOv8s, 28 GFLOPs) примерно в 3 раза тяжелее n-моделей. Если на TPU не уложимся по FPS — `ppe` на 480 или раз в N кадров только на кропах людей.
- Скоры классов `no_*` редкие и низкие (0.4–0.6). Квантование может сдвинуть их сильнее, чем `person`, поэтому сверку CPU fp32 и TPU int8 нужно делать по каждому классу, а не только по рамкам людей.

## Как проверить

```bash
python ml/check_onnx.py --model det --image data/images/samples/09_cctv_warehouse_workers_ppe.jpg -v
python ml/check_onnx.py --model all --image data/images/samples/ --json data/media/ml_checks/summary.json
python ml/check_onnx.py --model ppe --video data/videos/a3dev_worker_zone_detection/worker-zone-detection.mp4 --fps 5
python ml/export_onnx.py --preset coco          # пересобрать det и pose (torch — только через tools/heavy.sh)
python ml/eval_ppe.py --data <construction-ppe> --model имя=путь.pt ...   # сравнение СИЗ (torch, heavy.sh)
```

Безопасность весов: все скачанные `.pt` до загрузки в torch проверены тем же статическим сканером, что использует HF (0 опасных глобалов).
