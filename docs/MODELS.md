# Модели: внешний TPU и воспроизводимые кандидаты

Production-реестры `configs/models.yaml` и `configs/models.stand.yaml` содержат
программы LinQ TPU. Запуск `scripts/run_tpu.sh` использует `linq-tpu`, RAW и
запрещает неявный CPU fallback. CPU-голова декодирования после TPU допустима;
полный ONNX CPU inference не подтверждает работу TPU.

`configs/models.cpu-baseline.yaml` закрепляет прежний det/pose 640 для
повторяемых CPU-экспериментов. Более тяжёлая модель выбирается по качеству, а
её время на ноутбуке служит диагностикой. Перед включением кандидата нужны
компиляция, сверка CPU/TPU по классам и оценка событий на frozen test.

Для недостающих сигналов действующих правил подготовлен
[отдельный подбор моделей](MISSING_RULE_MODELS_20261024.md): страховочная
привязь, сигарета, действия человека и промышленный респиратор.

Ниже отдельно сохранены результаты upstream `e23ff4f` и локального CPU пакета.
Это разные выборки и окружения; их цифры нельзя объединять в одну таблицу
рейтинга. Упоминание доступных файлов относится к соответствующему запуску,
а не ко всем checkout. Свежая проверка объединённого TPU runtime ещё не завершена.

## Снимок upstream: модели и данные стенда

## Модели

Реестр — `configs/models.yaml` (ключ `models` — записи `ModelSpec`, ключ `meta` — источник, лицензия, маппинг классов).
Файлы лежат в `models/` (`$GUARD_MODELS_DIR`, вне git): `<файл>.onnx` и `<model_id>.json` (классы, вход/выход, операторы, sha256, источник).

Общее для всех ONNX:
- opset 12, batch = 1, статический shape, упрощено onnxslim, NMS внутри нет;
- вход `images` NCHW float32, RGB 0..1, letterbox с паддингом 114 по центру;
- выход detect `(1, 4 + nc, N)`: cx, cy, w, h в пикселях входа и sigmoid-скоры классов; выход pose `(1, 56, N)` — плюс 17 точек COCO (x, y, conf);
- N = 8400 (640), 4725 (480), 3549 (416).

### Таблица моделей

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

### СИЗ: выбор модели

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

#### Нормализация классов `ppe` (порядок = каналы ONNX)

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

### Качество на тестовых кадрах (просмотрено глазами)

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

### S2: СИЗ на кадрах сценария (24.10, кадры сценариев)

Проверено на `data/images/s2` (s2_01, s2_02 — 48 кадров) и `data/images/s2_ppe` (43 кадра); визуализации — в `mosprom-heavy.noindex/a2_work/s2/`.
- **s2_02** (склад, четверо, у троих нет ни каски, ни жилета): все 4 человека найдены почти на каждом кадре.
  - `no_helmet` 0.42–0.83 и `no_vest` 0.43–0.69 — на людях без СИЗ;
  - `helmet` 0.76 — у рабочего в каске.
- **s2_01** (коридор): каски 0.62–0.64 и жилеты 0.78–0.83 у людей в СИЗ, `no_vest` у женщины без жилета.
  - Ошибки:
    - её непокрытая голова на части кадров не помечена `no_helmet`, хотя на t40–t45 помечена (0.49);
    - пустые кадры коридора дают 0 детекций — ложных нет.
- В `s2_ppe` изредка срабатывает `no_goggles` — правило S2 его не использует.
- Для S2 этого достаточно при условии, что подтверждение идёт по треку (`min_duration_s`, `min_ratio`). Одиночные пропуски `no_helmet` правило переживёт.

### S7: огонь и курение (24.10, кадры сценариев)

- **s7_01** (курение в помещении, огня нет), 35 кадров `data/images/s7`:
  - при пороге 0.4 ложных `fire` нет: максимум 0.31, это оранжевые окна;
  - при пороге 0.15–0.2 появляются 4 ложных `fire` на окнах;
  - выдыхаемый дым у лица распознаётся как `smoke` 0.61–0.66 на 2 кадрах из 35 (t23–t24).
  - Вывод: **курение моделью огня не решается**. `smoke` у головы человека годится только как слабый вспомогательный признак.
- **Открытый огонь**:
  - 9 из 10 открытых кадров с пламенем найдены при пороге 0.4 (0.40–0.90);
  - ночной кадр 11 (огонь на земле, дым) — только 0.13–0.31, то есть пропуск;
  - ч/б кадр — пропуск.
- **Порог**: оставлен 0.4. Ниже начинаются ложные срабатывания на оранжевых окнах и бликах, а точность приоритетнее. Для S7 событие подтверждать по нескольким кадрам (`min_duration_s`).
- **На защите** показывать S7 только на ролике с хорошо видимым пламенем. Курение как сценарий не заявлять: нужен отдельный детектор сигареты, например `Beehzod/smoke_cigarette-detection*` — это YOLO11, не проверено.
- Альтернатива `mfranzon/fire-smoke-yolov8` (ранний бытовой огонь, в обучении есть сигареты) не проверена: чекпойнт сохранён через `dill`, на это не хватило времени.

### S4: техника вне COCO (24.10, кадры сценариев)

- **Экскаватор** (s4_03): `ppe` находит `excavator` на всех 14 кадрах (0.4–0.6). COCO `det` на этих кадрах выдаёт cup/boat/chair. Отдельная модель `vehicle` не нужна: брать `excavator`, `wheel_loader`, `dump_truck` из `ppe`, а `truck/car/bus` — из `det`.
  - Ошибка: жёлтый компрессор в углу кадра тоже помечается `excavator` (0.41–0.49).
- **Погрузчик** (вторая половина s4_02, тёмный проём): не находит ни одна модель.
  - Проверен `keremberke/yolov8s-forklift-detection` (классы forklift, person; экспортирован в `models/forklift-640.onnx`): он называет «forklift» грузовик и экскаватор, а настоящий погрузчик в темноте пропускает. **В реестр не включён.**

### TPU: компиляция ppe и fire (24.10, стенд, `tpu/compile.py`)

Команда:
```
tpu/compile.py --onnx ppe-640.onnx --name ppe_640 --cut-preset yolov8 --calib-dir models/calib_a2 --calib-n 100 --preproc letterbox --preset O5 --fallback O1
```
`calib_a2` — это `data/calib`, 100 кадров равномерно.

| модель | O5 | время | программа | cos quant/fp32: DFL / Sigmoid / output0 |
|---|---|---|---|---|
| ppe_640 | да | 149 с | 9.8 МБ | 0.959 / 0.816 / 0.998 |
| fire_640 | да | 107 с | 2.8 МБ | 0.988 / 0.820 / 0.999 |
| для сравнения: yolov8n_480_ft | да | — | 2.9 МБ | 0.990 / 0.845 / 0.999 |
| для сравнения: yolov8n_pose_640 | да | — | 3.0 МБ | 0.997 / 0.729 / 1.000 |

- Артефакты: `models/tpu/{ppe_640,fire_640}/` (`.tpu`, `manifest.json`, `*_tail.onnx`, `thresholds.json`, `compile.log`) и на стенде в `~/work/models/tpu/`.
- **Сверка TPU с CPU по детекциям не выполнена.** Воркер читает программы только из `/auto/cache/hackathon10864/`, а копирование туда в этой сессии не было разрешено.
- После копирования:
  - раскомментировать `tpu_program` у `ppe` и `fire` в `configs/models.yaml`;
  - запустить `tpu/yolo_check.py --model ppe_640 --onnx ~/work/models/ppe-640.onnx --images ~/work/models/ppe_check --n-images 43 --nc 14 --conf 0.4`. В `ppe_check` лежат 13 кадров samples и 30 eval_frames сценариев.
- Косинус по скорам (Sigmoid) у ppe в том же диапазоне, что у контрольных моделей, которые прошли сверку по детекциям. По координатам (DFL) ppe чуть хуже: 0.959 против 0.99. Как у yolov8n, ожидается Δconf около −0.05, поэтому для TPU порог `ppe` ставить 0.3–0.35.

### Калибровочный набор для квантования

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

### Риски компиляции под LinQ TPU

Операторы во всех графах одинаковые (detect: 232–234 узла, pose: 268): Conv, Mul + Sigmoid (SiLU), Add, Concat, Split, Slice, MaxPool (SPPF), Resize (nearest, ×2), Reshape, Transpose, Softmax (DFL), Sub, Div.

- **Голова декодирования** (Reshape → Transpose → Softmax по 16 бинам DFL → Conv 1×1 → Sub/Add/Div с якорями → Concat) плохо квантуется в int8: координаты в пикселях до 640 и скоры 0..1 оказываются в одном тензоре. Рекомендация — резать граф до головы (`tpu/compile.py --cut-preset yolov8`), DFL и декод делать на CPU.
- **Resize** (nearest, scale 2) и **Split** — проверить поддержку в SDK. Если не поддерживаются, Split заменяется на Slice, Resize — на ConvTranspose или ручной upsample.
- **SiLU** в виде Sigmoid + Mul — 57–66 пар на граф. Если в SDK нет слияния, это узкое место по скорости.
- **ppe и det_s640** (YOLOv8s, 28 GFLOPs) примерно в 3 раза тяжелее n-моделей. Если на TPU не уложимся по FPS — `ppe` на 480 или раз в N кадров только на кропах людей.
- Скоры классов `no_*` редкие и низкие (0.4–0.6). Квантование может сдвинуть их сильнее, чем `person`, поэтому сверку CPU fp32 и TPU int8 нужно делать по каждому классу, а не только по рамкам людей.

### Как проверить

```bash
python ml/check_onnx.py --model det --image data/images/samples/09_cctv_warehouse_workers_ppe.jpg -v
python ml/check_onnx.py --model all --image data/images/samples/ --json data/media/ml_checks/summary.json
python ml/check_onnx.py --model ppe --video data/videos/a3dev_worker_zone_detection/worker-zone-detection.mp4 --fps 5
python ml/export_onnx.py --preset coco          # пересобрать det и pose (torch — только через tools/heavy.sh)
python ml/eval_ppe.py --data <construction-ppe> --model имя=путь.pt ...   # сравнение СИЗ (torch, heavy.sh)
```

Безопасность весов: все скачанные `.pt` до загрузки в torch проверены тем же статическим сканером, что использует HF (0 опасных глобалов).

### Дообучение на открытых наборах (24.10)

Источники и состав наборов — `docs/DATASETS.md`, карточки версий с метриками — `ml/model_cards/`.

#### Погрузчик (S4)

**forklift-v1** — первый оборот маховика на открытых данных (карточка `ml/model_cards/forklift-v1.json`).
- YOLOv8n COCO (`models/weights/yolov8n.pt`) дообучен на `forklift-ds-v1` (`docs/DATASETS.md`), классы
  `[person, forklift]`. Обучение — `ml/flywheel/train.py`: MPS, imgsz 480, batch 16, workers 4, лимит 35 мин,
  44 эпохи, лучшая 41-я, val mAP50 0.774. Веса — `models/weights/forklift-v1.pt`.
- ONNX `models/forklift_v1-480.onnx` (12.7 МБ, выход `(1, 6, 4725)`), CPU p50 15.0 мс — как у `det_n480` (16.0 мс).
- В реестре — `model_id: forklift`, `conf_threshold` 0.3. Трек техники рождается от 0.45 (трекер),
  слабые рамки только продлевают его.
- Подключение — параметр правила: `vehicle_proximity` с `params.models: [det, forklift]` загружает модель только
  на своей камере. Трекер объектов берёт технику из всех моделей кадра (`guard/pipeline/worker.py::object_detections`),
  поэтому погрузчик из `forklift` и фура из `det` попадают в одно правило. Класс `person` модели не используется:
  люди по-прежнему из `det`.

**Тест набора** (238 кадров, в обучении не участвовали; AP50 COCO, 101 точка; CPU, onnxruntime, декодер `ml/yolo_decode.py`;
`ml/flywheel/gate.py`):

| модель | вход | forklift AP50, весь тест | forklift P / R при 0.35 | forklift AP50, тест `forklift_rf` (42 кадра) | person AP50 |
|---|---|---|---|---|---|
| **forklift-v1** | 480 | **0.867** | **0.71** / 0.84 | **0.918** | **0.551** |
| keremberke/yolov8s-forklift-detection | 640 | 0.503 | 0.25 / 0.88 | 0.896 | 0.505 |

На своём тесте (`forklift_rf`, его train — обучающая часть keremberke) модели почти равны. На полном тесте с фурами,
экскаваторами и автобусами keremberke даёт ложный forklift: precision 0.25.

**Покадрово на роликах оценки** (2 к/с, «видит» — рамка forklift с conf ≥ 0.45, порог рождения трека;
`ml/eval/frames_check.py`, `frames-forklift-v1-20261024-1503.json`):

| ролик, интервал | forklift-v1 | keremberke |
|---|---|---|
| s4_02, погрузчик выезжает из тёмного проёма, 20.5–26.5 с | **4 из 13 кадров**, max 0.68 | 0 из 13, max 0.18 |
| s4_02, погрузчик сдаёт назад, 32.0–34.3 с | 5 из 5, max 0.85 | 5 из 5, max 0.93 |
| ложный forklift: фура s4_02 и s4_01, экскаватор s4_03, clean_01, clean_02 (338 кадров) | **0** | 35 (фура 8, экскаватор и компрессор 23, clean_01 4) |

**События** (`ml/eval/run_offline.py`, 13 роликов, CPU; база — `scenarios.yaml`, кандидат — `scenarios_forklift.yaml`,
отличие только в `params.models`):

| вариант | TP / FP / FN / дубли | S4 (vehicle_proximity) | файл |
|---|---|---|---|
| база: det без модели погрузчика | 17 / 0 / 6 / 0 | 2 из 4 | `eval-onnx-cpu-forklift-base-20261024-1444.json` |
| det + forklift-v1 | 17 / 0 / 6 / 0 | 2 из 4 | `eval-onnx-cpu-forklift-v1-20261024-1514.json` |
| det + keremberke (для сравнения) | 17 / 0 / 6 / 0 | 2 из 4 | `eval-onnx-cpu-forklift-keremberke-20261024-1542.json` |

Новых ложных тревог нет ни на одном типе, остальные типы без изменений. **Событий S4 модель пока не добавила.**
Покадровая трассировка s4_02 (правило, треки, расстояния):
- 20.5–26.5 с. Погрузчик найден на 20.6–21.0 с, но до человека 243 пикс. при пороге 70 (0.5 высоты человека). Дальше
  погрузчик в профиль с паллетой не виден до 23.4 с. На 23.4–24.0 с он снова в треке, расстояние 121 → 50 пикс.
  при пороге 79–85, нарушение есть на 23.8 и 24.0 с. Затем паллета закрывает погрузчик анфас, и он теряется.
  Нарушение длится 0.4 с, а окно подтверждения правила — 2 с (`min_ratio` 0.7). Событий нет.
- 32.0–34.3 с. Погрузчик в треке на всех кадрах (0.65–0.86), но человек в 148–181 пикс. при пороге 41–129.
  По геометрии правила это не «сближение». Детектор здесь ни при чём.

Гейт (`ml/flywheel/gate.py`, правило контракта «не хуже»): **promote** — recall, precision, ложные в час и дубли
не хуже базы. mAP базы не применим: в det нет класса forklift. Выигрыша в событиях нет. Поэтому включение
на стенде (+1 модель на камере S4) требует отдельного решения о включении. Сборка под TPU — следующий шаг
(`ml/flywheel/deploy.py`).

Что поможет S4 дальше:
- больше кадров погрузчика с грузом, в профиль и анфас: паллета перекрывает корпус, на таких кадрах
  модель теряет погрузчик;
- данные маховика с камер склада: сэмплы `uncertain` и `flicker` по классу forklift;
- событие 32.0–34.3 с ограничено правилом (0.5 высоты человека в плоскости кадра), а не детектором.

**Второй и третий обороты (24.10, вечер).** Набор `forklift-ds-v2`: тот же test (238 кадров, совпадает с v1 по
sha1, с train и val v2 пересечений нет), в train ImageNet-кадры с рамками учителя от 0.4 вместо 0.6 (884 кадра против 833).
- **forklift-v2** — дообучение forklift-v1 на v2, mixup 0.15, штатный lr0 0.01. Лучшая эпоха — первая, дальше
  модель только хуже: высокий LR при дообучении от своей версии стирает выученное. Гейт: **отказ** (mAP50 0.659 против
  0.709). После этого в `ml/flywheel/train.py` появились `--lr0` и `--warmup-epochs`.
- **forklift-v3** — YOLOv8s COCO на v2, 480, batch 8, 27 эпох за 35 мин, val mAP50 0.765. Гейт: promote (события
  как у v1, mAP50 0.711 против 0.705), но **не выкатывается**: класс forklift хуже, а модель в 3.5 раза тяжелее.

Тест набора и кадры роликов — одним кодом, 24.10 21:05–21:20 (`ml/flywheel/gate.py`, `ml/eval/frames_check.py`,
`frames-forklift-v3-20261024-2110.json`). У forklift-v1 mAP50 0.705: днём на том же ONNX было 0.709, P/R при 0.35 те же.

| модель | mAP50 теста | forklift AP50 | forklift P / R при 0.35 | s4_02, тёмный проём 20.5–26.5 с | s4_02, 32.0–34.3 с | ложный forklift, 338 кадров |
|---|---|---|---|---|---|---|
| **forklift-v1** (n, 12.7 МБ) | 0.705 | **0.862** | **0.71** / 0.84 | **4 из 13**, max 0.68 | **5 из 5** | 0 |
| forklift-v2 (n) | 0.659 | 0.804 | 0.65 / 0.77 | — | — | — |
| forklift-v3 (s, 44.7 МБ) | **0.711** | 0.848 | 0.59 / **0.88** | 0 из 13, max 0.38 | 4 из 5 | 0 |

События на текущем коде (13 роликов, CPU, общий кэш детекций; все три прогона — один коммит):

| вариант | TP / FP / FN / дубли | S4 (vehicle_proximity) | файл |
|---|---|---|---|
| det без модели погрузчика (`scenarios.yaml`) | 18 / 0 / 5 / 0 | 2 из 4 | `eval-onnx-cpu-forklift-base-20261024-2105.json` |
| det + forklift-v1 (`scenarios_forklift.yaml`) | 18 / 0 / 5 / 0 | 2 из 4 | `events-base-20261024-2055.json` |
| det + forklift-v3 | 18 / 0 / 5 / 0 | 2 из 4 | `events-forklift-v3-20261024-2055.json` |

После объединения кода 24.10 (642a588) база стала 18 / 0 / 5: `phone_usage` теперь 1 из 1. Модель погрузчика на это
не влияет. На стенд идёт forklift-v1.

#### Лежащий человек (S3)

_Раздел заполняет ds-falls._

#### Сигарета (S7)

_Раздел заполняет ds-smoking._

#### СИЗ (S2): SH17 и Construction-PPE

_Раздел заполняет ds-ppe._

## Локальный CPU пакет: отдельный протокол

## Подготовка моделей для локального CPU запуска

`ml/export_onnx.py` и `ml/registry.py` используют каталог `models/` текущего checkout. Для общих артефактов между worktree задаётся `GUARD_MODELS_DIR`; `--models-dir` экспортера имеет приоритет. Чужих абсолютных путей и fallback в другой checkout больше нет. Импорт экспортера, `--help` и чтение реестра не загружают PyTorch/Ultralytics.

### Экспорт одного официального детектора

Из корня checkout с активированным Python 3.11 окружением:

```bash
uv pip install --python .venv/bin/python --requirements pyproject.toml --extra ml
export GUARD_MODELS_DIR="${GUARD_MODELS_DIR:-$PWD/models}"
export GUARD_DATA_DIR="${GUARD_DATA_DIR:-$PWD/data}"
export MPLCONFIGDIR="$GUARD_DATA_DIR/cache/matplotlib"
export YOLO_CONFIG_DIR="$GUARD_DATA_DIR/cache/ultralytics"
mkdir -p "$GUARD_MODELS_DIR/weights" "$MPLCONFIGDIR" "$YOLO_CONFIG_DIR/Ultralytics"
curl --fail --location --output "$GUARD_MODELS_DIR/weights/yolov8n.pt" \
  https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt
python ml/export_onnx.py --preset coco --model-id det --models-dir "$GUARD_MODELS_DIR" --threads 2
```

`--model-id pose` экспортирует только pose 640. Другие ID пресета: `det_n480`, `det_n416`, `det_s640`, `pose_n480`. Вызов `--preset coco` без ID сохраняет прежнее поведение: все шесть вариантов последовательно. `--imgsz` и `--task` относятся к пользовательским весам; размер и задача пресета определяются его ID.

```bash
python ml/export_onnx.py --weights /path/to/trusted/best.pt --model-id ppe \
  --out ppe-640.onnx --imgsz 640 --models-dir "$GUARD_MODELS_DIR"
```

Существующий относительный путь к весам разрешается от текущего каталога. Узнаваемое имя отсутствующих официальных весов использует кеш `<models-dir>/weights/`; неизвестный отсутствующий файл вызывает ошибку до загрузки ML runtime. `--out` — имя `.onnx` внутри каталога моделей. Для COCO использованы [официальные pretrained YOLOv8n weights](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt) и [YOLOv8s weights](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s.pt); сторонний PPE-кандидат загружен непосредственно в ONNX. Сторонние pickle не загружались.

Экспорт CPU: opset 12, FP32, статический batch 1, без NMS внутри графа; letterbox, декод и NMS выполняет runtime. После экспорта обязательно вызывается `onnxslim`, затем ONNX checker и один реальный ORT inference на нулевом входе. Ошибка simplifier видна вызывающему коду. Сам Ultralytics способен проглотить такую ошибку и вывести `export success`, поэтому его встроенный simplify отключён. Найденная несовместимость `onnxslim 0.1.71` с `onnx 1.23.2` устранена фиксацией `onnxslim 0.1.98`.

PyTorch intra/inter-op ограничены `--threads` (по умолчанию 2); внутренний сброс Ultralytics `select_device("cpu")` тоже учитывает этот лимит. Проверка ORT использует два intra-op потока и один inter-op. Экспорт идёт последовательно; TPU SDK, компиляция, GPU/MPS и обучение не запускаются.

### Проверенный артефакт 24.10.2026

Подготовлен `models/yolov8n-640.onnx` и `models/det.json`, исходные веса — `models/weights/yolov8n.pt`. Файлы находятся вне git. Для CPU S1 runtime достаточно `det` закреплённого `configs/models.cpu-baseline.yaml`; `pose` не требуется текущим трём правилам S1 и не экспортировалась.

| Свойство | Результат |
|---|---|
| Платформа экспорта | macOS arm64, Python 3.11.16, CPU |
| Ultralytics / torch / torchvision | 8.3.221 / 2.8.0 / 0.23.0 |
| ONNX / onnxslim / ORT | 1.23.2 / 0.1.98 / 1.31.0 |
| NumPy / OpenCV | 2.4.6 / 4.11.0.86 |
| torch intra-op / inter-op | 2 / 2 |
| ONNX размер / узлы | 12 851 095 байт / 234 |
| Вход → выход | `[1, 3, 640, 640]` → `[1, 84, 8400]` |
| Классы | 80 COCO, включая `person`; в `det.json` |
| Проверка | ONNX checker и ORT успешны; `problems=[]` |
| SHA256 исходных weights | `f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36` |
| SHA256 данного ONNX | `aa9699693737742033bc899670016413e1c3f9c2725b5fd0b302cfda44ce3a99` |

Основные зависимости экспортного окружения зафиксированы в optional-группе `ml`; версии, источник, лицензия, классы, граф и SHA256 записываются в `<model_id>.json`. Это фиксация проверенного основного стека, не полный lock транзитивных зависимостей. Повторный экспорт может менять ONNX SHA256 из-за метаданных Ultralytics; хеш выше относится к конкретному подготовленному файлу.

Модель и Ultralytics: [AGPL-3.0, исходный LICENSE версии 8.3.221](https://github.com/ultralytics/ultralytics/blob/v8.3.221/LICENSE). Поддерживаемая модель и формат экспорта описаны в [YOLOv8](https://docs.ultralytics.com/models/yolov8/) и [Export](https://docs.ultralytics.com/modes/export/). Экспорт подтверждает формат и работоспособность CPU inference; совместимость с TPU и качество на промышленном потоке не измерялись.

Реальное видео, отдельный demo-конфиг и результаты S1 с SQLite readback описаны в [DATA.md](DATA.md).

### Модельный пакет 24.10.2026

YOLOv8s-640 экспортирован отдельно в `models/candidates/yolov8s-640/` с
canonical ролью `det`; ONNX SHA256
`ddac89c188b95f774e40c0f76b40a48e3ea3752059d95f87acaccedc2009a0dd`,
44 869 889 байт. Парный S1 дал одинаковые 3 TP/1 FN/1 duplicate при стоимости
video-loop в 2.71 раза выше. На 141 размеченном Construction-PPE test image
person AP50 n/s составила 0.757/0.727. По этим проверкам baseline n сохраняется.
Команды и точный протокол — [MODEL_COMPARISON.md](MODEL_COMPARISON.md).

Экспериментальная opt-in роль `ppe` подготовлена в
`models/candidates/buildvision-hazards-v0.3/`. Она поддерживает helmet/vest и
явные negative классы, но `no_helmet` recall всего 0.20 при IoU 0.5/conf 0.35;
качество `no_vest` не измерено, respirator не поддерживается. Она не включена
по умолчанию. Источник и mapping — [PPE_MODEL_CANDIDATE.md](PPE_MODEL_CANDIDATE.md),
все результаты и следующие шаги — [MODEL_RESEARCH_20261024.md](MODEL_RESEARCH_20261024.md).
