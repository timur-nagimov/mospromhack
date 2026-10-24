# Открытые наборы под слабые места

Решение внутреннее, для демонстрации. Лицензии наборов не проверяем (решение человека 24.10), но источник каждого
набора записываем. Наборы лежат в `data/datasets/<имя>/` в YOLO-формате с `data.yaml` и `manifest.json`. Каталог
`data` — симлинк на `mosprom-heavy.noindex`, он не в git. Скачивание и конвертация — `ml/datasets/<имя>.py`.

**Отложенная выборка.** Кадры роликов из `ml/eval/scenarios.yaml` в обучение не попадают. Тестовые части наборов
используются только для оценки. Разбиение наборов падений — по людям и сериям, а не по кадрам.

Слабые места, под которые подбирались наборы (`docs/QUALITY.md`, разбор ошибок):

| Сценарий | Слабое место | Сейчас на роликах |
|---|---|---|
| S4 | погрузчика нет в классах моделей | 2 из 4 |
| S3 | детектор теряет лежащего человека: падение с кровати, человек на снегу | 6 из 7 |
| S7 | сигарету модель огня не видит | 0 из 1, не заявляем |
| S2 | каска у людей в кадре с первой секунды распознаётся с задержкой до 18 с; `no_helmet` AP50 0.135 | 5 из 6 |

## Погрузчик (S4)

Набор `forklift-ds-v1` — `data/datasets/forklift/` (`data.yaml`, `manifest.json`, списки теста по источникам
`test_<источник>.txt`). Классы: `0 person`, `1 forklift`. Собирает `ml/datasets/forklift.py`:

```bash
python ml/datasets/forklift.py download                   # zip и parquet с HF, кадры COCO по одному
tools/heavy.sh python ml/datasets/forklift.py teacher     # учитель: люди и рамки погрузчика на ImageNet
python ml/datasets/forklift.py build                      # конвертация, дедуп, разбиение, manifest.json
```

| Источник | Откуда | Что берём | train / val / test |
|---|---|---|---|
| `forklift_rf` | Roboflow forklift (images.cv), [HF keremberke/forklift-object-detection](https://huggingface.co/datasets/keremberke/forklift-object-detection) | погрузчики и люди, разметка Roboflow | 295 / 84 / 42 |
| `imagenet_forklift` | ImageNet-1k, класс forklift, [HF mlnomad/imnet1k_forklift](https://huggingface.co/datasets/mlnomad/imnet1k_forklift) | фото погрузчиков; рамки погрузчика ставит учитель (ниже) | 833 / — / — |
| `logistics_rf` | Roboflow Logistics v2, [HF SergioSCA/Logistics_v2_yolov8](https://huggingface.co/datasets/SergioSCA/Logistics_v2_yolov8) (300 кадров из 99 тыс.) | склад: люди; фуры, легковые, контейнеры — фон | 68 / 89 / 97 |
| `construction_rf` | Roboflow Construction Site Safety, [HF keremberke/construction-safety-object-detection](https://huggingface.co/datasets/keremberke/construction-safety-object-detection) | экскаваторы, самосвалы, фронтальные погрузчики, грузовики — фон; люди | 297 / 57 / 34 |
| `coco_vehicles` | COCO train2017, [метки Ultralytics](https://github.com/ultralytics/assets/releases/download/v0.0.0/coco2017labels.zip), кадры с images.cocodataset.org | 450 кадров с заметным truck и 150 с bus (рамка ≥ 2 % кадра); техника — фон, люди из COCO | 480 / 54 / 65 |

Итого 1973 / 284 / 238 кадров. Погрузчиков: 1201 / 98 / 43 рамки, людей: 3123 / 418 / 425. Кадров без погрузчика
(негативы): 848 / 199 / 197.

**Как размечено.**
- Техника, кроме погрузчика, остаётся без рамок. Для модели это фон, так она учится не называть фуру,
  экскаватор и фронтальный погрузчик «forklift». Это главная ошибка keremberke/yolov8s-forklift-detection.
- **Учитель** (звено 3 маховика, `docs/FLYWHEEL.md`):
  - В `construction_rf` каски размечены, а люди почти нет: 69 рамок person на 398 кадров. Людей добавляет COCO
    YOLOv8s (conf ≥ 0.5), рамки, совпавшие с разметкой (IoU ≥ 0.5), не дублируются. Добавлено 533 человека.
  - На ImageNet погрузчик есть на каждом кадре, рамок нет. Рамки ставит keremberke/yolov8s-forklift-detection
    (ONNX `models/forklift-640.onnx`, conf ≥ 0.6, до 3 на кадр), людей — он же (conf ≥ 0.5). Его слабость (фура и
    экскаватор как forklift) здесь не мешает: на этих кадрах погрузчик заведомо есть. Кадры без уверенной рамки
    (113 из 1350) не берём, иначе погрузчик стал бы фоном. ImageNet — только в train: val и test размечены людьми.
- COCO: кадры, в подписях которых есть forklift, fork lift или fork-lift, исключены (15 кадров). Подписи читаются
  HTTP Range из архива аннотаций, 253 МБ целиком не качаются.

**Разбиение и отложенные данные.**
- `forklift_rf`, `logistics_rf`, `construction_rf` — исходные train/valid/test Roboflow. Это отдельные фото, серий нет.
- `coco_vehicles` — по `sha1(id кадра) % 10`: 0 — test, 1 — val.
- `imagenet_forklift` — только train.
- **Дедуп** — dHash 64 бит, расстояние Хэмминга ≤ 5. Из пары дублей остаётся кадр более строгой части
  (test > val > train). Выброшено 460 кадров: 404 ImageNet (часть картинок images.cv взята из ImageNet,
  в том числе копии тестовых кадров `forklift_rf`), 46 `logistics_rf`, 10 `construction_rf`.
- В тесте 37 из 425 рамок person поставил учитель (`construction_rf`). На AP погрузчика это не влияет;
  AP person на этом тесте — ориентир, а не решающая метрика.
- **Ролики оценки.** Кадры 13 роликов `ml/eval/scenarios.yaml` (2 к/с, 1121 кадр) сверены по dHash со всем набором.
  Совпал 1 кадр COCO (ложное совпадение тёмного кадра), он выброшен. В обучение ролики не попали ни в каком виде.

**Не взяли и почему.**
- RF100 excavators ([LibreYOLO/excavators-czvg9](https://huggingface.co/datasets/LibreYOLO/excavators-czvg9),
  кадры роликов стройки): на HF по файлу, без токена ~5 с на файл. 300 кадров Logistics шли 29 минут, 2650 кадров
  не уложились бы. Код источника в скрипте есть (разбиение по ролику, люди — учитель).
- Полный Roboflow Logistics (99 тыс. кадров) и Roboflow Universe без ключа не скачиваются. В 300-кадровой
  подвыборке HF погрузчиков почти нет: 2 рамки в val.
- Ролики `tahaUgan/forklift-video*` — по 0.8–2.6 ГБ на ролик, разметка автоматическая (conf 0.40). Не брали.

Лицензии не проверяли: решение человека 24.10, демо внутреннее. Источники записаны выше и в `manifest.json`.

## Падения и лежащий человек (S3)

_Раздел заполняет ds-falls._

## Сигарета (S7)

_Раздел заполняет ds-smoking._

## СИЗ (S2): SH17 и Construction-PPE

_Раздел заполняет ds-ppe._
