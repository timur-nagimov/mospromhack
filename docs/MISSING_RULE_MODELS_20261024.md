# Модели для незакрытых сигналов действующих правил

Дата проверки: 24.10.2026. Ветка `feature/missing-rule-models-research`.
Я проверил публичные карточки, наличие файлов и версии моделей. Для двух
ONNX скачал payload, сверил размер/SHA256 и проверил граф. Инференс,
качество на наших видео и компиляцию LinQ в этом исследовании не запускал.
Машиночитаемые результаты: [каталог](../configs/rule-model-gap-catalog.json).

## Что требуется каждому правилу

| Правило | Незакрытый сигнал | Подходящий следующий эксперимент |
|---|---|---|
| `height_no_harness` | Явное отсутствие/наличие привязи; отдельно её соединение со страховочной линией | SafetyVision v2 для `No_Harness`; ProjectThor для harness/lanyard; затем отдельная модель отношений |
| `smoking_fire` | Сигарета и её использование конкретным человеком | MeetCool или YOLO11m cigarette + связь с рукой/ртом и история; огонь оставить отдельным сигналом |
| `ppe_missing` | Респиратор; остальные необходимые СИЗ вне текущего контракта | Отдельный industrial-respirator dataset/обучение; `Mask` не переименовывать в `respirator` |
| `unsafe_carry` | Связь человек–груз и действие переноски | VideoMAE Large AVA / SlowFast AVA + существующие bbox/pose/геометрия |
| `fence_climb` | Действие перелезания и конкретное ограждение | VideoMAE AVA `climb` + детектор/сегментация ограждения или заданная зона |
| `running` | Действие бега вместо одной скорости трека | VideoMAE AVA `run/jog` с контролем быстрого шага и движения камеры |
| `fall_immobile` | Падение во времени; видимость лежащего человека | AVA `fall down` / `lie/sleep`; Wardy TemporalFallGRU из прежнего каталога как отдельный контроль |
| `phone_usage` | Использование телефона, а не только присутствие объекта | AVA `answer phone` / `text on/look at a cellphone` + текущий телефон/pose |
| `vehicle_proximity` | Стабильная видимость погрузчика; физическое расстояние | Существующий forklift-v1 и YOLO11s forklift из каталога; калибровка камеры нужна отдельно от модели |
| `zone_intrusion` | Новый специализированный класс не требуется | Текущие person detector/tracker, проверка качества по камерам |
| `wrong_way` | Новый специализированный класс не требуется | Текущие person tracks, стрелка и зона |
| `zone_dwell` | Новый специализированный класс не требуется | Текущие person tracks и наблюдаемость присутствия |
| `zone_overcapacity` | Новый специализированный класс не требуется | Текущие person tracks; отдельная оценка пропусков и дублей при перекрытиях |

Классификатор действия даёт дополнительное наблюдение. Он не заменяет
условия зоны, непрерывность эпизода, принадлежность объекта человеку и
подтверждение события. Недостающие наблюдения сохраняют `unknown`.

## Страховочная привязь: первый приоритет

**SafetyVision v2 YOLOv8s** — первый кандидат для отсутствия привязи:
класс 10 `No_Harness`; готовые ONNX 640/896. Автор сообщает для этого класса
на 256 test instances P=0.728, R=0.773, AP50=0.830 при 896. Это авторский
результат, не наш замер. У v1 был один test instance и AP=0; версии не смешивать.
Положительного `Harness` и состояния крепления в class map нет.
[Карточка с per-class результатами](https://huggingface.co/ayushgupta7777/safetyvision-yolov8).

Локально проверен именно `v2/best_896.onnx`:

- revision `56a71758b55f0e9f2b4b2d6b51a779a1f882da10`;
- 44,926,046 байт, SHA256 `b250353639e01800f9cbe79c6002b8b041bdae7560328b8e18ad4a42dc3844e1`;
- FLOAT `[1,3,896,896]` → FLOAT `[1,17,16464]`, opset 20;
- class map из самого графа совпадает с карточкой; ONNX checker passed;
- внешних tensor data и локальных ONNX functions нет.

**Thalos Scaffold & Harness Safety v1** — резервный кандидат. В карточке
заявлены `harness_worn` / `harness_missing`, но точные имена оговорены как
переменные. Опубликован только `scaffold_harness_weights.pt`, 87,685,242 байта.
Примеры карточки используют другое имя файла; независимых метрик нет.
Сначала проверить реальный class map и получить воспроизводимый ONNX.
[Источник](https://huggingface.co/thalostech2025/thalos-scaffold-harness-safety-v1).

**ProjectThor Fullbody Safety Harness** — источник данных и обученной
модели для классов `fullbody-Harness`, `Lanyard+Hook`: 3,542 изображения,
YOLO11n; на странице P=80.1%, R=61.7%, mAP50=67.9%.
Подтверждён опубликованный API; прямые скачиваемые веса не установлены.
Для собственного сервера/TPU нужен экспорт весов либо своё обучение на наборе.
Найденный строп с крюком ещё не доказывает прикрепление к анкеру.
[Первичный проект](https://universe.roboflow.com/projectthor/fullbody-safety-harness-vkfei).

Правильное крепление — отдельная задача: наблюдения `harness_worn`,
`lanyard_visible`, `hook_visible`, `connected_to_anchor` и идентификаторы
человека/связей нужно различать. Скрытый карабин оставляет соединение
неизвестным. Готовую проверенную открытую модель всей цепочки в этом поиске
не установил. В [работе о распознавании действий со страховкой](https://ascelibrary.com/doi/10.1061/JCEMD4.COENG-12542)
рассматривается 3D skeleton + LSTM, но код/данные предлагаются по запросу.

Сейчас `height_no_harness` читает `track.attributes['harness']`, однако
`guard/tracking/attributes.py` этот атрибут не формирует. Поэтому один ONNX
не активирует распознавание страховки. Нужна отдельная интеграция: профиль
модели → class mapping → неоднозначная/однозначная связь с человеком →
атрибут с историей и наблюдаемостью → существующее правило. При наличии
только явного `No_Harness` можно доказывать отрицательный сигнал;
отсутствие этой детекции не означает `harness=True`.

## Курение: отдельная модель вместо fire

**MeetCool YOLOv8n smoking** — практически доступный первый контроль.
В карточке мало данных о качестве; реальный скачанный ONNX содержит
`cigarette`, `person`, `smoke`, а не готовый temporal smoking event.
[Первичный репозиторий](https://huggingface.co/MeetCool/yolov8n-smoking-detection).

Проверка графа: 12,252,672 байта, revision
`7d510c6aaa3fbc3048368565d4236993c077dc0a`, SHA256
`afcf1e6d1293427b9221b60c9505e76072f9a295120d0a78c71e35f9c444ff8e`;
FLOAT `[1,3,640,640]` → `[1,7,8400]`, opset 19, checker passed,
external data/functions отсутствуют. Качество не измерено.

Для сравнения по качеству есть два более крупных YOLO11m:
[Beehzod cigarette](https://huggingface.co/Beehzod/best_smoke_cigarette-detection)
и [HEIher smoking-detection](https://huggingface.co/HEIher/smoking-detection).
У обоих опубликованы `.pt`, готового ONNX в проверенном списке файлов нет.
У HEIher класс называется `cigarette`, несмотря на описание про дым.
Метрики карточек не сравнивают модели на общей выборке.

[SmokeGuardModel](https://huggingface.co/aliffatulmf/SmokeGuardModel) публикует
YOLOv5 FP32 ONNX (83,890,343 байта) и CPU INT8. Полный class map, качество
и IO по карточке не установлены, payload не скачивался. Его YOLOv5 output
нельзя направлять в текущий YOLOv8 decoder без проверки. Для LinQ исходным
контролем нужен FP32, чужое CPU-квантование не является TPU-программой.

Следующая интеграция должна учитывать сигарету у руки/рта конкретного
человека, временное подтверждение и отрицательные примеры: ручка, инструмент,
рация, еда, пар и пыль. Огонь и курение не объединять одним class mapping.

## Действия: бег, переноска, перелезание, падение, телефон

**VideoMAE ViT-Large AVA-Kinetics** — основной тяжёлый кандидат для этого
пакета. Это actor-level action detection, в отличие от ранее выбранного
Kinetics-400 clip classifier. В официальном MMAction2 есть config и
checkpoint; результат авторов на AVA2.2 — mAP 38.7. **SlowFast R50 AVA2.2**
остаётся вторым архитектурным контролем. Их benchmark не измеряет наши нарушения.
[VideoMAE](https://github.com/open-mmlab/mmaction2/tree/main/configs/detection/videomae),
[SlowFast](https://github.com/open-mmlab/mmaction2/tree/main/configs/detection/slowfast).

В [официальном label map](https://github.com/open-mmlab/mmaction2/blob/main/tools/data/ava/label_map.txt)
подтверждены `run/jog` (10), `carry/hold` (17), `climb` (20), `smoke` (54),
`fall down` (5), `lie/sleep` (8), `answer phone` (15), `text on/look at a
cellphone` (57). Общий `climb` не доказывает перелезание ограждения;
`carry/hold` не доказывает опасный груз, массу или закрытый обзор.

Проверен source config VideoMAE Large: 16 кадров с interval 4, RGB mean/std,
NCTHW, person proposals и RoIAlign, 81 выходной класс с background. Частота
исходного видео и fps текущего pipeline должны быть согласованы явно.
Веса/ONNX здесь не скачивались. Экспорт, Conv3D/tubelets, attention,
RoIAlign и переменное число людей требуют отдельной проверки LinQ.
Source revision и hashes записаны в каталоге.

## Респиратор

Готовый подтверждённый industrial-respirator ONNX с нужной семантикой в
этом поиске не найден. [TFM dataset](https://github.com/GibranBenitez/TFM_dataset)
различает surgical/cloth/respirator/valved лица, но доступность полного
набора/весов на странице не подтверждена. Это материал для проверки данных,
а не готовая замена текущей модели.
[Исследование industrial head PPE](https://doi.org/10.1038/s41598-022-20282-9)
отдельно оценивает N95 и cartridge respirator, причём для cartridge
respirator результаты хуже; ссылку на готовый подходящий артефакт не установил.
Нужны точный тип СИЗ и свои кадры: общее `Mask` нельзя автоматически считать
промышленным респиратором или подтверждением правильного прилегания.

## Отсеянные результаты поиска

- `Unlimitedlevel19/k3l-ppe-detection`: карточка обещает harness, но API
  репозитория не показывает файлов весов/графа.
- `introvert6162/yolov8m-ppe-detection`: классы harness_on/off названы, но
  сама карточка отмечает отсутствие данных; API не показывает весов.
- `afaq120/Harness_models`: фактические файлы относятся к bolts/plates/
  corrosion/welds, не к страховочной привязи.
- `Re-surgence/Safety-Harness-Detection`: название про harness, но README
  описывает обычные PPE; подтверждённого harness class map нет.

## Порядок следующей реализации

1. Проверить SafetyVision v2 896 и 640 на одинаковых person/torso samples:
   с привязью, без неё, со спины, с рюкзаком/жилетом и при перекрытии.
   Затем отдельной веткой добавить наблюдение harness и сквозной replay правила.
2. Сравнить MeetCool и YOLO11m cigarette на одном размеченном наборе;
   отдельно реализовать association/temporal evidence.
3. Проверить VideoMAE Large AVA на actor-labelled clips для бега, переноски
   и перелезания. Сначала offline model/decoder, затем интеграция по одному правилу.
4. Для крепления страховки и industrial respirator собрать специализированные
   данные; наличие класса в карточке не подменяет разметку и проверку.

До смены defaults нужны source-disjoint positive/negative видео, отдельные
метрики объекта, связи и события, unknown coverage и FP/час. Для внешнего
TPU — native/ONNX parity, реальная компиляция, калибровка и readback на тех же
входах. Проверенные здесь ONNX checker и SHA подтверждают только артефакты.
