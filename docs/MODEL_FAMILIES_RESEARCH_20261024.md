# Другие семейства и специалисты: каталог на 24.10.2026

**D-FINE-X Objects365→COCO в FP32 ONNX** уже прошёл парную проверку141 images после объединения adapter: person AP50 .771 против .757 у n, с меньшим precision .517 против .736 при .35. Результат и readback: [MODEL_FOLLOWUP_20261024.md](MODEL_FOLLOWUP_20261024.md). Проверка восстановления пропадающего человека в S3 остаётся отдельным экспериментом. Для событий стоит исследовать **VideoMAE** (`smoking`, `texting`), для позы — **ViTPose/ViTPose++** и более практичный кандидат **RTMPose**. Ни один из них здесь не признан улучшением event quality или совместимым с LinQ.

Это личный эксперимент по качеству. Хакатонные ролики используются как исторические диагностические примеры. Лицензии записаны в provenance каталога и не служат критерием отбора.

Дополнение: [RF-DETR2XL/Large2026, DEIMv2-X/DINOv3 и Grounding DINO](EXTENDED_MODEL_FAMILIES_20261024.md).
Каталог содержит13 кандидатов. RF-DETR2XL payload507.8MB уже проверен по SHA256
и ограниченно загружен: фактический head91. Static export прошёл checker,
native/ORT parity не пройден; quality и TPU не подтверждены.
[Фактический отрицательный gate](../.orchestra/metrics/rfdetr-export-20261024.json)
сохранён отдельно, tolerances не менялись.

Машиночитаемые ссылки, ревизии, ожидаемые SHA256, размеры, классы, IO и статусы находятся в [model-research-catalog.json](../configs/model-research-catalog.json). Это исследовательский каталог; production defaults и реестр моделей не меняются. ID ролей `det/ppe/pose/fire` остаются прежними.

## Что проверено

Прочитаны `docs/QUALITY.md`, `docs/MODELS.md`, выбранная provenance видео из `tools/data_manifest.json`, текущие ограничения декодера и YOLO cut presets. Во внешних системах прочитаны публичные карточки, JSON-метаданные и исходники авторов/авторов конверсии.

При подготовке этого обзора **веса не скачивались**, inference/export/обучение/компиляция не запускались. Ожидаемые SHA256 и размеры HF взяты из LFS metadata API. Отдельно прочитаны локальные отчёты по скачанному D-FINE: `models/candidates/dfine-x-obj2coco/{source,inspection}.json`. Для него подтверждены payload hash и инспекция графа по этим отчётам; inference, паритет и TPU не подтверждены. Для остальных моделей payload не проверен. `null` в каталоге означает неизвестное значение. Веса и локальные отчёты не входят в этот commit.

Состояния различаются: `discovered` → metadata reviewed → payload downloaded/hash verified → graph inspected → local inference validated → target TPU compiled → target TPU accepted. Наличие ONNX подтверждает только опубликованный формат. Оно не подтверждает паритет конверсии, поддержку операторов или выполнение на внешнем TPU.

## Какие ошибки имеет смысл исправлять

В историческом [QUALITY.md](QUALITY.md) сравнения настроены на тех же 13 роликах / 535 с; actor GT и независимый holdout не установлены. Это не текущая оценка новых моделей.

| Сценарий | Наблюдаемая проблема | Что должен проверить эксперимент |
|---|---|---|
| S3, падение/лежание | В `s3_02` человек на кровати пропадает у детектора; YOLOv8s оказался хуже YOLOv8n. В `s3_01` человек пропадает на снегу | Recall человека и позы во всём интервале; отдельно переход падения, лежание и неподвижность |
| S4, погрузчик | В COCO нет forklift; уже проверенный keremberke путает грузовик/экскаватор с погрузчиком | Настоящий forklift, тёмные сцены и hard negatives; затем связь с конкретным человеком и расстояние |
| S6, телефон | Телефон найден, исторический FN возникает в геометрии атрибута относительно головы | Сначала геометрия/ассоциация на фиксированных детекциях; temporal-модель полезна только при подходящей семантике |
| S7, курение | Fire/smoke не заменяют сигарету; включение дыма даёт FP на пыли | Присутствие сигареты и действие курения оценивать раздельно; отрицательные примеры еды, питья, телефона, пыли |

S3-ролики включают Mixkit и шесть фрагментов GMDCSA24, S4/S6/S7 — Mixkit. Объявленный другой датасет обучения не доказывает отсутствие совпадений: сравнение исходных видео, субъектов и копий ещё не сделано. `keremberke/yolov8s-forklift-detection` повторно не предлагается: локальная отрицательная проверка уже описана в [MODELS.md](MODELS.md).

## Три направления вне YOLO

### 1. D-FINE-X: первый кандидат для детекции

[Официальный D-FINE](https://github.com/Peterande/D-FINE) использует DETR-подобный decoder, HGNetv2 backbone и refinement распределения координат. Выбран X (~62M), pretrained Objects365 → finetuned COCO. Конечная голова имеет **80 COCO классов**: `person=0`, `cell phone=67`, **forklift отсутствует**. Больший датасет pretraining не добавляет классы в конечный COCO output. Авторские COCO AP и T4 latency не являются измерениями наших сцен или LinQ.

Доступны три разных происхождения весов:

| Артефакт | Пин и размер | Состояние |
|---|---|---|
| [ONNX Community FP32](https://huggingface.co/onnx-community/dfine_x_obj2coco-ONNX/resolve/4d3a85a6c29f55a5346fabaf9728f5f3751f947a/onnx/model.onnx) | rev `4d3a85a6c29f55a5346fabaf9728f5f3751f947a`, 251,138,448 B; SHA256 в JSON | Payload/граф и141/141 local inference проверены; паритет исходным весам и TPU не проверены |
| [Transformers safetensors](https://huggingface.co/ustc-community/dfine-xlarge-obj2coco) | rev `15f18d917eaddcedf9e3ffb082adcfb97a0b2d4d`, 251,619,400 B | HF port под организацией USTC, отдельно от original checkpoint |
| [Original release .pth](https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_x_obj2coco.pth) | release `dfinev1.0`, 252,140,594 B | Release API не отдаёт digest; полный byte pin пока отсутствует, загрузки pickle не было |

Проверенный [processor этой ONNX-конверсии](https://huggingface.co/onnx-community/dfine_x_obj2coco-ONNX/resolve/4d3a85a6c29f55a5346fabaf9728f5f3751f947a/preprocessor_config.json): RGB, прямой bilinear resize **640×640**, float32 NCHW `/255`, `do_pad=false`, `do_normalize=false`. Это **растяжение без YOLO letterbox**; mean/std в JSON при выключенной нормализации не применяются.

По прочитанному локальному inspection report настоящий ONNX input — float32 `pixel_values [B,3,H,W]`, outputs — float32 `logits [B,300,80]`, `pred_boxes [B,300,4]`; B/H/W динамические, opset16, 2328 nodes, checker passed, external tensors/functions отсутствуют. Для сравнения processor фиксирует H=W=640. Семантика boxes по HF API — normalized `cxcywh`. Для focal head ожидаются sigmoid scores, topK по query×class и восстановление `xyxy` в исходный размер; существующий YOLO decoder/NMS автоматически применять нельзя. Это отличается от [официального exporter](https://github.com/Peterande/D-FINE/blob/956d1709314c2c6a4df6f34de232054578a7449f/tools/deployment/export_onnx.py): `images + orig_target_sizes → labels/boxes/scores`, opset16, dynamic batch, пример с batch32.

Главный TPU-риск — default deformable attention: в [официальном source](https://github.com/Peterande/D-FINE/blob/956d1709314c2c6a4df6f34de232054578a7449f/src/zoo/dfine/utils.py) есть `grid_sample`, attention и динамическая индексация. В реальном inspection report: **GridSample18, MatMul123, ScatterND12, Erf1, Softmax20**. Нужны static B1, новый contract/cut и проверка компиляции. Переключение `default` на `discrete` без проверки качества не считается эквивалентным экспортом.

**Роль:** кандидат runtime после локального A/B и отдельной компиляции; FP32 также годится как offline detector teacher. Следующий близкий контроль — [RT-DETRv2-R101VD](https://huggingface.co/PekingU/rtdetr_v2_r101vd) из [официального семейства RT-DETR](https://github.com/lyuwenyu/RT-DETR); safetensors с ревизией/хешем есть в JSON. Он тоже COCO detector, а default sampling не снимает необходимость проверки TPU. DEIM в этот пинованный shortlist не включён: отдельная проверка артефакта ещё не проведена.

### 2. VideoMAE: действие на временном окне

[Авторский VideoMAE](https://github.com/MCG-NJU/VideoMAE) — video transformer с tubelet patch embedding. У [Kinetics-400 large](https://huggingface.co/MCG-NJU/videomae-large-finetuned-kinetics) есть safetensors 1,217,128,032 B, rev `0f6adcd5f6902900aa0281f9daacfe52bb3c4ad4`; у [base](https://huggingface.co/MCG-NJU/videomae-base-finetuned-kinetics) — 346,161,616 B, rev `488eb9a0565f257b32866000305c8178965eb9f6`. Large — offline teacher для проверки качества, base — меньший reference для последующего экспорта, а не заранее более подходящий production вариант.

В проверенном словаре **`316=smoking`, `317=smoking hookah`, `355=texting`**. Это подходящая исходная семантика для S7 и части S6. `texting` не покрывает весь `phone_usage`; отдельного `phone_at_ear`, падения или лежания в этом checkpoint не установлено. Выход — классификация клипа, без actor boxes и точных границ события.

IO по [Transformers API](https://huggingface.co/docs/transformers/model_doc/videomae): `[B,16,3,224,224]` → `[B,400]` logits; RGB, resize/crop и ImageNet normalization. 16 кадров не задают длительность в секундах: cadence/window нужно зафиксировать по рецепту и записывать timestamps. Pretraining/fine-tuning на Kinetics и авторский benchmark не устанавливают качество на CCTV или независимость от веб-роликов.

Готового авторского ONNX здесь не найдено. В просмотренном текущем `optimum-onnx/model_configs.py` регистрации VideoMAE не найдено, поэтому готовая команда CLI не заявляется. До runtime нужны явный static eager export из safetensors и паритет. TPU-риски: Conv3D tubelets, 5D input, MatMul/Softmax, GELU/LayerNorm, память temporal tokens. Мягкое распределение по 400 действиям ещё не calibrated smoking probability: обязательны background/hard negatives и привязка к человеку.

### 3. ViTPose/ViTPose++: качество позы на заданном человеке

[Авторский ViTPose](https://github.com/ViTAE-Transformer/ViTPose) — top-down transformer: сначала нужен person box, затем affine crop и keypoints. У [ViTPose++ large](https://huggingface.co/usyd-community/vitpose-plus-large) опубликован safetensors 1,737,705,492 B, rev `e211df377e49c89dba508ab3e83ddef13c0832b4`. Это offline pose teacher, 24 слоя / hidden1024 / 16 heads, dataset experts. Он не восстанавливает человека, которого не нашёл предыдущий detector.

Для меньшего runtime reference существует отдельная [ONNX-конверсия base-simple](https://huggingface.co/onnx-community/vitpose-base-simple): FP32 `onnx/model.onnx`, 343,893,670 B, rev `bbebbddd612af2728539103dd5b903a4f5a2da4c`. Это **другая модель**, не экспорт ++large. [Optimum-ONNX](https://github.com/huggingface/optimum-onnx/blob/main/optimum/exporters/onnx/model_configs.py) содержит `VitPoseOnnxConfig/keypoint-detection`; это не проверка экспорта конкретной large/MoE головы.

Ожидаемый contract: один RGB affine crop `[B,3,256,192]`, ImageNet normalization, heatmaps `[B,17,64,48]`; точные ONNX IO/opset требуют инспекции. Нужны heatmap decode, inverse affine и сохранение confidence/COCO joint order. Это не текущий YOLO pose tensor `[1,56,N]` и не classifier падения. Обучение ++ включает COCO/AIC/MPII/AP10K/APT36K/WholeBody; авторы отдельно отмечают возможные дубликаты CrowdPose train с validation других наборов. Независимость наших кадров не доказана.

TPU-риски: attention MatMul/Softmax, LayerNorm/GELU, upsampling, expert indexing, память и переменное число людей. Начать с fixed B1 на каждом человеке. Готовые INT8/4-bit ONNX Community варианты не доказывают переносимость квантования на LinQ.

### Дополнительный практический pose-контроль: RTMPose-l Body7

[Официальный RTMPose](https://github.com/open-mmlab/mmpose/tree/759b39c13fea6ba094afc1fa932f51dc1b11cbf9/projects/rtmpose) публикует [готовый ONNX SDK archive](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-l_simcc-body7_pt-body7_420e-256x192-4dba18fc_20230504.zip). Полного SHA256/размера payload в просмотренных metadata нет; токен `4dba18fc` в имени не подменяет полный byte pin. Перед скачиванием/запуском нужен отдельный acquisition record с фактическим SHA256.

Это CSPNeXt backbone + RTMCC/SimCC head на **17 COCO joints**, Body7 checkpoint. По [config](https://github.com/open-mmlab/mmpose/blob/759b39c13fea6ba094afc1fa932f51dc1b11cbf9/projects/rtmpose/rtmpose/body_2d_keypoint/rtmpose-l_8xb256-420e_coco-256x192.py) вход `[B,3,256,192]`, affine person crop, `bgr_to_rgb=true`, mean/std; ожидаются SimCC x `[B,17,384]` и y `[B,17,512]`, axis argmax `/2` и inverse affine. SDK pipeline/channel order и фактический IO ещё не прочитаны.

Conv backbone выглядит практичным кандидатом для компилятора — это гипотеза. [Head](https://github.com/open-mmlab/mmpose/blob/759b39c13fea6ba094afc1fa932f51dc1b11cbf9/mmpose/models/heads/coord_cls_heads/rtmcc_head.py) содержит GAU/gated attention, Linear и ScaleNorm, поэтому весь граф нельзя называть чисто свёрточным. Модель даёт позу, а не действие падения. Сравнивать её с ViTPose надо на одинаковых person crops; end-to-end результат отдельно включает пропуски detector.

## Специалисты для точечных пробелов

| Кандидат | Смысл выхода и веса | Данные и ограничения | Что проверять |
|---|---|---|---|
| [SafetyVision v2 YOLOv8s](https://huggingface.co/ayushgupta7777/safetyvision-yolov8) | 13 классов; `0=Fall-Detected`, `11=Person`; ready ONNX640/896, 44.76/44.93MB, opset20 | 80,304 изображений из 5 Roboflow sets, MD5 dedup, 68,253/8,025/4,026 train/val/test; разделение по людям/источникам и overlap неизвестны | **Frame posture**, не temporal fall event. Проверить человека на кровати/снегу и бытовые отрицательные позы; opset20/896/cuts |
| [Mustafahacioglu YOLO11s forklift/person](https://huggingface.co/mustafahacioglu/yolo11s-forklift-person-detection) | `0=forklift`, `1=person`; только `best.pt` 19.19MB, ONNX ещё не опубликован | 3,928/433/221, Roboflow forklift-dsitv и map_up; scene split/overlap не установлены | Тёмный forklift против truck/excavator. Author occupancy containment не равен опасному расстоянию. Нужен безопасно полученный ONNX |
| [Beehzod best YOLO11m](https://huggingface.co/Beehzod/best_smoke_cigarette-detection) | Единственный класс **cigarette**; только `best.pt` 40.52MB | smoking-tasfx v2, 12,046/318/122; web-source split/overlap не установлены | Маленький объект + связь с человеком. Присутствие сигареты не равно smoking; YOLO11 attention/export требуют проверки |
| [Wardy TemporalFallGRU](https://huggingface.co/mumallaeng/wardy-m4-fall) | Ready ONNX132,803 B; `fall_transition`; features `[B,20,80]` → `logits` | URFD pose sequences; публичная subject-disjoint оценка не установлена; другой declared source не доказывает отсутствие копий | **10fps/2s**, COCO xy/conf/visibility +12kinematics, точная normalization. Зависит от SSDLite/фиксированной pose версии; не лечит исчезнувшего человека и не доказывает immobility |

Все полные ревизии, SHA256 и sizes доступны в JSON. PT-only кандидаты не загружались через pickle. Для SafetyVision `Person/Hardhat/NO-Hardhat/Safety Vest` требуется явный export `--class-map` в существующие runtime имена `person/helmet/no_helmet/vest` (и остальные поддерживаемые классы). Нормализация `ml/eval_ppe.py` сама по себе runtime не меняет. COCO detector сохраняет runtime имя **`cell phone`**, не произвольное `cell_phone`. Отсутствующий PPE observation остаётся unknown; его нельзя превращать в `False` или «точно без каски». Новые `Fall-Detected`, cigarette и clip actions требуют отдельного исследовательского observation/event contract.

## План контролируемых экспериментов

1. **Зафиксировать baseline и данные.** Сохранить commit/config hash, веса/классы, все исходные video hashes, decoder/preprocess, реальные timestamps, fps, thresholds и event settings. Исторические 13 роликов — полный диагностический набор, не пилот для последующего утверждения независимого качества. Добавить отдельный source/subject-disjoint holdout и проверить web/corpus overlap. До сравнения разметить actor boxes/IDs и видимость в S3; отдельно frame observations и интервалы событий.
2. **Первый запуск: D-FINE-X FP32 ONNX.** Проверить скачанные bytes/SHA, external-data ссылки, opset, домены/операторы, настоящие IO/dtypes/static dimensions. Сначала корректный direct-resize processor и source postprocess, затем одинаковые кадры/весь диагностический набор. Сравнить person recall в каждом S3 интервале, длительность потерь и FP на отрицательных сценах; полный event replay при фиксированных правилах считать отдельно. Подбор thresholds делается на dev; held-out threshold не перенастраивается. Изменение parser/адаптера должно быть отдельной фичей, оно здесь не реализовано.
3. **Изолировать вклад позы и времени.** На одинаковых GT person crops сравнить current pose, RTMPose и ViTPose teacher; затем отдельно повторить с detected crops, чтобы видеть detector bottleneck. Wardy проверять при 10fps с исходной feature schema. VideoMAE smoke/texting проверять на одинаковых временных окнах с eating/drinking/phone/hookah/dust negatives и actor-aware метками. Closed-set top1 без background не превращать в событие. Телефонную геометрию оценить на фиксированных detections до замены detector.
4. **Считать явные знаменатели.** Frame box AP/recall, keypoint visibility/accuracy, coverage пропущенных акторов и actor-matched event precision/recall/delay — разные результаты. `uncertain` и отсутствие GT не считать отрицательными. Отчёт включает все сценарии/negatives, skipped/error rows и unsupported classes; ни один провал не выпадает из знаменателя. Авторский AP не основание для promotion.
5. **Только после локального выигрыша — внешний LinQ.** Зафиксировать B1/разрешение/IO и новый graph contract, выбрать cut без YOLO node-name предположений, проверить calibration/INT8 относительно FP32 и event regression. Фиксировать compile errors вместо implicit fallback. Проверить на target-server и внешний TPU реальными per-model counters/active state/fresh outputs и end-to-end latency; PID, ONNX smoke или локальный CPU успех не доказывают TPU acceptance. CPU decode после TPU оформляется явно, CPU inference — отдельная диагностика.

Пока эта работа даёт воспроизводимый выбор артефактов и порядок проверки. Она не выбирает production default и не устанавливает преимущество какого-либо кандидата на независимых данных.
