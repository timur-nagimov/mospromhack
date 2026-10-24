# RF-DETR, DEIMv2 и Grounding DINO — следующий эксперимент

Следующий тяжёлый кандидат — **RF-DETR 2XL 880**, с текущим RF-DETR Large2026
704 как контролем. Выбор основан на доступном артефакте и другой архитектуре;
преимущество на наших событиях ещё предстоит измерить. Скорость и лицензия
не ограничивают этот личный эксперимент. Production-модели выбираются по
качеству и подтверждённому выполнению на внешнем LinQ TPU.

Каталог расширен с9 до **13** кандидатов:
[model-research-catalog.json](../configs/model-research-catalog.json).
Отчёт с source pins и фактическим checkpoint inspection:
[extended-model-families-20261024.json](../.orchestra/metrics/extended-model-families-20261024.json).
RF-DETR2XL уже экспортирован и прошёл четыре native/ORT сравнения, однако
фиксированный численный паритет **не пройден**. Остальные новые веса пока
не запускались. Подробности: [RFDETR_CANDIDATE.md](RFDETR_CANDIDATE.md),
[rfdetr-export-20261024.json](../.orchestra/metrics/rfdetr-export-20261024.json).

| Семейство | Доступный артефакт | Следующая проверка |
|---|---|---|
| RF-DETR **2XL880** | Официальный checkpoint507,831,201B; payload и static export проверены, parity failed | Отдельная A/B/C+TopK диагностика, затем quality evaluation |
| RF-DETR **Large2026 704** | Официальный checkpoint135,954,129B; HTTP200 и metadata | Парный контроль того же поколения; payload hash ещё не получен |
| DEIMv2-X / DINOv3 | Официальный safetensors205,313,972B, закреплены revision и ожидаемый SHA256 | Строгая загрузка полного state, validation processor и static export |
| Grounding DINO base | Официальный safetensors933,400,872B, закреплены revision и ожидаемый SHA256 | Teacher-пробы forklift/helmet/phone с ручной проверкой фраз и ассоциации |

## RF-DETR: фактическая голова важнее package defaults

Исходник RF-DETR закреплён на
[`5b22dfc0940697ce1abdfa681a5f7392b3eddbfb`](https://github.com/roboflow/rf-detr/tree/5b22dfc0940697ce1abdfa681a5f7392b3eddbfb).
Plus wheel1.1.0 прочитан как ZIP без установки/исполнения, SHA256
`bf73c8b003c0327357947f50eb678efddceeafa7350e953b764f463f68573133`.
Его source и registry pins сохранены в локальном evidence JSON.

Фактически полученный [официальный RF-DETR 2XL checkpoint](https://storage.googleapis.com/rfdetr/platform-licensed/rf-detr-xxlarge.pth):

- 507,831,201bytes; SHA256 **`bf418652d5e07ad441599acfadc65bb1767029079e7a343a6d61675ec5d553ae`**;
- MD5 `e3204689c1f0280427e4c33e6a2ac6cd` совпал с registry/HEAD;
- `torch.load(weights_only=True, mmap=True, map_location='cpu')` прошёл под heavy lock;
  единственная разрешённая дополнительная структура — `argparse.Namespace`;
- checkpoint содержит `model` и `license`; tensor elements включая buffers — **126,903,202**;
- фактический `class_embed.weight` — **[91,512]**, bias[91]; такие же91 logits
  у encoder class heads. Package default `num_classes=365` не описывает эти bytes.

Это inspection state_dict, без native forward, graph export или quality metric.
Голова91 сама по себе не устанавливает label map. Нужна явная source sparse
COCO mapping: перенос YOLO80 index или удаление последнего logit без контракта
недопустимы. Constructor должен восстановить именно полученный head и проверить
полное покрытие ключей. Артефакты находятся в
`models/candidates/rfdetr-2xlarge-880/{source,inspection}.json`.

После этого checkpoint inspection выполнен настоящий export: static FP32
`images[1,3,880,880] → dets[1,300,4], labels[1,300,91]`, opset17,1767nodes,
460,466,535bytes, SHA256
`cd7a94b8c524905349a9639e78c72b66c04a223bbaf369c5d18811cc754aefa4`.
Checker прошёл, external/custom operators отсутствуют. Native strict load
проверил все531 checkpoint tensors; единственное дополнение — stateless empty
bool[0,0] buffer выключенной keypoint-ветки. Actual graph содержит GridSample5,
MatMul162 и LayerNormalization55; LinQ не проверялся.

Первичный parity при фиксированных `atol=1e-4, rtol=1e-3` завершился failed:
black matched, white/seeded failed, на realimage110 boxes failed, logits matched.
Все четыре input hashes и сравнения сохранены. Для реальной картинки max box
residual0.000735, для white0.827. Возможное влияние TopK на background — гипотеза;
новая A/B/C диагностика проверяет её отдельно. Raw-order gate и исходный failed
report сохраняются, модель не promoted. Event quality этой модели не измерено.

Текущий Large2026 — отдельный
[rf-detr-large-2026.pth](https://repo.roboflow.com/rfdetr/rf-detr-large-2026.pth),
small backbone,704px,300queries. Legacy `rf-detr-large.pth` — другое поколение;
опубликованные цифры старого Large129M нельзя смешивать с текущим Large33.9M.

Закреплённый [export preprocessing](https://github.com/roboflow/rf-detr/blob/5b22dfc0940697ce1abdfa681a5f7392b3eddbfb/src/rfdetr/export/_runtime/preprocess.py):
RGB float `/255`, direct torch bilinear resize `align_corners=False`,
`antialias=False`, ImageNet mean/std. PIL BILINEAR не является точной заменой.
Ожидаемый [ONNX contract](https://github.com/roboflow/rf-detr/blob/5b22dfc0940697ce1abdfa681a5f7392b3eddbfb/docs/exports/onnx.md):
`dets` — normalized cxcywh, `labels` — logits. Имена не означают готовые IDs.
Sigmoid/global query×class top-k, без YOLO NMS. StaticB1/opset17 — план экспорта,
фактический IO/opset/operators потребуют инспекции полученного графа.

## DEIMv2-X: другой processor и export contract

[Официальный HF safetensors](https://huggingface.co/Intellindust/DEIMv2_DINOv3_X_COCO/tree/b1d81486dce5686b2afcd38a231d086d142c996d):
revision `b1d81486dce5686b2afcd38a231d086d142c996d`, SHA256
`c15e1860118e6841f8f803219507dbfb630c2aa0164ab723c339b94e554bae7f`.
Эти payload bytes пока не скачаны. Source pin
[`f005f50ce565e63f6c298feccb51901a98c3d035`](https://github.com/Intellindust-AI-Lab/DEIMv2/tree/f005f50ce565e63f6c298feccb51901a98c3d035).

Validation recipe использует direct resize640² и ImageNet normalization;
demo использует letterbox. Это разные рецепты, выбор фиксируется до оценки.
[Официальный exporter](https://github.com/Intellindust-AI-Lab/DEIMv2/blob/f005f50ce565e63f6c298feccb51901a98c3d035/tools/deployment/export_onnx.py)
принимает `images, orig_target_sizes`, возвращает `labels, boxes, scores`,
оставляет dynamic batch и допускает random initialization без checkpoint.
Наш эксперимент должен требовать полный strict safetensors load и staticB1.
Копировать D-FINE adapter с RGB/255 и raw logits/boxes сюда автоматически нельзя.
Конечная COCO голова не добавляет forklift или PPE-классы от одного pretraining.

## Grounding DINO: отдельные open-vocabulary наблюдения

[Grounding DINO base](https://huggingface.co/IDEA-Research/grounding-dino-base/tree/12bdfa3120f3e7ec7b434d90674b3396eccf88eb)
закреплён на `12bdfa3120f3e7ec7b434d90674b3396eccf88eb`, safetensors SHA256
`5548f844c928c4b6f411fa8cbcc2bfa8dbbba437cb1d513975519f93c2a9ed21`.
Payload пока не скачан. Это кандидат для проверки forklift/PPE/phone и подготовки
ручной разметки. Teacher predictions не становятся готовым ground truth.

Processor включает shortest800/max1333, padding, ImageNet normalization,
tokenizer и lowercase prompts с точками. Голова выдаёт900boxes и token logits,
а не COCO class IDs. В закреплённом
[reference Transformers](https://github.com/huggingface/transformers/blob/8cb5963cc22174954e7dca2c0a3320b7dc2f4edc/src/transformers/models/grounding_dino/modeling_grounding_dino.py#L1349)
masked/padded token logits намеренно равны `-inf`. Проверка обязана учитывать
text mask: boxes и valid logits finite; NaN/+inf и неожиданные masked values
запрещены. Текущий blanket `isfinite(raw_outputs)` не переносится.
Prompt «человек без каски» требует собственной проверки; отсутствие найденной
каски остаётся `unknown`.

## Порядок дальнейшей проверки

1. Завершить разделение capture sessions и проверить replay сохранённых S2
   detections, чтобы новая модель не выигрывала за счёт чужого старого события.
2. RF-DETR2XL: strict constructor/head91/map → static FP32 export → одинаковые
   synthetic/real native-vs-ORT inputs → фактический graph inventory.
3. На фиксированных141 images и видео сравнить person misses, ID switches,
   PPE attribution и события. Все пороги/источники/frames зафиксировать;
   public overlap и отсутствие actor GT явно сохранить в ограничениях.
4. Для каждого потенциального runtime-кандидата отдельно проверить LinQ
   compiler operators, calibration/INT8 и настоящее выполнение на внешнем TPU.
   Attention, GridSample, LayerNorm, gather/top-k и positional operations пока
   являются рисками по source, а не результатом компиляции. Успех CPU или
   наличие checkpoint не подтверждают TPU. Неявный CPU fallback не добавляется.
