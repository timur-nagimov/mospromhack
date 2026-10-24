# RF-DETR 2XL: paired quality на frozen141

`tools.evaluate_rfdetr_candidate` сравнивает качество **обычного native PyTorch**
и **исходного ONNX** на одном Construction-PPE test snapshot. Это отдельная
оценка качества, а не замена численного gate: первичный native/ORT parity остаётся
**failed**. Нет production adapter/default switch, CPU fallback или LinQ TPU claim.

Контракты загрузки/экспорта находятся в [RFDETR_CANDIDATE.md](RFDETR_CANDIDATE.md),
численная диагностика — в [RFDETR_PARITY_DIAGNOSTICS.md](RFDETR_PARITY_DIAGNOSTICS.md).
Граф сохраняет SHA256
`cd7a94b8c524905349a9639e78c72b66c04a223bbaf369c5d18811cc754aefa4`,
460,466,535 bytes. Debug-копии, повторный экспорт и оптимизированные графы
не принимаются. Checkpoint закреплён SHA256
`bf418652d5e07ad441599acfadc65bb1767029079e7a343a6d61675ec5d553ae`;
используется существующий strict weights-only loader с полным531-key readback
и единственным документированным empty bool buffer. Native модель никогда
не вызывает `.export()` и проверяется на normal eval перед каждым inference.

Входы читает **существующий** `ml.eval.run_ppe_onnx.dataset_inputs`, а метрики
вычисляет **существующий** `ml.eval_ppe.evaluate_predictions`. AP не реализуется
повторно. Протокол фиксирован до запуска:

| Параметр | Значение |
|---|---|
| Split / scope | ровно141 test images, общие для двух arms |
| Dataset SHA256 | `800a03725b2c6fc37b8c80794735a03666fd424cc63b4eeecb54b548e37b6791` |
| GT person | 236 |
| Остальной GT | helmet192, vest178, no_helmet40, no_vest0 |
| AP | COCO101 AP50, IoU0.5, confidence floor `>=0.001` |
| Рабочая точка | confidence `>=0.35`, IoU0.5 |
| CPU | Torch2 threads; ORT intra2/inter1, CPU provider only |
| Input | один общий float32 tensor `[1,3,880,880]` на исходное изображение |
| Postprocessing | один pinned официальный RF PostProcess для обоих arms; top300; без NMS |

Dataset loader проверяет официальный class order, labels и bytes fingerprint.
Дополнительно проверяются уникальные image keys/paths/stems/content SHA, область
GT, положительная геометрия и отсутствие повторных GT boxes. Неоднородные или
ошибочные исходные GT не исправляются. RGB decoder отклоняет nontrivial EXIF
orientation, чтобы не менять координаты относительно frozen GT.

Каждый RGB кадр обрабатывается ровно один раз существующим
`prepare_rfdetr_candidate.preprocess_rgb`: float32 `/255`, direct Torch bilinear
resize880² (`align_corners=False`, `antialias=False`), ImageNet mean/std, без
letterbox/PIL resize. Native получает этот tensor, ORT — его NumPy view. SHA
проверяется перед/после каждого runtime; mutation запрещает второй arm получить
изменённый input и завершает сравнение ошибкой. Внутри каждого кадра порядок —
native, затем ONNX; native состояние не переключается в export mode.

Оба результата декодирует **один экземпляр** официального
[PostProcess на закреплённом source revision](https://github.com/roboflow/rf-detr/blob/5b22dfc0940697ce1abdfa681a5f7392b3eddbfb/src/rfdetr/models/postprocess.py).
Его SHA256 `86dc3aaa85283da17c36b0263473e8e2db6225eed90bd91920a6ffeb27c04e0e`,
box_ops SHA256 `b71ae71cd71889f9207374edc4807532494c6f76d31aae973388407933cd67c0`;
bytes проверены по source archive. Логика: sigmoid всех91 logits, stable sort
по descending score/ascending flattened query-class index, top300 пар,
cxcywh→xyxy, масштабирование в исходный размер и clip. Softmax, last-logit drop,
single-label argmax и YOLO NMS не добавляются. Zero-area clipped boxes не
выкидываются: остаются результатом официального postprocessor.

Sparse COCO map сохраняется полностью: **person=1, cell phone=77, toothbrush=90**.
Null slots неизвестны и не становятся person/background. Все91 logits/sigmoid
scores сохраняются. Только canonical person участвует в этом PPE benchmark;
phone и остальные COCO классы остаются в predictions и не имеют GT этого теста.
GT helmet/vest/no_helmet моделью не поддерживается. Поэтому общий
`core_ap50(person,helmet,vest)` имеет фиксированный знаменатель3 и включает нули
для helmet/vest; person AP не является общим PPE quality superiority.
Для отсутствующего GT no_vest возвращается null.

Каждый arm получает исходные raw float32 boxes/logits с контрактом
`[1,300,4]`/`[1,300,91]`; неверные shape/dtype/nonfinite значения — ошибка, а не
пустая детекция. Для всех141 кадров создаётся одна строка `images.jsonl` с
input/raw tensor SHA, статусом каждого arm, timing/error и всеми300 выбранными
predictions. `sample-000.npz` … сохраняют raw boxes/logits, все sigmoid scores,
decoded boxes/scores/native labels **обоих** arms; чтение — `allow_pickle=False`.
При сбое сохраняются доступные numeric raw arrays и явная ошибка. Ни один arm
не получает aggregate metrics, если был пропуск/дубликат/error хотя бы одного
кадра. Failed arm не заменяется другим runtime.

`inputs.json` содержит исходный file manifest, image-key fingerprint и полный
GT с отдельным fingerprint. `manifest.json` хранит общий recipe, coverage,
метрики/знаменатели, per-arm completed/attempted counts, версии, source archive,
Plus wheel, weights/graph/primary-failure refs и все code/data SHA. Проверка
before/after включает model, data, metrics, preprocessing/postprocessing и
helpers. Latency arm означает только runtime inference; общие decode/preprocess,
postprocessing и запись cache исключены. Это не production throughput benchmark.

До heavy stage в `code-sources/` автоматически архивируются exact evaluator,
RF/diagnostic/PPE helpers, frozen141 loader и metric engine. Исходный running
manifest уже содержит failed numeric parity/no promotion даже при прерывании.
Переиспользуется прежний watchdog `bounded_run`: новая output directory,
wall/RSS limits, защищённый SIGTERM/repeated SIGINT cleanup всей process group.

Запускайте последовательно из checkout этой фичи и существующего
изолированного export env; root `.venv` не меняется:

```bash
/Users/mikemoiseev/Documents/mospromhack/models/candidates/rfdetr-2xlarge-880/export-env/bin/python \
  -m tools.evaluate_rfdetr_candidate \
  --data /Users/mikemoiseev/Documents/mospromhack/data/model_research/construction-ppe-test/test-only \
  --weights /Users/mikemoiseev/Documents/mospromhack/models/candidates/rfdetr-2xlarge-880/rf-detr-xxlarge.pth \
  --export-dir /Users/mikemoiseev/Documents/mospromhack/models/candidates/rfdetr-2xlarge-880/static-fp32-880 \
  --primary-parity-dir /Users/mikemoiseev/Documents/mospromhack/data/model_research/rfdetr-2xl-native-ort-parity \
  --output-dir /Users/mikemoiseev/Documents/mospromhack/data/model_research/rfdetr-2xl-quality-141 \
  --timeout 900 --max-rss-gb 12
```

Публичный test split не доказан независимым от обучения автора. Нет подбора
порогов, GT или сцен. Native/ONNX comparison сохраняет веса/разрешение/
postprocessing; сравнение с YOLO/PPE моделями меняет архитектуру и source recipe,
поэтому не является weights-only S1. Даже совпавшие quality metrics не отменяют
failed numeric parity. Настоящий 141-image inference запускается отдельно;
эта фича подготовлена и проверена на synthetic/fake outputs без RF/ORT inference.

```bash
.venv/bin/python -m pytest -q tests/test_rfdetr_quality_comparison.py \
  tests/test_rfdetr_candidate.py tests/test_rfdetr_parity_diagnostics.py tests/test_ppe_eval.py
```

## Реальное парное измерение 24.10.2026

Оба arms завершили141/141 изображение (236 GT person) без ошибок: native и исходный ONNX дают округлённые person AP50/P/R **0.769/0.620/0.856**. Независимая NumPy AP101 даёт0.7687680065 и0.7691719599: совпадение округлённого JSON не означает равенства raw inference. Все141 person prediction lists имеют численные различия; диагностический raw gate не проходит для boxes на121 images и logits на66. Первичный parity остаётся **failed**.

На том же фиксированном пороге0.35/IoU0.5 baseline даёт0.757/0.736/0.780, D-FINE0.771/0.517/0.856. RF получил202TP/124FP/34FN. Более высокий recall сопровождается меньшим precision относительно baseline; общего превосходства не установлено, пороги на этом test set не подбирались.

Проверены464 input/source/tensor/model assets, shared preprocessing SHA каждого изображения, NPZ↔JSONL, оба полных metric JSON и отдельный NumPy AP101/P/R пересчёт. Native/ONNX выполнялись на одном immutable tensor; второго per-arm input SHA cache не содержит. Helmet/vest не поддержаны, общий core denominator3 сохранён.

Production default сохранён. Runtime RF adapter, LinQ compilation, INT8 calibration и внешний inference не подтверждены. [Полная evidence](../.orchestra/metrics/rfdetr-quality-20261024.json).
