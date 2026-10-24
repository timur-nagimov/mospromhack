# D-FINE raw ONNX: opt-in CPU adapter

`configs/models.dfine-x-640.yaml` подключает D-FINE как логическую роль `det`.
Production registry и выбор backend по умолчанию не меняются. Формат выбирается
явно через `ModelSpec.output_format: dfine`; количество классов или форма тензора
не используются для угадывания архитектуры.

Проверенный артефакт — [onnx-community/dfine_x_obj2coco-ONNX](https://huggingface.co/onnx-community/dfine_x_obj2coco-ONNX/tree/4d3a85a6c29f55a5346fabaf9728f5f3751f947a),
revision `4d3a85a6c29f55a5346fabaf9728f5f3751f947a`, файл `onnx/model.onnx`,
251 138 448 байт. Его Transformers-источник —
[ustc-community/dfine-xlarge-obj2coco](https://huggingface.co/ustc-community/dfine-xlarge-obj2coco/tree/15f18d917eaddcedf9e3ffb082adcfb97a0b2d4d).
Численная эквивалентность исходным весам отдельно не установлена.

| Файл рядом с локальным `model.onnx` | SHA256 |
|---|---|
| `model.onnx` | `486ee1cba40b25f3ea8d6783f664eca88b7c12d01078984834d36834d9b2c802` |
| `config.json` | `c3b580f1c68013a7c5737e0f2fdf6ddf3434cbb947b10d7388e0c5b652934317` |
| `preprocessor_config.json` | `cd38cd59999e7a95d68e487fbe5132df3d4e5c32a0836add57e6126ba0c4eaf1` |

Все 80 названий классов перечислены в YAML в точном порядке `config.json.id2label`:
`person` — 0, `cell phone` — 67. Встроенные метаданные YOLO и общий `det.json`
для этого профиля не нужны. Поддержки PPE-классов у него нет.

У графа opset 16, один FLOAT input `pixel_values[B,3,H,W]` и два упорядоченных
FLOAT output: `logits[B,300,80]`, затем `pred_boxes[B,300,4]`. Batch и пространственные
оси динамические; evaluation фиксирует `[1,3,640,640]`. Inspector проверяет три
SHA256, исходные processor/class contracts, имена, порядок и формы, отсутствие
custom operators/functions, external data и embedded NMS, затем ONNX checker.
Изменившийся артефакт требует отдельного проверенного pin.

Preprocessing соответствует закреплённому `RTDetrImageProcessor`: uint8 BGR кадр
превращается в RGB, PIL BILINEAR меняет его размер прямо до 640×640, далее float32
`/255`, NCHW. Aspect ratio меняется. Padding и mean/std normalization отсутствуют.
OpenCV interpolation здесь не подменяет PIL.

Postprocessing соответствует focal-loss ветке
[HF RT-DETR processor](https://github.com/huggingface/transformers/blob/main/src/transformers/models/rt_detr/image_processing_rt_detr.py):
стабильный sigmoid, top-Q по всем query/class парам, где Q=300. Один query может
дать несколько классов. При равных scores используется стабильный порядок
flattened index. Затем применяется строгий `score > conf_threshold`, normalized
cxcywh переводится в xyxy исходного кадра, координаты ограничиваются его границами.
Вырожденные boxes после clipping пропускаются. NMS отсутствует;
`iou_threshold` из общего ModelSpec на этот decoder не влияет.
NaN/Inf, неверный dtype или порядок/форма output дают явную ошибку всего кадра.
У FDR normalized — единицы координат, а не hard bounds `[0,1]`: выходной путь
проверенного ONNX заканчивается affine cxcywh без clamp/sigmoid. Конечные
off-frame/oversize proposals допускаются и проходят обычный clipping, включая
low-score queries, которые затем отбрасываются по score. Batcher проверяет
размер batch до разделения outputs.

Готовая команда для последовательного quality run из корня проекта:

```bash
.venv/bin/python -m ml.eval.run_ppe_onnx \
  --models-config configs/models.dfine-x-640.yaml \
  --model-id det \
  --models-dir models \
  --data data/model_research/construction-ppe-test/test-only \
  --output-root data/ppe_evaluation/dfine-x-640
```

Она явно выбирает `OnnxCpuBackend` с двумя threads и production `Detector`;
backend fallback не добавляется. Строгие pins/shape rules прежних PPE и YOLO
профилей сохранены. На тех же 141 изображении остаются исходный fingerprint
`800a03725b2c6fc37b8c80794735a03666fd424cc63b4eeecb54b548e37b6791`,
AP threshold .001, working point .35 и COCO 101-point AP50. В recipe значение
NMS IoU .7 остаётся как reference, а `nms_applied=false` явно указывает, что для
D-FINE оно не применяется. Это сравнение архитектур с их preprocessing/decoder,
а не weights-only контроль. Любая ошибка кадра делает весь отчёт failed без метрик.
На этапе decoder действует исходное строгое `score > .001`; working point
считается общим неизменённым evaluator по `score >= .35`. Оба сравнения записаны в recipe.
Отчёт содержит effective recipe, модель/config/processor и кодовые SHA, версии
зависимостей, все predictions, class coverage, latency и точные denominators.

Training overlap с public test неизвестен. Набор не объявляется независимым
held-out; отсутствие PPE-классов учитывается как unsupported с фиксированным
core denominator. `no_vest` GT отсутствует. S2 semantic smoke для COCO-модели
не проводится. Тесты адаптера и checker подтверждают контракт, а не качество.

При реализации прошли малые decoder/preprocessor/production-batcher и eval
регрессии, включая синтетический динамический ONNX на CPU. После объединения
реальный D-FINE прошёл141/141 images без ошибок: person AP50 .771, P .517,
R .856 при .35; у парного n .757/.736/.780. Полный readback и ограничения:
[MODEL_FOLLOWUP_20261024.md](MODEL_FOLLOWUP_20261024.md). TPU compilation, frozen artifact и
проверка SDK выполняются отдельным этапом; текущий dynamic graph не является
подтверждением TPU-совместимости.
