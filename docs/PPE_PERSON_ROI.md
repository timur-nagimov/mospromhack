# Opt-in PPE person ROI

`GUARD_PPE_INPUT_MODE=person_crop` enables person crops in `CameraWorker`.
The default is `full_frame`: the PPE model runs once per frame, even with no
people, in the original detector order; global assignment/voting is unchanged.
The model registry, weights, confidence/IoU thresholds and rule parameters do
not change. Invalid mode values fail settings validation.

In crop mode the worker runs non-PPE detectors and COCO tracking first. The
shared `guard.pipeline.ppe.infer_ppe` then processes only observed person tracks
(`last_seen == ts`), sequentially through the **existing** PPE Detector/backend.
Each bbox gains 10% padding on every side, with floor starts, ceil ends and
clipping to the frame. The detector undoes letterboxing in crop coordinates;
the helper adds only the actual clamped crop origin. No people means no PPE
calls. No missing detection is interpreted as a negative PPE observation.

Each ROI request forces logical batch one **inside the existing model
BatchScheduler**; it cannot merge with normal or other ROI requests. Ordinary
full-frame batching is unchanged. There is no scheduler bypass, extra backend,
device session or CPU fallback. A LinQ program compiled for a larger physical
batch can still pad internally, as before; this setting does not recompile it.

PPE observations belong to the ROI owner. The existing anatomical geometry
(center x within person ±10%, class-specific y bands) is checked against **all**
people, including stale tracks. Only a unique compatible owner is accepted.
Foreign/incompatible/ambiguous observations are rejected and never transferred.
Ambiguous observations mask the affected attribute for the owner and compatible
people; this conservative mask wins even if another ROI supplied an accepted
observation. Failed, invalid and stale ROIs mask all three PPE attributes.
Masks apply after aggregation, so an old `False` cannot leak into this frame;
the private history remains available for later valid frames. Successful empty
ROIs contribute an unknown model-run observation. There is one attribute update
per frame, including the original global pose/phone assignment. The existing
2-second voting window, three-negative minimum and 60% ratio are unchanged.

Production continues other ROIs after a failure, records `infer:ppe`, and makes
affected attributes unknown. `worker.last_ppe.audit_record()` contains each
track's original bbox, actual rectangle, status, decoded crop/global boxes,
ownership decisions, per-call timing and hashes of the **actual** preprocessed
input and returned raw tensors (shape/dtype/bytes/finite included). Only hashes
and decoded boxes are retained, not input/raw tensor copies. Audit hashing adds
CPU wall time; backend inference timings remain separately available.

## Reproducible S2 diagnostic

From the checkout root, with prepared assets and a new output directory:

```bash
python -m ml.eval.run_s2 --candidate vyra --ppe-input-mode full_frame \
  --data-dir /absolute/data/s2_public_20261024 --models-dir /absolute/models \
  --output-dir /absolute/data/evaluations/vyra-full-frame --timeout 1800 --json
python -m ml.eval.run_s2 --candidate vyra --ppe-input-mode person_crop \
  --data-dir /absolute/data/s2_public_20261024 --models-dir /absolute/models \
  --output-dir /absolute/data/evaluations/vyra-person-crop --timeout 1800 --json
```

The runner uses the same helper/ownership/mask code. It records the mode,
effective policy, per-sample ROI calls and actual tensor/output hashes. It keeps
the existing COCO pre-attribute identity fingerprint and fixed source samples,
model/config/source hashes, original/scoped GT and strict event metrics. Global
and per-video denominators include eligible/attempted/succeeded/error ROI counts,
call counts, observed/stale person samples, and true/false/unknown attribute
counts. Clean FP/hour retains its separate clean-video exposure. Unknown-heavy
outputs must be assessed alongside recall and coverage, rather than presented
as an FP improvement alone. An inference/invalid-ROI failure aborts with a failed
sample audit and **no aggregate evaluation**. Zero-ROI frames are valid; PPE
backend call coverage is checked against ROI calls, while det coverage remains
one call per selected frame. Output directories are never overwritten.

These commands are the continuous S2 diagnostic. Source-boundary paired replay
is a separate runner change; old flat-PPE caches cannot represent ROI ownership
or unknown masks and must not be reused as crop outputs. No real model run or
quality improvement is established by the model-free regression tests. Crops
also change the INT8 calibration distribution; CPU correctness does not prove
external TPU compilation, TPU accuracy, actor association or production quality.

## Зафиксированный A/B результат 24.10.2026

Свежие full-frame и person-crop прогоны BuildVision используют одинаковые веса, пороги, код обработки, все четыре видео и доказанную source boundary301 (reset перед selected frame303). Это отдельный controlled protocol: прежний continuous benchmark5/6 и cached diagnostic6/6 не заменены.

| Режим | Observations | PPE calls / errors | TP / FP / FN / duplicate | Helmet unknown / observed person samples |
|---|---:|---:|---:|---:|
| Full frame | 831 | 831 / 0 | 6 / 0 / 0 / 0 | 165 / 2064 |
| Person crop | 831 | 2064 / 0 | 4 / 0 / 2 / 0 | 688 / 2064 |

Во всех831 samples совпали decoded frame hashes, координаты, COCO raw detections/output hashes и pre-attribute tracks. Независимый SQLite scorer обоих metric scopes и полный Journal readback совпали; восемь реальных preprocessing probes подтвердили input hashes без model forward.

На этом наборе person-crop ухудшил event recall, поэтому **full_frame остаётся default**, person-crop — экспериментальный opt-in. Unknown-маски сохраняют консервативную семантику; не следует снимать их ради улучшения цифры на шести метках. В одном пропущенном окне crop часто не давал отрицательной детекции каски, в другом заметен вклад ambiguous/foreign observations и unknown masks. Разбор относится к одинаковым COCO tracks, физические actor IDs не размечены.

Шесть certain helmet events и68.933333s clean exposure не доказывают общее качество или ассоциацию СИЗ с человеком. Нового TPU/INT8 подтверждения нет. Подробные pins, coverage и readback: [PPE ROI evidence](../.orchestra/metrics/ppe-roi-quality-20261024.json).
