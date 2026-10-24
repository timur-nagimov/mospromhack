# RF-DETR 2XL 880: strict static export

`tools.prepare_rfdetr_candidate` — отдельный opt-in exporter/parity tool.
Production adapter, default registry и model research catalog не меняются.
Успешный CPU export/parity не подтверждает качество модели или совместимость LinQ TPU.

Источники закреплены до исполнения:

| Артефакт | Pin |
|---|---|
| [RF-DETR source archive](https://codeload.github.com/roboflow/rf-detr/tar.gz/5b22dfc0940697ce1abdfa681a5f7392b3eddbfb) | commit `5b22dfc0940697ce1abdfa681a5f7392b3eddbfb`; 2,751,009 B; SHA256 `18e2fb180f176709070fedf03229fc5edabd3ade271d4c4fe9ca7dbefcd42368` |
| [Plus wheel 1.1.0](https://files.pythonhosted.org/packages/4d/70/d2be010436a065b3b42096da156e6fbd986fb5f5d1eaa1050147d72b01b2/rfdetr_plus-1.1.0-py3-none-any.whl) | SHA256 `bf73c8b003c0327357947f50eb678efddceeafa7350e953b764f463f68573133` |
| [Official 2XL checkpoint](https://storage.googleapis.com/rfdetr/platform-licensed/rf-detr-xxlarge.pth) | 507,831,201 B; SHA256 `bf418652d5e07ad441599acfadc65bb1767029079e7a343a6d61675ec5d553ae` |

Core source is Apache-2.0; Plus/checkpoint provenance is PML-1.0. The experiment
does not alter the application's dependency environment or deployment policy.

Bootstrap uses a **new isolated Python3.11 env**. The existing root `.venv`
can supply large libraries read-only through one `.pth` entry. The tool emits
commands; it does not silently install dependencies:

```bash
.venv/bin/python -m tools.prepare_rfdetr_candidate env-plan \
  --env-dir models/candidates/rfdetr-2xlarge-880/export-env \
  --reuse-site "$PWD/.venv/lib/python3.11/site-packages" \
  --output-dir data/model_research/rfdetr-environment-plan
```

Inspect `environment-plan.json`, then run its argv commands in order. Before
the installation command, validate the dry-run pip report:

```bash
.venv/bin/python -m tools.prepare_rfdetr_candidate check-install-plan \
  data/model_research/rfdetr-environment-plan/resolution.json
```

The constraints retain torch2.8.0, torchvision0.23.0, NumPy2.4.6, ONNX1.23.2,
ORT1.31.0 and both OpenCV4.11.0.86 distributions. A resolver plan that would
install/replace any protected library fails. New direct pins are transformers5.1.0,
supervision0.29.1 and pyDeprecate0.9.0, plus hash-pinned core/Plus artifacts.
No training/LoRA extras are requested. Keep pip's installation report (download
URLs/hashes) and `export-env/bin/python -m pip freeze` beside the environment;
the first resolution's transitive dependencies are recorded rather than claimed
to have an existing universal lock. Export/parity assert all critical versions,
PEP610 archive hashes and the exact installed source bytes for config, builder,
backbone, defaults, sparse mapping and Plus architecture. The complete visible
distribution versions are also saved. The root environment is never a pip target.

The pinned [builder](https://github.com/roboflow/rf-detr/blob/5b22dfc0940697ce1abdfa681a5f7392b3eddbfb/src/rfdetr/models/lwdetr.py)
constructs `args.num_classes + 1` logits. Bootstrap therefore calls
`RFDETR2XLargeConfig(num_classes=90, pretrain_weights=None, device='cpu',
amp=False, compile=False, resolution=880, positional_encoding_size=44, ...)`,
then `build_model_from_config` with `force_no_pretrain=True` and
`pretrained_encoder=None` in `ModelDefaults`. Exact 2XL architecture fields are
asserted: DINOv2 windowed base, hidden512, five decoder layers, heads16/32,
patch20, windows2, P4, feature indices3/6/9/12, queries300/group13.
Socket access is blocked during package import, construction, export and parity;
HF offline flags are set. Any attempted backbone/model download is an error.

The loader always uses `torch.load(weights_only=True, mmap=True, map_location='cpu')`.
Only `argparse.Namespace` is additionally allowlisted, as found in this pinned
checkpoint. No pickle/trust/unsafe-load or mmap fallback exists. The checkpoint
must contain exactly `model` and `license`, 531 tensor keys and126,903,202 elements.
Actual class head `[91,512]`/bias91, every key, shape, dtype and post-load tensor
equality are checked. Final `load_state_dict(strict=True)` must cover the model.

One explicit source-compatibility case is permitted: current source adds
`_kp_active_mask`, a deterministic empty `bool[0,0]` buffer when keypoints are
disabled. It may be derived only with exactly those assertions and zero elements.
The manifest distinguishes the **531 checkpoint keys** from this derived buffer
and resulting model state count. No learned/nonempty missing state, resizing,
truncation, remapping or random parameter fallback is allowed.

The pinned [COCO map](https://github.com/roboflow/rf-detr/blob/5b22dfc0940697ce1abdfa681a5f7392b3eddbfb/src/rfdetr/assets/coco_classes.py)
uses sparse category IDs. The manifest records all91 logit slots, all80 valid
COCO IDs, and null for unnamed slots: person=1, cell phone=77, **toothbrush=90**.
`background_class_id=null`: no last-logit removal, YOLO80 renumbering or argmax
postprocessor is inserted. Export retains raw normalized-unit cxcywh `dets`
and raw class logits `labels`; names do not mean decoded class IDs.

Execute heavy stages sequentially using the isolated interpreter. Every output
directory must be new; partial artifacts remain reviewable after failure:

```bash
models/candidates/rfdetr-2xlarge-880/export-env/bin/python \
  -m tools.prepare_rfdetr_candidate export \
  --weights models/candidates/rfdetr-2xlarge-880/rf-detr-xxlarge.pth \
  --output-dir models/candidates/rfdetr-2xlarge-880/static-fp32-880 \
  --timeout 900 --max-rss-gb 12

models/candidates/rfdetr-2xlarge-880/export-env/bin/python \
  -m tools.prepare_rfdetr_candidate parity \
  --weights models/candidates/rfdetr-2xlarge-880/rf-detr-xxlarge.pth \
  --export-dir models/candidates/rfdetr-2xlarge-880/static-fp32-880 \
  --image data/model_research/construction-ppe-test/test-only/images/test/image110.jpg \
  --output-dir data/model_research/rfdetr-2xl-native-ort-parity \
  --timeout 900 --max-rss-gb 12
```

When running from a worktree, provide absolute paths to the shared model/image
assets and desired outputs. The child runs in the tool's project root. The parent
owns a process group, watches elapsed wall time and aggregate group RSS, then
terminates/reaps it on timeout/memory breach. A partial success manifest cannot
override a watchdog failure. SIGINT/SIGTERM are deferred until Popen ownership is
known; repeated signals cannot interrupt group cleanup or the final failed report.
Interrupted runs return128+signal. Limits are operational guards, not throughput claims.

Export is CPU FP32, static input `images[1,3,880,880]`, standard opset17, outputs
`dets[1,300,4]`, `labels[1,300,91]`. The source `.export()` conversion bakes the
fixed positional grid before legacy `torch.onnx.export(dynamo=False)`. Inspector
checks exact names/shapes/dtypes/opset, ONNX checker, no external weights/custom
functions/operators/embedded NMS and FP32 floating initializers. Actual operators
and counts are recorded, including any GridSample/attention/gather operations.
An export manifest status `exported_not_parity_verified` requires the next stage.

Parity reloads the complete original model and compares its **unmodified native
eval forward** against ORT CPU for three fixed RGB inputs (black480×640,
white333×517, seeded noise721×1279) plus every required real `--image`. Both
consume the identical preprocessed tensor. Source preprocessing is RGB uint8 →
torch float `/255` → direct bilinear resize880² with `align_corners=False`,
`antialias=False` → ImageNet mean/std; no PIL resize, letterbox or normalization
substitution. Per-input source/input/output hashes and max/mean absolute error
are saved. All raw values must be finite with exact shape/dtype. Fixed tolerances
are atol1e-4/rtol1e-3, never automatically relaxed after failure. Parity requires
the ONNX/checkpoint asset to match its initial pin after all
samples; a mid-run file replacement cannot be reported as a verified new graph.
This is graph parity evidence only; quality datasets, actor GT, sparse-class runtime mapping,
TPU compiler/calibration and genuine external execution remain separate stages.

The first local export on 2026-10-24 completed in 8.76 seconds under the bounded
runner. `models/candidates/rfdetr-2xlarge-880/static-fp32-880/model.onnx` is
460,466,535 bytes, SHA256
`cd7a94b8c524905349a9639e78c72b66c04a223bbaf369c5d18811cc754aefa4`.
The manifest confirms the static contract above, 1,767 standard nodes,
406 initializers, GridSample5, LayerNormalization55 and MatMul162. All531
checkpoint tensors match after strict load; model state has532 entries because
of the one documented empty buffer. The original export tool is preserved beside
the graph as `exporter-source.py`, SHA256
`cde9774b52ae49590f4fe1d3bf3b3c6dd9c016009edfc438f985be0875b7e16f`.
The original export manifest remains `exported_not_parity_verified`; a separate
parity manifest records the later verification and its tool version.

The first primary parity run completed with **failed** status after17.63 seconds.
Its immutable output is `data/model_research/rfdetr-2xl-native-ort-parity/`.
`comparisons.json` preserves every input/output hash and numerical comparison;
the fixed tolerances were not changed. Both outputs matched on the black input.
The other cases show the following maximum absolute differences:

| Case | Boxes | Box gate | Logits | Logit gate |
|---|---:|---|---:|---|
| White | 0.827380 | failed | 1.285440 | failed |
| Seeded RGB | 0.352778 | failed | 0.581083 | failed |
| Real `image110.jpg` | 0.000735 | failed | 0.005334 | matched |

The gate uses `abs(native-ORT) <= atol + rtol*abs(ORT)` elementwise through
`numpy.allclose`; maximum absolute difference alone does not determine a pass.
The artifact has not passed native/ORT parity and is not a runtime candidate.
TopK selection sensitivity/query ordering is a hypothesis for investigation,
not an established explanation. A follow-up should compare normal native,
source export-mode native and ORT outputs on these same input hashes, with
explicitly recorded selected token indices and separate diagnostic artifacts.
It must preserve this failed primary result and its original thresholds.

Lightweight regression coverage is available with:

```bash
.venv/bin/python -m pytest -q tests/test_rfdetr_candidate.py
```

The30 tests use tiny synthetic state/graphs or fake sessions. They cover strict
missing/shape/dtype/head failures, the sole empty-buffer exception, sparse class90,
source preprocessing, offline bootstrap, bounded descendant cleanup under
repeated SIGINT/SIGTERM, and rejection of an ONNX replacement during parity.

## Реальное парное измерение 24.10.2026

Оба arms завершили141/141 изображение (236 GT person) без ошибок: native и исходный ONNX дают округлённые person AP50/P/R **0.769/0.620/0.856**. Независимая NumPy AP101 даёт0.7687680065 и0.7691719599: совпадение округлённого JSON не означает равенства raw inference. Все141 person prediction lists имеют численные различия; диагностический raw gate не проходит для boxes на121 images и logits на66. Первичный parity остаётся **failed**.

На том же фиксированном пороге0.35/IoU0.5 baseline даёт0.757/0.736/0.780, D-FINE0.771/0.517/0.856. RF получил202TP/124FP/34FN. Более высокий recall сопровождается меньшим precision относительно baseline; общего превосходства не установлено, пороги на этом test set не подбирались.

Проверены464 input/source/tensor/model assets, shared preprocessing SHA каждого изображения, NPZ↔JSONL, оба полных metric JSON и отдельный NumPy AP101/P/R пересчёт. Native/ONNX выполнялись на одном immutable tensor; второго per-arm input SHA cache не содержит. Helmet/vest не поддержаны, общий core denominator3 сохранён.

Production default сохранён. Runtime RF adapter, LinQ compilation, INT8 calibration и внешний inference не подтверждены. [Полная evidence](../.orchestra/metrics/rfdetr-quality-20261024.json).
