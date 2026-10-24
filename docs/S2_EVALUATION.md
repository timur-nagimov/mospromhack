# Фиксированный S2 event baseline: отсутствие каски

`python -m ml.eval.run_s2` — локальный CPU diagnostic для четырёх полных
подготовленных роликов: `s2_01.mp4`, `s2_02.mp4`, `clean_01.mp4`, `clean_02.mp4`.
Выполняется последовательная production-цепочка `Detector` → COCO person
`Tracker` → `assign_attributes` → `ppe_missing` → `EventManager` → новая
`Journal` DB. Async/live/multicamera и external TPU этим runner не проверяются;
production target внешнего TPU требует отдельной проверки backend и модели.
Пропуски кадров ради скорости отсутствуют. CPU throughput — справочная метрика,
отдельное исполнение модели допускает приоритет качества.

Реальный inference, скачивание и кодирование не входят в реализацию/unit tests
этой фичи. Факт подготовки registry/runner не доказывает качество модели.
Последующие реальные результаты следует читать из нового полного report.

## Данные и обязательный preflight

Исходный manifest — `tools/data_manifest.json` из Git
`79a8920d49f0de35cc37d7deb4ed92e08c1af927`; snapshot SHA256:
`06642131278f2b484b3c7b859600f1d4e55c95676b8e51e26481da019e848449`.
Фиксированный рецепт находится в `configs/eval-s2.json`.

Подготовленный dataset root передаётся через `--data-dir`, например
`data/s2_public_20261024`. В нём обязательны:

- `manifest.snapshot.json` с указанным hash/Git pin.
- `raw/<source-id>.mp4`: пять оригинальных Intel/Mixkit source файлов.
- `download-report.json`: `{manifest_git, manifest_sha256, sources:[{id,path,bytes,sha256},...]}`.
- `videos/<clip>.mp4`: все четыре CFR 15 FPS, 1280×720 файла.
- `labels/<clipstem>.json`: `{video, events, ...}` для **каждого** ролика, включая
  два явно аннотированных clean файла с `events: []`. Отсутствующая разметка
  не считается отрицательным примером.
- `encode-report.json`: схема ниже.

```json
{
  "schema_version": 1,
  "manifest_sha256": "<snapshot SHA256>",
  "source_manifest_git": "79a8920d49f0de35cc37d7deb4ed92e08c1af927",
  "source_report_sha256": "<download-report.json SHA256>",
  "outputs": {
    "s2_01.mp4": {
      "relative_path": "videos/s2_01.mp4",
      "sha256": "<actual encoded bytes SHA256>",
      "bytes": 1,
      "frames": 900,
      "fps": 15.0,
      "duration_s": 60.0,
      "width": 1280,
      "height": 720
    }
  },
  "tool_versions": {"opencv": "<version>", "imageio_ffmpeg": "<version>"},
  "encoder": {"manifest_encode": {}, "ffmpeg_version": "<text>"}
}
```

Пример показывает один элемент; `outputs` обязан содержать все четыре.
`bytes`, SHA и frame counts записываются фактически после подготовки, не
копируются из примера. Duration каждого клипа равна `frames/fps`; округлённая
длительность из описания GT не используется. `s2_02` содержит склейку двух
источников; mapping сохраняет исходные manifest segments и отдельное время
подготовленного clip, начиная с нуля.

До ORT проверяются hash/size всех пяти raw sources по pinned manifest,
неизменность download/encode reports, hash/size всех encoded clips, все
original label events/notes и модельные файлы. Затем OpenCV metadata всех
четырёх файлов сверяется с frozen frame counts, FPS и размером **до** создания
ORT sessions. В inference-loop проверяется каждый source frame до полного EOF:
ранний EOF, лишний кадр, неуспешный selected decode и model/rule/storage ошибки
прерывают run. Частичный run не получает успешные метрики/negative exposure.

Выборка: первый source frame не раньше каждого фиксированного слота `n/5`;
source timestamp = `frame_index/native_fps`, `t0=0`. Все клипы проходятся целиком;
ограничения 60 секунд для этого runner нет. После EOF не добавляются пустые
кадры для подтверждения/закрытия событий.

## Модели и параметры

Baseline фиксирует COCO YOLOv8n-640 det SHA256
`aa9699693737742033bc899670016413e1c3f9c2725b5fd0b302cfda44ce3a99`,
input float32 640×640, conf `.35`, IoU `.5`. Person tracks строятся только из
COCO `det`; класс `person` PPE-модели не подменяет этот источник трекинга.

PPE выбирается явно:

| CLI candidate | Opt-in registry | PPE ONNX SHA256 |
| --- | --- | --- |
| `buildvision` | `configs/models.ppe-candidate.yaml` | `55f4dd2f53158eac4ff093d938b9fbda80c99454a03672145bbc36f539988249` |
| `vyra` | `configs/models.ppe-vyra.yaml` | `b99fed37bae1d111ddb622a0ef9077d42918a4daebd4a0bffbd8faaba273f246` |

Обе PPE модели: float32 640×640, batch 1, conf `.35`, IoU `.7`, 14 классов с
явным mapping в canonical `helmet/no_helmet/vest/no_vest/person`.
Registry выбранного кандидата должен быть доступен в исполняемом checkout;
отсутствие Vyra registry до интеграции соответствующей ветки — явная ошибка,
fallback на другую модель запрещён. Проверяются candidate SHA, filename,
sidecar classes, source/revision, checker status, input/output tensor metadata.
`ppe.json` берётся рядом с PPE ONNX, `det.json` — из models root.

Штатные rule defaults сохранены: `min_duration_s=2`, `min_ratio=.7`,
`close_after_s=3`, `cooldown_s=30`, severity high, `min_box_h_px=60`.
Единственный scope override — `required=[helmet]`, весь кадр без зон.
Main recipe с tuned `min_duration_s=1.5` не используется. Tracker:
high `.4`, low `.1`, new `.45`, match IoU `.25`, buffer 3 секунды, min hits 2.
`assign_attributes` использует штатное двухсекундное голосование PPE; его
unknown/None не превращается в отсутствие каски.
Точная temporal semantics задаётся исполняемым merged source snapshot и его
content fingerprint: интеграция CX-016 добавляет минимум три negative votes и
60% всех PPE model-run observations для False, включая пустой run как unknown.
Runner это поведение не переопределяет и не ослабляет ради still-image smoke;
прогоны разных source snapshots нельзя считать чистым сравнением моделей.

## GT scopes и знаменатели

Исходный GT сохраняется без исправлений: **6 certain no-helmet событий и
2 uncertain vest-only события**. `original_gt/` содержит byte copies четырёх
label файлов, `original_labels.json` сохраняет stable label IDs и все notes.

Primary `helmet_only` исключает ровно две метки:

- `s2_01.mp4:2`: мужчина с каской без жилета, интервал 37.5–46.5.
- `s2_02.mp4:3`: рабочий с оранжевой каской без жилета, интервал 0–20.1.

Для исключения рецепт фиксирует video/index, canonical original label SHA256,
полный note и rationale. Любое расхождение прерывает run. В primary остаётся
6 certain и 0 uncertain; vest-only uncertainty не может исключать helmet FP.
Secondary `original_legacy` отдельно оценивает неизменённые 6+2 исходные метки.
Оба scope используют только существующий `ml.eval.events_eval`; новый matcher
не вводится. Полученные TP/FP/FN/duplicates/uncertain IDs и reasons сохраняются.

Original precision исключает duplicates/ignored uncertain. Дополнительно
strict precision = TP/(TP+FP+duplicates), с явными raw denominators и частотой
`(FP+duplicates)/evaluation_exposure_hour`. Отдельно приводятся positive-only
метрики и **clean-only FP/hour**, где знаменатель — сумма полностью пройденных
`clean_01 + clean_02` длительностей, а не общая positive exposure. На clean
helmet GT пустой явно: все созданные PPE события там являются FP.

Actor IDs отсутствуют. Три certain метки `s2_02` совпадают по type/time 0–20.1
и различаются только notes; существующий greedy event matcher способен
сопоставить их по времени, но не устанавливает правильность person/PPE
association. Считать event TP доказательством actor-level качества нельзя.
Совпадение training data с этими публичными видео не проверено.

## Запуск и артефакты

Из корня checkout после подготовки данных и моделей, отдельно от других
тяжёлых jobs:

```bash
python -m ml.eval.run_s2 --candidate buildvision \
  --data-dir "$PWD/data/s2_public_20261024" \
  --models-dir "$PWD/models" --timeout 1800 --json

python -m ml.eval.run_s2 --candidate vyra \
  --data-dir "$PWD/data/s2_public_20261024" \
  --models-dir "$PWD/models" --timeout 1800 --json
```

В worktree используйте абсолютные пути общих data/models. CLI defaults:
dataset `GUARD_DATA_DIR/s2_public_20261024` либо `checkout/data/s2_public_20261024`,
models `GUARD_MODELS_DIR` либо `checkout/models`; timeout 900 секунд, максимум
1800. Новый output по умолчанию создаётся в dataset `evaluations/<UTC>-<candidate>-<uuid>`;
существующий `--output-dir` отклоняется. DB/results/weights находятся в ignored
data/models и не добавляются в Git.

ORT telemetry отключена, CPU provider, два ORT/OpenCV/BLAS потока, batch 1.
Родительский watchdog владеет process group и очищает потомков после grace
посредством TERM/KILL независимо от завершения leader. Exit 0 означает полный
валидный baseline, а не прохождение заранее выбранного quality threshold;
1 — runtime/coverage/provenance ошибка или timeout; 2 — неверный вызов.

`report.json` включает frozen recipe, effective specs, input bytes/mtime/hash,
полные prep reports, Git и версии, content fingerprint всех guard/ml Python
модулей до/после каждого видео и в финале, coverage/durations, metrics scopes,
per-model error/call/item/batch статистику и output hashes. На ошибке сохраняется
failed/timeout report с достигнутым coverage; успешные метрики не выдумываются.

`samples_<clip>.jsonl` содержит source coordinates/time, оба postprocessed raw
detection набора, COCO tracks **до** attribute assignment, PPE attribute votes/
smoothing state, violations, event IDs, per-model/processor timings. Для raw
ORT tensors сохраняются shape/dtype/bytes/finite/SHA256, массивы целиком не
дампятся. Non-finite outputs — ошибка. Backend timing summary может быть
rolling окном; полный набор покадровых timings находится в traces.

Per-sample `identity_sha256` и per-video/global identity fingerprints зависят
только от source coordinates и COCO pre-attribute tracks; PPE outputs/attributes
не входят. Их можно сверить между BuildVision/Vyra runs вместе с config/data
hashes, чтобы доказать одинаковый вход person tracking. Это не actor GT.
`lifecycle_<clip>.jsonl`, `predictions.json`, `journal_events.json` сохраняют
создание, закрытие и переоткрытие с исходным ID; open events на EOF отмечены
как right censored и принудительно не закрываются.
