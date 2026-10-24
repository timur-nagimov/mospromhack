# S2: source-boundary diagnostic из cached detections

`ml.eval.replay_s2` воспроизводит S2 из сохранённых **postprocessed**
`raw_detections`. Он не читает видео/веса и не запускает ONNX/ORT. В исходных
`raw_outputs` сохранены только SHA256, dtype и shape тензоров; самих тензоров
в cache нет. Это replay tracker → attributes → rule → EventManager/Journal.
Кадр из нулей задаёт только исходные размеры 720×1280 для геометрии правил.

Continuous-control без cuts обязан пройти перед treatment. Сравниваются все
831 исходных sample: координаты, pre-attribute COCO identities, PPE attributes,
violations, подтверждения, lifecycle closes/reopens, окончательный Journal,
right-censored events и обе исходные metric scopes с classifications.
UUID сопоставляются семантически по камере и порядку создания; поля событий,
времена, boxes и детали сравниваются полностью. Исходные per-model timings
переиспользуются для точного readback confirmation snapshots. Новые
tracking/rules/processor timings описывают replay, а не inference performance.

На реальных cache continuous-control подтвердил полное совпадение:

| Кандидат | Samples по `s2_01/s2_02/clean_01/clean_02` | Primary TP / FP / FN |
|---|---|---|
| BuildVision | 300 / 185 / 172 / 174 = 831 | 5 / 0 / 1 |
| Vyra | 300 / 185 / 172 / 174 = 831 | 3 / 0 / 3 |

Raw benchmark остаётся primary. Boundary treatment — отдельная диагностика
того, как склейка независимых captures влияет на перенос tracker IDs,
временные PPE votes и cooldown/reopening событий; это не новый quality benchmark
и не основание автоматически менять candidate/threshold/rule defaults.

Граница `s2_02` доказана независимо от GT тем же imageio-ffmpeg 7.1 и точным
подготовительным filter graph, выведенным в null без encoding/inference:
part0 = 301 кадров, part1 = 253, concat = 554. Showinfo checksums соседних кадров
подтвердили переход part0 → part1 на zero-based output frame **301**.
Точное время — `301/15 = 20.066666666666666` с. Последний old sample —
index100/frame300/20.0 с; первый new sample — index101/frame303/20.2 с.
Аннотация GT 20.1 с не использовалась для выбора cut и не меняется.
Showinfo проверяет filter frames перед lossy encoding, не равенство decoded pixels.

Evidence — `data/s2_public_20261024/source-boundary-verification.json`, SHA256
`4043bf7f430b04f131a0ef065a55408dfa43317cc82a9ed3193bbb6d844f684d`.
В нём сохранены argv/filter/ffmpeg version, SHA raw/encoded/manifest/encode-report,
encoder fingerprint, integer part counts и boundary checksums. Replay проверяет
их согласованность с pinned source report. Он не выводит boundary из GT, score,
confidence, FN или визуальной scene-cut эвристики. Map может содержать один такой
evidence object либо `schema_version: 1, boundaries: [evidence, ...]`; текущий
контракт поддерживает по одной доказанной двухчастной границе на видео.

Перед первым selected sample с `source_frame_index >= cut` treatment вызывает
production `EventManager.capture_closures(camera_id)`. Для каждого открытого
incident `ended_at` равен **его последнему violation**, а не времени cut или
последнего sample. Затем closures пишутся в Journal и читаются обратно. Только
после успешной durable записи выполняется `reset(camera_id)`, создаются новые
Tracker, EventManager и instances правил. Новый source не попадает в старые
history/candidates/aliases/cooldown. Сам cut и первый новый sample записываются
отдельно. Номера track IDs внутри нового сегмента могут повторяться; различать
их нужно вместе с новой capture identity. В treatment trace и event/Journal
origin получают deterministic `capture_id` с source segment и cut frame;
исходный `source_capture_id`, `segment_index` и confirmation source-row hash
сохраняются отдельно. Continuous-control origin остаётся исходным.

Dependency — commit `5349b70` (`capture_closures` + reset contract); в replay
ветку он перенесён как `70f9495`. Production `run_s2` и defaults не изменены.

Команды из root проекта; каждый output-dir должен быть новым:

```bash
.venv/bin/python -m ml.eval.replay_s2 \
  --source-dir data/model_followup_evaluation_20261024/s2-buildvision \
  --source-report-sha256 6ad3292e3da5432dbdcd08e26bea150b0570482b92d63572d937d03c038c8c15 \
  --boundary-map data/s2_public_20261024/source-boundary-verification.json \
  --boundary-map-sha256 4043bf7f430b04f131a0ef065a55408dfa43317cc82a9ed3193bbb6d844f684d \
  --output-dir data/model_followup_evaluation_20261024/s2-buildvision-boundary-diagnostic

.venv/bin/python -m ml.eval.replay_s2 \
  --source-dir data/model_followup_evaluation_20261024/s2-vyra \
  --source-report-sha256 aca4b7b98d9676735f16f00128ee17934d4582ac5498ebf0504dca06f87d3c43 \
  --boundary-map data/s2_public_20261024/source-boundary-verification.json \
  --boundary-map-sha256 4043bf7f430b04f131a0ef065a55408dfa43317cc82a9ed3193bbb6d844f684d \
  --output-dir data/model_followup_evaluation_20261024/s2-vyra-boundary-diagnostic
```

Без двух `--boundary-*` flags выполняется только continuous-control.
Source report требует явно заданный SHA256; все обязательные cached files
проверяются по его output hashes, составу, полному sample plan, class names,
thresholds и identity fingerprints. Missing/malformed/partial cache, coordinate
drift, несогласованный map или control mismatch приводят к failed report,
без продолжения treatment. Существующий output никогда не перезаписывается.
В `source_cache/` побайтово сохраняются исходные rows, labels, camera/recipe/model
provenance и report; отдельные `continuous-control/` и `source-boundary-diagnostic/`
содержат свои traces/lifecycle/Journal. Каждый replay row сохраняет исходные
raw detections, raw-output hashes, sampling coordinates и отдельные row/detection
fingerprints. Веса, thresholds, GT, sampling и правила не меняются.

У GT нет actor IDs: интервальные TP/FN не доказывают association precision для
конкретного человека. Доказанный source cut не решает scene cuts внутри source,
смену человека при похожем bbox, ошибки PPE geometry/разметки и live reconnect
semantics. Training overlap неизвестен. Любую разницу treatment нужно читать
как diagnosis continuity policy, сохраняя исходный benchmark рядом.

## Фактический diagnostic replay — 24.10.2026

Код `8d15ec2`, API dependency включён. Оба кандидата прошли сначала полный
continuous-control, затем treatment на всех831 samples. Исходные detections,
sample coordinates и raw-output hashes сохранены. Для трёх остальных видео
точно совпали track identities, attributes, violations и counters.

| PPE | Continuous TP / FP / FN | Known-boundary diagnostic TP / FP / FN |
|---|---|---|
| BuildVision | 5 / 0 / 1 | **6 / 0 / 0** |
| Vyra | 3 / 0 / 3 | 3 / 0 / 3 |

BuildVision создаёт отдельное событие второй сцены: start20.8s, confirmed22.8s,
с новой capture identity. Старые три открытых incidents завершены по своим
last-violation timestamps перед reset. Reopening внутри нового сегмента ещё
возможно и не означает переноса из первого сегмента; total reopen count не
используется как самостоятельный критерий. У Vyra трёх исходных подтверждений
в первой сцене достаточно для прежних3/6, новая сцена не добавляет события.

Независимый `events_eval --db` точно воспроизвёл обе metric scopes в обеих
ветках обоих кандидатов; полные SQLite rows совпали с final Journal.
Capture origins сегментов сохранены в predictions и Journal. Clean остаётся
0FP на68.933333s. Это результат continuity policy на известных источниках,
первичный benchmark и выбор модели не меняются.

Отчёт с hashes и readback:
[s2-source-boundary-20261024.json](../.orchestra/metrics/s2-source-boundary-20261024.json).
Полные локальные результаты —
`data/model_followup_evaluation_20261024/s2-{buildvision,vyra}-boundary-diagnostic/`.
