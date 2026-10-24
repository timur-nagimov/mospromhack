# S2 ROI A/B на доказанной границе источников

Это отдельный контролируемый протокол `ml.eval.run_s2`: сравниваются
`full_frame` и `person_crop` **с одинаковой доказанной source boundary**.
Continuous benchmark без boundary сохраняется. Добавление границы и изменение
PPE input mode нельзя смешивать в один вывод о качестве модели.

## Граница и состояние

CLI принимает `--boundary-map` только вместе с `--boundary-map-sha256`.
До импорта OpenCV/ORT/Detector общий `ml.eval.capture_boundary.load_boundaries`
проверяет SHA файла evidence, pins raw sources/encoded video/manifest/encode
report, encoder fingerprint и integer part counts/checksums. Граница задаётся
независимым encoder-part evidence, не GT, confidence или результатом детектора.
Текущий формат — одна доказанная двухчастная граница на видео; поддерживается
один evidence object либо `schema_version: 1, boundaries: [evidence, ...]`.

Для реального `s2_02` cut проходит перед zero-based frame301:
`301/15 = 20.066666666666666s`. Sampling остаётся5fps по исходной временной
шкале: последний старый sample100/frame300/20.0s, первый новый
sample101/frame303/20.2s. Ни кадры301/302 вне выборки, ни GT20.1s не становятся
дополнительными samples. Исходные GT и source-global coordinates не меняются.

Перед inference frame303 общий `close_capture` получает production
`EventManager.capture_closures`, сохраняет каждый closure и читает его обратно
из Journal. `ended_at` равен **последнему violation конкретного incident**,
а не времени границы/последнего sample. Только после подтверждённого readback
сбрасываются manager/candidates/aliases/cooldown и создаются свежие tracker и
instances правил. Ошибка записи/readback останавливает arm до reset и до
inference нового capture; failed report не содержит aggregate evaluation.

Повторное числовое значение `track_id` в новом сегменте допустимо. Его нужно
читать вместе с новым `capture_id`: effective identity включает source segment
и cut frame. Сохраняются `source_capture_id`, `segment_index`, evidence hash и
original source position. Report содержит `capture_boundary_protocol`,
per-video `boundary_resets` и число закрытых на границе incidents. При включённом
протоколе каждый sample всех четырёх видео также сохраняет SHA256 decoded BGR
frame, shape/dtype/bytes и описание byte layout. Это позволяет проверить, что
обе arms получили одинаковые pixels до crop/preprocessing.

Без `--boundary-*` прежний `full_frame` путь сохраняет исходную capture identity
и continuous temporal state; boundary-поля не добавляются.

## Воспроизводимый парный запуск

Каждая arm использует новый output directory, тот же финальный checkout,
BuildVision weights/metadata/registry, COCO detector, ORT/provider/threads,
frozen recipe, GT, source bytes, decoded samples и boundary evidence. Параметры
голосования, confidence/IoU, tracker и rule duration не подбираются по GT.
Запускать последовательно; длительность ограничена общим timeout.

```bash
python -m ml.eval.run_s2 --candidate buildvision --ppe-input-mode full_frame \
  --data-dir /absolute/data/s2_public_20261024 --models-dir /absolute/models \
  --boundary-map /absolute/data/s2_public_20261024/source-boundary-verification.json \
  --boundary-map-sha256 4043bf7f430b04f131a0ef065a55408dfa43317cc82a9ed3193bbb6d844f684d \
  --output-dir /absolute/data/evaluations/s2-boundary-full-frame --timeout 1800 --json

python -m ml.eval.run_s2 --candidate buildvision --ppe-input-mode person_crop \
  --data-dir /absolute/data/s2_public_20261024 --models-dir /absolute/models \
  --boundary-map /absolute/data/s2_public_20261024/source-boundary-verification.json \
  --boundary-map-sha256 4043bf7f430b04f131a0ef065a55408dfa43317cc82a9ed3193bbb6d844f684d \
  --output-dir /absolute/data/evaluations/s2-boundary-person-crop --timeout 1800 --json
```

Evidence SHA выше относится к существующей подготовке; при другой подготовке
нужно её собственное проверенное evidence. Сравнивать можно только завершённые
arms с полной coverage: реальные300/185/172/174 = **831samples**, одинаковые
source coordinates/frame hashes и COCO pre-attribute identities. Сохраняются
per-ROI owner/mask/per-call tensor audit; flat PPE boxes не заменяют эти данные.
Семантика final Journal/lifecycle сверяется с учётом различных event UUID.
Файлы timings и per-call PPE hashes/число вызовов между режимами могут различаться.

## Когда допустим cached full-frame control

Кэш пригоден только при совпадении producer pins исполняемого detector,
preprocessing/decoding, tracker/attributes/rules/manager/journal, backend/ORT,
model bytes/metadata/class mapping, frozen recipe/GT, исходных videos и sample
plan. Нужен полный exact continuous-control replay перед применением общей
границы. Наличие успешного старого report само по себе недостаточно. Любой
producer drift означает **fresh full-frame A**; нельзя объявлять такой cache
эквивалентным новому inference на основании одних event metrics.

В текущей итерации SHA `guard/pipeline/detector.py` отличается от исторического
producer. Поэтому новый A/B требует fresh full-frame A и fresh real person-crop
B, последовательно на одном финальном коде; ROI arm должна реально обработать
все831samples. Legacy `ml.eval.replay_s2` явно отвергает `person_crop` в report
или sample: старый flat cache не содержит owner-bound observations и masks.

## Интерпретация и границы доказанного

Исторический continuous BuildVision результат **5TP/0FP/1FN** не изменяется.
Его отдельная cached boundary-диагностика уже дала **6TP/0FP/0FN** на тех же
шести certain intervals. Это event-recall ceiling для текущих данных и
протокола, а не результат нового ROI A/B. Все старые артефакты сохраняются.
История и evidence описаны в [source-boundary diagnostic](S2_SOURCE_BOUNDARY_DIAGNOSTIC.md).

При таком ceiling ROI может не добавить TP. Нужно отдельно сравнить strict
precision с duplicates, FN, confirmations/latency, duplicate/uncertain IDs,
true/false/unknown PPE attribute counts и observed-person coverage. Снижение FP
при росте `None` без сохранения recall/coverage не доказывает улучшения. Clean
FP/hour использует только два явно размеченных clean videos и их фактическую
duration; positive-video exposure и общий alarm rate показываются отдельно.

У GT нет actor IDs; три certain `s2_02` intervals совпадают по time/type и
отличаются notes. Event6/6 не доказывает правильную привязку PPE к каждому
человеку. Нет независимого production test set; training overlap неизвестен.
Unit acceptance использует synthetic pixels/detections, реальные stateful
компоненты и SQLite, но не pretrained inference. Она доказывает reset и
durability ordering, не качество модели. ROI меняет calibration distribution:
даже успешный CPU A/B не доказывает INT8/external-TPU parity, compilation или
производительность на устройстве. Defaults автоматически не переключаются.
