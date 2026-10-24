# RF-DETR: диагностика native/export/ORT

`tools.diagnose_rfdetr_candidate` запускается отдельно от основного exporter.
Он исследует причину расхождения, сохраняя исходный failed gate, его четыре
входа и допуски `atol=1e-4`, `rtol=1e-3`. Завершённая диагностика имеет статус
`diagnostic_complete` и `primary_parity_status=failed`. Это не разрешение
использовать модель в runtime и не подтверждение качества или LinQ TPU.

Исходный RF-DETR 2XL экспорт и загрузка описаны в [RFDETR_CANDIDATE.md](RFDETR_CANDIDATE.md).
Первичный результат лежит в `data/model_research/rfdetr-2xl-native-ort-parity/`:
black совпал, white/seeded RGB не совпали, у real image110 не совпали boxes.
Исходник этого запуска сохранён рядом как `exporter-source.py`, SHA256
`90baca5df80b43027ff28c0b05678b246ed1bbd361bfd73c594032f5d926eeea`.
Диагностика проверяет его SHA по первичному manifest и не переписывает ни одного
входного артефакта. Единственное изменение существующего helper — необязательный
параметр module у `bounded_run`; прежний CLI/алгоритм паритета не меняются.

Протокол фиксирован:

1. Проверить failed manifest, полный `comparisons.json` из четырёх случаев,
   исходный exporter, static ONNX pin и изолированное окружение. Воспроизвести
   RGB/preprocessing и проверить все четыре исходных input SHA, размеры и
   bytes/SHA/path реального image110. Неполный cache завершает запуск ошибкой.
2. Строго загрузить checkpoint. Выполнить **все четыре A** обычным native eval.
   Затем единожды вызвать source `model.export()` и выполнить **все четыре B**.
   Поэтому ни один A не выполняется уже изменённой моделью.
3. Получить C из исходного ONNX через ORT CPU. Отдельно создать debug-копию
   графа D, добавив только graph outputs существующего единственного TopK:
   scores, values, indices. Проверить K300, axis/largest/sorted и путь indices
   к GatherElements. Новых вычислительных nodes нет.
4. Сравнить обычные выходы C/D побитово и исходным численным gate. Любое
   побитовое отличие отмечается `instrumentation_effect=true`; тогда индексы D
   не приписываются исходному C и поля C selection оставляются null.
5. Сохранить NPZ и JSON, повторно проверить bytes/SHA всех входных/code assets.
   Изменившийся граф, checkpoint, cache, image или исходник завершает запуск
   ошибкой. Каждая output directory должна быть новой.

Native hook оборачивает штатный `model.transformer.select_top_rows`.
Он единожды вызывает исходный selector и применяет **возвращённую им closure**
к `arange(token IDs)` для регистрации реально выбранного порядка. Та же closure
возвращается модели. Второго `torch.topk`, module hooks и подмены выбора нет.
На каждом eval/export ожидается ровно один вызов K300.

Для каждого входа сохраняются full score vector, selected IDs/scores,
`min(selected scores) - max(unselected scores)`, соседние rank gaps и exact ties.
`sample-00.npz` … `sample-03.npz` содержат float32 boxes/logits A/B/C/D, scores и
indices A/B/D. JSON хранит shape/dtype/SHA каждого tensor и SHA самого NPZ.
`learned-slots.npz` сохраняет первые300 `query_feat`/`refpoint_embed`; JSON также
содержит row SHA, диапазон, равенство строк и максимальное отклонение от первой.
Все NPZ — числовые массивы, читаемые с `allow_pickle=False`.

Сравнения A/B, B/C, A/C выводят исходный slot-aligned gate, число не совпавших
элементов/queries, max/mean ошибки и `max(abs(delta)/(atol+rtol*abs(right)))`.
Маска ошибок вычисляется непосредственно через `numpy.isclose` над float32,
точно как исходный `numpy.allclose`. Delta/ratio считаются в float64 для
диагностики; из-за округления на ULP-границе ratio может чуть превышать1 при
нулевом числе ошибок. Ratio не заменяет исходную маску.
Для A/C отдельно проверяются исходные native/ORT output SHA — это явный readback
воспроизводимости, включая возможное несовпадение SHA. Индексы сравниваются по
порядку и множеству; common anchor IDs имеют отдельное диагностическое сравнение.
Когда D не изменил обычные C outputs, его индексы также доступны в C selection.

**Одинаковое TopK-множество не доказывает безвредную перестановку.** Source
сочетает rank-упорядоченные anchors с фиксированными обученными query/refpoint
слотами (`transformer.py`, two-stage query construction). Изменение порядка
меняет это сочетание. Anchor-aligned ошибка не заменяет исходный gate; Hungarian
best-fit не используется. Изменение membership показывается отдельно.

Возможные направления интерпретации остаются гипотезами: A/B различаются —
проверять native export conversion; A/B близки, B/C различаются при малых TopK
gaps — проверять selection sensitivity; одинаковые IDs с остаточным отличием —
проверять арифметику/decoder. Ни одно из них не меняет failed статус первичного
паритета. Даже новый A/C pass не переписывает исторический failed gate.

Запускайте последовательно под тем же watchdog, без изменения
root `.venv`. CWD должен указывать на checkout с новой диагностической фичей:

```bash
/Users/mikemoiseev/Documents/mospromhack/models/candidates/rfdetr-2xlarge-880/export-env/bin/python \
  -m tools.diagnose_rfdetr_candidate \
  --weights /Users/mikemoiseev/Documents/mospromhack/models/candidates/rfdetr-2xlarge-880/rf-detr-xxlarge.pth \
  --export-dir /Users/mikemoiseev/Documents/mospromhack/models/candidates/rfdetr-2xlarge-880/static-fp32-880 \
  --primary-parity-dir /Users/mikemoiseev/Documents/mospromhack/data/model_research/rfdetr-2xl-native-ort-parity \
  --image /Users/mikemoiseev/Documents/mospromhack/data/model_research/construction-ppe-test/test-only/images/test/image110.jpg \
  --output-dir /Users/mikemoiseev/Documents/mospromhack/data/model_research/rfdetr-2xl-parity-diagnostic \
  --timeout 900 --max-rss-gb 12
```

Диагностика требует CPU-only ORT, исходные dependency/source pins и strict state
coverage экспортёра. Не меняются веса, preprocessing, пороги, primary evidence,
production registry/adapter или GT. Настоящий model inference запускается отдельно; focused tests используют tiny state/graphs и fake ORT sessions:

```bash
.venv/bin/python -m pytest -q tests/test_rfdetr_candidate.py tests/test_rfdetr_parity_diagnostics.py
```

Первый контролируемый запуск 2026-10-24 завершился за32.36 секунды и сохранил
четыре NPZ, learned slots и debug ONNX в
`data/model_research/rfdetr-2xl-parity-diagnostic/`. Его исходники архивированы
рядом manifest: `diagnostic-source.py` SHA256
`7481fddee960453dcc1a1d02a81ada7a9febf9d16d49aff9fd5416f16a57a524`,
`exporter-helper-source.py` SHA256
`17e287a5395346d8d09ad096406be78a73fa28ac490794d1bbe94811ede5948a`.
На всех четырёх входах A/B outputs, scores и IDs совпали побитово, обычные C/D
outputs совпали побитово, A/C output SHA точно воспроизвели первичный запуск.

TopK-множества из300 IDs совпали во всех случаях. Порядок различался только
для двух соседних **zero-based** слотов265/266 на white и253/254 на seeded RGB.
Их native rank-score gaps равны6.20e-6 и1.19e-5; максимальные native/ORT
расхождения ranking scores равны9.63e-5 и5.05e-5. Learned query/refpoint rows
не одинаковы: максимальное отклонение от первой строки равно5.83 и0.143.
Даже anchor-aligned сравнение white/seeded остаётся failed. Наблюдения совместимы
с усилением малых арифметических ошибок в дискретном ranking и последующем
slot↔anchor pairing; это не доказательство безвредной перестановки.

После уточнения float32 маски статистику можно пересчитать **без inference**
из этих же сохранённых NPZ в новую directory. Этот режим импортирует только
NumPy из численных библиотек, проверяет его версию2.4.6, все input file/tensor
SHA и архивированные исходники. Он проверяет manifest/comparisons согласованность
и before/after hashes; исходные отчёты не изменяются. В новом manifest явно
записаны `numbers_derived_from_cached_tensors`, `no_inference_performed`, новые
code SHA и исходный failed статус:

```bash
/Users/mikemoiseev/Documents/mospromhack/.venv/bin/python \
  -m tools.diagnose_rfdetr_candidate rescore \
  --diagnostic-dir /Users/mikemoiseev/Documents/mospromhack/data/model_research/rfdetr-2xl-parity-diagnostic \
  --output-dir /Users/mikemoiseev/Documents/mospromhack/data/model_research/rfdetr-2xl-parity-rescore
```

Этот рескоринг завершился успешно с `no_inference_performed=true` и NumPy2.4.6.
Ниже authoritative counts для A/C; B/C даёт те же числа. Знаменатели каждого
случая:300 queries, 1,200 box elements и27,300 logit elements (все91 logit slots).

| Случай | Ошибок box elements / queries | Ошибок logit elements / queries |
|---|---:|---:|
| Black | 0 / 0 | 0 / 0 |
| White | 462 / 215 | 5,568 / 175 |
| Seeded RGB | 477 / 193 | 2,290 / 72 |
| Real image110 | 5 / 4 | 0 / 0 |

На real image110 ошибочные box elements принадлежат zero-based queries
25,104,166,218. Их максимальные class confidences меньше0.011; это контекст
разбора сырых tensors, а не основание исключить queries или ослабить gate.
Первичный native/ORT parity остаётся **failed**. Диагностика не выполняла
quality evaluation и не меняла production adapter/defaults.
