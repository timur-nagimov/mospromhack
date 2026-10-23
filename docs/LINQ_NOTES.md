# LinQ TPU: конспект SDK, облачного API и замеров

Источники:
- `tpu_framework.pdf` (Release 2026.9.45, 56 стр.). **Ссылки «стр. N» — номер в колонтитуле pdf**; физическая страница = N + 4.
- `compile.py`, `README.txt` и `packets/requirements.txt` из архива SDK. Локальная копия: `vendor/linq_sdk/tpu_framework/`.
- `usage_example_python_api` и примеры C/C++ (`vendor/linq_api/`): локальный рантайм `pytpu`/`libtpu` с устройством `/dev/tpu*`. На стенде хакатона они **не нужны**.
- С Jupyter-стенда (`~/work/examples`, `~/work/packages`): `example.ipynb` и `tpu_pycloud_format-0.0.1`. Копия: `vendor/linq_examples/`. Это единственный источник по облачному протоколу; в pdf его нет.
- Метаданные программ (`metadata.json` внутри `.tpu`) и собственные замеры (`.orchestra/metrics/tpu-baseline.json`).

Пометка **[стенд]** — проверено на стенде 24.10.

## 1. Установка SDK и платформа

- Требования: GNU/Linux Ubuntu 18.04+, Python **3.12.11** (стр. 5). Все бинарные колёса собраны под `cp312-linux_x86_64`. На маке arm64 и в Python 3.11 SDK не встаёт.
- Установка по README.txt: venv → `pip install -r packets/requirements.txt` → `python compile.py` (имя `compile_onnx.py` в README ошибочно). Тянет tensorflow 2.20.0, torch 2.9.1, keras 3.14, numpy 1.26.4, onnx 1.16.2, onnxruntime 1.29.0.
- Пакеты SDK: `tpu_framework_bin` 2026.9.45 (фасад), `dnn_quant_bin` (квантование), `tpu_compiler_bin` 2026.9.493, `onnx_direct_bin` (ONNX→TF), `tpu_tlm_bin`/`tpu_tlm_is_bin`/`tpu_hw_is_bin` (модель TPU, ISA), `tpu_io_bin` (упаковка/распаковка, `pytpu_packer`), `optimize_onnx_bin`. Архитектура — на схеме стр. 4.
- Клиента облака `tpu-pycloud` в SDK нет. На стенде лежит только формат сообщений `tpu_pycloud_format` (см. §6).

Как ставим мы:
1. **Docker на маке** (`tpu/docker/Dockerfile`, amd64 + Rosetta). Образ собирается, импорт идёт ~100 с, калибровка и квантование работают. **Компиляция падает**: `compile_` в оптимизации времени (`get_trace`) создаёт TLM-модель DDR размером 268435456×32 байт = 8 ГиБ на банк и ловит `MemoryError` в VM podman на 8 ГБ (`models/tpu/resnet50/compile.log`). Тяжёлые шаги на ноутбуке запрещены (OOM 24.10).
2. **Jupyter-сервер стенда** — основной путь **[стенд]**. Там Linux x86_64, 64 ядра, 187 ГБ RAM, conda Python 3.11. Есть PyPI и download.pytorch.org, cloud.lin-q.ru недоступен. Шаги:
   - колёса залиты через `tools/jhub.py upload --parallel 4` (402 МБ за 381 с);
   - `tpu/server_setup.sh` ставит `uv`, создаёт `~/linq312` (Python 3.12.15) и устанавливает requirements (torch — CPU-сборка);
   - импорт `tpu_framework` занимает 6–12 с.

## 2. Поддерживаемые модели и операторы, ограничения

- Вход: TensorFlow `GraphDef` или ONNX (через встроенный конвертер ONNX→TF) (стр. 3).
- **Динамические входные формы не поддерживаются.** Форму надо заморозить через `freeze_shapes()` (compile.py SDK). Перед этим из входов убираются initializer'ы (`remove_initialized_inputs`), затем вызывается `preliminary_model_conditioning()`.
- Явного списка операторов в pdf нет. Есть список проверенных моделей (стр. 35–36):
  - классификаторы и детекторы: resnet/efficientnet/mobilenet/inception, ssd_mobilenet, tiny_yolo2/3, yolo2/3/4;
  - `yolo5s`, `yolov8n_pose`, `yolov8m_seg`, `yolo11n_obb`, `yolo12m`, scrfd и др.
- **Детекторы поставляются «головой»** (стр. 36, сноски 1–6): граф обрезан, остаток уходит в постпроцессинг на CPU. Для `yolov8n_pose` срез идёт по `/model.22/dfl/conv/Conv_output_0`, `/model.22/Sigmoid_output_0`, `/model.22/Sigmoid_1_output_0`, `/model.22/Concat_5/6/7_output_0`, `/model.22/Mul_6_output_0`. Резать умеет `get_sub_model(model, ordered_input_names, ordered_output_names)` (гл. 8, стр. 39–41).
- Наше решение для YOLOv8-det — `tpu/compile.py --cut-preset yolov8`:
  - на TPU уходит граф до `/model.22/dfl/conv/Conv_output_0` [1,1,4,8400] и `/model.22/Sigmoid_output_0` [1,80,8400];
  - хвост (anchors, dist2bbox, concat → `output0` [1,84,8400]) уходит в `*_tail.onnx` и считается на CPU в `LinqTpuBackend`;
  - причина: целый `output0` в одном int8-тензоре смешивает координаты 0..640 и вероятности 0..1, и при одном масштабе вероятности обнулятся.
- YOLO11 (C2PSA): `yolo11n_obb` есть в списке поддержанных (стр. 36), так что C2PSA, скорее всего, компилируется. Не проверяли, база — YOLOv8.
- O3 и O5 «иногда дают невалидную программу или компилируются слишком долго» → откат на O1 и баг-репорт (стр. 17).

## 3. Квантование

- Поток (стр. 9–15, 27–29; compile.py):
  1. `RegularModel(model=onnx)`;
  2. `calibrate_torch(calibration_data={вход: массив [N,C,H,W]})` или `calibrate(...)` (TF);
  3. → `thresholds` (`RegularModel.dump_thresholds/load_thresholds`, JSON);
  4. `quantize(thresholds)` → `InMemoryQuantizedModel`;
  5. `.save(dir, name)` → `*.qm.onnx` + `scales.json`.
- Калибровочные данные — словарь «имя входа → кадры, склеенные по оси batch» (стр. 24) **после того же препроцессинга, что в рантайме**. В compile.py SDK калибруют на `np.random.rand` с пометкой «нужны реальные изображения». В e2e-тесте калибруют на 2 кадрах, fine-tune — на 50 (стр. 27–29).
- Варианты (стр. 27–29): `qc` (калибровка), `qcf` (+ fine_tune), `qt` (готовые thresholds), `qtf` (+ fine_tune). `fine_tune_torch` дистиллирует fake-quant граф к fp32 и обучает только пороги активаций (стр. 13–14).
- Параметры калибровки (стр. 11–12):
  - `activation_percentile` — обрезка выбросов внутри тензора;
  - `samplewise_percentile` — устойчивость к выбросным кадрам на больших наборах (при max-калибровке больший набор часто даёт квантование хуже);
  - `RegularModel(percentile=100.0, use_safe_scaling=AUTO)` — safe scaling для LayerNorm (стр. 10, 15).
- Квантованная модель **чувствительна к batch и CWL** (cache word length = 128) (стр. 25). Batch программы фиксируется при компиляции: `network.set_batch(1)` (стр. 30), в имени файла — `_b1_`.
- **Форматы входа и выхода.** Пользователь подаёт `dict[str, np.ndarray]` в `user_shape`/`user_dtype` исходной модели (стр. 7). Для ONNX это NCHW float32 (TF-вариант resnet в Quick start — NHWC, стр. 37–38). Внутри TPU:
  - scale + cast → `tpu_dtype` int8 (или float16);
  - padding → tensor2words (блоки по ширине DDR-слова);
  - выход: words2tensor → de-pad/unmask → cast & scale → user dtype (стр. 7, глоссарий стр. 46–47).
- Пример из `metadata.json` resnet50 (`torch_resnet50_b1_o5_h1m.tpu`):
  - вход `input.1`: user [1,3,224,224] NCHW float32 → tpu [1,224,224,1,4] (C,H,W,N,B), int8, scale 48.106 (= 127/max|x|), каналы дополнены 3→4. В TPU-формате 200 704 байт, в float32 — 602 112;
  - выход `495`: tpu [8,1,128] int8, scale 0.329 → user [1,1000] float32.
- **uint8 на входе программы не бывает**: user_dtype берётся из исходной модели (float32). Масштаб и int8 делает упаковщик. Подавать uint8 и `/255` можно только на стороне клиента.
- Сверка quant vs fp32 на CPU (`tpu/compile.py`, без TPU): resnet50 даёт cos = 0.987 на логитах (64 кадра калибровки).

## 4. Уровни оптимизации компилятора

- `compiler.compile_(hardware_parameters, network, parameters, slicing_scheme=None)` → `(Executable, dict[str, TensorDescription])` (стр. 16). Платформа стенда — `TPU_H1M = get_hw_params('h1m')`:
  - systolic array 128×128;
  - DDR: 2 банка по 256 Мслов × 32 байта;
  - CWL 128.
- Пресеты (стр. 16–17):
  - **O1** — «безопасный, по умолчанию для неизвестной сети»;
  - **O3** — staging, бывают невалидные программы;
  - **O5** — параллельная работа конвейеров TPU: предзагрузка весов, вторая DDR для констант, плотнее кеш, layer fusion, гибкий slicer. Может дать невалидную программу → откат на O1 и баг-репорт;
  - **O5_FGS** — O5 с другим слайсером.
- В пакете есть ещё `O0, O2, O2_FGS, O6, O6_FGS` (`tpu_framework.presets`), в pdf они не описаны.
- Пример организаторов скомпилирован с O5 (`*_b1_o5_h1m.tpu`). `tpu/compile.py` по умолчанию пробует O5, при ошибке — O1.

## 5. Программа TPU и загрузка на сервис

- `TpuProgram.from_executable(executable, tensor_descriptions).to_file('x.tpu')` (стр. 31). Файл — zip:
  - `metadata.json` (hardware_parameters, inputs/outputs с user/tpu shape/order/dtype/scales/padding, ram_scheme);
  - потоки инструкций `DTS_L0/L1/S0/S1`, `ME.bin`, `VE.bin`;
  - `constants_<bank>/<addr>.bin`.
- Raw-программа: `pytpu.convert_to_raw(program, raw_program)` → codec. В ней нет metadata; клиент сам пакует (`codec.encode`) и распаковывает (`codec.decode`) плоские 2D uint8 (стр. 31–32, 46).
- **Загрузка на сервис [стенд]**: программа кладётся на общую ФС стенда в `/auto/cache/<login>/` (cp из Jupyter), воркер читает её по URI `file:///auto/cache/<login>/<prog>.tpu?raw=False`. С ноутбука: `tools/jhub.py upload --parallel 4 x.tpu tpu/x.tpu`, затем `jhub.py sh 'cp ~/work/tpu/x.tpu /auto/cache/hackathon10864/'`. По заданию в коде используем свой логин и свои программы.

## 6. Облачный протокол (RabbitMQ) [стенд]

- Брокер `10.0.114.30:5672` (с ноутбука — `127.0.0.1:5672` через контейнер VPN), логин `guest/guest`, vhost `/`.
- Задача уходит в exchange **`tpu.direct`** с routing key **`input`**.
- Тело — protobuf `Task` (`message.proto` из `tpu_pycloud_format`):
  - `dict_data: map<string, Array{dtype: str, shape: int32[], byte_data: bytes}>` — тензоры по именам входов;
  - либо `data_path` — путь к сэмплу на ФС стенда;
  - `list_metadata` — результаты постобработки, если в цепочке есть сервер постпроцессинга;
  - поля воркера в ответе: `worker_pid`, `thread_name`, `inference_time` (float, **секунды**), `status_code` (0 = ок), `error_message`, `inference_opening_timestamp`.
- Заголовки AMQP:
  - `uri` — программа, `file:///auto/cache/<login>/<prog>.tpu?raw=False|True`;
  - `response_routing_keys` — JSON-список ключей ответа, «последний хоп первым»: воркер снимает последний ключ и шлёт туда, сервер постобработки снимает следующий, в конце остаётся очередь клиента;
  - `tpu_pycloud_message_id` — id клиента, возвращается в заголовках ответа (корреляция).
- Клиент объявляет свою очередь и биндит её к `tpu.direct` ключом = имени очереди. Важно:
  - очередь `auto_delete` удаляется после отмены консьюмера, поэтому вторая задача падает `404 NOT_FOUND`;
  - мы держим `exclusive`-очередь и постоянного консьюмера.
- Ответ: `dict_data` — выходы программы в user-формате (float32), например `495`: (1,1000) у resnet50. При ошибке `dict_data` пустой, причина — в `error_message`.
- **raw**: при `raw=False` воркер сам пакует float32 → int8 и распаковывает обратно. По сети идёт float32 (4 байта на элемент). При `raw=True` клиент шлёт уже упакованный uint8 (int8 + паддинг), по стр. 31–32 и 46 это ~4× меньше. Но упаковщик `tpu_io` есть только в бинарном виде под linux x86_64/cp312. Наш бэкенд пока работает с `raw=False`.
- Параллелизм: у воркера минимум два потока (`Worker Thread #0/#1`). Задачи, отправленные окном, обрабатываются параллельно: см. замеры, ×2.4–2.8 к/с.
- Реализация: `guard/inference/linq_tpu.py`:
  - protobuf-дескриптор встроен, совместимость с `tpu_pycloud_format.serialize` проверена бит-в-бит;
  - соединение с очередью ответов на каждый поток, конвейер кадров батча, таймауты, ретраи с переподключением;
  - хвост модели — на CPU (onnxruntime).

## 7. Batch

- Batch программы статический, задаётся при компиляции (`network.set_batch`, стр. 30; квантованная модель batch-sensitive, стр. 25).
- Программы примера и наши — `b1`. `LinqTpuBackend` режет входной батч на задачи batch=1 и отправляет их **конвейером**: все сразу, затем ждёт ответы. Это даёт параллелизм воркера без перекомпиляции. `max_batch_size()` = 8.
- Компиляцию с batch>1 не пробовали. Выигрыш по сети она не даёт: объём тот же, накладные на сообщение малы по сравнению с передачей тензора.

## 8. End-to-end пример из pdf (гл. 5, стр. 19–33; гл. 7, стр. 37–38)

1. ONNX resnet50 `input.1` → `495`.
2. Препроцессинг: центр-кроп 0.85, resize 224, `(x/255 - mean)/std`, NCHW (стр. 21–22).
3. Проверка исходной модели в onnxruntime.
4. Квантование одним из 4 способов (стр. 27–29); проверка квантованной модели через `as_torch_graph` + `rescale(TpuDtype.Input/Output)` (стр. 25–26).
5. `Network.from_quantized_model` → `set_batch(1)` → `compile_(TPU_H1M, network, O1)` (стр. 30). TLM-прогон — внутренний инструмент (стр. 26, 31).
6. `TpuProgram.from_executable(...).to_file` (стр. 31).
7. `pytpu.Device.open` → `device.load(program)` → `program.inference()` → `inference.sync({input: tensor})` → dict выходов. Опционально raw + codec (стр. 31–32).
8. Постпроцессинг `argmax + 1`, ожидаемый класс 257 (Newfoundland).

На стенде вместо шага 7 — облачный воркер (§6). Пример организаторов `example.ipynb` повторён нами с ноутбука и из Jupyter: класс 257 **[стенд]**.

## 9. Замеры [стенд] (`.orchestra/metrics/tpu-baseline.json`, `tpu/bench.py`)

| Где | RTT до брокера | Петля через брокер 1 МБ / 5 МБ | resnet50 total p50 / p95 | инференс воркера p50 | к/с последовательно / окно 4 |
|---|---|---|---|---|---|
| ноутбук (VPN в контейнере) | 235 мс | 7.95 с / 27.6 с | 712 / 1329 мс | 68 мс | 1.23 / 3.4 |
| Jupyter-сервер | 1 мс | 0.10 с / 0.90 с | 101 / 212 мс | 17 мс | 7.8 / 19.1 |

- Канал ноутбук ↔ стенд **асимметричный**:
  - ноутбук → стенд ~1–1.9 МБ/с (upload колёс SDK 402 МБ за 381 с в 4 потока; запрос resnet50 0.6 МБ — ~0.65 с);
  - стенд → ноутбук 0.1–0.4 МБ/с (download 22 МБ — 58 с; петля 0.1 МБ — 2.1 с при RTT 0.24 с).
- Вывод для архитектуры — §10.

## 10. Что влияет на архитектуру

- Вход программы — float32 NCHW исходной ONNX. Упаковка в int8 идёт на воркере (`raw=False`). Тензор 640×640×3 float32 занимает 4.9 МБ; ответ головы YOLOv8 (`Sigmoid` 80×8400 + DFL 4×8400, float32) — ~2.8 МБ.
- С ноутбука через VPN это ≥ 4–5 с туда и ≥ 8–10 с обратно на кадр, то есть **YOLO 640 с ноутбука напрямую в Rabbit нежизнеспособен**. resnet50 224 проходит за 0.7 с только потому, что ответ всего 4 КБ.
- **Пре- и постпроцессинг надо делать рядом с TPU, в Jupyter.** Варианты:
  1. **Весь конвейер guard на Jupyter-сервере** (64 ядра, 187 ГБ, брокер в 1 мс). Наружу идут только события и превью. UI открывается через hub-прокси или туннель.
  2. **Гибрид.** Ноутбук шлёт JPEG кадра (~50–100 КБ) в нашу очередь, воркер в Jupyter декодирует, делает letterbox, отправляет задачу в TPU, считает хвост, декодирует и делает NMS. Обратно идут детекции (~1–5 КБ). Можно через `response_routing_keys = [наша_очередь_ноутбука, наш_постпроцессор]` — эту цепочку протокол и задумывал. Нужно разрешение на свои очереди (вопрос организаторам).
- Конвейер (несколько задач в полёте) даёт ×2.4–2.8 к/с. Держать окно 4–8 задач.
- `raw=True` уменьшил бы вход ~4×, но требует упаковщика `tpu_io` (бинарь x86_64). Имеет смысл только при варианте 2, если в воркер приходят тензоры, а не JPEG.

## 11. Инструменты

- `tools/jhub.py` — Jupyter/JupyterHub по REST и kernel websocket **[стенд]**. Команды:
  - `login` (форма `/hub/login` + `_xsrf`, запуск сервера, выпуск API-токена; есть вариант по токену);
  - `ls`, `upload [--parallel N]`, `download`, `exec`, `sh`, `run-bg`, `ps`, `kernels`, `tail`.
  
  С ноутбука: `JUPYTER_URL=http://127.0.0.1:8888 python tools/jhub.py ...`. Корень contents API — `~/work`.
- `tpu/compile.py` — ONNX → `.tpu` (+ head/tail ONNX, `manifest.json`, `compile.log`, сверка quant vs fp32). Запуск:
  - на сервере: `~/linq312/bin/python ~/work/guard_repo/tpu/compile.py ...`;
  - локально: `--docker`, но компиляция там упирается в память (см. §1).
- `tpu/sync_to_jupyter.py` — заливка кода в `~/work/guard_repo`. `tpu/server_setup.sh` — установка SDK на сервер.
- `tpu/bench.py` — замеры `net` и `tpu`; `--dry-run` работает на моках.
- `tpu/make_calib.py` — калибровочные кадры из видео. Сейчас это 120 кадров из Intel sample-videos (worker-zone, people, store-aisle; CC BY 4.0), лежат в `models/tpu/calib_frames/SOURCES.json`.
