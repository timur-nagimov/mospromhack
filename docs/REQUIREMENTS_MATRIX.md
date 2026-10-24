# Матрица требований кейса: R1–R7, D1–D5 и паттерны нарушений

Состояние на 24.10.2026, 17:55 МСК, main `af68cd2`. Документ отвечает на вопрос «чем доказано каждое требование кейса». Для каждой строки: как закрыто, где код, какие тесты, файлы метрик и скриншоты, честный статус.

Правила документа:
- Цифры берём только из файлов замеров (`tpu-baseline.json`, `edge-*.json`, `eval-*.json`) и документов `docs/`. Рядом с цифрой указан файл и ключ. Сырые замеры хранятся вне репозитория, как и в остальных документах.
- Тесты указаны как pytest node id. Все id проверены командой `pytest --collect-only -q`: 1190 тестов, 24.10 17:55.
- Скриншоты `deliverables/screens/real-*` сняты с живого бэкенда (local, CPU, 24.10 около 01:10). Остальные скриншоты там — из мок-режима (README, «Интерфейс»).
- Файлы `data/media/checks/*` лежат вне репозитория: каталог `data` в git не входит.
- Места под результаты треков A и B (они идут сейчас) помечены `<<…>>`.

| Статус | Что значит |
|---|---|
| **закрыто** | работает в живом режиме edge на TPU + шлюз, есть тесты и замер |
| **закрыто, с ограничением** | работает, но есть оговорка, которую говорим вслух |
| **частично** | часть требования не сделана или не проверена |
| **нет данных** | код и тесты есть, роликов для проверки нет |

## Сводка

| # | Требование кейса | Статус | Главное доказательство |
|---|---|---|---|
| R1 | Видеопотоки с камер или наборы изображений | закрыто, с ограничением: RTSP и веб-камера без тестов и без живой камеры | 4 камеры из видеофайлов на стенде; источники `file`, `images`, `rtsp`, `webcam` |
| R2 | Распознавание нейросетями | закрыто, с ограничением: качество по событиям мерили на CPU | 4 модели YOLOv8 на TPU; на 13 роликах P 1.00, R 0.78 |
| R3 | Оповещение в вебе: кадр с выделенным человеком, камера, зона, время, тип | закрыто | карточка события; сквозная задержка на TPU p50 2.20 с |
| R4 | Одно нарушение — строго одно оповещение | закрыто, с ограничением: в оценке 23 события | 0 дублей и 0 ложных тревог на 535 с видео |
| R5 | Журнал с фильтрами по типу, камере, зоне и времени | закрыто | плюс фильтры по статусу и важности, экспорт CSV |
| R6 | Инференс на LinQ, замеры FPS и задержки | закрыто | det 480 — 45.7 к/с; 4 камеры по 4.8–4.9 к/с; 76 мин без отката на CPU |
| R7 | Новый тип нарушения — отдельный модуль | закрыто | 13 плагинов: 10 паттернов кейса и 3 дополнительных правила; S9 и S10 добавлены двумя файлами без правки конвейера |
| D1 | Зоны и правила из UI, расписание | закрыто, с ограничением: в режиме edge правка доходит до стенда только после рестарта | редактор зон, `active_hours`, `PUT /api/cameras/{id}` |
| D2 | Подавление повторных срабатываний | закрыто | `cooldown_s`, переоткрытие события без оповещения, дедуп при смене трека |
| D3 | Дашборд по типам, камерам, времени суток и дням | закрыто, с ограничением: отчёт за период — только через API | экран «Аналитика», `/api/stats`, `/api/reports/shift` |
| D4 | Статусы оператора для оценки точности и дообучения | закрыто, с ограничением: время решения не хранится, маховик на edge не собирает | клавиши 1/2/3, доля ложных по правилам, `operator_label` в маховике |
| D5 | Несколько потоков одновременно | закрыто | 4 камеры на одном edge-узле, CPU 1.18 из 1.5 ядра |

## R1. Приём данных

**Как закрыто.**
- Источник камеры задаётся в `configs/cameras.yaml` или из UI. Типы:
  - `file` — видеофайл в темпе реального времени, с зацикливанием;
  - `images` — папка кадров, то есть стоп-кадры из одного места (вторая половина R1);
  - `rtsp` — IP-камера, транспорт TCP;
  - `webcam`.
- В демо 4 камеры читают видеофайлы. На стенде — субпоток 640×360 тех же роликов: его декод занимает 0.28 ядра вместо 0.92 (`docs/EDGE.md` §2).
- Наборы кадров для режима `images` — `data/images/<сценарий>/`: 23–49 стоп-кадров на сценарий и 40 фото со стройки RF100 (`docs/DATA.md`).

**Код:** `guard/sources/__init__.py` (`make_source`), `guard/sources/video.py` (`FileSource`, `LiveSource`, `WebcamSource`), `guard/sources/images.py` (`ImageFolderSource`), `guard/pipeline/config.py` (`resolve_source_uri`).

**Тесты:**
- `tests/test_pipeline_core.py::test_file_source_loops`
- `tests/test_pipeline_core.py::test_file_source_realtime_keeps_only_latest`
- `tests/test_pipeline_core.py::test_images_source_paces_by_fps`
- `tests/test_pipeline_core.py::test_resolve_source_uri`
- `tests/test_api_app.py::test_camera_crud_persists_and_hot_restarts`

**Метрики:** `edge-tpu-20261024-1002.json` → `fps.<камера>.fps_in`: на входе 24.9 к/с, в анализ идёт 4.81–4.88 к/с на камеру.

**Скриншоты:** `deliverables/screens/real-01-monitoring.jpg` (local, 3 камеры); вне git — `data/media/checks/orch_ui_tpu.png` (шлюз, бэкенд LinQ TPU, 4 камеры, 24.10 12:35).

**Статус: закрыто, с ограничением.**
- Файлы и папки кадров покрыты тестами. Видеофайлы работают вживую на стенде.
- RTSP и веб-камера реализованы, но тестов нет, а на живой камере их не проверяли: камеры у нас нет.
- Своих съёмок с производства нет. Все ролики — открытые (`docs/EXPERT_CALL.md` §3).

## R2. Распознавание нейросетями

**Как закрыто.** На TPU работают 4 сети (`configs/models.stand.yaml`):

| Модель | Вход | Что даёт | Кто использует |
|---|---|---|---|
| det — YOLOv8n COCO | 480 | люди, техника (`car`, `truck`, `bus`), телефон | все правила |
| pose — YOLOv8n-pose | 480 | 17 ключевых точек: лежит ли человек, телефон у уха, колено на ограждении | S3, S6, S8, S10 |
| ppe — YOLOv8s, 14 классов | 640 | каска и «нет каски», жилет и «нет жилета», экскаватор, фронтальный погрузчик, самосвал | S2 |
| fire — YOLOv8n | 640 | огонь и дым | S7 |

- Атрибуты человека (каска, жилет, телефон, поза) считаются в `assign_attributes` и сглаживаются по треку.
- Модель СИЗ выбрали из 6 открытых на отложенном наборе Construction-PPE (284 кадра). У выбранной AP50: человек 0.824, каска 0.826, жилет 0.835 (`docs/MODELS.md`).

**Код:** `guard/inference/onnx_cpu.py`, `guard/inference/linq_tpu.py`, `guard/vision/` (letterbox, декод YOLOv8, NMS), `guard/tracking/attributes.py`, `configs/models.yaml`, `configs/models.stand.yaml`.

**Тесты:**
- `tests/test_vision_yolo.py::test_three_known_objects_layouts_and_class_aware_nms`
- `tests/test_vision_yolo.py::test_pose_restores_keypoints_and_suppresses_duplicate`
- `tests/test_models_registry.py::test_every_entry_validates`
- `tests/test_models_registry.py::test_required_models_and_ppe_classes`
- `tests/test_tracking_attributes.py::test_phone_at_ear_uses_nearest_head_point`

**Метрики:**
- Качество по событиям на 13 роликах: P 1.00, R 0.78 (18 из 23), 0 ложных тревог — `eval-onnx-cpu-phone-nearest-stress-20261024-1342.json` → `overall`.
- Совпадение TPU (int8) с CPU (fp32) — `tpu-baseline.json` → `summary.holdout_recall_precision_tpu_conf_0.25` и `summary.ppe_fire` (таблица в R6).

**Статус: закрыто, с ограничением.**
- Качество по событиям мерили на CPU: det 640 fp32 из `configs/models.yaml`. На стенде работает det 480 int8. Оценка в живой конфигурации TPU идёт (сценарии `ml/eval/scenarios_tpu_live.yaml`): <<eval-tpu: P, R, ложные тревоги и задержка на TPU против CPU — eval-tpu-*.json>>.
- Нужных классов нет: сигарета, страховочная привязь, респиратор. Класс `no-mask` у модели СИЗ не обучен (`docs/MODELS.md`). Погрузчик: модель `forklift-v1` дообучена и прошла гейт, статус `candidate`. Она подключается к `vehicle_proximity` только на камерах S4, на стенде её пока нет (`docs/MODELS.md`, «Погрузчик»).
- Кроме `forklift-v1`, работают открытые веса под AGPL-3.0 (`docs/EXPERT_CALL.md` §3.2). Дообучение идёт в треке A: <<ds-forklift>>, <<ds-falls>>, <<ds-smoking>>, <<ds-ppe>>.

## R3. Оповещение в веб-интерфейсе

**Как закрыто.**
- Как только событие подтверждено, сохраняется снапшот JPEG с рамкой нарушителя. Клип H.264 ±5 с пишется в фоне (`guard/events/media.py`).
- Карточка приходит в ленту мониторинга по WebSocket `/ws/events`. Критичные закреплены сверху, со звуком. Баннер критической тревоги виден на любом экране (`frontend/README.md`).
- В карточке:
  - кадр с рамкой и клип ±5 с;
  - камера, участок, зона;
  - время начала нарушения, оповещения и окончания;
  - тип, важность, трек;
  - задержка по этапам.

  Решение оператор принимает клавишами 1/2/3.
- В режиме edge + шлюз снапшот приходит со стенда в самом событии (p50 25 КБ). Клип шлюз собирает из своего воспроизведения того же видеофайла (`guard/pipeline/gateway.py`).

**Код:** `guard/events/media.py`, `guard/api/app.py` (`/api/events/{id}/snapshot.jpg`, `/api/events/{id}/clip.mp4`, `/ws/events`), `guard/pipeline/gateway.py`, `frontend/src/features/event/EventModal.tsx`, `frontend/src/features/monitoring/AlertFeed.tsx`.

**Тесты:**
- `tests/test_api_app.py::test_snapshot_and_clip_files`
- `tests/test_api_app.py::test_ws_receives_new_event_and_metrics`
- `tests/test_pipeline_core.py::test_media_store_streaming_clip_and_snapshot`
- `tests/test_pipeline_core.py::test_write_clip_is_browser_h264`
- `tests/test_pipeline_gateway.py::test_edge_to_gateway_event_meta_metrics`
- `tests/test_pipeline_gateway.py::test_compact_snapshot_fits_slow_channel`

**Метрики** — задержка от первого нарушающего кадра до ноутбука:

| Замер | p50 / p95 | Файл |
|---|---|---|
| edge на TPU, 4 камеры, 300 с, 24.10 10:02 | **2.20 / 2.53 с**. Из них окно правила 1.71 с (p50), система 0.31 с | `edge-tpu-20261024-1002.json` → `events.e2e_from_started_s`, `confirm_window_s`, `e2e_from_confirmed_s` |
| итоговая конфигурация, 11:33 | 2.30 / 2.99 с | `edge-tpu-20261024-1133.json` |
| 10 минут при загруженном канале VPN, 13:38 | 2.84 / 4.05 с; канал стенд → ноутбук p50 1.02 с | `edge-tpu-stability-tap-20261024-1338.json` |
| настоящий шлюз: от начала нарушения до записи в журнал, 11:50 | 1.8–3.3 с, p50 около 2.6 с (12 событий) | `docs/EDGE.md` §7.1; `edge-gateway-a3-prefetch200-20261024-1150.json` |

**Скриншоты:** `deliverables/screens/real-02-event-card.jpg` (local, CPU: задержка 2.1 с = окно 2.0 с + обработка 65 мс), `02b-event-clip.jpg`, `real-01-monitoring.jpg`. Вне git — `data/media/checks/orch_ui_tpu.png`.

**Статус: закрыто.** Ограничения:
- Задержка колеблется вместе с каналом VPN и общей очередью TPU (`docs/EDGE.md` §6.4). Решение человека 24.10 в 14:10: главное — качество, задержка второстепенна (`docs/DATASETS.md`, «Сеть во время демо»).
- В режиме шлюза клип собирается из локального воспроизведения того же файла (`docs/ARCHITECTURE.md` §2–3). С реальными камерами шлюзу понадобится свой поток с камеры или клип с edge-узла.
- Скриншота карточки события в режиме edge + TPU в `deliverables/screens` нет.

## R4. Одно нарушение — одно оповещение

**Как закрыто.**
1. Правило не хранит состояние подтверждения. Метод `evaluate()` отвечает только на вопрос «есть ли нарушение на этом кадре».
2. `EventManager` поднимает событие, только если нарушение держится `min_duration_s` (в демо 1–2 с) на доле кадров не меньше `min_ratio` (0.5–0.7). Случайный кадр тревогу не поднимает.
3. Одно событие на тройку «камера, правило, трек». Пока трек жив, повторов нет.
4. Если у человека сменился ID трека, новое нарушение с IoU ≥ 0.3 к недавнему событию того же правила новым событием не считается. Падение отдельно учитывает смену позы (CX-021). ByteTrack держит ID при смене позы (CX-021B).
5. СИЗ: голосование по треку учитывает и кадры, где модель СИЗ ничего не нашла (CX-016). Один шумный `no_helmet` тревогу не поднимает.
6. Доставка на шлюз — at-least-once. Шлюз пишет событие по `Event.id`, поэтому повторная доставка второй карточки не даёт.

**Код:** `guard/events/manager.py`, `guard/tracking/bytetrack.py`, `guard/tracking/attributes.py`, `guard/pipeline/gateway.py`.

**Тесты:**
- `tests/test_events_manager.py::test_single_violating_frame_does_not_confirm`
- `tests/test_events_manager.py::test_sixty_seconds_still_produce_one_event`
- `tests/test_events_manager.py::test_temporarily_missing_track_id_then_new_id_keeps_one_event`
- `tests/test_events_manager.py::test_cx021_demo_posture_change_notifies_once`
- `tests/test_events_manager.py::test_two_simultaneous_people_have_separate_events`
- `tests/test_e2e_scenarios.py::test_single_false_no_helmet_frame_produces_no_event`
- `tests/test_e2e_scenarios.py::test_new_track_id_within_cooldown_does_not_notify_twice`
- `tests/test_tracking_bytetrack.py::test_posture_footpoint_recovery_preserves_track`
- `tests/test_pipeline_gateway.py::test_gateway_ignores_redelivery_and_bad_messages`

**Метрики** — офлайн-прогон всего конвейера по 13 роликам (535 с, 23 достоверных события, CPU):

| | База, штатные параметры | Итог |
|---|---|---|
| Дубли | 2 | **0** |
| Ложные тревоги | 3 | **0** |
| Precision / recall | 0.79 / 0.48 | **1.00 / 0.78** (18 из 23) |
| Задержка p50 / p95 | 3.1 / 6.0 с | 2.35 / 8.85 с |
| Файл | `eval-onnx-cpu-base-20261024-0913.json` | `eval-onnx-cpu-phone-nearest-stress-20261024-1342.json` |

На живом edge повторных доставок одного события — 0 (`edge-tpu-20261024-1002.json` → `events.duplicates`; подписчик считает повторы по `Event.id`).

**Статус: закрыто, с ограничением** (`docs/QUALITY.md`, «Ограничения»):
- В оценке 23 события: одно событие — это 4.3 п. п. полноты.
- «0 ложных тревог на 535 с» значит лишь, что частота, скорее всего, ниже ~20 в час (правило трёх).
- Параметры подбирали и проверяли на одних и тех же 13 роликах. Отложенной выборки нет. Для защиты от подгонки параметры общие для типа правила, взяты середины плато и есть стресс-тесты на чужих роликах.
- Демо-ролики зациклены, и каждый проход даёт новое событие. Это не дубль, но на показе это надо проговорить.

## R5. Журнал событий с фильтрами

**Как закрыто.**
- Журнал — SQLite (WAL, индексы по камере, типу, статусу, зоне и времени).
- `GET /api/events` фильтрует по `type`, `camera_id`, `zone_id`, `status`, `severity`, `from`, `to`. Значения можно перечислять через запятую, есть пагинация.
- Экран «Журнал»:
  - фильтры сохраняются в адресе;
  - превью кадров;
  - экспорт CSV для Excel (`;`, UTF-8 BOM) по всем страницам выдачи.
- В записи есть кадр, клип, решение и комментарий оператора.

**Код:** `guard/events/journal.py`, `guard/api/app.py` (`list_events`), `frontend/src/features/journal/JournalPage.tsx`, `frontend/src/features/journal/csv.ts`.

**Тесты:**
- `tests/test_events_journal.py::test_query_filters`
- `tests/test_events_journal.py::test_pagination_total_and_stable_order`
- `tests/test_events_journal.py::test_export_csv_has_russian_headers_and_all_filtered_rows`
- `tests/test_events_journal.py::test_parallel_writes_from_four_threads`
- `tests/test_api_app.py::test_events_list_filters_and_pagination`

**Скриншоты:** `deliverables/screens/real-03-journal.jpg`.

**Статус: закрыто.** Ограничения:
- Учётных записей и ролей нет.
- Время и автор решения оператора не записываются: в таблице `events` есть только `status` и `operator_note`. Для доказательной базы это стоит добавить.

## R6. Инференс на LinQ с замерами скорости

**Как закрыто.**
- Бэкенд `linq-tpu` работает по облачному протоколу LinQ (exchange `tpu.direct`).
- Raw-режим: вход пакуем в int8 сами на numpy. Это в 1.9–2.3 раза быстрее штатного режима, а выходы совпадают бит в бит.
- В полёте 4–8 задач. На TPU уходит только основа YOLO, хвост модели и NMS считаются на CPU (`docs/LINQ_NOTES.md` §13–17).
- Все 4 модели работают на TPU. Пороги подобраны под int8, потолок уверенности Sigmoid снят (`tpu/fix_thresholds.py`).
- Конвейер работает на стенде рядом с TPU (`docs/ARCHITECTURE.md` §3, D-005). С ноутбука через VPN кадр YOLO шёл бы 3.3 с, на стенде — 0.04 с (`tpu-baseline.json` → `summary.laptop_yolov8n_480_raw_total_p50_ms`; `docs/BENCHMARKS.md` §4.2).
- Отказ TPU: fail-fast, circuit breaker, откат на CPU и возврат на TPU (CX-018). Таймаут ответа TPU на стенде — 5 с.
- Замеры делают:
  - `tpu/bench.py` и `tpu/combo_bench.py` — скорость;
  - `tpu/yolo_check.py` — сверка с CPU;
  - `guard/edge_probe.py` — сквозная задержка с поправкой на расхождение часов стенда и ноутбука.

**Код:** `guard/inference/linq_tpu.py`, `tpu/compile.py`, `tpu/fix_thresholds.py`, `guard/edge.py`, `guard/edge_probe.py`, `tools/stand.py`, `scripts/run_tpu.sh`, `configs/cameras.stand.yaml`, `configs/models.stand.yaml`.

**Тесты:**
- `tests/test_linq_tpu.py::test_raw_pack_layout`
- `tests/test_linq_tpu.py::test_raw_unpack_crop_and_scale`
- `tests/test_linq_tpu.py::test_wire_roundtrip`
- `tests/test_linq_tpu.py::test_backend_pipeline_fake`
- `tests/test_linq_tpu_failfast.py::test_circuit_blocks_network_and_recovers`
- `tests/test_inference_fallback.py::test_switch_and_return_with_same_spec`
- `tests/test_edge_stand.py::test_stand_models_have_tpu_programs_and_match_main_registry`
- `tests/test_edge_transport.py::test_probe_summary_clock_latency_and_loss`

**Метрики:**

| Что | LinQ TPU | CPU того же контейнера | Источник |
|---|---|---|---|
| det 480, к/с при окне 1 / 4 / 8 | 22.8 / 38.0 / **45.7** | 8.3–9.3 | `tpu-baseline.json` → `summary.yolov8n_480_raw_fps_w1_w4_w8`; `docs/BENCHMARKS.md` §2 |
| pose 480, к/с при окне 1 / 4 / 8 | 24.1 / 44.7 / **50.1** | 7.1 | `summary.pose_480_raw_fps_w1_w4_w8` |
| det + pose на каждый кадр, батч 8 | 21.1 к/с | — | `summary["det+pose_480_raw_fps_b1_b4_b8"]` |
| Совпадение с CPU fp32, recall / precision | det 0.95 / 0.97 и pose 0.98 / 1.00 на 30 кадрах вне калибровки; ppe 0.91 / 0.95 на 43 кадрах; fire 0.87 / 0.93 на 28 кадрах | эталон | `summary.holdout_recall_precision_tpu_conf_0.25`, `summary.ppe_fire`; `docs/LINQ_NOTES.md` §16–17 |
| Живой edge, к/с на камеру (цель 5) | 4.81–4.88 | 0.29–1.84 | `edge-tpu-20261024-1002.json`, `edge-cpu-20261024-1109.json` → `fps` |
| CPU контейнера при квоте 1.5 ядра | 1.18 | 1.50 | те же файлы → `cpu.container_cores` |
| События за прогон | 41 за 300 с, все 4 типа | 13 за 240 с, СИЗ и падений — 0 | те же файлы → `events.by_type` |
| Сквозная задержка p50 / p95 | 2.20 / 2.53 с | 3.02 / 3.56 с | `edge-tpu-20261024-1002.json` → `comparison` |
| Стабильность | 76 мин без отката на CPU, потерь `frame_meta` 0 % | — | `edge-tpu-stability-tap-20261024-1338.json`; `docs/EDGE.md` §6.4 |

**Скриншоты:** `deliverables/screens/real-07-performance.jpg` снят в режиме local на CPU. Скриншота экрана «Производительность» с бэкендом `linq-tpu` в `deliverables/screens` нет. Вне git есть `data/media/checks/orch_ui_tpu.png`: мониторинг с бейджем «Инференс LinQ TPU».

**Статус: закрыто.** Ограничения:
- Очередь TPU общая с другими командами. `inference_time` для det 480 ночью был 8 мс, утром в пике — 34 мс (`docs/EDGE.md` §7.2).
- В контейнере стенда 1.5 CPU. Компиляция рядом с edge поднимает задержку до p50 2.40 с, p95 3.25 с (`edge-tpu-20261024-0954-contended.json`).
- TPU доступен через облачный брокер, а не по PCIe. Около 30–40 мс вызова — путь через брокер (`docs/EDGE.md` §6.3).
- «Около 9 камер на TPU для det 480 и около 4 для det + pose» — расчёт по формуле `docs/BENCHMARKS.md` §5. Вживую проверены 4 камеры.

## R7. Новый тип нарушения — отдельным модулем

**Как закрыто.**
- Тип нарушения — файл в `guard/rules/plugins/`. Это класс-наследник `RulePlugin` с декоратором `@register_rule` и полями `type`, `title`, `required_models`, `params_schema`, плюс метод `evaluate()`.
- Плагины находятся автоматически (`load_plugins`). Подтверждение и дедуп делает общий `EventManager`. UI берёт название из `GET /api/rules/types` и строит форму параметров по `params_schema`.
- Сейчас 13 плагинов: `zone_intrusion`, `ppe_missing`, `fall_immobile`, `vehicle_proximity`, `running`, `phone_usage`, `smoking_fire`, `fence_climb`, `height_no_harness`, `unsafe_carry` и три дополнительных правила после 17:55 — `wrong_way`, `zone_dwell`, `zone_overcapacity` (`docs/VIOLATION_EXPANSION_RESULTS.md`).
- Проверено на практике: S9 и S10 добавлены коммитом `69485c9` (CX-014). В нём два файла плагинов и два файла тестов, конвейер, API и UI не менялись.
- Новая модель — запись в `configs/models.yaml`, плагин запрашивает её через `required_models`. Маховик берёт целевые классы плагина из атрибута `flywheel_classes` (`docs/FLYWHEEL.md`).

**Код:** `guard/rules/base.py`, `guard/rules/plugins/*.py`; инструкция — README, раздел «Как добавить новый тип нарушения»; точки расширения — `docs/ARCHITECTURE.md` §6.

**Тесты:**
- `tests/test_e2e_scenarios.py::test_plugins_are_registered`
- `tests/test_api_app.py::test_health_and_rule_types`
- `tests/test_rules_zone_intrusion.py::test_plugin_registered_with_schema`
- `tests/test_rules_height_no_harness.py::test_plugin_discovered`
- `tests/test_rules_unsafe_carry.py::test_plugin_discovered`
- `tests/test_rules_default_severity.py::test_unset_severity_takes_plugin_default`

**Статус: закрыто.** Ограничение одно: новая модель требует компиляции под TPU (до 2.5 минуты CPU контейнера стенда: ppe_640 — 149 с, `docs/MODELS.md`) и сверки с CPU (`docs/LINQ_NOTES.md` §16–17). Во время показа этого не делаем.

## D1. Зоны и правила из UI

**Как закрыто.**
- Экран «Камеры и правила»:
  - полигон зоны рисуется мышью на стоп-кадре, вершины можно перетаскивать;
  - тип зоны: опасная, обязательны СИЗ, курение запрещено, место для курения, проезд техники, работа на высоте, ограждение;
  - расписание активности `active_hours` по МСК, например `07:55-12:00`, в том числе через полночь.
- Правила камеры: форма строится по `params_schema` плагина. Там же задаются окно подтверждения, `cooldown_s` и важность.
- Сохранение — `PUT /api/cameras/{id}`. Камера перезапускается на лету, процесс сервиса — нет.

**Код:** `frontend/src/features/cameras/ZoneEditor.tsx`, `RulesEditor.tsx`, `SchemaForm.tsx`; `guard/rules/geometry.py` (`is_zone_active`, `point_in_zone`); `guard/contracts.py` (`Zone.active_hours`); `guard/api/app.py` (`/api/cameras`).

**Тесты:**
- `tests/test_api_app.py::test_camera_crud_persists_and_hot_restarts`
- `tests/test_rules_zone_intrusion.py::test_inactive_zone_by_hours`
- `tests/test_rules_zone_intrusion.py::test_time_ranges_moscow_and_overnight`
- `tests/test_rules_zone_intrusion.py::test_active_hours_validation_and_failsafe`
- `tests/test_rules_zone_intrusion.py::test_head_in_zone_but_feet_outside_is_not_violation`
- `tests/test_pipeline_core.py::test_cameras_yaml_roundtrip`

**Скриншоты:** `deliverables/screens/real-05-cameras-zones.jpg` (живой бэкенд); `05b-zone-drawing.jpg` (мок: зона «Проход у стеллажей», 07:55–12:00, как в сквозном сценарии кейса); `06-cameras-rules.jpg` (мок).

**Статус: закрыто, с ограничением.** В режиме edge + шлюз правка в UI меняет только `configs/cameras.yaml` на ноутбуке. Edge читает `configs/cameras.stand.yaml` при старте, а канала конфигурации «шлюз → edge» нет (`docs/EDGE.md` §7.2, §7.3 п. 4). На стенде правка начнёт действовать после `tools/stand.py sync` и `restart`. В режиме local правка действует сразу.

## D2. Подавление повторных срабатываний

**Как закрыто.**
- `cooldown_s` задаётся у правила. После закрытия события тот же трек не даёт нового события, пока пауза не истекла.
- Если нарушения нет `close_after_s`, событие закрывается. Если оно вернулось в пределах паузы, переоткрывается то же событие (`event_update`), новой тревоги нет.
- При смене ID трека работает дедуп (IoU ≥ 0.3, `reid_gap_s` 5 с) — см. R4.
- Разные заходы в зону — разные события (CX-017). Так журнал честнее.

**Код:** `guard/events/manager.py`. Параметры — в `configs/cameras.yaml`: в демо `cooldown_s` 5–10 с.

**Тесты:**
- `tests/test_events_manager.py::test_same_track_cooldown_is_strict_and_measured_from_ended_at`
- `tests/test_events_manager.py::test_recently_closed_event_reopens_without_new_notification`
- `tests/test_events_manager.py::test_after_closure_and_cooldown_a_new_event_can_confirm`
- `tests/test_events_manager.py::test_s1_01_three_visits_at_same_place_produce_three_events`
- `tests/test_events_manager.py::test_concurrent_calls_do_not_duplicate_confirmation`

**Метрики:** дубли на 13 роликах — 2 в базе и 0 в итоге (таблица в R4).

**Статус: закрыто.** Ограничение: на склейке роликов дедуп может забрать реальное нарушение другого человека. На s2_02 пауза между людьми 3.8 с меньше `reid_gap_s` 5 с (`docs/QUALITY.md`, «После подбора»). В живой камере склеек нет.

## D3. Дашборд по типам, камерам, времени суток и дням

**Как закрыто.**
- Экран «Аналитика»:
  - нарушения по типам, участкам и камерам;
  - тепловая карта «день недели × час»;
  - график по дням;
  - решения оператора и доля ложных по каждому правилу.

  Периоды: сегодня, 24 часа, 7 дней, 30 дней, всё время, свой. У каждого графика есть табличный вид.
- `GET /api/stats?group_by=…` — группировки журнала.
- Отчёт за смену или период — `GET /api/reports/shift?from=…&to=…&format=json|html|md` (CX-020). В нём распределения, доля ложных, горячие точки, критические события и сравнение с прошлым периодом. Готовые периоды: текущая и прошлая смена, сутки, 7 дней.

**Код:** `frontend/src/features/analytics/AnalyticsPage.tsx`, `guard/api/app.py` (`/api/stats`), `guard/events/journal.py` (`stats`), `guard/api/reports.py`, `guard/reports/shift_report.py`.

**Тесты:**
- `tests/test_api_app.py::test_stats_groupings`
- `tests/test_events_journal.py::test_stats_timezone_and_filters`
- `tests/test_events_journal.py::test_stats_repeated_hour_during_dst_transition`
- `tests/test_api_reports.py::test_three_formats`
- `tests/test_api_reports.py::test_presets`
- `tests/test_shift_report.py::test_hotspots_and_shares`
- `tests/test_shift_report.py::test_half_open_periods_and_trend`

**Скриншоты:** `deliverables/screens/real-04-analytics.jpg`.

**Статус: закрыто, с ограничением.**
- Отчёт есть только в API: кнопки в UI нет. Готового периода «месяц» тоже нет, месяц задаётся параметрами `from` и `to`.
- Журнал живого демо — это события 24–25.10 по зацикленным роликам. Картина за 30 дней есть только в мок-режиме: там синтетический журнал примерно на 450 событий (`frontend/README.md`).

## D4. Статусы оператора

**Как закрыто.**
- В карточке события клавиши: `1` — «Подтверждено», `2` — «Ложное», `3` — «Передано мастеру», `C` — комментарий, `N` — следующее неразобранное. Решение уходит через `PATCH /api/events/{id}`, обновление — по WebSocket.
- Решения видны в журнале (фильтр по статусу) и в аналитике (доля ложных по правилу). В отчёте за период — доля ложных по камерам.
- Маховик данных переносит решение в сэмпл события как `operator_label` (`docs/FLYWHEEL.md`):
  - «подтверждено» → `positive`;
  - «ложное» → `hard_negative`;
  - «передано мастеру» → `forwarded`.

  Без проверки человеком метки в обучение не идут.

**Код:** `frontend/src/features/event/EventModal.tsx`, `frontend/src/domain/labels.ts` (`OPERATOR_DECISIONS`), `guard/api/app.py` (`patch_event`), `guard/flywheel/wiring.py` (`operator_decision`), `guard/flywheel/store.py`.

**Тесты:**
- `tests/test_api_app.py::test_event_get_patch_and_ws_update`
- `tests/test_events_journal.py::test_add_get_update_and_reopen`
- `tests/test_shift_report.py::test_totals_distributions_quality_and_critical_events`
- `tests/test_flywheel_operator.py::test_operator_decision_moves_to_event_sample`
- `tests/test_flywheel_operator.py::test_operator_decision_before_sample_is_written`
- `tests/test_flywheel_store.py::test_operator_label_goes_to_event_samples_only`

**Скриншоты:** `deliverables/screens/real-02-event-card.jpg` (кнопки 1/2/3 и поле комментария). Страница «Данные» на мок-данных — вне git: `data/media/checks/flywheel_ui_01-review.png` … `flywheel_ui_07-train-forbidden.png`.

**Статус: закрыто, с ограничением.**
- Время и автор решения не записываются. Поэтому шаг сквозного сценария «медпункт вызывают через 30 с» по журналу не проверить.
- «Передано мастеру» — только статус. Уведомления мастеру (мессенджер, сигнал у зоны) нет.
- Маховик собирает сэмплы в режиме local: живой прогон 14:01 на CPU дал 12 сэмплов событий, 17 неуверенных и 2 мигания трека (`docs/FLYWHEEL.md`, «Проверено»). С 24.10 вечера сэмплы собираются и на edge: они идут на шлюз конвертами `flywheel_sample` в свою очередь AMQP (`guard/flywheel/edge.py`, `guard/flywheel/amqp.py`). Канал проверен на RabbitMQ стенда 24.10 21:20: 4 сэмпла из 4 собраны на приёмной стороне. На стенде сбор заработает после перезапуска edge и шлюза с новым кодом. Решение оператора на шлюзе переносится в `operator_label` сэмпла события. <<fw-core: сбор на edge и приём сэмплов шлюзом — статус на 17:00>>
- Сверка TPU с CPU (`disagreement`) — `guard/flywheel/shadow.py`. Включается `GUARD_FLYWHEEL_SHADOW=1` и работает только на `linq-tpu`. На стенде не включалась.
- Дообучение и гейт (`ml/flywheel/train.py`, `ml/flywheel/gate.py`) влиты в 16:33 вместе с первой карточкой `ml/model_cards/forklift-v1.json`: гейт пропустил модель, статус `candidate`. Выкатки (`deploy.py`) и авторазметки учителем (`autolabel.py`) в main нет. <<ds-forklift: цифры до/после>>

## D5. Несколько потоков одновременно

**Как закрыто.**
- На одном edge-узле одновременно работают 4 камеры:
  - 4.81–4.88 к/с на камеру при цели 5, CPU 1.18 из 1.5 ядра (замер 10:02);
  - 4.85–4.90 к/с через 76 минут работы (замер 13:38).
- Кадры разных камер идут на TPU общим батчем (`BatchScheduler`, `GUARD_MAX_BATCH` = 8), в полёте 4 задачи.
- В режиме local на ноутбуке (M3 Pro, CPU) те же 4 камеры дают 4.79–4.92 к/с и занимают 3.6 ядра (`docs/EDGE.md` §6.3).
- Сколько камер выдержит один TPU: около 9 для det 480 и около 4 для det + pose при 5 к/с — это расчёт (`docs/BENCHMARKS.md` §5).

**Код:** `guard/pipeline/engine.py`, `guard/pipeline/detector.py` (`BatchScheduler`), `guard/pipeline/worker.py` (`CameraWorker`).

**Тесты:**
- `tests/test_pipeline_core.py::test_batch_scheduler_merges_cameras`
- `tests/test_events_manager.py::test_cameras_are_isolated_and_reset_only_removes_requested_camera`
- `tests/test_edge_stand.py::test_stand_cameras_match_demo_cameras_and_have_models`
- `tests/test_events_journal.py::test_parallel_writes_from_four_threads`

**Метрики:** `edge-tpu-20261024-1002.json` → `fps`, `cpu`; `edge-tpu-stability-tap-20261024-1338.json` → `fps`; `edge-cpu-laptop-20261024-1118.json`.

**Статус: закрыто.** Ограничение: вживую проверено не больше 4 камер. Предел здесь — CPU контейнера (1.18 из 1.5 ядра), а не TPU: батчер det собирает в среднем 1–3 кадра. Для большего числа камер нужен декод только анализируемых кадров и общий батч det (`docs/EDGE.md` §7.3 п. 5).

## Паттерны нарушений из кейса

Результат — офлайн-прогон всего конвейера на CPU по 13 роликам: `eval-onnx-cpu-phone-nearest-stress-20261024-1342.json` → `by_type`. Задержка считается от момента начала нарушения по разметке. Колонка «Живой edge» — сколько событий дала камера стенда на TPU за 300 с (`edge-tpu-20261024-1002.json` → `events.by_type`). Ролики там зациклены, поэтому эта колонка показывает, что сценарий работает на TPU, а не точность.

| Паттерн | Плагин | Модели (все на TPU) | Данные для проверки | Результат на роликах | Живой edge | Тесты | Что в работе |
|---|---|---|---|---|---|---|---|
| Опасная зона с расписанием | S1 `zone_intrusion` | det 480 | s1_01, s1_02: 4 события и 3 спорных | **4 из 4**, ложных 0, задержка 2.0 с (равна окну) | cam_s1: 15 | `tests/test_rules_zone_intrusion.py::test_person_foot_in_zone_gives_violation` | — |
| СИЗ: каска, жилет, респиратор | S2 `ppe_missing` | det 480 + ppe 640; сверка с CPU R 0.91 / P 0.95 | s2_01, s2_02: 6 событий и 2 спорных; 40 фото RF100 | каска **5 из 6**, ложных 0, задержка p50 6.2 с, p95 16.0 с. Жилет по умолчанию выключен: все 3 ложные тревоги базы были по жилету. Респиратора нет: класса нет | cam_s2: 5 | `tests/test_rules_ppe_missing.py::test_missing_helmet_in_zone` | <<ds-ppe: задержка S2 и AP no_helmet до/после>> |
| P1. Бег в зоне техники | S5 `running` | det 480 + трекер (скорость в высотах тела в секунду) | демо-ролика нет: проверенные стоки не подошли (`docs/DATA.md`, «Чего нет») | **нет данных** | — | `tests/test_rules_running.py::test_running_speed_violates` | своя съёмка (`docs/DATA.md`, «Чего нет и чем заменить») |
| P2. Сближение с движущейся техникой | S4 `vehicle_proximity` | det 480 (`car`, `truck`, `bus`). Классы техники из ppe в трекер не попадают (`docs/QUALITY.md`, рекомендация 2) | s4_01, s4_02, s4_03: 4 события и 3 спорных | **2 из 4**, ложных 0, задержка p50 0.9 с. Оба пропуска — погрузчик на s4_02: такого класса нет | cam_s4: 8 | `tests/test_rules_vehicle_proximity.py::test_moving_vehicle_nearby` | <<ds-forklift: S4 до/после, s4_02>> |
| P3. Телефон | S6 `phone_usage` | det 480 (`cell phone`) + pose 480 | s6_01: 1 событие и 1 спорное | **1 из 1**, задержка 2.2 с. Стресс-тест: правило включено на всех 13 роликах — 0 ложных | не на демо-камере | `tests/test_tracking_attributes.py::test_phone_at_ear_uses_nearest_head_point` | — |
| P4. Курение или открытый огонь | S7 `smoking_fire` | fire 640 (`fire_640s1`); сверка с CPU R 0.87 / P 0.93 | s7_01 (курение); 10 открытых кадров с огнём и 10 без (`docs/MODELS.md`) | курение **0 из 1**, не заявляем. Огонь: найден на 9 кадрах из 10, на 10 кадрах без огня ложных нет (CPU). Роликов с огнём в оценке нет | не на демо-камере | `tests/test_rules_smoking_fire.py::test_fire_in_no_smoking_zone` | <<ds-smoking: сигарета, s7_01 и новые ролики>> |
| P5. Перелезание через ограждение | S8 `fence_climb` | det 480 + pose 480 | демо-ролика нет: стоки — паркур, а не забор (`docs/DATA.md`) | **нет данных** | — | `tests/test_rules_fence_climb.py::test_crawling_from_head_position` | своя съёмка |
| P6. Работа на высоте без привязи | S9 `height_no_harness` | det 480; модели привязи нет | демо-ролика нет | **нет данных**. В режиме `strict` правило работает как контроль доступа в зону высоты | — | `tests/test_rules_height_no_harness.py::test_strict_is_access_control_for_unknown_harness` | — |
| P7. Падение и неподвижность | S3 `fall_immobile` | det 480 + pose 480 | s3_01 (падение в снег), s3_02 (6 постановочных падений GMDCSA24) | **6 из 7**, ложных 0, задержка p50 2.4 с, p95 4.15 с. Пропуск — падение с кровати: детектор видит лежащего урывками | cam_s3: 13 | `tests/test_rules_fall_immobile.py::test_lost_fall_with_tracker_attributes_and_event_manager` | <<ds-falls: recall лежачих и падений, ложные на ADL до/после>> |
| P8. Переноска груза | S10 `unsafe_carry` | det 480 + pose 480, осторожные эвристики | демо-ролика нет | **нет данных** | — | `tests/test_rules_unsafe_carry.py::test_carry_in_active_danger_zone` | — |

Итого:
- На роликах проверены 6 типов: зона, СИЗ, падение, техника, телефон, курение.
- Уверенно заявляем зону, каску, падение и технику с оговоркой про погрузчик. Телефон — по одному событию.
- Курение не заявляем. Для бега, перелезания, высоты и переноски груза роликов нет: только юнит-тесты на синтетике.

## Как перепроверить

```bash
unset PYTHONPATH

# Все тесты, на которые ссылается матрица
grep -o 'tests/test_[a-z0-9_]*\.py::[a-z0-9_]*' docs/REQUIREMENTS_MATRIX.md | sort -u | xargs .venv/bin/python -m pytest -q

# Файлы замеров хранятся вне репозитория: укажите их каталог
export M=<каталог с файлами замеров>

# Итоговые цифры качества по событиям
.venv/bin/python -c "import json, os; d = json.load(open(os.path.join(os.environ['M'], 'eval-onnx-cpu-phone-nearest-stress-20261024-1342.json'))); print(d['overall']); print(d['by_type'])"

# Живой edge на TPU: задержка, к/с, CPU
.venv/bin/python -c "import json, os; d = json.load(open(os.path.join(os.environ['M'], 'edge-tpu-20261024-1002.json'))); print(d['events']['e2e_from_started_s']); print({c: v['fps_processed']['p50'] for c, v in d['fps'].items()}); print(d['cpu']['container_cores'])"
```

Методика цифр описана в `docs/QUALITY.md` (качество), `docs/EDGE.md` (edge и задержка) и `docs/BENCHMARKS.md` (TPU и CPU).
