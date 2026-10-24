# Локальная проверка D-FINE, PPE и YOLO11 head/tail — 24.10.2026

На текущем диагностическом наборе замена модели по умолчанию не обоснована.
D-FINE-X увеличил person recall при снижении precision. BuildVision дал больше
S2-событий, чем Vyra, однако анализ выявил перенос tracker/event state через
склейку двух независимых источников. Его влияние проверяется отдельно.
Приоритет — качество; скорость CPU здесь описывает только локальный запуск.
Целевое развёртывание остаётся на сервере с внешним LinQ TPU.

Код всех пяти запусков: `bbfe8ab414a9fb710aa515a54bf766ac7d5f37ea`.
Машиночитаемый отчёт с 97 artifact references, SHA256, командами, знаменателями
и полным readback: [.orchestra/metrics/model-followup-20261024.json](../.orchestra/metrics/model-followup-20261024.json).
Исходные результаты находятся в `data/model_followup_evaluation_20261024/`.
Публикации и новой проверки внешнего TPU не было.

## D-FINE-X на изображениях

Обе модели прошли **141/141** исходных Construction-PPE test images без ошибок;
person GT — **236** объектов. Fingerprint набора
`800a03725b2c6fc37b8c80794735a03666fd424cc63b4eeecb54b548e37b6791`.
AP50 — COCO 101-point, working confidence .35 и IoU .5 зафиксированы заранее.

| Модель | Person AP50 | Precision при .35 | Recall при .35 |
|---|---:|---:|---:|
| YOLOv8n 640, frozen CPU baseline | .757 | .736 | .780 |
| D-FINE-X Objects365→COCO, FP32 ONNX | .771 | .517 | .856 |

Это сравнение архитектур с собственными processor/decoder: у D-FINE прямой PIL
resize и focal top-Q без NMS; у YOLO letterbox и NMS. Все метрики пересчитаны
из сохранённых postprocessed predictions и совпали с исходным отчётом целиком.
Person-only модели не поддерживают helmet/vest; общий core denominator не
уменьшался. Public test и неизвестное пересечение с обучением ограничивают
интерпретацию. Результат не измеряет event quality, no_vest или respirator.

Артефакт и runtime contract: [DFINE_DETECTOR_ADAPTER.md](DFINE_DETECTOR_ADAPTER.md).
Численный паритет community ONNX с исходными авторскими весами не установлен.
Динамические оси, GridSample и attention требуют самостоятельной TPU-проверки.

## Полные S2-видео

Подготовлены все четыре ролика: **2488** кадров CFR15, **165,866667 с**, затем
фиксированные **831/831** наблюдений при 5 FPS. COCO det и PPE вызваны по 831 раз
для каждого кандидата, batch1, два CPU threads, ошибок нет. Все sample coordinates,
COCO predictions и треки до атрибутов побайтно/семантически совпали в паре;
831 identity hash пересчитан для каждого кандидата. Код и все входные hashes
сохранялись во время прогонов.

| PPE | TP | FP | FN | Дубли | Precision | Recall | Median / p95 задержки, с |
|---|---:|---:|---:|---:|---:|---:|---:|
| BuildVision v0.3 | 5 | 0 | 1 | 0 | 1.0 | 5/6 | 4.6 / 15.84 |
| Vyra | 3 | 0 | 3 | 0 | 1.0 | 3/6 | 11.2 / 15.52 |

Основной helmet-only scope содержит **6 certain / 0 uncertain** меток.
Две явно закреплённые vest-only uncertain метки исключены из этого scope;
исходный legacy scope сохранён отдельным вторичным результатом. Независимый
CLI `ml.eval.events_eval --db` прочёл SQLite для обоих scopes и точно воспроизвёл
весь evaluator JSON. Полные строки SQLite совпали с `journal_events.json`,
а ID — с confirmation predictions. Confirmation snapshot может иметь
`ended_at=null`, когда финальная запись журнала уже закрыта; это разные моменты.

У обоих кандидатов 0 FP на **68,933333 с** размеченных clean-видео. Такой объём
не устанавливает производственную частоту ложных тревог. Actor IDs отсутствуют;
совпадение по времени/типу не доказывает правильную привязку к человеку.

Оригинальный manifest сохранён с SHA
`06642131278f2b484b3c7b859600f1d4e55c95676b8e51e26481da019e848449`.
Исходный encode recipe задавал 25 FPS / 4 threads; применён явный derived recipe
15 FPS / 2 threads для frozen S2 runner. Оба recipe и hashes записаны в
`data/s2_public_20261024/encode-report.json`. Это не изменение исходного manifest.

### Найденное смешение источников

`s2_02` склеивает склад и стройку. Точным повторением fps/scale/pad/setsar+concat
на исходных bytes установлены **301 + 253 = 554** кадра. Первый кадр стройки —
zero-based **301**, время **301/15 = 20,066667 с**. Контрольные суммы joined300
и joined301 совпали с последним кадром первой и первым кадром второй части.
Граница 20.1 в GT округлена и не использовалась как основание reset.

При 5 FPS старый sample100 относится к frame300/20.0 с, новый sample101 —
к frame303/20.2 с. В 29.8 с BuildVision выдаёт `no_helmet=.7278`, атрибут
`helmet=False` и S2 violation, но track4 с `first_seen=0` переоткрывает старое
событие, подтверждённое в 2.8 с. Нового confirmation prediction для другого
человека не возникает. Этот FN содержит системную причину и не может быть
приписан только распознаванию PPE.

Доказательство: `data/s2_public_20261024/source-boundary-verification.json`,
SHA `4043bf7f430b04f131a0ef065a55408dfa43317cc82a9ed3193bbb6d844f684d`.
Исходные **5/6** остаются primary. Диагностический replay из сохранённых
postprocessed detections должен сначала воспроизвести baseline, затем отдельно
проверить reset на доказанной границе. Production capture_id reset и replay
развиваются в отдельных ветках; автоматического scene-cut detector этот вывод
не устанавливает.

## YOLO11x: FP32 graph split

На трёх synthetic inputs (zeros, ones, uniform seed0) и одном настоящем
изображении полный ONNX и выделенный head→tail дали одинаковые FP32 outputs:
**4/4**, по **705600** элементов `[1,84,8400]`, `max_abs_diff=0`, failing0.
Обнаружен canonical head `.23`; tail11 nodes не зависит от исходного image input.
Артефакты в `data/model_followup_evaluation_20261024/tpu-parity-yolo11x/`.

Это SDK-free CPU parity. TPU compilation `not_attempted`, совместимость
`unknown`, quantized parity `not_checked`. Численный FP32 успех не подтверждает
INT8 calibration, внешнее выполнение или улучшение качества событий.

## Текущая интеграция и следующий эксперимент

До этого пакета объединены актуальный `origin/main=71302c7`, исследовательский
каталог, YOLO11x, D-FINE adapter и TPU preflight. Полный backend suite:
**1733 passed**, failures/errors/skips0; один прежний Starlette/httpx warning.
JUnit: `data/integration_validation_20261024/post-model-features.xml`.

Следующий приоритет — устранить перенос состояния между capture sessions и
проверить отдельный replay. Затем person ROI PPE при тех же весах/порогах;
другие семейства, pose и temporal specialists перечислены в
[MODEL_FAMILIES_RESEARCH_20261024.md](MODEL_FAMILIES_RESEARCH_20261024.md).
Модель/порог выбираются на отдельном dev, с итоговой source/subject-disjoint
проверкой. Лицензия не ограничивает этот личный эксперимент; происхождение
артефактов продолжает фиксироваться.

Продолжение после этого snapshot: capture-session fix и cached replay уже
объединены, полный suite **1775 passed**. На тех же detections доказанный
source-boundary treatment даёт BuildVision6/6 при сохранённом primary5/6;
Vyra остаётся3/6. Подробности и независимый readback:
[S2_SOURCE_BOUNDARY_DIAGNOSTIC.md](S2_SOURCE_BOUNDARY_DIAGNOSTIC.md).
Исследование новых RF-DETR/DEIMv2/Grounding DINO:
[EXTENDED_MODEL_FAMILIES_20261024.md](EXTENDED_MODEL_FAMILIES_20261024.md).
