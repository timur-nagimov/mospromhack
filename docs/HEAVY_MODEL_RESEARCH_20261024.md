# Тяжёлые и специализированные модели — 24.10.2026

Критерий выбора — качество системы. Время CPU на ноутбуке служит диагностикой,
не является гейтом. Целевой runtime — сервер и внешний LinQ TPU; CPU tail после
TPU допустим. Проверка ONNX на CPU не подтверждает компиляцию или инференс TPU.

Машиночитаемый отчёт с SHA256 артефактов:
`.orchestra/metrics/heavy-model-research-20261024.json`. Все замеры ниже сделаны
**до объединения upstream 71302c7**, на указанных в отчёте commit/config/data.
Обновлённые temporal/lifecycle правила требуют новых парных прогонов.

## Завершённое сравнение

S1: по 600 source-time observations на модель, два видео общей длительностью
120 секунд, 4 certain и 3 uncertain события. Параметры и исходный evaluator
одинаковые. Независимых clean-видео и actor identity GT в этом протоколе нет.

| Модель | TP / FP / FN / дубли | Strict precision / recall |
|---|---|---|
| YOLOv8n-640 | 3 / 0 / 1 / 1 | 0.75 / 0.75 |
| YOLOv8x-640 | 3 / 0 / 1 / 1 | 0.75 / 0.75 |

У x поздний человек и нарушение присутствуют во всех 60 observations интервала
48–60 с, но новое событие не создаётся: уже открыт инцидент, начавшийся раньше.
Поэтому FN текущего matcher не означает пропуск человека детектором. Выполнены
независимый запуск CLI evaluator и readback SQLite обоих прогонов.

Construction-PPE: одинаковые **141 test image**, GT person 236, helmet 192,
vest 178, no_helmet 40. AP50 — COCO 101 point; рабочий conf 0.35, IoU 0.5.
Пороги на test не подбирались. Training overlap публичных PPE весов неизвестен.

| Модель | person AP50 | helmet AP50 | vest AP50 | no_helmet AP50 |
|---|---:|---:|---:|---:|
| YOLOv8n-640 COCO | 0.757 | unsupported | unsupported | unsupported |
| YOLOv8x-640 COCO | 0.761 | unsupported | unsupported | unsupported |
| BuildVision / constructelligence v0.3 | 0.791 | 0.895 | 0.847 | 0.136 |
| Vyra YOLOv8m PPE | 0.026 | 0.543 | 0.586 | 0.054 |

На этих данных Vyra не улучшает текущий PPE-кандидат. No_helmet рабочие P/R:
BuildVision 0.25/0.20, Vyra 0.167/0.05. В GT неоднородные head/face boxes и
возможные пробелы; no_vest GT отсутствует, respirator не поддерживается.
Image AP не является оценкой S2 событий. Still-image semantic smoke сохраняет
unknown; отрицательный атрибут требует истории трека и подтверждения правила.

YOLOv8x-1280 экспортирован: статический `[1,3,1280,1280]` → `[1,84,33600]`,
ONNX checker и реальный zero-input ORT прошли. Его quality job остановлен по
смене приоритета на слияние/TPU; частичный результат не используется как метрика.
У завершённых jobs сохранены raw predictions, GT, конфиги, логи и provenance.

## Более новые и событийные кандидаты

| Кандидат | Для чего исследовать | Следующий проверяемый шаг |
|---|---|---|
| YOLO11x detect | люди и общие объекты | статический ONNX, явный head/tail, CPU parity, затем LinQ compiler |
| YOLO11x pose | S3, падение/неподвижность | keypoint quality и ADL negatives; pose AP не равен recall падений |
| YOLO26 detect/pose | новая архитектура и точность мелких объектов | отдельное export-окружение и проверка head/output; DFL preset v8 неприменим |
| SafetyVision v2 | явные negative PPE классы | сверить mapping и независимые S2 видео; не выбирать по авторскому AP |
| FireSmoke R3-E6 | отдельные fire/smoke события | проверить источник, классы Fire/Smoke и head 768; затем hard negatives |
| Специалисты forklift / cigarette / lying | слабые S4 / S7 / S3 | целевой frozen video test с погрузчиком в темноте, инструментами/рациями, ADL |

Официальный [YOLO11](https://docs.ultralytics.com/models/yolo11/) поддерживает
detect и pose. Для первого исследования скачан официальный YOLO11x:
HF `Ultralytics/YOLO11`, revision `5e0da476eb5def45e8080bd4b92ea63f0b16974c`,
114636239 байт, SHA256
`7bc158aa95c0ebfdd87f70f01653c1131b93e92522dbe15c228bcd742e773a24`.
Это подтверждённая загрузка, не quality/TPU результат.

У [YOLO26](https://docs.ultralytics.com/models/yolo26/) удалён DFL и изменены
режимы головы. Его нельзя подключать сменой имени файла в YOLOv8 preset.
Для [Vyra](https://huggingface.co/Hexmon/vyra-yolo-ppe-detection) source/mapping
и конфликт license metadata сохранены в candidate provenance; Mask не считается
respirator, Fall-Detected не является temporal fall event.

## Серверная приёмка

`configs/models.yaml` и `configs/models.stand.yaml` сохраняют production TPU
refs; `configs/models.cpu-baseline.yaml` закрепляет локальный CPU baseline.
Выбранный `linq-tpu` по умолчанию не переходит на CPU. Для уже готовых
VPN-forward предусмотрен `GUARD_VPN_MODE=external scripts/run_tpu.sh`.

Свежая проверка профиля `~/Downloads/LinQ/Hakathon.ovpn` дала два Connection
reset без туннеля. Пользователь после этого поручил продолжать локально с
прицелом на сервер. Новые remote-попытки отложены; исторические TPU замеры main
сохранены отдельно и не выдаются за приёмку объединённого кода.

Promotion кандидата требует: закреплённые weights/data; FP32 head+tail parity;
компиляцию LinQ с репрезентативной calibration; per-class quantization quality;
фиксированный event benchmark с clean negatives; фактические TPU counters и
edge→gateway readback на сервере. Серверный smoke должен проверять активный
backend и вызовы каждой модели, а не только PID процесса или доступность HTTP.
