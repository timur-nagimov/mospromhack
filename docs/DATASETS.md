# Открытые наборы под слабые места

Решение внутреннее, для демонстрации. Лицензии наборов не проверяем (решение человека 24.10), но источник каждого
набора записываем. Наборы лежат в `data/datasets/<имя>/` в YOLO-формате с `data.yaml` и `manifest.json`. Каталог
`data` — симлинк на `mosprom-heavy.noindex`, он не в git. Скачивание и конвертация — `ml/datasets/<имя>.py`.

**Отложенная выборка.** Кадры роликов из `ml/eval/scenarios.yaml` в обучение не попадают. Тестовые части наборов
используются только для оценки. Разбиение наборов падений — по людям и сериям, а не по кадрам.

Слабые места, под которые подбирались наборы (`docs/QUALITY.md`, разбор ошибок):

| Сценарий | Слабое место | Сейчас на роликах |
|---|---|---|
| S4 | погрузчика нет в классах моделей | 2 из 4 |
| S3 | детектор теряет лежащего человека: падение с кровати, человек на снегу | 6 из 7 |
| S7 | сигарету модель огня не видит | 0 из 1, не заявляем |
| S2 | каска у людей в кадре с первой секунды распознаётся с задержкой до 18 с; `no_helmet` AP50 0.135 | 5 из 6 |

## Погрузчик (S4)

_Раздел заполняет ds-forklift._

## Падения и лежащий человек (S3)

Скрипт — `ml/datasets/falls.py`, данные — `data/datasets/falls/` (`raw/`, `manifest.json`, `frames/`, `yolo/`).

| Набор | Источник | Что взяли | Как делили |
|---|---|---|---|
| UR Fall Detection (URFD) | http://fenix.ur.edu.pl/~mkepski/ds/uf.html | RGB камеры 0 (PNG 640×480, 30 к/с): 30 падений и 40 ADL; покадровые метки `urfall-cam0-{falls,adls}.csv`: −1 не лежит, 0 падает, 1 лежит | По людям, различённым по одежде: 5 человек в падениях (по 6 серий), в ADL ещё 5. **test** — люди P2, P5, B, E (12 падений, 20 ADL, в том числе «лёг на диван» и «лёг на кровать»); val — P4 |
| CAUCAFall | https://data.mendeley.com/datasets/7w7fccy7ky (v5) | 100 AVI 720×480, 20 к/с: 10 человек × 5 падений и 5 ADL (ходьба, прыжки, подъём предмета, сел, встал на колени). YOLO-разметка кадров `nofall/fall` — каждый 4-й кадр из общего zip по HTTP Range | По людям: **test** — 8, 9, 10; val — 7; train — 1–6 |
| GMDCSA24 | https://github.com/ekramalam/GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos | Subject 1 и 4: 33 падения и 36 ADL (сон на кровати, отжимания, сидение на полу, чтение) с интервалами действий из `Subject N/{Fall,ADL}.csv` | Только **test**. Subject 2 и 3 — источник s3_02, не скачиваются |
| COCO val2017 | http://images.cocodataset.org/zips/val2017.zip | 1500 изображений с людьми и ключевыми точками — повтор против забывания при дообучении pose | 200 — val, остальное — train |
| Le2i | — | **Не нашли зеркала без авторизации.** le2i.cnrs.fr не отвечает, страница imvia.u-bourgogne.fr — 404, Kaggle требует вход. На Hugging Face есть только `seanphan/le2i-sentinel-frames`: 3 кадра на ролик, для обучения и событийной оценки не годится | — |

Правила разбиения:
- один человек — только в одной части: train, val или test;
- тестовые части в обучение не попадают, кадры роликов `ml/eval/scenarios.yaml` — тоже (GMDCSA24 Subject 2 и 3 не скачиваются);
- test URFD — 40 % падений и 50 % ADL, CAUCAFall — 30 % людей, GMDCSA24 — целиком.

**Отложенная событийная выборка** (`falls.py holdout`): ролики `data/videos/falls_{urfd,gmdcsa,cauca}/*.mp4`,
метки `data/labels/falls/*.json`, длительности `data/durations_falls.json`, сценарии `ml/eval/scenarios_falls.yaml`.
- Падение — одно событие `fall_immobile` от начала падения до конца ролика.
- Событие достоверно, если после начала падения в ролике видно не меньше 3 с (около 1 с падение и ≥ 2 с лежания).
  Иначе метка `uncertain`: о неподвижности по ролику судить нельзя. У URFD ролики обрываются через 1–2 с после
  падения, поэтому достоверно только одно падение из 12.
- Падение на кровать (GMDCSA24 Subject 4) — `uncertain`: правило не отличает его от ADL «лёг на кровать».
- У ADL событий нет: любая тревога — ложная. В метке записан вид действия (`activity`): `lying_bed`, `lying_sofa`,
  `floor_activity` (на полу: отжимания, на четвереньках у кровати), `walk_sit_*`.

Команды (из корня репозитория, `GUARD_DATA_DIR` — каталог данных):
```bash
python ml/datasets/falls.py download                 # URFD, CAUCAFall, GMDCSA24 S1/S4, COCO, учитель yolov8x-pose
python ml/datasets/falls.py holdout                  # ролики, метки, длительности, scenarios_falls.yaml
python ml/datasets/falls.py recall --model det_n480 --model pose_n480   # покадровый recall лежачих
python ml/datasets/falls.py report <прогон run_offline>.json            # разбор по источникам и видам ADL
python ml/datasets/falls.py prepare && tools/heavy.sh python ml/datasets/falls.py teacher && python ml/datasets/falls.py yolo
```

## Сигарета (S7)

_Раздел заполняет ds-smoking._

## СИЗ (S2): SH17 и Construction-PPE

_Раздел заполняет ds-ppe._
