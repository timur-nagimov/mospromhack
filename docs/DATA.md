# Данные и воспроизводимость проверок

Production и локальные эксперименты используют разные наборы. Ниже сохранены
манифест и 13-видео протокол upstream `e23ff4f`, затем результаты локального CPU
пакета. Наличие файлов в прежнем запуске не означает наличие в данном checkout.
Замеры 13 роликов и фиксированного S1 CPU протокола имеют разные параметры и
знаменатели; сравнивать их как A/B нельзя.

Пять raw-источников S2/clean для нового frozen прогона скачаны и проверены по
SHA256 и размеру в `data/s2_public_20261024/`. Encode и итоговый S2 event benchmark
ещё не завершены. Публичный test не используется для подбора порогов.

## Снимок upstream: модели и данные стенда

## Данные для демо и оценки качества

Ролики, кадры и разметка лежат в `data/` и в git не попадают. В git хранятся только
манифест `tools/data_manifest.json` (источники, sha256, отрезки, разметка, полигоны зон) и скрипт
`tools/fetch_data.py`, который всё воспроизводит.

### Как получить

```bash
.venv/bin/python tools/fetch_data.py                  # скачать, перекодировать, записать разметку, кадры и картинки
.venv/bin/python tools/fetch_data.py --list           # список роликов манифеста
.venv/bin/python tools/fetch_data.py --only s1_01 --steps download encode labels
.venv/bin/python tools/fetch_data.py --data-dir /path/to/data --heavy /path/to/tools/heavy.sh
```

- Шаги: `download` → `encode` → `labels` → `frames` → `checks` → `images`.
- Исходники кэшируются в `data/raw/`, повторно не скачиваются, sha256 сверяется с манифестом.
- ffmpeg из imageio-ffmpeg запускается по одному файлу через `tools/heavy.sh` (очередь и проверка памяти).
- Перекодирование детерминированное: повторный прогон `s1_01` с нуля в пустой каталог дал файл с тем же md5.
- Без авторизации. Всё скачивается из Mixkit CDN, GitHub raw и Hugging Face; из РФ доступно (Pexels и Pixabay из РФ закрыты).

### Что получается в `data/`

| Путь | Что |
|---|---|
| `videos/<сценарий>_<nn>.mp4` | H.264, ≤1280×720, 25 fps, CRF 28, без звука, `+faststart`; 13 роликов, 535 с, 55 МБ |
| `labels/<stem>.json` | разметка событий, формат `ml/eval/events_eval.py` (ниже) |
| `durations.json` | `{"<video>": секунды}` для `--durations` |
| `images/<сценарий>/` | стоп-кадры из роликов для режима «наборы изображений»: s1 48, s2 49, s3 46, s4 45, clean 35, s6 23, s7 35 |
| `images/s2_ppe/`, `image_labels/s2_ppe.json` | 40 фото со стройки: на каждом есть люди в касках и без (CC BY 4.0, RF100) |
| `media/data_checks/` | 5–6 контрольных кадров на ролик; все просмотрены глазами |
| `raw/` | кэш исходников; можно удалить |

### Ролики

Длительность — после перекодирования. В графе «Разметка» — достоверные события и в скобках `uncertain`.

| Файл | Сценарий | Источник | Лицензия | Длит., с | Разрешение | Что происходит | Разметка |
|---|---|---|---|---|---|---|---|
| s1_01.mp4 | S1 зона | [Intel sample-videos: worker-zone-detection](https://github.com/intel-iot-devkit/sample-videos), 0–60 с | CC BY 4.0 | 60.0 | 1280×720 | Статичная камера в пустом цехе, на полу разметка и жёлто-чёрная лента. 1–2 рабочих в касках и жилетах ходят, один ~11 с стоит за лентой | 2 (1) + полигон |
| s1_02.mp4 | S1 зона | [Intel sample-videos: store-aisle-detection](https://github.com/intel-iot-devkit/sample-videos), 0–60 с | CC BY 4.0 | 60.0 | 720×404 | Статичная камера над проходом между стеллажами, 2–6 человек, перекрытия. Люди заходят в дальний конец правого прохода | 2 (2) + полигон |
| s2_01.mp4 | S2 СИЗ | [Intel safety-gear-detector: Safety_Full_Hat_and_Vest.mp4](https://github.com/intel-iot-devkit/safety-gear-detector-python), 0–60 с | BSD-3-Clause (лицензия репозитория) | 60.0 | 1280×720 | Статичная камера в коридоре. Большинство в касках и жилетах; женщина без каски и жилета, девушка без каски, мужчина без жилета | 2 (1) |
| s2_02.mp4 | S2 СИЗ | Склейка Mixkit [23237](https://mixkit.co/free-stock-video/business-people-walking-in-the-warehouse-23237/) + [24049](https://mixkit.co/free-stock-video/quality-control-check-at-construction-site-24049/) | Mixkit Free License | 36.9 | 1280×720 | 0–20.1 с: склад, рабочий в каске и трое в костюмах без касок. 20.1–36.9 с: стройка, трое в касках и женщина без каски | 4 (1) |
| s3_01.mp4 | S3 падение | [Mixkit 23814](https://mixkit.co/free-stock-video/tired-man-falls-in-the-snow-23814/) | Mixkit Free License | 21.7 | 1280×720 | Статичная камера, снежное поле. Человек идёт, на 16.8 с падает и лежит неподвижно до конца | 1 (1) |
| s3_02.mp4 | S3 падение | Склейка 6 роликов [GMDCSA24](https://github.com/ekramalam/GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos): Subject 3/07, 2/10, 3/05, 2/04, 3/18, 2/13 | MIT (цитировать датасет, doi:10.5281/zenodo.12921216) | 60.2 | 1280×720 | Комната, статичная камера. Шесть постановочных падений, после каждого лежание 4–7 с. Склейки на 9.0, 20.0, 30.0, 40.0, 51.5 с | 6 |
| s4_01.mp4 | S4 техника | [Mixkit 23852](https://mixkit.co/free-stock-video/working-giving-directions-to-freight-truck-23852/) | Mixkit Free License | 20.1 | 1280×720 | Рабочий в каске стоит в проезде между двумя фурами и направляет движущуюся фуру. Камера ручная | 1 |
| s4_02.mp4 | S4 техника | Склейка Mixkit [23174](https://mixkit.co/free-stock-video/worker-leading-a-freight-truck-in-the-parking-23174/) + [22983](https://mixkit.co/free-stock-video/loading-a-cargo-truck-22983/) | Mixkit Free License | 34.3 | 1280×720 | Грузовик подъезжает вплотную к рабочему. Затем погрузчик с паллетой проезжает в 1–2 м от рабочего | 3 |
| s4_03.mp4 | S4 техника (пограничный) | [Mixkit 25444](https://mixkit.co/free-stock-video/excavator-loading-up-a-truck-25444/) | Mixkit Free License | 55.0 | 1280×720 | Статичная камера сверху на котлован: экскаватор грузит самосвал, рабочие идут по краю в 1.5–3 ростах от техники | (3) + полигон |
| clean_01.mp4 | чистый | [Mixkit 23107](https://mixkit.co/free-stock-video/group-of-workers-in-an-electrical-plant-23107/) | Mixkit Free License | 34.3 | 1280×720 | Подстанция, камера на штативе. Четверо инженеров в касках разговаривают, на фоне двое в касках | пусто |
| clean_02.mp4 | чистый | [Mixkit 23224](https://mixkit.co/free-stock-video/workers-conversation-on-a-construction-site-23224/) | Mixkit Free License | 34.6 | 1280×720 | Стройплощадка: двое в касках идут к камере и разговаривают | пусто |
| s6_01.mp4 | S6 телефон | [Mixkit 24078](https://mixkit.co/free-stock-video/young-construction-worker-talking-on-the-phone-24078/) | Mixkit Free License | 22.6 | 1280×720 | Строитель в каске и жилете говорит по телефону 0–16 с, потом держит телефон у пояса | 1 (1) |
| s7_01.mp4 | S7 курение | [Mixkit 414](https://mixkit.co/free-stock-video/man-smoking-in-an-abandoned-building-414/) | Mixkit Free License | 35.1 | 1280×720 | Заброшенное здание, мужчина курит, на 19–25 с виден дым. Несколько планов | 1 |

Итого по S1–S4 — 9 роликов, «чистых» — 2. Достоверных событий 23, `uncertain` — 10.

| type | Достоверных | uncertain | Роликов |
|---|---|---|---|
| zone_intrusion | 4 | 3 | 2 |
| ppe_missing | 6 | 2 | 2 |
| fall_immobile | 7 | 1 | 2 |
| vehicle_proximity | 4 | 3 | 3 |
| phone_usage | 1 | 1 | 1 |
| smoking_fire | 1 | 0 | 1 |

Нагрузка для ложных срабатываний: «чистые» ролики — 68.9 с. Все 13 роликов — 535 с. Для FP/час можно
считать и по всем роликам, потому что в них размечены все нарушения включённых правил.

### Формат разметки `data/labels/<stem>.json`

```json
{
  "video": "s1_01.mp4",
  "events": [
    {"type": "zone_intrusion", "t_start": 15.4, "t_end": 26.6, "uncertain": false, "note": "Рабочий переходит ленту..."}
  ],
  "scenario": "S1", "description": "...", "rules_to_enable": ["zone_intrusion", "ppe_missing"],
  "zones": [{"zone_id": "behind_tape", "kind": "danger", "polygon": [[0.0, 0.30], "..."], "reference_frame_s": 11.1, "rationale": "..."}],
  "persons": "...", "sources": ["..."], "duration_s": 60.0, "size": [1280, 720]
}
```

- В `events` ровно пять ключей. Их принимает `LabelEvent`: лишний ключ ломает `load_labels`. Положение человека записано в `note`.
- `t_start` — момент начала нарушения: точка опоры вошла в зону, человек упал на пол, человек без каски появился в кадре, ТС приблизилось.
  `t_end` — нарушение закончилось или ролик/отрезок кончился.
- Нарушение СИЗ размечено отдельным событием на каждого человека, интервалы могут совпадать.
- `uncertain: true` ставится на спорные случаи: короткий заход, перекрытие, нарушение только по жилету, пограничное расстояние, низкая поза без падения.
  Совпавшие с ними предсказания не считаются ни TP, ни FP.
- Остальные поля справочные, `events_eval.py` их не читает. `rules_to_enable` — какие правила включать на камере с этим роликом,
  чтобы разметка была полной. Нарушения других типов в ролике не размечались.
- В `data/labels/` должны лежать только файлы разметки: загрузчик читает все `*.json`.

Оценка:

```bash
.venv/bin/python -m ml.eval.events_eval --labels data/labels --db data/guard.db \
  --mapping mapping.json --durations data/durations.json --out metrics.json
```

Проверено: `load_labels` читает все 13 файлов (33 метки). Набор `durations.json` совпадает с размеченными роликами.
На «идеальных» предсказаниях получается P = R = 1.

### Полигоны зон

Координаты нормированные (0..1) от кадра ролика; проверка — по точке опоры (низ-центр рамки).

| Ролик | zone_id | Полигон | Опорный кадр | Обоснование |
|---|---|---|---|---|
| s1_01 | `behind_tape` (danger) | (0.00,0.30) (0.07,0.31) (0.40,0.70) (0.22,1.00) (0.00,1.00) | 11.1 с, цех пуст | Сектор слева от жёлто-чёрной ленты. Лента идёт от (0.22,1.0) к основанию колонны (0.525,0.48). Сверху сектор ограничен красной линией, которая пересекает ленту в (0.40,0.70). Проверено наложением на кадры 20 и 37.6 с |
| s1_02 | `aisle_far_right` (danger) | (0.47,0.22) (0.60,0.22) (0.75,0.56) (0.50,0.56) | 3 с, проход пуст | Пол дальнего конца правого прохода: между правым краем островной выкладки и основанием правого стеллажа, от y = 0.56 до дальней стены. Условно — зона штабелирования. Проверено на 18 кадрах 14–58 с |
| s4_03 | `excavator_pit` (vehicle_path) | (0.14,0.00) (0.52,0.00) (0.56,1.00) (0.10,1.00) | 5 с | Дно котлована между отвалом слева и бровкой справа: там работают экскаватор и самосвал. Камера неподвижна, ориентир — фонарный столб x ≈ 0.43 |
| s4_01, s4_02 | — | — | — | Камера движется (в s4_02 ещё и склейка), поэтому фиксированный полигон не подходит. Правило должно работать по расстоянию между человеком и ТС |

### Наборы изображений

- `images/s1`, `s2`, `s3`, `s4`, `clean`, `s6`, `s7` — стоп-кадры роликов. Шаг 1–4 с, по 23–49 кадров на сценарий. Имя: `<ролик>_t<секунда>.jpg`,
  поэтому время кадра можно сверить с разметкой ролика.
- `images/s2_ppe` — 40 фото из Roboflow 100 «construction-safety» ([зеркало на HF](https://huggingface.co/datasets/LibreYOLO/construction-safety-gsnvb), CC BY 4.0).
  Отобраны кадры, где в одном кадре есть и `helmet`, и `no-helmet`. Рамки лежат в `image_labels/s2_ppe.json`.

### Ограничения и что учесть

- Погрузчика и экскаватора нет в классах COCO. На s4_02 (вторая половина) и s4_03 детектор COCO технику не увидит — нужна модель с соответствующим классом.
  s4_03 целиком пограничный: все события `uncertain`. Ролик годится для проверки ложных срабатываний.
- s1_02: 720×404, плотная толпа и перекрытия. Это стресс-тест трекинга, для показа жюри лучше s1_01.
- s3_01: лежащего на снегу YOLOv8n почти не видит (низкий контраст). Лежание длится ~5 с, поэтому `min_duration` для S3 должен быть ≤ 4 с.
- s3_02: лежание после склеек 4–7 с. Склейка обрывает событие: трек пропадает, и событие закрывается по `close_after_s`.
- S2: нарушение только по жилету помечено `uncertain`. В «чистых» роликах у части людей спецовка вместо жилета, поэтому правило жилета на них не включать.
- Камера движется в s2_02, s4_01, s4_02, clean_02 и частично s7_01 (стоки). Зоны там не задавать.
- Ролики Mixkit нельзя распространять как отдельные стоковые файлы. Поэтому в git они не лежат и скачиваются скриптом.
  Ролики Intel (CC BY 4.0) показывать с атрибуцией «Intel IoT DevKit sample videos, CC BY 4.0».
  GMDCSA24 (MIT) требует ссылки на статью: E. Alam et al., «GMDCSA24: A Dataset for Human Fall Detection in Videos», Data in Brief, 2024.
- `data/videos/a3dev_*` содержит исходники Intel sample-videos (people-detection, store-aisle-detection, worker-zone-detection),
  md5 совпадает с GitHub, лицензия CC BY 4.0. Из них сделаны s1_01 и s1_02.

### Для демо: роли роликов и параметры правил

Проверено по разметке, детекциям YOLOv8n-640 (порог 0.3) и коду правил в `guard/rules/plugins/`.

- **s1_01, S1.** Зону `behind_tape` из `labels/s1_01.json` можно подставлять в `configs/cameras.yaml` без изменений:
  она проходит `Zone.model_validate`, все координаты лежат в 0..1. Ролик — это исходник `a3dev_worker_zone_detection` с отрезком 0–60 с и тем же кадрированием,
  поэтому полигон и времена событий годятся и для исходника на 0–60 с. При `min_duration_s` 2 с подтвердятся события 15.4 и 36.2 с.
  Событие на 50.2 с длится 1.6 с и помечено `uncertain`.
- **s2_01, S2.** Без каски: женщина на 18–25 с и девушка на 37.5–46.5 с. У мужчины на 37.5–46.5 с нет только жилета, событие `uncertain`.
- **s3_01, S3 — риск.** YOLOv8n не находит человека, лежащего на снегу (с 17.4 с детекций нет). Лежание длится ~4.9 с,
  а при параметрах по умолчанию (`min_lying_s` 2 + `immobile_window_s` 3 + `min_duration_s` 2) событие не успеет подтвердиться до конца ролика.
  Для показа лучше s3_02: 0–9 с или 20–30 с (лежащего детектор видит). Параметры: `min_lying_s` 1, `immobile_window_s` 2, `min_duration_s` 1.
- **s4_01, S4 — риск.** Фуры детектируются (`truck`, conf 0.4–0.9), но точка опоры огромной рамки фуры смещается на ~0.01–0.11 её высоты в секунду.
  Это ниже `min_vehicle_speed_bh` = 0.15, и правило может молчать. Для этой камеры поставить `min_vehicle_speed_bh` ≈ 0.03 или показать s4_02 на 0–18 с:
  там подъезжающий грузовик даёт 0.2–0.6 высоты/с.

### Чего нет и чем заменить

- S5 (бег) — подходящего ролика нет. Проверены Mixkit 716 (камера едет за бегуном, скорость относительно кадра ≈ 0) и 21753 (паркур: 0.1–0.5 высоты/с,
  а правилу нужно ≥ 1.8). В `a3dev_people_detection` самый быстрый проход даёт ~0.7 высоты/с. Замена — снять самим 20–30 с бега поперёк кадра со штатива.
- S8 (перелезание) — подходящего ролика нет. Ближе всего Mixkit [21756](https://mixkit.co/free-stock-video/man-jumping-from-a-wall-21756/)
  (человек на стене и прыжок с неё, 13 с) и [21755](https://mixkit.co/free-stock-video/team-parkour-21755/) (перепрыгивают низкие стенки, есть склейки).
  Это паркур, а не забор. Замена — постановочная съёмка: перелезание через ограждение или стеллаж, камера на штативе.
- Падения на производстве у открытых стоков нет. Используются постановочные GMDCSA24 (комната) и падение в снег.
  Для показа лучше записать своё постановочное падение на телефон.
- Огонь (S7) — только курение. Если нужен огонь, подойдёт Mixkit [22587](https://mixkit.co/free-stock-video/building-burning-down-22587/) (горящее здание, 22 с).

## Локальный CPU пакет: отдельный протокол

## Публичные входные данные и реальный CPU smoke

Все три видео скачаны без изменения содержимого из авторского [intel-iot-devkit/sample-videos](https://github.com/intel-iot-devkit/sample-videos), автор Intel Corporation, [CC BY 4.0](https://github.com/intel-iot-devkit/sample-videos/blob/master/LICENSE). Для worker/store URL уже находятся в `tools/data_manifest.json`; people взят из того же официального репозитория. Источник `master` может измениться: SHA256 ниже фиксирует именно проверенные файлы.

Из корня checkout после [подготовки детектора](MODELS.md):

```bash
export GUARD_DATA_DIR="${GUARD_DATA_DIR:-$PWD/data}"
mkdir -p "$GUARD_DATA_DIR/videos/a3dev_worker_zone_detection" \
  "$GUARD_DATA_DIR/videos/a3dev_store_aisle_detection" \
  "$GUARD_DATA_DIR/videos/a3dev_people_detection"
curl --fail --location --output "$GUARD_DATA_DIR/videos/a3dev_worker_zone_detection/worker-zone-detection.mp4" \
  https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/worker-zone-detection.mp4
curl --fail --location --output "$GUARD_DATA_DIR/videos/a3dev_store_aisle_detection/store-aisle-detection.mp4" \
  https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/store-aisle-detection.mp4
curl --fail --location --output "$GUARD_DATA_DIR/videos/a3dev_people_detection/people-detection.mp4" \
  https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/people-detection.mp4
```

### Проверенные файлы 24.10.2026

Пути указаны относительно `GUARD_DATA_DIR`. Видео и результаты не добавляются в git.

| Файл | Байты | Геометрия / FPS / кадров | Проверено |
|---|---:|---|---|
| `videos/a3dev_worker_zone_detection/worker-zone-detection.mp4` | 12 596 893 | 1920×1080 / 59.9401 / 4548 | Декод и реальный S1 inference на первых 60 с |
| `videos/a3dev_store_aisle_detection/store-aisle-detection.mp4` | 9 214 573 | 720×404 / 59.9401 / 3921 | Файл доступен, первый кадр декодирован |
| `videos/a3dev_people_detection/people-detection.mp4` | 5 482 579 | 768×432 / 12 / 596 | Файл доступен, первый кадр декодирован |

SHA256:

```text
worker-zone-detection.mp4 b8b58b8100a81600bc42ee7ee082ced777b12a32a3619139ef289d27cbdcd284
store-aisle-detection.mp4 3526fee39cac70d6366471e4324a6d63b337c8f8ab99c015521bee0d60ed6e04
people-detection.mp4     18ffe8672d741e3e29c9d891d22c59d453720b086c25b35c88b393d55f92f693
```

С общими `GUARD_DATA_DIR` и `GUARD_MODELS_DIR` исходный `configs/cameras.yaml` теперь проходит `python -m guard.doctor --json`: **14 ok, 0 error, 0 warning, 1 skipped**, обязательна только модель `det`. В том CPU snapshot основной конфиг и его полигоны не менялись. Это подтверждает доступность трёх входов, но не качество их существующих зон.

### Реальный S1 smoke

Отдельный `configs/cameras.cpu-demo.yaml` содержит одну камеру, только правило `zone_intrusion` и полигон `behind_tape` из аннотации S1 `tools/data_manifest.json`. Модели/основные камеры не переписываются. Проверка воспроизводится на CPU без поднятия сервера:

```bash
export GUARD_BACKEND=onnx-cpu
export GUARD_CAMERAS_CONFIG="$PWD/configs/cameras.cpu-demo.yaml"
export GUARD_ORT_THREADS=2
python - <<'PY'
import json
import time
import cv2
import numpy as np
import onnxruntime as ort
from guard.contracts import FrameContext
from guard.events.journal import Journal
from guard.events.manager import EventManager
from guard.inference.onnx_cpu import OnnxCpuBackend
from guard.pipeline.config import load_cameras, load_models, resolve_source_uri
from guard.pipeline.detector import Detector
from guard.pipeline.settings import Settings
from guard.rules.base import build_rules, load_plugins
from guard.tracking import Tracker

st = Settings.from_env()
out = st.data_dir / "cpu-smoke-rerun"
db = out / "guard.db"
if db.exists():
    raise SystemExit("Выберите новый выходной каталог: БД уже существует")
out.mkdir(parents=True, exist_ok=True)
ort.disable_telemetry_events()
cv2.setNumThreads(2)
cam = load_cameras(st.cameras_config)[0]
spec = load_models(st.models_config, st.models_dir)["det"]
backend = OnnxCpuBackend(models_dir=st.models_dir, intra_op_threads=2)
backend.load(spec)
detector, tracker = Detector(spec, backend), Tracker(classes={"person"})
load_plugins()
rules, manager = build_rules(cam), EventManager()
cap = cv2.VideoCapture(resolve_source_uri(cam.source.uri, st.data_dir))
assert cap.isOpened()
fps, step = cap.get(cv2.CAP_PROP_FPS), round(cap.get(cv2.CAP_PROP_FPS) / cam.sample_fps)
idx, frames, violation_frames, created, latencies = 0, 0, 0, 0, []
started = time.perf_counter()
with Journal(db) as journal:
    while idx / fps < 60 and cap.grab():
        if idx % step:
            idx += 1
            continue
        ok, frame = cap.retrieve()
        assert ok
        ts, timings = idx / fps, {}
        dets = detector.detect(frame, timings)
        people = tracker.update([d for d in dets if d.cls == "person"], ts)
        ctx = FrameContext(camera=cam, ts=ts, frame_idx=idx, frame=frame,
                           detections={"det": dets}, people=people)
        violations = [v for rule in rules for v in rule.evaluate(ctx)]
        new, closed = manager.process(cam, ts, violations, timings)
        for event in new:
            journal.add(event)
        for event in closed:
            journal.update(event.id, ended_at=event.ended_at)
        frames += 1
        violation_frames += bool(violations)
        created += len(new)
        latencies.append(timings["infer"])
        idx += 1
    rows, total = journal.query(limit=1000)
cap.release()
result = {"frames": frames, "violation_frames": violation_frames, "events_created": created,
          "journal_events": total, "wall_s": time.perf_counter() - started,
          "infer_ms_p50": float(np.percentile(latencies, 50)),
          "infer_ms_p95": float(np.percentile(latencies, 95)),
          "events": [e.model_dump(mode="json") for e in rows]}
(out / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
print(json.dumps({k: v for k, v in result.items() if k != "events"}))
assert created > 0 and total == created
PY
```

Результат выполненного прогона 24.10.2026 на macOS arm64, ORT CPU 2 потока:

| Метрика | Результат |
|---|---:|
| Исходный интервал | Первые 60 с одного оригинального видео |
| Выборка | Каждый 12-й кадр, 4.995 FPS, 300 кадров |
| Детекции / сумма видимых person-треков по кадрам | 298 / 261 |
| Кадры с S1 нарушениями | 84 / 300 |
| Подтверждённые события / SQLite readback | 2 / 2 |
| ORT inference p50 / p95 | 57.38 / 60.45 мс |
| Время всего прогона / обработанные выборки в секунду | 18.99 с / 15.80 FPS |

Два подтверждения появились на 17.4174 и 38.2382 с видео; начало нарушений — 15.4154 и 36.2362 с. Менеджер переоткрывал первое событие в пределах cooldown: его последний `ended_at` — 51.8518 с, второго — 39.6396 с. Было три уведомления о закрытии при двух уникальных событиях. Эти значения показывают существующее объединение/переоткрытие, а не доказанную точность границ инцидента.

Артефакты выполненной проверки в общем каталоге данных: `cpu_demo/smoke-20261024-final.json`, `cpu_demo/smoke-20261024-final.db`, `cpu_demo/inputs-20261024.json`; использованный скрипт сохранён как `cpu_demo/run_smoke.py`. Эти файлы локальны и не входят в git. SHA256 модели в отчёте совпадает с указанным в [MODELS.md](MODELS.md). Предварительная незавершённая БД `cpu_demo/smoke-20261024.db` не является результатом финального прогона.

Проверена цепочка реальных компонентов: видео → ORT → декод/NMS → Tracker → правило S1 → EventManager → SQLite readback. Проверка синхронная и использует временные метки видео. Она не проверяет асинхронный sampler/worker, HTTP/UI, медиа-клипы, несколько камер одновременно, RTSP или TPU. Для store/people сделан только контроль доступа и первого декодированного кадра. Полный quality benchmark, precision/recall, промышленная пригодность и показатели TPU здесь не заявляются.

### Полный локальный асинхронный smoke

Постоянный helper `tools/cpu_smoke.py` проверяет итоговый runtime после объединения изменений. Реальный асинхронный прогон выполнен 24.10.2026 на интеграции `0ba66fe`; результаты выше относятся к отдельной синхронной проверке.

Из корня checkout в окружении с dev-зависимостями и подготовленными файлами:

```bash
export GUARD_DATA_DIR="${GUARD_DATA_DIR:-$PWD/data}"
export GUARD_MODELS_DIR="${GUARD_MODELS_DIR:-$PWD/models}"
GUARD_MODE=local python -m tools.cpu_smoke --data-dir "$GUARD_DATA_DIR" --models-dir "$GUARD_MODELS_DIR" \
  --timeout 60 --json
```

Для worktree можно указать абсолютные пути общих data/models. Поддерживается и прямой запуск `python tools/cpu_smoke.py`; установка проекта editable не требуется. Optional `--output-dir` должен указывать на новый каталог. По умолчанию создаётся уникальный `data/cpu_async/<UTC-время>-<id>/`; существующая БД и результаты не перезаписываются.

Helper использует неизменённый `configs/cameras.cpu-demo.yaml`, реальный `PipelineEngine`/`CameraWorker`, локальный file source, ORT CPU с двумя потоками и FastAPI `TestClient`. Нужны пакеты группы `dev` (`uv pip install --python .venv/bin/python --requirements pyproject.toml --extra dev`). Проверяются:

- запуск реальной камеры без fallback компонентов и одно подтверждённое S1 событие;
- совпадение HTTP/WS ID и неизменяемых полей события; операторский PATCH, его HTTP readback и WS `event_update`;
- JPEG камеры и снапшота через HTTP с декодированием;
- готовность клипа в HTTP/WS, загрузка MP4 через HTTP, H264/AVC codec и декодирование нескольких кадров;
- обработанные кадры, inference calls, ошибки камеры/источника/инференса/media и счётчик записанных клипов.

Все ожидания WS используют общий deadline. Родительский процесс ограничивает общий wall timeout 60 секундами; последние пять секунд отведены на завершение конвейера. При зависании завершается отдельная process group smoke и ffmpeg; принудительное завершение может добавить до двух секунд. Отсутствие события, WS update или декодируемого клипа означает ошибку. Загрузка файлов из сети, RTSP, TPU и внешние HTTP-сервисы не используются.

Итог — один JSON в stdout с `--json` и `report.json` в выходном каталоге. Код `0` означает успешную проверку, `1` — отказ/timeout, `2` — неверный вызов или уже существующий каталог результатов. Сохраняются `runtime.log`, HTTP JSON, наблюдённые WS сообщения, JPEG/MP4 и отдельные DB/media; отчёт содержит версии, SHA256 входов и явные знаменатели обработанных кадров, событий, PATCH и декодированных ответов. При ошибке сохраняется частичный отчёт; он не считается успешным smoke. Даже успешный результат не измеряет качество детекции, многокамерность, браузерный UI или TPU.

#### Выполненная асинхронная приёмка

| Проверка | Результат |
|---|---|
| Общий wall time / deadline | 23.74 / 60 с, exit 0 |
| Камеры / обязательные модели | 1 / 1 (det) |
| Захваченные кадры / обработанные кадры | 1379 / 115 при sample 5 FPS |
| События HTTP / WS / сверенные события | 1 / 1 / 1 |
| PATCH со сверкой HTTP/WS и сохранением | 1 / 1 |
| JPEG через HTTP с декодированием | 2 / 2 (кадр и snapshot, 1920×1080) |
| Клип через HTTP | H264, 187495 байт, 120 декодированных кадров |
| ORT infer p50 / p95 | 65.82 / 68.59 мс, 115 calls |
| Обработка кадра total p50 / p95 | 71.62 / 75.46 мс |
| Ошибки камеры / источника / inference / clips | 0 / 0 / 0 / 0 |

Артефакты: `data/cpu_async/20261024T005619Z-b18f9328/` (вне Git). После штатного завершения отдельное read-only SQLite-соединение подтвердило единственное событие, `status=confirmed`, операторскую заметку и наличие снимка/клипа в отдельном media-каталоге. Аудит: `post-shutdown-readback.json`. Компактная копия результатов, SHA256 отчёта, входы и версии сохранены в `.orchestra/metrics/system-improvements-20261024.json`.

Прогон остановлен после первого события и готового клипа: это не полный проход видео и не измерение recall/precision. Точная привязка границ клипа к окну ±5 с отдельно не сверялась. Браузерный UI, несколько камер, RTSP, AMQP и TPU в этот прогон не входят.
