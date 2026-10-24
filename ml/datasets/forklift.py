"""Набор «погрузчик» для S4: скачивание, конвертация в YOLO, дедуп, разбиение, manifest.json.

Классы: 0 person, 1 forklift. Источники (лицензии не проверяли — решение человека 24.10, демо внутреннее):
- forklift_rf — Roboflow forklift (images.cv), HF-зеркало keremberke/forklift-object-detection: позитивы;
- logistics_rf — Roboflow Logistics v2, HF-подвыборка SergioSCA/Logistics_v2_yolov8: склад, погрузчики, фуры;
- construction_rf — Roboflow Construction Site Safety, keremberke/construction-safety-object-detection:
  экскаваторы, самосвалы, фронтальные погрузчики — негативы;
- excavators_rf100 — RF100 excavators, LibreYOLO/excavators-czvg9: кадры роликов стройки — негативы;
- coco_vehicles — COCO train2017, кадры с truck и bus: негативы. Кадры, в подписях которых есть погрузчик, исключены.

Техника, кроме погрузчика, остаётся без рамок: для модели это фон. Люди размечены как person.
Где разметка людей неполная (construction_rf: каски размечены, люди почти нет; excavators_rf100: класса нет),
людей добавляет учитель — COCO YOLOv8s, conf ≥ 0.5 (шаг teacher, torch — только через tools/heavy.sh).

Разбиение:
- наборы Roboflow без серий — исходные train/valid/test;
- excavators_rf100 — кадры роликов: заново, по ролику (имя до «_mp4-»), чтобы кадры одного ролика не попали
  в разные части;
- coco_vehicles — по хэшу id кадра, 80/10/10.
Дедуп — dHash 64 бит, расстояние Хэмминга ≤ 5. Из пары дублей остаётся кадр более «строгой» части
(test > val > train). Отдельно кадры сверяются с роликами оценки (`ml/eval/scenarios.yaml`, data/videos, 2 к/с):
совпавшие выбрасываются, чтобы отложенные ролики не попали в обучение ни в каком виде.

Примеры:
    python ml/datasets/forklift.py download
    tools/heavy.sh python ml/datasets/forklift.py teacher      # люди для construction_rf и excavators_rf100
    python ml/datasets/forklift.py build
    python ml/datasets/forklift.py all                          # download + teacher + build
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import re
import shutil
import sys
import time
import urllib.request
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CLASSES = ("person", "forklift")
VERSION = "forklift-ds-v1"
SPLITS = ("train", "val", "test")
SPLIT_RANK = {"test": 0, "val": 1, "train": 2}  # кто остаётся из пары дублей
DUP_HAMMING = 5
TEACHER_CONF = 0.5
COCO_IDS = {"person": 0, "car": 2, "bus": 5, "train": 6, "truck": 7}
COCO_CAPTIONS_ZIP = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
COCO_CAPTIONS_MEMBER = "annotations/captions_train2017.json"
COCO_LABELS_ZIP = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco2017labels.zip"
COCO_IMAGE_URL = "http://images.cocodataset.org/train2017/{:012d}.jpg"
FORKLIFT_WORDS = re.compile(r"fork\s*-?\s*lift", re.IGNORECASE)

HF = "https://huggingface.co/datasets/{repo}/resolve/main/{path}"
SOURCES = {
    "forklift_rf": {
        "repo": "keremberke/forklift-object-detection",
        "url": "https://huggingface.co/datasets/keremberke/forklift-object-detection",
        "origin": "https://universe.roboflow.com/mohamed-traore-2ekkp/forklift-dsitv (images.cv)",
        "role": "позитивы: погрузчики и люди",
        "zips": ["data/train.zip", "data/valid.zip", "data/test.zip"],
    },
    "logistics_rf": {
        "repo": "SergioSCA/Logistics_v2_yolov8",
        "url": "https://huggingface.co/datasets/SergioSCA/Logistics_v2_yolov8",
        "origin": "https://universe.roboflow.com/large-benchmark-datasets/logistics-sz9jr/dataset/2",
        "role": "позитивы и негативы: склад, погрузчики, фуры, фургоны",
    },
    "construction_rf": {
        "repo": "keremberke/construction-safety-object-detection",
        "url": "https://huggingface.co/datasets/keremberke/construction-safety-object-detection",
        "origin": "https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety/dataset/1",
        "role": "негативы: экскаваторы, самосвалы, фронтальные погрузчики; люди — учитель",
        "zips": ["data/train.zip", "data/valid.zip", "data/test.zip"],
    },
    "excavators_rf100": {
        "repo": "LibreYOLO/excavators-czvg9",
        "url": "https://huggingface.co/datasets/LibreYOLO/excavators-czvg9",
        "origin": "https://universe.roboflow.com/roboflow-100/excavators-czvg9/dataset/1",
        "role": "негативы: кадры роликов стройки с экскаваторами, самосвалами, погрузчиками; люди — учитель",
    },
    "coco_vehicles": {
        "url": "https://cocodataset.org (train2017)",
        "origin": COCO_LABELS_ZIP,
        "role": "негативы: грузовики и автобусы, люди из разметки COCO",
    },
    "imagenet_forklift": {
        "repo": "mlnomad/imnet1k_forklift",
        "url": "https://huggingface.co/datasets/mlnomad/imnet1k_forklift",
        "origin": "ImageNet-1k, класс n03384352 forklift (train 1300 + validation 50)",
        "role": "позитивы без рамок: рамки погрузчика ставит учитель keremberke/yolov8s-forklift-detection, "
                "людей — COCO YOLOv8s; только в train",
        "files": ["data/train-00000-of-00001.parquet", "data/validation-00000-of-00001.parquet"],
    },
}


@dataclass
class Sample:
    source: str
    split: str
    group: str  # единица разбиения: кадр, ролик или серия
    image: Path
    boxes: list[tuple[int, float, float, float, float]] = field(default_factory=list)  # cls, cx, cy, w, h (0..1)
    dropped: Counter = field(default_factory=Counter)  # объекты, ставшие фоном
    teacher: int = 0  # сколько person добавил учитель
    name: str = ""
    dhash: int = 0


def log(msg: str) -> None:
    print(msg, flush=True)


# ---------- скачивание ----------


def http_get(url: str, dst: Path, attempts: int = 3) -> Path:
    if dst.exists() and dst.stat().st_size > 0:
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".part")
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
                shutil.copyfileobj(r, f, 1 << 20)
            tmp.rename(dst)
            return dst
        except OSError as exc:
            log(f"  {url}: попытка {attempt}: {exc}")
            time.sleep(2 * attempt)
    raise RuntimeError(f"не скачался {url}")


class RangeFile(io.RawIOBase):
    """Файл по HTTP Range: zipfile читает из большого архива только нужный член."""

    def __init__(self, url: str) -> None:
        self.url = url
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=60) as r:
            self.size = int(r.headers["Content-Length"])
        self.pos = 0

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        base = {0: 0, 1: self.pos, 2: self.size}[whence]
        self.pos = max(0, base + offset)
        return self.pos

    def read(self, n: int = -1) -> bytes:
        if self.pos >= self.size:
            return b""
        end = self.size - 1 if n is None or n < 0 else min(self.size, self.pos + n) - 1
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
        with urllib.request.urlopen(req, timeout=300) as r:
            data = r.read()
        self.pos += len(data)
        return data

    def readinto(self, b) -> int:
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)


def download_hf_zips(raw: Path, sid: str) -> None:
    src = SOURCES[sid]
    for path in src["zips"]:
        dst = raw / sid / Path(path).name
        http_get(HF.format(repo=src["repo"], path=path), dst)
        out = raw / sid / dst.stem
        if not out.exists():
            # Архив — недоверенные данные: распаковываем в свой пустой каталог, только картинки и json.
            tmp = out.with_name(out.name + ".tmp")
            shutil.rmtree(tmp, ignore_errors=True)
            tmp.mkdir(parents=True)
            with zipfile.ZipFile(dst) as z:
                for member in z.namelist():
                    name = Path(member).name
                    if member.endswith("/") or not name or name.startswith("."):
                        continue
                    if Path(name).suffix.lower() not in (".jpg", ".jpeg", ".png", ".json"):
                        continue
                    (tmp / name).write_bytes(z.read(member))
            tmp.rename(out)
        log(f"  {sid}/{dst.stem}: {sum(1 for _ in out.glob('*.jpg'))} кадров")


def download_hf_snapshot(raw: Path, sid: str, patterns: list[str] | None) -> None:
    from huggingface_hub import snapshot_download

    # max_workers=1: не больше двух загрузок на ноутбуке одновременно (правило проекта).
    snapshot_download(repo_id=SOURCES[sid]["repo"], repo_type="dataset", local_dir=str(raw / sid),
                      allow_patterns=patterns, max_workers=1)
    log(f"  {sid}: {sum(1 for _ in (raw / sid).rglob('*.jpg'))} кадров")


def coco_captions_forklift(raw: Path) -> set[int]:
    """id кадров COCO train2017, в подписях которых есть погрузчик. Читаем только один член архива 253 МБ."""
    cache = raw / "coco_vehicles" / "captions_forklift.json"
    if cache.exists():
        return set(json.loads(cache.read_text())["image_ids"])
    with zipfile.ZipFile(io.BufferedReader(RangeFile(COCO_CAPTIONS_ZIP), buffer_size=1 << 22)) as z:
        captions = json.loads(z.read(COCO_CAPTIONS_MEMBER))
    ids = sorted({a["image_id"] for a in captions["annotations"] if FORKLIFT_WORDS.search(a["caption"])})
    cache.write_text(json.dumps({"source": f"{COCO_CAPTIONS_ZIP}:{COCO_CAPTIONS_MEMBER}",
                                 "pattern": FORKLIFT_WORDS.pattern, "image_ids": ids}))
    return set(ids)


def coco_select(raw: Path, n_truck: int, n_bus: int, seed: int) -> list[int]:
    """Кадры COCO с заметным грузовиком или автобусом (площадь рамки ≥ 2 % кадра), без погрузчиков в подписях."""
    labels_zip = http_get(COCO_LABELS_ZIP, raw / "coco_vehicles" / "coco2017labels.zip")
    banned = coco_captions_forklift(raw)
    truck, bus = [], []
    with zipfile.ZipFile(labels_zip) as z:
        for member in z.namelist():
            if not member.startswith("coco/labels/train2017/") or not member.endswith(".txt"):
                continue
            image_id = int(Path(member).stem)
            if image_id in banned:
                continue
            classes = Counter()
            for line in z.read(member).decode().splitlines():
                parts = line.split()
                if len(parts) == 5 and float(parts[3]) * float(parts[4]) >= 0.02:
                    classes[int(parts[0])] += 1
            if classes[COCO_IDS["truck"]]:
                truck.append(image_id)
            elif classes[COCO_IDS["bus"]]:
                bus.append(image_id)
    rng = random.Random(seed)
    picked = sorted(rng.sample(sorted(truck), min(n_truck, len(truck))) +
                    rng.sample(sorted(bus), min(n_bus, len(bus))))
    log(f"  coco: кандидатов truck {len(truck)}, bus {len(bus)}; исключено по подписям {len(banned)}; "
        f"взято {len(picked)}")
    return picked


def download_coco(raw: Path, image_ids: list[int]) -> None:
    import requests

    out = raw / "coco_vehicles" / "images"
    out.mkdir(parents=True, exist_ok=True)
    todo = [i for i in image_ids if not (out / f"{i:012d}.jpg").exists()]
    # Одно соединение с keep-alive: ~0.5 с на кадр. Параллельно может идти загрузка HF — не больше двух.
    with requests.Session() as session:
        for k, image_id in enumerate(todo, 1):
            dst = out / f"{image_id:012d}.jpg"
            for attempt in range(1, 4):
                try:
                    r = session.get(COCO_IMAGE_URL.format(image_id), timeout=60)
                    r.raise_for_status()
                    tmp = dst.with_name(dst.name + ".part")
                    tmp.write_bytes(r.content)
                    tmp.rename(dst)
                    break
                except requests.RequestException as exc:
                    log(f"  coco {image_id}: попытка {attempt}: {exc}")
                    time.sleep(2 * attempt)
            if k % 100 == 0:
                log(f"  coco: {k}/{len(todo)}")
    log(f"  coco: {len(image_ids)} кадров ({len(todo)} скачано сейчас)")


def cmd_download(args: argparse.Namespace) -> None:
    raw = args.out / "raw"
    only = set(args.only.split(","))
    for sid in ("forklift_rf", "construction_rf"):
        if sid in only:
            log(f"download: {sid} (zip)")
            download_hf_zips(raw, sid)
    if "logistics_rf" in only:
        log("download: logistics_rf (HF, по файлу)")
        download_hf_snapshot(raw, "logistics_rf", None)
    if "excavators_rf100" in only:
        log("download: excavators_rf100 (HF, по файлу)")
        download_hf_snapshot(raw, "excavators_rf100", ["data.yaml", "README*", "*/images/*"])
    if "coco_vehicles" in only:
        log("download: coco_vehicles")
        ids = coco_select(raw, args.coco_truck, args.coco_bus, args.seed)
        (raw / "coco_vehicles" / "selected.json").write_text(json.dumps(ids))
        download_coco(raw, ids)
    if "imagenet_forklift" in only:
        log("download: imagenet_forklift (parquet)")
        download_imagenet(raw)


def download_imagenet(raw: Path) -> None:
    """Два parquet с картинками ImageNet класса forklift → JPEG в raw/imagenet_forklift/images."""
    import polars as pl

    src = SOURCES["imagenet_forklift"]
    out = raw / "imagenet_forklift" / "images"
    out.mkdir(parents=True, exist_ok=True)
    for path in src["files"]:
        dst = http_get(HF.format(repo=src["repo"], path=path), raw / "imagenet_forklift" / Path(path).name)
        part = "val" if "validation" in path else "train"
        df = pl.read_parquet(dst, columns=["image"])
        import cv2

        for k, cell in enumerate(df["image"].to_list()):
            img = cv2.imdecode(np.frombuffer(cell["bytes"], np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            # До 800 по большей стороне: обучение идёт в 480, а кадры по 9 Мп раздувают память учителя.
            scale = 800 / max(img.shape[:2])
            if scale < 1:
                img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(out / f"{part}_{k:05d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    log(f"  imagenet_forklift: {sum(1 for _ in out.glob('*.jpg'))} кадров")


# ---------- чтение источников ----------


def _yolo_box(x: float, y: float, w: float, h: float, width: int, height: int) -> tuple[float, float, float, float]:
    x1, y1 = max(0.0, x), max(0.0, y)
    x2, y2 = min(float(width), x + w), min(float(height), y + h)
    return ((x1 + x2) / 2 / width, (y1 + y2) / 2 / height, (x2 - x1) / width, (y2 - y1) / height)


def read_coco_json(sid: str, folder: Path, split: str) -> list[Sample]:
    data = json.loads((folder / "_annotations.coco.json").read_text())
    cats = {c["id"]: c["name"].strip().lower() for c in data["categories"]}
    by_image = defaultdict(list)
    for a in data["annotations"]:
        by_image[a["image_id"]].append(a)
    out = []
    for im in data["images"]:
        path = folder / im["file_name"]
        if not path.exists():
            continue
        s = Sample(sid, split, f"{sid}:{path.name}", path)
        for a in by_image[im["id"]]:
            name = cats[a["category_id"]]
            if name in CLASSES:
                box = _yolo_box(*a["bbox"], im["width"], im["height"])
                if box[2] > 0 and box[3] > 0:
                    s.boxes.append((CLASSES.index(name), *box))
            else:
                s.dropped[name] += 1
        out.append(s)
    return out


def read_yolo_dir(sid: str, root: Path, split_dirs: dict[str, str], names: list[str],
                  keep: dict[str, str]) -> list[Sample]:
    out = []
    for split, sub in split_dirs.items():
        for path in sorted((root / sub / "images").glob("*.jpg")):
            s = Sample(sid, split, f"{sid}:{path.name}", path)
            label = root / sub / "labels" / (path.stem + ".txt")
            if label.exists():
                for line in label.read_text().splitlines():
                    parts = line.split()
                    if len(parts) != 5:
                        continue
                    name = names[int(parts[0])].strip().lower()
                    if name in keep:
                        s.boxes.append((CLASSES.index(keep[name]), *map(float, parts[1:])))
                    else:
                        s.dropped[name] += 1
            out.append(s)
    return out


def _hash_split(key: str, test: int = 1, val: int = 1) -> str:
    bucket = int(hashlib.sha1(key.encode()).hexdigest(), 16) % 10
    return "test" if bucket < test else "val" if bucket < test + val else "train"


def load_sources(raw: Path, teacher: dict[str, list],
                 teacher_forklift: dict[str, list] | None = None) -> list[Sample]:
    samples: list[Sample] = []
    for split, sub in (("train", "train"), ("val", "valid"), ("test", "test")):
        samples += read_coco_json("forklift_rf", raw / "forklift_rf" / sub, split)
        samples += read_coco_json("construction_rf", raw / "construction_rf" / sub, split)

    root = raw / "logistics_rf" / "Logistics.v2i.yolov8"
    import yaml

    names = yaml.safe_load((root / "data.yaml").read_text())["names"]
    samples += read_yolo_dir("logistics_rf", root, {"train": "train", "val": "valid", "test": "test"}, names,
                             {"forklift": "forklift", "person": "person"})

    # RF100 excavators: кадры роликов, исходное разбиение случайное по кадрам. Делим заново по ролику.
    exc = raw / "excavators_rf100"
    for path in sorted(exc.glob("*/images/*.jpg")):
        video = re.split(r"_mp4-\d+", path.name)[0] if "_mp4-" in path.name else path.name.split("_jpg")[0]
        s = Sample("excavators_rf100", _hash_split(f"exc:{video}"), f"excavators_rf100:{video}", path)
        s.dropped["vehicle (EXCAVATORS, dump truck, wheel loader)"] += 1
        samples.append(s)

    coco_dir = raw / "coco_vehicles" / "images"
    labels_zip = raw / "coco_vehicles" / "coco2017labels.zip"
    if coco_dir.exists():
        with zipfile.ZipFile(labels_zip) as z:
            for path in sorted(coco_dir.glob("*.jpg")):
                image_id = int(path.stem)
                s = Sample("coco_vehicles", _hash_split(f"coco:{image_id}"), f"coco:{image_id}", path)
                for line in z.read(f"coco/labels/train2017/{path.stem}.txt").decode().splitlines():
                    parts = line.split()
                    if len(parts) != 5:
                        continue
                    if int(parts[0]) == COCO_IDS["person"]:
                        s.boxes.append((0, *map(float, parts[1:])))
                    else:
                        s.dropped[f"coco:{parts[0]}"] += 1
                samples.append(s)

    # ImageNet forklift: рамки погрузчика ставит учитель. Кадры без уверенной рамки пропускаем:
    # погрузчик на них есть, и без рамки он стал бы фоном. Только train — val и test размечены людьми.
    for path in sorted((raw / "imagenet_forklift" / "images").glob("*.jpg")):
        boxes = (teacher_forklift or {}).get(str(path))
        if not boxes:
            continue
        s = Sample("imagenet_forklift", "train", f"imagenet:{path.name}", path)
        s.boxes += [(1, *b) for b in boxes]
        samples.append(s)

    # Учитель добавляет людей там, где их разметка неполная.
    for s in samples:
        extra = teacher.get(str(s.image))
        if extra is None:
            continue
        gt = [b for b in s.boxes if b[0] == 0]
        for box in extra:
            if all(_iou_cxcywh(box, b[1:]) < 0.5 for b in gt):
                s.boxes.append((0, *box))
                s.teacher += 1
    return samples


def _iou_cxcywh(a, b) -> float:
    ax1, ay1, ax2, ay2 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1, bx2, by2 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    iw, ih = max(0.0, min(ax2, bx2) - max(ax1, bx1)), max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


# ---------- учитель ----------


TEACHER_SOURCES = ("construction_rf", "excavators_rf100", "imagenet_forklift")
FORKLIFT_TEACHER_CONF = 0.6
FORKLIFT_TEACHER_MAX = 3  # рамок погрузчика на кадр


def _predict(weights: Path, paths: list[str], conf: dict[int, float], device: str) -> dict[int, dict[str, list]]:
    """Рамки по классам: {cls: {путь: [[cx, cy, w, h], ...] по убыванию conf}}."""
    from ultralytics import YOLO

    model = YOLO(str(weights))
    result: dict[int, dict[str, list]] = {c: {} for c in conf}
    for r in model.predict(paths, imgsz=640, conf=min(conf.values()), classes=sorted(conf), device=device,
                           stream=True, verbose=False, batch=1):
        cls = r.boxes.cls.cpu().numpy().astype(int)
        scores = r.boxes.conf.cpu().numpy()
        xywhn = r.boxes.xywhn.cpu().numpy()
        for c, threshold in conf.items():
            keep = np.nonzero((cls == c) & (scores >= threshold))[0]
            keep = keep[np.argsort(-scores[keep])]
            result[c][str(r.path)] = xywhn[keep].round(5).tolist()
    return result


def _predict_onnx(onnx: Path, paths: list[str], conf: dict[int, float], classes: list[str]) -> dict[int, dict]:
    """То же через onnxruntime CPU. Чекпойнт keremberke под torch 2.14 на MPS падает уже на прогреве
    («MPSGraph does not support tensor dims larger than INT_MAX»), а на CPU раздувается до 8 ГБ."""
    import cv2
    import onnxruntime as ort

    sys.path.insert(0, str(ROOT))
    from guard.contracts import ModelSpec
    from ml.yolo_decode import decode, letterbox

    sess = ort.InferenceSession(str(onnx), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    size = tuple(inp.shape[2:4])
    spec = ModelSpec(model_id="teacher", onnx_path=str(onnx), input_size=size, classes=classes,
                     conf_threshold=min(conf.values()), iou_threshold=0.5)
    result: dict[int, dict[str, list]] = {c: {} for c in conf}
    for path in paths:
        img = cv2.imread(path)
        if img is None:
            continue
        h, w = img.shape[:2]
        x, meta = letterbox(img, size)
        dets = sorted(decode(sess.run(None, {inp.name: x}), meta, spec), key=lambda d: -d.conf)
        for c, threshold in conf.items():
            result[c][path] = [[round((d.bbox[0] + d.bbox[2]) / 2 / w, 5), round((d.bbox[1] + d.bbox[3]) / 2 / h, 5),
                                round((d.bbox[2] - d.bbox[0]) / w, 5), round((d.bbox[3] - d.bbox[1]) / h, 5)]
                               for d in dets if d.cls_id == c and d.conf >= threshold]
    return result


def cmd_teacher(args: argparse.Namespace) -> None:
    """Учитель: люди от COCO YOLOv8s там, где их разметка неполная; погрузчики (и люди) на ImageNet forklift —
    от keremberke/yolov8s-forklift-detection (ONNX models/forklift-640.onnx). На этих кадрах погрузчик есть
    заведомо, поэтому его слабость (фуры и экскаваторы как forklift) здесь не мешает.
    torch — запускать через tools/heavy.sh. Готовые ответы учителя из teacher_person.json не пересчитываются."""
    raw = args.out / "raw"
    t0 = time.time()
    out_people = raw / "teacher_person.json"
    people = json.loads(out_people.read_text())["images"] if out_people.exists() else {}
    paths = []
    for sid in ("construction_rf", "excavators_rf100"):
        paths += [str(p) for p in sorted((raw / sid).rglob("*.jpg")) if str(p) not in people]
    weights = Path(args.teacher_weights)
    if paths:
        people.update(_predict(weights, paths, {0: TEACHER_CONF}, args.device)[0])

    imagenet = sorted(str(p) for p in (raw / "imagenet_forklift" / "images").glob("*.jpg"))
    if imagenet:
        fw = Path(args.forklift_teacher_onnx)
        # у keremberke 0 = forklift, 1 = person
        pred = _predict_onnx(fw, imagenet, {0: FORKLIFT_TEACHER_CONF, 1: TEACHER_CONF}, ["forklift", "person"])
        boxes = {k: v[:FORKLIFT_TEACHER_MAX] for k, v in pred[0].items() if v}
        people.update(pred[1])
        out = raw / "teacher_forklift.json"
        out.write_text(json.dumps({"weights": fw.name, "conf": FORKLIFT_TEACHER_CONF,
                                   "max_per_image": FORKLIFT_TEACHER_MAX, "images": boxes}))
        log(f"teacher: imagenet_forklift {len(imagenet)} кадров, с погрузчиком {len(boxes)}, "
            f"рамок {sum(len(v) for v in boxes.values())} → {out}")
    out_people.write_text(json.dumps({"weights": weights.name, "conf": TEACHER_CONF, "images": people}))
    log(f"teacher: людей {sum(len(v) for v in people.values())} на {len(people)} кадрах → {out_people}; "
        f"{time.time() - t0:.0f} с")


# ---------- дедуп и сборка ----------


def dhash(path: Path) -> int:
    import cv2

    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"не читается {path}")
    small = cv2.resize(img, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def _popcount64(x: np.ndarray) -> np.ndarray:
    return np.unpackbits(x.view(np.uint8).reshape(-1, 8), axis=1).sum(axis=1)


def holdout_hashes(scenarios: Path, videos_dir: Path, fps: float = 2.0) -> np.ndarray:
    """dHash кадров роликов оценки: ни один похожий кадр не должен попасть в набор."""
    import cv2
    import yaml

    raw = yaml.safe_load(scenarios.read_text())
    hashes = []
    for entry in raw["scenarios"]:
        cap = cv2.VideoCapture(str(videos_dir / entry["video"]))
        native = cap.get(cv2.CAP_PROP_FPS) or 25.0
        step = max(1, round(native / fps))
        idx = 0
        while True:
            ok = cap.grab()
            if not ok:
                break
            if idx % step == 0:
                ok, frame = cap.retrieve()
                if ok:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
                    bits = (small[:, 1:] > small[:, :-1]).flatten()
                    hashes.append(int("".join("1" if b else "0" for b in bits), 2))
            idx += 1
        cap.release()
    return np.array(hashes, dtype=np.uint64)


def dedup(samples: list[Sample], holdout: np.ndarray) -> tuple[list[Sample], dict]:
    for s in samples:
        s.dhash = dhash(s.image)
    order = sorted(range(len(samples)), key=lambda i: (SPLIT_RANK[samples[i].split], samples[i].source,
                                                      samples[i].image.name))
    hashes = np.array([samples[i].dhash for i in order], dtype=np.uint64)
    keep = np.ones(len(order), dtype=bool)
    dropped_dup: Counter = Counter()
    dropped_holdout: Counter = Counter()
    for k in range(len(order)):
        s = samples[order[k]]
        if len(holdout) and _popcount64(holdout ^ hashes[k]).min() <= DUP_HAMMING:
            keep[k] = False
            dropped_holdout[s.source] += 1
            continue
        if k == 0:
            continue
        prev = np.nonzero(keep[:k])[0]
        if len(prev) and _popcount64(hashes[prev] ^ hashes[k]).min() <= DUP_HAMMING:
            keep[k] = False
            dropped_dup[f"{s.source}/{s.split}"] += 1
    kept = [samples[order[k]] for k in range(len(order)) if keep[k]]
    return kept, {"hamming_max": DUP_HAMMING, "dropped_duplicates": dict(dropped_dup),
                  "dropped_holdout_match": dict(dropped_holdout), "holdout_frames": int(len(holdout))}


def cap_groups(samples: list[Sample], per_group: int, seed: int) -> list[Sample]:
    """Не больше per_group кадров на ролик excavators_rf100: соседние кадры почти одинаковы."""
    rng = random.Random(seed)
    groups = defaultdict(list)
    for s in samples:
        groups[s.group if s.source == "excavators_rf100" else id(s)].append(s)
    out = []
    for key, items in groups.items():
        if isinstance(key, str) and len(items) > per_group:
            items = rng.sample(sorted(items, key=lambda s: s.image.name), per_group)
        out.extend(items)
    return out


def file_list_sha(names: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(names)).encode()).hexdigest()


def cmd_build(args: argparse.Namespace) -> None:
    out: Path = args.out
    raw = out / "raw"
    teacher_file = raw / "teacher_person.json"
    teacher = json.loads(teacher_file.read_text())["images"] if teacher_file.exists() else {}
    fork_file = raw / "teacher_forklift.json"
    teacher_forklift = json.loads(fork_file.read_text())["images"] if fork_file.exists() else {}
    samples = load_sources(raw, teacher, teacher_forklift)
    if not teacher:
        # Без учителя люди на негативах стали бы фоном: такие источники пропускаем.
        samples = [s for s in samples if s.source not in TEACHER_SOURCES]
        log("build: нет teacher_person.json — construction_rf и excavators_rf100 пропущены")
    before = Counter(f"{s.source}/{s.split}" for s in samples)
    samples = cap_groups(samples, args.per_video, args.seed)
    holdout = holdout_hashes(ROOT / "ml/eval/scenarios.yaml", args.videos)
    samples, dedup_stats = dedup(samples, holdout)

    for split in SPLITS:
        for kind in ("images", "labels"):
            d = out / kind / split
            shutil.rmtree(d, ignore_errors=True)
            d.mkdir(parents=True)
    stats = defaultdict(lambda: defaultdict(Counter))
    lists = defaultdict(list)
    per_source_test = defaultdict(list)
    for s in samples:
        name = f"{s.source}__{s.image.stem}"[:150] + s.image.suffix.lower()
        dst = out / "images" / s.split / name
        if dst.exists():
            raise ValueError(f"повтор имени {name}")
        try:
            dst.hardlink_to(s.image)
        except OSError:
            shutil.copy2(s.image, dst)
        lines = [f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for c, cx, cy, w, h in s.boxes]
        (out / "labels" / s.split / (dst.stem + ".txt")).write_text("\n".join(lines) + ("\n" if lines else ""))
        st = stats[s.split][s.source]
        st["images"] += 1
        st["negative_images"] += int(not any(b[0] == 1 for b in s.boxes))
        for c, *_ in s.boxes:
            st[CLASSES[c]] += 1
        st["person_teacher"] += s.teacher
        for k, v in s.dropped.items():
            stats["background_objects"][s.source][k] += v
        lists[s.split].append(name)
        if s.split == "test":
            per_source_test[s.source].append(str(dst))

    # Списки кадров теста по источникам: gate меряет и весь тест, и forklift_rf отдельно (как у keremberke).
    for sid, paths in per_source_test.items():
        (out / f"test_{sid}.txt").write_text("\n".join(sorted(paths)) + "\n")
    data_yaml = out / "data.yaml"
    data_yaml.write_text(
        f"# {VERSION}: собран ml/datasets/forklift.py, описание — docs/DATASETS.md\n"
        f"path: {out}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        f"names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASSES)))
    manifest = {
        "version": VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "classes": list(CLASSES),
        "data_yaml": str(data_yaml),
        "sources": {sid: {k: v for k, v in src.items() if k not in ("zips",)} for sid, src in SOURCES.items()},
        "licenses": "не проверяли: решение человека 24.10, демо внутреннее",
        "teacher": {"person": {"weights": "yolov8s.pt (COCO)", "conf": TEACHER_CONF,
                               "sources": list(TEACHER_SOURCES), "images": len(teacher)},
                    "forklift": {"weights": "forklift_keremberke_v8s.pt "
                                            "(https://huggingface.co/keremberke/yolov8s-forklift-detection)",
                                 "conf": FORKLIFT_TEACHER_CONF, "max_per_image": FORKLIFT_TEACHER_MAX,
                                 "sources": ["imagenet_forklift"], "images": len(teacher_forklift)}},
        "split_rule": {"forklift_rf, logistics_rf, construction_rf": "исходные train/valid/test",
                       "excavators_rf100": "по ролику, sha1(имя ролика) % 10: 0 test, 1 val",
                       "coco_vehicles": "по id кадра, sha1 % 10: 0 test, 1 val",
                       "imagenet_forklift": "только train: рамки от учителя"},
        "per_video_cap": args.per_video,
        "before_dedup": dict(sorted(before.items())),
        "dedup": dedup_stats,
        "splits": {split: {sid: dict(c) for sid, c in sorted(stats[split].items())} for split in SPLITS},
        "background_objects": {sid: dict(c) for sid, c in sorted(stats["background_objects"].items())},
        "totals": {split: {"images": len(lists[split]), "sha256_files": file_list_sha(lists[split])}
                   for split in SPLITS},
        "test_subsets": {sid: str(out / f"test_{sid}.txt") for sid in sorted(per_source_test)},
        "holdout_videos": "ml/eval/scenarios.yaml: кадры 13 роликов сверены по dHash и в набор не входят",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    for split in SPLITS:
        tot = Counter()
        for c in stats[split].values():
            tot.update(c)
        log(f"build {split}: кадров {tot['images']}, forklift {tot['forklift']}, person {tot['person']} "
            f"(учитель {tot['person_teacher']}), без погрузчика {tot['negative_images']}")
    log(f"build: дубли {dedup_stats['dropped_duplicates']}, совпадения с роликами оценки "
        f"{dedup_stats['dropped_holdout_match']} → {out / 'manifest.json'}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["download", "teacher", "build", "all"])
    ap.add_argument("--out", type=Path, default=ROOT / "data/datasets/forklift")
    ap.add_argument("--videos", type=Path, default=ROOT / "data/videos", help="ролики оценки для сверки")
    ap.add_argument("--only", default=",".join(SOURCES), help="источники для download через запятую")
    ap.add_argument("--coco-truck", type=int, default=450)
    ap.add_argument("--coco-bus", type=int, default=150)
    ap.add_argument("--per-video", type=int, default=25, help="кадров на ролик excavators_rf100")
    ap.add_argument("--seed", type=int, default=12)
    ap.add_argument("--teacher-weights", default=str(ROOT / "models/weights/yolov8s.pt"))
    ap.add_argument("--forklift-teacher-onnx", default=str(ROOT / "models/forklift-640.onnx"),
                    help="keremberke/yolov8s-forklift-detection, экспорт A2")
    ap.add_argument("--device", default="mps")
    args = ap.parse_args(argv)
    args.out = args.out.resolve()
    if args.cmd in ("download", "all"):
        cmd_download(args)
    if args.cmd in ("teacher", "all"):
        cmd_teacher(args)
    if args.cmd in ("build", "all"):
        cmd_build(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
