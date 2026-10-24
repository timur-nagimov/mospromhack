"""Наборы падений под слабое место S3: детектор теряет лежащего человека (docs/QUALITY.md).

Источники (лицензии не проверяем — решение 24.10, демо внутреннее; URL записываем в manifest.json):
- UR Fall Detection (URFD), http://fenix.ur.edu.pl/~mkepski/ds/uf.html — 30 падений и 40 ADL, RGB cam0
  (PNG 640×480, 30 к/с) и покадровые метки позы -1 / 0 / 1 (не лежит / падает / лежит).
- CAUCAFall, https://data.mendeley.com/datasets/7w7fccy7ky (v5) — 10 человек × (5 падений + 5 ADL),
  AVI 720×480, YOLO-разметка кадров (nofall / fall). Метки берём выборочно из общего zip по HTTP Range.
- GMDCSA24, https://github.com/ekramalam/GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos —
  только Subject 1 и Subject 4 и только в отложенную выборку. Subject 2 и 3 вошли в s3_02, их не трогаем.
- COCO val2017 (person keypoints) — повтор против забывания при дообучении pose.
- Le2i — зеркала без авторизации нет (docs/DATASETS.md).

Разбиение — по людям (URFD — по одежде, CAUCAFall — по номеру человека) и сериям, не по кадрам.
Кадры роликов из ml/eval/scenarios.yaml сюда не попадают: GMDCSA24 Subject 2 и 3 не скачиваются.

Шаги (каталог данных — $GUARD_DATA_DIR или <корень>/data):
  download  исходники в data/datasets/falls/raw/
  holdout   отложенные ролики data/videos/falls_*/, метки data/labels/falls/,
            длительности data/durations_falls.json, сценарии ml/eval/scenarios_falls.yaml
  recall    покадровый recall лежачих у моделей реестра на test-сериях
  prepare   кадры train и val в data/datasets/falls/frames/ и список frames.jsonl
  teacher   ключевые точки учителя yolov8x-pose (torch: только через tools/heavy.sh)
  yolo      набор YOLO-pose data/datasets/falls/yolo/ с data.yaml и manifest.json
  yolo-test тест детекции для гейта: test-серии CAUCAFall и COCO вне обучения (download --coco-test)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import zipfile
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:  # запуск файлом: python ml/datasets/falls.py
    sys.path.insert(0, str(ROOT))

URFD_PAGE = "http://fenix.ur.edu.pl/~mkepski/ds/uf.html"
URFD_BASE = "https://fenix.ur.edu.pl/~mkepski/ds/data/"
CAUCA_PAGE = "https://data.mendeley.com/datasets/7w7fccy7ky"
CAUCA_API = "https://data.mendeley.com/public-api/datasets/7w7fccy7ky"
CAUCA_ZIP = "https://data.mendeley.com/public-api/zip/7w7fccy7ky/download/5"
GMDCSA_PAGE = "https://github.com/ekramalam/GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos"
GMDCSA_RAW = (
    "https://raw.githubusercontent.com/ekramalam/"
    "GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos/master/"
)
GMDCSA_TREE = (
    "https://api.github.com/repos/ekramalam/"
    "GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos/git/trees/master?recursive=1"
)
COCO_IMAGES_ZIP = "http://images.cocodataset.org/zips/val2017.zip"
COCO_ANN_ZIP = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
TEACHER_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8x-pose.pt"

# GMDCSA24: в отложенную выборку только эти люди. Subject 2 и 3 — источник s3_02 (data/labels/s3_02.json).
GMDCSA_SUBJECTS = (1, 4)
GMDCSA_EXCLUDED = (2, 3)

# URFD: люди различены по одежде на контактных листах (24.10). Падения — по 6 серий на человека.
URFD_PERSON: dict[str, str] = {
    **{f"fall-{i:02d}": f"P{(i - 1) // 6 + 1}" for i in range(1, 31)},
    **{f"adl-{i:02d}": "A" for i in range(1, 12)},  # комната с диваном
    **{f"adl-{i:02d}": "B" for i in range(12, 24)},  # тёмная комната с диваном
    "adl-24": "C1", "adl-25": "C1", "adl-26": "C2", "adl-27": "C2",
    "adl-28": "P5", "adl-29": "P5",  # клетчатая рубашка и очки, как в fall-25…30
    "adl-30": "D", "adl-38": "D", "adl-39": "D", "adl-40": "D",
    "adl-31": "P5", "adl-32": "P5", "adl-33": "P4", "adl-34": "P2", "adl-35": "P2",
    "adl-36": "E", "adl-37": "E",  # спальня, лёг на кровать
}
URFD_TEST_PERSONS = frozenset({"P2", "P5", "B", "E"})
URFD_VAL_PERSONS = frozenset({"P4"})
# Что делает человек в ADL (для разбора ложных тревог по видам).
URFD_ADL_ACTIVITY: dict[str, str] = {
    **{f"adl-{i:02d}": "walk_sit_crouch" for i in range(1, 41)},
    "adl-10": "lying_bed", "adl-11": "lying_bed",
    "adl-21": "lying_sofa", "adl-22": "lying_sofa", "adl-23": "lying_sofa",
    "adl-36": "lying_bed", "adl-37": "lying_bed",
    **{f"adl-{i}": "floor_activity" for i in (30, 31, 32, 33, 34, 35, 38, 39, 40)},
}

CAUCA_TEST_SUBJECTS = frozenset({8, 9, 10})
CAUCA_VAL_SUBJECTS = frozenset({7})
CAUCA_ACTIVITIES = (
    "FallBackwards", "FallForward", "FallLeft", "FallRight", "FallSitting",
    "Hop", "Kneel", "Pickupobject", "SitDown", "Walk",
)

# Падение в отложенной выборке достоверно, если в ролике видно не меньше 3 с от начала падения:
# около 1 с само падение и не меньше 2 с лежания. Короче — нельзя судить о неподвижности, метка uncertain.
MIN_OBSERVED_AFTER_ONSET_S = 3.0

COCO_FLIP_IDX = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
YOLO_VERSION = "falls-pose-ds-v1"
UA = {"User-Agent": "curl/8.7.1", "Accept": "*/*"}


def data_dir() -> Path:
    return Path(os.environ.get("GUARD_DATA_DIR") or ROOT / "data").expanduser().resolve()


def falls_dir() -> Path:
    return data_dir() / "datasets" / "falls"


def raw_dir() -> Path:
    return falls_dir() / "raw"


# ---------------------------------------------------------------- загрузка


def _get(url: str, timeout: float = 120.0) -> bytes:
    request = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def fetch(url: str, dest: Path, sha256: str = "", attempts: int = 5) -> bool:
    """Скачать файл, если его нет или не совпала сумма. True — файл скачан сейчас."""
    if dest.is_file() and (not sha256 or _sha256(dest) == sha256):
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            data = _get(url, timeout=1800)
            if sha256 and hashlib.sha256(data).hexdigest() != sha256:
                raise ValueError(f"sha256 mismatch for {url}")
            tmp = dest.with_name(dest.name + ".part")
            tmp.write_bytes(data)
            tmp.replace(dest)
            return True
        except Exception as exc:  # noqa: BLE001 — повтор при обрыве
            last = exc
            time.sleep(2 + 3 * attempt)
    raise OSError(f"cannot download {url}: {last}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class HttpRangeFile(io.RawIOBase):
    """Файл только для чтения поверх HTTP Range: zipfile читает из большого архива отдельные файлы.

    Mendeley отдаёт подписанную ссылку S3 на 300 с, поэтому ссылка обновляется каждые 240 с и при ошибке.
    """

    def __init__(self, url: str, block: int = 256 * 1024) -> None:
        self.url = url
        self.block = block
        self.pos = 0
        self.requests = 0
        self._signed: str | None = None
        self._signed_at = 0.0
        self._buf_start = -1
        self._buf = b""
        size = self._resolve(force=True)
        if size is None:
            raise OSError(f"{url}: server does not report size")
        self.size = size

    def _resolve(self, force: bool = False) -> int | None:
        if not force and self._signed is not None and time.time() - self._signed_at < 240:
            return None
        request = urllib.request.Request(self.url, headers={**UA, "Range": "bytes=0-0"})
        with urllib.request.urlopen(request, timeout=60) as response:
            self._signed = response.geturl()
            total = response.headers.get("Content-Range", "").rsplit("/", 1)[-1]
        self._signed_at = time.time()
        return int(total) if total.isdigit() else None

    def _range(self, start: int, end: int) -> bytes:
        last: Exception | None = None
        for attempt in range(6):
            try:
                self._resolve()
                request = urllib.request.Request(
                    self._signed or self.url, headers={**UA, "Range": f"bytes={start}-{end}"}
                )
                with urllib.request.urlopen(request, timeout=180) as response:
                    self.requests += 1
                    data = response.read()
                if len(data) != end - start + 1:
                    raise OSError(f"short read {len(data)} of {end - start + 1}")
                return data
            except Exception as exc:  # noqa: BLE001
                last = exc
                self._signed = None
                time.sleep(1 + 2 * attempt)
        raise OSError(f"range {start}-{end} failed: {last}")

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        base = {0: 0, 1: self.pos, 2: self.size}[whence]
        self.pos = base + offset
        return self.pos

    def read(self, n: int | None = -1) -> bytes:
        if n is None or n < 0:
            n = self.size - self.pos
        if n <= 0 or self.pos >= self.size:
            return b""
        end = min(self.pos + n, self.size) - 1
        cached = self._buf_start <= self.pos and end < self._buf_start + len(self._buf)
        if not cached:
            fetch_end = min(max(end, self.pos + self.block - 1), self.size - 1)
            self._buf = self._range(self.pos, fetch_end)
            self._buf_start = self.pos
        data = self._buf[self.pos - self._buf_start : end - self._buf_start + 1]
        self.pos += len(data)
        return data

    def readinto(self, buffer: Any) -> int:
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)


def download_urfd() -> None:
    root = raw_dir() / "urfd"
    for name in ("urfall-cam0-falls.csv", "urfall-cam0-adls.csv"):
        fetch(URFD_BASE + name, root / name)
    for seq in URFD_PERSON:
        fetch(URFD_BASE + f"{seq}-data.csv", root / "sync" / f"{seq}-data.csv")
        dest = root / "zips" / f"{seq}-cam0-rgb.zip"
        if dest.is_file() and zipfile.is_zipfile(dest):
            continue
        fetch(URFD_BASE + f"{seq}-cam0-rgb.zip", dest)
        print("urfd", seq, flush=True)


def _cauca_activity(folder: str) -> str:
    """'Fall backwards' → 'FallBackwards', 'Pick up object' → 'Pickupobject'."""
    key = folder.replace(" ", "").lower()
    for activity in CAUCA_ACTIVITIES:
        if activity.lower() == key:
            return activity
    raise ValueError(f"unknown CAUCAFall activity folder: {folder!r}")


CAUCA_LABEL = re.compile(r"Subject\.(\d+)/([^/]+)/[a-z]+\d+?-?(\d{5})\.txt$")


def download_cauca(subjects: tuple[int, ...] = tuple(range(1, 11)), label_step: int = 4) -> None:
    """AVI — через API Mendeley (100 файлов), txt-метки — из общего zip (8.3 ГБ) по HTTP Range.

    Метки лежат в архиве вперемешку с PNG, поэтому одна метка — один запрос (~0.5 с). Берём нечётные
    кадры (label_step 2): для рамок и начала падения этого хватает (ошибка ≤ 1 кадра, 50 мс).
    Метки человека пишутся в labels/S<NN>.json — два процесса могут качать разных людей.
    """
    root = raw_dir() / "caucafall"
    meta_path = root / "meta.json"
    if not meta_path.is_file():
        root.mkdir(parents=True, exist_ok=True)
        meta_path.write_bytes(_get(CAUCA_API))
    meta = json.loads(meta_path.read_text())
    for item in meta["files"]:
        name = item["filename"]
        match = re.match(r"(.+?)S(\d+)\.avi$", name)
        if not match or int(match.group(2)) not in subjects:
            continue
        rel = f"S{int(match.group(2)):02d}/{match.group(1)}/{name}"
        details = item["content_details"]
        if fetch(details["download_url"], root / "files" / rel, details.get("sha256_hash", "")):
            print("cauca", rel, flush=True)

    archive = zipfile.ZipFile(HttpRangeFile(CAUCA_ZIP))
    for subject in subjects:
        labels_path = root / "labels" / f"S{subject:02d}.json"
        labels: dict[str, dict[str, str]] = (
            json.loads(labels_path.read_text()) if labels_path.is_file() else {}
        )
        members = []
        for info in archive.infolist():
            match = CAUCA_LABEL.search(info.filename)
            if not match or int(match.group(1)) != subject:
                continue
            series = f"S{subject:02d}_{_cauca_activity(match.group(2))}"
            frame = int(match.group(3))
            if (frame - 1) % label_step or str(frame) in labels.get(series, {}):
                continue
            members.append((info.header_offset, info, series, frame))
        members.sort(key=lambda item: item[0])  # по смещению: соседние метки приходят одним запросом
        labels_path.parent.mkdir(parents=True, exist_ok=True)
        for count, (_, info, series, frame) in enumerate(members, 1):
            labels.setdefault(series, {})[str(frame)] = archive.read(info).decode("utf-8", "replace")
            if count % 500 == 0:
                labels_path.write_text(json.dumps(labels))
                print(f"cauca S{subject:02d} labels {count}/{len(members)}", flush=True)
        labels_path.write_text(json.dumps(labels))


def download_gmdcsa() -> None:
    root = raw_dir() / "gmdcsa24"
    tree_path = root / "tree.json"
    if not tree_path.is_file():
        root.mkdir(parents=True, exist_ok=True)
        tree_path.write_bytes(_get(GMDCSA_TREE))
    tree = json.loads(tree_path.read_text())
    for item in tree["tree"]:
        path = item["path"]
        if item["type"] != "blob" or not path.startswith("Subject "):
            continue
        subject = int(path.split("/")[0].split()[1])
        if subject not in GMDCSA_SUBJECTS:
            continue
        if path.endswith((".mp4", ".csv")):
            if fetch(GMDCSA_RAW + urllib.parse.quote(path), root / path):
                print("gmdcsa", path, flush=True)


def coco_split_ids(with_people: list[int], limit: int, test: int) -> tuple[list[int], list[int]]:
    """Повтор (первые limit изображений с людьми) и тест забывания (следующие test): части не пересекаются."""
    return with_people[:limit], with_people[limit : limit + test]


def download_coco(limit: int = 1500, test: int = 0) -> None:
    """Повтор против забывания: изображения COCO val2017 с людьми и их ключевыми точками.

    test > 0 — ещё столько изображений для проверки забывания (coco/test.json): в обучение не идут.
    """
    root = raw_dir() / "coco"
    ann_path = root / "person_keypoints_val2017.json"
    if not ann_path.is_file():
        root.mkdir(parents=True, exist_ok=True)
        archive = zipfile.ZipFile(HttpRangeFile(COCO_ANN_ZIP))
        ann_path.write_bytes(archive.read("annotations/person_keypoints_val2017.json"))
    ann = json.loads(ann_path.read_text())
    with_people = sorted(
        {a["image_id"] for a in ann["annotations"] if a.get("num_keypoints", 0) > 0 and not a["iscrowd"]}
    )
    chosen, test_ids = coco_split_ids(with_people, limit, test)
    images = {img["id"]: img for img in ann["images"]}
    archive = zipfile.ZipFile(HttpRangeFile(COCO_IMAGES_ZIP))
    infos = {info.filename: info for info in archive.infolist()}
    todo = []
    for image_id in chosen + test_ids:
        name = images[image_id]["file_name"]
        dest = root / "images" / name
        if not dest.is_file():
            todo.append((infos[f"val2017/{name}"].header_offset, name, dest))
    todo.sort()
    for count, (_, name, dest) in enumerate(todo, 1):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(archive.read(f"val2017/{name}"))
        if count % 200 == 0:
            print(f"coco {count}/{len(todo)}", flush=True)
    (root / "chosen.json").write_text(json.dumps(chosen))
    if test_ids:
        (root / "test.json").write_text(json.dumps(test_ids))


def download_teacher() -> Path:
    dest = Path(os.environ.get("GUARD_MODELS_DIR") or ROOT / "models") / "weights" / "yolov8x-pose.pt"
    fetch(TEACHER_URL, dest)
    return dest


# ---------------------------------------------------------------- каталог серий


@dataclass
class Series:
    """Одна запись набора: ролик или последовательность кадров одного человека."""

    source: str  # urfd | cauca | gmdcsa
    name: str  # уникально внутри источника
    person: str
    kind: str  # fall | adl
    activity: str
    split: str  # train | val | test
    fps: float = 0.0
    n_frames: int = 0
    # Начало падения, с от начала ролика (для fall), и как получено.
    fall_onset_s: float | None = None
    # Метки кадров: позы URFD (-1/0/1) или рамки CAUCAFall [cls, cx, cy, w, h] по номеру кадра с 1.
    lying: dict[int, int] = field(default_factory=dict)
    boxes: dict[int, list[list[float]]] = field(default_factory=dict)
    path: str = ""

    @property
    def key(self) -> str:
        return f"{self.source}/{self.name}"

    @property
    def duration_s(self) -> float:
        return self.n_frames / self.fps if self.fps else 0.0


def _urfd_split(seq: str) -> str:
    person = URFD_PERSON[seq]
    if person in URFD_TEST_PERSONS:
        return "test"
    return "val" if person in URFD_VAL_PERSONS else "train"


def urfd_series() -> list[Series]:
    root = raw_dir() / "urfd"
    poses: dict[str, dict[int, int]] = defaultdict(dict)
    for name in ("urfall-cam0-falls.csv", "urfall-cam0-adls.csv"):
        with (root / name).open() as stream:
            for row in csv.reader(stream):
                poses[row[0]][int(row[1])] = int(row[2])
    out = []
    for seq, person in URFD_PERSON.items():
        zip_path = root / "zips" / f"{seq}-cam0-rgb.zip"
        if not zip_path.is_file():
            continue
        with zipfile.ZipFile(zip_path) as archive:
            n_frames = sum(name.endswith(".png") for name in archive.namelist())
        times = [
            float(row[1]) for row in csv.reader((root / "sync" / f"{seq}-data.csv").open()) if row
        ]
        steps = [b - a for a, b in zip(times, times[1:]) if b > a]
        fps = round(1000.0 / (sum(steps) / len(steps)), 3) if steps else 30.0
        kind = "fall" if seq.startswith("fall") else "adl"
        lying = poses.get(seq, {})
        onset = None
        if kind == "fall":
            falling = [frame for frame, value in lying.items() if value >= 0]
            onset = (min(falling) - 1) / fps if falling else None
        out.append(
            Series(
                source="urfd", name=seq, person=f"urfd:{person}", kind=kind,
                activity="fall" if kind == "fall" else URFD_ADL_ACTIVITY[seq],
                split=_urfd_split(seq), fps=fps, n_frames=n_frames, fall_onset_s=onset,
                lying=lying, path=str(zip_path),
            )
        )
    return out


def _parse_yolo(text: str) -> list[list[float]]:
    boxes = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 5:
            boxes.append([float(value) for value in parts])
    return boxes


def cauca_series() -> list[Series]:
    root = raw_dir() / "caucafall"
    labels: dict[str, dict[str, str]] = {}
    for labels_path in sorted((root / "labels").glob("S*.json")):
        labels.update(json.loads(labels_path.read_text()))
    import cv2

    out = []
    for avi in sorted((root / "files").glob("S*/*/*.avi")):
        subject = int(avi.parent.parent.name[1:])
        activity = avi.parent.name
        name = f"S{subject:02d}_{activity}"
        capture = cv2.VideoCapture(str(avi))
        fps = float(capture.get(cv2.CAP_PROP_FPS)) or 20.0
        n_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.release()
        boxes = {int(frame): _parse_yolo(text) for frame, text in labels.get(name, {}).items()}
        kind = "fall" if activity.startswith("Fall") else "adl"
        onset = None
        if kind == "fall":
            fallen = sorted(frame for frame, items in boxes.items() if any(int(b[0]) == 1 for b in items))
            onset = (fallen[0] - 1) / fps if fallen else None
        split = (
            "test" if subject in CAUCA_TEST_SUBJECTS
            else "val" if subject in CAUCA_VAL_SUBJECTS else "train"
        )
        out.append(
            Series(
                source="cauca", name=name, person=f"cauca:S{subject:02d}", kind=kind,
                activity=activity, split=split, fps=fps, n_frames=n_frames,
                fall_onset_s=onset, boxes=boxes, path=str(avi),
            )
        )
    return out


_INTERVAL = re.compile(
    r"([A-Za-z]+)\s*(?:\([A-Z]+\))?\s*(?:\[\s*([\d.]+)\s*(?:to)?\s*([\d.]+)\s*(?:;[^\]]*)?\])?"
)


def parse_gmdcsa_classes(text: str) -> dict[str, tuple[float, float] | None]:
    """'Falling (SW)[3.4 to 6]; Sitting[0 to 3.4]' → {'Falling': (3.4, 6.0), 'Sitting': (0.0, 3.4)}.

    В разметке бывают '[2.3 6]' без 'to' и 'Falling (FW);' без интервала — тогда значение None.
    """
    out: dict[str, tuple[float, float] | None] = {}
    for match in _INTERVAL.finditer(text):
        name = match.group(1)
        if not name[0].isupper():
            continue
        out[name] = (float(match.group(2)), float(match.group(3))) if match.group(2) else None
    return out


def gmdcsa_fall_onset(intervals: dict[str, tuple[float, float] | None]) -> float | None:
    """Начало падения; без интервала Falling — конец последнего предшествующего действия."""
    falling = intervals.get("Falling")
    if falling is not None:
        return falling[0]
    if "Falling" in intervals:
        ends = [value[1] for name, value in intervals.items() if name != "Falling" and value is not None]
        return max(ends) if ends else None
    return None


def gmdcsa_adl_activity(intervals: dict[str, Any], description: str) -> str:
    text = description.lower()
    if "Sleeping" in intervals:
        return "lying_bed"
    if "push-up" in text or "sitting on the floor" in text:
        return "floor_activity"
    return "walk_sit_read"


def gmdcsa_fall_activity(description: str) -> str:
    """Падение на кровать правило не отличает от «лёг на кровать» (ADL), поэтому выделяем его."""
    text = description.lower()
    floor = any(word in text for word in ("to the floor", "on the floor", "to the ground", "on the ground"))
    return "fall_on_bed" if "on the bed" in text and not floor else "fall"


def gmdcsa_series() -> list[Series]:
    root = raw_dir() / "gmdcsa24"
    import cv2

    out = []
    for subject in GMDCSA_SUBJECTS:
        for kind_dir, kind in (("Fall", "fall"), ("ADL", "adl")):
            table = root / f"Subject {subject}" / f"{kind_dir}.csv"
            if not table.is_file():
                continue
            rows = list(csv.reader(table.open(encoding="utf-8", errors="replace")))
            for row in rows[1:]:
                if not row or not row[0].strip().endswith(".mp4"):
                    continue
                video = root / f"Subject {subject}" / kind_dir / row[0].strip()
                if not video.is_file():
                    continue
                intervals = parse_gmdcsa_classes(",".join(row[4:]))
                capture = cv2.VideoCapture(str(video))
                fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
                n_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                capture.release()
                description = ",".join(row[4:])
                activity = (
                    gmdcsa_fall_activity(description) if kind == "fall"
                    else gmdcsa_adl_activity(intervals, description)
                )
                onset = gmdcsa_fall_onset(intervals) if kind == "fall" else None
                out.append(
                    Series(
                        source="gmdcsa", name=f"s{subject}_{kind}_{video.stem}",
                        person=f"gmdcsa:{subject}", kind=kind, activity=activity, split="test",
                        fps=fps, n_frames=n_frames, fall_onset_s=onset, path=str(video),
                    )
                )
    return out


def catalog(sources: tuple[str, ...] = ("urfd", "cauca", "gmdcsa")) -> list[Series]:
    builders = {"urfd": urfd_series, "cauca": cauca_series, "gmdcsa": gmdcsa_series}
    out: list[Series] = []
    for source in sources:
        out.extend(builders[source]())
    return out


def iter_frames(series: Series) -> Iterator[tuple[int, Any]]:
    """Кадры серии: (номер с 1, BGR)."""
    import cv2
    import numpy as np

    if series.source == "urfd":
        with zipfile.ZipFile(series.path) as archive:
            names = sorted(name for name in archive.namelist() if name.endswith(".png"))
            for index, name in enumerate(names, 1):
                image = cv2.imdecode(np.frombuffer(archive.read(name), np.uint8), cv2.IMREAD_COLOR)
                yield index, image
        return
    capture = cv2.VideoCapture(series.path)
    index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            index += 1
            yield index, frame
    finally:
        capture.release()


# ---------------------------------------------------------------- отложенная событийная выборка


def holdout_name(series: Series) -> str:
    return f"falls_{series.source}/{series.name}.mp4"


def holdout_label(series: Series) -> dict[str, Any]:
    """Метка ролика в формате data/labels: одно событие fall_immobile на падение, у ADL событий нет."""
    events = []
    duration = round(series.duration_s, 3)
    if series.kind == "fall":
        if series.fall_onset_s is None:
            raise ValueError(f"{series.key}: fall without onset")
        onset = round(series.fall_onset_s, 2)
        observed = duration - onset
        long_enough = observed + 1e-9 >= MIN_OBSERVED_AFTER_ONSET_S
        certain = long_enough and series.activity != "fall_on_bed"
        note = f"Падение ({series.activity}), после начала падения в ролике {observed:.1f} с"
        if not long_enough:
            note += f": меньше {MIN_OBSERVED_AFTER_ONSET_S:.0f} с, о неподвижности судить нельзя"
        elif not certain:
            note += ": падение на кровать, правило не отличает его от ADL «лёг на кровать»"
        events.append(
            {"type": "fall_immobile", "t_start": onset, "t_end": duration, "uncertain": not certain, "note": note}
        )
    sources = {
        "urfd": {"page": URFD_PAGE, "url": URFD_BASE + f"{series.name}-cam0-rgb.zip"},
        "cauca": {"page": CAUCA_PAGE, "url": CAUCA_PAGE},
        "gmdcsa": {"page": GMDCSA_PAGE, "url": GMDCSA_PAGE},
    }[series.source]
    return {
        "video": holdout_name(series),
        "events": events,
        "scenario": "S3",
        "description": f"{series.source} {series.name}: {series.kind}, {series.activity}, человек {series.person}",
        "rules_to_enable": ["fall_immobile"],
        "zones": [],
        "split": "test",
        "kind": series.kind,
        "activity": series.activity,
        "person": series.person,
        "fps": series.fps,
        "duration_s": duration,
        "sources": [sources],
    }


def _ffmpeg() -> str:
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def write_video(series: Series, dest: Path) -> int:
    """Перекодировать серию в H.264 mp4 с родной частотой; число кадров сохраняется."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.stem + ".part.mp4")
    process = None
    count = 0
    try:
        for _, frame in iter_frames(series):
            if process is None:
                height, width = frame.shape[:2]
                process = subprocess.Popen(
                    [
                        _ffmpeg(), "-nostdin", "-loglevel", "error", "-y",
                        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
                        "-r", f"{series.fps:.3f}", "-i", "-",
                        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                        "-pix_fmt", "yuv420p", "-g", "50", "-an", str(tmp),
                    ],
                    stdin=subprocess.PIPE,
                )
            assert process.stdin is not None
            process.stdin.write(frame.tobytes())
            count += 1
        if process is None:
            raise ValueError(f"{series.key}: no frames")
        process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError(f"ffmpeg failed for {series.key}")
        tmp.replace(dest)
    finally:
        if process is not None and process.poll() is None:
            process.kill()
        tmp.unlink(missing_ok=True)
    return count


def scenarios_yaml(items: list[Series]) -> str:
    """ml/eval/scenarios_falls.yaml: правило fall_immobile с ТЕМИ ЖЕ параметрами, что в scenarios.yaml."""
    import yaml

    main = yaml.safe_load((ROOT / "ml" / "eval" / "scenarios.yaml").read_text(encoding="utf-8"))
    fall = main["rule_defaults"]["fall"]
    lines = [
        "# Отложенная выборка падений и ADL (ml/datasets/falls.py holdout, docs/QUALITY.md).",
        "# Ролики собраны из test-серий открытых наборов: URFD (люди P2, P5, B, E), CAUCAFall (люди 8–10),",
        "# GMDCSA24 (Subject 1 и 4; Subject 2 и 3 — источник s3_02, не используются). В обучение не попадают.",
        "# Правило fall_immobile — с теми же параметрами, что rule_defaults.fall в ml/eval/scenarios.yaml,",
        "# без подгонки; файл генерируется, параметры копируются оттуда.",
        "# Падение достоверно, если после его начала в ролике видно не меньше 3 с, иначе uncertain.",
        "",
        "version: 1",
        "root: ../..",
        "videos_dir: data/videos",
        "labels_dir: data/labels/falls",
        "models_config: configs/models.yaml",
        "models_dir: models",
        f"sample_fps: {main.get('sample_fps', 5.0)}",
        f"t0: {main.get('t0', 1700000000.0)}",
        "",
        "rule_defaults:",
        "  fall: &fall",
    ]
    for key, value in fall.items():
        lines.append(f"    {key}: {json.dumps(value, ensure_ascii=False)}")
    lines += ["", "scenarios:"]
    for series in items:
        ident = f"{series.source}_{series.name}".replace("-", "_")
        lines += [
            f"  - id: {ident}",
            f"    video: {holdout_name(series)}",
            "    camera:",
            f"      camera_id: falls_{ident}",
            f"      name: \"{series.source} {series.name} ({series.kind}, {series.activity})\"",
            "      site: Отложенная выборка падений",
            "      zones: []",
            "      rules: [*fall]",
        ]
    return "\n".join(lines) + "\n"


def label_path(series: Series) -> Path:
    name = holdout_name(series).replace("/", "__").removesuffix(".mp4")
    return data_dir() / "labels" / "falls" / f"{name}.json"


def build_holdout(force: bool = False, sources: tuple[str, ...] = ("urfd", "cauca", "gmdcsa")) -> dict[str, Any]:
    """Ролики и метки test-серий выбранных источников; сценарии и длительности — по всем готовым меткам."""
    data = data_dir()
    for series in catalog(sources):
        if series.split != "test":
            continue
        dest = data / "videos" / holdout_name(series)
        if force or not dest.is_file():
            series.n_frames = write_video(series, dest)
        path = label_path(series)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(holdout_label(series), ensure_ascii=False, indent=2) + "\n")

    items = [series for series in catalog() if series.split == "test" and label_path(series).is_file()]
    durations = {}
    summary: dict[str, Any] = defaultdict(lambda: defaultdict(float))
    for series in items:
        label = json.loads(label_path(series).read_text())
        durations[label["video"]] = label["duration_s"]
        bucket = summary[series.source]
        bucket[series.kind] += 1
        for event in label["events"]:
            bucket["fall_certain" if not event["uncertain"] else "fall_uncertain"] += 1
        bucket["seconds"] = round(bucket["seconds"] + label["duration_s"], 2)
    (data / "durations_falls.json").write_text(json.dumps(durations, indent=2) + "\n")
    (ROOT / "ml" / "eval" / "scenarios_falls.yaml").write_text(scenarios_yaml(items), encoding="utf-8")
    return {key: dict(value) for key, value in summary.items()}


# ---------------------------------------------------------------- покадровый recall лежачих


def _iou(a: list[float], b: list[float]) -> float:
    width = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    height = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = width * height
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _xyxy(box: list[float], width: int, height: int) -> list[float]:
    _, cx, cy, w, h = box
    return [(cx - w / 2) * width, (cy - h / 2) * height, (cx + w / 2) * width, (cy + h / 2) * height]


def recall_targets(series: Series) -> dict[int, list[list[float]] | None]:
    """Кадры для покадрового recall: номер кадра → рамки лежачего (норм. YOLO) или None, если рамок нет.

    CAUCAFall: кадры с рамкой класса fall (человек падает или лежит). URFD: кадры с меткой 1 (лежит),
    человек в кадре один, рамки нет — попаданием считается любая рамка человека.
    """
    if series.source == "cauca":
        return {
            frame: [box for box in boxes if int(box[0]) == 1]
            for frame, boxes in series.boxes.items()
            if any(int(box[0]) == 1 for box in boxes)
        }
    if series.source == "urfd":
        return {frame: None for frame, value in series.lying.items() if value == 1}
    return {}


def upright_targets(series: Series) -> dict[int, list[list[float]]]:
    """Кадры CAUCAFall с человеком класса nofall — проверка, что дообучение не теряет стоящих."""
    if series.source != "cauca":
        return {}
    return {
        frame: [box for box in boxes if int(box[0]) == 0]
        for frame, boxes in series.boxes.items()
        if boxes and all(int(box[0]) == 0 for box in boxes)
    }


def recall_views(found: dict[str, list[Any]], unions: list[tuple[str, str]], thr: float) -> dict[str, list[Any]]:
    """Люди с conf ≥ thr: по каждой модели и по объединению det + pose, как его делает конвейер.

    Объединение — guard.pipeline.worker.person_detections с pose_fill: люди det и люди pose, которых det не видит.
    """
    from guard.pipeline.worker import person_detections

    kept = {model_id: [d for d in dets if d.conf >= thr] for model_id, dets in found.items()}
    views = {model_id: [d for d in dets if d.cls == "person"] for model_id, dets in kept.items()}
    for det_id, pose_id in unions:
        views[f"{det_id}+{pose_id}"] = person_detections({"det": kept[det_id], "pose": kept[pose_id]}, pose_fill=True)
    return views


def frame_recall(
    models_config: Path,
    models_dir: Path,
    model_ids: list[str],
    thresholds: tuple[float, ...] = (0.25, 0.35, 0.45),
    iou_min: float = 0.5,
    split: str = "test",
    step: int = 1,
    unions: list[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    """Recall лежачих (и стоящих — для контроля) по моделям реестра на кадрах split-серий.

    unions — пары (det, pose): recall объединения людей, как в конвейере с GUARD_PERSON_POSE_FILL.
    """
    from guard.inference import get_backend
    from guard.pipeline.config import load_models
    from guard.pipeline.detector import Detector

    unions = list(unions or [])
    model_ids = list(dict.fromkeys([*model_ids, *(model_id for pair in unions for model_id in pair)]))
    specs = load_models(models_config, models_dir)
    backend = get_backend("onnx-cpu", models_dir=models_dir, intra_op_threads=2, max_batch=1)
    detectors = {}
    for model_id in model_ids:
        spec = specs[model_id].model_copy(update={"conf_threshold": min(thresholds) - 0.05})
        backend.load(spec)
        detectors[model_id] = Detector(spec, backend)
    stats: dict[str, Any] = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: [0, 0])))
    try:
        for series in catalog(("urfd", "cauca")):
            if series.split != split:
                continue
            lying = recall_targets(series)
            upright = upright_targets(series)
            wanted = {frame for frame in (*lying, *upright) if frame % step == 0 or step == 1}
            if not wanted:
                continue
            for frame_no, frame in iter_frames(series):
                if frame_no not in wanted:
                    continue
                height, width = frame.shape[:2]
                found = {model_id: detector.detect(frame) for model_id, detector in detectors.items()}
                for thr in thresholds:
                    for view, kept in recall_views(found, unions, thr).items():
                        for group, targets in (("lying", lying), ("upright", upright)):
                            if frame_no not in targets:
                                continue
                            boxes = targets[frame_no]
                            cell = stats[view][f"{series.source}/{group}"][f"{thr:.2f}"]
                            if boxes is None:  # URFD: человек один, рамки нет
                                cell[0] += 1
                                cell[1] += int(bool(kept))
                                continue
                            for box in boxes:
                                gt = _xyxy(box, width, height)
                                cell[0] += 1
                                cell[1] += int(any(_iou(list(d.bbox), gt) >= iou_min for d in kept))
    finally:
        backend.close()
    out: dict[str, Any] = {}
    for view, groups in stats.items():
        out[view] = {
            group: {
                thr: {"targets": total, "hits": hits, "recall": round(hits / total, 4) if total else None}
                for thr, (total, hits) in sorted(cells.items())
            }
            for group, cells in sorted(groups.items())
        }
    return {
        "split": split,
        "iou_min": iou_min,
        "models_config": str(models_config),
        "specs": {
            model_id: {"onnx": specs[model_id].onnx_path, "input": list(specs[model_id].input_size)}
            for model_id in model_ids
        },
        "recall": out,
    }


# ---------------------------------------------------------------- кадры для обучения


def prepare_frames(step: int = 3, lying_step: int = 2) -> dict[str, int]:
    """JPEG-кадры train/val серий URFD и CAUCAFall в frames/<split>/ и frames.jsonl.

    Лежащих и падающих (метки 0/1 URFD, класс fall CAUCAFall) берём чаще: их и не хватает модели.
    """
    import cv2

    root = falls_dir()
    out_path = root / "frames.jsonl"
    counts: dict[str, int] = defaultdict(int)
    with out_path.open("w") as stream:
        for series in catalog(("urfd", "cauca")):
            if series.split not in ("train", "val"):
                continue
            for frame_no, frame in iter_frames(series):
                if series.source == "urfd":
                    pose = series.lying.get(frame_no)
                    fallen = pose is not None and pose >= 0
                    gt = None
                else:
                    if frame_no not in series.boxes:
                        continue  # кадр без метки: не знаем, есть ли человек
                    gt = series.boxes[frame_no]
                    fallen = any(int(box[0]) == 1 for box in gt)
                    pose = None
                every = lying_step if fallen else step
                if (frame_no - 1) % every:
                    continue
                name = f"{series.source}__{series.name}__{frame_no:05d}.jpg"
                dest = root / "frames" / series.split / name
                if not dest.is_file():
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(dest), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
                record = {
                    "image": str(dest.relative_to(root)),
                    "split": series.split,
                    "source": series.source,
                    "series": series.name,
                    "person": series.person,
                    "frame": frame_no,
                    "fallen": fallen,
                    "urfd_pose": pose,
                    "gt": gt,
                    "size": [frame.shape[1], frame.shape[0]],
                }
                stream.write(json.dumps(record) + "\n")
                counts[f"{series.split}/{series.source}"] += 1
                counts[f"{series.split}/fallen"] += int(fallen)
    return dict(counts)


def calib_replay() -> list[dict[str, Any]]:
    """Кадры калибровки без демо-камер (кадры demo_camera — из роликов оценки, их не берём)."""
    manifest = data_dir() / "calib" / "manifest.csv"
    out = []
    if not manifest.is_file():
        return out
    with manifest.open() as stream:
        for row in csv.DictReader(stream):
            if row["theme"] == "demo_camera":
                continue
            out.append({"image": str(data_dir() / "calib" / row["file"]), "theme": row["theme"]})
    return out


# ---------------------------------------------------------------- учитель


def run_teacher(weights: Path, conf: float = 0.25, imgsz: int = 640, device: str = "mps") -> Path:
    """yolov8x-pose по кадрам train/val и кадрам калибровки: рамки и 17 точек (torch, тяжёлое)."""
    from ultralytics import YOLO

    root = falls_dir()
    records = [json.loads(line) for line in (root / "frames.jsonl").open()]
    images = [str(root / record["image"]) for record in records]
    images += [item["image"] for item in calib_replay()]
    out_path = root / "teacher.jsonl"
    done = set()
    if out_path.is_file():
        done = {json.loads(line)["image"] for line in out_path.open()}
    model = YOLO(str(weights))
    with out_path.open("a") as stream:
        todo = [image for image in images if image not in done]
        for start in range(0, len(todo), 16):
            batch = todo[start : start + 16]
            results = model.predict(batch, imgsz=imgsz, conf=conf, device=device, verbose=False)
            for image, result in zip(batch, results):
                dets = []
                if result.boxes is not None and len(result.boxes):
                    xyxy = result.boxes.xyxy.cpu().numpy().tolist()
                    scores = result.boxes.conf.cpu().numpy().tolist()
                    kpts = result.keypoints.data.cpu().numpy().tolist() if result.keypoints is not None else []
                    for index, box in enumerate(xyxy):
                        dets.append({"bbox": box, "conf": scores[index], "kpts": kpts[index] if kpts else []})
                height, width = result.orig_shape
                stream.write(json.dumps({"image": image, "size": [width, height], "dets": dets}) + "\n")
            if start % 800 == 0:
                print(f"teacher {start + len(batch)}/{len(todo)}", flush=True)
    return out_path


# ---------------------------------------------------------------- набор YOLO-pose


def _yolo_pose_line(box: list[float], kpts: list[list[float]] | None, width: int, height: int) -> str:
    x1, y1, x2, y2 = box
    x1, x2 = max(0.0, x1), min(float(width), x2)
    y1, y2 = max(0.0, y1), min(float(height), y2)
    values = [0, (x1 + x2) / 2 / width, (y1 + y2) / 2 / height, (x2 - x1) / width, (y2 - y1) / height]
    points = kpts or [[0.0, 0.0, 0.0]] * 17
    for x, y, score in points:
        visible = 2 if score >= 0.5 else 0
        values += [x / width if visible else 0.0, y / height if visible else 0.0, visible]
    return " ".join(f"{value:.6f}" if isinstance(value, float) else str(value) for value in values)


def _teacher_index() -> dict[str, dict[str, Any]]:
    path = falls_dir() / "teacher.jsonl"
    index = {}
    for line in path.open():
        record = json.loads(line)
        index[record["image"]] = record
    return index


def _coco_ids(name: str) -> list[int]:
    path = raw_dir() / "coco" / name
    return json.loads(path.read_text()) if path.is_file() else []


def coco_label_lines(annotations: list[dict[str, Any]], width: int, height: int) -> list[str]:
    """Люди COCO в строки YOLO-pose: рамка и 17 точек; рамки меньше 4 пикселей пропускаем."""
    lines = []
    for a in annotations:
        x, y, w, h = a["bbox"]
        if w < 4 or h < 4:
            continue
        points = a["keypoints"]
        kpts = [[points[3 * k], points[3 * k + 1], 1.0 if points[3 * k + 2] > 0 else 0.0] for k in range(17)]
        lines.append(_yolo_pose_line([x, y, x + w, y + h], kpts, width, height))
    return lines


def _coco_items(ids: list[int]) -> Iterator[tuple[int, Path, list[str]]]:
    """(позиция в ids, изображение, строки меток) для скачанных изображений COCO; толпу (iscrowd) пропускаем."""
    coco_root = raw_dir() / "coco"
    ann_path = coco_root / "person_keypoints_val2017.json"
    if not ids or not ann_path.is_file():
        return
    ann = json.loads(ann_path.read_text())
    images = {img["id"]: img for img in ann["images"]}
    by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    crowd: set[int] = set()
    for a in ann["annotations"]:
        if a["iscrowd"]:
            crowd.add(a["image_id"])
        else:
            by_image[a["image_id"]].append(a)
    for position, image_id in enumerate(ids):
        if image_id in crowd:
            continue  # толпу с маской iscrowd не размечаем рамками
        img = images[image_id]
        path = coco_root / "images" / img["file_name"]
        if path.is_file():
            yield position, path, coco_label_lines(by_image[image_id], img["width"], img["height"])


def build_yolo(teacher_conf: float = 0.5, coco_val: int = 200) -> dict[str, Any]:
    """Набор YOLO-pose: кадры падений + повтор COCO и калибровки. Рамки — разметка набора или учитель.

    CAUCAFall: рамки из разметки набора, точки — от учителя при IoU ≥ 0.5, иначе точки невидимы.
    URFD: человек один, рамка и точки — лучшая детекция учителя ≥ teacher_conf; без неё кадр не берём,
    чтобы не учить «здесь никого нет». Калибровка: все детекции учителя ≥ teacher_conf.
    COCO val2017: разметка COCO (рамки и точки людей), часть изображений — в val.
    """
    root = falls_dir()
    out = root / "yolo"
    teacher = _teacher_index()
    counts: dict[str, int] = defaultdict(int)
    lists: dict[str, list[str]] = {"train": [], "val": []}

    def emit(split: str, image: Path, lines: list[str], tag: str) -> None:
        stem = f"{tag}__{image.stem}"
        image_dir = out / "images" / split
        label_dir = out / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        link = image_dir / f"{stem}{image.suffix}"
        if not link.exists():
            link.symlink_to(image.resolve())
        (label_dir / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        lists[split].append(str(link))
        counts[f"{split}/{tag}"] += 1

    for line in (root / "frames.jsonl").open():
        record = json.loads(line)
        image = root / record["image"]
        width, height = record["size"]
        found = teacher.get(str(image), {"dets": []})["dets"]
        lines = []
        if record["source"] == "cauca":
            for box in record["gt"] or []:
                gt = _xyxy(box, width, height)
                best = max(found, key=lambda det: _iou(det["bbox"], gt), default=None)
                kpts = best["kpts"] if best is not None and _iou(best["bbox"], gt) >= 0.5 else None
                lines.append(_yolo_pose_line(gt, kpts, width, height))
        else:
            best = max(found, key=lambda det: det["conf"], default=None)
            if best is None or best["conf"] < teacher_conf:
                counts["skipped/urfd_no_teacher"] += 1
                continue
            lines.append(_yolo_pose_line(best["bbox"], best["kpts"], width, height))
        emit(record["split"], image, lines, record["source"])

    for item in calib_replay():
        record = teacher.get(item["image"])
        if record is None:
            continue
        width, height = record["size"]
        lines = [
            _yolo_pose_line(det["bbox"], det["kpts"], width, height)
            for det in record["dets"]
            if det["conf"] >= teacher_conf
        ]
        emit("train", Path(item["image"]), lines, "calib")

    for position, path, lines in _coco_items(_coco_ids("chosen.json")):
        emit("val" if position < coco_val else "train", path, lines, "coco")

    data_yaml = out / "data.yaml"
    data_yaml.write_text(
        "# Набор YOLO-pose для дообучения на лежачих (ml/datasets/falls.py yolo).\n"
        f"path: {out}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test  # test-серии CAUCAFall и COCO вне обучения (ml/datasets/falls.py yolo-test)\n"
        "kpt_shape: [17, 3]\n"
        f"flip_idx: {COCO_FLIP_IDX}\n"
        "names:\n  0: person\n"
    )
    (out / "images" / "test").mkdir(parents=True, exist_ok=True)
    splits: dict[str, dict[str, dict[str, int]]] = defaultdict(dict)
    for key, value in counts.items():
        split, tag = key.split("/", 1)
        if split in ("train", "val"):
            splits[split][tag] = {"images": value}
    manifest = {
        "version": YOLO_VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "sources": {
            "urfd": {"url": URFD_PAGE}, "cauca": {"url": CAUCA_PAGE}, "gmdcsa": {"url": GMDCSA_PAGE},
            "coco": {"url": COCO_IMAGES_ZIP}, "calib": {"url": "data/calib/manifest.csv (без demo_camera)"},
        },
        "splits": dict(splits),
        "split": {
            "urfd_test_persons": sorted(URFD_TEST_PERSONS), "urfd_val_persons": sorted(URFD_VAL_PERSONS),
            "cauca_test_subjects": sorted(CAUCA_TEST_SUBJECTS), "cauca_val_subjects": sorted(CAUCA_VAL_SUBJECTS),
            "gmdcsa": f"Subject {GMDCSA_SUBJECTS} — только test; {GMDCSA_EXCLUDED} — s3_02, не используются",
        },
        "counts": dict(counts),
        "sha256_lists": {
            split: hashlib.sha256("\n".join(sorted(items)).encode()).hexdigest()
            for split, items in lists.items()
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest


def pick_test_frames(series: Series, upright_every: int = 2) -> dict[int, bool]:
    """Кадры test-серии CAUCAFall для теста детекции: номер кадра → лежит ли в нём человек.

    Берём все размеченные кадры с рамкой класса fall и каждый upright_every-й размеченный кадр без неё.
    Кадры без рамок пропускаем: в CAUCAFall это кадр без человека или без разметки.
    """
    out: dict[int, bool] = {}
    upright = 0
    for frame in sorted(series.boxes):
        boxes = series.boxes[frame]
        if not boxes:
            continue
        lying = any(int(box[0]) == 1 for box in boxes)
        if not lying:
            upright += 1
            if (upright - 1) % upright_every:
                continue
        out[frame] = lying
    return out


def build_yolo_test(upright_every: int = 2) -> dict[str, int]:
    """Тест детекции людей для гейта: test-серии CAUCAFall (люди 8–10, рамки разметки) и COCO вне обучения.

    Кадры — в yolo/images/test, метки — в yolo/labels/test; точки не размечены (видимость 0), гейт считает
    mAP50 рамок. Подмножества для гейта: test_lying.txt (лежит человек), test_upright.txt, test_coco.txt.
    """
    import cv2

    out = falls_dir() / "yolo"
    image_dir, label_dir = out / "images" / "test", out / "labels" / "test"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    lists: dict[str, list[str]] = {"lying": [], "upright": [], "coco": []}
    for series in cauca_series():
        if series.split != "test":
            continue
        wanted = pick_test_frames(series, upright_every)
        if not wanted:
            continue
        for frame_no, frame in iter_frames(series):
            if frame_no not in wanted:
                continue
            height, width = frame.shape[:2]
            name = f"cauca__{series.name}__{frame_no:05d}"
            image = image_dir / f"{name}.jpg"
            if not image.is_file():
                cv2.imwrite(str(image), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            lines = [_yolo_pose_line(_xyxy(box, width, height), None, width, height) for box in series.boxes[frame_no]]
            (label_dir / f"{name}.txt").write_text("\n".join(lines) + "\n")
            lists["lying" if wanted[frame_no] else "upright"].append(str(image))
    for _, path, lines in _coco_items(_coco_ids("test.json")):
        link = image_dir / f"coco__{path.name}"
        if not link.exists():
            link.symlink_to(path.resolve())
        (label_dir / f"coco__{path.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        lists["coco"].append(str(link))
    for key, items in lists.items():
        (out / f"test_{key}.txt").write_text("\n".join(items) + "\n")
    return {key: len(items) for key, items in lists.items()}


# ---------------------------------------------------------------- разбор прогона по источникам и видам ADL


def breakdown(result: dict[str, Any], tol_s: float = 2.0) -> dict[str, Any]:
    """Разбор JSON прогона run_offline по источникам и видам действий (данные — метки data/labels/falls)."""
    labels_dir = data_dir() / "labels" / "falls"
    labels = {}
    for path in labels_dir.glob("*.json"):
        label = json.loads(path.read_text())
        labels[label["video"]] = label
    by_video: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for item in result["items"]:
        mapping = result["mapping"][item["camera_id"]]
        by_video[mapping["video"]].append(
            (item["started_at"] - mapping["t0"], item["confirmed_at"] - mapping["t0"])
        )
    table: dict[str, dict[str, Any]] = defaultdict(lambda: defaultdict(float))
    for video, label in labels.items():
        if video not in result["durations"]:
            continue
        source = video.split("/")[0].removeprefix("falls_")
        row = table[f"{source}/{label['kind']}/{label['activity']}"]
        row["videos"] += 1
        row["seconds"] = round(row["seconds"] + label["duration_s"], 2)
        preds = by_video.get(video, [])
        events = label["events"]
        if not events:
            row["fp"] += len(preds)
            row["videos_with_fp"] += int(bool(preds))
            continue
        event = events[0]
        hit = [p for p in preds if p[0] - tol_s <= event["t_end"] and p[1] + tol_s >= event["t_start"]]
        if event["uncertain"]:
            row["uncertain"] += 1
            row["uncertain_alerted"] += int(bool(hit))
        else:
            row["tp" if hit else "fn"] += 1
            if hit:
                row["latency"] = row.get("latency", 0.0) + (min(p[1] for p in hit) - event["t_start"])
        row["fp"] += len(preds) - len(hit)
    out = {}
    for key, row in sorted(table.items()):
        row = dict(row)
        if row.get("tp"):
            row["latency_mean_s"] = round(row.pop("latency") / row["tp"], 2)
        out[key] = {k: (int(v) if isinstance(v, float) and v.is_integer() else v) for k, v in row.items()}
    return out


# ---------------------------------------------------------------- CLI


def write_manifest() -> dict[str, Any]:
    items = catalog()
    summary: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for series in items:
        summary[series.source][f"{series.split}/{series.kind}"] += 1
    manifest = {
        "sources": {
            "urfd": {"page": URFD_PAGE, "files": URFD_BASE + "<seq>-cam0-rgb.zip, urfall-cam0-*.csv"},
            "cauca": {"page": CAUCA_PAGE, "api": CAUCA_API, "zip": CAUCA_ZIP},
            "gmdcsa": {"page": GMDCSA_PAGE, "subjects": list(GMDCSA_SUBJECTS), "excluded": list(GMDCSA_EXCLUDED)},
            "coco": {"images": COCO_IMAGES_ZIP, "annotations": COCO_ANN_ZIP},
            "le2i": "зеркала без авторизации нет (le2i.cnrs.fr недоступен, imvia — 404, Kaggle — вход, "
            "HF seanphan/le2i-sentinel-frames — 3 кадра на ролик)",
        },
        "split_rule": "по людям и сериям; URFD — по одежде, CAUCAFall — по номеру человека",
        "series": [
            {k: v for k, v in asdict(series).items() if k not in ("lying", "boxes")}
            for series in items
        ],
        "summary": {key: dict(value) for key, value in summary.items()},
    }
    falls_dir().mkdir(parents=True, exist_ok=True)
    (falls_dir() / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest["summary"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    download = sub.add_parser("download")
    download.add_argument("--only", default="urfd,cauca,gmdcsa,coco,teacher")
    download.add_argument("--coco-images", type=int, default=1500)
    download.add_argument("--coco-test", type=int, default=0, help="изображений COCO для проверки забывания")
    download.add_argument("--subjects", default="8,9,10,1,2,3,4,5,6,7", help="люди CAUCAFall по порядку")
    download.add_argument("--label-step", type=int, default=4, help="метки CAUCAFall каждого N-го кадра")
    holdout = sub.add_parser("holdout")
    holdout.add_argument("--force", action="store_true")
    holdout.add_argument("--sources", default="urfd,cauca,gmdcsa")
    recall = sub.add_parser("recall")
    recall.add_argument("--models-config", type=Path, default=ROOT / "configs" / "models.yaml")
    recall.add_argument("--models-dir", type=Path, default=Path(os.environ.get("GUARD_MODELS_DIR") or ROOT / "models"))
    recall.add_argument("--model", action="append", default=[])
    recall.add_argument("--union", action="append", default=[], help="det+pose: объединение, как в конвейере")
    recall.add_argument("--split", default="test")
    recall.add_argument("--out", type=Path)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--step", type=int, default=3)
    prepare.add_argument("--lying-step", type=int, default=2)
    teacher = sub.add_parser("teacher")
    teacher.add_argument("--device", default="mps")
    yolo = sub.add_parser("yolo")
    yolo.add_argument("--teacher-conf", type=float, default=0.5)
    yolo_test = sub.add_parser("yolo-test")
    yolo_test.add_argument("--upright-every", type=int, default=2)
    sub.add_parser("manifest")
    report = sub.add_parser("report")
    report.add_argument("result", type=Path)
    args = parser.parse_args(argv)

    if args.command == "download":
        steps = {
            "urfd": download_urfd,
            "cauca": lambda: download_cauca(tuple(int(s) for s in args.subjects.split(",")), args.label_step),
            "gmdcsa": download_gmdcsa,
            "coco": lambda: download_coco(args.coco_images, args.coco_test),
            "teacher": download_teacher,
        }
        for name in args.only.split(","):
            steps[name]()
        print(json.dumps(write_manifest(), ensure_ascii=False, indent=2))
    elif args.command == "report":
        print(json.dumps(breakdown(json.loads(args.result.read_text())), ensure_ascii=False, indent=1))
    elif args.command == "manifest":
        print(json.dumps(write_manifest(), ensure_ascii=False, indent=2))
    elif args.command == "holdout":
        summary = build_holdout(force=args.force, sources=tuple(args.sources.split(",")))
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    elif args.command == "recall":
        unions = [tuple(item.split("+", 1)) for item in args.union]
        result = frame_recall(args.models_config, args.models_dir, args.model, split=args.split, unions=unions)
        text = json.dumps(result, ensure_ascii=False, indent=2)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(text + "\n")
        print(text)
    elif args.command == "prepare":
        print(json.dumps(prepare_frames(args.step, args.lying_step), indent=2))
    elif args.command == "teacher":
        print(run_teacher(download_teacher(), device=args.device))
    elif args.command == "yolo":
        print(json.dumps(build_yolo(args.teacher_conf), ensure_ascii=False, indent=2))
    elif args.command == "yolo-test":
        print(json.dumps(build_yolo_test(args.upright_every), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
