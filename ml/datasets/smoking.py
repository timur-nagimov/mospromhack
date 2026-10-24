"""Набор «сигарета» для S7: скачивание, авторазметка учителем, дедуп, разбиение, manifest.json.

Класс один: 0 cigarette. Источники (лицензии не проверяли — решение человека 24.10, демо внутреннее):
- zsc_cigarette — HF zsc-bdic/cigarette: экспорт Roboflow в формате YOLO, сигареты размечены людьми.
  Архив 10 ГБ, но почти весь объём — кэш .npy. По HTTP Range берём только jpg и txt (~1 ГБ);
- mixkit_smoking — открытые стоковые ролики Mixkit с курением. Кадры 1 к/с, рамки ставит учитель.
  Берём только кадры, где учитель уверенно нашёл сигарету: на остальных она может быть, но не найдена;
- workplace_neg — кадры стройки и склада из набора погрузчика (data/datasets/forklift): негативы.
  Кадр, где любой учитель видит сигарету с conf ≥ 0.25, выбрасывается — не делаем из возможной сигареты фон.

Учитель — две модели YOLO11m с HF, обученные на других наборах Roboflow:
Beehzod/smoking-detection-finetuned (xu-wei/cigarette-all) и Beehzod/smoke_cigarette-detection2-yolo11m
(smoking-detection-ggnhq). Рамка ролика принимается, если обе модели согласны (IoU ≥ 0.3) или одна уверена
(conf ≥ TEACHER_SOLO_CONF). Чекпойнты проверены torch.serialization.get_unsafe_globals_in_checkpoint:
только классы torch и ultralytics.

Разбиение:
- zsc_cigarette — исходные train/valid/test;
- mixkit_smoking — только train: рамки от учителя, в val и test нужна ручная разметка;
- workplace_neg — та же часть, что у кадра в наборе погрузчика.
Дедуп — dHash 64 бит, расстояние Хэмминга ≤ 5; из пары остаётся кадр более «строгой» части (test > val > train).
Кадры сверяются с роликами оценки (ml/eval/scenarios.yaml и ml/eval/scenarios_smoking.yaml, 2 к/с):
совпавшие выбрасываются. Ролики оценки и их серии (один автор, одна съёмка) в обучение не берутся.

Примеры:
    python ml/datasets/smoking.py download
    python ml/datasets/smoking.py frames
    tools/heavy.sh python ml/datasets/smoking.py teacher
    python ml/datasets/smoking.py build
    python ml/datasets/smoking.py sheet --split train --n 24      # лист для проверки разметки глазами
"""

from __future__ import annotations

import argparse
import io
import json
import random
import re
import shutil
import struct
import sys
import threading
import time
import urllib.request
import zipfile
import zlib
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.datasets.forklift import (  # noqa: E402
    RangeFile,
    Sample,
    _iou_cxcywh,
    dhash,
    file_list_sha,
    holdout_hashes,
    http_get,
)

CLASSES = ("cigarette",)
VERSION = "smoking-ds-v1"
SPLITS = ("train", "val", "test")
SPLIT_RANK = {"test": 0, "val": 1, "train": 2}
DUP_HAMMING = 5
FRAME_FPS = 1.0
TEACHER_CONF = 0.35  # порог каждой модели учителя
TEACHER_SOLO_CONF = 0.6  # одна модель без согласия второй
TEACHER_IOU = 0.3
TEACHER_IMGSZ = 960  # сигарета мелкая: учитель смотрит в большем разрешении
NEG_REJECT_CONF = 0.5  # негатив с уверенной вытянутой «сигаретой» выбрасываем
CIG_ASPECT = 2.0  # длинная сторона рамки сигареты к короткой
NEG_PER_SPLIT = {"train": 2500, "val": 300, "test": 300}

ZSC_URL = "https://huggingface.co/datasets/zsc-bdic/cigarette/resolve/main/data.zip"
MIXKIT_PAGE = "https://mixkit.co/free-stock-video/{slug}-{vid}/"
MIXKIT_MP4 = "https://assets.mixkit.co/videos/{vid}/{vid}-720.mp4"

SOURCES = {
    "zsc_cigarette": {
        "url": "https://huggingface.co/datasets/zsc-bdic/cigarette",
        "origin": "экспорт Roboflow (имена *.rf.*), разметка YOLO, один класс — сигарета",
        "role": "позитивы с ручной разметкой; train/valid/test — исходные",
    },
    "mixkit_smoking": {
        "url": "https://mixkit.co/free-stock-video/smoking/",
        "origin": "стоковые ролики Mixkit (Mixkit Free License), 720p",
        "role": "позитивы домена «человек курит в кадре»: кадры 1 к/с, рамки от учителя, только train",
    },
    "workplace_neg": {
        "url": "data/datasets/forklift (ml/datasets/forklift.py: Roboflow forklift, Logistics, Construction, COCO)",
        "origin": "кадры склада и стройки из набора погрузчика",
        "role": "негативы без сигарет; кадры с подозрением учителя на сигарету выброшены",
    },
}

# Ролики Mixkit для обучения: (id, slug, серия). Серия — одна съёмка; серии роликов оценки сюда не входят.
TRAIN_VIDEOS = [
    (41329, "man-smoking-a-cigarette", "41xxx-studio"),
    (41337, "woman-dancing-while-is-smoking", "41xxx-studio"),
    (41354, "woman-lighting-a-cigarette", "41xxx-studio"),
    (41485, "face-of-a-man-from-a-bygone-age-smoking-a", "41xxx-studio"),
    (13737, "stylish-man-smoking-a-cigarette-and-texting-a-message", "13737"),
    (39476, "young-man-smoking-in-the-street-on-a-cold-night", "39476"),
    (47192, "bearded-man-counting-money-while-he-smokes", "47192"),
    (44544, "stylish-woman-smoking-sitting-next-to-sports-car", "445xx-car"),
    (44550, "woman-lighting-a-cigarette-next-to-a-sports-car", "445xx-car"),
    (44558, "the-face-of-a-female-model-smoking-a-cigarette", "445xx-car"),
    (101050, "asian-woman-smoking-in-a-phone-booth", "1010xx-booth"),
    (3925, "depressed-guy-smoking-in-the-dark", "39xx-dark"),
    (3926, "girl-smoking-in-the-shadows-with-a-backlight", "39xx-dark"),
    (3918, "smoking-in-the-dark-with-a-taillight", "39xx-dark"),
    (927, "woman-smokes-under-window", "92x-window"),
    (928, "silhouette-of-woman-smoking", "92x-window"),
    (45, "biker-resting-next-to-a-highway", "biker"),
    (5, "man-sitting-on-his-motorcycle", "biker"),
    (35885, "man-smoking-on-the-seashore-at-dusk", "35885"),
    (8504, "man-smoking-a-cigarette-close-up", "85xx-close"),
    (8537, "hand-holding-cigarette-with-exhaled-smoke-on-black-background", "85xx-close"),
    (4732, "addicted-woman-surrounded-by-her-vices", "4732"),
    (2115, "clown-smoking-in-a-tunnel", "2115"),
]

# Отложенные ролики событийной оценки (ml/eval/scenarios_smoking.yaml). В обучение не попадают ни в каком виде.
# 398 и 399 — та же съёмка, что s7_01 (414): поэтому вся серия только в оценке.
EVAL_VIDEOS = [
    (398, "man-running-up-stairs-and-smoking", "smoking"),
    (399, "urban-man-smoking-at-a-ruined-building", "smoking"),
    (36446, "man-smoking-in-the-street-on-a-light-rainy-night", "smoking"),
    (36443, "man-smoking-out-of-his-car-at-night", "smoking"),
    (24063, "exhausted-worker-having-a-break-late-at-night", "clean"),
    (4801, "young-man-sitting-scrolling-on-his-cell-phone", "clean"),
    (4872, "two-coworkers-talking-and-laughing", "clean"),
    (47755, "workers-welding-heavy-metal-in-a-factory", "clean"),
]

TEACHERS = {
    "cigall_y11m": {
        "file": "teachers/beehzod_cigall_y11m/best.pt",
        "url": "https://huggingface.co/Beehzod/smoking-detection-finetuned",
        "dataset": "https://universe.roboflow.com/xu-wei/cigarette-all",
    },
    "cig2_y11m": {
        "file": "teachers/beehzod_cig2_y11m/best.pt",
        "url": "https://huggingface.co/Beehzod/smoke_cigarette-detection2-yolo11m",
        "dataset": "https://universe.roboflow.com/smoking-detection-5e8hh/smoking-detection-ggnhq",
    },
}


def log(msg: str) -> None:
    print(msg, flush=True)


# ---------- скачивание ----------


def _zip_member(url: str, info: zipfile.ZipInfo, dst: Path) -> None:
    """Один член zip по HTTP Range: локальный заголовок (30 байт + имя + extra), затем данные."""
    start = info.header_offset
    need = 30 + len(info.filename.encode()) + 1024 + info.compress_size
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{start + need - 1}"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                blob = r.read()
            break
        except OSError:
            if attempt == 3:
                raise
            time.sleep(2 * (attempt + 1))
    sig, _, _, method, _, _, _, _, _, nlen, elen = struct.unpack("<IHHHHHIIIHH", blob[:30])
    if sig != 0x04034B50:
        raise ValueError(f"нет локального заголовка: {info.filename}")
    data = blob[30 + nlen + elen: 30 + nlen + elen + info.compress_size]
    if method == 8:
        data = zlib.decompress(data, -15)
    elif method != 0:
        raise ValueError(f"{info.filename}: метод сжатия {method}")
    if len(data) != info.file_size:
        raise ValueError(f"{info.filename}: {len(data)} байт вместо {info.file_size}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".part")
    tmp.write_bytes(data)
    tmp.rename(dst)


def download_zsc(raw: Path, threads: int = 16) -> None:
    out = raw / "zsc_cigarette"
    rf = RangeFile(ZSC_URL)
    with urllib.request.urlopen(urllib.request.Request(ZSC_URL, method="HEAD"), timeout=60) as r:
        real_url = r.url  # прямой адрес CDN после редиректа: Range на нём быстрее
    z = zipfile.ZipFile(io.BufferedReader(rf, buffer_size=1 << 20))
    pattern = re.compile(r"^data/(train|valid|test)/(labels/[^/]+\.txt|images/[^/]+\.jpg)$")
    members = [i for i in z.infolist() if pattern.match(i.filename)]
    todo = [i for i in members
            if not ((out / i.filename).exists() and (out / i.filename).stat().st_size == i.file_size)]
    log(f"download: zsc_cigarette {len(members)} файлов, осталось {len(todo)}")
    lock = threading.Lock()
    done = [0]

    def one(info: zipfile.ZipInfo) -> None:
        _zip_member(real_url, info, out / info.filename)
        with lock:
            done[0] += 1
            if done[0] % 2000 == 0:
                log(f"  zsc_cigarette: {done[0]}/{len(todo)}")

    with ThreadPoolExecutor(threads) as ex:
        list(ex.map(one, todo))


def download_videos(out: Path) -> None:
    missing = []
    for videos, dst in ((TRAIN_VIDEOS, out / "raw" / "mixkit_videos"), (EVAL_VIDEOS, out / "eval_videos")):
        for vid, _, _ in videos:
            try:
                http_get(MIXKIT_MP4.format(vid=vid), dst / f"mixkit_{vid}.mp4")
            except RuntimeError as exc:  # часть новых роликов Mixkit отдаёт 403 по прямой ссылке
                missing.append(vid)
                log(f"  пропуск: {exc}")
    log(f"download: роликов train {len(TRAIN_VIDEOS)}, оценки {len(EVAL_VIDEOS)}, не скачались {missing}")


def cmd_download(args: argparse.Namespace) -> None:
    raw = args.out / "raw"
    only = set(args.only.split(","))
    if "videos" in only:
        download_videos(args.out)
    if "zsc_cigarette" in only:
        download_zsc(raw)


# ---------- кадры роликов ----------


def cmd_frames(args: argparse.Namespace) -> None:
    import cv2

    raw = args.out / "raw"
    for vid, _, _ in TRAIN_VIDEOS:
        src = raw / "mixkit_videos" / f"mixkit_{vid}.mp4"
        dst = raw / "mixkit_frames" / f"mixkit_{vid}"
        if dst.exists() and any(dst.iterdir()):
            continue
        dst.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(str(src))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        step = max(1, round(fps / FRAME_FPS))
        idx = n = 0
        while True:
            ok = cap.grab()
            if not ok:
                break
            if idx % step == 0:
                ok, frame = cap.retrieve()
                if ok:
                    cv2.imwrite(str(dst / f"mixkit_{vid}_t{idx / fps:06.1f}.jpg"), frame,
                                [cv2.IMWRITE_JPEG_QUALITY, 92])
                    n += 1
            idx += 1
        cap.release()
        log(f"frames: mixkit_{vid}: {n}")


# ---------- учитель ----------


def _letterbox_square(img: np.ndarray, size: int) -> tuple[np.ndarray, float, int, int]:
    """Кадр в квадрат size×size с полями. Одна форма входа: иначе MPS копит графы под каждую форму
    и процесс растёт до 9 ГБ (memwatch убил первый запуск учителя)."""
    import cv2

    h, w = img.shape[:2]
    scale = size / max(h, w)
    nw, nh = round(w * scale), round(h * scale)
    canvas = np.full((size, size, 3), 114, np.uint8)
    px, py = (size - nw) // 2, (size - nh) // 2
    canvas[py:py + nh, px:px + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    return canvas, scale, px, py


def _teacher_predict(weights: Path, paths: list[str], device: str, imgsz: int) -> dict[str, list]:
    """{путь: [[cx, cy, w, h, conf], ...]} — рамки сигареты (класс 0) с conf ≥ min(порогов), доли кадра."""
    import cv2
    import torch
    from ultralytics import YOLO

    model = YOLO(str(weights))
    out: dict[str, list] = {}
    for i, path in enumerate(paths):
        img = cv2.imread(path)
        if img is None:
            continue
        h, w = img.shape[:2]
        square, scale, px, py = _letterbox_square(img, imgsz)
        r = model.predict(square, imgsz=imgsz, conf=min(NEG_REJECT_CONF, TEACHER_CONF), classes=[0],
                          device=device, verbose=False)[0]
        rows = []
        for (x1, y1, x2, y2), c in zip(r.boxes.xyxy.cpu().numpy().tolist(), r.boxes.conf.cpu().numpy().tolist()):
            x1, x2 = (x1 - px) / scale / w, (x2 - px) / scale / w
            y1, y2 = (y1 - py) / scale / h, (y2 - py) / scale / h
            x1, y1, x2, y2 = (min(max(v, 0.0), 1.0) for v in (x1, y1, x2, y2))
            if x2 > x1 and y2 > y1:
                rows.append([round((x1 + x2) / 2, 5), round((y1 + y2) / 2, 5), round(x2 - x1, 5),
                             round(y2 - y1, 5), round(float(c), 3)])
        out[path] = rows
        if device == "mps" and i % 200 == 199:
            torch.mps.empty_cache()
    return out


def fuse_teachers(preds: list[list[list[float]]]) -> list[list[float]]:
    """Рамки двух учителей → итоговые: согласие (IoU ≥ TEACHER_IOU, обе ≥ TEACHER_CONF) или одна ≥ SOLO.

    Рамка согласия — средняя по двум, conf — минимальная. Возвращает [[cx, cy, w, h, conf], ...].
    """
    a = [b for b in preds[0] if b[4] >= TEACHER_CONF]
    b = [x for x in (preds[1] if len(preds) > 1 else []) if x[4] >= TEACHER_CONF]
    out: list[list[float]] = []
    used_b: set[int] = set()
    for box in sorted(a, key=lambda x: -x[4]):
        best, best_iou = None, 0.0
        for j, other in enumerate(b):
            if j in used_b:
                continue
            iou = _iou_cxcywh(box[:4], other[:4])
            if iou > best_iou:
                best, best_iou = j, iou
        if best is not None and best_iou >= TEACHER_IOU:
            used_b.add(best)
            other = b[best]
            out.append([round((box[k] + other[k]) / 2, 5) for k in range(4)] + [min(box[4], other[4])])
        elif box[4] >= TEACHER_SOLO_CONF:
            out.append(box)
    for j, other in enumerate(b):
        if j not in used_b and other[4] >= TEACHER_SOLO_CONF:
            out.append(other)
    return out


def _workplace_candidates(forklift: Path, seed: int) -> dict[str, list[Path]]:
    rng = random.Random(seed)
    out = {}
    for split in SPLITS:
        files = sorted((forklift / "images" / split).glob("*.jpg"))
        rng.shuffle(files)
        out[split] = files[: int(NEG_PER_SPLIT[split] * 1.3)]  # запас на отбраковку учителем
    return out


def cmd_teacher(args: argparse.Namespace) -> None:
    """Оба учителя на кадрах роликов (позитивы) и на кандидатах в негативы. torch — через tools/heavy.sh."""
    raw = args.out / "raw"
    out_file = raw / "teacher_cigarette.json"
    state = json.loads(out_file.read_text()) if out_file.exists() else {"models": {}, "images": {}}
    frames = sorted(str(p) for p in (raw / "mixkit_frames").rglob("*.jpg"))
    negs = [str(p) for files in _workplace_candidates(args.forklift, args.seed).values() for p in files]
    t0 = time.time()
    for name, spec in TEACHERS.items():
        weights = args.out / spec["file"]
        have = state["images"]
        paths = [p for p in frames + negs if name not in have.get(p, {})]
        if not paths:
            continue
        pred = _teacher_predict(weights, paths, args.device, TEACHER_IMGSZ)
        for p in paths:
            have.setdefault(p, {})[name] = pred.get(p, [])
        state["models"][name] = {**spec, "imgsz": TEACHER_IMGSZ}
        out_file.write_text(json.dumps(state))
        log(f"teacher: {name} — {len(paths)} кадров, {time.time() - t0:.0f} с")
    log(f"teacher: → {out_file}")


# ---------- сборка ----------


def _read_zsc(raw: Path) -> list[Sample]:
    root = raw / "zsc_cigarette" / "data"
    out = []
    for split, sub in (("train", "train"), ("val", "valid"), ("test", "test")):
        for path in sorted((root / sub / "images").glob("*.jpg")):
            label = root / sub / "labels" / (path.stem + ".txt")
            s = Sample("zsc_cigarette", split, f"zsc:{path.name}", path)
            if label.exists():
                for line in label.read_text().splitlines():
                    parts = line.split()
                    if len(parts) == 5 and int(float(parts[0])) == 0:
                        s.boxes.append((0, *map(float, parts[1:])))
                    elif parts:
                        s.dropped[f"zsc:{parts[0]}:{len(parts)}"] += 1
            out.append(s)
    return out


def cigarette_shaped(box: list[float], width: int, height: int) -> bool:
    """Вытянутая рамка: длинная сторона ≥ CIG_ASPECT × короткой (в пикселях).

    Учителя с HF часто ставят «сигарету» на всё лицо курящего (так размечены их наборы Roboflow):
    рамка почти квадратная. Такие рамки — не сигарета, по ним кадр не размечаем.
    """
    bw, bh = box[2] * width, box[3] * height
    return min(bw, bh) > 0 and max(bw, bh) / min(bw, bh) >= CIG_ASPECT


def load_sources(out: Path, teacher: dict[str, dict], forklift: Path, seed: int) -> list[Sample]:
    from PIL import Image

    raw = out / "raw"
    samples = _read_zsc(raw)
    series = {f"mixkit_{vid}": ser for vid, _, ser in TRAIN_VIDEOS}
    for path in sorted((raw / "mixkit_frames").rglob("*.jpg")):
        pred = teacher.get(str(path))
        if not pred or len(pred) < len(TEACHERS):
            continue
        with Image.open(path) as im:
            width, height = im.size
        # Кадр берём, только если все уверенные рамки обоих учителей вытянуты: квадратная «сигарета»
        # на лице означает, что учитель мог пропустить настоящую сигарету внутри неё.
        confident = [b for name in TEACHERS for b in pred[name] if b[4] >= TEACHER_CONF]
        if not confident or not all(cigarette_shaped(b, width, height) for b in confident):
            continue
        boxes = fuse_teachers([pred[name] for name in TEACHERS])
        if not boxes:
            continue
        s = Sample("mixkit_smoking", "train", f"mixkit:{series.get(path.parent.name, path.parent.name)}", path)
        s.boxes += [(0, *b[:4]) for b in boxes]
        s.teacher = len(boxes)
        samples.append(s)
    for split, files in _workplace_candidates(forklift, seed).items():
        kept = 0
        for path in files:
            pred = teacher.get(str(path))
            if not pred or len(pred) < len(TEACHERS):
                continue
            with Image.open(path) as im:
                width, height = im.size
            # Лица, на которых учителя ошибаются, — ценные трудные негативы. Выбрасываем только кадр
            # с уверенной вытянутой рамкой: там сигарета возможна.
            if any(b[4] >= NEG_REJECT_CONF and cigarette_shaped(b, width, height)
                   for name in TEACHERS for b in pred[name]):
                continue
            samples.append(Sample("workplace_neg", split, f"neg:{path.name}", path))
            kept += 1
            if kept >= NEG_PER_SPLIT[split]:
                break
    return samples


def _popcount(x: np.ndarray) -> np.ndarray:
    table = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
    return table[x.view(np.uint8).reshape(-1, 8)].sum(axis=1)


def dedup(samples: list[Sample], holdout: np.ndarray) -> tuple[list[Sample], dict]:
    for s in samples:
        s.dhash = dhash(s.image)
    order = sorted(range(len(samples)), key=lambda i: (SPLIT_RANK[samples[i].split], samples[i].source,
                                                      samples[i].image.name))
    hashes = np.array([samples[i].dhash for i in order], dtype=np.uint64)
    kept_hashes = np.zeros(len(order), dtype=np.uint64)
    n_kept = 0
    keep = np.zeros(len(order), dtype=bool)
    dropped_dup: Counter = Counter()
    dropped_holdout: Counter = Counter()
    for k in range(len(order)):
        s = samples[order[k]]
        if len(holdout) and _popcount(holdout ^ hashes[k]).min() <= DUP_HAMMING:
            dropped_holdout[s.source] += 1
            continue
        if n_kept and _popcount(kept_hashes[:n_kept] ^ hashes[k]).min() <= DUP_HAMMING:
            dropped_dup[f"{s.source}/{s.split}"] += 1
            continue
        keep[k] = True
        kept_hashes[n_kept] = hashes[k]
        n_kept += 1
    kept = [samples[order[k]] for k in range(len(order)) if keep[k]]
    return kept, {"hamming_max": DUP_HAMMING, "dropped_duplicates": dict(dropped_dup),
                  "dropped_holdout_match": dict(dropped_holdout), "holdout_frames": int(len(holdout))}


def cmd_build(args: argparse.Namespace) -> None:
    out: Path = args.out
    teacher_file = out / "raw" / "teacher_cigarette.json"
    teacher = json.loads(teacher_file.read_text())["images"] if teacher_file.exists() else {}
    samples = load_sources(out, teacher, args.forklift, args.seed)
    before = Counter(f"{s.source}/{s.split}" for s in samples)
    holdout = np.concatenate([
        holdout_hashes(ROOT / "ml/eval/scenarios.yaml", args.videos),
        holdout_hashes(ROOT / "ml/eval/scenarios_smoking.yaml", out / "eval_videos"),
    ])
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
        st["negative_images"] += int(not s.boxes)
        st["cigarette"] += len(s.boxes)
        st["cigarette_teacher"] += s.teacher
        for k, v in s.dropped.items():
            stats["dropped_rows"][s.source][k] += v
        lists[s.split].append(name)
        if s.split == "test":
            per_source_test[s.source].append(str(dst))

    for sid, paths in per_source_test.items():
        (out / f"test_{sid}.txt").write_text("\n".join(sorted(paths)) + "\n")
    data_yaml = out / "data.yaml"
    data_yaml.write_text(
        f"# {VERSION}: собран ml/datasets/smoking.py, описание — docs/DATASETS.md\n"
        f"path: {out}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        f"names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASSES)))
    manifest = {
        "version": VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "classes": list(CLASSES),
        "data_yaml": str(data_yaml),
        "sources": SOURCES,
        "videos": {
            "train": [{"id": v, "series": ser, "page": MIXKIT_PAGE.format(slug=slug, vid=v),
                       "url": MIXKIT_MP4.format(vid=v)} for v, slug, ser in TRAIN_VIDEOS],
            "eval_only": [{"id": v, "kind": kind, "page": MIXKIT_PAGE.format(slug=slug, vid=v),
                           "url": MIXKIT_MP4.format(vid=v)} for v, slug, kind in EVAL_VIDEOS],
        },
        "licenses": "не проверяли: решение человека 24.10, демо внутреннее",
        "teacher": {"models": TEACHERS, "imgsz": TEACHER_IMGSZ, "conf": TEACHER_CONF,
                    "solo_conf": TEACHER_SOLO_CONF, "agree_iou": TEACHER_IOU,
                    "negative_reject_conf": NEG_REJECT_CONF, "images": len(teacher)},
        "split_rule": {"zsc_cigarette": "исходные train/valid/test",
                       "mixkit_smoking": "только train (рамки от учителя)",
                       "workplace_neg": "часть кадра в наборе погрузчика"},
        "before_dedup": dict(sorted(before.items())),
        "dedup": dedup_stats,
        "splits": {split: {sid: dict(c) for sid, c in sorted(stats[split].items())} for split in SPLITS},
        "dropped_rows": {sid: dict(c) for sid, c in sorted(stats["dropped_rows"].items())},
        "totals": {split: {"images": len(lists[split]), "sha256_files": file_list_sha(lists[split])}
                   for split in SPLITS},
        "test_subsets": {sid: str(out / f"test_{sid}.txt") for sid in sorted(per_source_test)},
        "holdout_videos": "ml/eval/scenarios.yaml и ml/eval/scenarios_smoking.yaml: кадры сверены по dHash "
                          "и в набор не входят",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    for split in SPLITS:
        tot = Counter()
        for c in stats[split].values():
            tot.update(c)
        log(f"build {split}: кадров {tot['images']}, сигарет {tot['cigarette']} "
            f"(учитель {tot['cigarette_teacher']}), негативов {tot['negative_images']}")
    log(f"build: дубли {dedup_stats['dropped_duplicates']}, совпадения с роликами оценки "
        f"{dedup_stats['dropped_holdout_match']} → {out / 'manifest.json'}")


# ---------- проверка глазами ----------


def cmd_sheet(args: argparse.Namespace) -> None:
    """Лист кадров с рамками: для выборочной проверки разметки (особенно авторазметки учителя)."""
    import cv2

    img_dir = args.out / "images" / args.split
    files = sorted(p for p in img_dir.glob(f"{args.source}*.jpg"))
    random.Random(args.seed).shuffle(files)
    tiles = []
    for p in files[: args.n]:
        im = cv2.imread(str(p))
        h, w = im.shape[:2]
        label = args.out / "labels" / args.split / (p.stem + ".txt")
        for row in (label.read_text().splitlines() if label.exists() else []):
            _, cx, cy, bw, bh = map(float, row.split()[:5])
            cv2.rectangle(im, (int((cx - bw / 2) * w), int((cy - bh / 2) * h)),
                          (int((cx + bw / 2) * w), int((cy + bh / 2) * h)), (0, 0, 255), max(2, w // 300))
        s = args.tile / max(h, w)
        im = cv2.resize(im, (int(w * s), int(h * s)))
        canvas = np.zeros((args.tile, args.tile, 3), np.uint8)
        canvas[: im.shape[0], : im.shape[1]] = im
        tiles.append(canvas)
    while len(tiles) % args.cols:
        tiles.append(np.zeros((args.tile, args.tile, 3), np.uint8))
    rows = [np.hstack(tiles[i:i + args.cols]) for i in range(0, len(tiles), args.cols)]
    args.sheet_out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.sheet_out), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 75])
    log(f"sheet: {args.sheet_out}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["download", "frames", "teacher", "build", "sheet"])
    ap.add_argument("--out", type=Path, default=ROOT / "data/datasets/smoking")
    ap.add_argument("--videos", type=Path, default=ROOT / "data/videos", help="ролики основной оценки")
    ap.add_argument("--forklift", type=Path, default=ROOT / "data/datasets/forklift",
                    help="собранный набор погрузчика: источник негативов")
    ap.add_argument("--only", default="videos,zsc_cigarette", help="что скачивать, через запятую")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--split", default="train")
    ap.add_argument("--source", default="", help="префикс имени кадра для листа, например mixkit_smoking")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--cols", type=int, default=6)
    ap.add_argument("--tile", type=int, default=200)
    ap.add_argument("--sheet-out", type=Path, default=Path("/tmp/smoking_sheet.jpg"))
    args = ap.parse_args(argv)
    args.out = args.out.resolve()
    {"download": cmd_download, "frames": cmd_frames, "teacher": cmd_teacher,
     "build": cmd_build, "sheet": cmd_sheet}[args.cmd](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
