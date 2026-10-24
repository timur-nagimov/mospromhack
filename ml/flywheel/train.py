"""Звено 6 маховика, шаг 1: дообучение (ultralytics, MPS) и карточка-кандидат.

Вход:
- `--data` — data.yaml набора (открытый набор, data/datasets/<имя>/data.yaml) или
  `--dataset` — версия датасета маховика (`fw-YYYYMMDD-HHMM`, путь берётся из data/flywheel/flywheel.db);
- `--mix` — ещё наборы с теми же классами: их train и val добавляются к основному, test не трогается;
- `--base` — текущие веса: путь к .pt или версия из ml/model_cards (берутся её `weights`).

Выход:
- веса `models/weights/<version>.pt`;
- карточка `ml/model_cards/<version>.json` со `status: candidate`. Метрики на отложенных данных и решение
  пишет `ml/flywheel/gate.py`.

Обучение — только через очередь: `HEAVY_LANE=train tools/heavy.sh ...`. Лимиты памяти (tools/memwatch.sh убивает
процесс больше 6 ГБ): batch ≤ 16 для n-моделей и ≤ 8 для s, workers ≤ 4, cache=False. Время запуска
ограничено `--time-min` (по умолчанию 35 мин): ultralytics сам подстраивает число эпох и расписание LR.

Пример — первый оборот маховика на открытых данных (погрузчик, S4):
    HEAVY_LANE=train tools/heavy.sh .venv/bin/python ml/flywheel/train.py \\
        --data data/datasets/forklift/data.yaml --base models/weights/yolov8n.pt \\
        --model-id forklift --version forklift-v1 --parent yolov8n-coco --imgsz 480 --time-min 35
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.flywheel.cards import load_card, now_iso, save_card  # noqa: E402

MODELS_DIR = Path(os.environ.get("GUARD_MODELS_DIR", ROOT / "models"))
FLYWHEEL_DIR = Path(os.environ.get("GUARD_FLYWHEEL_DIR", ROOT / "data" / "flywheel"))
RUNS_DIR = ROOT / "data" / "flywheel" / "runs"
# Лимиты памяти по масштабу модели: memwatch убивает процесс проекта с RSS больше 6 ГБ.
MAX_BATCH = {"n": 16, "s": 8, "m": 4, "l": 2, "x": 2}
MAX_WORKERS = 4
MAX_TIME_MIN = 40


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--data", type=Path, help="data.yaml набора")
    src.add_argument("--dataset", help="версия датасета маховика fw-YYYYMMDD-HHMM или каталог с data.yaml")
    ap.add_argument("--mix", type=Path, action="append", default=[], help="ещё data.yaml с теми же классами")
    ap.add_argument("--base", required=True, help="веса .pt или версия модели из ml/model_cards")
    ap.add_argument("--model-id", required=True, help="model_id в configs/models.yaml")
    ap.add_argument("--version", required=True, help="версия, например forklift-v1")
    ap.add_argument("--parent", help="версия базы для карточки (по умолчанию — из --base)")
    ap.add_argument("--imgsz", type=int, default=480)
    ap.add_argument("--epochs", type=int, default=60, help="верхняя граница; реально ограничивает --time-min")
    ap.add_argument("--time-min", type=float, default=35.0)
    ap.add_argument("--patience", type=int, default=10, help="ранняя остановка по val")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hsv-v", type=float, default=0.5, help="разброс яркости: тёмные проёмы и ночные кадры")
    ap.add_argument("--mixup", type=float, default=0.0, help="доля mixup: устойчивость к перекрытиям")
    ap.add_argument("--lr0", type=float, default=0.01,
                    help="начальный LR; для дообучения от своей же версии — 0.001–0.002 (иначе модель «забывает»)")
    ap.add_argument("--warmup-epochs", type=float, default=3.0)
    ap.add_argument("--freeze", type=int, default=0, help="заморозить первые N слоёв базы")
    ap.add_argument("--note", default="", help="свободный комментарий в карточку")
    ap.add_argument("--dry-run", action="store_true", help="проверить входы и напечатать параметры без обучения")
    return ap


def model_scale(weights: Path) -> str:
    """Масштаб YOLOv8 (n/s/m/l/x) по имени файла или по числу параметров."""
    stem = weights.stem.lower()
    for scale in "nsmlx":
        if stem.startswith(("yolov8" + scale, "yolo11" + scale)):
            return scale
    try:
        import torch

        ckpt = torch.load(str(weights), map_location="cpu", weights_only=False)
        model = ckpt.get("ema") or ckpt.get("model")
        params = sum(p.numel() for p in model.parameters())
    except Exception:  # noqa: BLE001 — не смогли прочитать: считаем самой строгой
        return "x"
    return "n" if params < 5e6 else "s" if params < 15e6 else "m" if params < 30e6 else "l"


def check_limits(args: argparse.Namespace, scale: str) -> list[str]:
    """Ошибки параметров памяти и времени; пустой список — можно запускать."""
    errors = []
    if args.batch > MAX_BATCH.get(scale, 2):
        errors.append(f"batch {args.batch} > {MAX_BATCH.get(scale, 2)} для {scale}-модели (memwatch 6 ГБ)")
    if args.workers > MAX_WORKERS:
        errors.append(f"workers {args.workers} > {MAX_WORKERS}")
    if not 0 < args.time_min <= MAX_TIME_MIN:
        errors.append(f"time-min {args.time_min} вне (0, {MAX_TIME_MIN}]")
    return errors


def resolve_base(base: str) -> tuple[Path, str]:
    """Путь к весам и версия базы."""
    p = Path(base)
    if p.suffix == ".pt":
        p = p if p.is_absolute() else ROOT / p
        if not p.exists():
            raise FileNotFoundError(f"нет весов {p}")
        return p, p.stem
    card = load_card(base)
    weights = Path(card["weights"])
    weights = weights if weights.is_absolute() else ROOT / weights
    if not weights.exists():
        raise FileNotFoundError(f"нет весов {weights} из карточки {base}")
    return weights, card["version"]


def resolve_dataset(dataset: str, flywheel_dir: Path = FLYWHEEL_DIR) -> Path:
    """Версия датасета маховика → data.yaml. Путь из таблицы datasets (docs/FLYWHEEL.md)."""
    p = Path(dataset)
    if p.exists():
        return p / "data.yaml" if p.is_dir() else p
    db = flywheel_dir / "flywheel.db"
    if not db.exists():
        raise FileNotFoundError(f"нет {db}: версия {dataset} не найдена")
    with sqlite3.connect(db) as con:
        row = con.execute("SELECT path FROM datasets WHERE version = ?", (dataset,)).fetchone()
    if row is None:
        raise ValueError(f"версии {dataset} нет в {db}")
    path = Path(row[0])
    path = path if path.is_absolute() else flywheel_dir / path
    return path / "data.yaml" if path.is_dir() else path


def _names(meta: dict[str, Any]) -> list[str]:
    names = meta["names"]
    return [names[i] for i in sorted(names)] if isinstance(names, dict) else list(names)


def _split_dirs(yaml_path: Path, meta: dict[str, Any], key: str) -> list[str]:
    base = Path(meta.get("path") or yaml_path.parent)
    base = base if base.is_absolute() else (yaml_path.parent / base).resolve()
    value = meta.get(key)
    if value is None:
        return []
    items = value if isinstance(value, list) else [value]
    return [str(Path(v) if Path(v).is_absolute() else base / v) for v in items]


POSE_KEYS = ("kpt_shape", "flip_idx")


def data_task(data_yaml: Path) -> str:
    """Задача по data.yaml: pose, если задан kpt_shape (набор ключевых точек), иначе detect."""
    import yaml

    meta = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    return "pose" if meta.get("kpt_shape") else "detect"


def mix_data(primary: Path, mixes: list[Path], out: Path) -> tuple[Path, list[str]]:
    """data.yaml смеси: train и val — всех наборов, test — только основного. Классы обязаны совпадать.

    Для pose-наборов kpt_shape и flip_idx берутся из основного; у смешиваемых kpt_shape должен совпадать.
    """
    import yaml

    meta = yaml.safe_load(primary.read_text(encoding="utf-8"))
    names = _names(meta)
    train, val = _split_dirs(primary, meta, "train"), _split_dirs(primary, meta, "val")
    for extra in mixes:
        m = yaml.safe_load(extra.read_text(encoding="utf-8"))
        if _names(m) != names:
            raise ValueError(f"{extra}: классы {_names(m)} не совпадают с {names} из {primary}")
        if m.get("kpt_shape") != meta.get("kpt_shape"):
            raise ValueError(f"{extra}: kpt_shape {m.get('kpt_shape')} не совпадает с {meta.get('kpt_shape')}")
        train += _split_dirs(extra, m, "train")
        val += _split_dirs(extra, m, "val")
    mixed = {"train": train, "val": val, "test": _split_dirs(primary, meta, "test"),
             "names": dict(enumerate(names))}
    mixed.update({key: meta[key] for key in POSE_KEYS if meta.get(key) is not None})
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(mixed, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return out, names


def count_images(dirs: list[str]) -> int:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    total = 0
    for d in dirs:
        p = Path(d)
        if p.is_dir():
            total += sum(1 for f in p.rglob("*") if f.suffix.lower() in exts)
        elif p.suffix == ".txt" and p.exists():
            total += sum(1 for line in p.read_text().splitlines() if line.strip())
    return total


def dataset_entries(data_yaml: Path, mixes: list[Path]) -> list[dict[str, Any]]:
    """Состав обучения для карточки: из manifest.json рядом с data.yaml, иначе — сам data.yaml."""
    import yaml

    out = []
    for path in [data_yaml, *mixes]:
        manifest = path.parent / "manifest.json"
        meta = yaml.safe_load(path.read_text(encoding="utf-8"))
        train_dirs = _split_dirs(path, meta, "train")
        if manifest.exists():
            m = json.loads(manifest.read_text(encoding="utf-8"))
            train = m.get("splits", {}).get("train", {})
            for sid, src in m.get("sources", {}).items():
                if sid in train:
                    out.append({"name": sid, "images": train[sid].get("images", 0), "url": src.get("url", ""),
                                "dataset": m.get("version", path.parent.name)})
        else:
            out.append({"name": path.parent.name, "images": count_images(train_dirs), "url": str(path)})
    return out


def read_results(run_dir: Path) -> dict[str, Any]:
    """Итог обучения из results.csv ultralytics: эпохи, лучшая эпоха по mAP50 val."""
    path = run_dir / "results.csv"
    if not path.exists():
        return {}
    with path.open() as f:
        rows = [{k.strip(): v for k, v in row.items()} for row in csv.DictReader(f)]
    if not rows:
        return {}
    key50, key5095 = "metrics/mAP50(B)", "metrics/mAP50-95(B)"
    best = max(rows, key=lambda r: float(r.get(key50, 0) or 0))
    return {"epochs_done": len(rows), "best_epoch": int(float(best["epoch"])),
            "val_map50": round(float(best[key50]), 4), "val_map50_95": round(float(best[key5095]), 4)}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    weights, base_version = resolve_base(args.base)
    data_arg = args.data if args.data else resolve_dataset(args.dataset)
    data_yaml = (data_arg if data_arg.is_absolute() else ROOT / data_arg).resolve()
    run_dir = RUNS_DIR / args.version
    mixed, names = mix_data(data_yaml, [m.resolve() for m in args.mix], run_dir / "data_mix.yaml")
    scale = model_scale(weights)
    errors = check_limits(args, scale)
    if errors:
        print("train: " + "; ".join(errors), file=sys.stderr)
        return 2
    train_args = dict(data=str(mixed), imgsz=args.imgsz, epochs=args.epochs, time=args.time_min / 60.0,
                      patience=args.patience, batch=args.batch, workers=args.workers, device=args.device,
                      cache=False, seed=args.seed, hsv_v=args.hsv_v, mixup=args.mixup, freeze=args.freeze or None,
                      lr0=args.lr0, warmup_epochs=args.warmup_epochs,
                      project=str(run_dir), name="train", exist_ok=True, plots=False, verbose=False)
    print(f"train: база {weights.name} ({scale}), классы {names}, данные {mixed}")
    print("train: " + json.dumps({k: v for k, v in train_args.items() if k != "data"}, ensure_ascii=False))
    if args.dry_run:
        return 0

    from ultralytics import YOLO

    t0 = time.time()
    model = YOLO(str(weights))
    model.train(**train_args)
    minutes = round((time.time() - t0) / 60.0, 1)
    best = run_dir / "train" / "weights" / "best.pt"
    dst = MODELS_DIR / "weights" / f"{args.version}.pt"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, dst)
    card = {
        "model_id": args.model_id,
        "version": args.version,
        "parent": args.parent or base_version,
        "created_at": now_iso(),
        "task": data_task(data_yaml),
        "classes": names,
        "train": {
            "datasets": dataset_entries(data_yaml, [m.resolve() for m in args.mix]),
            # как передан в CLI: data/ — симлинк на общее хранилище, resolve дал бы путь вне репозитория
            "data_yaml": str(data_arg),
            "epochs": args.epochs, "imgsz": args.imgsz, "device": args.device, "minutes": minutes,
            "batch": args.batch, "workers": args.workers, "patience": args.patience, "time_min": args.time_min,
            "hsv_v": args.hsv_v, "mixup": args.mixup, "lr0": args.lr0, "warmup_epochs": args.warmup_epochs,
            "seed": args.seed, "base_weights": weights.name,
            **read_results(run_dir / "train"),
        },
        "weights": f"models/weights/{args.version}.pt",
        "onnx": None,
        "tpu_program": None,
        "holdout": {},
        "baseline": {},
        "tpu_check": None,
        "gate": None,
        "status": "candidate",
    }
    if args.note:
        card["note"] = args.note
    path = save_card(card)
    print(f"train: {minutes} мин, веса {dst}, карточка {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
