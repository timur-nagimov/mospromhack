"""Версия датасета маховика в YOLO-формате (docs/FLYWHEEL.md, звено 5).

    python -m ml.flywheel.export [--include-holdout] [--classes person,forklift] [--val-ratio 0.2]
                                 [--mix forklift-rf=data/datasets/forklift-rf] [--flywheel-dir DIR]

- в датасет идут только сэмплы со статусом accepted: метки проверил человек;
- кадры отложенных роликов (holdout = 1) пропускаются. С --include-holdout они входят, а манифест помечает,
  что оценка на этих роликах загрязнена (holdout_contaminated);
- hard negatives — принятые кадры без рамок: пустой файл меток, для YOLO это фон. Так в обучение уходят
  ложные срабатывания, которые оператор отметил «ложное», а человек принял без рамок;
- разбиение по камере и времени: кадры камеры группируются окнами по 10 минут, окно целиком уходит в train
  или в val (детерминированно, по хэшу). Почти одинаковые соседние кадры не попадают по разные стороны;
- изображения — жёсткие ссылки на кадры хранилища (копия, если ссылка невозможна);
- манифест: классы, состав по источникам и камерам, sha256 списков файлов, открытые наборы для смеси (--mix);
- версия fw-YYYYMMDD-HHMM регистрируется в таблице datasets, у сэмплов ставится dataset_version.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from guard.flywheel.store import FlywheelStore
from guard.rules.geometry import LOCAL_TZ

GROUP_S = 600.0  # окно группировки кадров одной камеры, с
DEFAULT_VAL_RATIO = 0.2


class ExportError(ValueError):
    """Экспорт невозможен (нет принятых сэмплов, неверные классы)."""


def default_flywheel_dir() -> Path:
    env = os.environ.get("GUARD_FLYWHEEL_DIR")
    if env:
        return Path(env).expanduser()
    data = os.environ.get("GUARD_DATA_DIR")
    root = Path(data).expanduser() if data else Path(__file__).resolve().parents[2] / "data"
    return root / "flywheel"


def make_version(store: FlywheelStore, out_root: Path, now: float) -> str:
    base = "fw-" + datetime.fromtimestamp(now, LOCAL_TZ).strftime("%Y%m%d-%H%M")
    version, n = base, 1
    while store.get_dataset(version) is not None or (out_root / version).exists():
        n += 1
        version = f"{base}-{n}"
    return version


def _group_key(sample: dict[str, Any]) -> str:
    return f"{sample['camera_id']}|{int(sample['created_at'] // GROUP_S)}"


def _in_val(key: str, val_ratio: float) -> bool:
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") / 2**32 < val_ratio


def split_samples(samples: list[dict[str, Any]], val_ratio: float) -> tuple[list[dict], list[dict], str]:
    """Разбиение по камере и времени: окно GROUP_S одной камеры целиком в train или в val."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for s in samples:
        groups[_group_key(s)].append(s)
    if len(groups) >= 2:
        val_keys = {k for k in groups if _in_val(k, val_ratio)}
        if not val_keys and val_ratio > 0:  # хэш не дал ни одного окна в val — берём самое позднее
            val_keys = {max(groups, key=lambda k: max(s["created_at"] for s in groups[k]))}
        if len(val_keys) == len(groups):  # и наоборот: хотя бы одно окно остаётся в train
            val_keys.discard(min(groups, key=lambda k: min(s["created_at"] for s in groups[k])))
        train = [s for k, g in groups.items() if k not in val_keys for s in g]
        val = [s for k, g in groups.items() if k in val_keys for s in g]
        return train, val, f"camera_time_{int(GROUP_S)}s"
    # одно окно: делим по времени внутри него (последние кадры — в val)
    ordered = sorted(samples, key=lambda s: s["created_at"])
    n_val = int(round(len(ordered) * val_ratio)) if len(ordered) >= 5 else 0
    cut = len(ordered) - n_val
    return ordered[:cut], ordered[cut:], "time_within_single_window"


def yolo_lines(sample: dict[str, Any], class_index: dict[str, int]) -> tuple[list[str], int]:
    """Строки YOLO (cls cx cy w h, нормировано) и число отброшенных меток чужих классов."""
    w, h = float(sample["width"]), float(sample["height"])
    lines, skipped = [], 0
    for label in sample.get("labels") or []:
        idx = class_index.get(label["cls"])
        if idx is None:
            skipped += 1
            continue
        x1, y1, x2, y2 = label["bbox"]
        x1, x2 = max(0.0, min(w, x1)), max(0.0, min(w, x2))
        y1, y2 = max(0.0, min(h, y1)), max(0.0, min(h, y2))
        if x2 - x1 < 1 or y2 - y1 < 1:
            skipped += 1
            continue
        cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
        lines.append(f"{idx} {cx:.6f} {cy:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
    return lines, skipped


def _link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _sha256_list(paths: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(paths)).encode("utf-8")).hexdigest()


def parse_mix(items: list[str] | None) -> list[dict[str, Any]]:
    """--mix имя=путь: открытые наборы (data/datasets/<имя>) для смеси при обучении — только в манифест."""
    out = []
    for item in items or []:
        name, _, path = item.partition("=")
        if not name or not path:
            raise ExportError(f"--mix expects name=path, got {item!r}")
        entry: dict[str, Any] = {"name": name, "path": path}
        manifest = Path(path) / "manifest.json"
        if manifest.is_file():
            try:
                meta = json.loads(manifest.read_text(encoding="utf-8"))
                entry.update({k: meta[k] for k in ("url", "source", "images", "license") if k in meta})
            except (OSError, json.JSONDecodeError):
                pass
        out.append(entry)
    return out


def export_dataset(
    store: FlywheelStore,
    *,
    include_holdout: bool = False,
    classes: list[str] | None = None,
    val_ratio: float = DEFAULT_VAL_RATIO,
    out_root: Path | None = None,
    mix: list[dict[str, Any]] | None = None,
    now: float | None = None,
) -> dict[str, Any]:
    """Собрать версию датасета из принятых сэмплов. Возвращает строку таблицы datasets."""
    now = time.time() if now is None else now
    samples, _ = store.query(status="accepted", holdout=None if include_holdout else False, limit=None,
                             order="created_at")
    if not samples:
        raise ExportError("нет принятых сэмплов для экспорта" + ("" if include_holdout else " (holdout исключён)"))
    if classes:
        names = [c.strip() for c in classes if c and c.strip()]
        if len(set(names)) != len(names):
            raise ExportError("classes must be unique")
    else:
        names = sorted({lb["cls"] for s in samples for lb in s.get("labels") or []})
    class_index = {c: i for i, c in enumerate(names)}
    out_root = Path(out_root) if out_root is not None else store.root / "datasets"
    version = make_version(store, out_root, now)
    root = out_root / version
    train, val, split_method = split_samples(samples, val_ratio)
    files: dict[str, list[str]] = {"train": [], "val": []}
    ids: dict[str, list[str]] = {"train": [], "val": []}
    n_hard_negative = 0
    skipped_labels = 0
    missing_frames: list[str] = []
    labels_by_class: Counter[str] = Counter()
    for part, items in (("train", train), ("val", val)):
        for sample in items:
            src = store.resolve_frame(sample["frame_path"])
            if src is None:
                missing_frames.append(sample["id"])
                continue
            lines, skipped = yolo_lines(sample, class_index)
            skipped_labels += skipped
            if not lines:
                n_hard_negative += 1
            for line in lines:
                labels_by_class[names[int(line.split(" ", 1)[0])]] += 1
            image = root / "images" / part / f"{sample['id']}.jpg"
            _link(src, image)
            label = root / "labels" / part / f"{sample['id']}.txt"
            label.parent.mkdir(parents=True, exist_ok=True)
            label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            files[part].append(f"images/{part}/{sample['id']}.jpg")
            ids[part].append(sample["id"])
    exported_ids = set(ids["train"]) | set(ids["val"])
    exported = [s for s in samples if s["id"] in exported_ids]
    if not exported:
        shutil.rmtree(root, ignore_errors=True)
        raise ExportError("кадры принятых сэмплов не найдены в хранилище")
    (root / "images" / "val").mkdir(parents=True, exist_ok=True)
    (root / "labels" / "val").mkdir(parents=True, exist_ok=True)
    data_yaml = (
        f"# {version}: маховик данных, docs/FLYWHEEL.md\n"
        f"path: {root.resolve()}\ntrain: images/train\nval: images/val\n"
        "names:\n" + "".join(f"  {i}: {json.dumps(c, ensure_ascii=False)}\n" for i, c in enumerate(names))
    )
    (root / "data.yaml").write_text(data_yaml, encoding="utf-8")
    holdout_in = sum(1 for s in exported if s.get("holdout"))
    manifest = {
        "version": version,
        "created_at": datetime.fromtimestamp(now, LOCAL_TZ).isoformat(timespec="seconds"),
        "flywheel_dir": str(store.root),
        "classes": names,
        "include_holdout": bool(include_holdout),
        "holdout_samples": holdout_in,
        "holdout_contaminated": bool(include_holdout and holdout_in),
        "warning": (
            "в датасете есть кадры отложенных роликов: оценка на них загрязнена" if include_holdout and holdout_in else None
        ),
        "split": {"method": split_method, "val_ratio": val_ratio, "group_s": GROUP_S},
        "n_train": len(ids["train"]),
        "n_val": len(ids["val"]),
        "n_hard_negative": n_hard_negative,
        "labels_by_class": dict(labels_by_class),
        "skipped_labels": skipped_labels,
        "missing_frames": missing_frames,
        "by_source": dict(Counter(s["source"] for s in exported)),
        "by_camera": dict(Counter(s["camera_id"] for s in exported)),
        "by_operator_label": dict(Counter(s.get("operator_label") or "none" for s in exported)),
        "sha256": {"train": _sha256_list(files["train"]), "val": _sha256_list(files["val"])},
        "samples": ids,
        "open_datasets": list(mix or []),
        "sources": [{"name": "flywheel", "path": str(store.root), "images": len(exported)}] + list(mix or []),
    }
    (root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    row = store.add_dataset(
        {
            "version": version,
            "created_at": now,
            "path": str(root),
            "classes": names,
            "n_train": len(ids["train"]),
            "n_val": len(ids["val"]),
            "n_hard_negative": n_hard_negative,
            "manifest": manifest,
        }
    )
    store.mark_dataset_version([*ids["train"], *ids["val"]], version)
    return row


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m ml.flywheel.export", description="Версия датасета маховика в YOLO")
    p.add_argument("--flywheel-dir", type=Path, default=None, help="хранилище (GUARD_FLYWHEEL_DIR или data/flywheel)")
    p.add_argument("--include-holdout", action="store_true", help="включить кадры отложенных роликов (оценка загрязнится)")
    p.add_argument("--classes", default=None, help="классы через запятую, порядок = индексы YOLO")
    p.add_argument("--val-ratio", type=float, default=DEFAULT_VAL_RATIO)
    p.add_argument("--out", type=Path, default=None, help="каталог версий (по умолчанию <хранилище>/datasets)")
    p.add_argument("--mix", action="append", default=None, help="открытый набор для смеси: имя=путь (в манифест)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    store = FlywheelStore(args.flywheel_dir or default_flywheel_dir())
    try:
        row = export_dataset(
            store,
            include_holdout=args.include_holdout,
            classes=[c for c in args.classes.split(",")] if args.classes else None,
            val_ratio=args.val_ratio,
            out_root=args.out,
            mix=parse_mix(args.mix),
        )
    except ExportError as exc:
        print(f"export: {exc}", file=sys.stderr)
        return 2
    finally:
        store.close()
    m = row["manifest"]
    print(
        f"{row['version']}: train {row['n_train']}, val {row['n_val']}, hard negatives {row['n_hard_negative']}, "
        f"classes {row['classes']} → {row['path']}"
    )
    if m.get("warning"):
        print(f"ВНИМАНИЕ: {m['warning']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
