"""Звено 6 маховика, шаг 5: гейт. Сравнивает кандидата с базой на отложенных данных и пишет решение в карточку.

Отложенные данные (docs/FLYWHEEL.md):
- тестовая часть набора (`--data`, split test) — mAP50 по классам, CPU, onnxruntime, тот же декодер, что в конвейере;
- ролики `ml/eval/scenarios.yaml` — события, офлайн-прогон всего конвейера (`ml/eval/run_offline.py`).

Правило гейта (контракт): кандидат выкатывается, если на отложенных событиях одновременно
- recall не ниже;
- precision не ниже;
- ложных тревог в час не больше;
- mAP50 на тесте набора ниже не больше чем на 0.01 (если у базы нет классов набора — критерий не применим).
Сверх контракта: дублей не больше — «одно нарушение — одно оповещение» главное требование кейса.

События можно дать готовыми отчётами run_offline (`--events-candidate/--events-baseline`) или посчитать здесь
(`--scenarios-candidate/--scenarios-baseline`, это минуты CPU — запускать через tools/heavy.sh).

Пример (погрузчик):
    python ml/flywheel/gate.py --candidate forklift-v1 --onnx models/forklift_v1-480.onnx \\
        --data data/datasets/forklift/data.yaml --subset forklift_rf=data/datasets/forklift/test_forklift_rf.txt \\
        --reference keremberke=models/forklift-640.onnx:forklift,person \\
        --events-baseline data/flywheel/runs/eval-forklift-base.json \\
        --events-candidate data/flywheel/runs/eval-forklift-v1.json --baseline-name "det (yolov8n COCO)" --write
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.flywheel.cards import load_card, now_iso, save_card  # noqa: E402

MAP_TOL = 0.01
EPS = 1e-9
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidate", required=True, help="версия кандидата или путь к его карточке")
    ap.add_argument("--onnx", type=Path, help="ONNX кандидата (по умолчанию — поле onnx карточки)")
    ap.add_argument("--data", type=Path, help="data.yaml набора: mAP50 на split")
    ap.add_argument("--split", default="test")
    ap.add_argument("--subset", action="append", default=[],
                    help="имя=файл со списком кадров теста: отдельный mAP50, например forklift_rf=...txt")
    ap.add_argument("--baseline-name", default="", help="как назвать базу в карточке")
    ap.add_argument("--baseline-onnx", help="ONNX базы:классы через запятую — для mAP50 базы на том же тесте")
    ap.add_argument("--reference", action="append", default=[],
                    help="имя=onnx:классы — справочная модель (mAP50 в карточку, на решение не влияет)")
    ap.add_argument("--events-candidate", type=Path, help="отчёт run_offline кандидата (JSON)")
    ap.add_argument("--events-baseline", type=Path, help="отчёт run_offline базы (JSON)")
    ap.add_argument("--scenarios-candidate", type=Path, help="посчитать события кандидата по этому scenarios")
    ap.add_argument("--scenarios-baseline", type=Path, help="посчитать события базы по этому scenarios")
    ap.add_argument("--only", help="ролики через запятую (для --scenarios-*)")
    ap.add_argument("--models-config", type=Path, default=ROOT / "configs/models.yaml",
                    help="реестр, в копии которого model_id кандидата смотрит на его ONNX (для --scenarios-candidate)")
    ap.add_argument("--events-out", type=Path, default=ROOT / "data/flywheel/runs",
                    help="куда сохранить посчитанные отчёты событий")
    ap.add_argument("--cache-dir", type=Path,
                    help="кэш детекций run_offline (ml/eval/replay_cache.py): det, pose, ppe и fire базы и кандидата "
                         "считаются один раз. Ключ кэша включает файл ONNX и запись модели, версии не смешиваются")
    ap.add_argument("--imgsz", type=int, help="вход ONNX (по умолчанию — из модели)")
    ap.add_argument("--map-tol", type=float, default=MAP_TOL)
    ap.add_argument("--write", action="store_true", help="записать holdout, baseline и gate в карточку")
    return ap


# ---------- mAP50 на тесте набора ----------


def _yaml(path: Path) -> dict[str, Any]:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def dataset_images(data_yaml: Path, split: str) -> tuple[list[Path], list[str]]:
    meta = _yaml(data_yaml)
    names = meta["names"]
    names = [names[i] for i in sorted(names)] if isinstance(names, dict) else list(names)
    base = Path(meta.get("path") or data_yaml.parent)
    base = base if base.is_absolute() else (data_yaml.parent / base)
    entries = meta[split] if isinstance(meta[split], list) else [meta[split]]
    images: list[Path] = []
    for entry in entries:
        p = Path(entry) if Path(entry).is_absolute() else base / entry
        if p.is_dir():
            images += sorted(f for f in p.rglob("*") if f.suffix.lower() in IMAGE_EXTS)
        elif p.suffix == ".txt":
            images += [Path(line.strip()) for line in p.read_text().splitlines() if line.strip()]
    return images, names


def label_path(image: Path) -> Path:
    """Как у ultralytics: …/images/<split>/x.jpg → …/labels/<split>/x.txt."""
    parts = list(image.parts)
    idx = len(parts) - 1 - parts[::-1].index("images")
    parts[idx] = "labels"
    return Path(*parts).with_suffix(".txt")


def load_gt(images: list[Path], names: list[str]) -> dict[str, dict[str, np.ndarray]]:
    """Рамки разметки в пикселях: {класс: {кадр: (N, 4)}}."""
    import cv2

    gts: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for img in images:
        lbl = label_path(img)
        if not lbl.exists():
            continue
        lines = [line.split() for line in lbl.read_text().splitlines() if line.strip()]
        if not lines:
            continue
        h, w = cv2.imread(str(img)).shape[:2]
        for parts in lines:
            cx, cy, bw, bh = map(float, parts[1:5])
            gts[names[int(parts[0])]][str(img)].append(
                [(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
    return {c: {k: np.array(v) for k, v in d.items()} for c, d in gts.items()}


def output_format_for(channels: int, classes: list[str]) -> str:
    """Формат выхода YOLOv8 по числу каналов: pose — рамка, скор person и 17 точек (56), иначе detect."""
    if len(classes) == 1 and channels == 4 + 1 + 17 * 3:
        return "yolov8-pose"
    return "yolov8"


def predict_onnx(onnx: Path, classes: list[str], images: list[Path], conf: float = 0.001,
                 iou: float = 0.6) -> dict[str, list]:
    """Предсказания ONNX на CPU: {класс: [(кадр, conf, box xyxy)]}. Декодер — ml/yolo_decode (как в сверке ONNX).

    Pose-модель (выход 1×56×N, класс person) декодируется как pose: mAP считается по рамкам людей.
    """
    import cv2
    import onnxruntime as ort

    from guard.contracts import ModelSpec
    from ml.yolo_decode import decode, letterbox

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    sess = ort.InferenceSession(str(onnx), opts, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    size = tuple(int(v) for v in inp.shape[2:4])
    channels = sess.get_outputs()[0].shape[1]
    output_format = output_format_for(channels if isinstance(channels, int) else 0, classes)
    spec = ModelSpec(model_id=onnx.stem, onnx_path=str(onnx), input_size=size, classes=classes,
                     conf_threshold=conf, iou_threshold=iou, output_format=output_format)
    preds: dict[str, list] = defaultdict(list)
    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        x, meta = letterbox(img, size)
        for d in decode(sess.run(None, {inp.name: x}), meta, spec, max_det=300):
            preds[d.cls].append((str(img_path), d.conf, np.array(d.bbox)))
    return preds


def dataset_map(onnx: Path, classes: list[str], images: list[Path], names: list[str],
                gts: dict[str, dict[str, np.ndarray]] | None = None, conf_op: float = 0.35) -> dict[str, Any]:
    """AP50 (COCO, 101 точка) по общим классам модели и набора; mAP50 — среднее по ним."""
    from ml.eval_ppe import ap_for_class

    gts = load_gt(images, names) if gts is None else gts
    preds = predict_onnx(onnx, classes, images)
    per_class = {}
    for c in names:
        if c not in classes:
            continue
        ap, p, r, n = ap_for_class(preds.get(c, []), gts.get(c, {}), 0.5, conf_op)
        if ap is not None:
            per_class[c] = {"ap50": round(ap, 4), "p": round(p, 3), "r": round(r, 3), "n_gt": n}
    aps = [v["ap50"] for v in per_class.values()]
    return {"images": len(images), "map50": round(float(np.mean(aps)), 4) if aps else None,
            "per_class": per_class, "conf_op": conf_op}


def parse_model_arg(value: str) -> tuple[str, Path, list[str]]:
    """«имя=путь.onnx:класс1,класс2» → (имя, путь, классы)."""
    name, rest = value.split("=", 1) if "=" in value else ("", value)
    path, _, classes = rest.rpartition(":") if ":" in rest else (rest, "", "")
    return name, Path(path), [c.strip() for c in classes.split(",") if c.strip()]


def parse_subset(value: str) -> tuple[str, Path]:
    name, path = value.split("=", 1)
    return name, Path(path)


# ---------- события ----------


def event_summary(report: dict[str, Any]) -> dict[str, Any]:
    """Сжатая сводка отчёта run_offline для карточки и гейта."""
    o = report["overall"]
    out = {k: o[k] for k in ("tp", "fp", "fn", "duplicates", "precision", "recall", "false_alarms_per_hour")}
    out["latency_p50_s"] = None if o.get("latency_median_s") is None else round(o["latency_median_s"], 2)
    out["latency_p95_s"] = None if o.get("latency_p95_s") is None else round(o["latency_p95_s"], 2)
    out["videos"] = sorted(Path(v).stem for v in report.get("videos", []))
    out["by_type"] = {t: {k: m[k] for k in ("tp", "fp", "fn", "duplicates")}
                      for t, m in report.get("by_type", {}).items()}
    return out


def candidate_registry(base: Path, model_id: str, onnx: Path, classes: list[str], out: Path) -> Path:
    """Копия реестра, где model_id смотрит на ONNX кандидата: конвейер меряет кандидата без правки configs."""
    import onnx as onnx_lib
    import yaml

    raw = yaml.safe_load(base.read_text(encoding="utf-8"))
    entry = next((m for m in raw["models"] if m.get("model_id") == model_id), None)
    if entry is None:
        entry = {"model_id": model_id, "task": "detect", "output_format": "yolov8",
                 "conf_threshold": 0.3, "iou_threshold": 0.5}
        raw["models"].append(entry)
    dims = [d.dim_value for d in onnx_lib.load(str(onnx)).graph.input[0].type.tensor_type.shape.dim]
    entry.update(onnx_path=str(onnx.resolve()), input_size=[int(dims[2]), int(dims[3])], classes=list(classes))
    entry.pop("tpu_program", None)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(raw, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return out


def rel_path(path: Path) -> str:
    """Путь в карточку: относительно корня репозитория (карточка одна для main и всех worktree)."""
    try:
        return str(Path(path).absolute().relative_to(ROOT))
    except ValueError:
        return str(path)


def code_version() -> str:
    """Коммит кода конвейера: события базы и кандидата сравнимы, только если посчитаны одним кодом."""
    import subprocess

    def git(*cmd: str) -> str:
        return subprocess.run(["git", *cmd], cwd=ROOT, capture_output=True, text=True).stdout.strip()

    head = git("rev-parse", "--short", "HEAD") or "unknown"
    dirty = git("status", "--porcelain", "--untracked-files=no", "--", "guard", "ml/eval")
    return head + ("+dirty" if dirty else "")


def run_events(scenarios: Path, only: str | None, out: Path, models_config: Path | None = None,
               cache_dir: Path | None = None) -> dict[str, Any]:
    """Прогон run_offline. Кэш детекций (CX-022) общий: модели базы не считаются второй раз."""
    from ml.eval import run_offline

    suite = run_offline.load_scenarios(scenarios, models_config=models_config)
    report = run_offline.run_evaluation(suite, only=only.split(",") if only else None, cache_dir=cache_dir)
    report["code"] = code_version()
    report["scenarios"] = str(scenarios)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


# ---------- решение ----------


def decide(candidate: dict[str, Any], baseline: dict[str, Any], map_tol: float = MAP_TOL) -> tuple[bool, list[str]]:
    """Правило гейта. Вход — {"events": сводка, "map50": число или None} кандидата и базы."""
    ce, be = candidate["events"], baseline["events"]
    ok = True
    reasons = []

    def check(passed: bool, text: str) -> None:
        nonlocal ok
        ok = ok and passed
        reasons.append(("" if passed else "НЕ ПРОЙДЕНО: ") + text)

    check(ce["recall"] + EPS >= be["recall"],
          f"recall {ce['recall']:.3f} против {be['recall']:.3f} (TP {ce['tp']} против {be['tp']}, "
          f"FN {ce['fn']} против {be['fn']})")
    # Без единого предсказания precision в events_eval = 0; сравниваем, только если предсказания есть.
    c_pred, b_pred = ce["tp"] + ce["fp"], be["tp"] + be["fp"]
    if c_pred and b_pred:
        check(ce["precision"] + EPS >= be["precision"],
              f"precision {ce['precision']:.3f} против {be['precision']:.3f} (FP {ce['fp']} против {be['fp']})")
    else:
        check(ce["fp"] <= be["fp"], f"precision: предсказаний {c_pred} и {b_pred}, FP {ce['fp']} против {be['fp']}")
    cf, bf = ce.get("false_alarms_per_hour") or 0.0, be.get("false_alarms_per_hour") or 0.0
    check(cf <= bf + EPS, f"ложных тревог в час {cf:.1f} против {bf:.1f}")
    check(ce["duplicates"] <= be["duplicates"], f"дублей {ce['duplicates']} против {be['duplicates']}")
    cm, bm = candidate.get("map50"), baseline.get("map50")
    if cm is None:
        check(False, "mAP50 кандидата на тесте набора не посчитан")
    elif bm is None:
        reasons.append(f"mAP50 на тесте {cm:.3f}; mAP50 базы на этом тесте нет (в базе нет класса набора "
                       "или не задан --baseline-onnx) — критерий mAP не применим")
    else:
        check(cm + map_tol + EPS >= bm, f"mAP50 на тесте {cm:.3f} против {bm:.3f} (допуск −{map_tol})")
    for event_type, cm_type in sorted(ce.get("by_type", {}).items()):
        bm_type = be.get("by_type", {}).get(event_type, {"tp": 0, "fp": 0, "fn": 0})
        if (cm_type["tp"], cm_type["fp"]) != (bm_type["tp"], bm_type["fp"]):
            total = cm_type["tp"] + cm_type["fn"]
            reasons.append(f"{event_type}: {cm_type['tp']}/{total} против {bm_type['tp']}/{total}, "
                           f"FP {cm_type['fp']} против {bm_type['fp']}")
    return ok, reasons


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    card = load_card(args.candidate)
    onnx = args.onnx or (ROOT / "models" / card["onnx"] if card.get("onnx") else None)
    if onnx is None:
        print("gate: нет ONNX кандидата (--onnx или поле onnx карточки)", file=sys.stderr)
        return 2
    onnx = onnx if onnx.is_absolute() else ROOT / onnx

    holdout: dict[str, Any] = {}
    baseline_holdout: dict[str, Any] = {}
    references: dict[str, Any] = {}
    cand_map = base_map = None
    if args.data:
        images, names = dataset_images(args.data, args.split)
        gts = load_gt(images, names)
        res = dataset_map(onnx, card["classes"], images, names, gts)
        holdout["dataset_test"] = {"name": f"{args.data.parent.name} {args.split}", **res}
        cand_map = res["map50"]
        subsets = [parse_subset(s) for s in args.subset]
        for sname, spath in subsets:
            sub_images = [Path(line.strip()) for line in spath.read_text().splitlines() if line.strip()]
            holdout[f"dataset_test_{sname}"] = {"name": f"{sname} {args.split}",
                                                **dataset_map(onnx, card["classes"], sub_images, names)}
        if args.baseline_onnx:
            _, bpath, bclasses = parse_model_arg(args.baseline_onnx)
            bres = dataset_map(bpath if bpath.is_absolute() else ROOT / bpath, bclasses, images, names, gts)
            baseline_holdout["dataset_test"] = {"name": f"{args.data.parent.name} {args.split}", **bres}
            base_map = bres["map50"]
        for ref in args.reference:
            rname, rpath, rclasses = parse_model_arg(ref)
            rpath = rpath if rpath.is_absolute() else ROOT / rpath
            references[rname] = {"onnx": rpath.name,
                                 "dataset_test": dataset_map(rpath, rclasses, images, names, gts)}
            for sname, spath in subsets:
                sub_images = [Path(line.strip()) for line in spath.read_text().splitlines() if line.strip()]
                references[rname][f"dataset_test_{sname}"] = dataset_map(rpath, rclasses, sub_images, names)

    stamp = time.strftime("%Y%m%d-%H%M")
    cand_path, base_path = args.events_candidate, args.events_baseline
    if args.events_candidate:
        cand_report = json.loads(args.events_candidate.read_text(encoding="utf-8"))
    elif args.scenarios_candidate:
        out_dir = args.events_out / card["version"]
        registry = candidate_registry(args.models_config, card["model_id"], onnx, card["classes"],
                                      out_dir / "models_candidate.yaml")
        cand_path = out_dir / f"events-{card['version']}-{stamp}.json"
        cand_report = run_events(args.scenarios_candidate, args.only, cand_path, registry, args.cache_dir)
    else:
        cand_report = None
    if args.events_baseline:
        base_report = json.loads(args.events_baseline.read_text(encoding="utf-8"))
    elif args.scenarios_baseline:
        base_path = args.events_out / f"events-base-{stamp}.json"
        base_report = run_events(args.scenarios_baseline, args.only, base_path, None, args.cache_dir)
    else:
        base_report = None
    if cand_report is None or base_report is None:
        print("gate: нужны события кандидата и базы (--events-* или --scenarios-*)", file=sys.stderr)
        return 2
    holdout["events"] = event_summary(cand_report)
    baseline_holdout["events"] = event_summary(base_report)
    for target, path, report in ((holdout, cand_path, cand_report), (baseline_holdout, base_path, base_report)):
        if path is not None:
            target["events"]["report"] = rel_path(path)
        if report.get("code"):
            target["events"]["code"] = report["code"]
    codes = {holdout["events"].get("code"), baseline_holdout["events"].get("code")}
    if None not in codes and len(codes) > 1:
        # Разный код конвейера — разница в событиях может быть не от модели (phone_usage в гейте forklift-v2).
        print(f"gate: внимание — события посчитаны разным кодом: {sorted(codes)}", file=sys.stderr)
    if holdout["events"]["videos"] != baseline_holdout["events"]["videos"]:
        print("gate: события кандидата и базы посчитаны на разных роликах", file=sys.stderr)
        return 2

    promote, reasons = decide({"events": holdout["events"], "map50": cand_map},
                              {"events": baseline_holdout["events"], "map50": base_map}, args.map_tol)
    gate = {"promote": promote, "reasons": reasons, "at": now_iso(),
            "rule": "recall и precision не ниже, ложных в час и дублей не больше, mAP50 −0.01"}
    print(json.dumps({"holdout": holdout, "baseline": baseline_holdout, "references": references, "gate": gate},
                     ensure_ascii=False, indent=2))
    if args.write:
        # Остальные поля holdout (например, покадровая проверка) сохраняются.
        card["holdout"] = {**(card.get("holdout") or {}), **holdout}
        card["onnx"] = card.get("onnx") or onnx.name
        card["baseline"] = {"version": args.baseline_name or "база", "holdout": baseline_holdout}
        if references:
            card["references"] = references
        card["gate"] = gate
        if not promote:
            card["status"] = "rejected"
        print(f"gate: карточка {save_card(card)}")
    return 0 if promote else 1


if __name__ == "__main__":
    raise SystemExit(main())
