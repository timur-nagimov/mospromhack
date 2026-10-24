"""Звено 6 маховика, шаги 2–4 и 6: экспорт ONNX → компиляция под LinQ TPU → сверка TPU с CPU → реестр.

Шаги (`--steps`, по умолчанию все по порядку; `--dry-run` печатает команды и ничего не меняет):
- `export`   — `ml/export_onnx.py` (torch, через tools/heavy.sh): `models/<версия>-<imgsz>.onnx`, поле `onnx` карточки;
- `calib`    — `models/calib_<имя>/`: кадры с погрузчиком (классом модели) из train набора и `data/calib`.
  Тестовые кадры и ролики оценки в калибровку не идут;
- `compile`  — на стенде (`HEAVY_LANE=stand`): загрузка ONNX, калибровки и `tpu/*.py` в `~/work`,
  `tpu/compile.py` с калибровкой, затем `tpu/fix_thresholds.py` (потолок Sigmoid 1.0, LINQ_NOTES §17) и
  повторная компиляция `<имя>s1`. Программы копируются в `/auto/cache/<логин>/` (новые файлы, чужие не трогаем);
- `check`    — `tpu/yolo_check.py` на стенде, raw: TPU против CPU fp32 на отложенных кадрах, recall и precision
  по порогам TPU. Итог — в `tpu_check` карточки;
- `registry` — `tpu_program` в записи модели `configs/models.yaml` и запись в `configs/models.stand.yaml`
  под якорем модели; карточка → `deployed`, прежняя `deployed` того же model_id → `archived`.

Edge не перезапускается: в конце печатаются команды для оркестратора.

Пример:
    HEAVY_LANE=stand tools/heavy.sh .venv/bin/python ml/flywheel/deploy.py --version forklift-v1 \\
        --data data/datasets/forklift/data.yaml --steps calib,compile,check
    .venv/bin/python ml/flywheel/deploy.py --version forklift-v1 --steps registry --tpu-conf 0.35 --dry-run
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.flywheel.cards import list_cards, load_card, save_card  # noqa: E402

MODELS_DIR = Path(os.environ.get("GUARD_MODELS_DIR", ROOT / "models"))
PYTHON = sys.executable
STEPS = ("export", "calib", "compile", "check", "registry")
STAND_LOGIN = "hackathon10864"
STAND_WORK = "work"                       # ~/work на стенде
STAND_TOOLS = "work/ds_flywheel/tpu"      # свои копии tpu/*.py, guard_repo и mosprom не трогаем
SDK_PY = "~/linq312/bin/python"
TPU_SCRIPTS = ("compile.py", "fix_thresholds.py", "yolo_check.py")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", required=True, help="версия из ml/model_cards")
    ap.add_argument("--steps", default=",".join(STEPS), help=f"через запятую из {','.join(STEPS)}")
    ap.add_argument("--dry-run", action="store_true", help="только напечатать команды и изменения")
    ap.add_argument("--imgsz", type=int, default=480)
    ap.add_argument("--data", type=Path, help="data.yaml набора: кадры для калибровки (train) и сверки (test)")
    ap.add_argument("--calib-n", type=int, default=100, help="кадров калибровки (половина — с классом модели)")
    ap.add_argument("--calib-extra", type=Path, default=ROOT / "data/calib", help="общий калибровочный набор")
    ap.add_argument("--calib-class", default="forklift", help="класс, кадры с которым обязательны в калибровке")
    ap.add_argument("--check-images", type=Path, action="append", default=[],
                    help="каталоги отложенных кадров для сверки (по умолчанию test набора)")
    ap.add_argument("--check-n", type=int, default=60)
    ap.add_argument("--check-conf", type=float, default=0.45, help="порог CPU в сверке (рождение трека)")
    ap.add_argument("--tpu-conf", type=float, help="conf_threshold модели на TPU в models.stand.yaml")
    ap.add_argument("--program", choices=["calib", "s1"], default="s1",
                    help="какую программу выкатывать: с порогами калибровки или с потолком Sigmoid 1.0")
    ap.add_argument("--anchor", help="подстрока строки-якоря в configs/models*.yaml (по умолчанию «<model_id>»)")
    ap.add_argument("--force", action="store_true", help="пересобрать готовое; выкатить без promote в гейте")
    ap.add_argument("--env", type=Path, default=Path(os.environ.get("GUARD_ROOT", "/Users/nagimov-timur/Desktop/mosprom"))
                    / ".env", help=".env с доступом к JupyterHub")
    return ap


# ---------- имена ----------


def onnx_name(version: str, imgsz: int) -> str:
    """forklift-v1, 480 → forklift_v1-480.onnx (как у yolov8n-480.onnx)."""
    return f"{version.replace('-', '_')}-{imgsz}.onnx"


def program_name(version: str, imgsz: int, s1: bool = False) -> str:
    return f"{version.replace('-', '_')}_{imgsz}{'s1' if s1 else ''}"


def program_path(name: str) -> str:
    """Путь программы относительно GUARD_MODELS_DIR, как в configs/models*.yaml."""
    return f"tpu/{name}/{name}_b1_o5_h1m.tpu"


# ---------- исполнение ----------


def run(cmd: list[str], dry: bool) -> int:
    print("+ " + " ".join(shlex.quote(c) for c in cmd), flush=True)
    if dry:
        return 0
    return subprocess.call(cmd, cwd=ROOT)


class Stand:
    """Операции на стенде через tools/jhub.py. В dry-run только печатает."""

    def __init__(self, env: Path, dry: bool) -> None:
        self.dry = dry
        self.jh = None
        if not dry:
            from tools.jhub import JHub

            self.jh = JHub.from_env(env)
            self.jh.ensure()

    def upload(self, local: Path, remote: str) -> None:
        print(f"+ upload {local} → ~/{remote}", flush=True)
        if not self.dry:
            self.jh.upload(local, remote)

    def sh(self, cmd: str, timeout: float = 1800.0) -> str:
        print(f"+ stand$ {cmd}", flush=True)
        if self.dry:
            return ""
        res = self.jh.sh(cmd, timeout=timeout)
        if not res.ok:
            raise RuntimeError(f"стенд: {cmd}\n{res.stdout[-2000:]}\n{res.error}")
        return res.stdout

    def download_text(self, remote: str) -> str:
        print(f"+ download ~/{remote}", flush=True)
        return "" if self.dry else self.jh.read_text(remote)


# ---------- шаги ----------


def step_export(card: dict[str, Any], args: argparse.Namespace) -> None:
    name = onnx_name(card["version"], args.imgsz)
    dst = MODELS_DIR / name
    if dst.exists() and not args.force:
        print(f"export: {dst} уже есть")
    else:
        weights = (ROOT / card["weights"]).resolve()
        cmd = [str(ROOT / "tools/heavy.sh"), PYTHON, "ml/export_onnx.py", "--weights", str(weights),
               "--imgsz", str(args.imgsz), "--out", name, "--model-id", card["version"].replace("-", "_"),
               "--source", f"маховик: {card['version']}, ml/model_cards/{card['version']}.json"]
        if run(cmd, args.dry_run):
            raise RuntimeError("export: ошибка экспорта")
    card["onnx"] = name


def _yaml(path: Path) -> dict[str, Any]:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _split_images(data_yaml: Path, split: str) -> list[Path]:
    meta = _yaml(data_yaml)
    base = Path(meta.get("path") or data_yaml.parent)
    entries = meta[split] if isinstance(meta[split], list) else [meta[split]]
    out: list[Path] = []
    for e in entries:
        p = Path(e) if Path(e).is_absolute() else base / e
        out += sorted(f for f in p.rglob("*") if f.suffix.lower() in (".jpg", ".jpeg", ".png"))
    return out


def _has_class(image: Path, cls_id: int) -> bool:
    parts = list(image.parts)
    idx = len(parts) - 1 - parts[::-1].index("images")
    parts[idx] = "labels"
    label = Path(*parts).with_suffix(".txt")
    return label.exists() and any(line.split()[:1] == [str(cls_id)] for line in label.read_text().splitlines())


def step_calib(card: dict[str, Any], args: argparse.Namespace) -> Path:
    """Калибровка: половина — кадры train с классом модели (иначе потолок Sigmoid по этому классу низкий,
    LINQ_NOTES §17), половина — общий data/calib. Отбор детерминирован (seed 12)."""
    if args.data is None:
        raise ValueError("calib: нужен --data")
    name = program_name(card["version"], args.imgsz)
    out = MODELS_DIR / f"calib_{name}"
    rng = random.Random(12)
    cls_id = card["classes"].index(args.calib_class)
    with_cls = [p for p in _split_images(args.data, "train") if _has_class(p, cls_id)]
    extra = sorted(p for p in args.calib_extra.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    half = args.calib_n // 2
    picked = [("train:" + args.calib_class, p) for p in rng.sample(with_cls, min(half, len(with_cls)))]
    picked += [("data/calib", p) for p in rng.sample(extra, min(args.calib_n - len(picked), len(extra)))]
    print(f"calib: {len(picked)} кадров ({sum(1 for s, _ in picked if s != 'data/calib')} с "
          f"{args.calib_class}) → {out}")
    if args.dry_run:
        return out
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    with (out / "manifest.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "source", "original"])
        for k, (src, p) in enumerate(picked):
            dst = f"{k:03d}{p.suffix.lower()}"
            shutil.copy2(p, out / dst)
            w.writerow([dst, src, str(p)])
    return out


def _tar(src: Path, dst: Path) -> Path:
    with tarfile.open(dst, "w") as t:
        t.add(src, arcname=src.name)
    return dst


def step_compile(card: dict[str, Any], args: argparse.Namespace, stand: Stand) -> None:
    onnx = card.get("onnx") or onnx_name(card["version"], args.imgsz)
    name = program_name(card["version"], args.imgsz)
    calib = MODELS_DIR / f"calib_{name}"
    if not args.dry_run and not calib.exists():
        raise FileNotFoundError(f"compile: нет {calib} (шаг calib)")
    for script in TPU_SCRIPTS:
        stand.upload(ROOT / "tpu" / script, f"{STAND_TOOLS}/{script}")
    stand.upload(MODELS_DIR / onnx, f"{STAND_WORK}/models/{onnx}")
    tar = Path("/tmp") / f"calib_{name}.tar"
    if not args.dry_run:
        _tar(calib, tar)
    stand.upload(tar, f"{STAND_WORK}/models/calib_{name}.tar")
    n = len(list(calib.glob("*.jpg"))) if calib.exists() else args.calib_n
    common = (f"--cut-preset yolov8 --calib-dir models/calib_{name} --calib-n {n} --preproc letterbox "
              f"--preset O5 --fallback O1 --out-root models/tpu")
    tools = f"$HOME/{STAND_TOOLS}"
    s1 = program_name(card["version"], args.imgsz, s1=True)
    script = " && ".join([
        f"cd ~/{STAND_WORK}",
        f"tar -xf models/calib_{name}.tar -C models",
        f"OMP_NUM_THREADS=2 nice -n 10 {SDK_PY} {tools}/compile.py --onnx models/{onnx} --name {name} {common}",
        f"{SDK_PY} {tools}/fix_thresholds.py models/tpu/{name}/thresholds.json models/tpu/{name}/thresholds_sig1.json",
        f"OMP_NUM_THREADS=2 nice -n 10 {SDK_PY} {tools}/compile.py --onnx models/{onnx} --name {s1} {common} "
        f"--thresholds models/tpu/{name}/thresholds_sig1.json",
        f"cp -n models/tpu/{name}/{name}_b1_o5_h1m.tpu models/tpu/{s1}/{s1}_b1_o5_h1m.tpu /auto/cache/{STAND_LOGIN}/",
        f"ls -la /auto/cache/{STAND_LOGIN}/ | grep {name}",
    ])
    out = stand.sh(script, timeout=3600)
    print(out[-3000:])


def step_check(card: dict[str, Any], args: argparse.Namespace, stand: Stand) -> dict[str, Any]:
    """Сверка обеих программ (калибровка и s1) с CPU fp32 на отложенных кадрах; лучшая — в карточку."""
    onnx = card.get("onnx") or onnx_name(card["version"], args.imgsz)
    name = program_name(card["version"], args.imgsz)
    folders = args.check_images or ([args.data.parent / "images" / "test"] if args.data else [])
    images = sorted(p for d in folders for p in d.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    rng = random.Random(7)
    images = sorted(rng.sample(images, min(args.check_n, len(images))))
    check_dir = MODELS_DIR / f"check_{name}"
    print(f"check: {len(images)} отложенных кадров из {', '.join(str(d) for d in folders)}")
    if not args.dry_run:
        shutil.rmtree(check_dir, ignore_errors=True)
        check_dir.mkdir(parents=True)
        for k, p in enumerate(images):
            shutil.copy2(p, check_dir / f"{k:03d}_{p.name[-60:]}")
        _tar(check_dir, Path("/tmp") / f"check_{name}.tar")
    stand.upload(Path("/tmp") / f"check_{name}.tar", f"{STAND_WORK}/models/check_{name}.tar")
    results = {}
    for prog in (name, program_name(card["version"], args.imgsz, s1=True)):
        out_json = f"models/tpu/{prog}/check.json"
        stand.sh(" && ".join([
            f"cd ~/{STAND_WORK}",
            f"tar -xf models/check_{name}.tar -C models",
            f"GUARD_MODELS_DIR=$HOME/{STAND_WORK}/models PYTHONPATH=$HOME/{STAND_WORK}/mosprom OMP_NUM_THREADS=2 "
            f"{SDK_PY} $HOME/{STAND_TOOLS}/yolo_check.py --model {prog} --onnx models/{onnx} "
            f"--images models/check_{name} --n-images {len(images) or args.check_n} --nc {len(card['classes'])} "
            f"--conf {args.check_conf} --raw -n 20 --windows 1,4 --out {out_json}",
        ]), timeout=1800)
        text = stand.download_text(f"{STAND_WORK}/{out_json}")
        results[prog] = json.loads(text) if text else {}
    return results


def summarize_check(results: dict[str, Any], conf: float) -> dict[str, Any]:
    """Сводка для карточки: recall/precision TPU относительно CPU по порогам TPU."""
    out = {}
    for prog, res in results.items():
        acc = res.get("accuracy", res)
        out[prog] = {
            "frames": acc.get("n_images") or acc.get("images"),
            "cpu_conf": conf,
            "recall": acc.get("recall_vs_cpu"),
            "precision": acc.get("precision_vs_cpu"),
            "sweep": acc.get("tpu_threshold_sweep"),
        }
    return out


# ---------- реестр ----------


def _entry_text(card: dict[str, Any], args: argparse.Namespace, stand: bool, comment: str) -> str:
    onnx = card.get("onnx") or onnx_name(card["version"], args.imgsz)
    prog = program_path(program_name(card["version"], args.imgsz, s1=args.program == "s1"))
    conf = args.tpu_conf if stand else card.get("conf_threshold", 0.3)
    lines = [
        f"  - model_id: {card['model_id']}             # {card['version']}{comment}",
        "    task: detect",
        f"    onnx_path: {onnx}",
        f"    tpu_program: {prog}",
        f"    input_size: [{args.imgsz}, {args.imgsz}]",
        "    output_format: yolov8",
        f"    conf_threshold: {conf}",
        "    iou_threshold: 0.5",
        f"    classes: [{', '.join(card['classes'])}]",
    ]
    return "\n".join(lines) + "\n"


def upsert_entry(text: str, anchor: str, model_id: str, entry: str) -> str:
    """Вставить или заменить запись model_id сразу под строкой-якорем. Остальной файл не трогается."""
    lines = text.splitlines(keepends=True)
    idx = next((i for i, line in enumerate(lines) if line.lstrip().startswith("#") and anchor in line), None)
    if idx is None:
        raise ValueError(f"нет якоря «{anchor}»")
    # существующая запись этого model_id под якорем — до следующей записи, якоря или пустой строки
    start = idx + 1
    if start < len(lines) and re.match(rf"\s*- model_id:\s*{re.escape(model_id)}\b", lines[start]):
        end = start + 1
        while end < len(lines) and lines[end].startswith("    ") and not lines[end].lstrip().startswith("- "):
            end += 1
        lines[start:end] = [entry]
    else:
        lines[start:start] = [entry]
    return "".join(lines)


def step_registry(card: dict[str, Any], args: argparse.Namespace) -> None:
    gate = card.get("gate") or {}
    if not gate.get("promote") and not args.force:
        raise RuntimeError(f"registry: {card['version']} не прошла гейт (gate.promote != true); --force — под свою ответственность")
    if args.tpu_conf is None:
        raise ValueError("registry: нужен --tpu-conf (порог на TPU по итогам check)")
    anchor = args.anchor or card["model_id"]
    for path, stand in ((ROOT / "configs/models.yaml", False), (ROOT / "configs/models.stand.yaml", True)):
        text = path.read_text(encoding="utf-8")
        comment = ", программа TPU и порог по сверке (ml/flywheel/deploy.py)" if stand else ""
        cpu_card = dict(card)
        cpu_card["conf_threshold"] = _current_conf(text, card["model_id"]) or 0.3
        new = upsert_entry(text, f"{_anchor_word(anchor)}", card["model_id"],
                           _entry_text(cpu_card, args, stand, comment))
        print(f"registry: {path.relative_to(ROOT)}")
        _print_diff(text, new)
        if not args.dry_run:
            path.write_text(new, encoding="utf-8")
    if not args.dry_run:
        for other in list_cards():
            if other["model_id"] == card["model_id"] and other["status"] == "deployed" \
                    and other["version"] != card["version"]:
                other["status"] = "archived"
                save_card(other)
        card["status"] = "deployed"
        card["tpu_program"] = program_path(program_name(card["version"], args.imgsz, s1=args.program == "s1"))


def _anchor_word(model_id: str) -> str:
    """Якоря в реестрах: «погрузчик (S4)…» и т. п.; для известных model_id — русское слово якоря."""
    return {"forklift": "погрузчик (S4)"}.get(model_id, model_id)


def _current_conf(text: str, model_id: str) -> float | None:
    m = re.search(rf"- model_id:\s*{re.escape(model_id)}\b.*?\n((?:    .*\n)+)", text)
    if not m:
        return None
    c = re.search(r"conf_threshold:\s*([0-9.]+)", m.group(1))
    return float(c.group(1)) if c else None


def _print_diff(old: str, new: str) -> None:
    import difflib

    for line in difflib.unified_diff(old.splitlines(), new.splitlines(), lineterm="", n=1):
        print("  " + line)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    steps = [s.strip() for s in args.steps.split(",") if s.strip()]
    unknown = set(steps) - set(STEPS)
    if unknown:
        print(f"deploy: неизвестные шаги {sorted(unknown)}", file=sys.stderr)
        return 2
    card = load_card(args.version)
    stand = Stand(args.env, args.dry_run) if {"compile", "check"} & set(steps) else None
    if "export" in steps:
        step_export(card, args)
    if "calib" in steps:
        step_calib(card, args)
    if "compile" in steps:
        step_compile(card, args, stand)
    if "check" in steps:
        res = step_check(card, args, stand)
        if not args.dry_run:
            card["tpu_check"] = summarize_check(res, args.check_conf)
    if "registry" in steps:
        step_registry(card, args)
    if not args.dry_run:
        print(f"deploy: карточка {save_card(card)}")
    print("\nEdge не перезапускался. Оркестратору, когда решит включить модель на стенде:\n"
          "  python tools/stand.py sync            # код и configs/*.stand.yaml\n"
          "  python tools/stand.py restart         # edge подхватит модель на камерах с params.models\n"
          f"Программы: ~/work/models/{program_path(program_name(card['version'], args.imgsz, args.program == 's1'))} "
          f"и /auto/cache/{STAND_LOGIN}/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
