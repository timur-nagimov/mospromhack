"""Звено 6 маховика, шаги 2–4 и 6: экспорт ONNX → компиляция под LinQ TPU → сверка TPU с CPU → реестр.

Шаги (`--steps`, по умолчанию все по порядку; `--dry-run` печатает команды и ничего не меняет):
- `export`   — `ml/export_onnx.py` (torch, через tools/heavy.sh): `models/<версия>-<imgsz>.onnx`, поле `onnx` карточки;
- `calib`    — `models/calib_<имя>/`: кадры с погрузчиком (классом модели) из train набора и `data/calib`
  без кадров роликов. Тестовые кадры набора и ролики оценки в калибровку не идут;
- `compile`  — на стенде (`HEAVY_LANE=stand`): загрузка ONNX, калибровки и `tpu/*.py` в `~/work`,
  `tpu/compile.py` с калибровкой, затем `tpu/fix_thresholds.py` (потолок Sigmoid 1.0, LINQ_NOTES §17) и
  повторная компиляция `<имя>s1`. Программы копируются в `/auto/cache/<логин>/` (новые файлы, чужие не трогаем);
- `check`    — `tpu/yolo_check.py` на стенде, raw: TPU против CPU fp32 на отложенных кадрах (тест набора и
  `--check-images`), recall и precision по порогам TPU для класса модели и для всех классов. Итог — `tpu_check`
  карточки, сырые ответы — `data/flywheel/runs/<версия>/`;
- `registry` — только после gate.promote: `tpu_program` в записи модели `configs/models.yaml` и запись
  в `configs/models.stand.yaml` под якорем модели (порог TPU — `--tpu-conf` или подсказка по сверке);
  карточка → `deployed`, прежняя `deployed` того же model_id → `archived`.

Edge и камеры не трогаются: в конце печатается, что сделать оркестратору.

Пример:
    HEAVY_LANE=stand tools/heavy.sh .venv/bin/python ml/flywheel/deploy.py --version forklift-v1 \\
        --data data/datasets/forklift/data.yaml --steps calib,compile,check --check-images <кадры S4>
    .venv/bin/python ml/flywheel/deploy.py --version forklift-v1 --steps registry --dry-run
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
STAND_URL = "http://127.0.0.1:8888/"       # как в tools/stand.py: стенд доступен только через контейнер infra/vpn
STAND_WORK = "work"                       # ~/work на стенде
STAND_TOOLS = "work/ds_flywheel/tpu"      # свои копии tpu/*.py, guard_repo и mosprom не трогаем
SDK_PY = "~/linq312/bin/python"
TPU_SCRIPTS = ("compile.py", "fix_thresholds.py", "yolo_check.py")
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")
# Как включить модель на камерах (deploy камеры не меняет — это решение оркестратора)
ENABLE_HINTS = {"forklift": "правилу vehicle_proximity камеры S4 — params.models: [det, forklift]"}


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
                    help="ещё каталоги отложенных кадров для сверки, кроме test набора (например, кадры роликов оценки)")
    ap.add_argument("--check-n", type=int, default=60, help="кадров test набора в сверке (все с классом модели + случайные)")
    ap.add_argument("--check-conf", type=float, default=0.3, help="порог CPU в сверке: conf_threshold модели в реестре")
    ap.add_argument("--birth-conf", type=float, default=0.45,
                    help="второй порог сверки: рождение трека (tracker_new_track_thresh, guard/pipeline/settings.py)")
    ap.add_argument("--tpu-conf", type=float,
                    help="conf_threshold модели на TPU в models.stand.yaml (по умолчанию — подсказка по сверке)")
    ap.add_argument("--program", choices=["calib", "s1"], default="s1",
                    help="какую программу выкатывать: с порогами калибровки или с потолком Sigmoid 1.0")
    ap.add_argument("--anchor", help="подстрока строки-якоря в configs/models*.yaml (по умолчанию «<model_id>»)")
    ap.add_argument("--force", action="store_true", help="пересобрать готовое; выкатить без promote в гейте")
    ap.add_argument("--env", type=Path, default=default_env(),
                    help=".env с доступом к JupyterHub (по умолчанию ближайший вверх от корня репозитория)")
    ap.add_argument("--url", default=STAND_URL, help="Jupyter стенда: через контейнер infra/vpn (%(default)s)")
    return ap


def default_env() -> Path:
    """.env ищется вверх от корня: в worktree это .env основного репозитория."""
    from guard.pipeline.settings import find_dotenv

    return find_dotenv(ROOT) or ROOT / ".env"


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

    def __init__(self, env: Path, dry: bool, url: str = STAND_URL) -> None:
        self.dry = dry
        self.jh = None
        if not dry:
            from tools.jhub import JHub

            self.jh = JHub.from_env(env, url=url)
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
        out += sorted(f for f in p.rglob("*") if f.suffix.lower() in IMAGE_SUFFIXES)
    return out


def _has_class(image: Path, cls_id: int) -> bool:
    """Есть ли класс в разметке YOLO (images/… → labels/….txt). Кадр без разметки (кадры роликов) — нет."""
    parts = list(image.parts)
    if "images" not in parts:
        return False
    idx = len(parts) - 1 - parts[::-1].index("images")
    parts[idx] = "labels"
    label = Path(*parts).with_suffix(".txt")
    return label.exists() and any(line.split()[:1] == [str(cls_id)] for line in label.read_text().splitlines())


def calib_extra_frames(calib_dir: Path, exclude_sha1: set[str]) -> list[Path]:
    """Кадры общего калибровочного набора без кадров видео (source data/videos/… в manifest.csv: демо-камеры
    сделаны из тех же исходников, что ролики оценки s1_01 и s1_02) и без копий кадров сверки (sha1)."""
    import hashlib

    from_videos = set()
    manifest = calib_dir / "manifest.csv"
    if manifest.exists():
        with manifest.open(newline="", encoding="utf-8") as f:
            from_videos = {row["file"] for row in csv.DictReader(f) if row.get("source", "").startswith("data/videos")}
    frames = sorted(p for p in calib_dir.glob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    return [p for p in frames if p.name not in from_videos
            and hashlib.sha1(p.read_bytes()).hexdigest() not in exclude_sha1]


def step_calib(card: dict[str, Any], args: argparse.Namespace) -> Path:
    """Калибровка: половина — кадры train с классом модели (иначе потолок Sigmoid по этому классу низкий,
    LINQ_NOTES §17), половина — общий data/calib без кадров роликов. Тестовые кадры набора не берутся.
    Отбор детерминирован (seed 12)."""
    import hashlib

    if args.data is None:
        raise ValueError("calib: нужен --data")
    name = program_name(card["version"], args.imgsz)
    out = MODELS_DIR / f"calib_{name}"
    rng = random.Random(12)
    cls_id = card["classes"].index(args.calib_class)
    with_cls = [p for p in _split_images(args.data, "train") if _has_class(p, cls_id)]
    test_sha1 = {hashlib.sha1(p.read_bytes()).hexdigest() for p in _split_images(args.data, "test")}
    extra = calib_extra_frames(args.calib_extra, test_sha1)
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
    n = sum(1 for p in calib.glob("*") if p.suffix.lower() in IMAGE_SUFFIXES) if calib.exists() else args.calib_n
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


def select_check_images(test_images: list[Path], cls_id: int, n: int, extra: list[Path],
                        seed: int = 7) -> list[Path]:
    """Кадры сверки: все тестовые кадры с классом модели (их мало, а важны именно они), остальное до n —
    случайные тестовые кадры (фуры, экскаваторы, люди: ложные рамки класса). Плюс все кадры из extra
    (например, кадры роликов оценки). Детерминированно."""
    with_cls = [p for p in test_images if _has_class(p, cls_id)]
    others = [p for p in test_images if p not in set(with_cls)]
    rng = random.Random(seed)
    picked = sorted(with_cls)[:n]
    picked += sorted(rng.sample(others, max(0, min(n - len(picked), len(others)))))
    return picked + sorted(extra)


def check_variants(cls_id: int, cls_name: str, conf: float, birth_conf: float) -> list[dict[str, Any]]:
    """Прогоны yolo_check на программу: класс модели при пороге реестра и при пороге рождения трека,
    все классы при пороге реестра (там же замер скорости)."""
    return [
        {"tag": f"{cls_name}@{conf}", "classes": [cls_id], "conf": conf, "sweep": "0,0.05,0.1,0.15", "perf": False},
        {"tag": f"{cls_name}@{birth_conf}", "classes": [cls_id], "conf": birth_conf, "sweep": "0,0.05,0.1,0.15",
         "perf": False},
        {"tag": f"all@{conf}", "classes": None, "conf": conf, "sweep": "0,0.05,0.1", "perf": True},
    ]


def step_check(card: dict[str, Any], args: argparse.Namespace, stand: Stand) -> dict[str, Any]:
    """Сверка обеих программ (калибровка и s1) с CPU fp32 на отложенных кадрах (raw, tpu/yolo_check.py)."""
    onnx = card.get("onnx") or onnx_name(card["version"], args.imgsz)
    name = program_name(card["version"], args.imgsz)
    cls_id = card["classes"].index(args.calib_class)
    test_images = _split_images(args.data, "test") if args.data else []
    extra = sorted(p for d in args.check_images for p in d.glob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    images = select_check_images(test_images, cls_id, args.check_n, extra)
    n_cls = sum(1 for p in images if _has_class(p, cls_id))
    check_dir = MODELS_DIR / f"check_{name}"
    print(f"check: {len(images)} отложенных кадров ({n_cls} с {args.calib_class} в тесте набора, "
          f"{len(extra)} из {', '.join(str(d) for d in args.check_images) or '—'})")
    if not args.dry_run:
        shutil.rmtree(check_dir, ignore_errors=True)
        check_dir.mkdir(parents=True)
        with (check_dir.parent / f"check_{name}.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["file", "original"])
            for k, p in enumerate(images):
                dst = f"{k:03d}_{p.name[-60:]}"
                shutil.copy2(p, check_dir / dst)
                w.writerow([dst, str(p)])
        _tar(check_dir, Path("/tmp") / f"check_{name}.tar")
    stand.upload(Path("/tmp") / f"check_{name}.tar", f"{STAND_WORK}/models/check_{name}.tar")
    stand.sh(f"cd ~/{STAND_WORK} && rm -rf models/check_{name} && tar -xf models/check_{name}.tar -C models")
    results: dict[str, Any] = {}
    for prog in (name, program_name(card["version"], args.imgsz, s1=True)):
        results[prog] = {}
        for v in check_variants(cls_id, args.calib_class, args.check_conf, args.birth_conf):
            out_json = f"models/tpu/{prog}/check_{v['tag'].replace('@', '_')}.json"
            stand.sh(
                f"cd ~/{STAND_WORK} && rm -f {out_json} && "
                f"GUARD_MODELS_DIR=$HOME/{STAND_WORK}/models PYTHONPATH=$HOME/{STAND_WORK}/mosprom OMP_NUM_THREADS=2 "
                f"nice -n 10 {SDK_PY} $HOME/{STAND_TOOLS}/yolo_check.py --model {prog} --onnx models/{onnx} "
                f"--images models/check_{name} --n-images {len(images) or args.check_n} --nc {len(card['classes'])} "
                f"--conf {v['conf']} --sweep {v['sweep']} --raw -n 20 --windows 1,4 "
                + (f"--classes {','.join(map(str, v['classes']))} " if v["classes"] else "")
                + ("" if v["perf"] else "--no-perf ") + f"--out {out_json}",
                timeout=1800)
            text = stand.download_text(f"{STAND_WORK}/{out_json}")
            results[prog][v["tag"]] = json.loads(text) if text else {}
    return results


def summarize_check(results: dict[str, Any]) -> dict[str, Any]:
    """Сводка для карточки по формату tpu/yolo_check.py ({<программа>[_raw]: {accuracy_vs_cpu, perf}}):
    recall и precision TPU относительно CPU fp32 по порогам TPU, IoU, сдвиг уверенности, скорость."""
    out: dict[str, Any] = {}
    for prog, variants in results.items():
        out[prog] = {}
        for tag, res in variants.items():
            entry = res.get(f"{prog}_raw") or res.get(prog) or next(iter(res.values()), {}) if res else {}
            acc = entry.get("accuracy_vs_cpu") or {}
            item = {
                "frames": acc.get("images"),
                "cpu_conf": acc.get("conf"),
                "cpu_dets": acc.get("cpu_dets"),
                "recall": acc.get("recall_vs_cpu"),
                "precision": acc.get("precision_vs_cpu"),
                "iou_mean": acc.get("iou_mean"),
                "dconf_mean": acc.get("dconf_signed_mean"),
                "sweep": {k: {"recall": v.get("recall_vs_cpu"), "precision": v.get("precision_vs_cpu")}
                          for k, v in (acc.get("tpu_threshold_sweep") or {}).items()},
            }
            perf = {k: {"fps": v.get("fps"), "total_p50_ms": ((v.get("call_ms") or {}).get("total") or {}).get("p50")}
                    for k, v in (entry.get("perf") or {}).items() if k.startswith("window_")}
            if perf:
                item["perf"] = perf
            out[prog][tag] = item
    return out


def suggest_tpu_conf(summary: dict[str, Any], tag: str, min_recall: float = 0.95,
                     min_precision: float = 0.9) -> float | None:
    """Порог на TPU: самый высокий из перебора, при котором recall TPU относительно CPU ≥ min_recall и
    precision ≥ min_precision. Если такого нет — порог с наибольшей суммой recall + precision."""
    sweep = (summary.get(tag) or {}).get("sweep") or {}
    rows = [(float(k), v.get("recall") or 0.0, v.get("precision") or 0.0) for k, v in sweep.items()]
    if not rows:
        return None
    ok = [r for r in rows if r[1] >= min_recall and r[2] >= min_precision]
    if ok:
        return max(ok)[0]
    return max(rows, key=lambda r: (r[1] + r[2], r[0]))[0]


# ---------- реестр ----------


def entry_text(card: dict[str, Any], imgsz: int, program: str, conf: float, comment: str) -> str:
    """Запись реестра (ModelSpec) в стиле configs/models*.yaml."""
    onnx = card.get("onnx") or onnx_name(card["version"], imgsz)
    lines = [
        f"  - model_id: {card['model_id']}             # {card['version']} (ml/model_cards/{card['version']}.json){comment}",
        "    task: detect",
        f"    onnx_path: {onnx}",
        f"    tpu_program: {program}",
        f"    input_size: [{imgsz}, {imgsz}]",
        "    output_format: yolov8",
        f"    conf_threshold: {conf}",
        "    iou_threshold: 0.5",
        f"    classes: [{', '.join(card['classes'])}]",
    ]
    return "\n".join(lines) + "\n"


_KV = re.compile(r"^(\s+)([A-Za-z_]+):\s*(.*?)\s*(#.*)?$")


def _keep_comments(old: list[str], new: str) -> str:
    """Комментарии старой записи остаются у строк, где ключ и значение не изменились."""
    old_kv = {}
    for line in old:
        m = _KV.match(line.rstrip("\n"))
        if m and m.group(4):
            old_kv[m.group(2)] = (m.group(3), m.group(4))
    out = []
    for line in new.splitlines(keepends=True):
        m = _KV.match(line.rstrip("\n"))
        if m and not m.group(4) and m.group(2) in old_kv and old_kv[m.group(2)][0] == m.group(3):
            line = f"{line.rstrip()}{' ' * max(1, 35 - len(line.rstrip()))}{old_kv[m.group(2)][1]}\n"
        out.append(line)
    return "".join(out)


def upsert_entry(text: str, anchor: str, model_id: str, entry: str) -> str:
    """Вставить или заменить запись model_id сразу под строкой-якорем. Остальной файл не трогается.
    Запись этого model_id в другом месте файла — ошибка: две записи одного model_id реестр не различит."""
    lines = text.splitlines(keepends=True)
    idx = next((i for i, line in enumerate(lines) if line.lstrip().startswith("#") and anchor in line), None)
    if idx is None:
        raise ValueError(f"нет якоря «{anchor}»")
    pattern = re.compile(rf"\s*- model_id:\s*{re.escape(model_id)}(\s|$)")
    # существующая запись этого model_id под якорем — до следующей записи, якоря или пустой строки
    start = idx + 1
    if start < len(lines) and pattern.match(lines[start]):
        end = start + 1
        while end < len(lines) and lines[end].startswith("    ") and not lines[end].lstrip().startswith("- "):
            end += 1
        lines[start:end] = [_keep_comments(lines[start + 1:end], entry)]
    else:
        lines[start:start] = [entry]
    if sum(1 for line in lines for one in line.splitlines() if pattern.match(one)) != 1:
        raise ValueError(f"model_id {model_id} уже есть в файле не под якорем «{anchor}»")
    return "".join(lines)


def tpu_conf_for(card: dict[str, Any], args: argparse.Namespace) -> float:
    """Порог на TPU: --tpu-conf или подсказка по сверке выбранной программы (класс модели, порог реестра)."""
    if args.tpu_conf is not None:
        return args.tpu_conf
    prog = program_name(card["version"], args.imgsz, s1=args.program == "s1")
    summary = (card.get("tpu_check") or {}).get(prog) or {}
    conf = suggest_tpu_conf(summary, f"{args.calib_class}@{args.check_conf}")
    if conf is None:
        raise ValueError(f"registry: нет сверки {prog} в карточке (шаг check) — задайте --tpu-conf")
    return conf


def step_registry(card: dict[str, Any], args: argparse.Namespace) -> None:
    gate = card.get("gate") or {}
    if not gate.get("promote") and not args.force:
        raise RuntimeError(f"registry: {card['version']} не прошла гейт (gate.promote != true); --force — под свою ответственность")
    tpu_conf = tpu_conf_for(card, args)
    program = program_path(program_name(card["version"], args.imgsz, s1=args.program == "s1"))
    anchor = args.anchor or _anchor_word(card["model_id"])
    for path, stand in ((ROOT / "configs/models.yaml", False), (ROOT / "configs/models.stand.yaml", True)):
        text = path.read_text(encoding="utf-8")
        if stand:
            conf, comment = tpu_conf, f", порог TPU по сверке с CPU (tpu_check карточки)"
        else:
            conf, comment = _current_conf(text, card["model_id"]) or 0.3, ""
        new = upsert_entry(text, anchor, card["model_id"], entry_text(card, args.imgsz, program, conf, comment))
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
        card["tpu_program"] = program


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
    stand = Stand(args.env, args.dry_run, args.url) if {"compile", "check"} & set(steps) else None
    if "export" in steps:
        step_export(card, args)
    if "calib" in steps:
        step_calib(card, args)
    if "compile" in steps:
        step_compile(card, args, stand)
    if "check" in steps:
        res = step_check(card, args, stand)
        if not args.dry_run:
            raw = ROOT / "data/flywheel/runs" / card["version"] / f"tpu_check-{time.strftime('%Y%m%d-%H%M')}.json"
            raw.parent.mkdir(parents=True, exist_ok=True)
            raw.write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
            card["tpu_check"] = summarize_check(res)
            for prog, summary in card["tpu_check"].items():
                tag = f"{args.calib_class}@{args.check_conf}"
                print(f"check: {prog}: {json.dumps(summary.get(tag), ensure_ascii=False)}; "
                      f"порог TPU по сверке {suggest_tpu_conf(summary, tag)}")
            print(f"check: сырые результаты {raw}")
    if "registry" in steps:
        step_registry(card, args)
    if not args.dry_run:
        print(f"deploy: карточка {save_card(card)}")
    print(edge_hint(card, args))
    return 0


def edge_hint(card: dict[str, Any], args: argparse.Namespace) -> str:
    """Что сделать оркестратору: deploy edge не перезапускает и камеры не меняет."""
    prog = program_path(program_name(card["version"], args.imgsz, args.program == "s1"))
    enable = ENABLE_HINTS.get(card["model_id"], f"подключить {card['model_id']} у правила, которому нужна модель")
    return ("\nEdge не перезапускался, камеры не менялись. Оркестратору, когда решит включить модель на стенде:\n"
            f"  1) configs/cameras.stand.yaml: {enable};\n"
            "  2) python tools/stand.py sync      # код и configs/*.stand.yaml на стенд;\n"
            "  3) перезапуск edge (tools/stand.py restart) — только оркестратор, вне окон показа.\n"
            f"Программа: ~/work/models/{prog} и /auto/cache/{STAND_LOGIN}/{Path(prog).name}.")


if __name__ == "__main__":
    raise SystemExit(main())
