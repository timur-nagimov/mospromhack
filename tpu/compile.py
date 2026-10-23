#!/usr/bin/env python3
"""ONNX -> программа LinQ TPU (*.tpu): оптимизация, калибровка, квантование, компиляция.

Повторяет поток из tpu_framework.pdf (гл. 4-5, стр. 9-33) и compile.py из SDK, плюс:
  * отрезание «хвоста» сети (--cut / --cut-preset): на TPU уходит тяжёлая часть, хвост сохраняется
    отдельным ONNX и выполняется на CPU (так поставляются yolov8n_pose, yolo12m, yolo5s — pdf стр. 36);
  * калибровка на реальных кадрах (--calib-dir) с нужным препроцессингом;
  * откат пресета O5 -> O1, если O5 не компилируется (pdf стр. 17);
  * сверка квантованной модели с fp32 ONNX (косинус, max|diff|) и manifest.json для бэкенда.

SDK есть только под linux/x86_64 + Python 3.12 (pdf стр. 5), поэтому на маке:
    python tpu/compile.py --docker --onnx models/yolov8n-640.onnx --name yolov8n_640 --cut-preset yolov8 \
        --calib-dir models/tpu/calib_frames --preproc letterbox
Флаг --docker перезапускает этот же скрипт в образе linq-sdk:2026.9.45 (tpu/docker/Dockerfile),
монтируя корень проекта по тому же абсолютному пути. В Jupyter-стенде (SDK уже стоит) — без --docker.

Результат в <out-root>/<name>/: <name>_b<B>_<preset>_<platform>.tpu, *_head.onnx, *_tail.onnx (если cut),
thresholds.json, quant_*.qm.onnx, manifest.json, compile.log.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import platform
import shlex
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(os.environ.get("GUARD_ROOT", "/Users/nagimov-timur/Desktop/mosprom"))
MODELS_DIR = Path(os.environ.get("GUARD_MODELS_DIR", ROOT / "models"))
DOCKER_IMAGE = os.environ.get("LINQ_SDK_IMAGE", "localhost/linq-sdk:2026.9.45")
SDK_VERSION = "2026.9.45"

# Точки отреза «головы» (выходы части, которая пойдёт на TPU).
# yolov8 (detect): как у вендора для yolov8n_pose (pdf стр. 36, сноска 6) — после DFL-свёртки и Sigmoid классов.
# Иначе выход output0 [1,84,8400] смешивает координаты 0..640 и вероятности 0..1 в одном int8-тензоре.
CUT_PRESETS: dict[str, list[str]] = {
    "yolov8": ["/model.22/dfl/conv/Conv_output_0", "/model.22/Sigmoid_output_0"],
    "yolov8-pose": [  # дословно из pdf стр. 36, сноска 6
        "/model.22/Mul_6_output_0",
        "/model.22/Concat_6_output_0",
        "/model.22/Concat_5_output_0",
        "/model.22/Sigmoid_1_output_0",
        "/model.22/Concat_7_output_0",
        "/model.22/Sigmoid_output_0",
        "/model.22/dfl/conv/Conv_output_0",
    ],
    # голые свёртки детект-головы (3 масштаба x box/cls) — запасной вариант, если DFL/Softmax не компилируется
    "yolov8-conv": [
        f"/model.22/cv{k}.{i}/cv{k}.{i}.2/Conv_output_0" for i in range(3) for k in (2, 3)
    ],
}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ----------------------------------------------------------------------------- utils


class Tee:
    def __init__(self, path: Path):
        self.f = path.open("a", encoding="utf-8")
        self.out = sys.__stdout__

    def write(self, s: str) -> int:
        self.out.write(s)
        self.f.write(s)
        return len(s)

    def flush(self) -> None:
        self.out.flush()
        self.f.flush()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ----------------------------------------------------------------------------- препроцессинг калибровки


def _load_rgb(path: str):
    from PIL import Image

    with Image.open(path) as im:
        return im.convert("RGB")


def preprocess(path: str, kind: str, hw: tuple[int, int]):
    """Картинка -> тензор (1, C, H, W) float32 в формате входа модели."""
    import numpy as np
    from PIL import Image

    h, w = hw
    img = _load_rgb(path)
    if kind == "letterbox":  # YOLOv8: RGB 0..1, letterbox с серым 114 (как guard.vision.letterbox)
        iw, ih = img.size
        r = min(w / iw, h / ih)
        nw, nh = max(1, round(iw * r)), max(1, round(ih * r))
        canvas = Image.new("RGB", (w, h), (114, 114, 114))
        canvas.paste(img.resize((nw, nh), Image.BILINEAR), ((w - nw) // 2, (h - nh) // 2))
        arr = np.asarray(canvas, dtype=np.float32) / 255.0
    elif kind == "imagenet":  # torch resnet50 из примера SDK (pdf стр. 21-22)
        iw, ih = img.size
        m = int(min(iw, ih) * 0.85)
        top, left = (ih - m) // 2, (iw - m) // 2
        img = img.crop((left, top, left + m, top + m)).resize((w, h))
        arr = (np.asarray(img, dtype=np.float32) / 255.0 - np.array(IMAGENET_MEAN, np.float32)) / np.array(
            IMAGENET_STD, np.float32
        )
    elif kind == "resize01":
        arr = np.asarray(img.resize((w, h), Image.BILINEAR), dtype=np.float32) / 255.0
    else:
        raise ValueError(f"неизвестный препроцессинг {kind}")
    return np.ascontiguousarray(arr.transpose(2, 0, 1)[None], dtype=np.float32)


def calib_files(calib_dir: str | None, n: int) -> list[str]:
    if not calib_dir:
        return []
    exts = ("*.jpg", "*.jpeg", "*.png", "*.JPEG", "*.JPG", "*.bmp")
    files = sorted({f for e in exts for f in glob.glob(os.path.join(calib_dir, "**", e), recursive=True)})
    if len(files) > n:  # равномерная подвыборка
        step = len(files) / n
        files = [files[int(i * step)] for i in range(n)]
    return files


# ----------------------------------------------------------------------------- ONNX


def input_shapes(model) -> dict[str, list[int]]:
    inits = {i.name for i in model.graph.initializer}
    return {
        i.name: [d.dim_value for d in i.type.tensor_type.shape.dim]
        for i in model.graph.input
        if i.name not in inits
    }


def remove_initialized_inputs(model) -> None:  # из compile.py SDK
    name_to_input = {i.name: i for i in model.graph.input}
    for init in model.graph.initializer:
        if init.name in name_to_input:
            model.graph.input.remove(name_to_input[init.name])


def split_model(onnx_path: Path, cut: list[str], out_dir: Path, name: str) -> tuple[Path, Path, list[str], list[str]]:
    """Голова (вход -> cut) для TPU и хвост (cut -> исходные выходы) для CPU."""
    import onnx

    m = onnx.load(str(onnx_path))
    in_names = list(input_shapes(m))
    out_names = [o.name for o in m.graph.output]
    head_path = out_dir / f"{name}_head.onnx"
    tail_path = out_dir / f"{name}_tail.onnx"
    try:
        from tpu_framework import get_sub_model  # pdf стр. 39-41

        head = get_sub_model(model=m, ordered_input_names=in_names, ordered_output_names=cut)
        tail = get_sub_model(model=m, ordered_input_names=cut, ordered_output_names=out_names)
        onnx.save(head, str(head_path))
        onnx.save(tail, str(tail_path))
        how = "tpu_framework.get_sub_model"
    except Exception as e:  # noqa: BLE001
        log(f"get_sub_model не сработал ({e!r}), режем onnx.utils.extract_model")
        onnx.utils.extract_model(str(onnx_path), str(head_path), in_names, cut)
        onnx.utils.extract_model(str(onnx_path), str(tail_path), cut, out_names)
        how = "onnx.utils.extract_model"
    log(f"разрезано ({how}): head {in_names} -> {cut}; tail -> {out_names}")
    return head_path, tail_path, in_names, out_names


def ort_run(path: Path, feeds: dict) -> dict:
    import onnxruntime as ort

    s = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    names = [o.name for o in s.get_outputs()]
    return dict(zip(names, s.run(names, feeds)))


def compare(ref, got) -> dict:
    import numpy as np

    a = np.asarray(ref, np.float64).ravel()
    b = np.asarray(got, np.float64).ravel()
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    return {
        "cosine": round(cos, 6),
        "max_abs_diff": float(np.max(np.abs(a - b))),
        "mean_abs_diff": float(np.mean(np.abs(a - b))),
        "ref_absmax": float(np.max(np.abs(a))),
    }


# ----------------------------------------------------------------------------- квантованная модель на CPU


def run_quantized(quant_model, inputs: dict, out_names: list[str], batch: int, cwl: int) -> dict:
    """Инференс квантованной модели на CPU (torch), как _test_quantized_model в pdf стр. 22-26."""
    import torch
    from dnn_quant.api import TpuDtype

    g = quant_model.as_torch_graph(batch_size=batch, cache_word_length=cwl)
    gm = g.gm
    gm.train(False)  # режим инференса
    name_map = g.get_outputs_anchors() | g.get_inputs_anchors()
    data = quant_model.rescale({name_map[k]: v for k, v in inputs.items()}, TpuDtype.Input)
    with torch.no_grad():
        outs = gm(*[torch.as_tensor(data[n]) for n in g._input_layer_names])
    if not isinstance(outs, (tuple, list)):
        outs = (outs,)
    # порядок выходов torch-графа: по get_outputs_anchors (имя исходного тензора -> имя в графе)
    out_anchor = g.get_outputs_anchors()
    graph_out_names = list(out_anchor.keys()) if len(out_anchor) == len(outs) else out_names
    res = quant_model.rescale(
        {n: o.detach().cpu().numpy() for n, o in zip(graph_out_names, outs)}, TpuDtype.Output
    )
    return {n: res[n] for n in out_names if n in res}


# ----------------------------------------------------------------------------- main


def program_summary(tpu_path: Path) -> dict:
    import zipfile

    with zipfile.ZipFile(tpu_path) as z:
        meta = json.loads(z.read("metadata.json"))

    def strip(t: dict) -> dict:
        return {k: t[k] for k in t if k not in ("address",)}

    return {
        "inputs": {n: strip(t) for d in meta.get("inputs", {}).values() for n, t in d.items()},
        "outputs": {n: strip(t) for d in meta.get("outputs", {}).values() for n, t in d.items()},
        "ram_scheme": meta.get("ram_scheme"),
        "iva_tpu_ip_version": meta.get("iva_tpu_ip_version"),
        "iva_tpu_api_version": meta.get("iva_tpu_api_version"),
    }


def run_in_docker(argv: list[str]) -> int:
    script = Path(__file__).resolve()
    args = [a for a in argv if a != "--docker"]
    mounts = {str(ROOT)}
    if not str(script).startswith(str(ROOT)):
        mounts.add(str(script.parent.parent))
    cmd = ["docker", "run", "--rm", "--platform", "linux/amd64", "-e", "PYTHONUNBUFFERED=1",
           "-e", f"GUARD_ROOT={ROOT}", "-e", f"GUARD_MODELS_DIR={MODELS_DIR}"]
    for m in sorted(mounts):
        cmd += ["-v", f"{m}:{m}"]
    cmd += ["-w", os.getcwd(), DOCKER_IMAGE, "python", str(script), *args]
    print("+", " ".join(shlex.quote(c) for c in cmd), flush=True)
    return subprocess.call(cmd)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx", required=True, help="исходная ONNX-модель (статический shape)")
    ap.add_argument("--name", help="имя модели/каталога (по умолчанию — имя файла)")
    ap.add_argument("--out-root", default=str(MODELS_DIR / "tpu"), help="куда класть результат (%(default)s)")
    ap.add_argument("--preset", default="O5", help="пресет компилятора: O1|O3|O5|O5_FGS|... (pdf стр. 16-17)")
    ap.add_argument("--fallback", default="O1", help="пресет при ошибке основного ('' — без отката)")
    ap.add_argument("--platform", default="h1m", help="платформа: h1m (TPU_H1M) или имя для get_hw_params")
    ap.add_argument("--input-shape", action="append", default=[],
                    help="заморозить динамический вход: name:1x3x640x640 (freeze_shapes)")
    ap.add_argument("--cut", default="", help="имена тензоров-выходов головы через запятую")
    ap.add_argument("--cut-preset", choices=sorted(CUT_PRESETS), help="готовый набор точек отреза")
    ap.add_argument("--calib-dir", help="каталог с кадрами для калибровки")
    ap.add_argument("--calib-n", type=int, default=100, help="сколько кадров взять (50-200)")
    ap.add_argument("--preproc", default="letterbox", choices=["letterbox", "imagenet", "resize01"])
    ap.add_argument("--calib-method", default="torch", choices=["torch", "tf"],
                    help="calibrate_torch (как в compile.py SDK) или calibrate (TF, pdf стр. 11)")
    ap.add_argument("--activation-percentile", type=float, default=None)
    ap.add_argument("--samplewise-percentile", type=float, default=None,
                    help="устойчивость к выбросам на больших наборах (pdf стр. 12)")
    ap.add_argument("--thresholds", help="готовые thresholds.json (пропустить калибровку)")
    ap.add_argument("--no-validate", action="store_true", help="не сверять квантованную модель с fp32")
    ap.add_argument("--validate-image", help="картинка для сверки (по умолчанию первый кадр калибровки)")
    ap.add_argument("--docker", action="store_true", help="выполнить в Docker-образе SDK (для мака)")
    a = ap.parse_args(argv)

    if a.docker:
        return run_in_docker(argv)

    onnx_path = Path(a.onnx).resolve()
    name = a.name or onnx_path.stem
    out_dir = Path(a.out_root) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    sys.stdout = Tee(out_dir / "compile.log")  # type: ignore[assignment]
    sys.stderr = sys.stdout  # type: ignore[assignment]
    t_all = time.time()
    log(f"=== compile {name}: {onnx_path} | {platform.platform()} | python {platform.python_version()}")
    log("args: " + " ".join(shlex.quote(x) for x in argv))

    import numpy as np
    import onnx

    t0 = time.time()
    import tpu_framework  # noqa: F401
    from tpu_framework import (Network, QuantizedModel, RegularModel, TpuProgram, compiler,
                               freeze_shapes, presets, preliminary_model_conditioning)
    from tpu_tlm_is import get_hw_params

    log(f"tpu_framework импортирован за {time.time() - t0:.1f} с")
    hw = get_hw_params(a.platform)
    cwl = hw.tpu_parameters.cache_word_length

    manifest: dict = {
        "name": name,
        "sdk": SDK_VERSION,
        "source_onnx": str(onnx_path),
        "source_onnx_sha256": sha256(onnx_path),
        "platform": a.platform,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "command": "python tpu/compile.py " + " ".join(shlex.quote(x) for x in argv),
    }

    # --- 0. разрез
    cut = [c for c in a.cut.split(",") if c] or (CUT_PRESETS[a.cut_preset] if a.cut_preset else [])
    work_onnx = onnx_path
    if cut:
        head_path, tail_path, in_names, orig_out = split_model(onnx_path, cut, out_dir, name)
        work_onnx = head_path
        manifest.update({"cut": cut, "head_onnx": str(head_path), "tail_onnx": str(tail_path),
                         "orig_outputs": orig_out})
    else:
        manifest.update({"cut": [], "tail_onnx": None})

    # --- 1. оптимизация (pdf: preliminary_model_conditioning; compile.py шаг 1/3)
    log("1/3 оптимизация")
    m = onnx.load(str(work_onnx))
    remove_initialized_inputs(m)
    shapes = input_shapes(m)
    for spec in a.input_shape:
        n, s = spec.split(":")
        shapes[n] = [int(x) for x in s.lower().split("x")]
    if any(0 in s for s in shapes.values()) or a.input_shape:
        freeze_shapes(m, shapes)
    if any(0 in s for s in shapes.values()):
        log(f"ОШИБКА: динамический вход {shapes}; задайте --input-shape")
        return 2
    m_opt = preliminary_model_conditioning(m)
    opt_path = out_dir / f"{name}_opt.onnx"
    onnx.save(m_opt, str(opt_path))
    shapes = input_shapes(m_opt)
    batch = list(shapes.values())[0][0]
    out_names = [o.name for o in m_opt.graph.output]
    manifest.update({"inputs": {k: {"shape": v, "dtype": "float32"} for k, v in shapes.items()},
                     "tpu_outputs": out_names, "batch": batch, "preproc": a.preproc})
    log(f"входы {shapes}, выходы {out_names}")

    # --- 2. калибровка и квантование (pdf стр. 10-15, 27-29)
    log("2/3 калибровка и квантование")
    files = calib_files(a.calib_dir, a.calib_n)
    in_name = list(shapes)[0]
    hwsz = tuple(shapes[in_name][2:4])
    if files:
        t0 = time.time()
        calib = np.concatenate([preprocess(f, a.preproc, hwsz) for f in files], axis=0)
        log(f"калибровка: {len(files)} кадров из {a.calib_dir} -> {calib.shape} ({time.time() - t0:.1f} с)")
        manifest["calibration"] = {"dir": str(a.calib_dir), "n": len(files), "preproc": a.preproc,
                                   "files_sha256_first": [sha256(Path(f))[:12] for f in files[:3]]}
    else:
        calib = np.random.rand(*shapes[in_name]).astype(np.float32)
        log("ВНИМАНИЕ: калибровка на случайных данных (как в compile.py SDK) — точность не гарантирована")
        manifest["calibration"] = {"dir": None, "n": 0, "random": True}
    calib_dict = {in_name: calib}

    regular = RegularModel(model=m_opt)
    thr_path = out_dir / "thresholds.json"
    t0 = time.time()
    if a.thresholds:
        thresholds = RegularModel.load_thresholds(a.thresholds)
        log(f"thresholds из {a.thresholds}")
    else:
        kw = {}
        if a.activation_percentile is not None:
            kw["activation_percentile"] = a.activation_percentile
        if a.calib_method == "torch":
            if a.samplewise_percentile is not None:
                kw["samplewise_percentile"] = a.samplewise_percentile
            try:
                thresholds = regular.calibrate_torch(calibration_data=calib_dict, **kw)
            except KeyError:
                am = regular.anchors_mapping
                thresholds = regular.calibrate_torch(calibration_data={am[k]: v for k, v in calib_dict.items()}, **kw)
        else:
            am = regular.anchors_mapping
            thresholds = regular.calibrate({am[k]: v for k, v in calib_dict.items()}, **kw)
    RegularModel.dump_thresholds(str(thr_path), thresholds)
    log(f"калибровка {time.time() - t0:.1f} с -> {thr_path}")
    t0 = time.time()
    qm = regular.quantize(thresholds)
    qm_name = f"quant_{name}.qm.onnx"
    qm.save(file_dir=str(out_dir), file_name=qm_name)
    log(f"квантование {time.time() - t0:.1f} с -> {qm_name}")
    manifest["quant_model"] = str(out_dir / qm_name)

    # --- сверка квантованной модели с fp32 (на CPU, без TPU)
    if not a.no_validate:
        try:
            vimg = a.validate_image or (files[0] if files else None)
            x = preprocess(vimg, a.preproc, hwsz) if vimg else calib[:1]
            if batch > 1:
                x = np.repeat(x, batch, axis=0)
            ref = ort_run(opt_path, {in_name: x})
            got = run_quantized(qm, {in_name: x}, out_names, batch, cwl)
            val = {n: compare(ref[n], got[n]) for n in out_names if n in got}
            if cut and set(got) == set(out_names):
                ref_full = ort_run(onnx_path, {in_name: x})
                tail_out = ort_run(Path(manifest["tail_onnx"]), {n: got[n].astype(np.float32) for n in out_names})
                for n in manifest["orig_outputs"]:
                    val[f"tail:{n}"] = compare(ref_full[n], tail_out[n])
            manifest["validation_quant_vs_fp32"] = {"image": vimg, "metrics": val}
            for k, v in val.items():
                log(f"сверка quant vs fp32 [{k}]: cos={v['cosine']} max|d|={v['max_abs_diff']:.4g} "
                    f"(|ref|max={v['ref_absmax']:.4g})")
        except Exception as e:  # noqa: BLE001
            log(f"сверка не удалась: {e!r}")
            traceback.print_exc()
            manifest["validation_quant_vs_fp32"] = {"error": repr(e)}

    # --- 3. компиляция (pdf стр. 16-17, 30-31)
    log("3/3 компиляция")
    qloaded = QuantizedModel.load(str(out_dir / qm_name))
    network, _ = Network.from_quantized_model(qloaded)
    tried = [a.preset] + ([a.fallback] if a.fallback and a.fallback != a.preset else [])
    program_path = None
    for p in tried:
        preset = getattr(presets, p, None) or getattr(tpu_framework, p)
        t0 = time.time()
        try:
            log(f"compile_ preset={p} platform={a.platform}")
            executable, tensor_descriptions = compiler.compile_(hardware_parameters=hw, network=network,
                                                                parameters=preset)
            prog = TpuProgram.from_executable(executable, tensor_descriptions)
            program_path = out_dir / f"{name}_b{batch}_{p.lower()}_{a.platform}.tpu"
            prog.to_file(str(program_path))
            manifest["compile"] = {"preset": p, "seconds": round(time.time() - t0, 1)}
            log(f"OK {p}: {program_path} ({program_path.stat().st_size / 1e6:.1f} MB, {time.time() - t0:.1f} с)")
            break
        except Exception as e:  # noqa: BLE001
            log(f"ОШИБКА компиляции {p} за {time.time() - t0:.1f} с: {e!r}")
            traceback.print_exc()
            manifest.setdefault("compile_errors", {})[p] = repr(e)
    if program_path is None:
        manifest["status"] = "failed"
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False, default=str))
        return 1

    manifest["program"] = str(program_path)
    manifest["program_sha256"] = sha256(program_path)
    manifest["program_meta"] = program_summary(program_path)
    manifest["status"] = "ok"
    manifest["total_seconds"] = round(time.time() - t_all, 1)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False, default=str))
    for n, t in manifest["program_meta"]["inputs"].items():
        log(f"вход  {n}: user {t['user_shape']} {t['user_dtype']} -> tpu {t['tpu_shape']} {t['tpu_order']} "
            f"{t['tpu_dtype']} size={t['size']} scales={t['scales'][:2]}")
    for n, t in manifest["program_meta"]["outputs"].items():
        log(f"выход {n}: tpu {t['tpu_shape']} {t['tpu_order']} {t['tpu_dtype']} size={t['size']} -> "
            f"user {t['user_shape']} {t['user_dtype']} scales={t['scales'][:2]}")
    log(f"=== готово за {manifest['total_seconds']} с: {program_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
