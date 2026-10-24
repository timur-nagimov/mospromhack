"""Opt-in RF-DETR 2XL strict CPU bootstrap/static export/parity (no runtime adapter).

``env-plan`` is light. ``export`` and ``parity`` are explicit bounded heavy
stages for an isolated environment; neither stage downloads model/backbone data.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
SOURCE_REVISION = "5b22dfc0940697ce1abdfa681a5f7392b3eddbfb"
SOURCE_SHA = "18e2fb180f176709070fedf03229fc5edabd3ade271d4c4fe9ca7dbefcd42368"
SOURCE_URL = f"https://codeload.github.com/roboflow/rf-detr/tar.gz/{SOURCE_REVISION}"
PLUS_SHA = "bf73c8b003c0327357947f50eb678efddceeafa7350e953b764f463f68573133"
PLUS_URL = "https://files.pythonhosted.org/packages/4d/70/d2be010436a065b3b42096da156e6fbd986fb5f5d1eaa1050147d72b01b2/rfdetr_plus-1.1.0-py3-none-any.whl"
WEIGHTS_SHA = "bf418652d5e07ad441599acfadc65bb1767029079e7a343a6d61675ec5d553ae"
WEIGHTS_BYTES = 507831201
CHECKPOINT_KEYS, CHECKPOINT_NUMEL = 531, 126903202
PROTECTED = {"torch": "2.8.0", "torchvision": "0.23.0", "numpy": "2.4.6", "onnx": "1.23.2", "onnxruntime": "1.31.0",
             "opencv-python": "4.11.0.86", "opencv-python-headless": "4.11.0.86"}
OPTIONAL = {"transformers": "5.1.0", "supervision": "0.29.1", "pyDeprecate": "0.9.0", "rfdetr-plus": "1.1.0"}
SOURCE_FILES = {
    "config.py": "5968e352e67bed9dabaaccfe0ee23caaa07569455e1eab47023d5622be9e5675",
    "models/lwdetr.py": "248faf15d673bc18f51f60bb55245f64eee40eb9c8adacea398aebdadad2243b",
    "models/_defaults.py": "57c129ab56d94f06a6f4fbf71abf5099b5f73a70970ad17ee7f2de9bf79715aa",
    "models/backbone/__init__.py": "f1df4341c4c8050139a430d0609656d40f0f3bf84d1774865dbc26dd48bd6ffa",
    "models/backbone/dinov2.py": "5c5c5b88cf09cab1f963427445bcf41d76d316707d9404d7491bf4feeb2b7040",
    "assets/coco_classes.py": "e839cf911b26ff3b363bb23e4d07861cf5bf85a5610b2025472b71b36941703e",
}
PLUS_CONFIG_SHA = "497852c400585763d07eb01ac0a14cd27196eae9cf2e823e51740a00f2f32b3b"
PREPROCESS = {"input": "RGB uint8", "resize": "torch bilinear", "align_corners": False, "antialias": False,
              "rescale": "float32 /255 before resize", "mean": [.485, .456, .406], "std": [.229, .224, .225],
              "shape": [1, 3, 880, 880], "padding": False}
COCO_IDS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
            27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53,
            54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
            82, 84, 85, 86, 87, 88, 89, 90)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def asset(path):
    path = Path(path).resolve()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)}


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def constructor_kwargs():
    # The pinned builder constructs args.num_classes + 1 logits.
    return {"num_classes": 90, "pretrain_weights": None, "device": "cpu", "amp": False, "compile": False,
            "resolution": 880, "positional_encoding_size": 44, "num_queries": 300, "group_detr": 13,
            "use_grouppose_keypoints": False, "num_keypoints_per_class": [], "segmentation_head": False,
            "backbone_lora": False, "gradient_checkpointing": False}


def validate_install_plan(report):
    names = {item["metadata"]["name"].lower().replace("_", "-") for item in report["install"]}
    require(not names.intersection(PROTECTED), "Dependency resolution would replace protected shared runtime libraries")


def env_plan(env: Path, reuse_site: Path, output: Path):
    require(env.resolve() != (ROOT / ".venv").resolve() and not env.exists(), "Use a NEW isolated env, never root .venv")
    require(reuse_site.is_dir(), "Shared read-only runtime site-packages is missing")
    output.mkdir(parents=True, exist_ok=False)
    constraints = output / "constraints.txt"
    constraints.write_text("".join(f"{name}=={version}\n" for name, version in PROTECTED.items()))
    requirements = output / "requirements.in"
    requirements.write_text(f"rfdetr @ {SOURCE_URL}#sha256={SOURCE_SHA}\n{PLUS_URL}#sha256={PLUS_SHA}\n"
                            + "".join(f"{name}=={version}\n" for name, version in OPTIONAL.items() if name != "rfdetr-plus"))
    python = env / "bin/python"
    pth_code = ("import pathlib,site; pathlib.Path(site.getsitepackages()[0],'root-runtime.pth').write_text("
                + repr(str(reuse_site.resolve()) + "\n") + ")")
    pip = [str(python), "-m", "pip", "install", "-c", str(constraints.resolve()), "-r", str(requirements.resolve())]
    plan = {"root_environment_mutation": False, "source_revision": SOURCE_REVISION,
            "commands": [[sys.executable, "-m", "venv", str(env)], [str(python), "-c", pth_code],
                         [*pip, "--dry-run", "--report", str(output / "resolution.json")],
                         [*pip, "--report", str(output / "installation.json")]],
            "gate": "validate_install_plan(resolution.json) must pass before installation; save pip freeze after install",
            "shared_read_only_site": str(reuse_site.resolve())}
    write(output / "environment-plan.json", plan)
    return plan


@contextlib.contextmanager
def offline_network():
    """Downloads during imports/construction/forward are errors, not fallbacks."""
    from unittest.mock import patch
    def blocked(*args, **kwargs):
        raise RuntimeError("Network access is disabled during RF-DETR export/parity")
    with patch.object(socket.socket, "connect", blocked), patch.object(socket.socket, "connect_ex", blocked), \
            patch.object(socket, "create_connection", blocked), \
            patch.dict(os.environ, {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "WANDB_MODE": "disabled"}):
        yield


def verify_environment():
    require(Path(sys.prefix).resolve() != (ROOT / ".venv").resolve() and sys.prefix != sys.base_prefix,
            "Export/parity must run in an isolated env; root .venv is read-only")
    versions = {name: importlib.metadata.version(name) for name in (*PROTECTED, *OPTIONAL, "rfdetr")}
    require(all(versions[name] == version for name, version in {**PROTECTED, **OPTIONAL}.items()), "Pinned dependency version drift")
    for name, expected in (("rfdetr", SOURCE_SHA), ("rfdetr-plus", PLUS_SHA)):
        raw = importlib.metadata.distribution(name).read_text("direct_url.json")
        require(raw is not None and json.loads(raw).get("archive_info", {}).get("hashes", {}).get("sha256") == expected,
                f"{name} was not installed from the pinned source archive/wheel")
    core = Path(importlib.metadata.distribution("rfdetr").locate_file("rfdetr"))
    plus = Path(importlib.metadata.distribution("rfdetr-plus").locate_file("rfdetr_plus"))
    for relative, expected in SOURCE_FILES.items():
        require(sha(core / relative) == expected, f"Installed RF source drift: {relative}")
    require(sha(plus / "models/detection.py") == PLUS_CONFIG_SHA, "Installed Plus config drift")
    return {"pinned": versions, "all_distributions": {dist.metadata["Name"]: dist.version for dist in importlib.metadata.distributions()}}


def restricted_checkpoint(path, torch):
    info = asset(path)
    require(info["bytes"] == WEIGHTS_BYTES and info["sha256"] == WEIGHTS_SHA, "RF-DETR checkpoint bytes/SHA256 mismatch")
    # This is the only additional allowlisted structure found in the pinned artifact.
    # There is deliberately no weights_only=False, pickle, trust or mmap fallback.
    with torch.serialization.safe_globals([argparse.Namespace]):
        checkpoint = torch.load(path, weights_only=True, mmap=True, map_location="cpu")
    require(isinstance(checkpoint, dict) and set(checkpoint) == {"model", "license"}, "Unexpected checkpoint container")
    state = checkpoint["model"]
    require(isinstance(state, dict) and len(state) == CHECKPOINT_KEYS
            and all(isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in state.items()), "Incomplete checkpoint tensor state")
    require(sum(value.numel() for value in state.values()) == CHECKPOINT_NUMEL, "Checkpoint tensor coverage/numel drift")
    return state, info


def strict_load(model, state, torch):
    """Every learned/nonempty tensor must come from the checkpoint, unchanged."""
    target = model.state_dict()
    derived = {}
    missing, unexpected = set(target) - set(state), set(state) - set(target)
    if missing == {"_kp_active_mask"}:
        value = target["_kp_active_mask"]
        require(not model.use_grouppose_keypoints and model.num_keypoints_per_class == []
                and tuple(value.shape) == (0, 0) and value.dtype == torch.bool and value.numel() == 0,
                "Only the documented disabled-keypoint empty bool[0,0] buffer may be derived")
        derived["_kp_active_mask"] = value.clone()
        missing.clear()
    require(not missing and not unexpected, f"Strict state keys mismatch: missing={sorted(missing)}, unexpected={sorted(unexpected)}")
    for key, value in state.items():
        require(tuple(value.shape) == tuple(target[key].shape) and value.dtype == target[key].dtype,
                f"Strict state shape/dtype mismatch: {key}")
    require(tuple(state["class_embed.weight"].shape) == (91, 512) and tuple(state["class_embed.bias"].shape) == (91,),
            "Actual head must be91x512, not package365 or contiguousCOCO80")
    report = model.load_state_dict({**state, **derived}, strict=True)
    require(not report.missing_keys and not report.unexpected_keys, "Strict load did not cover all state")
    loaded = model.state_dict()
    require(all(torch.equal(loaded[key], value) for key, value in state.items()), "Loaded model differs from checkpoint tensors")
    return {"strict": True, "checkpoint_keys": len(state), "checkpoint_numel": sum(value.numel() for value in state.values()),
            "model_state_keys": len(target), "derived_stateless_buffers": {key: {"shape": list(value.shape), "dtype": str(value.dtype),
              "numel": value.numel(), "reason": "pinned source disabled-keypoint deterministic empty buffer"} for key, value in derived.items()},
            "all_checkpoint_tensors_equal_after_load": True, "missing_keys": [], "unexpected_keys": []}


def build_native(weights):
    import torch
    from rfdetr.models._defaults import MODEL_DEFAULTS
    from rfdetr.models.lwdetr import build_model_from_config
    from rfdetr_plus.models.detection import RFDETR2XLargeConfig
    torch.set_num_threads(2)
    state, weights_info = restricted_checkpoint(weights, torch)
    config = RFDETR2XLargeConfig(**constructor_kwargs())
    require(config.encoder == "dinov2_windowed_base" and config.hidden_dim == 512 and config.dec_layers == 5
            and config.sa_nheads == 16 and config.ca_nheads == 32 and config.num_windows == 2 and config.patch_size == 20
            and config.projector_scale == ["P4"] and config.out_feature_indexes == [3, 6, 9, 12], "2XL architecture drift")
    defaults = dataclasses.replace(MODEL_DEFAULTS, force_no_pretrain=True, pretrained_encoder=None)
    model = build_model_from_config(config, defaults=defaults)
    coverage = strict_load(model, state, torch)
    del state
    model = model.to(device="cpu", dtype=torch.float32).eval()
    torch.set_num_threads(2)
    return model, {"weights": weights_info, "config": config.model_dump(mode="json"), "state_coverage": coverage,
                   "force_no_pretrain": True, "device": "cpu", "dtype": "float32"}


def class_layout():
    from rfdetr.assets.coco_classes import COCO_CLASSES
    return make_class_layout(COCO_CLASSES)


def make_class_layout(names):
    require(tuple(sorted(names)) == COCO_IDS and names[1] == "person" and names[77] == "cell phone"
            and names[90] == "toothbrush", "Pinned sparse COCO mapping drift")
    return {"logit_count": 91, "logit_index_to_name": [names.get(index) for index in range(91)],
            "valid_coco_category_ids": sorted(names), "background_class_id": None,
            "policy": "raw91 logits preserved; unnamed sparse slots stay unnamed; no last-logit removal or YOLO80 remap"}


def preprocess_rgb(rgb, size=880):
    import numpy as np
    import torch
    require(isinstance(rgb, np.ndarray) and rgb.dtype == np.uint8 and rgb.ndim == 3 and rgb.shape[2] == 3
            and min(rgb.shape[:2]) > 0, "Expected nonempty RGB uint8 HWC image")
    tensor = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).to(torch.float32).div(255).unsqueeze(0)
    tensor = torch.nn.functional.interpolate(tensor, size=(size, size), mode="bilinear", align_corners=False, antialias=False)
    mean, std = (torch.tensor(values, dtype=torch.float32).view(1, 3, 1, 1) for values in (PREPROCESS["mean"], PREPROCESS["std"]))
    return ((tensor - mean) / std).contiguous()


def raw_pair(value):
    if isinstance(value, dict):
        return value["pred_boxes"], value["pred_logits"]
    require(isinstance(value, (tuple, list)) and len(value) == 2, "Expected raw detection boxes/logits only")
    return tuple(value)


def validate_outputs(values):
    import numpy as np
    expected = ((1, 300, 4), (1, 300, 91))
    require(len(values) == 2, "Raw output count mismatch")
    for value, shape in zip(values, expected):
        require(value.shape == shape and value.dtype == np.float32 and np.isfinite(value).all(), "Raw output shape/dtype/finite mismatch")


def inspect_graph(path):
    import onnx
    model = onnx.load(path, load_external_data=False)
    require(not model.functions and all(node.domain in ("", "ai.onnx") for node in model.graph.node), "Custom operators/functions unsupported")
    require(not any(t.external_data or t.data_location == onnx.TensorProto.EXTERNAL for t in model.graph.initializer), "External weights unsupported")
    require(not any(t.data_type in (onnx.TensorProto.FLOAT16, onnx.TensorProto.BFLOAT16, onnx.TensorProto.DOUBLE)
                    for t in model.graph.initializer), "Floating model initializers must be FP32")
    onnx.checker.check_model(model, full_check=True)
    values = [*model.graph.input, *model.graph.output]
    shapes = [[dim.dim_value for dim in value.type.tensor_type.shape.dim] for value in values]
    require([value.name for value in values] == ["images", "dets", "labels"] and shapes == [[1, 3, 880, 880], [1, 300, 4], [1, 300, 91]],
            "Strict static graph IO contract mismatch")
    require(all(value.type.tensor_type.elem_type == onnx.TensorProto.FLOAT for value in values), "Graph IO must be FP32")
    require({item.domain: item.version for item in model.opset_import} == {"": 17}, "Graph must use standard opset17")
    require(not any(node.op_type == "NonMaxSuppression" for node in model.graph.node), "Expected raw graph without NMS")
    return {"checker": "passed", "inputs": {"images": shapes[0]}, "outputs": {"dets": shapes[1], "labels": shapes[2]},
            "opset": 17, "operators": dict(sorted(Counter((node.domain or "ai.onnx") + ":" + node.op_type for node in model.graph.node).items())),
            "nodes": len(model.graph.node), "initializer_count": len(model.graph.initializer), "external_data": False,
            "custom_operators": False, "target_tpu": "not_compiled_or_verified"}


def export_stage(args):
    import torch
    model, provenance = build_native(args.weights)
    class Raw(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner
        def forward(self, images):
            return self.inner(images)
    # Source export conversion precomputes positional encodings on the fixed grid.
    model.export()
    wrapper = Raw(model).eval()
    image = torch.zeros((1, 3, 880, 880), dtype=torch.float32)
    path = args.output_dir / "model.onnx"
    with torch.inference_mode():
        torch.onnx.export(wrapper, image, path, input_names=["images"], output_names=["dets", "labels"],
                          opset_version=17, dynamo=False, dynamic_axes=None, external_data=False,
                          export_params=True, keep_initializers_as_inputs=False, do_constant_folding=True)
    inspection = inspect_graph(path)
    require(asset(args.weights) == provenance["weights"], "Weights changed during export")
    return {**provenance, "onnx": asset(path), "inspection": inspection, "class_layout": class_layout(),
            "preprocessing": PREPROCESS, "status": "exported_not_parity_verified", "parity": "required_separate_stage"}


def parity_stage(args):
    import numpy as np
    import torch
    import onnxruntime as ort
    from PIL import Image
    export = json.loads((args.export_dir / "manifest.json").read_text())
    require(export["ok"] is True and export["source_revision"] == SOURCE_REVISION, "Export manifest is incomplete/unpinned")
    path = args.export_dir / "model.onnx"
    original_onnx = asset(path)
    require(original_onnx["sha256"] == export["onnx"]["sha256"], "Exported ONNX bytes changed")
    inspect_graph(path)
    model, provenance = build_native(args.weights)
    options = ort.SessionOptions()
    options.intra_op_num_threads, options.inter_op_num_threads = 2, 1
    session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    require(session.get_providers() == ["CPUExecutionProvider"], "Parity must use CPU only")
    samples = [("zeros", np.zeros((480, 640, 3), np.uint8), None),
               ("white", np.full((333, 517, 3), 255, np.uint8), None),
               ("seeded_rgb", np.random.default_rng(20261024).integers(0, 256, (721, 1279, 3), dtype=np.uint8), None)]
    for index, image in enumerate(args.image):
        with Image.open(image) as opened:
            samples.append((f"real-{index}", np.asarray(opened.convert("RGB")).copy(), asset(image)))
    rows = []
    with torch.inference_mode():
        for name, rgb, source in samples:
            tensor = preprocess_rgb(rgb)
            native = [value.detach().cpu().numpy() for value in raw_pair(model(tensor))]
            converted = session.run(["dets", "labels"], {"images": tensor.numpy()})
            validate_outputs(native)
            validate_outputs(converted)
            comparisons = []
            for output_name, left, right in zip(("dets", "labels"), native, converted):
                delta = np.abs(left.astype(np.float64) - right.astype(np.float64))
                comparisons.append({"output": output_name, "max_abs": float(delta.max()), "mean_abs": float(delta.mean()),
                    "native_sha256": hashlib.sha256(left.tobytes()).hexdigest(), "ort_sha256": hashlib.sha256(right.tobytes()).hexdigest(),
                    "matched": bool(np.allclose(left, right, atol=1e-4, rtol=1e-3))})
            rows.append({"name": name, "source": source, "rgb_shape": list(rgb.shape),
                         "input_sha256": hashlib.sha256(tensor.numpy().tobytes()).hexdigest(), "outputs": comparisons})
    write(args.output_dir / "comparisons.json", rows)
    require(asset(path) == original_onnx, "ONNX asset changed during parity; the session may hold earlier bytes")
    require(asset(args.weights) == provenance["weights"], "Checkpoint asset changed during parity")
    require(all(item["matched"] for row in rows for item in row["outputs"]), "Native/ORT parity failed; preserve evidence, do not loosen tolerances automatically")
    return {**provenance, "onnx": original_onnx, "status": "native_ort_parity_passed", "samples": len(rows),
            "atol": 1e-4, "rtol": 1e-3, "synthetic_inputs": 3, "real_inputs": len(args.image), "comparisons": rows,
            "target_tpu": "not_compiled_or_verified", "quality_evidence": False}


def worker(args):
    manifest = {"schema_version": 1, "ok": False, "status": "running", "stage": args.command,
                "source_revision": SOURCE_REVISION, "source_archive_sha256": SOURCE_SHA, "plus_wheel_sha256": PLUS_SHA,
                "tool": asset(__file__), "target_tpu": "not_compiled_or_verified", "production_defaults_changed": False}
    try:
        with offline_network():
            manifest["versions"] = verify_environment()
            manifest.update(export_stage(args) if args.command == "export" else parity_stage(args))
        manifest["ok"] = True
    except Exception as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    write(args.output_dir / "manifest.json", manifest)
    return 0 if manifest["ok"] else 1


def group_rss_kib(pgid):
    text = subprocess.check_output(["ps", "-axo", "pgid=,rss="], text=True)
    return sum(int(fields[1]) for line in text.splitlines() if len(fields := line.split()) == 2 and int(fields[0]) == pgid)


def bounded_run(args, argv, *, module="tools.prepare_rfdetr_candidate"):
    require(not args.output_dir.exists(), "Output exists; artifacts are never overwritten")
    require(0 < args.timeout <= 3600 and 0 < args.max_rss_gb <= 64, "Invalid wall/memory limits")
    args.output_dir.mkdir(parents=True)
    command = [sys.executable, "-m", module, *argv, "--_worker"]
    started, reason, process, pending_signal, cleanup_error = time.monotonic(), None, None, None, None
    def interrupted(signum, frame):
        nonlocal pending_signal
        # Record, never raise: Popen must return ownership before interruption is
        # acted on, and repeated signals must not interrupt TERM→KILL/readback.
        pending_signal = pending_signal or signum
    previous = {signum: signal.signal(signum, interrupted) for signum in (signal.SIGINT, signal.SIGTERM)}
    try:
        try:
            with (args.output_dir / "runtime.log").open("w") as log:
                if pending_signal is None:
                    process = subprocess.Popen(command, cwd=ROOT, start_new_session=True, stdout=log, stderr=log)
                while process is not None and process.poll() is None:
                    if pending_signal is not None:
                        break
                    if time.monotonic() - started > args.timeout:
                        reason = "wall_timeout"
                    elif group_rss_kib(process.pid) > args.max_rss_gb * 1024 * 1024:
                        reason = "memory_limit"
                    if reason:
                        break
                    time.sleep(.25)
        except KeyboardInterrupt:
            pending_signal = pending_signal or signal.SIGINT
        except Exception as exc:
            reason = f"worker_monitor_error: {type(exc).__name__}: {exc}"
        finally:
            if process is not None:
                try:
                    for sig in (signal.SIGTERM, signal.SIGKILL):
                        try:
                            os.killpg(process.pid, sig)
                        except ProcessLookupError:
                            pass
                        if sig == signal.SIGTERM:
                            time.sleep(.2)
                    process.wait(timeout=5)
                except Exception as exc:
                    cleanup_error = f"{type(exc).__name__}: {exc}"
        # Cleanup is finished. Freeze its cancellation outcome while committing
        # the report; a repeated signal cannot interrupt a failed-manifest write.
        for signum in previous:
            signal.signal(signum, signal.SIG_IGN)
        manifest_path = args.output_dir / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"ok": False}
        except (OSError, ValueError):
            manifest = {"ok": False, "error": "worker manifest missing or malformed"}
        returncode = process.returncode if process is not None else None
        if pending_signal is not None:
            manifest.update(ok=False, status="interrupted", error="parent_signal", interrupt_signal=pending_signal)
        elif reason or returncode or not manifest.get("ok"):
            manifest.update(ok=False, status="failed", error=reason or manifest.get("error", "worker failed; see runtime.log"))
        if cleanup_error:
            manifest.update(ok=False, status="failed", cleanup_error=cleanup_error)
        manifest.update(wall_s=time.monotonic() - started, returncode=returncode,
                        limits={"timeout_s": args.timeout, "max_rss_gb": args.max_rss_gb},
                        owned_process_reaped=process is not None and returncode is not None and cleanup_error is None)
        write(manifest_path, manifest)
        if pending_signal is not None:
            return 128 + pending_signal
        return 0 if manifest["ok"] else 1
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("env-plan")
    plan.add_argument("--env-dir", type=Path, required=True)
    plan.add_argument("--reuse-site", type=Path, required=True)
    plan.add_argument("--output-dir", type=Path, required=True)
    check = sub.add_parser("check-install-plan")
    check.add_argument("report", type=Path)
    for name in ("export", "parity"):
        child = sub.add_parser(name)
        child.add_argument("--weights", type=Path, required=True)
        child.add_argument("--output-dir", type=Path, required=True)
        child.add_argument("--timeout", type=float, default=900.)
        child.add_argument("--max-rss-gb", type=float, default=12.)
        child.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
        if name == "parity":
            child.add_argument("--export-dir", type=Path, required=True)
            child.add_argument("--image", type=Path, action="append", required=True)
    arguments = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(arguments)
    if args.command == "env-plan":
        env_plan(args.env_dir, args.reuse_site, args.output_dir)
        return 0
    if args.command == "check-install-plan":
        validate_install_plan(json.loads(args.report.read_text()))
        return 0
    if args._worker:
        return worker(args)
    return bounded_run(args, arguments)


if __name__ == "__main__":
    raise SystemExit(main())
