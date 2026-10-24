"""Tiny strict-state/contract tests; never imports RF-DETR or loads real weights."""

import argparse
import copy
import json
import socket
import os
import select
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tools import prepare_rfdetr_candidate as candidate


class Tiny(torch.nn.Module):
    def __init__(self, shape=(0, 0), enabled=False):
        super().__init__()
        self.class_embed = torch.nn.Linear(512, 91)
        self.extra = torch.nn.Parameter(torch.arange(3, dtype=torch.float32))
        self.register_buffer("_kp_active_mask", torch.zeros(shape, dtype=torch.bool))
        self.use_grouppose_keypoints = enabled
        self.num_keypoints_per_class = []


def state(model):
    return {key: value.clone() for key, value in model.state_dict().items() if key != "_kp_active_mask"}


def test_constructor_uses_90_plus_one_not_package365_and_disables_implicit_paths():
    values = candidate.constructor_kwargs()
    assert values["num_classes"] + 1 == 91
    assert values["pretrain_weights"] is None and values["backbone_lora"] is False
    assert values["device"] == "cpu" and values["amp"] is False and values["compile"] is False
    assert values["resolution"] == 880 and values["positional_encoding_size"] == 44
    assert values["num_queries"] == 300 and values["group_detr"] == 13


def test_strict_load_preserves_every_checkpoint_tensor_and_reports_only_empty_derived_buffer():
    source = Tiny()
    checkpoint = state(source)
    target = Tiny()
    report = candidate.strict_load(target, checkpoint, torch)
    assert report["strict"] and report["checkpoint_keys"] == 3 and report["model_state_keys"] == 4
    assert report["checkpoint_numel"] == sum(tensor.numel() for tensor in checkpoint.values())
    assert report["derived_stateless_buffers"]["_kp_active_mask"]["shape"] == [0, 0]
    assert report["derived_stateless_buffers"]["_kp_active_mask"]["numel"] == 0
    assert all(torch.equal(target.state_dict()[key], value) for key, value in checkpoint.items())


@pytest.mark.parametrize("change,error", [("missing", "state keys"), ("extra", "state keys"), ("shape", "shape/dtype"),
                                          ("dtype", "shape/dtype"), ("wrong_head", "shape/dtype")])
def test_missing_learned_weight_extra_tensor_shape_dtype_or_head_mismatch_is_never_ignored(change, error):
    target = Tiny()
    checkpoint = state(target)
    if change == "missing": del checkpoint["extra"]
    if change == "extra": checkpoint["unrecognized"] = torch.ones(1)
    if change == "shape": checkpoint["extra"] = torch.ones(4)
    if change == "dtype": checkpoint["extra"] = checkpoint["extra"].double()
    if change == "wrong_head": checkpoint["class_embed.weight"] = torch.zeros(365, 512)
    with pytest.raises(ValueError, match=error):
        candidate.strict_load(target, checkpoint, torch)


@pytest.mark.parametrize("shape,enabled", [((1, 1), False), ((0, 0), True)])
def test_nonempty_or_enabled_keypoint_buffer_cannot_be_derived(shape, enabled):
    target = Tiny(shape=shape, enabled=enabled)
    with pytest.raises(ValueError, match="empty bool"):
        candidate.strict_load(target, state(target), torch)


def test_existing_empty_checkpoint_buffer_is_loaded_without_synthesizing_state():
    target = Tiny()
    report = candidate.strict_load(target, copy.deepcopy(target.state_dict()), torch)
    assert report["checkpoint_keys"] == report["model_state_keys"] == 4
    assert report["derived_stateless_buffers"] == {}


def test_weights_only_load_has_one_explicit_allowlist_and_no_unsafe_retry(tmp_path, monkeypatch):
    model = Tiny()
    checkpoint = state(model)
    path = tmp_path / "tiny.pth"
    torch.save({"model": checkpoint, "license": argparse.Namespace(name="synthetic")}, path)
    monkeypatch.setattr(candidate, "WEIGHTS_SHA", candidate.sha(path))
    monkeypatch.setattr(candidate, "WEIGHTS_BYTES", path.stat().st_size)
    monkeypatch.setattr(candidate, "CHECKPOINT_KEYS", len(checkpoint))
    monkeypatch.setattr(candidate, "CHECKPOINT_NUMEL", sum(value.numel() for value in checkpoint.values()))
    loaded, _ = candidate.restricted_checkpoint(path, torch)
    assert set(loaded) == set(checkpoint)
    calls = []
    def failed(*args, **kwargs):
        calls.append(kwargs)
        raise RuntimeError("strict loader refused")
    monkeypatch.setattr(torch, "load", failed)
    with pytest.raises(RuntimeError, match="refused"):
        candidate.restricted_checkpoint(path, torch)
    assert calls == [{"weights_only": True, "mmap": True, "map_location": "cpu"}]


def test_sparse_coco_keeps_class90_and_does_not_remap_person_to_yolo_zero():
    names = {index: f"class-{index}" for index in candidate.COCO_IDS}
    names.update({1: "person", 77: "cell phone", 90: "toothbrush"})
    layout = candidate.make_class_layout(names)
    assert layout["logit_count"] == 91 and layout["background_class_id"] is None
    assert layout["logit_index_to_name"][0] is None and layout["logit_index_to_name"][12] is None
    assert layout["logit_index_to_name"][1] == "person" and layout["logit_index_to_name"][90] == "toothbrush"
    with pytest.raises(ValueError, match="mapping"):
        candidate.make_class_layout(dict(enumerate(names.values())))


def test_source_preprocessing_is_rgb_float_bilinear_without_antialias_or_padding():
    # Downsampling4x4→1 without AA samples the central2x2, not the whole image.
    rgb = np.zeros((4, 4, 3), np.uint8)
    rgb[1:3, 1:3] = [255, 128, 0]
    original = rgb.copy()
    actual = candidate.preprocess_rgb(rgb, size=1).numpy()
    expected = (np.array([1., 128 / 255, 0.], np.float32) - candidate.PREPROCESS["mean"]) / candidate.PREPROCESS["std"]
    np.testing.assert_allclose(actual[0, :, 0, 0], expected, atol=1e-6)
    np.testing.assert_array_equal(rgb, original)
    assert actual.shape == (1, 3, 1, 1) and actual.dtype == np.float32


def test_bootstrap_plan_never_changes_root_env_and_refuses_protected_replacement(tmp_path):
    shared = tmp_path / "shared"
    shared.mkdir()
    report = candidate.env_plan(tmp_path / "new-env", shared, tmp_path / "plan")
    text = (tmp_path / "plan/requirements.in").read_text()
    assert candidate.SOURCE_REVISION in text and candidate.SOURCE_SHA in text and candidate.PLUS_SHA in text
    assert report["root_environment_mutation"] is False
    candidate.validate_install_plan({"install": [{"metadata": {"name": "transformers"}}]})
    with pytest.raises(ValueError, match="protected"):
        candidate.validate_install_plan({"install": [{"metadata": {"name": "torch"}}]})
    with pytest.raises(ValueError, match="root .venv"):
        candidate.env_plan(candidate.ROOT / ".venv", shared, tmp_path / "bad")


def test_offline_constructor_guard_blocks_network_instead_of_auto_downloading():
    with candidate.offline_network(), socket.socket() as connection:
        with pytest.raises(RuntimeError, match="Network access"):
            connection.connect(("127.0.0.1", 1))


@pytest.mark.parametrize("change", ["nan", "inf", "head80", "batch2", "dtype"])
def test_parity_contract_refuses_invalid_outputs(change):
    values = [np.zeros((1, 300, 4), np.float32), np.zeros((1, 300, 91), np.float32)]
    if change == "nan": values[0][0, 0, 0] = np.nan
    if change == "inf": values[1][0, 299, 90] = np.inf
    if change == "head80": values[1] = values[1][:, :, :80]
    if change == "batch2": values[0] = np.repeat(values[0], 2, axis=0)
    if change == "dtype": values[1] = values[1].astype(np.float16)
    with pytest.raises(ValueError, match="shape/dtype/finite"):
        candidate.validate_outputs(values)


@pytest.mark.parametrize("size,head,opset,ok", [(880, 91, 17, True), (640, 91, 17, False),
                                            (880, 80, 17, False), (880, 91, 18, False)])
def test_static_graph_manifest_records_actual_io_and_rejects_wrong_contract(tmp_path, size, head, opset, ok):
    import onnx
    from onnx import helper, numpy_helper, TensorProto
    boxes = numpy_helper.from_array(np.zeros((1, 300, 4), np.float32), name="boxes")
    labels = numpy_helper.from_array(np.zeros((1, 300, head), np.float32), name="classes")
    graph = helper.make_graph([helper.make_node("Identity", ["boxes"], ["dets"]), helper.make_node("Identity", ["classes"], ["labels"])],
                             "tiny-static", [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, size, size])],
                             [helper.make_tensor_value_info("dets", TensorProto.FLOAT, [1, 300, 4]),
                              helper.make_tensor_value_info("labels", TensorProto.FLOAT, [1, 300, head])], [boxes, labels])
    path = tmp_path / "tiny.onnx"
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)], ir_version=10), path)
    if ok:
        result = candidate.inspect_graph(path)
        assert result["checker"] == "passed" and result["operators"] == {"ai.onnx:Identity": 2}
        assert result["target_tpu"] == "not_compiled_or_verified" and result["outputs"]["labels"][-1] == 91
    else:
        with pytest.raises(ValueError):
            candidate.inspect_graph(path)


@pytest.mark.parametrize("reason", ["wall_timeout", "memory_limit"])
def test_watchdog_overrides_partial_success_and_reaps_owned_group(tmp_path, monkeypatch, reason):
    args = SimpleNamespace(output_dir=tmp_path / "export", timeout=1., max_rss_gb=1.)
    class Process:
        pid, returncode = 987654321, None
        def poll(self): return self.returncode
        def wait(self, timeout=None):
            self.returncode = -15
            return self.returncode
    process = Process()
    def spawn(*args_, **kwargs):
        candidate.write(args.output_dir / "manifest.json", {"ok": True, "status": "partial-success"})
        return process
    monkeypatch.setattr(candidate.subprocess, "Popen", spawn)
    clock = iter([0., 2., 3.] if reason == "wall_timeout" else [0., 0., 1.])
    monkeypatch.setattr(candidate.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(candidate, "group_rss_kib", lambda pgid: 2 * 1024 * 1024)
    signals = []
    monkeypatch.setattr(candidate.os, "killpg", lambda pid, sig: signals.append((pid, sig)))
    assert candidate.bounded_run(args, ["export"]) == 1
    report = json.loads((args.output_dir / "manifest.json").read_text())
    assert report["ok"] is False and report["error"] == reason and report["owned_process_reaped"] is True
    assert {sig for _, sig in signals} == {candidate.signal.SIGTERM, candidate.signal.SIGKILL}


@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGINT])
def test_parent_signal_reaps_real_worker_and_descendant_before_interrupted_report(tmp_path, signum):
    output, ready, child_ready = tmp_path / "export", tmp_path / "ready", tmp_path / "child-ready"
    read_fd, write_fd = os.pipe()
    child = f'''
import os,signal,time
from pathlib import Path
signal.signal(signal.SIGTERM,signal.SIG_IGN)
os.fstat({write_fd})
Path({str(child_ready)!r}).write_text(str(os.getpid()))
time.sleep(30)
Path({str(output / 'manifest.json')!r}).write_text('{{"ok":true,"status":"late_success"}}')
'''
    worker = f'''
import os,signal,subprocess,sys,time
from pathlib import Path
signal.signal(signal.SIGTERM,signal.SIG_IGN)
subprocess.Popen([sys.executable,'-c',{child!r}],pass_fds=({write_fd},))
while not Path({str(child_ready)!r}).exists():time.sleep(.01)
Path({str(ready)!r}).write_text(str(os.getpgrp()))
time.sleep(30)
'''
    parent = f'''
import subprocess,sys
from tools import prepare_rfdetr_candidate as candidate
original = subprocess.Popen
def owned(command,**kwargs):
    kwargs['pass_fds']=({write_fd},)
    return original([sys.executable,'-c',{worker!r}],**kwargs)
candidate.subprocess.Popen=owned
candidate.group_rss_kib=lambda pgid:0
raise SystemExit(candidate.main(sys.argv[1:]))
'''
    process = subprocess.Popen([sys.executable, "-c", parent, "export", "--weights", "unused.pth", "--output-dir", str(output),
                                "--timeout", "20"], cwd=candidate.ROOT, pass_fds=(write_fd,), start_new_session=True,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    os.close(write_fd)
    stopped = False
    try:
        deadline = time.monotonic() + 5
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(.01)
        assert ready.exists() and not select.select([read_fd], [], [], 0)[0]
        os.kill(process.pid, signum)
        time.sleep(.05)
        os.kill(process.pid, signal.SIGINT)  # Repeated cancellation during cleanup cannot strand descendants.
        _, stderr = process.communicate(timeout=8)
        assert process.returncode == 128 + signum, stderr
        report = json.loads((output / "manifest.json").read_text())
        assert not report["ok"] and report["status"] == "interrupted" and report["interrupt_signal"] == signum
        assert report["owned_process_reaped"] is True
        assert select.select([read_fd], [], [], 2)[0], "Owned worker/descendant survived parent interruption"
        assert os.read(read_fd, 1) == b""
        stopped = True
    finally:
        if not stopped and ready.exists():
            try: os.killpg(int(ready.read_text()), signal.SIGKILL)
            except ProcessLookupError: pass
        if process.poll() is None: process.kill()
        process.communicate(timeout=2)
        os.close(read_fd)


@pytest.mark.parametrize("mutate", [False, True])
def test_parity_reports_only_pinned_session_graph_and_rejects_midrun_swap(tmp_path, monkeypatch, mutate):
    from PIL import Image
    export_dir, output = tmp_path / "export", tmp_path / "parity"
    export_dir.mkdir()
    output.mkdir()
    graph = export_dir / "model.onnx"
    graph.write_bytes(b"pinned tiny fixture graph; real ORT is mocked")
    original = candidate.asset(graph)
    candidate.write(export_dir / "manifest.json", {"ok": True, "source_revision": candidate.SOURCE_REVISION, "onnx": original})
    weights = tmp_path / "weights.pth"
    weights.write_bytes(b"synthetic; strict bootstrap is mocked")
    image = tmp_path / "real.png"
    Image.new("RGB", (2, 3)).save(image)
    values = [np.zeros((1, 300, 4), np.float32), np.zeros((1, 300, 91), np.float32)]
    class Native:
        def __call__(self, tensor):
            return {"pred_boxes": torch.from_numpy(values[0]), "pred_logits": torch.from_numpy(values[1])}
    monkeypatch.setattr(candidate, "build_native", lambda path: (Native(), {"weights": candidate.asset(path)}))
    monkeypatch.setattr(candidate, "inspect_graph", lambda path: {})
    monkeypatch.setattr(candidate, "preprocess_rgb", lambda rgb: torch.zeros((1, 3, 2, 2)))
    class Session:
        def __init__(self, *args, **kwargs): pass
        def get_providers(self): return ["CPUExecutionProvider"]
        def run(self, *args):
            if mutate: graph.write_bytes(b"different graph not held by the running session")
            return values
    monkeypatch.setitem(sys.modules, "onnxruntime", SimpleNamespace(SessionOptions=SimpleNamespace, InferenceSession=Session))
    args = SimpleNamespace(export_dir=export_dir, output_dir=output, weights=weights, image=[image])
    if mutate:
        with pytest.raises(ValueError, match="changed during parity"):
            candidate.parity_stage(args)
    else:
        report = candidate.parity_stage(args)
        assert report["onnx"] == original and report["samples"] == 4
