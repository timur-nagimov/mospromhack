"""Real tiny FP32 ONNX graphs; no SDK, remote service or pretrained inference."""

import copy
import json
import os
import select
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import pytest

from tpu import preflight


def detect_graph(prefix="/model.22"):
    """Small nonconstant DFL detect graph with actual Conv/Softmax/classes."""
    initializers = []

    def const(name, value):
        initializers.append(numpy_helper.from_array(np.asarray(value), name))
        return name

    rng = np.random.default_rng(42)
    weights = const("backbone.weight", rng.normal(0, .3, (66, 3, 1, 1)).astype(np.float32))
    reshape = const("concat.shape", np.array([1, 66, 4], np.int64))
    splits = const("split.sizes", np.array([64, 2], np.int64))
    dfl_shape = const("dfl.shape", np.array([1, 4, 16, 4], np.int64))
    dfl_weight = const("dfl.weight", np.arange(16, dtype=np.float32).reshape(1, 16, 1, 1))
    boxes_shape = const("boxes.shape", np.array([1, 4, 4], np.int64))
    scale = const("scale", np.array(2, np.float32))
    box_cut, class_cut = prefix + "/dfl/conv/Conv_output_0", prefix + "/Sigmoid_output_0"
    nodes = [
        helper.make_node("Conv", ["images", weights], ["logits_4d"], name="/backbone/Conv"),
        helper.make_node("Reshape", ["logits_4d", reshape], ["logits"], name=prefix + "/Reshape"),
        helper.make_node("Split", ["logits", splits], ["box_logits", "class_logits"], axis=1, name=prefix + "/Split"),
        helper.make_node("Reshape", ["box_logits", dfl_shape], ["distribution"], name=prefix + "/dfl/Reshape"),
        helper.make_node("Transpose", ["distribution"], ["distribution_t"], perm=[0, 2, 1, 3], name=prefix + "/dfl/Transpose"),
        helper.make_node("Softmax", ["distribution_t"], ["probabilities"], axis=1, name=prefix + "/dfl/Softmax"),
        helper.make_node("Conv", ["probabilities", dfl_weight], [box_cut], name=prefix + "/dfl/conv/Conv"),
        helper.make_node("Sigmoid", ["class_logits"], [class_cut], name=prefix + "/Sigmoid"),
        helper.make_node("Reshape", [box_cut, boxes_shape], ["boxes"], name=prefix + "/tail/Reshape"),
        helper.make_node("Mul", ["boxes", scale], ["scaled_boxes"], name=prefix + "/tail/Mul"),
        helper.make_node("Concat", ["scaled_boxes", class_cut], ["output0"], axis=1, name=prefix + "/tail/Concat"),
    ]
    graph = helper.make_graph(nodes, "tiny_detect", [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 2, 2])],
        [helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, 6, 4])], initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 10
    return model, [box_cut, class_cut]


@pytest.fixture
def tiny(tmp_path):
    model, cut = detect_graph()
    path = tmp_path / "full.onnx"
    onnx.save(model, path)
    return path, model, cut


def args_for(path, output):
    output.mkdir()
    return SimpleNamespace(onnx=path, output_dir=output, cut=None, graph_only=False, image=[], atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("prefix", ["/model.22", "/model.23", "/detector.7"])
def test_graph_scoped_discovery_and_real_fp32_full_head_tail_parity(tmp_path, prefix):
    model, cut = detect_graph(prefix)
    path = tmp_path / "full.onnx"
    onnx.save(model, path)
    args = args_for(path, tmp_path / "result")
    assert preflight.run(args) == 0
    report = json.loads((args.output_dir / "report.json").read_text())
    assert report["status"] == "fp32_parity_passed" and report["head"]["cut"] == cut
    assert report["head"]["prefix"] == prefix
    assert report["tpu_compilation"] == "not_attempted" and report["tpu_compatibility"] == "unknown"
    assert report["dependency_check"]["original_input_dependencies"] == []
    assert report["operators"]["head"]["ai.onnx:Conv"] == 2
    assert "ai.onnx:Softmax" not in report["operators"]["tail"]
    cases = report["parity"]["cases"]
    assert [case["input"] for case in cases] == ["zeros", "ones", "uniform_seed_0"]
    assert len({case["input_tensor_sha256"] for case in cases}) == 3
    assert all(case["outputs"]["output0"]["elements"] == 24 for case in cases)
    assert all(case["outputs"]["output0"]["failing_elements"] == 0 for case in cases)
    assert report["parity"]["threads"] == {"intra": 2, "inter": 2}
    for name in ("source", "head", "tail"):
        assert report["assets"][name]["sha256"] == preflight.sha256(report["assets"][name]["path"])


def test_extra_original_dependency_is_rejected_before_extraction_or_ort(tiny, tmp_path, monkeypatch):
    path, model, _ = tiny
    # Valid ONNX bypasses both cuts through a scalar from the source input.
    model.graph.node[-1].output[0] = "original_prediction"
    model.graph.node.extend([
        helper.make_node("Reshape", ["images", "input.flat.shape"], ["input.flat"]),
        helper.make_node("Gather", ["input.flat", "first.index"], ["input.first"], axis=0),
        helper.make_node("Add", ["original_prediction", "input.first"], ["output0"]),
    ])
    model.graph.initializer.extend([
        numpy_helper.from_array(np.array([12], np.int64), "input.flat.shape"),
        numpy_helper.from_array(np.array(0, np.int64), "first.index"),
    ])
    onnx.save(model, path)
    args = args_for(path, tmp_path / "result")
    monkeypatch.setattr(preflight, "extract_pair", lambda *a: pytest.fail("Invalid cut must not be extracted"))
    assert preflight.run(args) == 1
    report = json.loads((args.output_dir / "report.json").read_text())
    assert "original input" in report["error"]["message"]


def test_missing_cut_and_unused_cut_are_rejected(tiny):
    _, model, cut = tiny
    with pytest.raises(ValueError, match="Missing cut"):
        preflight.validate_cut(model, [cut[0], "missing"])
    model.graph.node.append(helper.make_node("Identity", [cut[0]], ["unused"]))
    with pytest.raises(ValueError, match="both contribute"):
        preflight.validate_cut(model, ["output0", "unused"])
    # Existing but irrelevant backbone output cannot replace the class boundary.
    with pytest.raises(ValueError):
        preflight.validate_cut(model, [cut[0], "logits_4d"])


@pytest.mark.parametrize("change", ["duplicate_head", "wrong_split", "wrong_softmax_axis", "not_dfl", "dynamic", "nonfinite", "output_shape", "half", "custom"])
def test_ambiguous_unsupported_or_invalid_graphs_fail_closed(tiny, tmp_path, change):
    path, model, cut = tiny
    if change == "duplicate_head":
        other = copy.deepcopy(model.graph.node[6])
        other.output[0] = "duplicate_dfl"
        model.graph.node.insert(7, other)
    elif change == "wrong_split":
        model.graph.node[7].input[0] = "box_logits"
        model.graph.output[0].type.tensor_type.shape.dim[1].dim_value = 68
    elif change == "wrong_softmax_axis":
        model.graph.node[5].attribute[0].i = 2
    elif change == "not_dfl":
        model.graph.node[6].name = "/model.22/ordinary/Conv"
    elif change == "dynamic":
        dim = model.graph.input[0].type.tensor_type.shape.dim[2]
        dim.ClearField("dim_value")
        dim.dim_param = "height"
    elif change == "nonfinite":
        model.graph.initializer[0].CopyFrom(numpy_helper.from_array(np.full((66, 3, 1, 1), np.nan, np.float32), "backbone.weight"))
    elif change == "output_shape":
        model.graph.output[0].type.tensor_type.shape.dim[2].dim_value = 99
    elif change == "half":
        model.graph.input[0].type.tensor_type.elem_type = TensorProto.FLOAT16
    else:
        model.graph.node[0].domain = "custom"
    onnx.save(model, path)
    args = args_for(path, tmp_path / "result")
    assert preflight.run(args) == 1
    assert not json.loads((args.output_dir / "report.json").read_text())["ok"]


def test_tolerance_failure_is_not_hidden_and_nan_shape_dtype_fail():
    expected = {"output": np.array([0, 1, 100], dtype=np.float32)}
    actual = {"output": np.array([.01, 1, 100], dtype=np.float32)}
    result = preflight.compare_outputs(expected, actual, atol=1e-4, rtol=1e-4)
    assert not result["output"]["passed"] and result["output"]["failing_elements"] == 1
    for got, message in ((np.array([np.nan, 1, 100], np.float32), "Nonfinite"),
                         (np.zeros((1, 3), np.float32), "shape"), (np.zeros(3, np.float64), "FP32")):
        with pytest.raises(ValueError, match=message):
            preflight.compare_outputs(expected, {"output": got}, atol=1e-4, rtol=1e-4)


def test_changed_tail_fails_real_parity_and_keeps_difference_metrics(tiny, tmp_path, monkeypatch):
    path, _, _ = tiny
    original = preflight.extract_pair

    def corrupted(model, cut, output):
        head, tail, paths = original(model, cut, output)
        for tensor in tail.graph.initializer:
            if tensor.name == "scale":
                tensor.CopyFrom(numpy_helper.from_array(np.array(3, np.float32), "scale"))
        onnx.save(tail, paths[1])
        return head, tail, paths

    monkeypatch.setattr(preflight, "extract_pair", corrupted)
    args = args_for(path, tmp_path / "result")
    assert preflight.run(args) == 1
    report = json.loads((args.output_dir / "report.json").read_text())
    assert report["status"] == "error" and "tolerance failed" in report["error"]["message"]
    assert report["parity"]["cases"][0]["outputs"]["output0"]["max_abs_diff"] > 1


def test_optional_image_uses_reproducible_fp32_input_hash_and_source_hash(tmp_path):
    from PIL import Image

    path = tmp_path / "frame.png"
    Image.new("RGB", (3, 5), (10, 20, 30)).save(path)
    source = {"shape": [1, 3, 2, 2]}
    first, second = list(preflight.parity_inputs(source, [path])), list(preflight.parity_inputs(source, [path]))
    assert len(first) == 4 and first[-1][1].shape == (1, 3, 2, 2)
    assert first[-1][1].dtype == np.float32 and first[-1][2]["sha256"] == preflight.sha256(path)
    assert all(np.array_equal(a[1], b[1]) for a, b in zip(first, second))


def test_graph_only_cli_never_imports_ort_or_sdk_and_preserves_existing_results(tiny, tmp_path):
    path, _, _ = tiny
    output = tmp_path / "result"
    code = '''
import builtins, sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name in {"onnxruntime", "tpu_framework", "torch", "dnn_quant.api"}:
        raise AssertionError("Unexpected inference/SDK import")
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from tpu import preflight
raise SystemExit(preflight.run(preflight.parser().parse_args(sys.argv[1:])))
'''
    output.mkdir()
    result = subprocess.run([sys.executable, "-c", code, "--onnx", str(path), "--output-dir", str(output), "--graph-only"],
                            text=True, capture_output=True, timeout=15)
    assert result.returncode == 0, result.stderr
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "graph_validated" and report["parity"]["status"] == "skipped"
    completed = subprocess.run([sys.executable, "-m", "tpu.preflight", "--onnx", str(path), "--output-dir", str(tmp_path / "cli"), "--json"],
                               text=True, capture_output=True, timeout=15)
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["status"] == "fp32_parity_passed"
    again = subprocess.run([sys.executable, "-m", "tpu.preflight", "--onnx", str(path), "--output-dir", str(output)],
                           text=True, capture_output=True, timeout=15)
    assert again.returncode == 2 and json.loads((output / "report.json").read_text()) == report


@pytest.mark.parametrize("argument,value", [("--timeout", "1801"), ("--timeout", "0"), ("--atol", "nan"), ("--rtol", "-1")])
def test_invalid_parity_or_deadline_arguments_are_usage_errors(argument, value):
    with pytest.raises(SystemExit) as error:
        preflight.parser().parse_args(["--onnx", "model.onnx", argument, value])
    assert error.value.code == 2


def test_timeout_kills_owned_group_after_leader_exits(monkeypatch):
    signals, waits = [], []
    monkeypatch.setattr(preflight.os, "killpg", lambda pid, sig: signals.append((pid, sig)))
    monkeypatch.setattr(preflight.time, "sleep", lambda seconds: None)
    process = SimpleNamespace(pid=123456789, returncode=0, wait=lambda **kwargs: waits.append(True))
    preflight.terminate_group(process)
    assert signals == [(process.pid, signal.SIGTERM), (process.pid, signal.SIGKILL)] and waits == [True]


@pytest.mark.parametrize("leader_already_exited", [False, True])
def test_real_term_ignoring_descendant_is_killed_even_if_leader_has_exited(tmp_path, leader_already_exited):
    ready, term = tmp_path / "ready", tmp_path / "term"
    read_fd, write_fd = os.pipe()
    child = f'''
import os, signal, time
from pathlib import Path
signal.signal(signal.SIGTERM, lambda *args: Path({str(term)!r}).write_text("ignored"))
os.fstat({write_fd})
Path({str(ready)!r}).write_text("ready")
while True:
    time.sleep(.01)
'''
    leader = f'''
import subprocess, sys, time
from pathlib import Path
subprocess.Popen([sys.executable, "-c", {child!r}], pass_fds=({write_fd},))
while not Path({str(ready)!r}).exists():
    time.sleep(.01)
if {leader_already_exited!r}:
    raise SystemExit(0)
while True:
    time.sleep(.01)
'''
    process = subprocess.Popen([sys.executable, "-c", leader], start_new_session=True, pass_fds=(write_fd,),
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    os.close(write_fd)
    fully_stopped = False
    try:
        deadline = time.monotonic() + 5
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(.01)
        assert ready.exists(), "Own descendant did not start"
        if leader_already_exited:
            assert process.wait(timeout=2) == 0
        assert not select.select([read_fd], [], [], 0)[0]
        preflight.terminate_group(process)
        assert term.read_text() == "ignored"
        assert select.select([read_fd], [], [], 2)[0], "Own descendant survived group KILL"
        assert os.read(read_fd, 1) == b""  # EOF proves every inherited writer exited.
        fully_stopped = True
    finally:
        if not fully_stopped:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        process.wait(timeout=2)
        os.close(read_fd)


def test_cli_timeout_is_a_nonzero_json_result(monkeypatch, tmp_path, capsys):
    class Process:
        pid, returncode, calls = 123456789, None, 0

        def wait(self, timeout):
            self.calls += 1
            if self.calls == 1:
                raise subprocess.TimeoutExpired("worker", timeout)
            self.returncode = -signal.SIGKILL

    signals = []
    monkeypatch.setattr(preflight.subprocess, "Popen", lambda *a, **kw: Process())
    monkeypatch.setattr(preflight.os, "killpg", lambda pid, sig: signals.append(sig))
    monkeypatch.setattr(preflight.time, "sleep", lambda seconds: None)
    assert preflight.main(["--onnx", "missing.onnx", "--output-dir", str(tmp_path / "result"), "--timeout", "1", "--json"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "timeout" and not report["ok"]
    assert report["tpu_compilation"] == "not_attempted" and signals == [signal.SIGTERM, signal.SIGKILL]


def test_sdk_compiler_defaults_are_portable_without_importing_sdk(tmp_path):
    code = '''
import builtins, json
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name in {"onnxruntime", "tpu_framework", "torch", "dnn_quant.api"}:
        raise AssertionError("Unexpected inference/SDK import")
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from tpu import compile
print(json.dumps([str(compile.ROOT), str(compile.MODELS_DIR)]))
'''
    environment = {key: value for key, value in os.environ.items() if key not in {"GUARD_ROOT", "GUARD_MODELS_DIR"}}
    result = subprocess.run([sys.executable, "-c", code], env=environment, text=True, capture_output=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == [str(preflight.ROOT), str(preflight.ROOT / "models")]
    environment.update(GUARD_ROOT=str(tmp_path / "repo"), GUARD_MODELS_DIR=str(tmp_path / "assets"))
    result = subprocess.run([sys.executable, "-c", code], env=environment, text=True, capture_output=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == [str(tmp_path / "repo"), str(tmp_path / "assets")]


@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM])
def test_parent_interruption_cleans_real_worker_and_cannot_later_write_success(tmp_path, signum):
    ready, output = tmp_path / "ready", tmp_path / "result"
    read_fd, write_fd = os.pipe()
    worker = f'''
import json, os, signal, time
from pathlib import Path
signal.signal(signal.SIGTERM, signal.SIG_IGN)
os.fstat({write_fd})
Path({str(ready)!r}).write_text(str(os.getpid()))
time.sleep(30)
Path({str(output / "report.json")!r}).write_text(json.dumps({{"status": "fp32_parity_passed", "ok": True}}))
'''
    parent = f'''
import subprocess, sys
from tpu import preflight
original = subprocess.Popen
def start_owned(command, **kwargs):
    kwargs["pass_fds"] = ({write_fd},)
    return original([sys.executable, "-c", {worker!r}], **kwargs)
preflight.subprocess.Popen = start_owned
raise SystemExit(preflight.main(sys.argv[1:]))
'''
    process = subprocess.Popen([sys.executable, "-c", parent, "--onnx", "unused.onnx", "--output-dir", str(output), "--timeout", "20", "--json"],
        cwd=preflight.ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, pass_fds=(write_fd,), start_new_session=True)
    os.close(write_fd)
    fully_stopped = False
    try:
        deadline = time.monotonic() + 5
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(.01)
        assert ready.exists(), "Owned worker did not start"
        assert not select.select([read_fd], [], [], 0)[0]
        os.kill(process.pid, signum)
        stdout, stderr = process.communicate(timeout=8)
        assert process.returncode == 128 + signum, stderr
        report = json.loads(stdout)
        assert report["status"] == "interrupted" and not report["ok"] and report["interrupt_signal"] == signum
        assert json.loads((output / "report.json").read_text()) == report
        assert select.select([read_fd], [], [], 2)[0], "ORT-style worker survived parent interruption"
        assert os.read(read_fd, 1) == b""
        fully_stopped = True
    finally:
        if not fully_stopped and ready.exists():
            try:
                os.killpg(int(ready.read_text()), signal.SIGKILL)
            except OSError:
                pass
        if process.poll() is None:
            process.kill()
        process.communicate(timeout=2)
        os.close(read_fd)


def test_exception_after_spawn_cleans_group_and_restores_signal_handlers(monkeypatch, tmp_path, capsys):
    class Process:
        pid, returncode, calls = 123456789, None, 0

        def wait(self, timeout):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("wait failure")
            self.returncode = -signal.SIGKILL

    signals = []
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    monkeypatch.setattr(preflight.subprocess, "Popen", lambda *a, **kw: Process())
    monkeypatch.setattr(preflight.os, "killpg", lambda pid, sig: signals.append(sig))
    monkeypatch.setattr(preflight.time, "sleep", lambda seconds: None)
    assert preflight.main(["--onnx", "unused.onnx", "--output-dir", str(tmp_path / "result"), "--json"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "error" and report["error"]["message"] == "wait failure"
    assert signals == [signal.SIGTERM, signal.SIGKILL]
    assert previous == {sig: signal.getsignal(sig) for sig in previous}
