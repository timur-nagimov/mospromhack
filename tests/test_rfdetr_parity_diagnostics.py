"""Tiny diagnosis regressions; no RF package/weights, export, or ORT execution."""

import json
from pathlib import Path
import shutil
from types import SimpleNamespace
import sys

import numpy as np
import pytest
import torch

from tools import diagnose_rfdetr_candidate as diagnostic
from tools import prepare_rfdetr_candidate as source


def selector(scores, k):
    indices = torch.topk(scores, k, dim=1)[1]
    return lambda rows: torch.gather(rows, 1, indices.unsqueeze(-1).expand(-1, -1, rows.shape[-1]))


class Tiny:
    def __init__(self):
        self.transformer = SimpleNamespace(select_top_rows=selector)
        self.query_feat = SimpleNamespace(weight=torch.arange(300 * 2, dtype=torch.float32).reshape(300, 2))
        self.refpoint_embed = SimpleNamespace(weight=torch.zeros((300, 4), dtype=torch.float32))
        self.log, self.export_count, self.exported = [], 0, False

    def export(self):
        self.export_count += 1
        self.exported = True

    def __call__(self, tensor):
        self.log.append(("B" if self.exported else "A", float(tensor.flatten()[0])))
        self.transformer.select_top_rows(torch.arange(320, dtype=torch.float32).view(1, -1), 300)
        raw = [torch.zeros((1, 300, 4)), torch.zeros((1, 300, 91))]
        return tuple(raw) if self.exported else {"pred_boxes": raw[0], "pred_logits": raw[1]}


def observation(ids):
    values = np.asarray(ids, dtype=np.int64).reshape(1, -1)
    raw = [np.repeat(values[..., None], width, axis=-1).astype(np.float32) for width in (4, 91)]
    return {"selection": {"ids": values}, "raw": raw}


def test_actual_closure_captures_tied_topk_without_a_second_selection(monkeypatch):
    original, calls = torch.topk, []
    def tracked(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(torch, "topk", tracked)
    model = SimpleNamespace(transformer=SimpleNamespace(select_top_rows=selector))
    scores = torch.tensor([[1., 1., 1., 0.]])
    with diagnostic.capture_selection(model, k=2) as records:
        choose = model.transformer.select_top_rows(scores, 2)
        actual_ids = choose(torch.arange(4).view(1, 4, 1)).squeeze(-1).numpy()
    assert calls == [1]
    np.testing.assert_array_equal(records[0]["ids"], actual_ids)
    assert records[0]["stats"]["boundary_margin"] == 0
    assert records[0]["stats"]["exact_tie_groups"] == 1
    assert model.transformer.select_top_rows is selector


def test_four_normal_forwards_all_precede_the_single_export_conversion():
    model = Tiny()
    result = diagnostic.native_phases(model, [torch.tensor([float(i)]) for i in range(4)])
    assert model.log == [(phase, float(i)) for phase in ("A", "B") for i in range(4)]
    assert model.export_count == 1
    assert all(row["selection_call_count"] == 1 for phase in result.values() for row in phase)


def test_same_anchor_set_permutation_is_only_diagnostic_and_keeps_slot_failure():
    report = diagnostic.compare_selection(observation([0, 1, 2]), observation([2, 0, 1]))
    assert report["same_id_set"] and not report["ordered_ids_equal"] and report["common_ids"] == 3
    assert not report["slot_aligned"]["matched"]
    assert report["common_anchor_aligned_diagnostic_only"]["matched"]
    assert "learned" in report["caution"] and "harmless" in report["caution"]


def test_membership_difference_is_reported_even_when_common_anchors_match():
    report = diagnostic.compare_selection(observation([0, 1, 2]), observation([0, 1, 3]))
    assert not report["same_id_set"] and report["common_ids"] == 2
    assert report["left_only_ids"] == [2] and report["right_only_ids"] == [3]
    assert report["common_anchor_aligned_diagnostic_only"]["matched"]
    assert report["slot_aligned"]["outputs"]["dets"]["failed_queries"] == 1


def test_fixed_error_ratio_counts_elements_and_queries_and_rejects_nonfinite():
    left, right = np.zeros((1, 3, 4), np.float32), np.zeros((1, 3, 4), np.float32)
    left[0, 1, :2] = .001
    result = diagnostic.compare_arrays(left, right)
    assert result["failed_elements"] == 2 and result["failed_queries"] == 1
    assert result["max_error_to_allowed_ratio"] == pytest.approx(10.)
    left[0, 0, 0] = np.inf
    with pytest.raises(ValueError, match="Invalid comparison"):
        diagnostic.compare_arrays(left, right)


def test_failed_mask_exactly_matches_original_float32_isclose_at_ulp_boundary():
    right = np.full((1, 1, 4), 1.091843504497092e-08, np.float32)
    left = np.full((1, 1, 4), 0.00010001093323808163, np.float32)
    assert np.allclose(left, right, atol=1e-4, rtol=1e-3)
    report = diagnostic.compare_arrays(left, right)
    assert report["matched"] and report["failed_elements"] == 0
    assert report["max_error_to_allowed_ratio"] > 1  # Float64 ratio is diagnostic; it does not replace NumPy's gate.


def test_learned_slots_report_inequality_and_recoverable_row_fingerprints():
    values = np.arange(600, dtype=np.float32).reshape(300, 2)
    report = diagnostic.slot_info(values)
    assert not report["all_rows_equal"] and report["max_deviation_from_first_row"] == 598
    assert len(set(report["row_sha256"])) == 300
    assert diagnostic.slot_info(np.zeros((300, 4), np.float32))["all_rows_equal"]


@pytest.mark.parametrize("axis,k,gather,ok", [(1, 300, True, True), (0, 300, True, False),
                                           (1, 299, True, False), (1, 300, False, False)])
def test_debug_copy_exposes_existing_topk_without_new_nodes_and_rejects_wrong_contract(tmp_path, axis, k, gather, ok):
    import onnx
    from onnx import helper, TensorProto, numpy_helper
    nodes = [helper.make_node("TopK", ["scores", "K"], ["values", "ids"], axis=axis, largest=1, sorted=1)]
    if gather:
        nodes.append(helper.make_node("GatherElements", ["scores", "ids"], ["selected"], axis=1))
    else:
        nodes.append(helper.make_node("Identity", ["values"], ["selected"]))
    graph = helper.make_graph(nodes, "tiny-topk", [helper.make_tensor_value_info("scores", TensorProto.FLOAT, [1, 320])],
                              [helper.make_tensor_value_info("selected", TensorProto.FLOAT, [1, k])],
                              [numpy_helper.from_array(np.array([k], np.int64), name="K")])
    original, copy = tmp_path / "original.onnx", tmp_path / "debug.onnx"
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=10), original)
    before = source.asset(original)
    if ok:
        report = diagnostic.make_debug_graph(original, copy)
        result = onnx.load(copy)
        assert len(result.graph.node) == len(nodes) == report["debug_nodes"]
        assert report["new_computation_nodes"] == 0
        assert [value.name for value in result.graph.output] == ["selected", "scores", "values", "ids"]
        assert source.asset(original) == before
        with pytest.raises(ValueError, match="new copy"):
            diagnostic.make_debug_graph(original, copy)
    else:
        with pytest.raises(ValueError):
            diagnostic.make_debug_graph(original, copy)


def setup_protocol(tmp_path, monkeypatch, effect=False, mutation=None):
    from PIL import Image
    primary, exported, output = (tmp_path / name for name in ("primary", "export", "diagnosis"))
    for path in (primary, exported, output): path.mkdir()
    image, weights, graph = tmp_path / "real.png", tmp_path / "weights.pth", exported / "model.onnx"
    Image.new("RGB", (2, 3)).save(image)
    weights.write_bytes(b"synthetic state, bootstrap mocked")
    graph.write_bytes(b"original graph, ORT mocked")
    archive = primary / "exporter-source.py"
    archive.write_bytes(b"frozen primary tool fixture")
    source.write(primary / "manifest.json", {"ok": False, "status": "failed", "source_revision": source.SOURCE_REVISION,
                                            "error": "Native/ORT parity failed", "tool": {"sha256": source.sha(archive)}})
    source.write(exported / "manifest.json", {"ok": True, "source_revision": source.SOURCE_REVISION, "onnx": source.asset(graph)})
    monkeypatch.setattr(source, "preprocess_rgb", lambda rgb: torch.tensor([float(rgb[0, 0, 0])], dtype=torch.float32))
    blank = [np.zeros((1, 300, 4), np.float32), np.zeros((1, 300, 91), np.float32)]
    first_random = int(np.random.default_rng(20261024).integers(0, 256, (721, 1279, 3), dtype=np.uint8)[0, 0, 0])
    rows = []
    for name, value, shape in zip(("zeros", "white", "seeded_rgb", "real-0"), (0, 255, first_random, 0),
                                  ((480, 640, 3), (333, 517, 3), (721, 1279, 3), (3, 2, 3))):
        rows.append({"name": name, "input_sha256": diagnostic.array_info(np.array([float(value)], np.float32))["sha256"],
                     "rgb_shape": list(shape), "source": source.asset(image) if name == "real-0" else None,
                     "outputs": [{"output": label, "native_sha256": diagnostic.array_info(raw)["sha256"],
                                  "ort_sha256": diagnostic.array_info(raw)["sha256"]} for label, raw in zip(("dets", "labels"), blank)]})
    source.write(primary / "comparisons.json", rows)
    monkeypatch.setattr(source, "build_native", lambda path: (Tiny(), {"weights": source.asset(path)}))
    monkeypatch.setattr(source, "inspect_graph", lambda path: {})
    monkeypatch.setattr(diagnostic, "code_assets", lambda: [Path(diagnostic.__file__), Path(source.__file__)])
    def debug_graph(original, target):
        target.write_bytes(b"debug graph fixture")
        return {"onnx": source.asset(target), "scores": "scores", "values": "values", "indices": "ids"}
    monkeypatch.setattr(diagnostic, "make_debug_graph", debug_graph)
    class Session:
        def __init__(self, path, **kwargs): self.debug = Path(path).name == "debug-topk.onnx"
        def get_providers(self): return ["CPUExecutionProvider"]
        def run(self, names, feed):
            if mutation == "graph": graph.write_bytes(b"replaced during diagnosis")
            if mutation == "primary": (primary / "comparisons.json").write_text("[]")
            outputs = [value.copy() for value in blank]
            if not self.debug: return outputs
            if effect: outputs[0][0, 0, 0] = .00001  # even an allclose-only change invalidates index attribution
            scores = np.arange(320, dtype=np.float32).reshape(1, -1)
            ids = np.arange(319, 19, -1, dtype=np.int64).reshape(1, -1)
            return [*outputs, scores, scores[0, ids[0]].reshape(1, -1), ids]
    monkeypatch.setitem(sys.modules, "onnxruntime", SimpleNamespace(SessionOptions=SimpleNamespace, InferenceSession=Session))
    return SimpleNamespace(primary_parity_dir=primary, export_dir=exported, output_dir=output, weights=weights, image=image)


@pytest.mark.parametrize("effect", [False, True])
def test_complete_protocol_preserves_primary_failure_and_detects_instrumentation_effect(tmp_path, monkeypatch, effect):
    args = setup_protocol(tmp_path, monkeypatch, effect=effect)
    before = source.asset(args.primary_parity_dir / "manifest.json")
    report = diagnostic.diagnosis(args)
    assert report["status"] == "diagnostic_complete" and report["primary_parity_status"] == "failed"
    assert report["primary_gate_passed"] is False and report["primary_failure_unchanged"] is True
    assert report["current_A_C_allclose"] is True  # A new result cannot rewrite the historical failed gate.
    assert source.asset(args.primary_parity_dir / "manifest.json") == before
    assert len(report["samples"]) == 4
    for row in report["samples"]:
        assert row["instrumentation_effect"] == effect
        assert row["debug_ids_proven_for_original_C"] != effect
        assert (row["B_vs_C_selection"] is None) == effect
        assert all(row["exact_primary_output_sha_reproduced"][phase][output] for phase in ("A", "C") for output in ("dets", "labels"))
        with np.load(row["tensor_file"]["path"], allow_pickle=False) as cache:
            assert {"A_dets", "A_labels", "B_dets", "B_labels", "C_dets", "C_labels", "D_ids"}.issubset(cache.files)
            assert all(cache[name].dtype != object for name in cache.files)


@pytest.mark.parametrize("mutation", ["graph", "primary"])
def test_graph_or_primary_cache_replacement_during_diagnosis_fails_closed(tmp_path, monkeypatch, mutation):
    args = setup_protocol(tmp_path, monkeypatch, mutation=mutation)
    with pytest.raises(ValueError, match="asset changed during diagnosis"):
        diagnostic.diagnosis(args)


def test_incomplete_four_case_cache_is_not_interpreted_as_empty_success(tmp_path, monkeypatch):
    args = setup_protocol(tmp_path, monkeypatch)
    source.write(args.primary_parity_dir / "comparisons.json", [])
    with pytest.raises(ValueError, match="exactly the four"):
        diagnostic.load_primary(args.primary_parity_dir)


@pytest.mark.parametrize("target", ["graph", "primary"])
def test_asset_replacement_between_bootstrap_read_and_global_snapshot_is_rejected(tmp_path, monkeypatch, target):
    args = setup_protocol(tmp_path, monkeypatch)
    if target == "graph":
        monkeypatch.setattr(source, "inspect_graph", lambda path: path.write_bytes(b"replaced after initial pin"))
    else:
        original = diagnostic.load_primary
        def replaced(path):
            result = original(path)
            (path / "comparisons.json").write_text("[]")
            return result
        monkeypatch.setattr(diagnostic, "load_primary", replaced)
    monkeypatch.setattr(source, "build_native", lambda path: pytest.fail("Must reject bootstrap drift before model loading"))
    with pytest.raises(ValueError, match="pinned asset changed during bootstrap"):
        diagnostic.diagnosis(args)


def test_pinned_json_detects_replacement_during_the_read(tmp_path, monkeypatch):
    path = tmp_path / "data.json"
    path.write_text("{}")
    original, calls = source.asset, []
    def changed(value):
        result = original(value)
        if not calls:
            path.write_text('{"new":true}')
        calls.append(1)
        return result
    monkeypatch.setattr(source, "asset", changed)
    with pytest.raises(ValueError, match="changed during read"):
        diagnostic.pinned_json(path)


def complete_cached_diagnosis(tmp_path, monkeypatch):
    args = setup_protocol(tmp_path, monkeypatch)
    report = diagnostic.diagnosis(args)
    report.update(ok=True, source_revision=source.SOURCE_REVISION, tool=source.asset(diagnostic.__file__),
                  watchdog_tool=source.asset(source.__file__))
    source.write(args.output_dir / "manifest.json", report)
    shutil.copyfile(diagnostic.__file__, args.output_dir / "diagnostic-source.py")
    shutil.copyfile(source.__file__, args.output_dir / "exporter-helper-source.py")
    return args.output_dir


def test_rescore_uses_only_pinned_npz_and_writes_new_derived_report_without_runtime_imports(tmp_path, monkeypatch):
    directory = complete_cached_diagnosis(tmp_path, monkeypatch)
    before = source.asset(directory / "manifest.json")
    output = tmp_path / "rescore"
    monkeypatch.setattr(source, "build_native", lambda path: pytest.fail("Rescore cannot construct a model"))
    for name in ("torch", "onnxruntime", "rfdetr"):
        monkeypatch.setitem(sys.modules, name, None)
    argv = ["rescore", "--diagnostic-dir", str(directory), "--output-dir", str(output)]
    assert diagnostic.main(argv) == 0
    report = json.loads((output / "manifest.json").read_text())
    assert report["ok"] and report["status"] == "rescored_from_cached_tensors" and report["no_inference_performed"]
    assert report["primary_gate_passed"] is False and report["primary_failure_unchanged"]
    assert len(report["samples"]) == 4 and report["numbers_derived_from_cached_tensors"]
    assert source.asset(directory / "manifest.json") == before
    output_before = source.asset(output / "manifest.json")
    with pytest.raises(ValueError, match="never overwritten"):
        diagnostic.main(argv)
    assert source.asset(output / "manifest.json") == output_before


@pytest.mark.parametrize("target", ["npz", "comparisons"])
def test_rescore_rejects_missing_or_changed_cached_evidence_instead_of_success(tmp_path, monkeypatch, target):
    directory = complete_cached_diagnosis(tmp_path, monkeypatch)
    if target == "npz":
        with (directory / "sample-00.npz").open("ab") as handle:
            handle.write(b"tampered")
    else:
        source.write(directory / "comparisons.json", [])
    output = tmp_path / "failed-rescore"
    assert diagnostic.main(["rescore", "--diagnostic-dir", str(directory), "--output-dir", str(output)]) == 1
    report = json.loads((output / "manifest.json").read_text())
    assert not report["ok"] and report["status"] == "failed" and report["primary_gate_passed"] is False


def test_diagnostic_cli_uses_shared_watchdog_and_refuses_existing_output(tmp_path, monkeypatch):
    args = ["--" + name + "=" + str(tmp_path / name) for name in ("weights", "export-dir", "primary-parity-dir", "image", "output-dir")]
    captured = []
    def watchdog(config, argv, **kwargs):
        captured.append((config.command, argv, kwargs))
        return 17
    with monkeypatch.context() as context:
        context.setattr(source, "bounded_run", watchdog)
        assert diagnostic.main(args) == 17
    assert captured[0][0] == "diagnose" and captured[0][2]["module"] == "tools.diagnose_rfdetr_candidate"
    (tmp_path / "output-dir").mkdir()
    with pytest.raises(ValueError, match="never overwritten"):
        diagnostic.main(args)
