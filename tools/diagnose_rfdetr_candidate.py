"""Offline RF-DETR native/export/ORT diagnosis; never changes the primary gate."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

from tools import prepare_rfdetr_candidate as source

ATOL, RTOL = 1e-4, 1e-3
CAUTION = ("Anchor rank is paired with fixed learned query/refpoint slots. "
           "Equal selected sets or an anchor-aligned pass do not establish a harmless permutation.")


def array_info(value):
    return {"shape": list(value.shape), "dtype": str(value.dtype),
            "sha256": hashlib.sha256(value.tobytes()).hexdigest()}


def compare_arrays(left, right):
    import numpy as np
    source.require(left.shape == right.shape and left.dtype == right.dtype == np.float32
                   and left.size and np.isfinite(left).all() and np.isfinite(right).all(), "Invalid comparison tensors")
    delta = np.abs(left.astype(np.float64) - right.astype(np.float64))
    allowed = ATOL + RTOL * np.abs(right.astype(np.float64))
    # Match the original NumPy float32 gate exactly, including ULP boundaries.
    failed = ~np.isclose(left, right, atol=ATOL, rtol=RTOL)
    return {"matched": not bool(failed.any()), "bitwise_equal": bool(np.array_equal(left, right)),
            "elements": int(left.size), "failed_elements": int(failed.sum()),
            "queries": int(left.shape[1]), "failed_queries": int(failed.any(axis=-1).sum()),
            "max_abs": float(delta.max()), "mean_abs": float(delta.mean()),
            "max_error_to_allowed_ratio": float((delta / allowed).max())}


def compare_outputs(left, right):
    items = {name: compare_arrays(a, b) for name, a, b in zip(("dets", "labels"), left, right)}
    return {"matched": all(item["matched"] for item in items.values()),
            "bitwise_equal": all(item["bitwise_equal"] for item in items.values()), "outputs": items}


def selection_stats(scores, ids, k=300):
    import numpy as np
    source.require(scores.ndim == 2 and scores.shape[0] == 1 and scores.dtype == np.float32
                   and np.isfinite(scores).all(), "Invalid TopK scores")
    source.require(ids.shape == (1, k) and ids.dtype == np.int64
                   and np.unique(ids).size == k and ids.min() >= 0 and ids.max() < scores.shape[1], "Invalid TopK IDs")
    selected = scores[0, ids[0]]
    source.require(np.all(selected[:-1] >= selected[1:]), "TopK is not sorted descending")
    other = np.ones(scores.shape[1], dtype=bool)
    other[ids[0]] = False
    _, counts = np.unique(scores, return_counts=True)
    return {"k": k, "tokens": int(scores.shape[1]), "selected_scores": selected.tolist(),
            "boundary_margin": float(selected.min() - scores[0, other].max()) if other.any() else None,
            "adjacent_selected_rank_gaps": (selected[:-1] - selected[1:]).tolist(),
            "exact_tie_groups": int((counts > 1).sum()), "tokens_in_exact_ties": int(counts[counts > 1].sum()),
            "adjacent_selected_exact_ties": int((selected[:-1] == selected[1:]).sum())}


@contextlib.contextmanager
def capture_selection(model, k=300):
    """Observe the actual returned closure; do not compute a second model TopK."""
    import torch
    records = []
    original = model.transformer.select_top_rows
    def observed(scores, count):
        source.require(count == k, "Unexpected model TopK count")
        select = original(scores, count)
        rows = torch.arange(scores.shape[1], device=scores.device, dtype=torch.int64).view(1, -1, 1)
        rows = rows.expand(scores.shape[0], -1, -1)
        ids = select(rows).squeeze(-1).detach().cpu().numpy().copy()
        values = scores.detach().cpu().numpy().copy()
        records.append({"scores": values, "ids": ids, "stats": selection_stats(values, ids, k)})
        return select
    model.transformer.select_top_rows = observed
    try:
        yield records
    finally:
        model.transformer.select_top_rows = original


def slot_info(values):
    import numpy as np
    source.require(values.ndim == 2 and values.shape[0] == 300 and values.dtype == np.float32
                   and np.isfinite(values).all(), "Invalid learned query/refpoint slots")
    return {**array_info(values), "row_sha256": [hashlib.sha256(row.tobytes()).hexdigest() for row in values],
            "all_rows_equal": bool(np.array_equal(values, np.broadcast_to(values[0], values.shape))),
            "max_deviation_from_first_row": float(np.abs(values.astype(np.float64) - values[0]).max()),
            "min": float(values.min()), "max": float(values.max())}


def native_phases(model, tensors):
    """All normal forwards precede the single destructive source export conversion."""
    import torch
    result = {"A": [], "B": []}
    with torch.inference_mode():
        for phase in ("A", "B"):
            if phase == "B":
                model.export()
            for tensor in tensors:
                with capture_selection(model) as records:
                    raw = [value.detach().cpu().numpy().copy() for value in source.raw_pair(model(tensor))]
                source.validate_outputs(raw)
                source.require(len(records) == 1, "Expected exactly one eval/export TopK call")
                result[phase].append({"raw": raw, "selection": records[0], "selection_call_count": len(records)})
    return result


def compare_selection(left, right):
    import numpy as np
    a, b = left["selection"]["ids"][0], right["selection"]["ids"][0]
    common, ia, ib = np.intersect1d(a, b, return_indices=True)
    aligned = compare_outputs([value[:, ia] for value in left["raw"]], [value[:, ib] for value in right["raw"]]) if common.size else None
    return {"ordered_ids_equal": bool(np.array_equal(a, b)), "same_id_set": bool(common.size == a.size == b.size),
            "same_slot_ids": int((a == b).sum()), "common_ids": int(common.size),
            "left_only_ids": np.setdiff1d(a, b).tolist(), "right_only_ids": np.setdiff1d(b, a).tolist(),
            "common_anchor_ids": common.tolist(), "left_slots_for_common_ids": ia.tolist(), "right_slots_for_common_ids": ib.tolist(),
            "slot_aligned": compare_outputs(left["raw"], right["raw"]),
            "common_anchor_aligned_diagnostic_only": aligned, "caution": CAUTION}


def analyse_cached(arrays):
    """Numerical analysis only; usable without Torch, RF-DETR, ONNX or ORT."""
    import numpy as np
    observations = {}
    for phase in ("A", "B", "C", "D"):
        raw = [arrays[phase + "_dets"], arrays[phase + "_labels"]]
        source.validate_outputs(raw)
        observations[phase] = {"raw": raw}
        if phase != "C":
            scores, ids = arrays[phase + "_scores"], arrays[phase + "_ids"]
            observations[phase]["selection"] = {"scores": scores, "ids": ids, "stats": selection_stats(scores, ids)}
    a, b, c, d = (observations[phase] for phase in ("A", "B", "C", "D"))
    source.require(np.array_equal(arrays["D_values"], arrays["D_scores"][np.arange(1)[:, None], arrays["D_ids"]]),
                   "Debug TopK values/indices inconsistent")
    effect = compare_outputs(c["raw"], d["raw"])
    return {"A_vs_B": compare_selection(a, b), "B_vs_C_slot_aligned": compare_outputs(b["raw"], c["raw"]),
            "A_vs_C_slot_aligned": compare_outputs(a["raw"], c["raw"]), "B_vs_D": compare_selection(b, d),
            "A_vs_D": compare_selection(a, d), "C_vs_D_ordinary_outputs": effect,
            "B_vs_C_selection": compare_selection(b, d) if effect["bitwise_equal"] else None,
            "A_vs_C_selection": compare_selection(a, d) if effect["bitwise_equal"] else None,
            "instrumentation_effect": not effect["bitwise_equal"], "debug_ids_proven_for_original_C": effect["bitwise_equal"],
            "selection": {phase: observations[phase]["selection"]["stats"] for phase in ("A", "B", "D")}}


def make_debug_graph(original, target):
    import numpy as np
    import onnx
    from onnx import helper, numpy_helper, TensorProto
    source.require(not target.exists(), "Debug graph must be a new copy")
    model = onnx.load(original, load_external_data=False)
    topks = [node for node in model.graph.node if node.op_type == "TopK"]
    source.require(len(topks) == 1, "Expected exactly one TopK")
    topk = topks[0]
    attrs = {item.name: helper.get_attribute_value(item) for item in topk.attribute}
    source.require(attrs.get("axis", -1) in (1, -1) and attrs.get("largest", 1) == 1 and attrs.get("sorted", 1) == 1,
                   "Unexpected TopK axis/largest/sorted")
    constants = {item.name: numpy_helper.to_array(item) for item in model.graph.initializer if item.name == topk.input[1]}
    for node in model.graph.node:
        if node.op_type == "Constant" and topk.input[1] in node.output:
            tensors = [item.t for item in node.attribute if item.name == "value"]
            source.require(len(tensors) == 1, "Unsupported TopK K constant")
            constants[topk.input[1]] = numpy_helper.to_array(tensors[0])
    source.require(topk.input[1] in constants and constants[topk.input[1]].dtype == np.int64
                   and np.array_equal(constants[topk.input[1]], np.array([300], np.int64)), "Expected constant K300")
    # Track the index tensor through only index-preserving shape operations.
    derived = {topk.output[1]}
    for _ in range(len(model.graph.node)):
        before = len(derived)
        for node in model.graph.node:
            if node.op_type in {"Unsqueeze", "Squeeze", "Expand", "Reshape", "Identity", "Cast"} and node.input[0] in derived:
                derived.update(node.output)
        if len(derived) == before:
            break
    source.require(any(node.op_type == "GatherElements" and node.input[1] in derived for node in model.graph.node),
                   "TopK indices do not feed the expected GatherElements path")
    names = [topk.input[0], topk.output[0], topk.output[1]]
    source.require(not set(names).intersection(value.name for value in model.graph.output), "Debug output already exposed")
    node_count = len(model.graph.node)
    model.graph.output.extend([helper.make_tensor_value_info(names[0], TensorProto.FLOAT, [1, None]),
                               helper.make_tensor_value_info(names[1], TensorProto.FLOAT, [1, 300]),
                               helper.make_tensor_value_info(names[2], TensorProto.INT64, [1, 300])])
    onnx.checker.check_model(model)
    onnx.save_model(model, target, save_as_external_data=False)
    return {"onnx": source.asset(target), "scores": names[0], "values": names[1], "indices": names[2],
            "topk_name": topk.name, "topk_attributes": attrs, "k": 300, "original_nodes": node_count,
            "debug_nodes": len(model.graph.node), "new_computation_nodes": 0, "indices_feed_gather_elements": True}


def pinned_json(path):
    initial = source.asset(path)
    value = json.loads(path.read_text())
    source.require(source.asset(path) == initial, f"JSON asset changed during read: {path}")
    return value, initial


def load_primary(directory):
    manifest_path, comparisons_path = directory / "manifest.json", directory / "comparisons.json"
    manifest, manifest_asset = pinned_json(manifest_path)
    rows, comparisons_asset = pinned_json(comparisons_path)
    source.require(manifest.get("ok") is False and manifest.get("status") == "failed"
                   and manifest.get("source_revision") == source.SOURCE_REVISION
                   and "Native/ORT parity failed" in manifest.get("error", ""), "Expected the original failed primary parity")
    source.require(isinstance(rows, list) and [row.get("name") for row in rows] == ["zeros", "white", "seeded_rgb", "real-0"],
                   "Primary cache must contain exactly the four fixed cases")
    for row in rows:
        source.require([item.get("output") for item in row["outputs"]] == ["dets", "labels"]
                       and all(len(item.get(key, "")) == 64 for item in row["outputs"] for key in ("native_sha256", "ort_sha256")),
                       "Incomplete primary output hashes")
    archived = directory / "exporter-source.py"
    archived_asset = source.asset(archived)
    source.require(archived_asset["sha256"] == manifest["tool"]["sha256"], "Primary exporter source archive changed/missing")
    return manifest, rows, {info["path"]: info for info in (manifest_asset, comparisons_asset, archived_asset)}


def fixed_inputs(rows, image):
    import numpy as np
    from PIL import Image
    image_asset = source.asset(image)
    source.require(rows[3]["source"] == image_asset, "Real image differs from the primary input")
    with Image.open(image) as opened:
        real = np.asarray(opened.convert("RGB")).copy()
    values = [np.zeros((480, 640, 3), np.uint8), np.full((333, 517, 3), 255, np.uint8),
              np.random.default_rng(20261024).integers(0, 256, (721, 1279, 3), dtype=np.uint8), real]
    tensors = [source.preprocess_rgb(rgb) for rgb in values]
    for row, rgb, tensor in zip(rows, values, tensors):
        source.require(row["rgb_shape"] == list(rgb.shape) and row["input_sha256"] == array_info(tensor.numpy())["sha256"],
                       "Diagnostic input does not reproduce the primary coordinates/hash")
    return tensors, image_asset


def code_assets():
    root = Path(importlib.metadata.distribution("rfdetr").locate_file("rfdetr"))
    paths = [Path(__file__), Path(source.__file__), root / "models/transformer.py", root / "models/lwdetr.py"]
    paths.extend(sorted((root / "models/backbone").rglob("*.py")))
    return paths


def diagnosis(args):
    import numpy as np
    import onnxruntime as ort
    primary, rows, initial_assets = load_primary(args.primary_parity_dir)
    export_manifest = args.export_dir / "manifest.json"
    export, export_asset = pinned_json(export_manifest)
    graph = args.export_dir / "model.onnx"
    graph_asset = source.asset(graph)
    source.require(export.get("ok") is True and export.get("source_revision") == source.SOURCE_REVISION
                   and graph_asset == export["onnx"], "Export is incomplete or graph pin changed")
    initial_assets.update({info["path"]: info for info in (export_asset, graph_asset)})
    source.inspect_graph(graph)
    paths = [*initial_assets, args.weights, args.image, *code_assets()]
    before = {str(Path(path).resolve()): source.asset(path) for path in paths}
    source.require(all(before[path] == value for path, value in initial_assets.items()), "A pinned asset changed during bootstrap")
    tensors, image_asset = fixed_inputs(rows, args.image)
    model, native = source.build_native(args.weights)
    slots = {name: getattr(model, name).weight[:300].detach().cpu().numpy().copy() for name in ("query_feat", "refpoint_embed")}
    slot_report = {name: slot_info(value) for name, value in slots.items()}
    np.savez_compressed(args.output_dir / "learned-slots.npz", **slots)
    phases = native_phases(model, tensors)
    del model
    debug = make_debug_graph(graph, args.output_dir / "debug-topk.onnx")
    options = ort.SessionOptions()
    options.intra_op_num_threads, options.inter_op_num_threads = 2, 1
    original_session = ort.InferenceSession(str(graph), sess_options=options, providers=["CPUExecutionProvider"])
    debug_session = ort.InferenceSession(debug["onnx"]["path"], sess_options=options, providers=["CPUExecutionProvider"])
    source.require(original_session.get_providers() == debug_session.get_providers() == ["CPUExecutionProvider"], "CPU only")
    reports = []
    for index, (row, tensor) in enumerate(zip(rows, tensors)):
        c = original_session.run(["dets", "labels"], {"images": tensor.numpy()})
        d = debug_session.run(["dets", "labels", debug["scores"], debug["values"], debug["indices"]], {"images": tensor.numpy()})
        a, b = phases["A"][index], phases["B"][index]
        arrays = {"C_dets": c[0], "C_labels": c[1], "D_dets": d[0], "D_labels": d[1],
                  "D_scores": d[2], "D_values": d[3], "D_ids": d[4]}
        for phase, values in (("A", a), ("B", b)):
            arrays.update({phase + "_dets": values["raw"][0], phase + "_labels": values["raw"][1],
                           phase + "_scores": values["selection"]["scores"], phase + "_ids": values["selection"]["ids"]})
        path = args.output_dir / f"sample-{index:02d}.npz"
        np.savez_compressed(path, **arrays)
        exact_primary = {phase: {name: array_info(values[j])["sha256"] == row["outputs"][j][key]
                                for j, name in enumerate(("dets", "labels"))}
                         for phase, values, key in (("A", a["raw"], "native_sha256"), ("C", c, "ort_sha256"))}
        reports.append({"name": row["name"], "input_sha256": row["input_sha256"], "rgb_shape": row["rgb_shape"],
                        "tensor_file": source.asset(path), "tensors": {name: array_info(value) for name, value in arrays.items()},
                        **analyse_cached(arrays),
                        "selection_calls": {"A": a["selection_call_count"], "B": b["selection_call_count"]},
                        "exact_primary_output_sha_reproduced": exact_primary})
    source.write(args.output_dir / "comparisons.json", reports)
    source.require(all(source.asset(path) == value for path, value in before.items()), "An input/code asset changed during diagnosis")
    return {"status": "diagnostic_complete", "primary_parity_status": "failed", "primary_gate_passed": False,
            "primary_failure_unchanged": True, "current_A_C_allclose": all(row["A_vs_C_slot_aligned"]["matched"] for row in reports),
            "primary_manifest_sha256": initial_assets[str((args.primary_parity_dir / "manifest.json").resolve())]["sha256"],
            "primary_tool_sha256": primary["tool"]["sha256"],
            "assets_before_and_after_equal": True, "input_assets": before, "image": image_asset,
            "native": native, "debug_graph": debug, "learned_slots": slot_report,
            "learned_slots_file": source.asset(args.output_dir / "learned-slots.npz"),
            "phase_order": ["A:all_four_normal_eval", "B:single_export_then_all_four", "C:original_ORT", "D:instrumented_ORT"],
            "samples": reports, "atol": ATOL, "rtol": RTOL, "caution": CAUTION,
            "quality_evidence": False, "target_tpu": "not_compiled_or_verified"}


def rescore_cached(directory, output):
    """Derive a new report from pinned NPZ; do not execute any model runtime."""
    import numpy as np
    source.require(np.__version__ == source.PROTECTED["numpy"], "Rescore NumPy version differs from the pinned primary runtime")
    code_before = {str(Path(path).resolve()): source.asset(path) for path in (__file__, source.__file__)}
    manifest, manifest_asset = pinned_json(directory / "manifest.json")
    rows, comparisons_asset = pinned_json(directory / "comparisons.json")
    source.require(manifest.get("ok") is True and manifest.get("status") == "diagnostic_complete"
                   and manifest.get("source_revision") == source.SOURCE_REVISION
                   and manifest.get("primary_gate_passed") is False and manifest.get("primary_failure_unchanged") is True,
                   "Expected a complete diagnosis preserving the failed primary gate")
    source.require(rows == manifest["samples"] and [row["name"] for row in rows] == ["zeros", "white", "seeded_rgb", "real-0"],
                   "Diagnostic cache is incomplete or differs from its manifest")
    before = {item["path"]: item for item in (manifest_asset, comparisons_asset)}
    for filename, key in (("diagnostic-source.py", "tool"), ("exporter-helper-source.py", "watchdog_tool")):
        archived = source.asset(directory / filename)
        source.require(archived["sha256"] == manifest[key]["sha256"], "Frozen diagnostic source archive changed/missing")
        before[archived["path"]] = archived
    derived = []
    for row in rows:
        path = Path(row["tensor_file"]["path"])
        initial = source.asset(path)
        source.require(initial == row["tensor_file"], "Cached NPZ pin changed")
        before[initial["path"]] = initial
        with np.load(path, allow_pickle=False) as stored:
            source.require(set(stored.files) == set(row["tensors"]), "Incomplete/extra cached tensors")
            arrays = {name: stored[name].copy() for name in stored.files}
        source.require(all(array_info(value) == row["tensors"][name] for name, value in arrays.items()), "Cached tensor shape/dtype/SHA changed")
        derived.append({"name": row["name"], "input_sha256": row["input_sha256"], "tensor_file": initial,
                        "numbers_derived_from_cached_tensors": True, **analyse_cached(arrays),
                        "source_exact_primary_output_sha_reproduced": row["exact_primary_output_sha_reproduced"]})
    source.write(output / "comparisons.json", derived)
    source.require(all(source.asset(path) == item for path, item in before.items()), "A rescore input asset changed")
    source.require(all(source.asset(path) == item for path, item in code_before.items()), "Rescore code changed")
    return {"status": "rescored_from_cached_tensors", "numbers_derived_from_cached_tensors": True,
            "no_inference_performed": True, "source_diagnostic_assets": before, "samples": derived,
            "rescore_code_assets": code_before, "numpy_version": np.__version__,
            "primary_parity_status": "failed", "primary_gate_passed": False, "primary_failure_unchanged": True,
            "current_A_C_allclose": all(row["A_vs_C_slot_aligned"]["matched"] for row in derived),
            "atol": ATOL, "rtol": RTOL, "element_gate": "numpy.isclose(float32, atol=1e-4, rtol=1e-3)",
            "caution": CAUTION, "quality_evidence": False, "target_tpu": "not_compiled_or_verified"}


def rescore_main(argv):
    parser = argparse.ArgumentParser(description="Rescore cached RF-DETR tensors; no model/ORT execution")
    parser.add_argument("--diagnostic-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    source.require(not args.output_dir.exists(), "Output exists; artifacts are never overwritten")
    args.output_dir.mkdir(parents=True)
    report = {"schema_version": 1, "ok": False, "status": "running", "stage": "rescore",
              "tool": source.asset(__file__), "source_revision": source.SOURCE_REVISION,
              "primary_gate_passed": False, "no_inference_performed": True}
    try:
        report.update(rescore_cached(args.diagnostic_dir, args.output_dir))
        report["ok"] = True
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    source.write(args.output_dir / "manifest.json", report)
    return 0 if report["ok"] else 1


def worker(args):
    report = {"schema_version": 1, "ok": False, "status": "running", "stage": "diagnose",
              "tool": source.asset(__file__), "watchdog_tool": source.asset(source.__file__),
              "source_revision": source.SOURCE_REVISION, "primary_gate_passed": False,
              "quality_evidence": False, "target_tpu": "not_compiled_or_verified"}
    try:
        with source.offline_network():
            report["versions"] = source.verify_environment()
            report.update(diagnosis(args))
        report["ok"] = True
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    source.write(args.output_dir / "manifest.json", report)
    return 0 if report["ok"] else 1


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == "rescore":
        return rescore_main(arguments[1:])
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("weights", "export-dir", "primary-parity-dir", "image", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=900.)
    parser.add_argument("--max-rss-gb", type=float, default=12.)
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(arguments)
    args.command = "diagnose"
    return worker(args) if args._worker else source.bounded_run(args, arguments, module="tools.diagnose_rfdetr_candidate")


if __name__ == "__main__":
    raise SystemExit(main())
