"""SDK-free static YOLO detect head/tail validation and CPU FP32 parity.

This produces ONNX evidence, never a TPU program or compiler compatibility claim.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
TAIL_OPS = frozenset({"Constant", "Reshape", "Transpose", "Slice", "Split", "Concat", "Add", "Sub", "Mul",
    "Div", "Shape", "Gather", "Unsqueeze", "Squeeze", "Cast", "Identity"})


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def asset(path):
    path = Path(path).resolve()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def write_report(path, report):
    temporary = path.with_suffix(".json.part")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def tensor_info(value):
    import onnx

    require(value.type.HasField("tensor_type"), f"Unsupported non-tensor value: {value.name}")
    tensor = value.type.tensor_type
    require(tensor.HasField("shape"), f"Unknown shape: {value.name}")
    dims = []
    for dim in tensor.shape.dim:
        require(dim.HasField("dim_value") and dim.dim_value > 0, f"Dynamic/invalid shape: {value.name}")
        dims.append(dim.dim_value)
    return {"name": value.name, "dtype": onnx.TensorProto.DataType.Name(tensor.elem_type), "shape": dims}


def inventory(model):
    return dict(sorted(Counter(f"{node.domain or 'ai.onnx'}:{node.op_type}" for node in model.graph.node).items()))


def load_static_model(path):
    import numpy as np
    import onnx
    from onnx import numpy_helper

    model = onnx.load(str(path), load_external_data=False)
    require(not model.functions and not model.graph.sparse_initializer, "Functions/sparse initializers are unsupported")
    tensors = list(model.graph.initializer)
    for node in model.graph.node:
        require(node.domain in ("", "ai.onnx"), f"Custom operator domain is unsupported: {node.domain}")
        require(node.op_type not in {"If", "Loop", "Scan", "Dropout"} and not node.op_type.startswith("Random"),
                f"Control flow/stochastic operator is unsupported: {node.op_type}")
        for attribute in node.attribute:
            require(attribute.type not in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS), "Nested graphs are unsupported")
            if attribute.type == onnx.AttributeProto.TENSOR:
                tensors.append(attribute.t)
            if attribute.type in (onnx.AttributeProto.FLOAT, onnx.AttributeProto.FLOATS):
                values = [attribute.f] if attribute.type == onnx.AttributeProto.FLOAT else attribute.floats
                require(all(math.isfinite(value) for value in values), f"Nonfinite attribute: {node.name}")
    for tensor in tensors:
        require(tensor.data_location != onnx.TensorProto.EXTERNAL and not tensor.external_data, "External tensor data is unsupported; embed weights first")
        dtype = onnx.TensorProto.DataType.Name(tensor.data_type)
        require(dtype not in {"COMPLEX64", "COMPLEX128", "STRING", "UNDEFINED"}, f"Unsupported tensor dtype: {dtype}")
        if dtype.startswith("FLOAT") or dtype in {"DOUBLE", "BFLOAT16"}:
            require(dtype == "FLOAT", f"FP32 weights/constants required: {tensor.name}")
        values = numpy_helper.to_array(tensor)
        if np.issubdtype(values.dtype, np.floating):
            require(values.dtype == np.float32, f"FP32 weights/constants required: {tensor.name}")
            require(np.isfinite(values).all(), f"Nonfinite weights/constants: {tensor.name}")
    onnx.checker.check_model(model)
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    values = {value.name: value for value in [*model.graph.value_info, *model.graph.input, *model.graph.output]}
    for value in values.values():
        dtype = tensor_info(value)["dtype"]
        require(dtype not in {"DOUBLE", "BFLOAT16"} and not (dtype.startswith("FLOAT") and dtype != "FLOAT"),
                f"Non-FP32 floating tensor is unsupported: {value.name} ({dtype})")
    inputs = [value for value in model.graph.input if value.name not in {item.name for item in model.graph.initializer}]
    require(len(inputs) == len(model.graph.output) == 1, "Static detect requires one image input and one prediction output")
    source, output = tensor_info(inputs[0]), tensor_info(model.graph.output[0])
    require(source["dtype"] == output["dtype"] == "FLOAT", "FP32 input/output required")
    require(len(source["shape"]) == 4 and source["shape"][:2] == [1, 3]
            and math.prod(source["shape"]) <= 3 * 2048 * 2048, "Expected bounded batch-1 NCHW RGB input")
    require(len(output["shape"]) == 3 and output["shape"][0] == 1 and output["shape"][1] > 4,
            "Expected static detect output [1,4+classes,anchors]; pose/segmentation/NMS exports unsupported")
    return model, values, source, output


def graph_index(model):
    producers = {}
    for index, node in enumerate(model.graph.node):
        for name in node.output:
            require(name not in producers, f"Duplicate tensor producer: {name}")
            producers[name] = index
    return producers


def identify_cut(model, values, output):
    """Names select a head scope; shared Split/Softmax/DFL topology verifies it."""
    import numpy as np
    import onnx
    from onnx import numpy_helper

    nodes, producers = model.graph.node, graph_index(model)
    initializers = {item.name: item for item in model.graph.initializer}

    def ancestors(name):
        seen, tensors, pending = set(), set(), [name]
        while pending:
            tensor = pending.pop()
            tensors.add(tensor)
            current = producers.get(tensor)
            if current is not None and current not in seen:
                seen.add(current)
                pending.extend(nodes[current].input)
        return seen, tensors

    candidates = []
    for conv in nodes:
        match = re.fullmatch(r"(.+)/dfl/conv/Conv", conv.name)
        if conv.op_type != "Conv" or not match:
            continue
        prefix = match[1]
        classes = [node for node in nodes if node.name == prefix + "/Sigmoid" and node.op_type == "Sigmoid"]
        for sigmoid in classes:
            if len(conv.output) != 1 or len(sigmoid.output) != 1 or len(conv.input) != 2 or len(sigmoid.input) != 1:
                continue
            split_index = producers.get(sigmoid.input[0])
            if split_index is None:
                continue
            split = nodes[split_index]
            lineage, lineage_tensors = ancestors(conv.input[0])
            if (split.op_type != "Split" or len(split.output) != 2 or sigmoid.input[0] != split.output[1]
                    or split.output[0] not in lineage_tensors):
                continue
            weight = initializers.get(conv.input[1])
            if weight is None:
                continue
            weights = numpy_helper.to_array(weight)
            if weights.ndim != 4 or weights.shape[0] != 1 or weights.shape[2:] != (1, 1) or weights.shape[1] < 2:
                continue
            if not np.array_equal(weights.ravel(), np.arange(weights.shape[1], dtype=np.float32)):
                continue
            # Trace layout-only operations back to Softmax. Its normalized
            # bin axis must become Conv's input-channel axis, not an anchor axis.
            cursor, axis = conv.input[0], 1
            while cursor in producers:
                parent = nodes[producers[cursor]]
                if parent.op_type == "Identity":
                    cursor = parent.input[0]
                elif parent.op_type == "Transpose" and cursor in values:
                    rank = len(tensor_info(values[cursor])["shape"])
                    perm = next((list(attribute.ints) for attribute in parent.attribute if attribute.name == "perm"), list(reversed(range(rank))))
                    axis, cursor = perm[axis], parent.input[0]
                else:
                    break
            softmax = nodes[producers[cursor]] if cursor in producers else None
            if softmax is None or softmax.op_type != "Softmax" or not softmax.name.startswith(prefix + "/dfl/") or cursor not in values:
                continue
            shape = tensor_info(values[cursor])["shape"]
            opset = next(item.version for item in model.opset_import if item.domain in ("", "ai.onnx"))
            default_axis = onnx.defs.get_schema("Softmax", opset).attributes["axis"].default_value.i
            softmax_axis = next((attribute.i for attribute in softmax.attribute if attribute.name == "axis"), default_axis)
            if not -len(shape) <= softmax_axis < len(shape) or softmax_axis % len(shape) != axis or shape[axis] != weights.shape[1]:
                continue
            cut = [conv.output[0], sigmoid.output[0]]
            if not all(name in values for name in cut):
                continue
            box, scores = [tensor_info(values[name]) for name in cut]
            if (box["dtype"] == scores["dtype"] == "FLOAT" and box["shape"] == [1, 1, 4, output["shape"][2]]
                    and scores["shape"] == [1, output["shape"][1] - 4, output["shape"][2]]):
                candidates.append({"prefix": prefix, "cut": cut, "reg_max": weights.shape[1]})
    require(len(candidates) == 1, f"Expected unique supported detect head; found {len(candidates)}")
    return candidates[0]


def validate_cut(model, cut):
    """Every original output path must stop at a cut tensor or a constant."""
    require(len(cut) == 2 and len(set(cut)) == 2, "Exactly two distinct cut tensors required")
    producers = graph_index(model)
    require(all(name in producers for name in cut), "Missing cut tensor")
    constants = {item.name for item in model.graph.initializer}
    original_inputs = {item.name for item in model.graph.input} - constants
    visited, reached, pending = set(), set(), [item.name for item in model.graph.output]
    while pending:
        name = pending.pop()
        if not name or name in constants:
            continue
        if name in cut:
            reached.add(name)
            continue
        require(name not in original_inputs, f"Cut does not cover tail dependency on original input: {name}")
        require(name in producers, f"Missing tail dependency: {name}")
        index = producers[name]
        if index not in visited:
            visited.add(index)
            node = model.graph.node[index]
            require(node.op_type in TAIL_OPS, f"Unsupported detect tail operator: {node.op_type} ({node.name})")
            pending.extend(node.input)
    require(reached == set(cut), "Cut tensors must both contribute to original output")
    return {"required_cut_inputs": sorted(reached), "tail_nodes": len(visited), "original_input_dependencies": []}


def extract_pair(model, cut, output_dir):
    import onnx

    inputs = [value.name for value in model.graph.input if value.name not in {item.name for item in model.graph.initializer}]
    outputs = [value.name for value in model.graph.output]
    extractor = onnx.utils.Extractor(model)
    head = extractor.extract_model(inputs, cut)
    tail = extractor.extract_model(cut, outputs)
    require([value.name for value in tail.graph.input] == cut, "Extracted tail input mismatch")
    paths = []
    for name, part in (("head", head), ("tail", tail)):
        onnx.checker.check_model(part)
        path = output_dir / f"{name}.onnx"
        onnx.save(part, str(path))
        paths.append(path)
    return head, tail, paths


def compare_outputs(reference, actual, *, atol, rtol):
    import numpy as np

    require(reference.keys() == actual.keys(), "Parity output names mismatch")
    rows = {}
    for name, expected in reference.items():
        got = actual[name]
        require(expected.shape == got.shape, f"Parity shape mismatch: {name}")
        require(expected.dtype == got.dtype == np.float32, f"Parity requires FP32 tensors: {name}")
        require(np.isfinite(expected).all() and np.isfinite(got).all(), f"Nonfinite parity output: {name}")
        diff = np.abs(expected.astype(np.float64) - got.astype(np.float64))
        failures = int(np.count_nonzero(diff > atol + rtol * np.abs(expected)))
        rows[name] = {"shape": list(got.shape), "elements": got.size, "max_abs_diff": float(diff.max()),
            "mean_abs_diff": float(diff.mean()), "failing_elements": failures, "passed": failures == 0}
    return rows


def parity_inputs(source, images):
    import numpy as np

    shape = source["shape"]
    for name, array in (("zeros", np.zeros(shape, np.float32)), ("ones", np.ones(shape, np.float32)),
                        ("uniform_seed_0", np.random.default_rng(0).random(shape, dtype=np.float32))):
        yield name, array, None
    for image in images:
        from PIL import Image

        image_asset = asset(image)
        with Image.open(image) as opened:
            rgb = opened.convert("RGB")
            h, w = shape[2:]
            scale = min(w / rgb.width, h / rgb.height)
            size = (max(1, round(rgb.width * scale)), max(1, round(rgb.height * scale)))
            canvas = Image.new("RGB", (w, h), (114, 114, 114))
            canvas.paste(rgb.resize(size, Image.Resampling.BILINEAR), ((w - size[0]) // 2, (h - size[1]) // 2))
            array = np.ascontiguousarray(np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1)[None] / 255)
        require(sha256(image) == image_asset["sha256"], f"Image changed during preflight: {image}")
        yield "image:" + str(image.resolve()), array, image_asset


def run_parity(full_path, head_path, tail_path, source, output, cut, images, atol, rtol):
    import numpy as np
    import onnxruntime as ort

    ort.disable_telemetry_events()
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 2
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.log_severity_level = 3
    sessions = [ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
                for path in (full_path, head_path, tail_path)]
    cases = []
    for name, array, image in parity_inputs(source, images):
        started = time.monotonic()
        feeds = {source["name"]: array}
        full = sessions[0].run([output["name"]], feeds)
        head = sessions[1].run(cut, feeds)
        require(all(value.dtype == np.float32 and np.isfinite(value).all() for value in head), "Nonfinite/non-FP32 head output")
        tail = sessions[2].run([output["name"]], dict(zip(cut, head)))
        metrics = compare_outputs(dict(zip([output["name"]], full)), dict(zip([output["name"]], tail)), atol=atol, rtol=rtol)
        cases.append({"input": name, "input_tensor_sha256": hashlib.sha256(array.tobytes()).hexdigest(), "image": image,
                      "wall_s": time.monotonic() - started, "outputs": metrics})
    return {"provider": "CPUExecutionProvider", "dtype": "float32", "threads": {"intra": 2, "inter": 2},
            "atol": atol, "rtol": rtol, "cases": cases,
            "passed": all(row["passed"] for case in cases for row in case["outputs"].values())}


def run(args):
    report_path = args.output_dir / "report.json"
    report = {"schema_version": 1, "status": "running", "ok": False, "scope": "SDK-free static detect ONNX head/tail preflight",
        "tpu_compilation": "not_attempted", "tpu_compatibility": "unknown", "quantized_parity": "not_checked",
        "limitations": ["CPU FP32 parity does not establish SDK operator support, quantization accuracy, TPU execution or model quality",
                        "only batch-1 static FP32 detect exports with a unique canonical DFL/class head are supported"], "assets": {}}
    started = time.monotonic()
    try:
        report["assets"].update(source=asset(args.onnx), runner=asset(Path(__file__)))
        report["versions"] = {}
        for name in ("onnx", "onnxruntime", "numpy"):
            try:
                report["versions"][name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                report["versions"][name] = None
        model, values, source, output = load_static_model(args.onnx)
        report.update(input=source, output=output, operators={"full": inventory(model)},
                      opsets={item.domain or "ai.onnx": item.version for item in model.opset_import})
        selected = identify_cut(model, values, output)
        cut = args.cut or selected["cut"]
        require(cut == selected["cut"], "Explicit cut must match the unique verified DFL/classes boundary")
        report["head"] = {**selected, "tensors": [tensor_info(values[name]) for name in cut]}
        report["dependency_check"] = validate_cut(model, cut)
        head, tail, paths = extract_pair(model, cut, args.output_dir)
        report["operators"] = {name: inventory(part) for name, part in (("full", model), ("head", head), ("tail", tail))}
        report["assets"].update(head=asset(paths[0]), tail=asset(paths[1]))
        report["status"] = "graph_validated"
        report["parity"] = {"status": "skipped", "reason": "--graph-only"} if args.graph_only else {"status": "pending"}
        write_report(report_path, report)
        if not args.graph_only:
            report["parity"] = run_parity(args.onnx, *paths, source, output, cut, args.image, args.atol, args.rtol)
            require(report["parity"]["passed"], "FP32 full vs head→tail tolerance failed")
            report["status"] = "fp32_parity_passed"
        require(sha256(args.onnx) == report["assets"]["source"]["sha256"], "Source ONNX changed during preflight")
        for name, path in zip(("head", "tail"), paths):
            require(sha256(path) == report["assets"][name]["sha256"], f"Extracted {name} changed during preflight")
        report["ok"] = True
    except Exception as error:  # noqa: BLE001 — preserve inspectable JSON on every validation/ORT failure
        report.update(status="error", error={"type": type(error).__name__, "message": str(error)})
    report["wall_s"] = time.monotonic() - started
    write_report(report_path, report)
    return 0 if report["ok"] else 1


def finite_nonnegative(value):
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("must be finite and nonnegative")
    return number


def timeout_seconds(value):
    number = float(value)
    if not math.isfinite(number) or not 0 < number <= 1800:
        raise argparse.ArgumentTypeError("must be finite, >0 and <=1800 seconds")
    return number


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--onnx", required=True, type=Path)
    result.add_argument("--output-dir", type=Path)
    result.add_argument("--cut", nargs=2, metavar=("DFL_TENSOR", "CLASS_TENSOR"))
    result.add_argument("--graph-only", action="store_true", help="validate/extract without importing ORT or running inference")
    result.add_argument("--image", type=Path, action="append", default=[], help="optional image parity input (at most 16), in addition to 3 synthetic cases")
    result.add_argument("--atol", type=finite_nonnegative, default=1e-4)
    result.add_argument("--rtol", type=finite_nonnegative, default=1e-4)
    result.add_argument("--timeout", type=timeout_seconds, default=300)
    result.add_argument("--json", action="store_true")
    result.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    return result


def terminate_group(process):
    """The leader exiting after TERM must not leave an owned ORT descendant."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    time.sleep(.2)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass


class ParentInterrupted(Exception):
    def __init__(self, signum):
        self.signum = signum


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    if len(args.image) > 16:
        cli.error("at most 16 --image inputs")
    args.onnx = args.onnx.expanduser().resolve()
    if args.output_dir is None:
        data = Path(os.environ.get("GUARD_DATA_DIR") or ROOT / "data").expanduser()
        args.output_dir = data / "tpu_preflight" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8])
    args.output_dir = args.output_dir.expanduser().resolve()
    if args._worker:
        return run(args)
    if args.output_dir.exists():
        cli.error("output directory already exists; preserve earlier evidence")
    args.output_dir.mkdir(parents=True)
    command = [sys.executable, "-m", "tpu.preflight", "--_worker", "--onnx", str(args.onnx), "--output-dir", str(args.output_dir),
               "--atol", str(args.atol), "--rtol", str(args.rtol)]
    if args.graph_only:
        command.append("--graph-only")
    if args.cut:
        command.extend(["--cut", *args.cut])
    for image in args.image:
        command.extend(["--image", str(image.expanduser().resolve())])
    process, pending_signal, outcome, cleanup_error = None, None, None, None

    def interrupt(signum, frame):
        nonlocal pending_signal
        if process is None:  # Do not interrupt Popen before ownership is recorded.
            pending_signal = signum
        else:
            raise ParentInterrupted(signum)

    previous = {signum: signal.signal(signum, interrupt) for signum in (signal.SIGINT, signal.SIGTERM)}
    try:
        with (args.output_dir / "worker.log").open("w") as log:
            if pending_signal is not None:
                raise ParentInterrupted(pending_signal)
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                       env={**os.environ, "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2"})
            if pending_signal is not None:
                raise ParentInterrupted(pending_signal)
            process.wait(timeout=args.timeout)
    except subprocess.TimeoutExpired:
        outcome = {"status": "timeout", "timeout_s": args.timeout}
    except ParentInterrupted as error:
        outcome = {"status": "interrupted", "interrupt_signal": error.signum}
    except KeyboardInterrupt:
        outcome = {"status": "interrupted", "interrupt_signal": signal.SIGINT}
    except Exception as error:  # noqa: BLE001 — failure after spawn must also clean up
        outcome = {"status": "error", "error": {"type": type(error).__name__, "message": str(error)}}
    finally:
        # Repeated Ctrl-C/TERM must not interrupt the bounded TERM→KILL cleanup.
        for signum in previous:
            signal.signal(signum, signal.SIG_IGN)
        try:
            if process is not None:
                terminate_group(process)
        except Exception as error:  # noqa: BLE001
            cleanup_error = {"type": type(error).__name__, "message": str(error)}
        finally:
            for signum, handler in previous.items():
                signal.signal(signum, handler)
    report_path = args.output_dir / "report.json"
    returncode = process.returncode if process is not None else None
    report = json.loads(report_path.read_text()) if report_path.is_file() else {
        "schema_version": 1, "status": "error", "ok": False, "tpu_compilation": "not_attempted", "tpu_compatibility": "unknown",
        "error": {"message": f"Worker exited {returncode} without report"}}
    if outcome is not None:
        report.update(**outcome, ok=False)
    elif returncode != 0 and report["status"] != "error":
        report.update(status="error", ok=False, error={"message": f"Worker exited {returncode} before completing preflight"})
    if cleanup_error is not None:
        report.update(ok=False, cleanup_error=cleanup_error)
        if outcome is None:
            report["status"] = "error"
    write_report(report_path, report)  # Worker is stopped before its outcome can be overwritten.
    print(json.dumps(report) if args.json else f"{report['status']}: {report_path}")
    if outcome is not None and outcome["status"] == "interrupted":
        return 128 + outcome["interrupt_signal"]
    return 0 if report["ok"] and returncode == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
