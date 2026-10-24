"""Pinned HF raw D-FINE contract inspection, without starting ONNX Runtime."""

from __future__ import annotations

import json
from pathlib import Path

from tools.prepare_ppe_candidate import asset, require

DFINE_REVISION = "4d3a85a6c29f55a5346fabaf9728f5f3751f947a"
DFINE_FILES = {
    "model.onnx": "486ee1cba40b25f3ea8d6783f664eca88b7c12d01078984834d36834d9b2c802",
    "config.json": "c3b580f1c68013a7c5737e0f2fdf6ddf3434cbb947b10d7388e0c5b652934317",
    "preprocessor_config.json": "cd38cd59999e7a95d68e487fbe5132df3d4e5c32a0836add57e6126ba0c4eaf1",
}


def inspect_dfine_onnx(path: Path, classes: list[str], input_size=(640, 640)) -> dict:
    """Validate the specific raw ONNX + text pair before creating any session.

    A dynamic graph is accepted only for batch/height/width; evaluation still
    sends B=1,640x640. An upstream artifact change requires a reviewed new pin.
    """
    import onnx

    require(tuple(input_size) == (640, 640), "Pinned D-FINE profile requires 640x640")
    files = {name: asset(path if name == "model.onnx" else path.parent / name) for name in DFINE_FILES}
    for name, expected in DFINE_FILES.items():
        require(files[name]["sha256"] == expected, f"Pinned D-FINE {name} SHA256 mismatch")
    config = json.loads((path.parent / "config.json").read_text(encoding="utf-8"))
    labels = config["id2label"]
    require(set(labels) == {str(index) for index in range(80)}, "D-FINE needs contiguous 80-class id2label")
    source_classes = [labels[str(index)] for index in range(80)]
    require(classes == source_classes and source_classes[0] == "person" and source_classes[67] == "cell phone",
            "D-FINE registry classes differ from pinned config id2label")
    require(config["use_focal_loss"] is True and config["num_queries"] == 300
            and config["model_type"] == "d_fine", "Unsupported D-FINE decoding configuration")
    processor = json.loads((path.parent / "preprocessor_config.json").read_text(encoding="utf-8"))
    expected_processor = {"do_resize": True, "do_rescale": True, "do_normalize": False, "do_pad": False,
                          "size": {"height": 640, "width": 640}, "resample": 2,
                          "rescale_factor": 1 / 255, "image_processor_type": "RTDetrImageProcessor"}
    require(all(processor.get(key) == value for key, value in expected_processor.items()),
            "D-FINE processor differs from direct PIL bilinear RGB /255 contract")
    model = onnx.load(str(path), load_external_data=False)
    require(not model.functions and all(node.domain in ("", "ai.onnx") for node in model.graph.node),
            "Custom ONNX operators/functions are unsupported")
    require(not any(t.external_data or t.data_location == onnx.TensorProto.EXTERNAL
                    for t in model.graph.initializer), "External model data is unsupported")
    require(not any(node.op_type == "NonMaxSuppression" for node in model.graph.node), "Embedded NMS is unsupported")
    onnx.checker.check_model(model, full_check=True)
    values = [*model.graph.input, *model.graph.output]
    require(len(model.graph.input) == 1 and len(model.graph.output) == 2
            and [value.name for value in values] == ["pixel_values", "logits", "pred_boxes"],
            "D-FINE requires ordered pixel_values -> logits, pred_boxes")
    require(all(value.type.tensor_type.elem_type == onnx.TensorProto.FLOAT for value in values),
            "D-FINE input/output must be float32")
    shapes = [[dim.dim_value if dim.HasField("dim_value") else dim.dim_param
               for dim in value.type.tensor_type.shape.dim] for value in values]
    require([len(shape) for shape in shapes] == [4, 3, 3], "D-FINE graph rank mismatch")
    batch, channels, height, width = shapes[0]
    dynamic_or_equal = lambda actual, expected: actual == expected or isinstance(actual, str) and bool(actual)
    require(dynamic_or_equal(batch, 1) and channels == 3
            and dynamic_or_equal(height, 640) and dynamic_or_equal(width, 640)
            and shapes[1] == [batch, 300, 80] and shapes[2] == [batch, 300, 4],
            "D-FINE graph shape mismatch")
    return {"checker": "passed", "input_name": "pixel_values", "output_names": ["logits", "pred_boxes"],
            "input_shape": shapes[0], "output_shapes": shapes[1:], "effective_input_shape": [1, 3, 640, 640],
            "input_dtype": "float32", "output_dtype": "float32", "output_format": "dfine",
            "classes": source_classes, "queries": 300, "top_k": 300, "use_focal_loss": True,
            "preprocessing": "RGB PIL BILINEAR direct resize; float32 /255 NCHW; no normalize/pad",
            "source_repository": "onnx-community/dfine_x_obj2coco-ONNX", "revision": DFINE_REVISION,
            "files": files, "tpu_status": "not_compiled_or_verified"}
