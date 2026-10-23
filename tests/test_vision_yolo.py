import numpy as np
import pytest

from guard.contracts import ModelSpec
from guard.vision.preprocess import letterbox, make_batch
from guard.vision.yolo import decode_batch, decode_yolov8, decode_yolov8_pose


def _spec(**overrides):
    values = dict(
        model_id="det",
        onnx_path="unused.onnx",
        classes=["person", "helmet", "forklift"],
        input_size=(200, 200),
        conf_threshold=0.35,
        iou_threshold=0.5,
    )
    values.update(overrides)
    return ModelSpec(**values)


def _outputs(rows, channels_first):
    output = np.asarray(rows, dtype=np.float32)[None]
    if channels_first:
        output = output.transpose(0, 2, 1)
    return [output.copy()]


@pytest.mark.parametrize("dtype", ["float32", "uint8"])
def test_letterbox_shape_rgb_padding_and_dtype(dtype):
    frame = np.zeros((2, 4, 3), dtype=np.uint8)
    frame[:] = [10, 20, 30]
    original = frame.copy()
    tensor, meta = letterbox(frame, size=(8, 8), dtype=dtype)

    assert tensor.shape == (1, 3, 8, 8)
    assert tensor.dtype == np.dtype(dtype)
    assert tensor.flags.c_contiguous
    assert (meta.scale, meta.pad_x, meta.pad_y) == (2.0, 0, 2)
    assert (meta.orig_w, meta.orig_h) == (4, 2)
    divisor = 255 if dtype == "float32" else 1
    np.testing.assert_allclose(
        tensor[0, :, 2, 0], np.array([30, 20, 10]) / divisor, atol=1e-7
    )
    np.testing.assert_allclose(tensor[:, :, :2], 114 / divisor, atol=1e-7)
    np.testing.assert_allclose(tensor[:, :, 6:], 114 / divisor, atol=1e-7)
    if dtype == "float32":
        assert 0 <= tensor.min() <= tensor.max() <= 1
    np.testing.assert_array_equal(frame, original)


@pytest.mark.parametrize(
    "shape,size",
    [
        ((101, 203), (320, 512)),
        ((203, 101), (512, 320)),
        ((333, 517), (640, 640)),
    ],
)
def test_letterbox_coordinate_roundtrip_with_resize_rounding(shape, size):
    height, width = shape
    _, meta = letterbox(np.zeros((height, width, 3), np.uint8), size)
    original_box = np.array(
        [0.1 * width, 0.15 * height, 0.8 * width, 0.85 * height]
    )

    # Use actual rounded resize dimensions, independently of meta.scale.
    ratio = min(size[1] / width, size[0] / height)
    resized_w = round(width * ratio)
    resized_h = round(height * ratio)
    mapped = original_box * np.array(
        [resized_w / width, resized_h / height] * 2
    )
    mapped += np.array([meta.pad_x, meta.pad_y] * 2)
    cx, cy = (mapped[:2] + mapped[2:]) / 2
    box_w, box_h = mapped[2:] - mapped[:2]
    outputs = _outputs([[cx, cy, box_w, box_h, 0.9]], True)
    result = decode_yolov8(outputs, meta, _spec(classes=["person"]))

    assert len(result) == 1
    np.testing.assert_allclose(result[0].bbox, original_box, atol=1, rtol=0)


def test_odd_padding_uses_actual_top_and_left_offsets():
    image = np.zeros((3, 6, 3), np.uint8)
    tensor, meta = letterbox(image, (9, 10), dtype="uint8")
    assert meta.pad_y == 2
    assert meta.pad_x == 0
    np.testing.assert_array_equal(tensor[:, :, :2], 114)
    np.testing.assert_array_equal(tensor[:, :, 2:7], 0)
    np.testing.assert_array_equal(tensor[:, :, 7:], 114)

    tensor, meta = letterbox(image.transpose(1, 0, 2), (10, 9), dtype="uint8")
    assert meta.pad_x == 2
    np.testing.assert_array_equal(tensor[:, :, :, :2], 114)
    np.testing.assert_array_equal(tensor[:, :, :, 2:7], 0)
    np.testing.assert_array_equal(tensor[:, :, :, 7:], 114)


def test_make_batch_and_validation():
    first, _ = letterbox(np.zeros((10, 20, 3), np.uint8), (32, 32))
    second, _ = letterbox(np.full((20, 10, 3), 255, np.uint8), (32, 32))
    batch = make_batch([first, second])
    assert batch.shape == (2, 3, 32, 32)
    assert batch.dtype == np.float32
    assert batch.flags.c_contiguous
    np.testing.assert_array_equal(batch[:1], first)
    np.testing.assert_array_equal(batch[1:], second)
    with pytest.raises(ValueError):
        make_batch([])
    with pytest.raises(ValueError):
        make_batch([first, second.astype(np.uint8)])
    with pytest.raises(ValueError):
        make_batch([first, second[:, :, :16]])
    with pytest.raises(ValueError):
        make_batch([first[0]])


@pytest.mark.parametrize("channels_first", [True, False])
def test_three_known_objects_layouts_and_class_aware_nms(channels_first):
    _, meta = letterbox(np.zeros((100, 200, 3), np.uint8), (200, 200))
    rows = [
        [40, 80, 40, 40, 0.95, 0.02, 0.01],
        [41, 80, 40, 40, 0.80, 0.01, 0.01],
        [120, 100, 40, 60, 0.01, 0.90, 0.02],
        [40, 80, 40, 40, 0.01, 0.02, 0.85],
        [170, 100, 20, 20, 0.1, 0.1, 0.1],
    ]
    outputs = _outputs(rows, channels_first)
    original_output = outputs[0].copy()
    detections = decode_yolov8(outputs, meta, _spec())

    assert [item.cls for item in detections] == ["person", "helmet", "forklift"]
    assert [item.cls_id for item in detections] == [0, 1, 2]
    assert all(item.model == "det" for item in detections)
    assert all(item.keypoints is None for item in detections)
    np.testing.assert_allclose(
        [item.bbox for item in detections],
        [[20, 10, 60, 50], [100, 20, 140, 80], [20, 10, 60, 50]],
        atol=1e-5,
    )
    np.testing.assert_allclose([item.conf for item in detections], [0.95, 0.9, 0.85])
    np.testing.assert_array_equal(outputs[0], original_output)


def test_clips_boxes_and_discards_invalid_or_fully_outside_candidates():
    _, meta = letterbox(np.zeros((100, 200, 3), np.uint8), (200, 200))
    outputs = _outputs(
        [
            [0, 50, 40, 40, 0.9],
            [400, 400, 20, 20, 0.95],
            [100, 100, -20, 20, 0.99],
            [np.nan, 100, 20, 20, 0.99],
            [100, 100, 20, 20, np.nan],
            # ниже порога: делает тензор неквадратным, иначе раскладка (B, N, C) неоднозначна
            [10, 10, 5, 5, 0.01],
        ],
        False,
    )
    detections = decode_yolov8(outputs, meta, _spec(classes=["person"]))
    assert len(detections) == 1
    np.testing.assert_allclose(detections[0].bbox, [0, 0, 20, 20])


def test_confidence_threshold_includes_equality():
    _, meta = letterbox(np.zeros((100, 100, 3), np.uint8), (100, 100))
    result = decode_yolov8(
        _outputs([[50, 50, 20, 20, 0.5]], True),
        meta,
        _spec(classes=["person"], conf_threshold=0.5),
    )
    assert len(result) == 1


@pytest.mark.parametrize("channels_first", [True, False])
def test_pose_restores_keypoints_and_suppresses_duplicate(channels_first):
    _, meta = letterbox(np.zeros((100, 200, 3), np.uint8), (200, 200))
    keypoints = np.column_stack(
        [np.linspace(20, 180, 17), np.linspace(10, 90, 17), np.full(17, 0.75)]
    ).astype(np.float32)
    model_keypoints = keypoints.copy()
    model_keypoints[:, :2] *= meta.scale
    model_keypoints[:, :2] += [meta.pad_x, meta.pad_y]
    row = np.concatenate(([100, 100, 160, 80, 0.9], model_keypoints.ravel()))
    duplicate = row.copy()
    duplicate[0] += 1
    duplicate[4] = 0.6
    outputs = _outputs([row, duplicate], channels_first)
    original_output = outputs[0].copy()
    spec = _spec(model_id="pose", task="pose", output_format="yolov8-pose", classes=[])

    detections = decode_yolov8_pose(outputs, meta, spec)
    assert len(detections) == 1
    detection = detections[0]
    assert (detection.model, detection.cls, detection.cls_id) == ("pose", "person", 0)
    assert detection.keypoints.shape == (17, 3)
    np.testing.assert_allclose(detection.bbox, [20, 10, 180, 90])
    np.testing.assert_allclose(detection.keypoints, keypoints, atol=1e-5)
    np.testing.assert_array_equal(outputs[0], original_output)
    batched = decode_batch(outputs, [meta], spec)
    np.testing.assert_allclose(batched[0][0].keypoints, keypoints, atol=1e-5)


@pytest.mark.parametrize("channels_first", [True, False])
def test_batch_restores_each_frame_with_its_own_meta(channels_first):
    _, wide = letterbox(np.zeros((100, 200, 3), np.uint8), (200, 200))
    _, tall = letterbox(np.zeros((200, 100, 3), np.uint8), (200, 200))
    one = _outputs([[100, 100, 40, 60, 0.9]], channels_first)[0]
    outputs = [np.concatenate([one, one], axis=0)]
    spec = _spec(classes=["person"])
    detections = decode_batch(outputs, [wide, tall], spec)

    assert [len(items) for items in detections] == [1, 1]
    np.testing.assert_allclose(detections[0][0].bbox, [80, 20, 120, 80])
    np.testing.assert_allclose(detections[1][0].bbox, [30, 70, 70, 130])
    with pytest.raises(ValueError, match="batch size"):
        decode_batch(outputs, [wide], spec)
    with pytest.raises(ValueError, match="batch size"):
        decode_yolov8(outputs, wide, spec)


@pytest.mark.parametrize("pose", [False, True])
@pytest.mark.parametrize("channels_first", [True, False])
def test_empty_output(pose, channels_first):
    _, meta = letterbox(np.zeros((20, 30, 3), np.uint8), (32, 32))
    spec = _spec(
        task="pose" if pose else "detect",
        output_format="yolov8-pose" if pose else "yolov8",
    )
    decoder = decode_yolov8_pose if pose else decode_yolov8
    channels = 56 if pose else 7
    rows = np.empty((0, channels), dtype=np.float32)
    outputs = _outputs(rows, channels_first)
    assert decoder(outputs, meta, spec) == []
    assert decoder([], meta, spec) == []
    assert decode_batch(outputs, [meta], spec) == [[]]
    assert decode_batch([], [meta, meta], spec) == [[], []]
    assert decode_batch([], [], spec) == []


def test_all_candidates_below_threshold():
    _, meta = letterbox(np.zeros((20, 30, 3), np.uint8), (32, 32))
    outputs = _outputs([[10, 10, 5, 5, 0.1, 0.2, 0.1]], True)
    assert decode_yolov8(outputs, meta, _spec()) == []


def test_unsupported_output_shapes_are_explicit_errors():
    _, meta = letterbox(np.zeros((20, 30, 3), np.uint8), (32, 32))
    spec = _spec()
    with pytest.raises(ValueError, match="channels"):
        decode_yolov8([np.zeros((1, 8, 2))], meta, spec)
    with pytest.raises(ValueError, match="shape"):
        decode_yolov8([np.zeros((7, 2))], meta, spec)
    with pytest.raises(ValueError, match="one concatenated"):
        decode_yolov8([np.zeros((1, 7, 2)), np.zeros((1, 7, 2))], meta, spec)


@pytest.mark.parametrize(
    "image,size,dtype",
    [
        (np.zeros((0, 10, 3), np.uint8), (32, 32), "float32"),
        (np.zeros((10, 10), np.uint8), (32, 32), "float32"),
        (np.zeros((10, 10, 3), np.float32), (32, 32), "float32"),
        (np.zeros((10, 10, 3), np.uint8), (0, 32), "float32"),
        (np.zeros((10, 10, 3), np.uint8), (32, 32), "float16"),
    ],
)
def test_letterbox_invalid_inputs(image, size, dtype):
    with pytest.raises(ValueError):
        letterbox(image, size, dtype)
