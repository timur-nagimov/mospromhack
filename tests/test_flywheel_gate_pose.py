"""Гейт маховика для pose-моделей (ds-falls): формат выхода ONNX определяется по числу каналов."""

from __future__ import annotations

import pytest

from ml.flywheel import gate


@pytest.mark.parametrize(
    "channels, classes, expected",
    [
        (56, ["person"], "yolov8-pose"),  # рамка, скор person и 17 точек × 3
        (5, ["person"], "yolov8"),
        (6, ["person", "forklift"], "yolov8"),
        (84, [f"c{i}" for i in range(80)], "yolov8"),
    ],
)
def test_output_format_by_channels(channels, classes, expected):
    assert gate.output_format_for(channels, classes) == expected
