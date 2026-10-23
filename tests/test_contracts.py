import numpy as np

from guard.contracts import CameraConfig, FrameContext, RuleConfig, SourceConfig, Zone
from guard.rules.base import available_rules


def test_camera_config_roundtrip():
    cam = CameraConfig(
        camera_id="cam1",
        name="Склад",
        site="Склад",
        source=SourceConfig(type="file", uri="data/videos/demo.mp4"),
        zones=[Zone(zone_id="z1", name="Стеллажи", polygon=[(0.1, 0.1), (0.5, 0.1), (0.5, 0.9)])],
        rules=[RuleConfig(rule_id="r1", type="zone_intrusion", zones=["z1"])],
    )
    assert CameraConfig.model_validate_json(cam.model_dump_json()) == cam


def test_frame_context_size():
    cam = CameraConfig(camera_id="c", name="c", site="s", source=SourceConfig(type="file", uri="x"))
    ctx = FrameContext(camera=cam, ts=0.0, frame_idx=0, frame=np.zeros((480, 640, 3), np.uint8))
    assert ctx.size == (640, 480)


def test_available_rules_is_list():
    assert isinstance(available_rules(), list)
