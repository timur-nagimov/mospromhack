"""Маховик: holdout по источнику камеры — кадры роликов офлайн-оценки не идут в обучение."""

from types import SimpleNamespace

import numpy as np
import yaml

from guard.contracts import CameraConfig, FrameContext, SourceConfig
from guard.flywheel.collector import FlywheelCollector
from guard.flywheel.holdout import holdout_names, is_holdout_uri, load_holdout_names


def test_repo_scenarios_are_holdout():
    names = holdout_names()
    for video in ("s1_01", "s2_01", "s3_02", "s4_02", "clean_01", "s7_01"):
        assert video in names
    # демо-камеры configs/cameras.yaml и субпоток стенда — ролики оценки
    assert is_holdout_uri("videos/s1_01.mp4", names)
    assert is_holdout_uri("/Users/x/work/mosprom/data_360/videos/s3_02.mp4", names)
    assert is_holdout_uri("data/images/s2_01_frames", names)
    assert is_holdout_uri("images/s2", names)  # каталог кадров серии s2_01 и s2_02
    # чужие ролики и потоки — не отложенные
    assert not is_holdout_uri("videos/a3dev_worker_zone_detection/worker-zone-detection.mp4", names)
    assert not is_holdout_uri("rtsp://10.0.0.5/stream1", names)
    assert not is_holdout_uri("videos/s4_04.mp4", names)
    assert not is_holdout_uri("", names) and not is_holdout_uri(None, names)


def test_extra_scenarios_files_are_holdout_too(tmp_path):
    (tmp_path / "scenarios.yaml").write_text(yaml.safe_dump({"scenarios": [{"id": "s1_01", "video": "s1_01.mp4"}]}),
                                             encoding="utf-8")
    (tmp_path / "scenarios_forklift.yaml").write_text(
        yaml.safe_dump({"scenarios": [{"id": "fk_a", "video": "forklift_dock_03.mp4"}]}), encoding="utf-8")
    (tmp_path / "other.yaml").write_text(yaml.safe_dump({"scenarios": [{"video": "nope.mp4"}]}), encoding="utf-8")
    names = load_holdout_names(tmp_path)
    assert names == {"s1_01", "fk_a", "forklift_dock_03"}
    assert is_holdout_uri("videos/forklift_dock_03.mp4", names)
    assert not is_holdout_uri("videos/nope.mp4", names)


class _Out:
    def __init__(self):
        self.jobs = []

    def put(self, job):
        self.jobs.append(job)
        return True


def _ctx(cam):
    return FrameContext(camera=cam, ts=1.0, frame_idx=0, frame=np.zeros((8, 8, 3), np.uint8))


def test_collector_marks_samples_by_camera_source():
    names = frozenset({"s1_01"})
    out = _Out()
    col = FlywheelCollector(out, holdout=names)
    ev = SimpleNamespace(id="e", type="zone_intrusion", rule_id="r", zone_id=None, track_id=None, bbox=(0, 0, 1, 1))
    held = CameraConfig(camera_id="a", name="a", site="s", source=SourceConfig(type="file", uri="videos/s1_01.mp4"))
    live = CameraConfig(camera_id="b", name="b", site="s", source=SourceConfig(type="rtsp", uri="rtsp://cam/1"))
    col.on_event(_ctx(held), ev)
    col.on_event(_ctx(live), ev)
    assert [j.sample["holdout"] for j in out.jobs] == [1, 0]
    # камеру перенастроили на другой источник — признак пересчитывается
    moved = held.model_copy(update={"source": SourceConfig(type="rtsp", uri="rtsp://cam/2")})
    col.on_event(_ctx(moved), ev)
    assert out.jobs[-1].sample["holdout"] == 0 and col.is_holdout(moved) is False
