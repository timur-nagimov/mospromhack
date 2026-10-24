"""Дообучение pose через ml/flywheel/train.py: kpt_shape и flip_idx не теряются в смеси, задача в карточке."""

from __future__ import annotations

import pytest
import yaml

from ml.flywheel import train

FLIP = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]


def _pose_yaml(root, kpt_shape=(17, 3)):
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
    path = root / "data.yaml"
    meta = {"path": str(root), "train": "images/train", "val": "images/val", "names": {0: "person"}}
    if kpt_shape:
        meta.update(kpt_shape=list(kpt_shape), flip_idx=FLIP)
    path.write_text(yaml.safe_dump(meta))
    return path


def test_mix_keeps_kpt_shape_and_flip_idx(tmp_path):
    a = _pose_yaml(tmp_path / "a")
    b = _pose_yaml(tmp_path / "b")
    out, names = train.mix_data(a, [b], tmp_path / "run" / "mix.yaml")
    mixed = yaml.safe_load(out.read_text())
    assert names == ["person"]
    assert mixed["kpt_shape"] == [17, 3] and mixed["flip_idx"] == FLIP
    assert mixed["train"] == [str(tmp_path / "a/images/train"), str(tmp_path / "b/images/train")]


def test_mix_refuses_detect_set_in_pose_mix(tmp_path):
    a = _pose_yaml(tmp_path / "a")
    b = _pose_yaml(tmp_path / "b", kpt_shape=None)
    with pytest.raises(ValueError, match="kpt_shape"):
        train.mix_data(a, [b], tmp_path / "mix.yaml")


def test_detect_mix_has_no_pose_keys(tmp_path):
    a = _pose_yaml(tmp_path / "a", kpt_shape=None)
    out, _ = train.mix_data(a, [], tmp_path / "mix.yaml")
    mixed = yaml.safe_load(out.read_text())
    assert "kpt_shape" not in mixed and "flip_idx" not in mixed


def test_task_from_data_yaml(tmp_path):
    assert train.data_task(_pose_yaml(tmp_path / "a")) == "pose"
    assert train.data_task(_pose_yaml(tmp_path / "b", kpt_shape=None)) == "detect"
