"""Маховик: версия датасета в YOLO — структура, разбиение по камере и времени, holdout, hard negatives, манифест."""

import json

import pytest
import yaml

from guard.flywheel.store import FlywheelStore
from ml.flywheel import export as fw_export
from ml.flywheel.export import ExportError, export_dataset, split_samples


def _add(store, sid, *, camera="cam1", ts=0.0, labels=(), status="accepted", holdout=0, source="event",
         operator_label=None):
    store.write_frame(f"frames/{camera}/{sid}.jpg", b"\xff\xd8\xff" + sid.encode())
    store.add({"id": sid, "camera_id": camera, "source": source, "width": 200, "height": 100, "created_at": ts,
               "holdout": holdout, "operator_label": operator_label,
               "event_id": f"ev-{sid}" if source == "event" else None})
    if labels:
        store.update(sid, labels=list(labels))
    store.update(sid, status=status)


@pytest.fixture()
def store(tmp_path):
    s = FlywheelStore(tmp_path / "fw")
    yield s
    s.close()


def _forklift(x=20):
    return {"cls": "forklift", "bbox": [x, 10, x + 100, 90], "origin": "teacher", "conf": 0.8}


def _person():
    return {"cls": "person", "bbox": [0, 0, 50, 100], "origin": "operator"}


def test_export_structure_labels_and_registry(store):
    t0 = 1_700_000_000.0
    _add(store, "a", ts=t0, labels=[_forklift(), _person()])
    _add(store, "b", ts=t0 + 5, labels=[])  # принят без рамок — hard negative
    _add(store, "c", ts=t0 + 1300, camera="cam2", labels=[_forklift(50)])
    _add(store, "n", ts=t0 + 10, labels=[_forklift()], status="auto")  # не проверен человеком — не идёт
    _add(store, "r", ts=t0 + 20, labels=[_forklift()], status="rejected")
    row = export_dataset(store, now=t0 + 2000)
    root = store.root / "datasets" / row["version"]
    assert row["version"].startswith("fw-") and row["path"] == str(root)
    assert row["classes"] == ["forklift", "person"]  # без --classes: отсортированное объединение меток
    assert row["n_train"] + row["n_val"] == 3 and row["n_hard_negative"] == 1
    data = yaml.safe_load((root / "data.yaml").read_text(encoding="utf-8"))
    assert data["train"] == "images/train" and data["val"] == "images/val" and data["names"] == {0: "forklift", 1: "person"}
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest == row["manifest"]
    assert manifest["holdout_contaminated"] is False and manifest["by_camera"] == {"cam1": 2, "cam2": 1}
    assert set(manifest["sha256"]) == {"train", "val"} and manifest["labels_by_class"] == {"forklift": 2, "person": 1}
    found = {p.stem: p for p in root.glob("labels/*/*.txt")}
    assert set(found) == {"a", "b", "c"}
    lines = found["a"].read_text(encoding="utf-8").splitlines()
    assert lines[0] == "0 0.350000 0.500000 0.500000 0.800000"  # forklift 20..120 × 10..90 в кадре 200×100
    assert lines[1].startswith("1 0.125000 0.500000 0.250000 1.000000")
    assert found["b"].read_text(encoding="utf-8") == ""
    image = next(root.glob("images/*/a.jpg"))
    assert image.read_bytes() == store.resolve_frame("frames/cam1/a.jpg").read_bytes()
    assert store.get_dataset(row["version"]) is not None
    assert store.get("a")["dataset_version"] == row["version"] and store.get("n")["dataset_version"] is None
    second = export_dataset(store, now=t0 + 2000)  # та же минута — суффикс версии
    assert second["version"] == row["version"] + "-2"


def test_holdout_is_skipped_unless_flag(store):
    _add(store, "live", ts=0.0, labels=[_forklift()])
    _add(store, "held", ts=5.0, labels=[_forklift()], holdout=1)
    row = export_dataset(store, now=100.0)
    ids = row["manifest"]["samples"]["train"] + row["manifest"]["samples"]["val"]
    assert ids == ["live"] and row["manifest"]["holdout_contaminated"] is False
    dirty = export_dataset(store, include_holdout=True, now=200.0)
    ids = dirty["manifest"]["samples"]["train"] + dirty["manifest"]["samples"]["val"]
    assert set(ids) == {"live", "held"}
    assert dirty["manifest"]["holdout_contaminated"] is True and "загрязнена" in dirty["manifest"]["warning"]


def test_explicit_classes_order_and_foreign_labels(store):
    _add(store, "a", labels=[_forklift(), _person(), {"cls": "helmet", "bbox": [5, 5, 15, 15], "origin": "teacher"}])
    row = export_dataset(store, classes=["person", "forklift"], now=0.0)
    root = store.root / "datasets" / row["version"]
    lines = next(root.glob("labels/*/a.txt")).read_text(encoding="utf-8").splitlines()
    assert sorted(int(line.split()[0]) for line in lines) == [0, 1]  # person = 0, forklift = 1
    assert row["manifest"]["skipped_labels"] == 1  # helmet не в списке классов
    with pytest.raises(ExportError):
        export_dataset(store, classes=["a", "a"])


def test_split_keeps_camera_time_windows_together():
    samples = []
    for cam in ("cam1", "cam2", "cam3"):
        for window in range(6):
            for k in range(5):  # пять соседних кадров одного окна
                samples.append({"id": f"{cam}-{window}-{k}", "camera_id": cam, "created_at": window * 600.0 + k})
    train, val, method = split_samples(samples, 0.2)
    assert method == "camera_time_600s" and train and val and len(train) + len(val) == len(samples)
    side = {}
    for part, items in (("train", train), ("val", val)):
        for s in items:
            key = (s["camera_id"], int(s["created_at"] // 600))
            assert side.setdefault(key, part) == part  # окно целиком по одну сторону
    one_window = [{"id": f"w{k}", "camera_id": "cam1", "created_at": 30.0 * k} for k in range(10)]
    single, single_val, single_method = split_samples(one_window, 0.2)  # одно окно одной камеры
    assert single_method == "time_within_single_window" and len(single_val) == 2
    assert max(s["created_at"] for s in single) <= min(s["created_at"] for s in single_val)


def test_nothing_to_export_and_cli(store, tmp_path, capsys):
    with pytest.raises(ExportError):
        export_dataset(store)
    _add(store, "a", labels=[_forklift()])
    code = fw_export.main(["--flywheel-dir", str(store.root), "--classes", "forklift",
                           "--mix", f"forklift-rf={tmp_path / 'nope'}"])
    assert code == 0
    out = capsys.readouterr().out
    assert "train 1" in out and "classes ['forklift']" in out
    row = store.last_dataset()
    assert row["manifest"]["open_datasets"] == [{"name": "forklift-rf", "path": str(tmp_path / "nope")}]
    assert fw_export.main(["--flywheel-dir", str(tmp_path / "empty")]) == 2
