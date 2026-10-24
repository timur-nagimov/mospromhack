"""CLI дообучения маховика: разбор аргументов, лимиты памяти, смесь наборов, карточки. Без обучения."""

from __future__ import annotations

import json
import sqlite3

import pytest
import yaml

from ml.flywheel import cards, train


def _yaml(path, names, root=None):
    root = root or path.parent
    for split in ("train", "val", "test"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump({"path": str(root), "train": "images/train", "val": "images/val",
                                    "test": "images/test", "names": dict(enumerate(names))}))
    return path


def test_parser_requires_one_data_source():
    p = train.build_parser()
    with pytest.raises(SystemExit):
        p.parse_args(["--base", "b.pt", "--model-id", "m", "--version", "v"])
    with pytest.raises(SystemExit):
        p.parse_args(["--data", "a.yaml", "--dataset", "fw-1", "--base", "b.pt", "--model-id", "m", "--version", "v"])
    args = p.parse_args(["--dataset", "fw-20261024-1500", "--mix", "a.yaml", "--mix", "b.yaml",
                         "--base", "forklift-v1", "--model-id", "forklift", "--version", "forklift-v2"])
    assert args.dataset == "fw-20261024-1500" and len(args.mix) == 2
    assert args.imgsz == 480 and args.batch == 16 and args.workers == 4 and args.time_min == 35.0


@pytest.mark.parametrize(
    "scale, batch, workers, minutes, ok",
    [("n", 16, 4, 35, True), ("n", 32, 4, 35, False), ("s", 8, 4, 35, True), ("s", 16, 4, 35, False),
     ("n", 16, 8, 35, False), ("n", 16, 4, 60, False), ("n", 16, 4, 0, False)],
)
def test_memory_and_time_limits(scale, batch, workers, minutes, ok):
    args = train.build_parser().parse_args(["--data", "a.yaml", "--base", "b.pt", "--model-id", "m",
                                            "--version", "v", "--batch", str(batch), "--workers", str(workers),
                                            "--time-min", str(minutes)])
    assert (train.check_limits(args, scale) == []) is ok


@pytest.mark.parametrize("name, scale", [("yolov8n.pt", "n"), ("yolov8s-pose.pt", "s"), ("yolo11m.pt", "m")])
def test_model_scale_by_name(tmp_path, name, scale):
    assert train.model_scale(tmp_path / name) == scale


def test_mix_data_combines_train_and_val_keeps_primary_test(tmp_path):
    a = _yaml(tmp_path / "a" / "data.yaml", ["person", "forklift"], tmp_path / "a")
    b = _yaml(tmp_path / "b" / "data.yaml", ["person", "forklift"], tmp_path / "b")
    out, names = train.mix_data(a, [b], tmp_path / "run" / "mix.yaml")
    mixed = yaml.safe_load(out.read_text())
    assert names == ["person", "forklift"]
    assert mixed["train"] == [str(tmp_path / "a/images/train"), str(tmp_path / "b/images/train")]
    assert mixed["val"] == [str(tmp_path / "a/images/val"), str(tmp_path / "b/images/val")]
    assert mixed["test"] == [str(tmp_path / "a/images/test")]
    assert mixed["names"] == {0: "person", 1: "forklift"}


def test_mix_data_refuses_different_classes(tmp_path):
    a = _yaml(tmp_path / "a" / "data.yaml", ["person", "forklift"], tmp_path / "a")
    b = _yaml(tmp_path / "b" / "data.yaml", ["forklift", "person"], tmp_path / "b")
    with pytest.raises(ValueError, match="классы"):
        train.mix_data(a, [b], tmp_path / "mix.yaml")


def test_resolve_dataset_from_flywheel_db(tmp_path):
    fw = tmp_path / "flywheel"
    (fw / "datasets" / "fw-20261024-1500").mkdir(parents=True)
    with sqlite3.connect(fw / "flywheel.db") as con:
        con.execute("CREATE TABLE datasets (version TEXT PRIMARY KEY, path TEXT)")
        con.execute("INSERT INTO datasets VALUES ('fw-20261024-1500', 'datasets/fw-20261024-1500')")
    path = train.resolve_dataset("fw-20261024-1500", fw)
    assert path == fw / "datasets" / "fw-20261024-1500" / "data.yaml"
    with pytest.raises(ValueError):
        train.resolve_dataset("fw-none", fw)


def test_read_results_picks_best_epoch(tmp_path):
    (tmp_path / "results.csv").write_text(
        "epoch,metrics/mAP50(B),metrics/mAP50-95(B)\n1,0.5,0.3\n2,0.7,0.4\n3,0.6,0.45\n")
    assert train.read_results(tmp_path) == {"epochs_done": 3, "best_epoch": 2, "val_map50": 0.7,
                                            "val_map50_95": 0.4}


def test_dataset_entries_from_manifest(tmp_path):
    data = _yaml(tmp_path / "data.yaml", ["person", "forklift"])
    (tmp_path / "manifest.json").write_text(json.dumps({
        "version": "forklift-ds-v1",
        "sources": {"forklift_rf": {"url": "https://hf/forklift"}, "coco_vehicles": {"url": "https://coco"}},
        "splits": {"train": {"forklift_rf": {"images": 295}, "coco_vehicles": {"images": 480}}},
    }))
    entries = train.dataset_entries(data, [])
    assert {e["name"]: e["images"] for e in entries} == {"forklift_rf": 295, "coco_vehicles": 480}


def test_dry_run_checks_inputs_without_training(tmp_path, monkeypatch, capsys):
    data = _yaml(tmp_path / "data.yaml", ["person", "forklift"])
    base = tmp_path / "yolov8n.pt"
    base.write_bytes(b"")
    monkeypatch.setattr(train, "RUNS_DIR", tmp_path / "runs")
    code = train.main(["--data", str(data), "--base", str(base), "--model-id", "forklift",
                       "--version", "forklift-vt", "--dry-run"])
    assert code == 0
    assert (tmp_path / "runs" / "forklift-vt" / "data_mix.yaml").exists()
    assert "классы ['person', 'forklift']" in capsys.readouterr().out


def test_too_big_batch_stops_before_training(tmp_path, monkeypatch):
    data = _yaml(tmp_path / "data.yaml", ["person"])
    base = tmp_path / "yolov8s.pt"
    base.write_bytes(b"")
    monkeypatch.setattr(train, "RUNS_DIR", tmp_path / "runs")
    assert train.main(["--data", str(data), "--base", str(base), "--model-id", "m", "--version", "v",
                       "--batch", "16"]) == 2


def test_card_roundtrip_and_validation(tmp_path):
    card = {"model_id": "forklift", "version": "forklift-vt", "parent": "yolov8n-coco",
            "created_at": cards.now_iso(), "task": "detect", "classes": ["person", "forklift"],
            "status": "candidate"}
    path = cards.save_card(card, tmp_path)
    assert path == tmp_path / "forklift-vt.json"
    assert cards.load_card(path)["version"] == "forklift-vt"
    assert [c["version"] for c in cards.list_cards(tmp_path)] == ["forklift-vt"]
    with pytest.raises(ValueError):
        cards.save_card({**card, "status": "maybe"}, tmp_path)
    with pytest.raises(ValueError):
        cards.validate_card({"version": "x"})


def test_repository_cards_follow_schema():
    for path in cards.CARDS_DIR.glob("*.json"):
        cards.validate_card(json.loads(path.read_text(encoding="utf-8")))
