"""ml/datasets/falls.py: разбиение по людям, метки отложенной выборки, разбор GMDCSA24, формат YOLO-pose."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from ml.datasets import falls

ROOT = Path(__file__).resolve().parents[1]


def test_urfd_split_is_by_person() -> None:
    by_split: dict[str, set[str]] = {}
    for seq, person in falls.URFD_PERSON.items():
        by_split.setdefault(falls._urfd_split(seq), set()).add(person)
    # Один человек — только в одной части.
    assert not by_split["test"] & by_split["train"]
    assert not by_split["test"] & by_split.get("val", set())
    assert not by_split["train"] & by_split.get("val", set())
    falls_total = sum(seq.startswith("fall") for seq in falls.URFD_PERSON)
    falls_test = sum(seq.startswith("fall") and falls._urfd_split(seq) == "test" for seq in falls.URFD_PERSON)
    assert falls_total == 30 and falls_test / falls_total >= 0.25
    adl = [seq for seq in falls.URFD_PERSON if seq.startswith("adl")]
    assert len(adl) == 40
    assert sum(falls._urfd_split(seq) == "test" for seq in adl) / len(adl) >= 0.25


def test_cauca_split_is_by_subject() -> None:
    assert len(falls.CAUCA_TEST_SUBJECTS) / 10 >= 0.25
    assert not falls.CAUCA_TEST_SUBJECTS & falls.CAUCA_VAL_SUBJECTS


def test_gmdcsa_subjects_of_s3_02_are_excluded() -> None:
    """Люди GMDCSA24 из ролика s3_02 (tools/data_manifest.json) в набор падений не берутся."""
    import json

    manifest = json.loads((ROOT / "tools" / "data_manifest.json").read_text(encoding="utf-8"))
    urls = [
        value["url"]
        for value in manifest["sources"].values()
        if isinstance(value, dict) and "GMDCSA24" in value.get("url", "")
    ]
    assert urls
    used = {int(url.split("Subject%20")[1].split("/")[0]) for url in urls}
    assert used == set(falls.GMDCSA_EXCLUDED)
    assert not used & set(falls.GMDCSA_SUBJECTS)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Falling (SW)[3.4 to 6]; Sitting[0 to 3.4]", {"Falling": (3.4, 6.0), "Sitting": (0.0, 3.4)}),
        ("Falling (BW)[2.3 6]; Sitting[0 to 2]", {"Falling": (2.3, 6.0), "Sitting": (0.0, 2.0)}),
        ("Falling (SW) [0.3 to 5]; Standing [0 to 0.2]", {"Falling": (0.3, 5.0), "Standing": (0.0, 0.2)}),
        ("Standing[0 to 0.3; 1.5 to 8]; Walking[0.4 to 1.4]", {"Standing": (0.0, 0.3), "Walking": (0.4, 1.4)}),
    ],
)
def test_parse_gmdcsa_classes(text: str, expected: dict) -> None:
    parsed = {key: value for key, value in falls.parse_gmdcsa_classes(text).items() if value is not None}
    assert parsed == expected


def test_gmdcsa_fall_onset_without_interval_uses_previous_action() -> None:
    intervals = falls.parse_gmdcsa_classes("Falling (FW); Sitting[0 to 1.7]")
    assert intervals["Falling"] is None
    assert falls.gmdcsa_fall_onset(intervals) == pytest.approx(1.7)
    assert falls.gmdcsa_fall_onset(falls.parse_gmdcsa_classes("Falling (SW)[1.8 to 4]")) == pytest.approx(1.8)


def test_gmdcsa_activity_classification() -> None:
    assert falls.gmdcsa_fall_activity("Falling (LS) on the bed from sitting on the bed") == "fall_on_bed"
    assert falls.gmdcsa_fall_activity("Falling (LS) from the bed to the floor") == "fall"
    assert falls.gmdcsa_fall_activity("Walking then falling (forward) view 1") == "fall"
    sleeping = falls.parse_gmdcsa_classes("Sitting[0 to 0.5]; Sleeping[1.6 to 5]")
    assert falls.gmdcsa_adl_activity(sleeping, "Sitting to sleeping on the bed") == "lying_bed"
    assert falls.gmdcsa_adl_activity({}, "Doing exercise (push-up); view 1") == "floor_activity"
    assert falls.gmdcsa_adl_activity({}, "Walking in the room") == "walk_sit_read"


def _series(kind: str, onset: float | None, n_frames: int, activity: str = "fall") -> falls.Series:
    return falls.Series(
        source="gmdcsa", name="s1_fall_01", person="gmdcsa:1", kind=kind, activity=activity,
        split="test", fps=30.0, n_frames=n_frames, fall_onset_s=onset,
    )


def test_holdout_label_certain_only_with_enough_lying() -> None:
    long = falls.holdout_label(_series("fall", 1.0, 150))  # 5 с, после начала 4 с
    assert long["events"] == [
        {
            "type": "fall_immobile", "t_start": 1.0, "t_end": 5.0, "uncertain": False,
            "note": long["events"][0]["note"],
        }
    ]
    short = falls.holdout_label(_series("fall", 1.0, 105))  # после начала 2.5 с
    assert short["events"][0]["uncertain"] is True
    on_bed = falls.holdout_label(_series("fall", 1.0, 150, activity="fall_on_bed"))
    assert on_bed["events"][0]["uncertain"] is True
    adl = falls.holdout_label(_series("adl", None, 150, activity="lying_bed"))
    assert adl["events"] == [] and adl["activity"] == "lying_bed"
    assert long["video"] == "falls_gmdcsa/s1_fall_01.mp4"
    with pytest.raises(ValueError):
        falls.holdout_label(_series("fall", None, 150))


def test_scenarios_falls_copy_rule_params_from_main() -> None:
    """Параметры правила в отложенной выборке — те же, что в основном scenarios.yaml, без подгонки."""
    main = yaml.safe_load((ROOT / "ml" / "eval" / "scenarios.yaml").read_text(encoding="utf-8"))
    text = falls.scenarios_yaml([_series("fall", 1.0, 150), _series("adl", None, 90, "lying_bed")])
    generated = yaml.safe_load(text)
    assert generated["rule_defaults"]["fall"] == main["rule_defaults"]["fall"]
    assert generated["sample_fps"] == main["sample_fps"]
    assert generated["labels_dir"] == "data/labels/falls"
    rules = generated["scenarios"][0]["camera"]["rules"]
    assert rules == [main["rule_defaults"]["fall"]]


def test_committed_scenarios_falls_matches_main_params() -> None:
    path = ROOT / "ml" / "eval" / "scenarios_falls.yaml"
    main = yaml.safe_load((ROOT / "ml" / "eval" / "scenarios.yaml").read_text(encoding="utf-8"))
    suite = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert suite["rule_defaults"]["fall"] == main["rule_defaults"]["fall"]
    ids = [item["id"] for item in suite["scenarios"]]
    assert len(ids) == len(set(ids)) and ids
    for item in suite["scenarios"]:
        assert item["video"].startswith("falls_")
        assert item["camera"]["rules"] == [main["rule_defaults"]["fall"]]
    # Ролики основного набора в отложенную выборку не входят.
    assert not {item["video"] for item in suite["scenarios"]} & {item["video"] for item in main["scenarios"]}


def test_yolo_pose_line_format() -> None:
    kpts = [[10.0 + i, 20.0 + i, 0.9 if i % 2 else 0.1] for i in range(17)]
    line = falls._yolo_pose_line([10.0, 20.0, 110.0, 70.0], kpts, 200, 100).split()
    assert len(line) == 5 + 17 * 3
    assert line[0] == "0"
    assert [float(v) for v in line[1:5]] == pytest.approx([0.3, 0.45, 0.5, 0.5])
    # Точка с низкой уверенностью учителя невидима и обнулена.
    assert line[5:8] == ["0.000000", "0.000000", "0"]
    assert float(line[8]) == pytest.approx(11 / 200) and line[10] == "2"
    empty = falls._yolo_pose_line([-5.0, 0.0, 50.0, 100.0], None, 200, 100).split()
    assert float(empty[1]) == pytest.approx(0.125)  # рамка обрезана по кадру
    assert set(empty[5:]) == {"0.000000", "0"}


def test_recall_targets() -> None:
    cauca = falls.Series(
        source="cauca", name="S08_FallLeft", person="cauca:S08", kind="fall", activity="FallLeft",
        split="test", fps=20.0, n_frames=10,
        boxes={1: [[0, 0.5, 0.5, 0.1, 0.4]], 5: [[1, 0.5, 0.6, 0.4, 0.2]], 9: []},
    )
    assert falls.recall_targets(cauca) == {5: [[1, 0.5, 0.6, 0.4, 0.2]]}
    assert falls.upright_targets(cauca) == {1: [[0, 0.5, 0.5, 0.1, 0.4]]}
    urfd = falls.Series(
        source="urfd", name="fall-07", person="urfd:P2", kind="fall", activity="fall",
        split="test", fps=30.0, n_frames=4, lying={1: -1, 2: 0, 3: 1, 4: 1},
    )
    assert falls.recall_targets(urfd) == {3: None, 4: None}


def test_iou_and_xyxy() -> None:
    assert falls._iou([0, 0, 10, 10], [0, 0, 10, 10]) == pytest.approx(1.0)
    assert falls._iou([0, 0, 10, 10], [10, 10, 20, 20]) == 0.0
    assert falls._xyxy([1, 0.5, 0.5, 0.2, 0.4], 100, 50) == pytest.approx([40.0, 15.0, 60.0, 35.0])
