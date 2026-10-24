"""Маховик: хранилище FlywheelStore (SQLite + кадры), фильтры, решение человека, сводка, версии датасетов."""

import time

import pytest

from guard.flywheel.store import FlywheelStore, SampleError, day_start, frame_rel_path, safe_name


def _sample(**kw):
    base = {
        "camera_id": "cam1",
        "source": "uncertain",
        "width": 640,
        "height": 360,
        "created_at": 1_700_000_000.0,
        "detections": [{"model_id": "det", "cls": "person", "conf": 0.42, "bbox": [10, 20, 110, 220], "track_id": 3}],
        "reason": "person conf 0.42 ∈ [0.30, 0.45]",
    }
    base.update(kw)
    return base


@pytest.fixture()
def store(tmp_path):
    s = FlywheelStore(tmp_path / "fw")
    yield s
    s.close()


def test_add_get_roundtrip_and_idempotent_insert(store):
    stored = store.add(_sample(id="a1", event_id=None))
    assert stored["id"] == "a1" and stored["status"] == "new" and stored["holdout"] == 0
    assert stored["frame_path"] == "frames/cam1/a1.jpg"
    assert stored["detections"][0] == {"model_id": "det", "cls": "person", "conf": 0.42,
                                       "bbox": [10.0, 20.0, 110.0, 220.0], "track_id": 3}
    assert stored["labels"] == [] and stored["meta"] == {}
    again = store.add(_sample(id="a1", reason="другая причина"))  # повторная доставка не меняет строку
    assert again["reason"] == stored["reason"]
    assert store.count() == 1


def test_validation(store):
    with pytest.raises(SampleError):
        store.add(_sample(source="random"))
    with pytest.raises(SampleError):
        store.add(_sample(width=0))
    with pytest.raises(SampleError):
        store.add(_sample(detections=[{"cls": "person", "conf": 0.5, "bbox": [10, 10, 5, 20]}]))
    with pytest.raises(SampleError):
        store.add(_sample(operator_label="maybe"))


def test_frames_are_written_atomically_and_resolved_inside_store(store, tmp_path):
    rel = frame_rel_path("cam/../../x", "id1")
    assert ".." not in rel and rel.startswith("frames/")
    path = store.write_frame(rel, b"\xff\xd8jpeg")
    assert path.read_bytes() == b"\xff\xd8jpeg"
    assert store.resolve_frame(rel) == path
    assert store.resolve_frame("../outside.jpg") is None
    assert store.resolve_frame("frames/none.jpg") is None
    with pytest.raises(SampleError):
        store.write_frame("../../escape.jpg", b"x")
    assert safe_name("") == "camera"


def test_query_filters_order_and_paging(store):
    store.add(_sample(id="e1", source="event", event_id="ev1", created_at=100.0))
    store.add(_sample(id="u1", source="uncertain", created_at=200.0, holdout=1))
    store.add(_sample(id="f1", source="flicker", camera_id="cam2", created_at=300.0,
                      detections=[{"model_id": "ppe", "cls": "no_helmet", "conf": 0.4, "bbox": [1, 1, 5, 5]}]))
    store.update("u1", labels=[{"cls": "forklift", "bbox": [0, 0, 50, 50], "origin": "teacher", "conf": 0.8}],
                 status="auto")

    items, total = store.query()
    assert total == 3 and [s["id"] for s in items] == ["f1", "u1", "e1"]  # -created_at по умолчанию
    assert [s["id"] for s in store.query(order="created_at")[0]] == ["e1", "u1", "f1"]
    assert [s["id"] for s in store.query(source="event,flicker")[0]] == ["f1", "e1"]
    assert [s["id"] for s in store.query(camera_id="cam2")[0]] == ["f1"]
    assert [s["id"] for s in store.query(status="auto")[0]] == ["u1"]
    assert [s["id"] for s in store.query(holdout=True)[0]] == ["u1"]
    assert {s["id"] for s in store.query(holdout=False)[0]} == {"e1", "f1"}
    assert [s["id"] for s in store.query(event_id="ev1")[0]] == ["e1"]
    assert [s["id"] for s in store.query(cls="forklift")[0]] == ["u1"]  # класс в метках
    assert [s["id"] for s in store.query(cls="no_helmet")[0]] == ["f1"]  # класс в детекциях
    page, total = store.query(limit=1, offset=1)
    assert total == 3 and [s["id"] for s in page] == ["u1"]
    assert store.query(limit=0)[1] == 3
    with pytest.raises(SampleError):
        store.query(order="name")


def test_review_marks_changed_labels_as_operator(store):
    store.add(_sample(id="s1"))
    teacher = [
        {"cls": "person", "bbox": [10, 20, 110, 220], "origin": "teacher", "conf": 0.91},
        {"cls": "forklift", "bbox": [200, 50, 400, 300], "origin": "teacher", "conf": 0.55},
    ]
    store.update("s1", labels=teacher, status="auto")
    t0 = time.time()
    out = store.review(
        "s1",
        status="accepted",
        labels=[
            {"cls": "person", "bbox": [10, 20, 110, 220]},  # без изменений — origin teacher сохраняется
            {"cls": "truck", "bbox": [200, 50, 400, 300]},  # смена класса — origin operator
            {"cls": "helmet", "bbox": [40, 20, 70, 45], "origin": "teacher"},  # новая рамка — operator
        ],
    )
    assert out["status"] == "accepted" and out["reviewed_at"] >= t0
    origins = {lb["cls"]: (lb["origin"], lb["conf"]) for lb in out["labels"]}
    assert origins == {"person": ("teacher", 0.91), "truck": ("operator", None), "helmet": ("operator", None)}
    with pytest.raises(SampleError):
        store.review("s1", status="auto")
    assert store.review("missing", status="accepted") is None


def test_operator_label_goes_to_event_samples_only(store):
    store.add(_sample(id="e1", source="event", event_id="ev7"))
    store.add(_sample(id="u1", source="uncertain", event_id=None))
    assert store.set_operator_label("ev7", "hard_negative") == 1
    assert store.get("e1")["operator_label"] == "hard_negative"
    assert store.get("u1")["operator_label"] is None
    assert store.set_operator_label("ev7", None) == 1  # оператор вернул статус new
    assert store.get("e1")["operator_label"] is None
    with pytest.raises(SampleError):
        store.set_operator_label("ev7", "wrong")


def test_summary_counts_and_daily_cap_counter(store):
    now = time.time()
    store.add(_sample(id="e1", source="event", event_id="ev1", created_at=now))
    store.add(_sample(id="u1", source="uncertain", created_at=now, holdout=1))
    store.add(_sample(id="o1", source="operator", created_at=now))
    store.add(_sample(id="old", source="flicker", created_at=day_start(now) - 10))
    store.update("u1", labels=[{"cls": "forklift", "bbox": [0, 0, 5, 5], "origin": "teacher"}], status="auto")
    store.review("e1", status="rejected")
    s = store.summary(now=now)
    assert s["total"] == 4 and s["today"] == 3
    assert s["by_source"]["event"] == 1 and s["by_source"]["disagreement"] == 0
    assert s["by_status"] == {"new": 2, "auto": 1, "accepted": 0, "rejected": 1}
    assert s["by_class"] == {"forklift": 1}
    assert s["by_detected_class"]["person"] == 4
    assert s["reviewed"] == 1 and s["holdout"] == 1 and s["last_dataset"] is None
    # под суточный лимит попадает всё, кроме event и operator, и только за сегодня
    assert store.capped_today(now=now) == {"cam1": 1}


def test_datasets_registry_and_marks(store):
    store.add(_sample(id="a"))
    store.add(_sample(id="b"))
    row = store.add_dataset({"version": "fw-20261024-1500", "path": "/x", "classes": ["person"], "n_train": 1,
                             "n_val": 1, "n_hard_negative": 0, "manifest": {"k": 1}, "created_at": 10.0})
    assert row["classes"] == ["person"] and row["manifest"] == {"k": 1}
    store.add_dataset({"version": "fw-20261024-1600", "path": "/y", "created_at": 20.0})
    assert [d["version"] for d in store.list_datasets()] == ["fw-20261024-1600", "fw-20261024-1500"]
    assert store.last_dataset()["version"] == "fw-20261024-1600"
    assert store.mark_dataset_version(["a", "b"], "fw-20261024-1600") == 2
    assert store.get("a")["dataset_version"] == "fw-20261024-1600"


def test_second_connection_sees_writes(tmp_path):
    """CLI (export, autolabel) и сервис работают с одной БД одновременно."""
    a = FlywheelStore(tmp_path / "fw")
    b = FlywheelStore(tmp_path / "fw")
    try:
        a.add(_sample(id="x"))
        assert b.get("x") is not None
        b.update("x", status="auto")
        assert a.get("x")["status"] == "auto"
    finally:
        a.close()
        b.close()
