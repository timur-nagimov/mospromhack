"""Model-free ROI contracts with the production detector, tracker and worker."""

import copy
import hashlib
import json
import threading
from concurrent.futures import Future
from types import SimpleNamespace

import numpy as np
import pytest

from guard.contracts import CameraConfig, Detection, ModelSpec, Track
from guard.pipeline import compat
from guard.pipeline.detector import BatchScheduler, Detector
from guard.pipeline.ppe import PpeFrameError, PpePolicy, attribute_coverage, crop_rect, infer_ppe
from guard.pipeline.testing import make_settings
from guard.pipeline.worker import CameraWorker
from guard.metrics import MetricsRegistry
from guard.sources import FramePacket
from guard.tracking import assign_attributes
from ml.eval import run_s2


def person(track_id=1, bbox=(30, 20, 130, 120), ts=1):
    return Track(track_id, "person", bbox, .9, 0, ts, hits=4)


def ppe(cls="helmet", bbox=(40, 20, 60, 30), conf=.9):
    return Detection("ppe", cls, 0, conf, bbox)


class ScriptedPpe:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.calls = []

    def detect(self, frame, timings=None, **kwargs):
        self.calls.append((frame.shape, kwargs.copy()))
        value = next(self.outputs)
        if isinstance(value, Exception):
            raise value
        if timings is not None:
            timings["infer"] = 1
        return value


def apply(result, people, ts, others=None):
    assign_attributes(people, {"ppe": result.detections, **(others or {})}, ts,
                      ppe_by_track=result.by_track, ppe_unknown=result.unknown)


@pytest.mark.parametrize("box,expected", [
    ((30.2, 20.4, 130.3, 120.6), (20, 10, 141, 131)),
    ((-20, 10, 80, 110), (0, 0, 90, 120)),
    ((170, 170, 220, 220), (165, 165, 200, 200)),
    ((1, 1, 1, 20), None), ((300, 0, 320, 10), None), ((0, 0, float("nan"), 3), None),
])
def test_padding_floor_ceil_and_actual_clipping(box, expected):
    assert crop_rect(box, 200, 200) == expected


@pytest.mark.parametrize("box,raw,expected,shape", [
    ((30, 20, 130, 120), [50, 25, 20, 10, .9], (60, 30, 80, 40), (120, 120, 3)),
    # Non-square crop: real letterbox undoes x padding before actual origin offset.
    ((-20, 10, 80, 110), [45, 17, 20, 10, .9], (20, 12, 40, 22), (120, 90, 3)),
])
def test_real_preprocessor_decoder_existing_backend_and_per_call_hashes(box, raw, expected, shape):
    class Backend:
        def __init__(self):
            self.inputs, self.outputs = [], []

        def infer_timed(self, model_id, tensor):
            assert model_id == "ppe"
            self.inputs.append(tensor.copy())
            value = np.asarray(raw, np.float32).reshape(1, 5, 1)
            self.outputs.append(value.copy())
            return [value], {"infer": 2}

    backend = Backend()
    detector = Detector(ModelSpec(model_id="ppe", onnx_path="never-opened", input_size=(120, 120), classes=["helmet"]), backend)
    frame = np.arange(200 * 200 * 3, dtype=np.uint8).reshape(200, 200, 3)
    people = [person(bbox=box)]
    result = infer_ppe(frame, people, detector, 1, PpePolicy("person_crop"))
    assert result.complete and result.detections[0].bbox == pytest.approx(expected)
    call = result.rois[0]["call"]
    assert call["input"]["sha256"] == hashlib.sha256(backend.inputs[0].tobytes()).hexdigest()
    assert call["outputs"][0]["sha256"] == hashlib.sha256(backend.outputs[0].tobytes()).hexdigest()
    assert call["input"]["shape"] == [1, 3, 120, 120] and call["request_batch"] == 1
    assert tuple(np.array(frame)[slice(result.rois[0]["actual_rect"][1], result.rois[0]["actual_rect"][3]),
                                  slice(result.rois[0]["actual_rect"][0], result.rois[0]["actual_rect"][2])].shape) == shape
    assert len(backend.inputs) == 1 and detector.backend is backend
    json.dumps(result.audit_record(), allow_nan=False)


def test_overlapping_people_one_helmet_is_ambiguous_and_conservative_mask_wins():
    people = [person(1, (0, 0, 100, 200)), person(2, (50, 0, 150, 200))]
    # ROI1 x0=0; ROI2 x0=40. Same global helmet appears in both crops.
    result = infer_ppe(np.zeros((240, 180, 3), np.uint8), people,
        ScriptedPpe([[ppe(bbox=(60, 10, 80, 30))], [ppe(bbox=(20, 10, 40, 30))]]), 1, PpePolicy("person_crop"))
    apply(result, people, 1)
    assert result.detections == [] and result.counts["succeeded"] == 2
    assert all(track.attributes["helmet"] is None for track in people)
    assert all(row["observations"][0]["decision"] == "ambiguous" for row in result.rois)
    assert all(set(row["observations"][0]["candidate_track_ids"]) == {1, 2} for row in result.rois)


def test_same_helmet_in_two_rois_is_not_reassigned_or_double_voted():
    people = [person(1, (0, 0, 100, 200)), person(2, (20, 60, 120, 260))]
    # Global helmet y50 fits only owner1 but is visible inside both padded ROIs.
    result = infer_ppe(np.zeros((300, 200, 3), np.uint8), people,
        ScriptedPpe([[ppe(bbox=(40, 42, 60, 58))], [ppe(bbox=(30, 2, 50, 18))]]), 1, PpePolicy("person_crop"))
    apply(result, people, 1)
    assert len(result.by_track[1]) == 1 and not result.by_track[2]
    assert people[0].attributes["helmet"] is True and people[1].attributes["helmet"] is None
    history = people[0].attributes["_tracking_attribute_state"]["ppe"]["helmet"]
    assert history == [(1, True)]
    apply(result, people, 1)
    assert history == people[0].attributes["_tracking_attribute_state"]["ppe"]["helmet"]


def test_ambiguous_from_other_roi_masks_independent_accepted_observation():
    people = [person(1, (0, 0, 100, 200)), person(2, (50, 0, 150, 200))]
    result = infer_ppe(np.zeros((240, 180, 3), np.uint8), people,
        ScriptedPpe([[ppe(bbox=(10, 10, 30, 30))], [ppe(bbox=(20, 10, 40, 30))]]), 1, PpePolicy("person_crop"))
    apply(result, people, 1)
    assert len(result.by_track[1]) == 1 and people[0].attributes["helmet"] is None
    assert people[0].attributes["_tracking_attribute_state"]["ppe"]["helmet"] == [(1, True)]


@pytest.mark.parametrize("failure", ["error", "stale", "invalid", "ambiguous"])
def test_old_false_cannot_leak_after_unusable_roi_but_history_survives(failure):
    track = person(1, (0, 0, 100, 200), 0)
    for ts in (0, .2, .4):
        assign_attributes([track], {"ppe": [ppe("no_helmet", (40, 10, 60, 30))]}, ts)
    assert track.attributes["helmet"] is False
    prior = copy.deepcopy(track.attributes["_tracking_attribute_state"]["ppe"]["helmet"])
    track.last_seen = .6 if failure != "stale" else .4
    people = [track]
    outputs = [RuntimeError("device failed")] if failure == "error" else [[]]
    if failure == "invalid":
        track.bbox = (400, 0, 500, 200)
    if failure == "ambiguous":
        people.append(person(2, (0, 0, 100, 200), .6))
        outputs = [[ppe("no_helmet", (40, 10, 60, 30))], []]
    detector = ScriptedPpe(outputs)
    result = infer_ppe(np.zeros((240, 200, 3), np.uint8), people, detector, .6, PpePolicy("person_crop"))
    apply(result, people, .6)
    assert track.attributes["helmet"] is None
    assert track.attributes["_tracking_attribute_state"]["ppe"]["helmet"][:3] == prior
    track.bbox, track.last_seen = (0, 0, 100, 200), .8
    recovered = infer_ppe(np.zeros((240, 200, 3), np.uint8), [track],
                          ScriptedPpe([[ppe("no_helmet", (40, 10, 60, 30))]]), .8, PpePolicy("person_crop"))
    apply(recovered, [track], .8)
    assert track.attributes["helmet"] is False


def test_failed_roi_does_not_stop_other_owner_and_does_not_invent_absence():
    people = [person(1, (0, 0, 60, 100)), person(2, (100, 0, 160, 100))]
    result = infer_ppe(np.zeros((140, 200, 3), np.uint8), people, ScriptedPpe([RuntimeError("unavailable"), []]), 1, PpePolicy("person_crop"))
    apply(result, people, 1)
    assert not result.complete and result.counts == dict(eligible=2, attempted=2, succeeded=1, errors=1, calls=2)
    assert result.by_track == {2: []}
    assert all(track.attributes["helmet"] is None for track in people)
    assert people[1].attributes["_tracking_attribute_state"]["ppe"]["helmet"] == [(1, None)]


def test_no_people_zero_crop_calls_but_full_frame_still_runs_once_without_batch_override():
    detector = ScriptedPpe([[]])
    frame = np.zeros((10, 10, 3), np.uint8)
    assert infer_ppe(frame, [], detector, 1, PpePolicy("person_crop")).counts["calls"] == 0
    result = infer_ppe(frame, [], detector, 1)
    assert result.complete and result.counts["calls"] == 1
    assert detector.calls == [((10, 10, 3), {})]


def test_global_pose_phone_are_identical_and_owner_votes_once_per_timestamp():
    points = np.zeros((17, 3), dtype=float)
    points[[5, 6, 11, 12]] = [(35, 30, .9), (65, 30, .9), (35, 65, .9), (65, 65, .9)]
    people = [person(1, (0, 0, 100, 200))]
    baseline = copy.deepcopy(people)
    others = {"pose": [Detection("pose", "person", 0, .9, people[0].bbox, points)],
              "det": [Detection("det", "cell phone", 1, .9, (45, 60, 55, 70))]}
    result = infer_ppe(np.zeros((240, 140, 3), np.uint8), people,
                       ScriptedPpe([[ppe(bbox=(40, 10, 60, 30))]]), 1, PpePolicy("person_crop"))
    apply(result, people, 1, others)
    assign_attributes(baseline, {**others, "ppe": result.detections}, 1)
    assert people[0].attributes == baseline[0].attributes
    np.testing.assert_equal(people[0].keypoints, baseline[0].keypoints)
    assert attribute_coverage(people, 1)["attributes"]["helmet"] == dict(true=1, false=0, unknown=0)


def test_nonfinite_output_fails_closed_without_nonfinite_json():
    result = infer_ppe(np.zeros((200, 200, 3), np.uint8), [person()],
                       ScriptedPpe([[ppe(bbox=(1, 2, float("nan"), 3))]]), 1, PpePolicy("person_crop"))
    assert not result.complete and not result.by_track
    json.dumps(result.audit_record(), allow_nan=False)


def test_stale_neighbor_still_participates_in_anatomical_ambiguity():
    people = [person(1, (0, 0, 100, 200)), person(2, (0, 0, 100, 200), ts=.5)]
    detector = ScriptedPpe([[ppe(bbox=(40, 10, 60, 30))]])
    result = infer_ppe(np.zeros((240, 140, 3), np.uint8), people, detector, 1, PpePolicy("person_crop"))
    apply(result, people, 1)
    assert len(detector.calls) == 1 and result.rois[1]["status"] == "stale"
    assert all(track.attributes["helmet"] is None for track in people)


@pytest.mark.parametrize("mode", ["full_frame", "person_crop"])
def test_worker_and_s2_share_order_ownership_and_one_attribute_update(tmp_path, monkeypatch, mode):
    camera = CameraConfig.model_validate({"camera_id": "ppe_roi", "name": "synthetic", "site": "unit",
        "source": {"type": "file", "uri": "never-opened", "loop": False},
        "rules": [{"rule_id": "s2", "type": "ppe_missing", "min_duration_s": 100,
                   "params": {"required": ["helmet"], "min_box_h_px": 60}}]})
    from guard.rules.base import load_plugins
    load_plugins()
    settings = make_settings(tmp_path, ppe_input_mode=mode, tracker_min_hits=1)
    outputs = [[ppe("no_helmet", (40, 10, 60, 30))] for _ in range(4)]
    worker_ppe, eval_ppe = ScriptedPpe(copy.deepcopy(outputs)), ScriptedPpe(copy.deepcopy(outputs))
    coco = [Detection("det", "person", 0, .95, (0, 0, 100, 200))]
    worker_det, eval_det = ScriptedPpe([coco] * 4), ScriptedPpe([coco] * 4)
    detectors = {"det": worker_det, "ppe": worker_ppe}
    engine = SimpleNamespace(settings=settings, metrics=MetricsRegistry(), get_detector=detectors.__getitem__,
        events=compat.EventManager(), run_id="unit", on_analysis=lambda *a: None,
        handle_events=lambda *a, **kw: pytest.fail("No event expected before frozen rule duration"))
    worker = CameraWorker(camera, engine)
    worker.detectors = detectors  # Explicit production-order fixture.
    tracker = compat.Tracker(classes={"person"}, **{**worker._tracker_kwargs})
    manager = compat.EventManager()
    original_assign = compat.assign_attributes
    assignments = []

    def assign(*args, **kwargs):
        assignments.append(kwargs)
        return original_assign(*args, **kwargs)

    monkeypatch.setattr(compat, "assign_attributes", assign)
    frame = np.zeros((240, 140, 3), np.uint8)
    for index in range(4):
        ts = index / 5
        context = worker.process(FramePacket(frame, ts, index + 1, 0, capture_id="one-session"))
        evaluated = run_s2.process_frame(camera, frame, {"source_time_s": ts, "source_frame_index": index},
            {"det": eval_det, "ppe": eval_ppe}, tracker, manager, worker.rules, ppe_policy=PpePolicy(mode))
        assert [track.attributes for track in context.people] == [track["attributes"] for track in evaluated["trace"]["person_tracks"]]
        assert worker.last_ppe.counts == evaluated["trace"]["ppe"]["counts"]
    assert len(assignments) == 8 and not worker.errors
    assert len(worker_ppe.calls) == len(eval_ppe.calls) == 4
    assert all(bool(kwargs) == (mode == "person_crop") for kwargs in assignments)
    assert all(call[1].get("force_batch_one", False) == (mode == "person_crop") for call in worker_ppe.calls)
    assert context.people[0].attributes["helmet"] is False
    if mode == "person_crop":
        worker_ppe.outputs = iter([RuntimeError("device unavailable")])
        worker_det.outputs = iter([coco])
        failed_context = worker.process(FramePacket(frame, .8, 5, 0, capture_id="one-session"))
        assert failed_context.people[0].attributes["helmet"] is None
        assert "infer:ppe" in worker.errors and worker.metrics.errors == 1
        assert not worker.last_ppe.complete


def test_s2_crop_failure_is_incomplete_instead_of_negative_prediction():
    class Tracker:
        def update(self, detections, ts):
            return [person(ts=ts)]

    class NeverManager:
        def process_with_updates(self, *a):
            pytest.fail("Failed ROI must not reach event matching")

    with pytest.raises(PpeFrameError) as error:
        run_s2.process_frame(None, np.zeros((200, 200, 3), np.uint8), {"source_time_s": 1, "source_frame_index": 0},
            {"det": ScriptedPpe([[]]), "ppe": ScriptedPpe([RuntimeError("device failure")])},
            Tracker(), NeverManager(), [], ppe_policy=PpePolicy("person_crop"))
    assert error.value.audit["counts"]["errors"] == 1 and not error.value.audit["complete"]


def test_s2_mode_and_coverage_keep_unknown_denominators_explicit():
    assert run_s2.parser().parse_args(["--candidate", "vyra"]).ppe_input_mode == "full_frame"
    assert run_s2.parser().parse_args(["--candidate", "vyra", "--ppe-input-mode", "person_crop"]).ppe_input_mode == "person_crop"
    target = {}
    run_s2.add_counts(target, attribute_coverage([person()], 1))
    run_s2.add_counts(target, attribute_coverage([person(), person(2, ts=0)], 1))
    assert target["observed_person_samples"] == 2 and target["stale_person_samples"] == 1
    assert target["attributes"]["helmet"] == dict(true=0, false=0, unknown=2)


class QueuedBackend:
    """One controlled first request lets tests enqueue a deterministic FIFO."""
    def __init__(self, fail=()):
        self.started, self.release = threading.Event(), threading.Event()
        self.calls = []
        self.fail = set(fail)
        self.active = 0
        self.max_active = 0

    def infer_timed(self, model, batch):
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        ids = batch[:, 0, 0, 0].tolist()
        self.calls.append(ids)
        try:
            if ids == [0]:
                self.started.set()
                assert self.release.wait(3)
            if any(value in self.fail for value in ids):
                raise RuntimeError("planned request failure")
            return [batch.copy()], {"infer": 1}
        finally:
            self.active -= 1


def tensor(index):
    return np.full((1, 1, 1, 1), index, np.float32)


def test_existing_scheduler_interleaves_normal_and_roi_fifo_without_concurrent_backend_calls():
    backend = QueuedBackend()
    scheduler = BatchScheduler(backend, "ppe", 4, .05)
    try:
        first = scheduler.submit(tensor(0), force_batch_one=True)
        assert backend.started.wait(2)
        values = [(1, False), (2, False), (3, True), (4, True), (5, False), (6, False)]
        futures = [scheduler.submit(tensor(index), force_batch_one=single) for index, single in values]
        backend.release.set()
        first.result(2)
        for (index, single), future in zip(values, futures):
            outputs, times = future.result(2)
            assert outputs[0].item() == index
            assert times["batch"] == (1 if single else 2)
    finally:
        backend.release.set()
        scheduler.stop()
    assert backend.calls == [[0], [1, 2], [3], [4], [5, 6]] and backend.max_active == 1


def test_scheduler_roi_failure_and_cancellation_preserve_following_normal_batch():
    backend = QueuedBackend(fail={2})
    scheduler = BatchScheduler(backend, "ppe", 4, .05)
    try:
        first = scheduler.submit(tensor(0), force_batch_one=True)
        assert backend.started.wait(2)
        cancelled = scheduler.submit(tensor(1), force_batch_one=True)
        assert cancelled.cancel()
        failed = scheduler.submit(tensor(2), force_batch_one=True)
        normal = [scheduler.submit(tensor(i)) for i in (3, 4)]
        backend.release.set()
        first.result(2)
        with pytest.raises(RuntimeError, match="planned"):
            failed.result(2)
        assert [future.result(2)[0][0].item() for future in normal] == [3, 4]
    finally:
        backend.release.set()
        scheduler.stop()
    assert backend.calls == [[0], [2], [3, 4]] and cancelled.cancelled()


def test_shutdown_rejects_pending_roi_and_cancelled_future_without_invalidstate():
    backend = QueuedBackend()
    scheduler = BatchScheduler(backend, "ppe", 4, .05)
    first = scheduler.submit(tensor(0), force_batch_one=True)
    assert backend.started.wait(2)
    pending = scheduler.submit(tensor(1), force_batch_one=True)
    cancelled = scheduler.submit(tensor(2))
    assert cancelled.cancel()
    stopped = threading.Thread(target=scheduler.stop)
    stopped.start()
    assert scheduler._stop.wait(1)
    backend.release.set()
    stopped.join(3)
    first.result(2)
    with pytest.raises(RuntimeError, match="stopped"):
        pending.result(2)
    assert cancelled.cancelled() and not scheduler._thread.is_alive() and not stopped.is_alive()
    assert backend.calls == [[0]]


@pytest.mark.parametrize("action", ["cancel", "stop"])
def test_deferred_roi_future_is_cancelled_or_rejected_without_losing_fifo(action):
    class Backend(QueuedBackend):
        def __init__(self):
            super().__init__()
            self.second_started, self.second_release = threading.Event(), threading.Event()

        def infer_timed(self, model, batch):
            if batch[:, 0, 0, 0].tolist() == [1]:
                self.second_started.set()
                assert self.second_release.wait(3)
            return super().infer_timed(model, batch)

    backend = Backend()
    scheduler = BatchScheduler(backend, "ppe", 4, .05)
    try:
        first = scheduler.submit(tensor(0), force_batch_one=True)
        assert backend.started.wait(2)
        normal = scheduler.submit(tensor(1))
        deferred = scheduler.submit(tensor(2), force_batch_one=True)
        following = scheduler.submit(tensor(3))
        backend.release.set()
        first.result(2)
        assert backend.second_started.wait(2)  # _collect has moved ROI2 into _deferred.
        if action == "cancel":
            assert deferred.cancel()
            backend.second_release.set()
            assert following.result(2)[0][0].item() == 3
        else:
            stopping = threading.Thread(target=scheduler.stop)
            stopping.start()
            assert scheduler._stop.wait(1)
            backend.second_release.set()
            stopping.join(3)
            for future in (deferred, following):
                with pytest.raises(RuntimeError, match="stopped"):
                    future.result(2)
        normal.result(2)
    finally:
        backend.release.set()
        backend.second_release.set()
        scheduler.stop()
    assert backend.calls == ([[0], [1], [3]] if action == "cancel" else [[0], [1]])
    assert not scheduler._thread.is_alive()


def test_multiple_roi_audits_keep_each_actual_input_and_output_instead_of_last_call():
    class Backend:
        def __init__(self):
            self.inputs, self.outputs = [], []

        def infer_timed(self, model_id, batch):
            self.inputs.append(batch.copy())
            raw = np.array([60, 17, 10, 10, .8 + .01 * len(self.inputs)], np.float32).reshape(1, 5, 1)
            self.outputs.append(raw.copy())
            return [raw], {"infer": 1}

    backend = Backend()
    detector = Detector(ModelSpec(model_id="ppe", onnx_path="never-opened", input_size=(120, 120), classes=["helmet"]), backend)
    frame = np.zeros((140, 340, 3), np.uint8)
    frame[:, 180:] = 255
    people = [person(1, (0, 0, 100, 100)), person(2, (200, 0, 300, 100))]
    result = infer_ppe(frame, people, detector, 1, PpePolicy("person_crop"))
    assert result.complete and len(result.rois) == len(backend.inputs) == 2
    for index, row in enumerate(result.rois):
        assert row["call"]["input"]["sha256"] == hashlib.sha256(backend.inputs[index].tobytes()).hexdigest()
        assert row["call"]["outputs"][0]["sha256"] == hashlib.sha256(backend.outputs[index].tobytes()).hexdigest()
    assert result.rois[0]["call"]["input"]["sha256"] != result.rois[1]["call"]["input"]["sha256"]
    assert result.rois[0]["call"]["outputs"] != result.rois[1]["call"]["outputs"]


def test_cancel_races_with_deferred_shutdown_claim_and_following_future_still_rejected():
    class RacingFuture(Future):
        def __init__(self):
            super().__init__()
            self.claim_entered, self.claim_release = threading.Event(), threading.Event()

        def set_running_or_notify_cancel(self):
            self.claim_entered.set()
            assert self.claim_release.wait(3)
            return super().set_running_or_notify_cancel()

    class Backend(QueuedBackend):
        def __init__(self):
            super().__init__()
            self.second_started, self.second_release = threading.Event(), threading.Event()

        def infer_timed(self, model, batch):
            if batch[:, 0, 0, 0].tolist() == [1]:
                self.second_started.set()
                assert self.second_release.wait(3)
            return super().infer_timed(model, batch)

    backend, racing = Backend(), RacingFuture()
    scheduler = BatchScheduler(backend, "ppe", 4, .05)
    try:
        first = scheduler.submit(tensor(0), force_batch_one=True)
        assert backend.started.wait(2)
        normal = scheduler.submit(tensor(1))
        scheduler.submit(tensor(2), force_batch_one=True)
        following = scheduler.submit(tensor(3))
        backend.release.set()
        first.result(2)
        assert backend.second_started.wait(2)
        assert scheduler._deferred is not None
        scheduler._deferred.future = racing
        stopping = threading.Thread(target=scheduler.stop)
        stopping.start()
        assert scheduler._stop.wait(1)
        backend.second_release.set()
        assert racing.claim_entered.wait(2)
        assert racing.cancel()  # Wins immediately before the atomic claim.
        racing.claim_release.set()
        stopping.join(3)
        normal.result(2)
        with pytest.raises(RuntimeError, match="stopped"):
            following.result(2)
        assert racing.cancelled() and not scheduler._thread.is_alive()
    finally:
        racing.claim_release.set()
        backend.release.set()
        backend.second_release.set()
        scheduler.stop()
