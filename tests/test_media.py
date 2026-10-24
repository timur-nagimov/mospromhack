"""Streaming completion must drain bounded ring snapshots before publishing."""

import threading
import time

import numpy as np

from guard.contracts import Event
from guard.events import media as media_module


def test_post_frames_arriving_during_encoding_are_drained_before_completion(tmp_path, monkeypatch):
    entered, resume = threading.Event(), threading.Event()
    written, done = [], []

    class Encoder:
        def __init__(self, path, fps):
            self.frames = 0

        def write(self, frame):
            if self.frames == 0:
                entered.set()
                assert resume.wait(3), "Test did not publish the post frames"
            written.append(int(frame[0, 0, 0]))
            self.frames += 1

        def close(self):
            return True

        def abort(self):
            raise AssertionError("The complete streaming clip must not abort")

    monkeypatch.setattr(media_module, "ClipEncoder", Encoder)
    media = media_module.MediaStore(tmp_path, pre_s=1, post_s=2, clip_fps=2, max_frames=20,
                                   max_writers=1, max_pending=1)
    base = time.time() + 3600
    for index in range(4):
        media.push_frame("cam", np.full((8, 8, 3), index, np.uint8), base + index)
    event = Event(id="clip", camera_id="cam", site="unit", rule_id="rule", type="zone_intrusion", severity="high",
                  started_at=base + 1, confirmed_at=base + 3, bbox=(0, 0, 8, 8))
    job = media_module._ClipJob(event, media.ring("cam"), base, base + 5, lambda ev, path: done.append(path))
    writer = threading.Thread(target=media._write_streaming, args=(job,))
    writer.start()
    try:
        assert entered.wait(3), "Encoder did not receive the initial snapshot"
        for index in (4, 5, 6):
            media.push_frame("cam", np.full((8, 8, 3), index, np.uint8), base + index)
        resume.set()
        writer.join(timeout=3)
        assert not writer.is_alive()
        assert written == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5]
        assert media.clips_written == 1 and media.clip_errors == 0 and done == [media.rel_path(event, ".mp4")]
        assert len(media.ring("cam")) <= media.max_frames
        assert media.max_writers == 1 and media._jobs.maxsize == 1
    finally:
        resume.set()
        writer.join(timeout=3)



def test_source_capture_rotation_seals_active_and_queued_clips_without_scene_mixing(tmp_path, monkeypatch):
    entered, resume, completed = threading.Event(), threading.Event(), threading.Event()
    written, done = {}, []

    class Encoder:
        def __init__(self, path, fps):
            self.frames = 0
            self.name = path.stem
            written[self.name] = []

        def write(self, frame):
            if self.name == "old-a" and self.frames == 0:
                entered.set()
                assert resume.wait(3)
            written[self.name].append(int(frame[0, 0, 0]))
            self.frames += 1

        def close(self):
            return True

        def abort(self):
            raise AssertionError("A sealed clip has two observed frames")

    monkeypatch.setattr(media_module, "ClipEncoder", Encoder)
    media = media_module.MediaStore(tmp_path, pre_s=1, post_s=5, clip_fps=2, max_frames=4, max_writers=1, max_pending=2)
    base = time.time() + 3600

    def push(capture, start, value):
        for index in range(2):
            media.push_frame("cam", np.full((8, 8, 3), value + index, np.uint8), base + start + index / 2, capture_id=capture)

    def event(name, capture, start):
        return Event(id=name, camera_id="cam", site="unit", rule_id="zone", type="zone_intrusion", severity="high",
            started_at=base + start, confirmed_at=base + start + .5, bbox=(0, 0, 8, 8),
            details={"origin": {"capture_id": capture}})

    def on_done(ev, path):
        media.stats()  # A callback may acquire media/SQLite locks; no store lock may be held here.
        done.append((ev.id, path))
        if len(done) == 2:
            completed.set()

    push("a", 0, 10)
    old_ring = media.ring("cam")
    push("b", 1, 100)
    assert old_ring.sealed
    assert media.schedule_clip(event("old-a", "a", 0), on_done=on_done)
    try:
        assert entered.wait(2)
        push("c", 2, 200)
        assert media.schedule_clip(event("old-b", "b", 1), on_done=on_done)
        push("d", 3, 240)
        assert not media.schedule_clip(event("late-a", "a", 0), on_done=on_done)
        assert media.clips_dropped == 1
        rings = list(media._all_rings)
        assert len(rings) == 4
        assert all(ring.budget_bytes == media.budget_bytes // 4 for ring in rings)
        assert all(len(ring) <= media.max_frames for ring in rings)
        assert len(media._threads) == media.max_writers == 1 and media._jobs.maxsize == 2
        resume.set()
        assert completed.wait(2), "Sealed rings must finish without a synthetic post-roll/wall timeout"
        assert written == {"old-a": [10, 11], "old-b": [100, 101]}
        assert media.clips_written == 2 and media.clip_errors == 0
        assert all(path is not None for _, path in done)
    finally:
        resume.set()
        media.stop()


def test_capture_lookup_retention_is_bounded_across_many_source_loops(tmp_path):
    media = media_module.MediaStore(tmp_path, clip_fps=2, max_frames=4, budget_mb=1)
    for generation in range(100):
        for index in range(10):
            media.push_frame("cam", np.zeros((8, 8, 3), np.uint8), generation * 10 + index / 2, capture_id=str(generation))
    assert len(media._previous) == 1 and media._previous["cam"][0] == "98"
    assert media._capture_ids["cam"] == "99"
    assert media.stats()["retained_rings"] == 2
    assert all(len(ring) <= 4 for ring in media._all_rings)
    assert sum(ring.budget_bytes for ring in media._all_rings) <= media.budget_bytes


def test_empty_capture_and_first_known_identity_keep_legacy_ring_but_known_change_seals_it(tmp_path):
    media = media_module.MediaStore(tmp_path, clip_fps=2)
    frame = np.zeros((8, 8, 3), np.uint8)
    media.push_frame("cam", frame, 100)
    original = media.ring("cam")
    media.push_frame("cam", frame, 100.5, capture_id="first")
    media.push_frame("cam", frame, 101, capture_id="")
    assert media.ring("cam") is original and not original.sealed
    media.push_frame("cam", frame, 101.5, capture_id="second")
    assert media.ring("cam") is not original and original.sealed
    assert original.push(frame, 102) is False
    media.remove_camera("cam")
    assert not media._rings and not media._capture_ids and not media._previous
