from __future__ import annotations

import copy
import json
import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import yaml

from guard.contracts import Detection
from guard.pipeline import settings as pipeline_settings
from guard.tracking import Tracker as RealTracker
from ml.eval import events_eval
from ml.eval import run_offline as offline


@pytest.fixture(autouse=True)
def isolate_guard_environment(monkeypatch):
    """Не читаем .env основного репозитория и не наследуем его GUARD_*."""
    for key in tuple(os.environ):
        if key.startswith("GUARD_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(
        pipeline_settings,
        "load_dotenv_guard",
        lambda *args, **kwargs: None,
    )


def _write_manifest(bundle):
    bundle.path.write_text(
        yaml.safe_dump(bundle.raw, allow_unicode=True),
        encoding="utf-8",
    )


@pytest.fixture
def bundle(tmp_path):
    """Три секунды настоящего видео; никаких моделей или скачиваний."""
    video = tmp_path / "sample.avi"
    writer = cv2.VideoWriter(
        str(video),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (128, 96),
    )
    try:
        assert writer.isOpened(), "OpenCV could not create an MJPG test video"
        for index in range(30):
            frame = np.full((96, 128, 3), 40 + index, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()

    labels = tmp_path / "labels"
    labels.mkdir()
    (labels / "sample.json").write_text(
        json.dumps(
            {
                "video": "sample.avi",
                "events": [
                    {
                        "type": "zone_intrusion",
                        "t_start": 0.0,
                        "t_end": 3.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    models = tmp_path / "models.yaml"
    models.write_text(
        yaml.safe_dump(
            {
                "models": [
                    {
                        "model_id": "det",
                        "onnx_path": "not-needed.onnx",
                        "classes": [],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    raw = {
        "version": 1,
        "root": ".",
        "videos_dir": ".",
        "labels_dir": "labels",
        "models_config": "models.yaml",
        "models_dir": "models",
        "sample_fps": 5.0,
        "t0": 1_700_000_000.0,
        "scenarios": [
            {
                "id": "s1_01",
                "video": "sample.avi",
                "camera": {
                    "camera_id": "offline_test",
                    "name": "Synthetic camera",
                    "site": "Tests",
                    "zones": [
                        {
                            "zone_id": "danger",
                            "name": "Danger zone",
                            "kind": "danger",
                            "polygon": [
                                [0.1, 0.1],
                                [0.9, 0.1],
                                [0.9, 0.95],
                                [0.1, 0.95],
                            ],
                        }
                    ],
                    "rules": [
                        {
                            "rule_id": "intrusion",
                            "type": "zone_intrusion",
                            "zones": ["danger"],
                            "params": {"min_conf": 0.35, "min_box_h_px": 40},
                            "min_duration_s": 1.0,
                            "min_ratio": 1.0,
                            "close_after_s": 0.4,
                            "cooldown_s": 30.0,
                        }
                    ],
                },
            }
        ],
    }
    result = SimpleNamespace(
        path=tmp_path / "scenarios.yaml",
        raw=raw,
        video=video,
        labels=labels,
        models=models,
    )
    _write_manifest(result)
    return result


@pytest.fixture
def fake_models(monkeypatch):
    """Подменяется инференс; алгоритмы трекинга, правил и событий настоящие."""
    state = SimpleNamespace(
        detectors=[],
        backend_calls=[],
        timestamps=[],
        tracker_calls=[],
        event_manager_calls=0,
        settings_calls=0,
    )

    class Backend:
        def __init__(self):
            self.loaded = []
            self.closed = False

        def load(self, spec):
            self.loaded.append(spec.model_id)

        def class_names(self, model_id):
            assert model_id == "det"
            return ["person"]

        def close(self):
            self.closed = True

    state.backend = Backend()

    class FixedDetector:
        def __init__(self, spec, backend):
            assert backend is state.backend
            assert spec.classes == ["person"]
            self.model_id = spec.model_id
            self.calls = 0
            state.detectors.append(self)

        def detect(self, frame, timings=None):
            assert frame.shape == (96, 128, 3)
            self.calls += 1
            return [
                Detection(
                    model=self.model_id,
                    cls="person",
                    cls_id=0,
                    conf=0.99,
                    bbox=(40.0, 10.0, 80.0, 80.0),
                )
            ]

    def backend_factory(name, **kwargs):
        state.backend_calls.append((name, kwargs))
        return state.backend

    # Тесты времён явно используют min_hits=1 через Settings. Фабрика
    # трекера не исправляет переданные runner параметры и сохраняет их
    # для проверки соответствия боевому конвейеру.
    state.settings = replace(
        offline.Settings.from_env(), tracker_min_hits=1
    )

    def settings_factory(cls):
        state.settings_calls += 1
        return replace(state.settings)

    def tracker_factory(**kwargs):
        state.tracker_calls.append(dict(kwargs))
        return RealTracker(**kwargs)

    original_assign = offline.compat.assign_attributes
    original_manager = offline.compat.EventManager

    def record_attributes(people, detections, ts):
        state.timestamps.append(ts)
        return original_assign(people, detections, ts)

    def manager_factory():
        state.event_manager_calls += 1
        return original_manager()

    monkeypatch.setattr(offline, "get_backend", backend_factory)
    monkeypatch.setattr(offline, "Detector", FixedDetector)
    monkeypatch.setattr(
        offline.Settings, "from_env", classmethod(settings_factory)
    )
    monkeypatch.setattr(offline.compat, "Tracker", tracker_factory)
    monkeypatch.setattr(offline.compat, "assign_attributes", record_attributes)
    monkeypatch.setattr(offline.compat, "EventManager", manager_factory)
    return state


def test_cli_events_timestamps_and_evaluation(
    bundle, fake_models, monkeypatch, tmp_path, capsys
):
    evaluations = []
    original_evaluate = events_eval.evaluate

    def evaluate_spy(labels, preds, tol_s=2.0, *, durations=None):
        evaluations.append((labels, preds, dict(durations)))
        return original_evaluate(labels, preds, tol_s, durations=durations)

    monkeypatch.setattr(events_eval, "evaluate", evaluate_spy)
    output = tmp_path / "metrics" / "eval-onnx-cpu-test.json"
    assert offline.main(
        [
            "--scenarios", str(bundle.path),
            "--backend", "onnx-cpu",
            "--out", str(output),
            "--tol-s", "0",
        ]
    ) == 0

    result = json.loads(output.read_text(encoding="utf-8"))
    assert len(evaluations) == 1
    assert result["total"] == len(result["items"]) == 1
    assert result["overall"]["tp"] == 1
    assert result["overall"]["fp"] == 0
    assert result["overall"]["fn"] == 0
    assert result["overall"]["duplicates"] == 0
    assert result["overall"]["precision"] == 1.0
    assert result["overall"]["recall"] == 1.0
    assert result["overall"]["latency_median_s"] == pytest.approx(1.0)
    assert result["overall"]["false_alarms_per_hour"] == 0.0
    assert result["durations"] == {"sample.avi": pytest.approx(3.0)}

    mapping = result["mapping"]["offline_test"]
    assert mapping == {"video": "sample.avi", "t0": bundle.raw["t0"]}
    event = result["items"][0]
    assert event["started_at"] == pytest.approx(mapping["t0"], rel=0, abs=1e-6)
    assert event["confirmed_at"] == pytest.approx(
        mapping["t0"] + 1.0, rel=0, abs=1e-6
    )
    assert event["ended_at"] is None

    # Сохранённый отчёт — допустимый вход штатного загрузчика предсказаний.
    predictions = events_eval.load_predictions(output, result["mapping"])
    assert len(predictions) == 1
    assert predictions[0].start == pytest.approx(0.0, abs=1e-6)
    assert predictions[0].confirmed == pytest.approx(1.0, abs=1e-6)
    assert evaluations[0][2] == {"sample.avi": pytest.approx(3.0)}

    assert fake_models.timestamps == pytest.approx(
        [mapping["t0"] + index / 5 for index in range(15)],
        rel=0,
        abs=1e-6,
    )
    assert fake_models.detectors[0].calls == 15
    assert result["processing"]["frames_decoded"] == 30
    assert result["processing"]["frames_analyzed"] == 15
    assert result["processing"]["elapsed_s"] > 0
    assert result["processing"]["fps"] > 0
    assert result["processing"]["wall_elapsed_s"] >= result["processing"]["elapsed_s"]
    assert fake_models.backend.loaded == ["det"]
    assert fake_models.backend.closed
    assert fake_models.backend_calls[0][1]["intra_op_threads"] == 2
    assert fake_models.settings_calls == 1
    assert fake_models.event_manager_calls == 1
    assert os.environ["OMP_NUM_THREADS"] == "2"
    text = capsys.readouterr().out
    assert "TOTAL" in text
    assert "Precision" in text
    assert "analyzed FPS" in text


@pytest.mark.parametrize(
    ("sample_fps", "indices"),
    [
        (4.0, [0, 3, 5, 8, 10, 13, 15, 18, 20, 23, 25, 28]),
        (20.0, list(range(30))),
    ],
)
def test_sampling_uses_source_indices_without_duplicate_frames(
    bundle, fake_models, sample_fps, indices
):
    bundle.raw["sample_fps"] = sample_fps
    _write_manifest(bundle)
    result = offline.run_evaluation(offline.load_scenarios(bundle.path))
    assert fake_models.timestamps == pytest.approx(
        [bundle.raw["t0"] + index / 10 for index in indices],
        rel=0,
        abs=1e-6,
    )
    assert fake_models.detectors[0].calls == len(indices)
    assert result["processing"]["frames_decoded"] == 30
    assert result["processing"]["frames_analyzed"] == len(indices)
    assert result["durations"]["sample.avi"] == pytest.approx(3.0)


def test_only_excludes_other_labels_and_does_not_open_other_video(
    bundle, fake_models, tmp_path
):
    other = copy.deepcopy(bundle.raw["scenarios"][0])
    other["id"] = "s3_02"
    other["video"] = "missing.avi"
    other["camera"]["camera_id"] = "offline_other"
    bundle.raw["scenarios"].append(other)
    (bundle.labels / "other.json").write_text(
        json.dumps(
            {
                "video": "missing.avi",
                "events": [
                    {"type": "zone_intrusion", "t_start": 0.0, "t_end": 2.0}
                ],
            }
        ),
        encoding="utf-8",
    )
    _write_manifest(bundle)
    output = tmp_path / "only.json"
    assert offline.main(
        [
            "--scenarios", str(bundle.path),
            "--only", "s1_01",
            "--out", str(output),
        ]
    ) == 0
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["videos"] == ["sample.avi"]
    assert set(result["durations"]) == {"sample.avi"}
    assert set(result["mapping"]) == {"offline_test"}
    assert result["overall"]["tp"] == 1
    assert result["overall"]["fn"] == 0
    assert len(result["runs"]) == 1


def test_unknown_only_is_an_error_before_model_loading(bundle, fake_models):
    with pytest.raises(ValueError, match="Unknown scenario IDs"):
        offline.run_evaluation(
            offline.load_scenarios(bundle.path), only=["typo"]
        )
    assert fake_models.backend_calls == []


def test_missing_model_is_not_silently_disabled(bundle, fake_models):
    bundle.models.write_text("models: []\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Missing required models: det"):
        offline.run_evaluation(offline.load_scenarios(bundle.path))
    assert fake_models.backend_calls == []


def test_detector_failure_closes_backend_and_does_not_write_report(
    bundle, fake_models, monkeypatch, tmp_path
):
    def fail(self, frame, timings=None):
        raise RuntimeError("detector failed")

    monkeypatch.setattr(offline.Detector, "detect", fail)
    output = tmp_path / "failed.json"
    with pytest.raises(SystemExit) as exc:
        offline.main(
            ["--scenarios", str(bundle.path), "--out", str(output)]
        )
    assert exc.value.code == 2
    assert fake_models.backend.closed
    assert not output.exists()


def test_disabled_rule_keeps_ground_truth_as_false_negative(bundle, fake_models):
    bundle.raw["scenarios"][0]["camera"]["rules"][0]["enabled"] = False
    _write_manifest(bundle)
    result = offline.run_evaluation(offline.load_scenarios(bundle.path))
    assert result["items"] == []
    assert result["overall"]["fn"] == 1
    assert result["overall"]["recall"] == 0.0
    assert result["durations"]["sample.avi"] == pytest.approx(3.0)
    assert fake_models.backend_calls == []


def test_invalid_fps_is_rejected_and_capture_released(
    bundle, fake_models, monkeypatch
):
    class InvalidCapture:
        released = False

        def isOpened(self):
            return True

        def get(self, prop):
            return 0.0 if prop == cv2.CAP_PROP_FPS else 30.0

        def read(self):
            raise AssertionError("Invalid FPS must fail before decoding")

        def release(self):
            self.released = True

    capture = InvalidCapture()
    monkeypatch.setattr(offline.cv2, "VideoCapture", lambda path: capture)
    with pytest.raises(ValueError, match="invalid source FPS"):
        offline.run_evaluation(offline.load_scenarios(bundle.path))
    assert capture.released
    assert fake_models.backend.closed


def test_help_needs_no_model_loading(fake_models, capsys):
    with pytest.raises(SystemExit) as exc:
        offline.main(["--help"])
    assert exc.value.code == 0
    assert "--scenarios" in capsys.readouterr().out
    assert fake_models.backend_calls == []


def test_explicit_scenario_paths_override_guard_environment(
    bundle, fake_models, monkeypatch, tmp_path
):
    missing = tmp_path / "does-not-exist"
    assert not missing.exists()
    monkeypatch.setenv("GUARD_ROOT", str(missing))
    monkeypatch.setenv("GUARD_DATA_DIR", str(missing / "data"))
    monkeypatch.setenv("GUARD_MODELS_DIR", str(missing / "models"))
    monkeypatch.setenv("GUARD_MODELS_CONFIG", str(missing / "models.yaml"))

    suite = offline.load_scenarios(bundle.path)
    assert suite.labels_dir == bundle.labels.resolve()
    assert suite.models_config == bundle.models.resolve()
    assert suite.models_dir == (tmp_path / "models").resolve()
    assert Path(suite.scenarios[0].camera.source.uri) == bundle.video.resolve()

    result = offline.run_evaluation(suite)
    assert result["overall"]["tp"] == 1
    assert result["processing"]["frames_decoded"] == 30
    assert fake_models.backend_calls[0][1]["models_dir"] == suite.models_dir


def test_explicit_data_dir_overrides_guard_data_dir(
    bundle, monkeypatch, tmp_path
):
    bundle.raw.pop("videos_dir")
    bundle.raw.pop("labels_dir")
    bundle.raw["data_dir"] = "explicit-data"
    _write_manifest(bundle)
    monkeypatch.setenv("GUARD_DATA_DIR", str(tmp_path / "wrong-data"))

    suite = offline.load_scenarios(bundle.path)
    assert suite.labels_dir == (tmp_path / "explicit-data" / "labels").resolve()
    assert Path(suite.scenarios[0].camera.source.uri) == (
        tmp_path / "explicit-data" / "videos" / "sample.avi"
    ).resolve()


def test_environment_paths_are_used_when_yaml_omits_them(
    bundle, monkeypatch, tmp_path
):
    for key in ("root", "videos_dir", "labels_dir", "models_dir", "models_config"):
        bundle.raw.pop(key)
    _write_manifest(bundle)
    env_root = tmp_path / "shared"
    env_data = tmp_path / "shared-data"
    env_models = tmp_path / "shared-models"
    monkeypatch.setenv("GUARD_ROOT", str(env_root))
    monkeypatch.setenv("GUARD_DATA_DIR", str(env_data))
    monkeypatch.setenv("GUARD_MODELS_DIR", str(env_models))
    monkeypatch.setenv("GUARD_MODELS_CONFIG", str(bundle.models))

    suite = offline.load_scenarios(bundle.path)
    assert suite.labels_dir == (env_data / "labels").resolve()
    assert suite.models_dir == env_models.resolve()
    assert suite.models_config == bundle.models.resolve()
    assert Path(suite.scenarios[0].camera.source.uri) == (
        env_data / "videos" / "sample.avi"
    ).resolve()

    monkeypatch.delenv("GUARD_DATA_DIR", raising=False)
    suite = offline.load_scenarios(bundle.path)
    assert suite.labels_dir == (env_root / "data" / "labels").resolve()


def test_model_registry_directory_overrides_environment(
    bundle, monkeypatch, tmp_path
):
    bundle.raw.pop("models_dir")
    _write_manifest(bundle)
    registry = yaml.safe_load(bundle.models.read_text(encoding="utf-8"))
    registry["models_dir"] = "registry-weights"
    bundle.models.write_text(yaml.safe_dump(registry), encoding="utf-8")
    monkeypatch.setenv("GUARD_MODELS_DIR", str(tmp_path / "wrong-models"))

    suite = offline.load_scenarios(bundle.path)
    assert suite.models_dir == (bundle.models.parent / "registry-weights").resolve()

    # Более конкретный сценарий может переопределить каталог реестра.
    bundle.raw["models_dir"] = "scenario-weights"
    _write_manifest(bundle)
    suite = offline.load_scenarios(bundle.path)
    assert suite.models_dir == (tmp_path / "scenario-weights").resolve()


def test_cli_model_paths_override_yaml_and_environment(
    bundle, fake_models, monkeypatch, tmp_path
):
    bundle.raw["models_config"] = "missing-registry.yaml"
    bundle.raw["models_dir"] = "wrong-yaml-models"
    _write_manifest(bundle)
    monkeypatch.setenv("GUARD_MODELS_CONFIG", str(tmp_path / "wrong-env.yaml"))
    monkeypatch.setenv("GUARD_MODELS_DIR", str(tmp_path / "wrong-env-models"))

    cli_models = tmp_path / "cli-models"
    output = tmp_path / "cli-paths.json"
    assert offline.main(
        [
            "--scenarios", str(bundle.path),
            "--models-config", str(bundle.models),
            "--models-dir", str(cli_models),
            "--out", str(output),
        ]
    ) == 0
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["models_config"] == str(bundle.models.resolve())
    assert result["models_dir"] == str(cli_models.resolve())
    assert fake_models.backend_calls[0][1]["models_dir"] == cli_models.resolve()
    assert result["overall"]["tp"] == 1


def test_tracker_uses_live_settings_from_env(bundle, fake_models):
    result = offline.run_evaluation(offline.load_scenarios(bundle.path))
    expected = {
        "high_thresh": 0.4,
        "low_thresh": 0.1,
        "match_iou": 0.25,
        "new_track_thresh": 0.45,
        "track_buffer_s": 3.0,
        # Явная настройка тестовой Settings, остальные значения штатные.
        "min_hits": 1,
    }
    assert fake_models.settings_calls == 1
    assert len(fake_models.tracker_calls) == 2
    for call in fake_models.tracker_calls:
        assert {key: value for key, value in call.items() if key != "classes"} == expected
    assert fake_models.tracker_calls[0]["classes"] == set(offline.PERSON_CLASSES)
    assert fake_models.tracker_calls[1]["classes"] == set(offline.OBJECT_CLASSES)
    assert fake_models.event_manager_calls == 1
    assert len(fake_models.timestamps) == 15
    assert result["tracker"] == expected


def test_custom_settings_reach_both_trackers(bundle, fake_models):
    settings = replace(
        fake_models.settings,
        tracker_high_thresh=0.83,
        tracker_low_thresh=0.17,
        tracker_match_iou=0.41,
        tracker_new_track_thresh=0.87,
        tracker_buffer_s=4.5,
        tracker_min_hits=2,
    )
    result = offline.run_evaluation(
        offline.load_scenarios(bundle.path), settings=settings
    )
    expected = {
        "high_thresh": 0.83,
        "low_thresh": 0.17,
        "match_iou": 0.41,
        "new_track_thresh": 0.87,
        "track_buffer_s": 4.5,
        "min_hits": 2,
    }
    assert fake_models.settings_calls == 0
    assert len(fake_models.tracker_calls) == 2
    for call in fake_models.tracker_calls:
        assert {key: value for key, value in call.items() if key != "classes"} == expected
    assert result["tracker"] == expected
    assert fake_models.event_manager_calls == 1
    assert len(fake_models.timestamps) == 15
