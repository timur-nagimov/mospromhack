"""Конфиги стенда и секция edge (A10): согласованность с реестром A2 и демо-камерами A3, приоритеты."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from guard import edge
from guard.pipeline.config import load_cameras, load_models
from guard.pipeline.testing import make_settings
from guard.rules.base import available_rules, load_plugins

ROOT = Path(__file__).resolve().parents[1]
STAND_CAMERAS = ROOT / "configs" / "cameras.stand.yaml"
# Разрешённые отличия правил стенда от configs/cameras.yaml — только явные поправки под int8.
# min_conf зоны: уверенность int8 на TPU в среднем на ~0.05 ниже, чем fp32 на CPU (сверка A1, docs/LINQ_NOTES.md §16;
# на стенде 0.45 вместо 0.5 A9 — A10, docs/EDGE.md §3).
STAND_INT8_OVERRIDES: dict[tuple[str, str], dict[str, float]] = {("cam_s1", "s1_tape"): {"min_conf": 0.45}}
STAND_MODELS = ROOT / "configs" / "models.stand.yaml"


def test_stand_section_parses_and_points_to_tpu_registry():
    section = edge.load_edge_section(STAND_CAMERAS)
    assert section.backend == "linq-tpu"
    assert section.models_config == "configs/models.stand.yaml"
    assert section.linq.raw is True and section.linq.window == 4
    assert section.gateway_guard.enabled and section.gateway_guard.backlog_limit > 0


def test_stand_models_have_tpu_programs_and_match_main_registry():
    stand = load_models(STAND_MODELS)
    main = load_models(ROOT / "configs" / "models.yaml")
    assert {"det", "pose", "ppe"} <= set(stand)
    for model_id, spec in stand.items():
        assert spec.tpu_program and spec.tpu_program.endswith(".tpu"), model_id
        assert Path(spec.tpu_program).parent.name == Path(spec.tpu_program).stem.split("_b1_")[0]
        # классы и формат выхода те же, что у A2: правила работают с одними именами
        assert spec.classes == main[model_id].classes, model_id
        assert spec.output_format == main[model_id].output_format
    assert stand["det"].input_size == (480, 480) and stand["pose"].input_size == (480, 480)
    assert stand["det"].conf_threshold <= 0.3  # int8 на TPU занижает уверенность на ~0.05


def test_stand_cameras_match_demo_cameras_and_have_models():
    load_plugins()
    stand = {c.camera_id: c for c in load_cameras(STAND_CAMERAS)}
    demo = {c.camera_id: c for c in load_cameras(ROOT / "configs" / "cameras.yaml")}
    assert set(stand) == {"cam_s1", "cam_s2", "cam_s3", "cam_s4"}
    required = {r["type"]: set(r["required_models"]) for r in available_rules()}
    models = set(load_models(STAND_MODELS))
    videos = set()
    for cid, cam in stand.items():
        assert 4.0 <= cam.sample_fps <= 5.0
        assert all(z.active_hours is None for z in cam.zones)
        videos.add(Path(cam.source.uri).name)
        for rule in cam.rules:
            assert required[rule.type] <= models, (cid, rule.type)
        if cid in demo:  # gateway показывает камеры по своему конфигу: зоны и правила совпадают
            assert [z.model_dump() for z in cam.zones] == [z.model_dump() for z in demo[cid].zones]
            # и параметры правил тоже: иначе на TPU работают одни пороги, а в отчёте о качестве — другие.
            # Исключение — только явные поправки STAND_INT8_OVERRIDES: на стенде стоят именно они, остальное совпадает.
            stand_rules = []
            for r in cam.rules:
                dump = r.model_dump()
                override = STAND_INT8_OVERRIDES.get((cid, r.rule_id))
                if override:
                    assert {k: dump["params"].get(k) for k in override} == override, (cid, r.rule_id)
                    demo_rule = next(d for d in demo[cid].rules if d.rule_id == r.rule_id)
                    dump["params"] = {**dump["params"], **{k: demo_rule.params[k] for k in override}}
                stand_rules.append(dump)
            assert stand_rules == [r.model_dump() for r in demo[cid].rules]
            assert Path(cam.source.uri).name == Path(demo[cid].source.uri).name
    assert videos == {"s1_01.mp4", "s2_01.mp4", "s3_02.mp4", "s4_02.mp4"}


def test_section_rejects_unknown_keys(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text(yaml.safe_dump({"edge": {"bakend": "linq-tpu"}, "cameras": []}), encoding="utf-8")
    with pytest.raises(ValueError):
        edge.load_edge_section(cfg)
    assert edge.load_edge_section(tmp_path / "missing.yaml") == edge.EdgeSection()


def test_apply_section_respects_env_and_coerces_settings(tmp_path, monkeypatch):
    for name in ("GUARD_BACKEND", "GUARD_MODELS_CONFIG", "GUARD_MODELS_DIR", "GUARD_DATA_DIR",
                 "GUARD_MEDIA_DIR", "GUARD_DB_PATH", "GUARD_RABBIT_URL", "GUARD_FRAME_META"):
        monkeypatch.delenv(name, raising=False)
    settings = make_settings(tmp_path)
    section = edge.EdgeSection(
        backend="linq-tpu", models_config="configs/models.stand.yaml", models_dir=str(tmp_path / "m"),
        data_dir=str(tmp_path / "d"), settings={"max_batch": "4", "batch_timeout_ms": 2},
    )
    applied = edge.apply_edge_section(settings, section)
    assert settings.backend == "linq-tpu"
    assert settings.models_config == ROOT / "configs" / "models.stand.yaml"
    assert settings.data_dir == tmp_path / "d" and settings.media_dir == tmp_path / "d" / "media"
    assert settings.max_batch == 4 and isinstance(settings.max_batch, int)
    assert settings.batch_timeout_ms == 2.0 and isinstance(settings.batch_timeout_ms, float)
    assert "settings.max_batch" in applied

    monkeypatch.setenv("GUARD_BACKEND", "onnx-cpu")  # окружение важнее секции
    settings2 = make_settings(tmp_path, backend="onnx-cpu")
    edge.apply_edge_section(settings2, section)
    assert settings2.backend == "onnx-cpu"

    with pytest.raises(ValueError):
        edge.apply_edge_section(make_settings(tmp_path), edge.EdgeSection(settings={"data_dir": "/x"}))
    with pytest.raises(ValueError):
        edge.apply_edge_section(make_settings(tmp_path), edge.EdgeSection(settings={"no_such": 1}))


def test_default_rabbit_url_on_stand(monkeypatch):
    monkeypatch.setenv("JUPYTERHUB_USER", "hackathon10864")
    assert edge.default_rabbit_url() == "amqp://guest:guest@10.0.114.30:5672/"
    monkeypatch.delenv("JUPYTERHUB_USER")
    monkeypatch.setattr(edge, "on_stand", lambda: False)
    assert edge.default_rabbit_url() is None


def test_build_applies_stand_section(tmp_path, monkeypatch):
    for name in ("GUARD_BACKEND", "GUARD_MODELS_CONFIG", "GUARD_MODELS_DIR", "GUARD_RABBIT_URL", "LINQ_RAW"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("GUARD_DATA_DIR", str(tmp_path / "data"))
    cams = tmp_path / "cams.yaml"
    cams.write_text(yaml.safe_dump({
        "edge": {"backend": "linq-tpu", "node_id": "n1", "linq": {"raw": True, "window": 2}},
        "cameras": [],
    }), encoding="utf-8")
    args = edge.parse_args(["--config", str(cams), "--transport", "log", "--backend", "onnx-cpu"])
    settings, engine = edge.build(args)
    try:
        assert settings.mode == "edge" and settings.backend == "onnx-cpu"  # флаг CLI важнее секции
        assert engine.edge_node_id == "n1"
        assert isinstance(engine.sink, edge.EdgeSink) and engine.sink.guard is None  # log: защита не нужна
    finally:
        engine.stop()
