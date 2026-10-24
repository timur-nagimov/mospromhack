"""Выкатка маховика (ml/flywheel/deploy.py): имена, отбор кадров, сводка сверки, порог TPU, запись в реестры.
Без стенда, torch и моделей: шаги compile и check проверяются только в --dry-run."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from ml.flywheel import deploy

STAND_YAML = """\
models:
  - model_id: det
    task: detect
    conf_threshold: 0.25

  # --- погрузчик (S4): записи ds-forklift ниже ---

  # --- лежащий человек (S3): записи ds-falls ниже ---
"""

CPU_YAML = """\
models:
  - model_id: det
    task: detect

  # --- погрузчик (S4), дообучение на открытых наборах: записи ds-forklift ниже ---
  - model_id: forklift             # forklift-v1: старый комментарий
    task: detect
    onnx_path: forklift_v1-480.onnx
    input_size: [480, 480]
    output_format: yolov8
    conf_threshold: 0.3            # трек рождается от 0.45 (трекер)
    iou_threshold: 0.5
    classes: [person, forklift]    # person не используется: люди берутся из det

  # --- лежащий человек (S3), наборы падений: записи ds-falls ниже ---
"""


def _card(**kw):
    card = {"model_id": "forklift", "version": "forklift-v1", "parent": "yolov8n-coco", "created_at": "x",
            "task": "detect", "classes": ["person", "forklift"], "status": "candidate",
            "onnx": "forklift_v1-480.onnx", "weights": "models/weights/forklift-v1.pt", "gate": {"promote": True}}
    card.update(kw)
    return card


def _img(path: Path, label: str | None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(path.name.encode())  # разное содержимое — разный sha1
    lab = Path(str(path.parent).replace("images", "labels")) / (path.stem + ".txt")
    lab.parent.mkdir(parents=True, exist_ok=True)
    lab.write_text(label or "")
    return path


def test_names_match_registry_style():
    assert deploy.onnx_name("forklift-v1", 480) == "forklift_v1-480.onnx"
    assert deploy.program_name("forklift-v1", 480) == "forklift_v1_480"
    assert deploy.program_name("forklift-v1", 480, s1=True) == "forklift_v1_480s1"
    assert deploy.program_path("forklift_v1_480s1") == "tpu/forklift_v1_480s1/forklift_v1_480s1_b1_o5_h1m.tpu"


def test_parser_and_unknown_step(monkeypatch):
    with pytest.raises(SystemExit):
        deploy.build_parser().parse_args([])
    args = deploy.build_parser().parse_args(["--version", "forklift-v1"])
    assert args.check_conf == 0.3 and args.birth_conf == 0.45 and args.program == "s1"
    monkeypatch.setattr(deploy, "load_card", lambda v: pytest.fail("карточка не нужна при ошибке шагов"))
    assert deploy.main(["--version", "forklift-v1", "--steps", "export,publish"]) == 2


def test_select_check_images_takes_all_class_frames_then_random(tmp_path):
    test = tmp_path / "images" / "test"
    with_cls = [_img(test / f"f{i}.jpg", "1 0.5 0.5 0.2 0.2\n") for i in range(3)]
    others = [_img(test / f"o{i}.jpg", "0 0.5 0.5 0.2 0.2\n" if i % 2 else None) for i in range(10)]
    extra = [tmp_path / "s4_02_t020.0.jpg"]
    picked = deploy.select_check_images(with_cls + others, 1, 6, extra)
    assert picked[:3] == sorted(with_cls)
    assert len(picked) == 7 and picked[-1] == extra[0]
    assert set(picked[3:6]) <= set(others)
    assert picked == deploy.select_check_images(with_cls + others, 1, 6, extra)  # детерминированно


def test_calib_extra_skips_video_frames_and_check_copies(tmp_path):
    calib = tmp_path / "calib"
    calib.mkdir()
    for name in ("calib_000_cs.jpg", "calib_001_fk.jpg", "calib_170_video.jpg"):
        (calib / name).write_bytes(name.encode())
    with (calib / "manifest.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "theme", "source", "original", "license"])
        w.writerow(["calib_000_cs.jpg", "construction", "https://hf", "a.jpg", "CC"])
        w.writerow(["calib_001_fk.jpg", "warehouse", "https://hf", "b.jpg", "CC"])
        w.writerow(["calib_170_video.jpg", "demo_camera", "data/videos/a3dev/x.mp4", "frame 1", "-"])
    import hashlib

    dup = hashlib.sha1(b"calib_001_fk.jpg").hexdigest()
    frames = deploy.calib_extra_frames(calib, {dup})
    assert [p.name for p in frames] == ["calib_000_cs.jpg"]


def _yolo_check(prog, recall, precision, sweep, perf=None):
    acc = {"images": 100, "conf": 0.3, "cpu_dets": 40, "recall_vs_cpu": recall, "precision_vs_cpu": precision,
           "iou_mean": 0.93, "dconf_signed_mean": -0.04,
           "tpu_threshold_sweep": {k: {"recall_vs_cpu": r, "precision_vs_cpu": p, "tpu_dets": 40}
                                   for k, (r, p) in sweep.items()}}
    return {f"{prog}_raw": {"model": prog, "accuracy_vs_cpu": acc, "perf": perf or {}}}


def test_summarize_check_reads_yolo_check_format_and_suggests_conf():
    perf = {"cpu_ort_2threads_ms": {"p50": 90}, "window_1": {"fps": 14.0, "call_ms": {"total": {"p50": 70.0}}}}
    res = {"forklift_v1_480s1": {
        "forklift@0.3": _yolo_check("forklift_v1_480s1", 0.9, 1.0,
                                    {"0.3": (0.9, 1.0), "0.25": (0.96, 0.97), "0.2": (0.98, 0.85)}),
        "all@0.3": _yolo_check("forklift_v1_480s1", 0.92, 0.98, {"0.3": (0.92, 0.98)}, perf),
    }}
    summary = deploy.summarize_check(res)
    item = summary["forklift_v1_480s1"]["forklift@0.3"]
    assert item["recall"] == 0.9 and item["frames"] == 100 and item["dconf_mean"] == -0.04
    assert item["sweep"]["0.25"] == {"recall": 0.96, "precision": 0.97}
    assert summary["forklift_v1_480s1"]["all@0.3"]["perf"] == {"window_1": {"fps": 14.0, "total_p50_ms": 70.0}}
    # самый высокий порог, где recall ≥ 0.95 и precision ≥ 0.9
    assert deploy.suggest_tpu_conf(summary["forklift_v1_480s1"], "forklift@0.3") == 0.25
    # нет подходящего — лучший по сумме recall + precision
    weak = {"t": {"sweep": {"0.3": {"recall": 0.7, "precision": 1.0}, "0.25": {"recall": 0.8, "precision": 0.95}}}}
    assert deploy.suggest_tpu_conf(weak, "t") == 0.25
    assert deploy.suggest_tpu_conf({}, "t") is None


def test_upsert_inserts_under_anchor_and_keeps_rest():
    entry = deploy.entry_text(_card(), 480, "tpu/forklift_v1_480s1/forklift_v1_480s1_b1_o5_h1m.tpu", 0.25, "")
    new = deploy.upsert_entry(STAND_YAML, "погрузчик (S4)", "forklift", entry)
    lines = new.splitlines()
    i = lines.index("  # --- погрузчик (S4): записи ds-forklift ниже ---")
    assert lines[i + 1].startswith("  - model_id: forklift ")
    assert "    conf_threshold: 0.25" in lines
    assert new.replace(entry, "") == STAND_YAML  # остальной файл не тронут


def test_upsert_replaces_entry_and_keeps_comments_of_unchanged_lines():
    prog = "tpu/forklift_v1_480s1/forklift_v1_480s1_b1_o5_h1m.tpu"
    new = deploy.upsert_entry(CPU_YAML, "погрузчик (S4)", "forklift",
                              deploy.entry_text(_card(), 480, prog, 0.3, ""))
    assert new.count("model_id: forklift") == 1
    assert f"    tpu_program: {prog}" in new
    assert "conf_threshold: 0.3            # трек рождается от 0.45 (трекер)" in new
    assert "# person не используется: люди берутся из det" in new
    assert "старый комментарий" not in new
    assert new.endswith("  # --- лежащий человек (S3), наборы падений: записи ds-falls ниже ---\n")


def test_upsert_errors_without_anchor_or_with_duplicate_elsewhere():
    entry = deploy.entry_text(_card(), 480, "tpu/x/x_b1_o5_h1m.tpu", 0.25, "")
    with pytest.raises(ValueError, match="нет якоря"):
        deploy.upsert_entry("models:\n  - model_id: det\n", "погрузчик (S4)", "forklift", entry)
    elsewhere = STAND_YAML.replace("  - model_id: det\n", "  - model_id: forklift\n    task: detect\n")
    with pytest.raises(ValueError, match="не под якорем"):
        deploy.upsert_entry(elsewhere, "погрузчик (S4)", "forklift", entry)


def _registry_env(tmp_path, monkeypatch):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/models.yaml").write_text(CPU_YAML, encoding="utf-8")
    (tmp_path / "configs/models.stand.yaml").write_text(STAND_YAML, encoding="utf-8")
    monkeypatch.setattr(deploy, "ROOT", tmp_path)
    saved = []
    monkeypatch.setattr(deploy, "save_card", lambda c: saved.append(dict(c)) or tmp_path / "card.json")
    old = _card(version="forklift-v0", status="deployed")
    monkeypatch.setattr(deploy, "list_cards", lambda: [old])
    return saved


def test_registry_uses_check_suggestion_and_archives_previous(tmp_path, monkeypatch):
    saved = _registry_env(tmp_path, monkeypatch)
    card = _card(tpu_check={"forklift_v1_480s1": {"forklift@0.3": {
        "sweep": {"0.3": {"recall": 0.9, "precision": 1.0}, "0.25": {"recall": 0.97, "precision": 0.95}}}}})
    args = deploy.build_parser().parse_args(["--version", "forklift-v1", "--steps", "registry"])
    deploy.step_registry(card, args)
    stand = (tmp_path / "configs/models.stand.yaml").read_text(encoding="utf-8")
    cpu = (tmp_path / "configs/models.yaml").read_text(encoding="utf-8")
    assert "    conf_threshold: 0.25" in stand and "forklift_v1_480s1_b1_o5_h1m.tpu" in stand
    assert "conf_threshold: 0.3            # трек рождается" in cpu and "forklift_v1_480s1_b1_o5_h1m.tpu" in cpu
    assert card["status"] == "deployed"
    assert card["tpu_program"] == "tpu/forklift_v1_480s1/forklift_v1_480s1_b1_o5_h1m.tpu"
    assert [c["version"] for c in saved] == ["forklift-v0"] and saved[0]["status"] == "archived"


def test_registry_dry_run_writes_nothing_and_gate_is_required(tmp_path, monkeypatch, capsys):
    saved = _registry_env(tmp_path, monkeypatch)
    args = deploy.build_parser().parse_args(["--version", "forklift-v1", "--tpu-conf", "0.3", "--dry-run"])
    card = _card()
    deploy.step_registry(card, args)
    assert (tmp_path / "configs/models.stand.yaml").read_text(encoding="utf-8") == STAND_YAML
    assert card["status"] == "candidate" and saved == []
    assert "+  - model_id: forklift" in capsys.readouterr().out
    with pytest.raises(RuntimeError, match="не прошла гейт"):
        deploy.step_registry(_card(gate={"promote": False}), args)
    no_check = deploy.build_parser().parse_args(["--version", "forklift-v1", "--dry-run"])
    with pytest.raises(ValueError, match="нет сверки"):
        deploy.step_registry(_card(), no_check)


def test_main_dry_run_prints_stand_commands_without_stand(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(deploy, "load_card", lambda v: _card())
    monkeypatch.setattr(deploy, "save_card", lambda c: pytest.fail("dry-run не пишет карточку"))
    monkeypatch.setattr(deploy, "MODELS_DIR", tmp_path / "models")
    data = tmp_path / "ds"
    _img(data / "images" / "train" / "a.jpg", "1 0.5 0.5 0.2 0.2\n")
    _img(data / "images" / "test" / "b.jpg", "1 0.5 0.5 0.2 0.2\n")
    (data / "data.yaml").write_text(f"path: {data}\ntrain: images/train\nval: images/train\ntest: images/test\n"
                                    "names: [person, forklift]\n")
    calib = tmp_path / "calib"
    calib.mkdir()
    (calib / "c.jpg").write_bytes(b"c")
    code = deploy.main(["--version", "forklift-v1", "--data", str(data / "data.yaml"), "--calib-extra", str(calib),
                        "--steps", "export,calib,compile,check", "--dry-run"])
    out = capsys.readouterr().out
    assert code == 0
    assert "ml/export_onnx.py" in out and "calib: 2 кадров (1 с forklift)" in out
    assert "compile.py --onnx models/forklift_v1-480.onnx --name forklift_v1_480 " in out
    assert "fix_thresholds.py" in out and "--name forklift_v1_480s1" in out
    assert "yolo_check.py --model forklift_v1_480s1" in out and "--classes 1" in out
    assert "params.models: [det, forklift]" in out and "restart" in out
