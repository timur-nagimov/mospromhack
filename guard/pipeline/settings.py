"""Настройки сервиса из окружения (GUARD_*) и .env корня проекта.

Код и конфиги (configs/) берутся из каталога пакета (CODE_ROOT), данные и модели —
из GUARD_DATA_DIR и GUARD_MODELS_DIR: worktree агентов делят общие data/ и models/.
"""

from __future__ import annotations

import os
import math
from dataclasses import dataclass, field
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[2]
_BACKEND_ALIASES = {"onnx": "onnx-cpu", "cpu": "onnx-cpu", "ort": "onnx-cpu", "tpu": "linq-tpu", "linq": "linq-tpu"}


def normalize_backend_name(name: str) -> str:
    """Configuration-only normalization; importing Settings must not load ML."""
    key = name.strip().lower().replace("_", "-")
    return _BACKEND_ALIASES.get(key, key)


def project_root() -> Path:
    """Portable shared asset/environment root; code/config defaults stay CODE_ROOT."""
    return Path(os.environ.get("GUARD_ROOT") or CODE_ROOT).expanduser()


def find_dotenv(start: Path | None = None) -> Path | None:
    start = Path(start) if start is not None else project_root()
    return next((directory / ".env" for directory in (start, *start.parents)
                 if (directory / ".env").is_file()), None)


def load_dotenv_guard(start: Path | None = None) -> Path | None:
    """Найти .env вверх от start и загрузить только GUARD_* (секреты в окружение не тянем).

    Уже заданные переменные окружения не перезаписываются."""
    env = find_dotenv(start)
    if env is None:
        return None
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip().removeprefix("export ").strip()
        if key.startswith("GUARD_"):
            os.environ.setdefault(key, value.strip().strip("'\""))
    return env


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return float(raw) if raw else default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw else default


@dataclass
class Settings:
    data_dir: Path
    models_dir: Path
    cameras_config: Path
    models_config: Path
    db_path: Path
    media_dir: Path
    frontend_dist: Path
    backend: str = "onnx-cpu"
    mode: str = "local"
    edge_id: str | None = None
    rabbit_url: str | None = field(default=None, repr=False)
    preview_fps: float = 1.0
    preview_max_width: int = 480
    preview_stale_after_s: float = 10.0
    telemetry_interval_s: float = 2.0
    autostart: bool = True
    # режим запуска (D-005): local — всё в одном процессе; edge — конвейер у TPU, публикует
    # event/frame_meta/metrics в транспорт; gateway — API/UI/журнал/клипы на ноутбуке
    transport: str = "rabbit"  # memory | rabbit | log
    publish_frame_meta: bool = True
    clips_enabled: bool = True  # на edge выключено: клипы собирает gateway из своего буфера
    edge_snapshot_width: int = 640
    edge_snapshot_quality: int = 70
    edge_snapshot_max_kb: int = 60
    # батчинг между камерами: включается, если бэкенд принимает batch > 1
    batching: bool = True
    max_batch: int = 8
    batch_timeout_ms: float = 4.0
    ppe_input_mode: str = "full_frame"
    # клипы и кольцевой буфер
    clip_pre_s: float = 5.0
    clip_post_s: float = 5.0
    clip_fps: float = 10.0
    clip_max_width: int = 640
    clip_buffer_mb: int = 512
    # поток MJPEG
    stream_fps: float = 10.0
    stream_max_width: int = 1280
    jpeg_quality: int = 80
    # трекер
    tracker_high_thresh: float = 0.4
    tracker_low_thresh: float = 0.1
    tracker_new_track_thresh: float = 0.45
    tracker_match_iou: float = 0.25
    tracker_buffer_s: float = 3.0
    tracker_min_hits: int = 2
    # люди для трекера: к det добавить людей pose, которых det не видит (лежачие, S3; docs/QUALITY.md)
    person_pose_fill: bool = False
    ort_threads: int = 2
    cors_origins: list[str] = field(default_factory=lambda: ["http://localhost:5173", "http://127.0.0.1:5173"])
    # маховик данных (docs/FLYWHEEL.md): сбор кадров в обучающую выборку
    flywheel: bool = True
    flywheel_dir: Path | None = None  # None — <data_dir>/flywheel
    flywheel_period_s: float | None = None  # период uncertain и flicker на камеру; None — 10 с, на edge 30 с
    flywheel_daily_cap: int = 500  # сэмплов в сутки на камеру, кроме event и operator
    flywheel_shadow: bool = False  # сверка TPU с CPU на выборочных кадрах (только linq-tpu)
    flywheel_shadow_period_s: float = 30.0
    flywheel_allow_train: bool = False  # задачи train и deploy из UI
    flywheel_edge_kbps: float = 4.0  # edge: прибавка трафика на камеру, КБ/с (контракт — не больше 10)

    def __post_init__(self) -> None:
        if self.ppe_input_mode not in {"full_frame", "person_crop"}:
            raise ValueError("GUARD_PPE_INPUT_MODE must be full_frame or person_crop")
        if self.mode not in {"local", "edge", "gateway"}:
            raise ValueError("GUARD_MODE must be local, edge or gateway")
        if self.mode != "local" and not self.edge_id:
            raise ValueError("GUARD_EDGE_ID is required for edge and gateway")
        if self.edge_id and (len(self.edge_id) > 100 or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-" for c in self.edge_id)):
            raise ValueError("Invalid GUARD_EDGE_ID")
        if not math.isfinite(self.preview_fps) or not 0 < self.preview_fps <= 1:
            raise ValueError("GUARD_PREVIEW_FPS must be in (0, 1]")
        if not 1 <= self.preview_max_width <= 480:
            raise ValueError("GUARD_PREVIEW_MAX_WIDTH must be in [1, 480]")
        if not math.isfinite(self.preview_stale_after_s) or self.preview_stale_after_s <= 0:
            raise ValueError("GUARD_PREVIEW_STALE_AFTER_S must be positive")
        if not math.isfinite(self.telemetry_interval_s) or self.telemetry_interval_s <= 0:
            raise ValueError("GUARD_TELEMETRY_INTERVAL_S must be positive")

    @property
    def root(self) -> Path:
        return self.data_dir.parent

    @property
    def flywheel_root(self) -> Path:
        return Path(self.flywheel_dir).expanduser() if self.flywheel_dir else self.data_dir / "flywheel"

    @classmethod
    def from_env(cls) -> Settings:
        load_dotenv_guard()
        data_dir = Path(os.environ.get("GUARD_DATA_DIR") or project_root() / "data").expanduser()
        models_dir = Path(os.environ.get("GUARD_MODELS_DIR") or project_root() / "models").expanduser()
        return cls(
            data_dir=data_dir,
            models_dir=models_dir,
            cameras_config=Path(os.environ.get("GUARD_CAMERAS_CONFIG") or CODE_ROOT / "configs" / "cameras.yaml"),
            models_config=Path(os.environ.get("GUARD_MODELS_CONFIG") or CODE_ROOT / "configs" / "models.yaml"),
            db_path=Path(os.environ.get("GUARD_DB_PATH") or data_dir / "guard.db"),
            media_dir=Path(os.environ.get("GUARD_MEDIA_DIR") or data_dir / "media"),
            frontend_dist=Path(os.environ.get("GUARD_FRONTEND_DIST") or CODE_ROOT / "frontend" / "dist"),
            backend=normalize_backend_name(os.environ.get("GUARD_BACKEND") or "onnx-cpu"),
            edge_id=os.environ.get("GUARD_EDGE_ID") or None,
            preview_fps=_env_float("GUARD_PREVIEW_FPS", 1.0),
            preview_max_width=_env_int("GUARD_PREVIEW_MAX_WIDTH", 480),
            preview_stale_after_s=_env_float("GUARD_PREVIEW_STALE_AFTER_S", 10.0),
            telemetry_interval_s=_env_float("GUARD_TELEMETRY_INTERVAL_S", 2.0),
            autostart=_env_bool("GUARD_AUTOSTART", True),
            mode=(os.environ.get("GUARD_MODE") or "local").strip().lower(),
            transport=(os.environ.get("GUARD_TRANSPORT") or "rabbit").strip().lower(),
            rabbit_url=os.environ.get("GUARD_RABBIT_URL") or None,
            publish_frame_meta=_env_bool("GUARD_FRAME_META", True),
            clips_enabled=_env_bool("GUARD_CLIPS", (os.environ.get("GUARD_MODE") or "local").strip().lower() != "edge"),
            edge_snapshot_width=_env_int("GUARD_EDGE_SNAPSHOT_WIDTH", 640),
            edge_snapshot_quality=_env_int("GUARD_EDGE_SNAPSHOT_QUALITY", 70),
            edge_snapshot_max_kb=_env_int("GUARD_EDGE_SNAPSHOT_MAX_KB", 60),
            batching=_env_bool("GUARD_BATCHING", True),
            max_batch=_env_int("GUARD_MAX_BATCH", 8),
            batch_timeout_ms=_env_float("GUARD_BATCH_TIMEOUT_MS", 4.0),
            ppe_input_mode=(os.environ.get("GUARD_PPE_INPUT_MODE") or "full_frame").strip().lower(),
            clip_pre_s=_env_float("GUARD_CLIP_PRE_S", 5.0),
            clip_post_s=_env_float("GUARD_CLIP_POST_S", 5.0),
            clip_fps=_env_float("GUARD_CLIP_FPS", 10.0),
            clip_max_width=_env_int("GUARD_CLIP_MAX_WIDTH", 640),
            clip_buffer_mb=_env_int("GUARD_CLIP_BUFFER_MB", 512),
            stream_fps=_env_float("GUARD_STREAM_FPS", 10.0),
            person_pose_fill=_env_bool("GUARD_PERSON_POSE_FILL", False),
            ort_threads=_env_int("GUARD_ORT_THREADS", 2),
            flywheel=_env_bool("GUARD_FLYWHEEL", True),
            flywheel_dir=Path(os.environ["GUARD_FLYWHEEL_DIR"]).expanduser() if os.environ.get("GUARD_FLYWHEEL_DIR") else None,
            flywheel_period_s=_env_float("GUARD_FLYWHEEL_PERIOD_S", 0.0) or None,
            flywheel_daily_cap=_env_int("GUARD_FLYWHEEL_DAILY_CAP", 500),
            flywheel_shadow=_env_bool("GUARD_FLYWHEEL_SHADOW", False),
            flywheel_shadow_period_s=_env_float("GUARD_FLYWHEEL_SHADOW_PERIOD_S", 30.0),
            flywheel_allow_train=_env_bool("GUARD_FLYWHEEL_ALLOW_TRAIN", False),
            flywheel_edge_kbps=_env_float("GUARD_FLYWHEEL_EDGE_KBPS", 4.0),
        )
