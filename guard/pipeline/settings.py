"""Настройки сервиса из окружения (GUARD_*) и .env корня проекта.

Код и конфиги (configs/) берутся из каталога пакета (CODE_ROOT), данные и модели —
из GUARD_DATA_DIR и GUARD_MODELS_DIR: worktree агентов делят общие data/ и models/.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[2]


def load_dotenv_guard(start: Path = CODE_ROOT) -> Path | None:
    """Найти .env вверх от start и загрузить только GUARD_* (секреты в окружение не тянем).

    Уже заданные переменные окружения не перезаписываются."""
    for d in (start, *start.parents):
        env = d / ".env"
        if env.is_file():
            for line in env.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip().removeprefix("export ").strip()
                if key.startswith("GUARD_"):
                    os.environ.setdefault(key, value.strip().strip("'\""))
            return env
    return None


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
    autostart: bool = True
    # режим запуска (D-005): local — всё в одном процессе; edge — конвейер у TPU, публикует
    # event/frame_meta/metrics в транспорт; gateway — API/UI/журнал/клипы на ноутбуке
    mode: str = "local"
    transport: str = "rabbit"  # memory | rabbit | log
    rabbit_url: str | None = None
    publish_frame_meta: bool = True
    clips_enabled: bool = True  # на edge выключено: клипы собирает gateway из своего буфера
    edge_snapshot_width: int = 640
    edge_snapshot_quality: int = 70
    edge_snapshot_max_kb: int = 60
    # батчинг между камерами: включается, если бэкенд принимает batch > 1
    batching: bool = True
    max_batch: int = 8
    batch_timeout_ms: float = 4.0
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

    @property
    def root(self) -> Path:
        return self.data_dir.parent

    @property
    def flywheel_root(self) -> Path:
        return Path(self.flywheel_dir).expanduser() if self.flywheel_dir else self.data_dir / "flywheel"

    @classmethod
    def from_env(cls) -> Settings:
        load_dotenv_guard()
        from guard.inference import backend_name_from_env

        data_dir = Path(os.environ.get("GUARD_DATA_DIR") or CODE_ROOT / "data").expanduser()
        models_dir = Path(os.environ.get("GUARD_MODELS_DIR") or CODE_ROOT / "models").expanduser()
        return cls(
            data_dir=data_dir,
            models_dir=models_dir,
            cameras_config=Path(os.environ.get("GUARD_CAMERAS_CONFIG") or CODE_ROOT / "configs" / "cameras.yaml"),
            models_config=Path(os.environ.get("GUARD_MODELS_CONFIG") or CODE_ROOT / "configs" / "models.yaml"),
            db_path=Path(os.environ.get("GUARD_DB_PATH") or data_dir / "guard.db"),
            media_dir=Path(os.environ.get("GUARD_MEDIA_DIR") or data_dir / "media"),
            frontend_dist=Path(os.environ.get("GUARD_FRONTEND_DIST") or CODE_ROOT / "frontend" / "dist"),
            backend=backend_name_from_env(),
            autostart=_env_bool("GUARD_AUTOSTART", True),
            mode=(os.environ.get("GUARD_MODE") or "local").strip().lower(),
            transport=(os.environ.get("GUARD_TRANSPORT") or "rabbit").strip().lower(),
            rabbit_url=os.environ.get("GUARD_RABBIT_URL") or None,
            publish_frame_meta=_env_bool("GUARD_FRAME_META", True),
            clips_enabled=_env_bool("GUARD_CLIPS", (os.environ.get("GUARD_MODE") or "local").strip().lower() != "edge"),
            batching=_env_bool("GUARD_BATCHING", True),
            max_batch=_env_int("GUARD_MAX_BATCH", 8),
            batch_timeout_ms=_env_float("GUARD_BATCH_TIMEOUT_MS", 4.0),
            clip_pre_s=_env_float("GUARD_CLIP_PRE_S", 5.0),
            clip_post_s=_env_float("GUARD_CLIP_POST_S", 5.0),
            clip_fps=_env_float("GUARD_CLIP_FPS", 10.0),
            clip_max_width=_env_int("GUARD_CLIP_MAX_WIDTH", 640),
            clip_buffer_mb=_env_int("GUARD_CLIP_BUFFER_MB", 512),
            stream_fps=_env_float("GUARD_STREAM_FPS", 10.0),
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
