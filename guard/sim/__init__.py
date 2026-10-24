"""Воспроизводимые синтетические сценарии без моделей и видеофайлов."""

from .scenarios import (
    FPS,
    FRAME_SIZE,
    SCENARIO_NAMES,
    START_TS,
    Scenario,
    generate_frames,
    make_scenario,
)

__all__ = [
    "FPS",
    "FRAME_SIZE",
    "SCENARIO_NAMES",
    "START_TS",
    "Scenario",
    "generate_frames",
    "make_scenario",
]
