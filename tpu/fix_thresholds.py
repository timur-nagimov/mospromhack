#!/usr/bin/env python3
"""Поднять пороги квантования выходов Sigmoid/Softmax до 1.0 в thresholds.json (dnn_quant).

Зачем: калибровка max задаёт порог выхода Sigmoid по максимуму на калибровочных кадрах. Если там мало
объектов класса, int8-диапазон обрезает уверенность сверху. У fire_640 потолок 0.42/0.50/0.57 по страйдам
8/16/32, у yolov8n_640 — 0.705 на страйде 8. Сигмоида ограничена [0, 1], поэтому порог 1.0 безопасен:
шаг 1/127 ≈ 0.008.

Правило: слой квантования с min_value >= 0 и порогом < 1.0 — выход сигмоиды или softmax (у SiLU min < 0).

    python tpu/fix_thresholds.py in/thresholds.json out/thresholds_sig1.json
    python tpu/compile.py ... --thresholds out/thresholds_sig1.json   # калибровка пропускается
"""

from __future__ import annotations

import json
import sys


def fix(t: dict, eps: float = 1e-6) -> list[tuple[str, float]]:
    changed = []
    for name, layer in t.get("layers", {}).items():
        td = layer.get("thresholds_data") or {}
        out = td.get("outputs_thresholds_data")
        if not out or name == "images" or layer.get("type") == "input_node":
            continue
        mn = ((td.get("min_value") or {}).get("0") or {}).get("$array")
        mx = ((td.get("max_value") or {}).get("0") or {}).get("$array")
        th = (out.get("0") or {}).get("$array")
        if not isinstance(th, (int, float)) or mn is None or mn < -eps or th >= 1.0:
            continue
        out["0"]["$array"] = 1.0
        if isinstance(mx, (int, float)):
            td["max_value"]["0"]["$array"] = max(mx, 1.0)
        changed.append((name, round(float(th), 4)))
    return changed


def main() -> int:
    src, dst = sys.argv[1], sys.argv[2]
    t = json.load(open(src))
    changed = fix(t)
    json.dump(t, open(dst, "w"))
    for n, th in changed:
        print(f"{n}: {th} -> 1.0")
    print(f"изменено слоёв: {len(changed)} -> {dst}")
    return 0 if changed else 1


if __name__ == "__main__":
    raise SystemExit(main())
