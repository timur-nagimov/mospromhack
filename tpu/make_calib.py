#!/usr/bin/env python3
"""Нарезка калибровочных кадров для квантования (50-200 кадров производственной тематики).

    python tpu/make_calib.py --out /Users/nagimov-timur/Desktop/mosprom/models/tpu/calib_frames \
        --video data/videos/a3dev_worker_zone_detection/worker-zone-detection.mp4:80 \
        --video data/videos/a3dev_people_detection/people-detection.mp4:30

Кадры берутся равномерно по длине видео (кроме первых/последних 2 %), пишутся JPEG q=95,
источник и параметры — в SOURCES.json рядом с кадрами.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import cv2


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--video", action="append", required=True, help="путь[:кол-во кадров], по умолчанию 50")
    ap.add_argument("--note", default="", help="примечание об источнике/лицензии")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sources = []
    for spec in a.video:
        path, _, n = spec.partition(":")
        n = int(n or 50)
        cap = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        lo, hi = int(total * 0.02), int(total * 0.98)
        idxs = [lo + int(i * (hi - lo) / n) for i in range(n)]
        stem = Path(path).stem
        saved = 0
        for i in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, frame = cap.read()
            if not ok:
                continue
            cv2.imwrite(str(out / f"{stem}_{i:06d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved += 1
        h = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        sources.append({"video": str(Path(path).resolve()), "sha256": h, "frames_total": total, "saved": saved,
                        "size": [int(cap.get(3)), int(cap.get(4))]})
        print(f"{path}: {saved} кадров из {total}")
    (out / "SOURCES.json").write_text(json.dumps({"created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                                                  "note": a.note, "sources": sources}, indent=1,
                                                 ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
