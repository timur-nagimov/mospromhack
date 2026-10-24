"""Листы кадров ролика для разметки: кадр каждые N секунд с отметкой времени, сеткой листов JPEG.

Разметчик смотрит листы и записывает начало и конец нарушений. Сетка `--grid` рисует линии с шагом 0.1
в нормированных координатах кадра — по ней удобно снимать вершины полигонов зон.

usage: python tools/contact_sheet.py VIDEO --out DIR [--every 1.0] [--start 0] [--end END]
                                     [--cols 4] [--rows 4] [--tile-w 480] [--grid] [--quality 80]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


def _label(img: np.ndarray, text: str) -> None:
    cv2.rectangle(img, (0, 0), (12 + 11 * len(text), 26), (0, 0, 0), -1)
    cv2.putText(img, text, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


def _grid(img: np.ndarray) -> None:
    h, w = img.shape[:2]
    for i in range(1, 10):
        x, y = round(w * i / 10), round(h * i / 10)
        cv2.line(img, (x, 0), (x, h), (0, 255, 255), 1)
        cv2.line(img, (0, y), (w, y), (0, 255, 255), 1)
        cv2.putText(img, f"{i / 10:.1f}", (x + 2, h - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
        cv2.putText(img, f"{i / 10:.1f}", (2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)


def frames(path: Path, every: float, start: float, end: float | None):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"не открыть {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
    stop = min(end, duration) if end is not None else duration
    t = start
    while t < stop:
        cap.set(cv2.CAP_PROP_POS_FRAMES, round(t * fps))
        ok, frame = cap.read()
        if not ok:
            break
        yield t, frame
        t = round(t + every, 3)
    cap.release()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("video", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--every", type=float, default=1.0, help="шаг между кадрами, с")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=None)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--tile-w", type=int, default=480)
    ap.add_argument("--grid", action="store_true", help="сетка 0.1 в нормированных координатах")
    ap.add_argument("--quality", type=int, default=80, help="качество JPEG; 70 и ширина плитки 256 — лёгкие листы")
    args = ap.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    per_sheet = args.cols * args.rows
    tiles: list[np.ndarray] = []
    written: list[Path] = []

    def flush() -> None:
        if not tiles:
            return
        th, tw = tiles[0].shape[:2]
        sheet = np.zeros((th * args.rows, tw * args.cols, 3), np.uint8)
        for i, tile in enumerate(tiles):
            r, c = divmod(i, args.cols)
            sheet[r * th:(r + 1) * th, c * tw:(c + 1) * tw] = tile
        path = args.out / f"{args.video.stem}_sheet{len(written):03d}.jpg"
        cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, args.quality])
        written.append(path)
        tiles.clear()

    for t, frame in frames(args.video, args.every, args.start, args.end):
        h, w = frame.shape[:2]
        tile = cv2.resize(frame, (args.tile_w, round(h * args.tile_w / w)), interpolation=cv2.INTER_AREA)
        if args.grid:
            _grid(tile)
        _label(tile, f"t={t:.1f}s")
        tiles.append(tile)
        if len(tiles) == per_sheet:
            flush()
    flush()
    print("\n".join(str(p) for p in written))
    return 0 if written else 1


if __name__ == "__main__":
    sys.exit(main())
