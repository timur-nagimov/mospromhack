"""Сборка калибровочного набора для квантования под LinQ TPU (data/calib/, JPEG, 200 кадров).

Смесь тематик: стройка, склад и погрузчики, цех и производство, плюс кадры демо-видео (домен камер).
Отбор детерминированный (seed), тестовые кадры из data/images/samples исключаются.
Рядом пишется manifest.csv: файл, тематика, источник, исходное имя, лицензия.

    python -m ml.build_calib --work ./downloaded --out ./data/calib

По умолчанию data/calib и demo videos берутся из GUARD_DATA_DIR либо checkout/data.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import shutil
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]

SOURCES = {
    # ключ: (тематика, лицензия, url)
    "cs": ("construction", "CC BY 4.0",
           "https://huggingface.co/datasets/keremberke/construction-safety-object-detection (Roboflow Construction Site Safety)"),
    "forklift": ("warehouse", "CC BY 4.0",
                 "https://huggingface.co/datasets/keremberke/forklift-object-detection (Roboflow, images.cv)"),
    "sh17": ("factory", "Pexels License (фото); зеркало Roboflow — CC BY 4.0",
             "https://huggingface.co/datasets/fathansanum/SH-17-Dataset (SH17, фото Pexels)"),
    "video": ("demo_camera", "см. docs/DATA.md (A5)", "data/videos/*"),
}


def data_dir() -> Path:
    return Path(os.environ.get("GUARD_DATA_DIR") or ROOT / "data").expanduser()


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", required=True, help="каталог со скачанными наборами: cs/, forklift/, sh17/")
    ap.add_argument("--out", default=str(data_dir() / "calib"))
    ap.add_argument("--n-cs", type=int, default=60)
    ap.add_argument("--n-forklift", type=int, default=50)
    ap.add_argument("--n-sh17", type=int, default=60)
    ap.add_argument("--n-video", type=int, default=30)
    ap.add_argument("--seed", type=int, default=12)
    return ap


def main(argv=None) -> int:
    a = parser().parse_args(argv)

    work, out = Path(a.work).expanduser(), Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("calib_*.jpg"):
        old.unlink()
    rnd = random.Random(a.seed)
    # исходные имена тестовых кадров не должны попасть в калибровку
    exclude = {
        "NX_img_177_jpg.rf.ce6664324494f1b3c9e13f45911e452c.jpg",
        "helmet999939_jpg.rf.c667b976d357ac84e5ca128824a6d753.jpg",
        "6OV3JZN7K67E_jpg.rf.29878e2efe37780fc047fa3e0af24a9e.jpg",
        "26AJCLLVBKTB_jpg.rf.8786181411b6b396915d3f07e2efdd34.jpg",
        "pexels-photo-10871621_jpeg.rf.f4eae129b2553911aabeea93a47a5fe9.jpg",
        "pexels-photo-8877378_jpeg.rf.0fc693d9744f13c54648528b60d96e4c.jpg",
        "pexels-photo-13914845_jpeg.rf.437d83997be8597a5775b04a6b426128.jpg",
    }

    def pick(paths, n):
        paths = sorted(p for p in paths if p.name not in exclude)
        rnd.shuffle(paths)
        return paths[:n]

    rows = []
    chosen = [
        ("cs", pick(list((work / "cs").rglob("*.jpg")), a.n_cs)),
        ("forklift", pick(list((work / "forklift").rglob("*.jpg")), a.n_forklift)),
        ("sh17", pick(list((work / "sh17").glob("*.jpg")), a.n_sh17)),
    ]
    idx = 0
    for key, paths in chosen:
        for p in paths:
            name = f"calib_{idx:03d}_{key}.jpg"
            shutil.copyfile(p, out / name)
            rows.append([name, SOURCES[key][0], SOURCES[key][2], p.name, SOURCES[key][1]])
            idx += 1

    # кадры демо-видео: равномерно, исключая крайние 5 % и кадры, взятые в samples
    videos = sorted((data_dir() / "videos").glob("*/*.mp4"))
    weights = {"worker-zone-detection": 0.5, "store-aisle-detection": 0.25, "people-detection": 0.25}
    counts = {v.stem: int(a.n_video * weights.get(v.stem, 1 / max(len(videos), 1))) for v in videos}
    for v in videos[: a.n_video - sum(counts.values())]:  # остаток от округления вниз
        counts[v.stem] += 1
    for v in videos:
        n = counts[v.stem]
        cap = cv2.VideoCapture(str(v))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        lo, hi = int(total * 0.05), int(total * 0.95)
        for k in range(n):
            f = lo + int((k + 0.5) * (hi - lo) / n)
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, frame = cap.read()
            if not ok:
                continue
            name = f"calib_{idx:03d}_video.jpg"
            cv2.imwrite(str(out / name), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            rows.append([name, SOURCES["video"][0], str(v.relative_to(data_dir())), f"frame {f}", SOURCES["video"][1]])
            idx += 1

    with open(out / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "theme", "source", "original", "license"])
        w.writerows(rows)
    by_theme: dict[str, int] = {}
    for r in rows:
        by_theme[r[1]] = by_theme.get(r[1], 0) + 1
    print(f"{len(rows)} кадров → {out}: {by_theme}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
