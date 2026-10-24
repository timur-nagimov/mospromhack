#!/usr/bin/env python3
"""Воспроизводимая загрузка демо- и тестовых данных по манифесту tools/data_manifest.json.

Что делает (по шагам, каждый можно включить отдельно):
  download — скачивает исходники в <data>/raw/ (кэш, повторно не качает, сверяет sha256);
  encode   — режет и перекодирует в <data>/videos/<name>: H.264, ≤1280×720, 25 fps, CRF 28, без звука,
             +faststart. Ролики из нескольких отрезков склеиваются в один файл (кадр 1280×720 с полями);
  labels   — пишет разметку <data>/labels/<stem>.json (формат ml/eval/events_eval.py) и <data>/durations.json;
  frames   — стоп-кадры для режима «наборы изображений»: <data>/images/<scenario>/;
  checks   — 4–6 контрольных кадров на ролик в <data>/media/data_checks/ (проверка глазами).

ffmpeg берётся из imageio-ffmpeg и запускается строго по одному файлу через tools/heavy.sh
(очередь тяжёлых задач и проверка свободной памяти), если он есть.

Примеры:
  python tools/fetch_data.py                       # всё: download encode labels frames checks
  python tools/fetch_data.py --only s1_01 --steps download encode labels
  python tools/fetch_data.py --data-dir /path/to/data --heavy /path/to/tools/heavy.sh
  python tools/fetch_data.py --list
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = Path(__file__).resolve().parent / "data_manifest.json"
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
STEPS = ("download", "encode", "labels", "frames", "checks", "images")
EVENT_KEYS = ("type", "t_start", "t_end", "uncertain", "note")  # ровно то, что принимает LabelEvent


def log(msg: str) -> None:
    print(msg, flush=True)


def ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001
        exe = shutil.which("ffmpeg")
        if not exe:
            sys.exit("ffmpeg не найден: поставьте imageio-ffmpeg или ffmpeg")
        return exe


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def raw_path(data: Path, sid: str, src: dict) -> Path:
    ext = Path(src["url"].split("?")[0]).suffix or ".mp4"
    return data / "raw" / f"{sid}{ext}"


YTDLP_HOSTS = ("youtube.com", "youtu.be", "vimeo.com")
# Видео без звука; avc1 — чтобы OpenCV и ffmpeg из imageio читали без av1-декодера
YTDLP_FORMAT = "bv*[height<=720][vcodec^=avc1]/bv*[height<=720][ext=mp4]/b[height<=720]/b"


def is_ytdlp(src: dict) -> bool:
    host = urllib.parse.urlparse(src["url"]).netloc
    return bool(src.get("ytdlp")) or any(host == h or host.endswith("." + h) for h in YTDLP_HOSTS)


def download_ytdlp(out: Path, src: dict) -> None:
    exe = shutil.which("yt-dlp")
    if not exe:
        sys.exit("yt-dlp не найден: brew install yt-dlp")
    cmd = [exe, "-q", "--no-warnings", "--no-playlist", "--ffmpeg-location", ffmpeg_exe(),
           "-f", src.get("format", YTDLP_FORMAT), "--remux-video", "mp4",
           "-o", str(out.with_suffix("")) + ".%(ext)s", src["url"]]
    subprocess.run(cmd, check=True, timeout=900)
    if not out.exists():
        sys.exit(f"yt-dlp не создал {out}")


def download(data: Path, sid: str, src: dict, retries: int = 3) -> Path:
    out = raw_path(data, sid, src)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and out.stat().st_size > 0:
        status = "кэш"
    elif is_ytdlp(src):
        t0 = time.time()
        download_ytdlp(out, src)
        status = f"скачан yt-dlp за {time.time() - t0:.0f} с"
    else:
        for attempt in range(1, retries + 1):
            try:
                t0 = time.time()
                req = urllib.request.Request(src["url"], headers={"User-Agent": UA})
                tmp = out.with_suffix(out.suffix + ".part")
                with urllib.request.urlopen(req, timeout=120) as r, tmp.open("wb") as f:
                    shutil.copyfileobj(r, f, 1 << 20)
                tmp.replace(out)
                status = f"скачан за {time.time() - t0:.0f} с"
                break
            except Exception as e:  # noqa: BLE001
                log(f"  ! {sid}: попытка {attempt}: {e}")
                if attempt == retries:
                    raise
                time.sleep(3 * attempt)
    digest = sha256(out)
    want = src.get("sha256")
    if want and want != digest:
        log(f"  ! {sid}: sha256 не совпадает с манифестом ({digest[:12]} != {want[:12]}) — источник изменился?")
    log(f"  {sid}: {out.name} {out.stat().st_size / 1e6:.1f} МБ, {status}, sha256 {digest[:12]}")
    return out


def encode(data: Path, video: dict, manifest: dict, heavy: str | None, force: bool) -> Path:
    enc = {**manifest["encode"], **video.get("encode", {})}
    out = data / "videos" / video["name"]
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and not force:
        log(f"  {video['name']}: уже есть (--force для перекодирования)")
        return out
    w, h, fps = enc["width"], enc["height"], enc["fps"]
    segs = video["segments"]
    cmd = [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y"]
    for seg in segs:
        src = manifest["sources"][seg["source"]]
        if "ss" in seg:
            cmd += ["-ss", str(seg["ss"])]
        if "to" in seg:
            cmd += ["-to", str(seg["to"])]
        cmd += ["-i", str(raw_path(data, seg["source"], src))]
    if len(segs) == 1:
        # один отрезок: сохраняем пропорции, не увеличиваем, чётные стороны
        vf = (f"fps={fps},scale='min({w},iw)':'min({h},ih)':force_original_aspect_ratio=decrease,"
              "scale=trunc(iw/2)*2:trunc(ih/2)*2,setsar=1")
        cmd += ["-vf", vf, "-map", "0:v:0"]
    else:
        # склейка: каждый отрезок вписывается в w×h с полями, затем concat
        parts = [f"[{i}:v:0]fps={fps},scale={w}:{h}:force_original_aspect_ratio=decrease,"
                 f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,setsar=1[v{i}]" for i in range(len(segs))]
        chain = "".join(f"[v{i}]" for i in range(len(segs)))
        cmd += ["-filter_complex", ";".join(parts) + f";{chain}concat=n={len(segs)}:v=1:a=0[out]",
                "-map", "[out]"]
    cmd += ["-c:v", "libx264", "-preset", enc["preset"], "-crf", str(enc["crf"]), "-pix_fmt", "yuv420p",
            "-g", str(fps * 2), "-threads", str(enc.get("threads", 4)), "-an", "-movflags", "+faststart",
            str(out.with_suffix(".tmp.mp4"))]
    if heavy:
        cmd = [heavy, *cmd]
    t0 = time.time()
    subprocess.run(cmd, check=True)
    out.with_suffix(".tmp.mp4").replace(out)
    info = probe(out)
    log(f"  {video['name']}: {info['width']}×{info['height']} {info['duration']:.1f} с "
        f"{out.stat().st_size / 1e6:.1f} МБ за {time.time() - t0:.0f} с")
    return out


def probe(path: Path) -> dict:
    import cv2

    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    info = {"width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": fps, "frames": n, "duration": n / fps if fps else 0.0}
    cap.release()
    return info


def write_labels(data: Path, videos: list[dict], manifest: dict, labels: Path | None = None) -> None:
    # Отдельный каталог для отложенных выборок: run_offline требует, чтобы сценарии покрывали все *.json каталога
    labels = labels or data / "labels"
    labels.mkdir(parents=True, exist_ok=True)
    durations_path = data / "durations.json"
    durations = json.loads(durations_path.read_text()) if durations_path.exists() else {}
    for v in videos:
        path = data / "videos" / v["name"]
        events = [{k: e[k] for k in EVENT_KEYS if k in e} for e in v.get("events", [])]
        for e in events:
            e.setdefault("uncertain", False)
            e.setdefault("note", "")
        doc = {
            "video": v["name"],
            "events": events,
            # всё ниже — справочно, events_eval.py это игнорирует
            "scenario": v["scenario"],
            "description": v.get("description", ""),
            "rules_to_enable": v.get("rules_to_enable", []),
            "zones": v.get("zones", []),
            "persons": v.get("persons", ""),
            "sources": [
                {**{k: manifest["sources"][s["source"]].get(k) for k in ("page", "url", "license", "author")},
                 "ss": s.get("ss"), "to": s.get("to")}
                for s in v["segments"]
            ],
        }
        if path.exists():
            info = probe(path)
            doc["duration_s"] = round(info["duration"], 2)
            doc["size"] = [info["width"], info["height"]]
            durations[v["name"]] = round(info["duration"], 2)
        (labels / f"{Path(v['name']).stem}.json").write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n")
        log(f"  labels/{Path(v['name']).stem}.json: событий {len(events)}")
    durations_path.write_text(json.dumps(durations, ensure_ascii=False, indent=2) + "\n")
    log(f"  durations.json: {len(durations)} роликов")


def grab(path: Path, times: list[float], out_dir: Path, prefix: str, quality: int = 88) -> int:
    import cv2

    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(path))
    n = 0
    for t in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if ok:
            cv2.imwrite(str(out_dir / f"{prefix}_t{t:05.1f}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
            n += 1
    cap.release()
    return n


def frames(data: Path, videos: list[dict]) -> None:
    for v in videos:
        path = data / "videos" / v["name"]
        if not path.exists():
            continue
        dur = probe(path)["duration"]
        step = v.get("frame_step_s", 2.0)
        times = [round(t * step + 0.5, 1) for t in range(int((dur - 0.6) // step) + 1)]
        folder = data / "images" / v["scenario"].lower()
        for old in folder.glob(f"{Path(v['name']).stem}_t*.jpg"):  # повторный запуск не копит кадры
            old.unlink()
        n = grab(path, times, folder, Path(v["name"]).stem)
        log(f"  images/{folder.name}/: +{n} кадров из {v['name']}")


def checks(data: Path, videos: list[dict]) -> None:
    for v in videos:
        path = data / "videos" / v["name"]
        if not path.exists():
            continue
        dur = probe(path)["duration"]
        events = [e for e in v.get("events", []) if not e.get("uncertain")][:2]
        times = {round(dur * (k + 0.5) / (6 - len(events)), 1) for k in range(6 - len(events))}
        times |= {round((e["t_start"] + e["t_end"]) / 2, 1) for e in events}  # середины достоверных событий
        times = sorted(t for t in times if t < dur - 0.2)
        folder = data / "media" / "data_checks"
        for old in folder.glob(f"{Path(v['name']).stem}_t*.jpg"):
            old.unlink()
        n = grab(path, times, folder, Path(v["name"]).stem, 80)
        log(f"  data_checks: {n} кадров из {v['name']}")


def fetch_url(url: str, dest: Path, attempts: int = 3) -> Path:
    """Скачать маленький файл с повторами (CDN Hugging Face иногда не резолвится с первого раза)."""
    for attempt in range(1, attempts + 1):
        if dest.exists() and dest.stat().st_size > 0:
            break
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=60) as r:
                dest.write_bytes(r.read())
        except Exception as e:  # noqa: BLE001
            log(f"  ! {dest.name}: попытка {attempt}: {e}")
            if attempt == attempts:
                raise
            time.sleep(5 * attempt)
    return dest


def image_sets(data: Path, sets: list[dict]) -> None:
    """Наборы картинок из открытых датасетов (формат YOLO):
    <data>/images/<name>/*.jpg и рамки <data>/image_labels/<name>.json."""
    import cv2

    for s in sets:
        raw = data / "raw" / "image_sets" / s["name"]
        out = data / "images" / s["name"]
        raw.mkdir(parents=True, exist_ok=True)
        out.mkdir(parents=True, exist_ok=True)
        labels = {}
        for rel in s["files"]:
            img = fetch_url(s["base_url"] + rel, raw / Path(rel).name)
            lab_rel = rel.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
            lab = fetch_url(s["base_url"] + lab_rel, raw / Path(lab_rel).name)
            short = Path(rel).stem.split("_jpg")[0] + ".jpg"  # ppe_0123_jpg.rf.<hash> -> ppe_0123.jpg
            name = short if short not in labels else Path(rel).stem + ".jpg"
            cv2.imwrite(str(out / name), cv2.imread(str(img)), [cv2.IMWRITE_JPEG_QUALITY, 92])
            boxes: dict[str, list] = {c: [] for c in s["classes"]}
            for line in lab.read_text().splitlines():
                if line.strip():
                    c, x, y, w, h = line.split()[:5]
                    x, y, w, h = map(float, (x, y, w, h))
                    boxes[s["classes"][int(c)]].append([round(x - w / 2, 4), round(y - h / 2, 4), round(w, 4), round(h, 4)])
            labels[name] = boxes
        (data / "image_labels").mkdir(exist_ok=True)
        doc = {"set": s["name"], "scenario": s["scenario"], "license": s["license"], "source": s["page"],
               "origin": s.get("origin", ""), "bbox_format": "x, y, w, h — нормированные, левый верхний угол",
               "images": labels}
        (data / "image_labels" / f"{s['name']}.json").write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n")
        counts = {c: sum(len(v[c]) for v in labels.values()) for c in s["classes"]}
        log(f"  images/{s['name']}/: {len(labels)} картинок, рамки: {counts}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=MANIFEST)
    ap.add_argument("--data-dir", type=Path, default=Path(os.environ.get("GUARD_DATA_DIR", ROOT / "data")))
    ap.add_argument("--labels-dir", type=Path, default=None, help="куда писать разметку (по умолчанию <data>/labels)")
    ap.add_argument("--only", nargs="*", help="имена роликов без .mp4, например s1_01 clean_02")
    ap.add_argument("--steps", nargs="*", choices=STEPS, default=list(STEPS))
    ap.add_argument("--heavy", default=os.environ.get("GUARD_HEAVY_SH", str(ROOT / "tools" / "heavy.sh")),
                    help="обёртка очереди тяжёлых задач; пустая строка — без неё")
    ap.add_argument("--force", action="store_true", help="перекодировать, даже если файл есть")
    ap.add_argument("--list", action="store_true", help="показать ролики манифеста и выйти")
    args = ap.parse_args(argv)

    manifest = json.loads(args.manifest.read_text())
    videos = manifest["videos"]
    if args.only:
        wanted = {Path(x).stem for x in args.only}
        videos = [v for v in videos if Path(v["name"]).stem in wanted]
        if not videos:
            sys.exit(f"нет таких роликов: {sorted(wanted)}")
    if args.list:
        for v in manifest["videos"]:
            print(f"{v['name']:16} {v['scenario']:6} {len(v.get('events', [])):2} соб.  {v.get('title', '')}")
        return 0
    heavy = args.heavy if args.heavy and Path(args.heavy).is_file() else None
    data = args.data_dir
    log(f"data: {data}; шаги: {', '.join(args.steps)}; heavy: {heavy or 'нет'}")

    if "download" in args.steps:
        log("== download")
        for sid in sorted({s["source"] for v in videos for s in v["segments"]}):
            download(data, sid, manifest["sources"][sid])
    if "encode" in args.steps:
        log("== encode (по одному файлу)")
        for v in videos:
            encode(data, v, manifest, heavy, args.force)
    if "labels" in args.steps:
        log("== labels")
        write_labels(data, videos, manifest, args.labels_dir)
    if "frames" in args.steps:
        log("== frames")
        frames(data, videos)
    if "checks" in args.steps:
        log("== checks")
        checks(data, videos)
    if "images" in args.steps and not args.only:
        log("== images (наборы картинок из датасетов)")
        image_sets(data, manifest.get("image_sets", []))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
