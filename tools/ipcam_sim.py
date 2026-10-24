"""Стенд-имитатор IP-камер на ноутбуке: RTSP и HLS через mediamtx, HTTP MJPEG — встроенный сервер.

Ролики data/videos публикуются по кругу (ffmpeg -re -stream_loop -1, без перекодирования):
  rtsp://127.0.0.1:8554/<имя>              RTSP (TCP)
  http://127.0.0.1:8890/<имя>/index.m3u8   HLS (сегменты режет mediamtx)
  http://127.0.0.1:8891/<имя>.mjpg         HTTP MJPEG (multipart/x-mixed-replace)
Порты свои: 8888 на ноутбуке занят туннелем к стенду, 8000 — демо. WebRTC, SRT, RTMP, MoQ, API и метрики
mediamtx выключены, UDP для RTP уведён на 8892/8893. Всё слушает только 127.0.0.1 (без --lan),
поэтому macOS не спрашивает разрешение на входящие подключения.

  python -m tools.ipcam_sim start [--videos s1_01,s2_01,s3_02,s4_02] [--mjpeg s4_02] [--stamped s1_01]
  python -m tools.ipcam_sim status [--json]
  python -m tools.ipcam_sim stop [ИМЯ]        # без имени — всё; с именем — обрыв одной публикации
  python -m tools.ipcam_sim resume ИМЯ        # вернуть публикацию
  python -m tools.ipcam_sim latency URL [--seconds 30] [--json]   # задержка по метке времени в кадре
  python -m tools.ipcam_sim install           # скачать mediamtx для этой платформы

--stamped ИМЯ — дополнительная публикация ИМЯ_ts: каждый кадр несёт в верхней строке метку времени
(60 чёрно-белых блоков), перекодирование libx264 ultrafast/zerolatency. MJPEG-сервер ставит такую же метку.
latency открывает поток источником сервиса (guard.sources) и считает «кадр сформирован → кадр у анализа».
mediamtx ищется так: GUARD_MEDIAMTX, mediamtx в PATH, <тяжёлый каталог>/bin/mediamtx-*/mediamtx
(тяжёлый каталог — родитель реального пути GUARD_DATA_DIR, у нас mosprom-heavy.noindex).
"""

from __future__ import annotations

import argparse
import contextlib
import glob
import http.server
import json
import os
import platform
import shutil
import signal
import socket
import subprocess
import sys
import tarfile
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any

CODE_ROOT = Path(__file__).resolve().parents[1]
RTSP_PORT, HLS_PORT, MJPEG_PORT, RTP_PORT, RTCP_PORT = 8554, 8890, 8891, 8892, 8893
FORBIDDEN_PORTS = {8000, 8888}
MEDIAMTX_VERSION = "v1.21.2"
DEFAULT_VIDEOS = ("s1_01", "s2_01", "s3_02", "s4_02")

# ---------------------------------------------------------------- метка времени в кадре

STAMP_MARK = (1, 0, 1, 0)
STAMP_BITS = 48  # epoch, миллисекунды
STAMP_PARITY_BITS = 8


def _stamp_block(width: int) -> int:
    return max(6, min(16, width // 64))


def stamp_frame(frame: Any, ms: int) -> None:
    """Метка времени в верхней строке кадра: маркер, 48 бит epoch-мс, 8 бит контрольной суммы."""
    bits = [(ms >> (STAMP_BITS - 1 - i)) & 1 for i in range(STAMP_BITS)]
    parity = sum((ms >> (8 * k)) & 0xFF for k in range(6)) & 0xFF
    pattern = [*STAMP_MARK, *bits, *[(parity >> (7 - i)) & 1 for i in range(STAMP_PARITY_BITS)]]
    b = _stamp_block(frame.shape[1])
    for i, bit in enumerate(pattern):
        frame[0:b, i * b : (i + 1) * b] = 255 if bit else 0


def read_stamp(frame: Any) -> int | None:
    """epoch-мс из метки или None (метки нет, кадр повреждён)."""
    b = _stamp_block(frame.shape[1])
    n = len(STAMP_MARK) + STAMP_BITS + STAMP_PARITY_BITS
    if frame.shape[0] < b or frame.shape[1] < n * b:
        return None
    y = b // 2
    row = frame[y, : n * b]
    gray = row.mean(axis=1) if row.ndim == 2 else row
    bits = [1 if gray[i * b + b // 2] > 127 else 0 for i in range(n)]
    if tuple(bits[: len(STAMP_MARK)]) != STAMP_MARK:
        return None
    ms = 0
    for bit in bits[len(STAMP_MARK) : len(STAMP_MARK) + STAMP_BITS]:
        ms = (ms << 1) | bit
    parity = 0
    for bit in bits[len(STAMP_MARK) + STAMP_BITS :]:
        parity = (parity << 1) | bit
    if parity != sum((ms >> (8 * k)) & 0xFF for k in range(6)) & 0xFF:
        return None
    return ms


# ---------------------------------------------------------------- пути и бинарники


def data_dir() -> Path:
    return Path(os.environ.get("GUARD_DATA_DIR") or CODE_ROOT / "data").expanduser()


def run_dir() -> Path:
    return data_dir() / "ipcam_sim"


def heavy_dir() -> Path:
    return data_dir().resolve().parent


def find_mediamtx() -> str | None:
    exe = os.environ.get("GUARD_MEDIAMTX") or shutil.which("mediamtx")
    if exe:
        return exe
    found = sorted(glob.glob(str(heavy_dir() / "bin" / "mediamtx-*" / "mediamtx")))
    return found[-1] if found else None


def find_ffmpeg() -> str:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001
        exe = shutil.which("ffmpeg")
        if not exe:
            raise SystemExit("нужен ffmpeg: pip install imageio-ffmpeg или brew install ffmpeg") from None
        return exe


def video_path(name: str) -> Path:
    p = Path(name)
    if p.suffix and p.is_file():
        return p
    cand = data_dir() / "videos" / f"{name}.mp4"
    if not cand.is_file():
        raise SystemExit(f"нет ролика {cand}")
    return cand


def urls(name: str, bind: str = "127.0.0.1") -> dict[str, str]:
    return {
        "rtsp": f"rtsp://{bind}:{RTSP_PORT}/{name}",
        "hls": f"http://{bind}:{HLS_PORT}/{name}/index.m3u8",
        "mjpeg": f"http://{bind}:{MJPEG_PORT}/{name}.mjpg",
    }


def mediamtx_config(bind: str, rtsp_port: int = RTSP_PORT, hls_port: int = HLS_PORT, rtp_port: int = RTP_PORT, rtcp_port: int = RTCP_PORT) -> str:
    bad = FORBIDDEN_PORTS & {rtsp_port, hls_port, rtp_port, rtcp_port}
    if bad:
        raise ValueError(f"порты {sorted(bad)} заняты демо или туннелем к стенду")
    return f"""# создано tools/ipcam_sim.py; порты свои: 8888 — туннель к стенду, 8000 — демо
logLevel: info
logDestinations: [stdout]
api: false
metrics: false
pprof: false
playback: false
rtsp: true
rtspTransports: [tcp]
rtspAddress: {bind}:{rtsp_port}
rtpAddress: {bind}:{rtp_port}
rtcpAddress: {bind}:{rtcp_port}
rtmp: false
hls: true
hlsAddress: {bind}:{hls_port}
hlsAlwaysRemux: true
hlsVariant: mpegts
hlsSegmentCount: 7
hlsSegmentDuration: 1s
webrtc: false
srt: false
moq: false
paths:
  all_others:
"""


# ---------------------------------------------------------------- процессы


def _state_path() -> Path:
    return run_dir() / "state.json"


def load_state() -> dict[str, Any]:
    p = _state_path()
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {"procs": {}}


def save_state(state: dict[str, Any]) -> None:
    run_dir().mkdir(parents=True, exist_ok=True)
    _state_path().write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def _alive(entry: dict[str, Any]) -> bool:
    pid = entry.get("pid")
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    out = subprocess.run(["ps", "-p", str(pid), "-o", "command="], capture_output=True, text=True).stdout
    return entry.get("match", "") in out  # pid мог достаться чужому процессу


def spawn(key: str, cmd: list[str], match: str, state: dict[str, Any]) -> None:
    log = run_dir() / f"{key.replace(':', '_')}.log"
    # mediamtx создаёт в рабочем каталоге auto.crt/auto.key — запускаем его в каталоге имитатора
    cwd = run_dir() if key == "mediamtx" else CODE_ROOT
    with open(log, "ab") as fh:
        proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True, cwd=cwd)
    state["procs"][key] = {"pid": proc.pid, "cmd": cmd, "match": match, "log": str(log), "started": time.time()}


def terminate(entry: dict[str, Any], timeout: float = 5.0) -> bool:
    if not _alive(entry):
        return False
    pid = entry["pid"]
    with contextlib.suppress(OSError):
        os.killpg(pid, signal.SIGTERM)
    deadline = time.time() + timeout
    while time.time() < deadline and _alive(entry):
        time.sleep(0.1)
    if _alive(entry):
        with contextlib.suppress(OSError):
            os.killpg(pid, signal.SIGKILL)
    return True


def port_open(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def publisher_cmd(ffmpeg: str, src: Path, url: str) -> list[str]:
    # без перекодирования: H.264 из mp4 как есть, только видео; -re — темп реального времени
    return [ffmpeg, "-hide_banner", "-loglevel", "warning", "-nostdin", "-re", "-stream_loop", "-1", "-i", str(src),
            "-map", "0:v:0", "-c:v", "copy", "-an", "-f", "rtsp", "-rtsp_transport", "tcp", url]


def cmd_start(args: argparse.Namespace) -> int:
    state = load_state()
    if any(_alive(e) for e in state.get("procs", {}).values()):
        print("имитатор уже запущен: python -m tools.ipcam_sim status / stop", file=sys.stderr)
        return 1
    mediamtx = find_mediamtx()
    if not mediamtx:
        print("mediamtx не найден: python -m tools.ipcam_sim install или GUARD_MEDIAMTX", file=sys.stderr)
        return 1
    bind = "0.0.0.0" if args.lan else "127.0.0.1"
    ffmpeg = find_ffmpeg()
    videos = [v for v in args.videos.split(",") if v]
    mjpeg = [v for v in (args.mjpeg or "").split(",") if v]
    stamped = [v for v in (args.stamped or "").split(",") if v]
    for name in videos + mjpeg + stamped:
        video_path(name)
    run_dir().mkdir(parents=True, exist_ok=True)
    conf = run_dir() / "mediamtx.yml"
    conf.write_text(mediamtx_config(bind), encoding="utf-8")
    state = {"procs": {}, "bind": bind, "started": time.time(), "videos": videos, "mjpeg": mjpeg, "stamped": stamped}
    spawn("mediamtx", [mediamtx, str(conf)], "mediamtx", state)
    for _ in range(50):
        if port_open(RTSP_PORT):
            break
        time.sleep(0.1)
    else:
        save_state(state)
        print(f"mediamtx не открыл порт {RTSP_PORT}, см. {state['procs']['mediamtx']['log']}", file=sys.stderr)
        return 1
    for name in videos:
        spawn(f"pub:{name}", publisher_cmd(ffmpeg, video_path(name), f"rtsp://127.0.0.1:{RTSP_PORT}/{name}"), "ffmpeg", state)
    for name in stamped:
        cmd = [sys.executable, "-m", "tools.ipcam_sim", "publish-stamped", "--video", str(video_path(name)),
               "--url", f"rtsp://127.0.0.1:{RTSP_PORT}/{name}_ts"]
        spawn(f"pub:{name}_ts", cmd, "ipcam_sim", state)
    if mjpeg:
        cmd = [sys.executable, "-m", "tools.ipcam_sim", "mjpeg-serve", "--bind", bind, "--port", str(MJPEG_PORT),
               "--fps", str(args.mjpeg_fps), *[f"{n}={video_path(n)}" for n in mjpeg]]
        spawn("mjpeg", cmd, "ipcam_sim", state)
    save_state(state)
    time.sleep(1.5)
    return cmd_status(argparse.Namespace(json=False))


def cmd_stop(args: argparse.Namespace) -> int:
    state = load_state()
    procs = state.get("procs", {})
    if args.name:
        keys = [k for k in procs if k in (f"pub:{args.name}", f"pub:{args.name}_ts")]
        if not keys:
            print(f"нет публикации {args.name}; есть: {', '.join(k[4:] for k in procs if k.startswith('pub:'))}", file=sys.stderr)
            return 1
        for k in keys:
            terminate(procs[k])
            procs[k]["stopped"] = time.time()
            print(f"остановлена публикация {k[4:]} ({time.strftime('%H:%M:%S')})")
        save_state(state)
        return 0
    for key in sorted(procs, key=lambda k: k == "mediamtx"):  # mediamtx последним
        if terminate(procs[key]):
            print(f"остановлен {key}")
    save_state({"procs": {}})
    return 0


def cmd_resume(args: argparse.Namespace) -> int:
    state = load_state()
    procs = state.get("procs", {})
    keys = [k for k in procs if k in (f"pub:{args.name}", f"pub:{args.name}_ts")]
    if not keys:
        print(f"нет публикации {args.name}", file=sys.stderr)
        return 1
    for k in keys:
        entry = procs[k]
        if _alive(entry):
            print(f"{k[4:]} уже публикуется")
            continue
        spawn(k, entry["cmd"], entry["match"], state)
        print(f"публикация {k[4:]} возобновлена ({time.strftime('%H:%M:%S')})")
    save_state(state)
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    state = load_state()
    procs = state.get("procs", {})
    bind = "127.0.0.1"
    out: dict[str, Any] = {
        "ports": {"rtsp": port_open(RTSP_PORT), "hls": port_open(HLS_PORT), "mjpeg": port_open(MJPEG_PORT)},
        "procs": {k: {"pid": e.get("pid"), "alive": _alive(e), "log": e.get("log")} for k, e in procs.items()},
        "streams": {},
    }
    for name in state.get("videos", []) + [f"{n}_ts" for n in state.get("stamped", [])]:
        u = urls(name, bind)
        out["streams"][name] = {"rtsp": u["rtsp"], "hls": u["hls"], "publishing": out["procs"].get(f"pub:{name}", {}).get("alive", False)}
    for name in state.get("mjpeg", []):
        out["streams"].setdefault(name, {})["mjpeg"] = urls(name, bind)["mjpeg"]
    if args.json:
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 0
    print("порты:", ", ".join(f"{k} {'открыт' if v else 'закрыт'}" for k, v in out["ports"].items()))
    for k, v in out["procs"].items():
        print(f"  {k:<16} pid {v['pid']:<7} {'работает' if v['alive'] else 'остановлен'}")
    for name, s in out["streams"].items():
        print(f"  {name}: " + "  ".join(f"{k}={v}" for k, v in s.items() if k != "publishing")
              + ("" if s.get("publishing", True) else "  [публикация остановлена]"))
    return 0


# ---------------------------------------------------------------- MJPEG-сервер


class MjpegFeed(threading.Thread):
    """Читает ролик по кругу в реальном темпе и держит последний JPEG (с меткой времени)."""

    def __init__(self, name: str, path: Path, fps: float) -> None:
        super().__init__(daemon=True, name=f"mjpeg-{name}")
        self.name_, self.path, self.fps = name, path, fps
        self.cond = threading.Condition()
        self.jpeg: bytes | None = None
        self.seq = 0

    def run(self) -> None:
        import cv2

        while True:
            cap = cv2.VideoCapture(str(self.path))
            native = cap.get(cv2.CAP_PROP_FPS) or 25.0
            step = max(1, round(native / self.fps))
            t0, idx = time.monotonic(), 0
            while True:
                due = t0 + idx / native
                delay = due - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                if idx % step:
                    if not cap.grab():
                        break
                    idx += 1
                    continue
                ok, frame = cap.read()
                if not ok:
                    break
                idx += 1
                stamp_frame(frame, int(time.time() * 1000))
                data = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()
                with self.cond:
                    self.jpeg, self.seq = data, self.seq + 1
                    self.cond.notify_all()
            cap.release()


def make_mjpeg_handler(feeds: dict[str, MjpegFeed]) -> type[http.server.BaseHTTPRequestHandler]:
    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:
            pass

        def do_GET(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0].strip("/")
            name, _, ext = path.rpartition(".")
            feed = feeds.get(name)
            if feed is None or ext not in ("mjpg", "jpg"):
                body = ("MJPEG: " + ", ".join(f"/{n}.mjpg" for n in feeds) + "\n").encode()
                self.send_response(404 if path else 200)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if ext == "jpg":
                with feed.cond:
                    feed.cond.wait_for(lambda: feed.jpeg is not None, timeout=5)
                    data = feed.jpeg or b""
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            seq = -1
            try:
                while True:
                    with feed.cond:
                        feed.cond.wait_for(lambda: feed.seq != seq, timeout=5)
                        seq, data = feed.seq, feed.jpeg
                    if data is None:
                        continue
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(data)).encode() + b"\r\n\r\n" + data + b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                return

    return Handler


def cmd_mjpeg_serve(args: argparse.Namespace) -> int:
    if args.port in FORBIDDEN_PORTS:
        raise SystemExit(f"порт {args.port} занят демо или туннелем")
    feeds = {}
    for item in args.items:
        name, _, path = item.partition("=")
        feeds[name] = MjpegFeed(name, Path(path), args.fps)
        feeds[name].start()
    server = http.server.ThreadingHTTPServer((args.bind, args.port), make_mjpeg_handler(feeds))
    server.daemon_threads = True
    signal.signal(signal.SIGTERM, lambda *a: threading.Thread(target=server.shutdown, daemon=True).start())
    print(f"MJPEG http://{args.bind}:{args.port}/ : {', '.join(feeds)}", flush=True)
    server.serve_forever()
    return 0


# ---------------------------------------------------------------- публикация с меткой времени


def cmd_publish_stamped(args: argparse.Namespace) -> int:
    """Ролик по кругу → метка времени в каждом кадре → libx264 zerolatency → RTSP."""
    import cv2

    ffmpeg = find_ffmpeg()
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "warning", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
           "-r", f"{fps:g}", "-i", "-", "-an", "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency",
           "-g", str(int(fps * 2)), "-pix_fmt", "yuv420p", "-b:v", "2500k", "-f", "rtsp", "-rtsp_transport", "tcp", args.url]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    signal.signal(signal.SIGTERM, lambda *a: (proc.terminate(), sys.exit(0)))
    t0, idx = time.monotonic(), 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                cap.release()
                cap = cv2.VideoCapture(args.video)
                continue
            delay = t0 + idx / fps - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            idx += 1
            stamp_frame(frame, int(time.time() * 1000))
            proc.stdin.write(frame.tobytes())
    except (BrokenPipeError, KeyboardInterrupt):
        return 1
    finally:
        with contextlib.suppress(Exception):
            proc.stdin.close()
        proc.terminate()


# ---------------------------------------------------------------- задержка


def cmd_latency(args: argparse.Namespace) -> int:
    """Источник сервиса (LiveSource) на URL; задержка = момент выдачи кадра анализу − метка в кадре."""
    import numpy as np

    from guard.contracts import SourceConfig
    from guard.sources import make_source

    lat: list[float] = []
    bad = {"n": 0}

    def on_frame(pkt: Any) -> None:
        ms = read_stamp(pkt.frame)
        if ms is None:
            bad["n"] += 1
            return
        lat.append(time.time() * 1000.0 - ms)

    src = make_source(SourceConfig(type="rtsp", uri=args.url), args.url, "latency", on_frame=on_frame)
    t0 = time.monotonic()
    src.start()
    first = None
    while time.monotonic() - t0 < args.seconds + 60:
        if first is None and lat:
            first = time.monotonic() - t0
        if first is not None and time.monotonic() - t0 >= first + args.seconds:
            break
        time.sleep(0.2)
    status = src.status()
    src.stop()
    arr = np.array(lat) if lat else np.array([np.nan])
    res = {
        "url": args.url,
        "stream": status.get("stream"),
        "connect_ms": status.get("connect_ms"),
        "seconds": args.seconds,
        "frames": len(lat),
        "frames_without_stamp": bad["n"],
        "fps": round(len(lat) / args.seconds, 2),
        "latency_ms": {"p50": round(float(np.percentile(arr, 50)), 1), "p95": round(float(np.percentile(arr, 95)), 1),
                       "min": round(float(np.min(arr)), 1), "max": round(float(np.max(arr)), 1)},
        "state": status.get("state"),
        "last_error": status.get("last_error"),
    }
    print(json.dumps(res, ensure_ascii=False, indent=None if args.json else 2))
    return 0 if lat else 1


# ---------------------------------------------------------------- установка mediamtx


def cmd_install(args: argparse.Namespace) -> int:
    machine = platform.machine().lower()
    arch = {"arm64": "arm64", "aarch64": "arm64", "x86_64": "amd64", "amd64": "amd64"}.get(machine, machine)
    osname = {"darwin": "darwin", "linux": "linux"}.get(sys.platform.rstrip("0123456789"), sys.platform)
    asset = f"mediamtx_{args.version}_{osname}_{arch}.tar.gz"
    url = f"https://github.com/bluenviron/mediamtx/releases/download/{args.version}/{asset}"
    dest = Path(args.dest) if args.dest else heavy_dir() / "bin" / f"mediamtx-{args.version}"
    dest.mkdir(parents=True, exist_ok=True)
    archive = dest / asset
    print(f"скачиваю {url} → {dest}")
    urllib.request.urlretrieve(url, archive)  # noqa: S310 — фиксированный адрес релиза GitHub
    with tarfile.open(archive) as tar:
        member = tar.getmember("mediamtx")
        tar.extract(member, dest, filter="data")
    (dest / "mediamtx").chmod(0o755)
    print(dest / "mediamtx")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m tools.ipcam_sim", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("start", help="запустить mediamtx, публикации и MJPEG")
    p.add_argument("--videos", default=",".join(DEFAULT_VIDEOS), help="ролики data/videos для RTSP и HLS (через запятую)")
    p.add_argument("--mjpeg", default="", help="ролики для HTTP MJPEG")
    p.add_argument("--mjpeg-fps", type=float, default=15.0)
    p.add_argument("--stamped", default="", help="ролики для публикации ИМЯ_ts с меткой времени в кадре")
    p.add_argument("--lan", action="store_true", help="слушать все интерфейсы (телефон, другие машины); macOS спросит разрешение")
    p.set_defaults(func=cmd_start)
    p = sub.add_parser("stop", help="остановить всё или одну публикацию")
    p.add_argument("name", nargs="?")
    p.set_defaults(func=cmd_stop)
    p = sub.add_parser("resume", help="вернуть остановленную публикацию")
    p.add_argument("name")
    p.set_defaults(func=cmd_resume)
    p = sub.add_parser("status")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_status)
    p = sub.add_parser("latency", help="задержка по метке времени в кадре")
    p.add_argument("url")
    p.add_argument("--seconds", type=float, default=30.0)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_latency)
    p = sub.add_parser("install", help="скачать mediamtx с GitHub")
    p.add_argument("--version", default=MEDIAMTX_VERSION)
    p.add_argument("--dest")
    p.set_defaults(func=cmd_install)
    p = sub.add_parser("mjpeg-serve", help=argparse.SUPPRESS)
    p.add_argument("--bind", default="127.0.0.1")
    p.add_argument("--port", type=int, default=MJPEG_PORT)
    p.add_argument("--fps", type=float, default=15.0)
    p.add_argument("items", nargs="+", help="имя=путь")
    p.set_defaults(func=cmd_mjpeg_serve)
    p = sub.add_parser("publish-stamped", help=argparse.SUPPRESS)
    p.add_argument("--video", required=True)
    p.add_argument("--url", required=True)
    p.set_defaults(func=cmd_publish_stamped)
    args = ap.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
