"""Живой RTSP и HLS через mediamtx: подключение, обрыв публикации, переподключение без перезапуска.

Пропускается без mediamtx (GUARD_MEDIAMTX, PATH или <тяжёлый каталог>/bin/mediamtx-*/mediamtx) и ffmpeg.
Порты берутся свободные, поэтому тест не мешает запущенному имитатору (tools/ipcam_sim.py).
"""

from __future__ import annotations

import socket
import subprocess
import time

import pytest

from guard.contracts import SourceConfig
from guard.sources.base import ONLINE, RECONNECTING
from guard.sources.probe import probe_source
from guard.sources.video import LiveSource
from tools import ipcam_sim

pytestmark = pytest.mark.integration

MEDIAMTX = ipcam_sim.find_mediamtx()
try:
    FFMPEG = ipcam_sim.find_ffmpeg()
except SystemExit:
    FFMPEG = None


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait(predicate, timeout: float, step: float = 0.1) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(step)
    return predicate()


@pytest.fixture()
def server(tmp_path):
    if not MEDIAMTX or not FFMPEG:
        pytest.skip("нужны mediamtx и ffmpeg")
    ports = {"rtsp_port": _free_port(), "hls_port": _free_port(), "rtp_port": _free_port(), "rtcp_port": _free_port()}
    conf = tmp_path / "mediamtx.yml"
    conf.write_text(ipcam_sim.mediamtx_config("127.0.0.1", **ports), encoding="utf-8")
    log = open(tmp_path / "mediamtx.log", "wb")  # noqa: SIM115
    proc = subprocess.Popen([MEDIAMTX, str(conf)], cwd=tmp_path, stdout=log, stderr=subprocess.STDOUT)
    assert _wait(lambda: ipcam_sim.port_open(ports["rtsp_port"]), 10), "mediamtx не открыл порт"
    publishers: list[subprocess.Popen] = []

    def publish(name: str) -> subprocess.Popen:
        cmd = [FFMPEG, "-hide_banner", "-loglevel", "error", "-nostdin", "-re", "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=25",
               "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency", "-g", "25", "-pix_fmt", "yuv420p",
               "-f", "rtsp", "-rtsp_transport", "tcp", f"rtsp://127.0.0.1:{ports['rtsp_port']}/{name}"]
        p = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        publishers.append(p)
        return p

    yield ports, publish
    for p in publishers:
        p.kill()
        p.wait(5)
    proc.terminate()
    proc.wait(10)
    log.close()


def test_probe_real_rtsp_and_missing_path(server):
    ports, publish = server
    publish("cam")
    url = f"rtsp://127.0.0.1:{ports['rtsp_port']}/cam"
    assert _wait(lambda: probe_source(SourceConfig(type="rtsp", uri=url), timeout_s=5)["ok"], 20, step=0.5)
    res = probe_source(SourceConfig(type="rtsp", uri=url), timeout_s=10)
    assert res["ok"] and (res["width"], res["height"]) == (320, 240)
    assert res["codec"] == "h264" and res["fps"] == pytest.approx(25, abs=2)
    assert res["preview_jpeg_b64"]
    missing = probe_source(SourceConfig(type="rtsp", uri=f"rtsp://127.0.0.1:{ports['rtsp_port']}/nope"), timeout_s=5)
    assert not missing["ok"] and "404" in missing["error"]


def test_live_source_reconnects_after_publisher_restart(server):
    ports, publish = server
    pub = publish("cam")
    url = f"rtsp://127.0.0.1:{ports['rtsp_port']}/cam"
    src = LiveSource(SourceConfig(type="rtsp", uri=url), url, "it-cam")
    src.start()
    try:
        assert _wait(lambda: src.state == ONLINE and src.fps_in() > 10, 30), src.status()
        pub.kill()
        pub.wait(5)
        assert _wait(lambda: src.state != ONLINE, 15), src.status()
        assert src.state in (RECONNECTING, "offline")
        assert src.drops == 1
        t_resume = time.monotonic()
        publish("cam")
        assert _wait(lambda: src.state == ONLINE, 30), src.status()
        recovered_s = time.monotonic() - t_resume
        assert src.reconnects == 1 and src.last_outage_s is not None
        assert recovered_s < 15  # пауза не больше 10 с плюс подключение
    finally:
        src.stop()


def test_live_hls_from_mediamtx(server):
    ports, publish = server
    publish("hlscam")
    url = f"http://127.0.0.1:{ports['hls_port']}/hlscam/index.m3u8"
    src = LiveSource(SourceConfig(type="rtsp", uri=url), url, "it-hls")
    assert src.paced and src.stream_kind == "hls"
    src.start()
    try:
        assert _wait(lambda: src.state == ONLINE and src.fps_in() > 15, 45), src.status()
        assert src.status()["width"] == 320
    finally:
        src.stop()
