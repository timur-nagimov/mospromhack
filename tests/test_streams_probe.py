"""POST /api/cameras/probe и диагностика сбоев подключения: фейковые захваты и локальные RTSP/HTTP-серверы."""

from __future__ import annotations

import base64
import hashlib
import http.server
import socket
import threading
import time

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from guard.api.app import create_app
from guard.contracts import SourceConfig
from guard.pipeline.engine import PipelineEngine
from guard.pipeline.testing import FakeBackend, make_settings, write_images
from guard.sources import probe as probe_mod
from guard.sources.probe import diagnose_stream, probe_source
from guard.sources.stream_url import StreamResolveError
from guard.sources.testing import FakeCapture


def rtsp(uri="rtsp://cam.local/stream"):
    return SourceConfig(type="rtsp", uri=uri)


# ---------------------------------------------------------------- probe_source


def test_probe_ok_returns_size_fps_codec_and_preview():
    res = probe_source(rtsp(), timeout_s=2, capture_factory=lambda *a: FakeCapture(frames=50, size=(1280, 720)))
    assert res["ok"] is True and res["error"] is None
    assert (res["width"], res["height"]) == (1280, 720)
    assert res["fps"] == 25.0 and res["fps_nominal"] == 25.0 and res["fps_measured"] == pytest.approx(25.0)
    assert res["codec"] == "h264"
    assert res["first_frame_ms"] is not None and res["open_ms"] is not None
    assert res["frames"] > 1
    img = cv2.imdecode(np.frombuffer(base64.b64decode(res["preview_jpeg_b64"]), np.uint8), cv2.IMREAD_COLOR)
    assert img.shape[1] == 640  # превью уменьшено до ширины 640


def test_probe_measures_fps_by_timestamps_when_nominal_is_wrong():
    # HLS Уфанета: CAP_PROP_FPS = 100, а метки времени идут через 40 мс → 25 к/с
    res = probe_source(rtsp("http://h/a/index.m3u8"), timeout_s=2, capture_factory=lambda *a: FakeCapture(frames=50, fps=100.0))
    assert res["ok"] and res["fps_nominal"] is None
    assert res["fps"] == pytest.approx(25.0) and res["stream"] == "hls"
    assert any("HLS" in h for h in res["hints"])


def test_probe_open_failed_explains_reason():
    res = probe_source(
        rtsp("rtsp://admin:secret@10.0.0.5/Streaming/Channels/101"),
        timeout_s=1,
        capture_factory=lambda *a: FakeCapture(opened=False),
        diagnose=lambda uri: "порт 554 на 10.0.0.5 закрыт",
    )
    assert res["ok"] is False and res["error_code"] == "open_failed"
    assert "порт 554" in res["error"]
    assert "secret" not in res["uri"] and "secret" not in res["error"]


def test_probe_opened_without_frames():
    res = probe_source(rtsp(), timeout_s=1, capture_factory=lambda *a: FakeCapture(frames=0), diagnose=lambda u: None)
    assert res["error_code"] == "no_frames" and "кадр не пришёл" in res["error"]


def test_probe_bad_inputs(tmp_path):
    assert probe_source(rtsp(""))["error_code"] == "bad_uri"
    assert probe_source(rtsp("камера в цехе"))["error_code"] == "bad_uri"
    res = probe_source(SourceConfig(type="file", uri="videos/none.mp4"), uri=str(tmp_path / "none.mp4"))
    assert res["error_code"] == "not_found"
    assert probe_source(SourceConfig(type="images", uri=str(tmp_path)))["error_code"] == "not_found"


def test_probe_images_folder(tmp_path):
    folder = write_images(tmp_path / "imgs", n=3, size=(320, 240))
    res = probe_source(SourceConfig(type="images", uri=str(folder)))
    assert res["ok"] and (res["width"], res["height"]) == (320, 240) and res["frames"] == 3


def test_probe_file_reads_real_video(tmp_path):
    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (160, 120))
    for i in range(15):
        writer.write(np.full((120, 160, 3), i * 10, np.uint8))
    writer.release()
    res = probe_source(SourceConfig(type="file", uri="clip.avi"), uri=str(path), timeout_s=5)
    assert res["ok"] and (res["width"], res["height"]) == (160, 120)
    assert res["fps"] == pytest.approx(10.0) and res["codec"] == "mjpeg"


def test_probe_resolve_error():
    def resolver(uri):
        raise StreamResolveError("трансляция не идёт")

    res = probe_source(rtsp("https://youtu.be/abc"), timeout_s=1, resolver=resolver)
    assert res["error_code"] == "resolve_failed" and "трансляция не идёт" in res["error"]


def test_probe_answer_is_bounded_when_capture_hangs(monkeypatch):
    monkeypatch.setattr(probe_mod, "DIAG_TIMEOUT_S", 0.0)

    def hang(*args):
        time.sleep(1.0)
        return FakeCapture(frames=1)

    t0 = time.monotonic()
    res = probe_source(rtsp(), timeout_s=0.1, capture_factory=hang)
    assert time.monotonic() - t0 < 0.9
    assert res["error_code"] == "timeout"


# ---------------------------------------------------------------- diagnose_stream


class FakeRtspServer:
    """RTSP DESCRIBE: 404 для пути /missing, Digest 401 без верного пароля, 200 с ним."""

    def __init__(self, password="right"):
        self.password = password
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(4)
        self.port = self.sock.getsockname()[1]
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()

    def _serve(self):
        while True:
            try:
                conn, _ = self.sock.accept()
            except OSError:
                return
            with conn:
                conn.settimeout(2)
                while True:
                    try:
                        data = conn.recv(65536).decode()
                    except OSError:
                        break
                    if not data:
                        break
                    conn.sendall(self._answer(data).encode())

    def _answer(self, request):
        line = request.split("\r\n")[0]
        url = line.split(" ")[1]
        cseq = next(ln.split(":", 1)[1].strip() for ln in request.split("\r\n") if ln.lower().startswith("cseq"))
        if url.endswith("/missing"):
            return f"RTSP/1.0 404 Not Found\r\nCSeq: {cseq}\r\n\r\n"
        auth = next((ln.split(":", 1)[1].strip() for ln in request.split("\r\n") if ln.lower().startswith("authorization")), "")
        ha1 = hashlib.md5(f"admin:cam:{self.password}".encode()).hexdigest()
        ha2 = hashlib.md5(f"DESCRIBE:{url}".encode()).hexdigest()
        expected = hashlib.md5(f"{ha1}:nonce1:{ha2}".encode()).hexdigest()
        if f'response="{expected}"' in auth:
            return f"RTSP/1.0 200 OK\r\nCSeq: {cseq}\r\nContent-Length: 0\r\n\r\n"
        return f'RTSP/1.0 401 Unauthorized\r\nCSeq: {cseq}\r\nWWW-Authenticate: Digest realm="cam", nonce="nonce1"\r\n\r\n'

    def close(self):
        self.sock.close()


@pytest.fixture()
def rtsp_server():
    srv = FakeRtspServer()
    yield srv
    srv.close()


def test_diagnose_rtsp_auth_and_path(rtsp_server):
    base = f"127.0.0.1:{rtsp_server.port}"
    assert "требует логин и пароль" in diagnose_stream(f"rtsp://{base}/live")
    assert "неверный логин или пароль" in diagnose_stream(f"rtsp://admin:wrong@{base}/live")
    assert "отвечает (200)" in diagnose_stream(f"rtsp://admin:right@{base}/live")
    assert "поток не найден (RTSP 404)" in diagnose_stream(f"rtsp://{base}/missing")


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_diagnose_refused_port_and_bad_host():
    port = free_port()
    assert f"порт {port} на 127.0.0.1 закрыт" in diagnose_stream(f"rtsp://127.0.0.1:{port}/live")
    assert "не найден (DNS)" in diagnose_stream("rtsp://no-such-host.invalid/live", timeout_s=2)
    assert diagnose_stream("file:///tmp/x.mp4") is None


class _Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        if self.path.startswith("/page"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(b"<html></html>")
        elif self.path.startswith("/token"):
            self.send_response(403)
            self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *args):
        pass


def test_diagnose_http_page_token_and_missing():
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        base = f"http://127.0.0.1:{srv.server_address[1]}"
        assert "веб-страница, а не видеопоток" in diagnose_stream(f"{base}/page")
        assert "временный токен" in diagnose_stream(f"{base}/token/x.m3u8?token=old")
        assert "HTTP 404" in diagnose_stream(f"{base}/nothing.m3u8")
    finally:
        srv.shutdown()


# ---------------------------------------------------------------- API


@pytest.fixture()
def client(tmp_path):
    settings = make_settings(tmp_path)
    engine = PipelineEngine(settings, backend=FakeBackend())
    with TestClient(create_app(settings, engine=engine)) as c:
        yield c, engine, tmp_path


def test_api_probe_images_and_closed_port(client):
    c, engine, tmp_path = client
    folder = write_images(tmp_path / "imgs", n=2)
    r = c.post("/api/cameras/probe", json={"source": {"type": "images", "uri": str(folder)}})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True and body["width"] == 320 and body["preview_jpeg_b64"]
    port = free_port()
    r = c.post("/api/cameras/probe", json={"source": {"type": "rtsp", "uri": f"rtsp://127.0.0.1:{port}/x"}, "timeout_s": 2})
    body = r.json()
    assert r.status_code == 200 and body["ok"] is False and body["error_code"] == "open_failed"
    assert "закрыт" in body["error"]
    # маршрут probe не перехвачен /api/cameras/{camera_id}
    assert c.get("/api/cameras/probe").status_code == 404


def test_api_probe_validation_and_gateway(client):
    c, engine, _ = client
    assert c.post("/api/cameras/probe", json={"source": {"type": "ftp", "uri": "x"}}).status_code == 422
    assert c.post("/api/cameras/probe", json={"source": {"type": "rtsp", "uri": "rtsp://h/x"}, "timeout_s": 999}).status_code == 422
    engine.settings.mode = "gateway"
    try:
        assert c.post("/api/cameras/probe", json={"source": {"type": "rtsp", "uri": "rtsp://h/x"}}).status_code == 409
    finally:
        engine.settings.mode = "local"


def test_api_cameras_and_metrics_expose_source_state(client):
    c, engine, tmp_path = client
    folder = write_images(tmp_path / "cam_imgs", n=3)
    body = {"camera_id": "cam1", "name": "Цех", "site": "Цех 1", "source": {"type": "images", "uri": str(folder)}, "sample_fps": 5}
    assert c.post("/api/cameras", json=body).status_code == 201
    deadline = time.time() + 5
    while time.time() < deadline:
        src = c.get("/api/cameras/cam1").json()["status"]["source"]
        if src["state"] == "online" and src["frames"] > 0:
            break
        time.sleep(0.05)
    assert src["state"] == "online" and src["width"] == 320 and src["height"] == 240
    assert {"fps_in", "last_frame_age_s", "reconnects", "drops", "last_error", "attempts"} <= set(src)
    cams = c.get("/api/metrics").json()["cameras"]
    assert cams["cam1"]["source"]["state"] == "online"
