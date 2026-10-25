"""Сетевой источник без сети: переходы состояний, паузы переподключения, темп HLS, адреса, превью проверки."""

from __future__ import annotations

import pytest

from guard.contracts import SourceConfig
from guard.sources import video as video_mod
from guard.sources.base import CONNECTING, ERROR, OFFLINE, ONLINE, RECONNECTING
from guard.sources.stream_url import (
    StreamResolveError,
    is_hls,
    mask_uri,
    paced_kind,
    resolve_stream_uri,
    resolve_youtube,
    stream_kind,
    strip_secrets,
)
from guard.sources.testing import FakeCapture
from guard.sources.video import BACKOFF_S, LiveSource, Pacer, _CatchUp, fourcc_name, open_capture, sane_fps

class Script:
    """Фабрика захватов: по одному FakeCapture на попытку; сценарий кончился — источник останавливается."""

    def __init__(self, source_ref, captures):
        self.captures = list(captures)
        self.source_ref = source_ref
        self.calls = []

    def __call__(self, *args):
        self.calls.append(args)
        if not self.captures:
            self.source_ref[0]._stop.set()
            return FakeCapture(opened=False)
        cap = self.captures.pop(0)
        cap.args = args
        return cap


def make_live(captures, uri="rtsp://cam.local/stream", realtime=True, resolver=None):
    ref = [None]
    script = Script(ref, captures)
    src = LiveSource(
        SourceConfig(type="rtsp", uri=uri, realtime=realtime),
        uri,
        "cam1",
        capture_factory=script,
        resolver=resolver or (lambda u: (u, None)),
        diagnose=lambda u: None,
    )
    ref[0] = src
    log = []  # (пауза, состояние, попыток подряд, last_error) — только паузы между попытками

    def fake_sleep(seconds):
        if seconds >= 0.5:
            log.append((seconds, src.state, src.attempts, src.last_error))
        return not src._stop.is_set()

    src._sleep = fake_sleep
    frames = []
    src.on_frame = lambda pkt: frames.append((src.state, pkt.capture_id, pkt.ts))
    return src, script, log, frames


def test_backoff_1_2_5_10_and_offline_after_4_failures():
    src, script, log, _ = make_live([FakeCapture(opened=False) for _ in range(6)])
    src._run()
    pauses = [p for p, *_ in log]
    assert pauses == [1.0, 2.0, 5.0, 10.0, 10.0, 10.0]
    states = [s for _, s, *_ in log]
    assert states == [CONNECTING, CONNECTING, CONNECTING, OFFLINE, OFFLINE, OFFLINE]
    assert src.attempts == 6
    assert src.last_error and "не удалось открыть поток" in src.last_error
    assert BACKOFF_S == (1.0, 2.0, 5.0, 10.0)
    # таймауты открытия и чтения передаются в cv2
    uri, api, params = script.calls[0]
    assert api == video_mod.cv2.CAP_FFMPEG
    assert params == [video_mod.cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000, video_mod.cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000,
                      video_mod.cv2.CAP_PROP_N_THREADS, 1]  # один поток декодера — без кадровой задержки


def test_online_drop_reconnect_counts_and_new_capture():
    first = FakeCapture(frames=3)
    second = FakeCapture(frames=2)
    src, _, log, frames = make_live([first, FakeCapture(opened=False), second])
    src._run()
    # три кадра первого подключения, два — после переподключения; во время кадров источник online
    assert [s for s, _, _ in frames] == [ONLINE] * 5
    capture_ids = [c for _, c, _ in frames]
    assert len(set(capture_ids[:3])) == 1 and len(set(capture_ids[3:])) == 1 and capture_ids[0] != capture_ids[3]
    # обрыв → пауза 1 с (reconnecting) → неудача → пауза 2 с → снова online → обрыв → пауза 1 с
    assert [(p, s) for p, s, *_ in log] == [(1.0, RECONNECTING), (2.0, RECONNECTING), (1.0, RECONNECTING)]
    assert "поток оборвался" in log[0][3]
    assert src.drops == 2
    assert src.reconnects == 1
    assert src.last_outage_s is not None and src.last_outage_s >= 0
    assert first.released and second.released
    st = src.status()
    assert st["width"] == 64 and st["height"] == 48
    assert st["codec"] == "h264"
    assert st["stream"] == "rtsp" and st["paced"] is False
    assert st["frames"] == 5
    assert st["connect_ms"] is not None


def test_open_without_frames_is_failed_attempt():
    src, _, log, frames = make_live([FakeCapture(opened=True, frames=0)])
    src._run()
    assert frames == []
    assert log[0][:2] == (1.0, CONNECTING)
    assert "кадров нет" in log[0][3]
    assert src.drops == 0


def test_stall_by_read_timeout_breaks_connection():
    cap = FakeCapture(frames=1)
    src, _, log, _ = make_live([cap])
    src.stall_s = 0.0  # одно чтение, упавшее по таймауту, уже дольше stall_s
    src._run()
    assert src.drops == 1
    assert cap.read_calls == 2  # кадр и одно неудачное чтение — без пяти повторов


def test_resolve_error_is_error_state_and_retries():
    def bad(uri):
        raise StreamResolveError("трансляция не идёт")

    src, _, log, _ = make_live([], uri="https://www.youtube.com/watch?v=x", resolver=bad)
    calls = {"n": 0}

    def resolver(uri):
        calls["n"] += 1
        if calls["n"] >= 3:
            src._stop.set()
        return bad(uri)

    src._resolver = resolver
    src._run()
    assert [s for _, s, *_ in log][:2] == [ERROR, ERROR]
    assert "адрес не разрешён" in src.last_error


def test_youtube_link_is_resolved_on_every_attempt():
    resolved = []

    def resolver(uri):
        resolved.append(uri)
        return f"https://cdn.example/live{len(resolved)}.m3u8", "yt-dlp"

    src, script, _, frames = make_live([FakeCapture(frames=1), FakeCapture(frames=1)], uri="https://youtu.be/abc", resolver=resolver)
    src._run()
    assert len(resolved) == 3  # две удачные попытки и одна после конца сценария
    assert [c[0] for c in script.calls[:2]] == ["https://cdn.example/live1.m3u8", "https://cdn.example/live2.m3u8"]
    assert src.status()["resolved_via"] == "yt-dlp"
    assert src.paced  # YouTube — это HLS: выдаём в темпе меток времени
    assert src.open_timeout_s >= 30 and src.read_timeout_s >= 30


def test_status_brief_and_fps_window():
    src, _, _, _ = make_live([])
    assert src.state == CONNECTING
    brief = src.brief_status()
    assert set(brief) == {"state", "fps_in", "last_frame_age_s", "width", "height", "reconnects", "drops", "last_error"}
    assert brief["fps_in"] == 0.0 and brief["last_frame_age_s"] is None
    # 26 кадров ровно за секунду → 25 к/с
    src._frame_times.extend(100.0 + i * 0.04 for i in range(26))
    src._last_frame_mono = 101.0
    assert src.fps_in(now=101.0) == pytest.approx(25.0, rel=0.01)
    assert src.fps_in(now=200.0) == 0.0  # поток встал — к/с падают к нулю
    assert src.last_frame_age_s(now=103.5) == pytest.approx(2.5)


def test_stop_sets_offline():
    src, _, _, _ = make_live([])
    src.stop(timeout=0.1)
    assert src.state == OFFLINE


def test_webcam_opens_by_index_without_ffmpeg_params():
    ref = [None]
    script = Script(ref, [FakeCapture(frames=1)])
    src = video_mod.WebcamSource(SourceConfig(type="webcam", uri="0"), "0", "cam", capture_factory=script, diagnose=lambda u: None)
    ref[0] = src
    src._sleep = lambda s: not src._stop.is_set()
    src._run()
    assert script.calls[0] == (0,)
    assert src.status()["stream"] == "webcam"


# ---------------------------------------------------------------- темп HLS


def test_pacer_spreads_burst_by_timestamps():
    pacer = Pacer(wall=lambda: 1000.0)
    now = 50.0
    delays = []
    for i in range(5):  # пачка: все кадры декодированы в один момент
        delay, ts = pacer.schedule(i * 40.0, now)
        delays.append(round(delay, 3))
    assert delays == [0.0, 0.04, 0.08, 0.12, 0.16]


def test_pacer_late_frames_go_immediately_and_ts_monotonic():
    clock = {"now": 10.0}
    pacer = Pacer(wall=lambda: 1000.0 + clock["now"])  # настенные часы идут вместе с monotonic
    _, ts0 = pacer.schedule(0.0, clock["now"])
    assert ts0 == pytest.approx(1010.0)
    clock["now"] = 10.5
    delay, ts1 = pacer.schedule(40.0, clock["now"])  # опоздали на 0.46 с — без ожидания, метка в прошлом
    assert delay == 0.0 and ts1 == pytest.approx(1010.04)
    delay, ts2 = pacer.schedule(80.0, clock["now"])
    assert delay == 0.0 and ts2 == pytest.approx(1010.08) and ts2 > ts1


def test_pacer_reanchors_on_jumps_and_ignores_missing_pts():
    pacer = Pacer(wall=lambda: 0.0)
    pacer.schedule(0.0, 0.0)
    assert pacer.schedule(120_000.0, 0.1)[0] == 0.0  # метки прыгнули на 2 мин вперёд — отсчёт заново
    assert pacer.schedule(120_040.0, 0.1)[0] == pytest.approx(0.04)
    assert pacer.schedule(float("nan"), 0.2) == (0.0, None)
    assert pacer.schedule(-1.0, 0.2) == (0.0, None)
    assert pacer.schedule(120_040.0, 0.3) == (0.0, None)  # метка не сдвинулась — нет темпа


def test_http_file_is_paced_and_uses_long_timeouts():
    cap = FakeCapture(frames=3, pos_step_ms=40.0)
    src, script, log, frames = make_live([cap], uri="http://127.0.0.1:8080/clip.mp4")
    waits = []
    stop_sleep = src._sleep

    def record(seconds):
        if 0 < seconds < 0.5:
            waits.append(round(seconds, 2))
        return stop_sleep(seconds)

    src._sleep = record
    src._run()
    assert src.paced and src.stream_kind == "http-file"
    assert len(frames) == 3
    # часы в тесте стоят: второй кадр ждёт 40 мс, третий — 80 мс от опорного (дальше — паузы между неудачными чтениями)
    assert waits[:2] == [0.04, 0.08]
    ts = [t for _, _, t in frames]
    assert ts[1] - ts[0] == pytest.approx(0.04, abs=0.01) and ts[2] - ts[1] == pytest.approx(0.04, abs=0.01)
    _, _, params = script.calls[0]
    assert params[1] == 30000 and params[3] == 30000


def test_live_hls_skips_backlog_before_first_frame(monkeypatch):
    monkeypatch.setattr(video_mod, "CATCHUP_MAX_WALL_S", 0.0)  # край эфира «наступает» сразу после первого кадра
    src, _, _, frames = make_live([FakeCapture(frames=5)], uri="http://127.0.0.1:8890/s1_01/index.m3u8")
    src._run()
    assert src.stream_kind == "hls" and src.paced
    assert len(frames) == 4  # первый кадр запаса пропущен


def test_catchup_skips_backlog_until_live_edge():
    c = _CatchUp()
    assert c.backlog(0.0, 1.0, 0.0)  # первый кадр после открытия: долгое чтение не в счёт
    for i in range(1, 150):  # запас: 3 сегмента по 2 с из буфера, загрузка по 5 мс
        assert c.backlog(i * 40.0, 0.005, 0.1)
    # ждали 1.5 с, но от начала запаса прошло 6 с потока — это ещё не край
    assert c.backlog(6000.0, 1.5, 1.6)
    for i in range(1, 50):
        assert c.backlog(6000.0 + i * 40.0, 0.005, 1.7)
    assert not c.backlog(8000.0, 1.9, 3.6)  # следующий сегмент ждали почти его длительность — край эфира


def test_catchup_limits_and_missing_pts():
    assert not _CatchUp().backlog(float("nan"), 0.0, 0.0)
    c = _CatchUp()
    c.backlog(0.0, 0.0, 0.0)
    assert not c.backlog(61_000.0, 0.0, 1.0)  # больше минуты потока — выдаём как есть
    c = _CatchUp()
    c.backlog(0.0, 0.0, 0.0)
    assert not c.backlog(40.0, 0.0, 31.0)  # больше 30 с ожидания — выдаём как есть
    c = _CatchUp()
    c.backlog(0.0, 0.0, 0.0)
    assert c.backlog(2000.0, 0.6, 1.0)  # загрузка сегмента 0.6 с при сегменте 2 с — ещё запас


def test_unpaced_when_realtime_off():
    src, _, _, _ = make_live([], uri="http://h/x.m3u8", realtime=False)
    assert not src.paced


# ---------------------------------------------------------------- адреса


def test_mask_uri_hides_password_and_tokens():
    assert mask_uri("rtsp://admin:S3cret@10.0.0.5:554/Streaming/Channels/101") == "rtsp://admin:***@10.0.0.5:554/Streaming/Channels/101"
    assert mask_uri("не удалось открыть http://h/x.m3u8?token=abc123&q=1") == "не удалось открыть http://h/x.m3u8?token=***&q=1"
    assert mask_uri("rtsp://10.0.0.5/live") == "rtsp://10.0.0.5/live"
    assert mask_uri(None) is None
    assert strip_secrets("rtsp://u:p@h:554/a?token=t&x=1") == "rtsp://h:554/a?token=%2A%2A%2A&x=1"


@pytest.mark.parametrize(
    "uri,kind",
    [
        ("rtsp://h/s", "rtsp"),
        ("http://h:8890/cam/index.m3u8", "hls"),
        ("https://cdn/hls/playlists/ts.m3u8?uuid=1", "hls"),
        ("https://www.youtube.com/watch?v=abc", "youtube"),
        ("https://youtu.be/abc", "youtube"),
        ("http://192.168.1.5:8080/video", "mjpeg"),
        ("http://h/clip.mp4", "http-file"),
        ("rtmp://h/live", "rtmp"),
    ],
)
def test_stream_kind(uri, kind):
    assert stream_kind(uri) == kind


def test_paced_kinds():
    assert paced_kind("hls") and paced_kind("youtube") and paced_kind("http-file")
    assert not paced_kind("rtsp") and not paced_kind("mjpeg")
    assert is_hls("http://h/a.m3u8") and not is_hls("rtsp://h/a.m3u8")


class FakeRun:
    def __init__(self, stdout="", stderr="", code=0):
        self.result = type("P", (), {"stdout": stdout, "stderr": stderr, "returncode": code})()
        self.cmd = None

    def __call__(self, cmd, **kw):
        self.cmd = cmd
        return self.result


def test_resolve_youtube_via_ytdlp(monkeypatch):
    monkeypatch.setenv("GUARD_YTDLP", "/bin/yt-dlp-test")
    run = FakeRun(stdout="https://manifest.googlevideo.com/x/index.m3u8\n")
    assert resolve_youtube("https://youtu.be/abc", run=run) == "https://manifest.googlevideo.com/x/index.m3u8"
    assert run.cmd[0] == "/bin/yt-dlp-test" and "-g" in run.cmd
    assert resolve_stream_uri("rtsp://h/s") == ("rtsp://h/s", None)
    with pytest.raises(StreamResolveError, match="Sign in"):
        resolve_youtube("https://youtu.be/abc", run=FakeRun(stderr="ERROR: Sign in to confirm", code=1))


def test_youtube_format_selects_video_only_low_resolution_hls_first(monkeypatch):
    # YouTube Live отдаёт видео и звук отдельными HLS-вариантами: «best» (общий поток со звуком)
    # там нет, yt-dlp падает с «Requested format is not available» (проверено со стенда 25.10).
    # Анализу звук не нужен, а 360p дешевле декодировать на стенде.
    monkeypatch.setenv("GUARD_YTDLP", "/bin/yt-dlp-test")
    run = FakeRun(stdout="https://manifest.googlevideo.com/x/index.m3u8\n")
    resolve_youtube("https://www.youtube.com/watch?v=abc", run=run)
    selector = run.cmd[run.cmd.index("-f") + 1]
    alternatives = selector.split("/")
    assert alternatives[0].startswith("bv*[height<=360]") and "[protocol^=m3u8]" in alternatives[0]
    assert all("height<=" in alt for alt in alternatives[:-1])
    assert max(int(alt.split("height<=")[1].split("]")[0]) for alt in alternatives if "height<=" in alt) <= 480


def test_codec_and_fps_helpers():
    assert fourcc_name(FakeCapture()) == "h264"
    assert fourcc_name(FakeCapture(fourcc=int.from_bytes(b"hvc1", "little"))) == "h265"
    assert fourcc_name(FakeCapture(fourcc=0)) is None
    assert sane_fps(25.0) == 25.0 and sane_fps(90000.0) is None and sane_fps(0) is None and sane_fps(100.0) is None
    cap = open_capture("rtsp://h/s", 3.0, 2.0, factory=lambda *a: FakeCapture())
    assert cap.isOpened()
