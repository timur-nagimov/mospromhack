"""Офлайн-тесты LinqTpuBackend: формат tpu_pycloud, raw-кодек, конвейер infer/submit на фейковом транспорте."""

from __future__ import annotations

import json

import numpy as np
import pytest

from guard.contracts import ModelSpec
from guard.inference import get_backend
from guard.inference import linq_tpu as lt

# раскладка как у yolov8n (metadata.json): вход NCHW -> [C/B, H, W, N, B], выход (N,W,C) -> [C/B, W, N, B]
IN_META = {"x": {"user_shape": [1, 3, 8, 8], "user_order": ["N", "C", "H", "W"], "user_dtype": "float32",
                 "tpu_shape": [1, 8, 8, 1, 4], "tpu_order": ["C", "H", "W", "N", "B"], "tpu_dtype": "int8",
                 "scales": [127.0], "scale_shape": [1], "zero_points": [0.0],
                 "padding": [[0, 0], [0, 1], [0, 0], [0, 0]], "size": 256}}
OUT_META = {"y": {"user_shape": [1, 5, 300], "user_order": ["N", "W", "C"], "user_dtype": "float32",
                  "tpu_shape": [3, 5, 1, 128], "tpu_order": ["C", "W", "N", "B"], "tpu_dtype": "int8",
                  "scales": [0.01], "scale_shape": [1], "zero_points": [0.0],
                  "padding": [[0, 0], [0, 0], [0, -84]], "size": 1920}}


@pytest.fixture(autouse=True)
def isolated_linq_settings(monkeypatch):
    # Never read the real repository .env from fake-transport tests.
    monkeypatch.setattr(lt, "_dotenv", lambda: {})
    monkeypatch.setenv("GUARD_BACKEND_FALLBACK", "none")


def test_wire_roundtrip():
    x = {"a": np.random.rand(1, 3, 4, 4).astype(np.float32), "b": np.arange(6, dtype=np.uint8).reshape(2, 3)}
    data, worker = lt.deserialize_task(lt.serialize_task(x))
    assert set(data) == {"a", "b"}
    for k in x:
        assert data[k].dtype == x[k].dtype and np.array_equal(data[k], x[k])
    assert worker["status_code"] == 0


def test_raw_pack_layout():
    codec = lt.RawCodec(IN_META, OUT_META)
    x = np.random.rand(1, 3, 8, 8).astype(np.float32)
    packed = codec.pack("x", x)
    assert packed.dtype == np.uint8 and packed.size == IN_META["x"]["size"]
    t = packed.view(np.int8).reshape(1, 8, 8, 1, 4)  # C/B, H, W, N, B
    assert np.array_equal(t[0, :, :, 0, :3], np.rint(x[0] * 127).astype(np.int8).transpose(1, 2, 0))
    assert not t[0, :, :, 0, 3].any()  # добитый канал — нули


def test_raw_unpack_crop_and_scale():
    codec = lt.RawCodec(IN_META, OUT_META)
    q = np.random.randint(-128, 128, size=(3, 5, 1, 128), dtype=np.int8)  # C/B, W, N, B
    y = codec.unpack("y", q.view(np.uint8).reshape(-1, 32))
    assert y.shape == (1, 5, 300)
    ref = q.transpose(2, 1, 0, 3).reshape(1, 5, 384)[:, :, :300].astype(np.float32) * 0.01
    np.testing.assert_allclose(y, ref, rtol=0, atol=1e-6)


def test_raw_unpack_float16_output():
    # Видимость ключевых точек pose (Sigmoid_1) программа отдаёт в float16, а не int8:
    # 2 байта на значение, scale 1.0. Чтение байтов как int8 давало шум −128..127.
    meta = {"v": {**OUT_META["y"], "tpu_dtype": "float16", "scales": [1.0], "size": 3840}}
    codec = lt.RawCodec(IN_META, meta)
    v = np.random.rand(3, 5, 1, 128).astype(np.float16)  # C/B, W, N, B
    y = codec.unpack("v", v.view(np.uint8).reshape(-1, 32))
    assert y.dtype == np.float32 and y.shape == (1, 5, 300)
    ref = v.transpose(2, 1, 0, 3).reshape(1, 5, 384)[:, :, :300].astype(np.float32)
    np.testing.assert_array_equal(y, ref)


def test_raw_unpack_rejects_unknown_dtype():
    meta = {"v": {**OUT_META["y"], "tpu_dtype": "bfloat16"}}
    with pytest.raises(NotImplementedError):
        lt.RawCodec(IN_META, meta).unpack("v", np.zeros(3840, np.uint8))


class FakeClient:
    """Имитация RabbitClient: отвечает тензором y=сумма входа, проверяет формат raw-запроса."""

    def __init__(self, raw: bool):
        self.raw, self.pending, self.is_open, self.reply_queue = raw, {}, True, "fake"

    def submit(self, tensors, uri, body=None):
        data, _ = lt.deserialize_task(body)
        x = data["x"]
        if self.raw:
            assert "?raw=True" in uri and x.dtype == np.uint8 and x.shape[1] == 32
            out = {"y": np.zeros((60, 32), np.uint8)}
        else:
            assert "?raw=False" in uri and x.dtype == np.float32
            out = {"y": np.full((1, 5, 300), float(x.sum()), np.float32)}
        msg = lt._task_cls()()
        for k, v in out.items():
            a = msg.dict_data[k]
            a.dtype, a.byte_data = v.dtype.name, v.tobytes()
            a.shape.extend(v.shape)
        msg.inference_time = 0.005
        mid = str(len(self.pending))
        self.pending[mid] = msg.SerializeToString()
        return mid

    def collect(self, ids, timeout):
        out = {}
        for i in ids:
            data, worker = lt.deserialize_task(self.pending.pop(i))
            out[i] = lt.Reply(data, worker, 0.0, 0.0, 100, 100)
        return out

    def close(self):
        pass


@pytest.mark.parametrize("raw", [False, True])
def test_backend_pipeline_fake(tmp_path, raw):
    d = tmp_path / "tpu" / "m"
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps({
        "inputs": {"x": {"shape": [1, 3, 8, 8]}}, "tpu_outputs": ["y"], "batch": 1,
        "program_meta": {"inputs": IN_META, "outputs": OUT_META}}))
    be = get_backend("linq-tpu", host="fake", models_dir=tmp_path, raw=raw, window=2)
    assert be.name == "linq-tpu"
    be._client = lambda: FakeClient(raw)  # type: ignore[method-assign]
    be.load(ModelSpec(model_id="m", onnx_path="none.onnx", tpu_program="file:///auto/cache/u/m.tpu", input_size=(8, 8)))
    x = np.random.rand(3, 3, 8, 8).astype(np.float32)
    outs, t = be.infer_timed("m", x)  # батч 3 -> 3 задачи конвейером
    assert outs[0].shape == (3, 5, 300)
    if not raw:
        np.testing.assert_allclose(outs[0][:, 0, 0], x.reshape(3, -1).sum(1), rtol=1e-5)
    assert {"transfer", "infer", "serialize"} <= set(t)
    fut = be.submit("m", x[:1])
    assert fut.result(timeout=5)[0][0].shape == (1, 5, 300)
    st = be.stats()
    assert st["models"]["m"]["items"] >= 4 and st["models"]["m"]["errors"] == 0
    be.close()
