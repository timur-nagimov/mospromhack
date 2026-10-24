"""Отказ соединения, circuit breaker и завершение потоков без сети."""

from __future__ import annotations

import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from guard.contracts import ModelSpec
from guard.inference import linq_tpu as lt


@pytest.fixture(autouse=True)
def isolated_config(monkeypatch):
    monkeypatch.setattr(lt, "_cfg", lambda *names, default=None: default)
    monkeypatch.setattr(lt, "_default_host", lambda: "fake")
    monkeypatch.delenv("LINQ_RABBIT_HOST", raising=False)
    monkeypatch.setattr(lt, "serialize_task", lambda tensors: b"task")


def make_backend(**kwargs):
    backend = lt.LinqTpuBackend(host="fake", raw=False, window=2, **kwargs)
    spec = ModelSpec(
        model_id="m",
        onnx_path="unused.onnx",
        tpu_program="file:///fake/m.tpu",
        input_size=(8, 8),
    )
    backend._models["m"] = lt._TpuModel(
        spec, "file:///fake/m.tpu?raw=False", "x", 1, ["y"], ["y"],
    )
    return backend


@pytest.fixture
def batch():
    return np.ones((1, 3, 8, 8), np.float32)


@pytest.fixture
def broken_pika(monkeypatch):
    parameters = []

    def connection_params(*args, **kwargs):
        return SimpleNamespace(args=args, **kwargs)

    def connect(params):
        parameters.append(params)
        raise OSError("broker unreachable")

    fake = SimpleNamespace(
        PlainCredentials=lambda user, password: (user, password),
        ConnectionParameters=connection_params,
        BlockingConnection=connect,
    )
    monkeypatch.setitem(sys.modules, "pika", fake)
    return parameters


class SuccessClient:
    def __init__(self):
        self.is_open = True
        self.counter = 0
        self.closed = False

    def submit(self, tensors, uri, body=None):
        self.counter += 1
        return str(self.counter)

    def collect(self, ids, timeout):
        return {
            mid: lt.Reply(
                {"y": np.ones((1, 2), np.float32)},
                {"status_code": 0, "error_message": "", "inference_time": 0.001},
                0.0, 0.0, 4, 8,
            )
            for mid in ids
        }

    def close(self):
        self.closed = True
        self.is_open = False


def test_connection_failure_is_fast_and_has_bounded_parameters(broken_pika, batch):
    backend = make_backend()
    try:
        start = time.perf_counter()
        with pytest.raises(lt.BackendUnavailable, match="broker unreachable") as caught:
            backend.infer("m", batch)
        assert time.perf_counter() - start < 2.5
        assert isinstance(caught.value.__cause__, OSError)
        assert len(broken_pika) == 1
        params = broken_pika[0]
        assert 0 < params.socket_timeout <= 2.0
        assert params.socket_timeout < params.stack_timeout <= 2.0
        assert params.blocked_connection_timeout <= 2.0
        assert params.connection_attempts == 1
        assert params.retry_delay == 0
        state = backend.stats()
        assert state["consecutive_failures"] == 1
        assert state["circuit_open"] is False
        assert state["available"] is False
        assert "broker unreachable" in state["last_error"]
    finally:
        backend.close()


@pytest.mark.parametrize("retries, attempts", [(0, 1), (1, 2), (10, 2), (-1, 1)])
def test_no_more_than_one_explicit_retry(broken_pika, batch, retries, attempts):
    backend = make_backend(retries=retries, connect_timeout_s=50)
    try:
        with pytest.raises(lt.BackendUnavailable):
            backend.infer("m", batch)
        assert len(broken_pika) == attempts
        assert all(params.stack_timeout <= 2 for params in broken_pika)
        # Размыкатель считает вызовы инференса, а не внутренние попытки.
        assert backend.stats()["consecutive_failures"] == 1
    finally:
        backend.close()


def test_circuit_blocks_network_and_recovers(monkeypatch, broken_pika, batch):
    now = [100.0]
    monkeypatch.setattr(
        lt, "time", SimpleNamespace(monotonic=lambda: now[0], perf_counter=time.perf_counter),
    )
    backend = make_backend()
    try:
        for number in range(1, 4):
            with pytest.raises(lt.BackendUnavailable):
                backend.infer("m", batch)
            assert backend.stats()["consecutive_failures"] == number
        assert backend.stats()["circuit_open"] is True
        assert len(broken_pika) == 3
        for moment in (100.0, 105.0, 109.999):
            now[0] = moment
            with pytest.raises(lt.BackendUnavailable, match="circuit open"):
                backend.infer("m", batch)
        assert len(broken_pika) == 3
        assert backend.stats()["consecutive_failures"] == 3

        now[0] = 110.0
        with pytest.raises(lt.BackendUnavailable, match="broker unreachable"):
            backend.infer("m", batch)
        assert len(broken_pika) == 4
        assert backend.stats()["circuit_open"] is True
        now[0] = 119.999
        with pytest.raises(lt.BackendUnavailable, match="circuit open"):
            backend.infer("m", batch)

        client = SuccessClient()
        monkeypatch.setattr(backend, "_client", lambda: client)
        now[0] = 120.0
        outputs = backend.infer("m", batch)
        assert outputs[0].shape == (1, 2)
        state = backend.stats()
        assert state["circuit_open"] is False
        assert state["consecutive_failures"] == 0
        assert state["last_error"] is None
        assert state["available"] is True
    finally:
        backend.close()


def test_success_resets_consecutive_failures(monkeypatch, broken_pika, batch):
    backend = make_backend()
    real_client = backend._client
    try:
        for _ in range(2):
            with pytest.raises(lt.BackendUnavailable):
                backend.infer("m", batch)
        monkeypatch.setattr(backend, "_client", lambda: SuccessClient())
        backend.infer("m", batch)
        monkeypatch.setattr(backend, "_client", real_client)
        with pytest.raises(lt.BackendUnavailable):
            backend.infer("m", batch)
        assert backend.stats()["consecutive_failures"] == 1
        assert backend.stats()["circuit_open"] is False
    finally:
        backend.close()


def test_worker_error_is_not_retried(monkeypatch, batch):
    backend = make_backend(retries=1)
    calls = []

    class BadWorker(SuccessClient):
        def collect(self, ids, timeout):
            calls.append(1)
            replies = super().collect(ids, timeout)
            for reply in replies.values():
                reply.worker.update(status_code=1, error_message="bad program")
            return replies

    monkeypatch.setattr(backend, "_client", lambda: BadWorker())
    try:
        with pytest.raises(lt.TpuWorkerError, match="bad program"):
            backend.infer("m", batch)
        assert calls == [1]
        assert backend.stats()["consecutive_failures"] == 1
    finally:
        backend.close()


def test_only_one_half_open_call(monkeypatch, broken_pika, batch):
    now = [100.0]
    monkeypatch.setattr(
        lt, "time", SimpleNamespace(monotonic=lambda: now[0], perf_counter=time.perf_counter),
    )
    backend = make_backend()
    entered = threading.Event()
    release = threading.Event()

    class SlowSuccess(SuccessClient):
        def collect(self, ids, timeout):
            entered.set()
            assert release.wait(timeout=3)
            return super().collect(ids, timeout)

    try:
        for _ in range(3):
            with pytest.raises(lt.BackendUnavailable):
                backend.infer("m", batch)
        now[0] = 110.0
        monkeypatch.setattr(backend, "_client", lambda: SlowSuccess())
        future = backend.submit("m", batch)
        assert entered.wait(timeout=2)
        with pytest.raises(lt.BackendUnavailable, match="circuit open"):
            backend.infer("m", batch)
        release.set()
        future.result(timeout=3)
        assert backend.stats()["consecutive_failures"] == 0
    finally:
        release.set()
        backend.close()


def test_close_interrupts_wait_and_joins_pool(monkeypatch, batch):
    backend = make_backend()
    entered = threading.Event()
    stop = threading.Event()

    class WaitingClient(SuccessClient):
        def collect(self, ids, timeout):
            entered.set()
            assert stop.wait(timeout=3)
            raise lt.BackendUnavailable("stopped")

        def request_close(self):
            stop.set()

    client = WaitingClient()
    backend._clients.append(client)
    monkeypatch.setattr(backend, "_client", lambda: client)
    future = backend.submit("m", batch)
    try:
        assert entered.wait(timeout=2)
        pool = backend._pool
        backend.close()
        assert client.closed
        assert backend._clients == []
        assert backend._pool is None
        assert pool is not None
        assert not any(thread.is_alive() for thread in pool._threads)
        assert backend.stats()["available"] is False
        with pytest.raises(lt.BackendUnavailable):
            future.result(timeout=1)
        with pytest.raises(lt.BackendUnavailable, match="закрыт"):
            backend.infer("m", batch)
        with pytest.raises(lt.BackendUnavailable, match="закрыт"):
            backend.submit("m", batch).result(timeout=1)
        backend.close()
    finally:
        stop.set()
        backend.close()


def test_rabbit_client_close_releases_connection_and_buffers():
    closed = []
    client = lt.RabbitClient("fake", 5672)
    client.conn = SimpleNamespace(is_open=True, close=lambda: closed.append(True))
    client.ch = SimpleNamespace(is_open=True)
    client._pending["m"] = (0.0, 1)
    client._replies["m"] = object()
    client.close()
    assert closed == [True]
    assert client.conn is None and client.ch is None
    assert not client._pending and not client._replies
    client.close()
    assert closed == [True]


def test_partial_connection_is_closed_on_channel_failure(monkeypatch):
    closed = []

    def broken_channel():
        raise OSError("channel failed")

    connection = SimpleNamespace(
        is_open=True,
        channel=broken_channel,
        close=lambda: closed.append(True),
    )
    monkeypatch.setitem(sys.modules, "pika", SimpleNamespace(
        PlainCredentials=lambda *args: args,
        ConnectionParameters=lambda *args, **kwargs: kwargs,
        BlockingConnection=lambda params: connection,
    ))
    client = lt.RabbitClient("fake", 5672)
    with pytest.raises(OSError, match="channel failed"):
        client.connect()
    assert closed == [True]
    assert client.conn is None and client.ch is None
