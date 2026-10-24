"""Offline tests of opt-in, safety boundaries and actual confirmation barriers."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import rabbit_acceptance as acceptance
from guard.contracts import Event
from guard.transport.protocol import Envelope, encode, event_payload
from uuid import uuid4


def envelope():
    producer = str(uuid4())
    ev = Event(id='one', camera_id='cam', site='site', rule_id='zone', type='zone_intrusion', severity='high',
               started_at=100, confirmed_at=102, bbox=(1, 2, 3, 4))
    return Envelope(edge_id='accept-test', producer_id=producer, run_id=str(uuid4()), message_id=producer + ':1',
                    sequence=1, kind='event', sent_at=103, payload=event_payload(ev, snapshot_jpeg=acceptance.synthetic_jpeg('red')))


@pytest.mark.parametrize('url', ['http://localhost:5672', 'amqp://host.example:5672', 'amqp://127.0.0.2:5672', ''])
def test_only_explicit_loopback_broker(url):
    with pytest.raises((acceptance.AcceptanceError, ValueError)):
        acceptance.broker_address(url)


def test_control_rejects_scope_or_address_without_executing(tmp_path, monkeypatch):
    path = tmp_path / 'control.json'
    path.write_text(json.dumps({'scope': 'system_service', 'host': '127.0.0.1', 'port': 5672, 'start': ['start'], 'stop': ['stop']}))
    calls = []
    monkeypatch.setattr(acceptance.subprocess, 'run', lambda *a, **kw: calls.append(a))
    with pytest.raises(acceptance.AcceptanceError):
        acceptance.BrokerControl(path, 'amqp://guest:secret@127.0.0.1:5672/')
    assert calls == []


def test_control_uses_explicit_argv_no_shell_and_tracks_restart(tmp_path, monkeypatch):
    path = tmp_path / 'control.json'
    path.write_text(json.dumps({'scope': 'owned_loopback_broker', 'host': '127.0.0.1', 'port': 5672,
                               'start': ['python', '/owned path/control.py', 'start'], 'stop': ['python', '/owned path/control.py', 'stop']}))
    calls = []
    monkeypatch.setattr(acceptance.subprocess, 'run', lambda argv, **kw: (calls.append((argv, kw)) or SimpleNamespace(returncode=0)))
    control = acceptance.BrokerControl(path, 'amqp://guest:secret@127.0.0.1:5672/')
    control.run('stop')
    assert control.stopped
    control.run('start')
    assert not control.stopped
    assert calls[0][0] == ['python', '/owned path/control.py', 'stop']
    assert all('shell' not in kw and kw['timeout'] == 150 for _, kw in calls)


def test_no_opt_in_or_missing_url_cannot_pass(monkeypatch, tmp_path, capsys):
    monkeypatch.delenv('GUARD_RABBIT_URL', raising=False)
    assert acceptance.main(['--output', str(tmp_path / 'absent')]) == 2
    assert acceptance.main(['--run', '--output', str(tmp_path / 'absent')]) == 2
    assert not (tmp_path / 'absent').exists()
    assert 'unavailable' in capsys.readouterr().err


def test_credential_error_never_prints_raw_network_exception(monkeypatch, tmp_path, capsys):
    secret = 'amqp://user:private-password@127.0.0.1:5672/'
    monkeypatch.setenv('GUARD_RABBIT_URL', secret)
    def failure(_):
        raise RuntimeError(secret)
    monkeypatch.setattr(acceptance, 'run_acceptance', failure)
    assert acceptance.main(['--run']) == 2
    output = capsys.readouterr()
    assert 'RuntimeError' in output.err and 'private-password' not in output.err + output.out


def test_confirm_crash_happens_after_real_publish_return_and_freezes_evidence(tmp_path, monkeypatch):
    message = envelope()
    calls = []
    class Publisher:
        def publish(self, ev, *, durable):
            calls.append(('publish-return', encode(ev)))
        def close(self):
            pass
    class ExitBarrier(BaseException):
        pass
    def crash(code):
        assert calls and code == 72
        evidence = json.loads((tmp_path / 'last-confirm.json').read_text())
        assert evidence['sha256'] == acceptance.digest(encode(message))
        calls.append(('exit', code))
        raise ExitBarrier
    monkeypatch.setattr(acceptance.os, '_exit', crash)
    publisher = acceptance.ConfirmBarrier(Publisher(), tmp_path)
    publisher.crash = True
    with pytest.raises(ExitBarrier):
        publisher.publish(message)
    assert calls[0][0] == 'publish-return' and calls[-1] == ('exit', 72)
    assert (tmp_path / ('wire-' + acceptance.digest(encode(message)) + '.json')).read_bytes() == encode(message)


def test_confirm_failure_cannot_create_success_evidence(tmp_path):
    class Publisher:
        def publish(self, *args, **kwargs):
            raise ConnectionError('not confirmed')
    publisher = acceptance.ConfirmBarrier(Publisher(), tmp_path)
    with pytest.raises(ConnectionError):
        publisher.publish(envelope())
    assert list(tmp_path.iterdir()) == []


def test_wait_is_bounded_without_sleep(monkeypatch):
    times = iter([0, 0, 0.1, 0.2, 0.3])
    monkeypatch.setattr(acceptance.time, 'monotonic', lambda: next(times))
    monkeypatch.setattr(acceptance.time, 'sleep', lambda _: None)
    with pytest.raises(acceptance.AcceptanceError, match='Timeout: test'):
        acceptance.wait_for(lambda: False, 0.15, 'test')


def test_exact_durable_callback_barrier_ignores_telemetry_and_counts_replays():
    ledger = acceptance.DurableCallbacks()
    ev = envelope()
    ledger.record(ev.model_copy(update={'kind': 'metrics'}), None)
    assert ledger.snapshot() == {}
    ledger.record(ev, SimpleNamespace(changed=False))
    ledger.record(ev, SimpleNamespace(changed=False))
    assert ledger.snapshot() == {ev.message_id: {'completed': 2, 'changed': 0}}


@pytest.mark.parametrize('code, verified, allowed', [(1, None, False), (72, None, False), (72, 72, True), (73, 73, True), (0, None, True)])
def test_cleanup_rejects_unexpected_dead_actor(code, verified, allowed):
    actor = acceptance.Actor.__new__(acceptance.Actor)
    stream = SimpleNamespace(close=lambda: None)
    actor.log = stream
    actor.verified_exit = verified
    actor.process = SimpleNamespace(poll=lambda: code, returncode=code, stdin=stream, stdout=stream)
    if allowed:
        actor.stop()
    else:
        with pytest.raises(acceptance.AcceptanceError, match='unexpectedly'):
            actor.stop()


def test_admin_preserves_topology_error_when_broker_already_closed_connection(monkeypatch):
    import guard.transport.rabbit as rabbit
    connection = SimpleNamespace(is_open=False, channel=lambda: object(), close=lambda: pytest.fail('Already closed connection'))
    monkeypatch.setattr(rabbit, '_connection', lambda _: connection)
    def fail(*args):
        raise ValueError('Original topology failure')
    monkeypatch.setattr(rabbit, '_topology', fail)
    with pytest.raises(ValueError, match='Original topology failure'):
        with acceptance.admin('amqp://127.0.0.1:5672/', 'accept-test'):
            pytest.fail('Topology must not be ready')
