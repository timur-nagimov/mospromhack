"""Opt-in two-process acceptance of real RabbitMQ + edge/gateway SQLite.

Fault barriers wrap real adapters only in these actors. No production failpoints,
models, camera sources or fake broker are used. The AMQP URL is environment-only.
"""
from __future__ import annotations

import argparse
import base64
import contextlib
import hashlib
import io
import json
import logging
import os
from pathlib import Path
import queue
import signal
import socket
import sqlite3
import subprocess
import sys
import threading
import time
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class AcceptanceError(RuntimeError):
    pass


class AcceptanceUnavailable(AcceptanceError):
    pass


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix('.part')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def wait_for(check, timeout, label):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            result = check()
            if result:
                return result
        except (OSError, sqlite3.Error):
            pass
        time.sleep(min(0.05, max(0, deadline - time.monotonic())))
    raise AcceptanceError('Timeout: ' + label)


def check(condition, label):
    if not condition:
        raise AcceptanceError(label)


def broker_address(url):
    parsed = urlsplit(url)
    if parsed.scheme not in {'amqp', 'amqps'} or parsed.hostname not in {'127.0.0.1', 'localhost', '::1'}:
        raise AcceptanceError('Acceptance requires an explicit loopback AMQP URL')
    return parsed.hostname, parsed.port or (5671 if parsed.scheme == 'amqps' else 5672)


class BrokerControl:
    def __init__(self, path, url):
        data = json.loads(Path(path).read_text())
        host, port = broker_address(url)
        if data.get('scope') != 'owned_loopback_broker' or data.get('host') != host or data.get('port') != port:
            raise AcceptanceError('Broker control scope/address does not match the explicit AMQP URL')
        self.commands = {}
        for action in ('start', 'stop'):
            argv = data.get(action, data.get(action + 'argv'))
            if not isinstance(argv, list) or not argv or not all(isinstance(value, str) and value for value in argv):
                raise AcceptanceError('Broker control must contain explicit argv arrays')
            self.commands[action] = argv
        self.stopped = False

    def run(self, action):
        # Only this explicitly supplied owned control is executed; never a shell.
        result = subprocess.run(self.commands[action], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=150)
        check(result.returncode == 0, 'Owned broker ' + action + ' failed')
        self.stopped = action == 'stop'


def actor_settings(root, mode, edge_id):
    from guard.pipeline.settings import Settings
    return Settings(data_dir=root / 'data', models_dir=root / 'no-models', cameras_config=root / 'cameras.yaml',
                    models_config=root / 'models.yaml', db_path=root / 'data' / 'guard.db', media_dir=root / 'data' / 'media',
                    frontend_dist=root / 'no-frontend', mode=mode, edge_id=edge_id,
                    rabbit_url=os.environ['GUARD_RABBIT_URL'], autostart=True)


def synthetic_jpeg(color):
    from PIL import Image
    output = io.BytesIO()
    Image.new('RGB', (48, 32), color).save(output, 'JPEG')
    return output.getvalue()


class ConfirmBarrier:
    """Wrap real publish-confirm, then crash before the runner marks outbox."""
    def __init__(self, publisher, root):
        self.publisher, self.root = publisher, root
        self.gate = threading.Event()
        self.gate.set()
        self.crash = False

    def publish(self, envelope, *, durable=True):
        from guard.transport.protocol import encode
        check(self.gate.wait(30), 'Publisher barrier not released')
        self.publisher.publish(envelope, durable=durable)
        wire = encode(envelope)
        sha = digest(wire)
        (self.root / ('wire-' + sha + '.json')).write_bytes(wire)
        write_json(self.root / 'last-confirm.json', {'message_id': envelope.message_id, 'sha256': sha})
        if self.crash:
            os._exit(72)

    def close(self):
        self.gate.set()
        self.publisher.close()

    def status(self):
        return self.publisher.status()


class DurableCallbacks:
    """Actor-only exact-message barrier; telemetry never advances it."""
    def __init__(self):
        self.lock = threading.Lock()
        self.messages = {}

    def record(self, envelope, result):
        if envelope.kind not in {'event', 'event_update'}:
            return
        with self.lock:
            item = self.messages.setdefault(envelope.message_id, {'completed': 0, 'changed': 0})
            item['completed'] += 1
            item['changed'] += bool(result.changed)

    def snapshot(self):
        with self.lock:
            return {key: dict(value) for key, value in self.messages.items()}


def run_actor(args):
    # All log output is sanitized: Pika exceptions can contain credentials/URL.
    logging.disable(logging.CRITICAL)
    import guard.pipeline.settings as settings_module
    settings_module.load_dotenv_guard = lambda *a, **kw: None
    from guard.contracts import Event
    from guard.transport.protocol import decode
    from guard.transport.rabbit import RabbitConfig, RabbitPublisher
    root = args.directory.resolve()
    root.mkdir(parents=True, exist_ok=True)
    settings = actor_settings(root, args.actor, args.edge_id)
    config = RabbitConfig(settings.rabbit_url, args.edge_id)
    server = server_thread = None
    callbacks = DurableCallbacks()
    if args.actor == 'edge':
        from guard.edge.runner import EdgeEngine
        from guard.inference.base import InferenceBackend
        class NoInference(InferenceBackend):
            name = 'acceptance-synthetic'
            def load(self, spec):
                raise AssertionError('Acceptance must not load models')
            def infer(self, model_id, batch):
                raise AssertionError('Acceptance must not run inference')
            def stats(self):
                return {'backend': self.name, 'models': {}}
        (root / 'models.yaml').write_text('models: []\n')
        (root / 'cameras.yaml').write_text('cameras: []\n')
        publisher = ConfirmBarrier(RabbitPublisher(config), root)
        engine = EdgeEngine(settings, backend=NoInference(), publisher=publisher)
        engine.start()
    else:
        from guard.gateway.engine import GatewayEngine
        from guard.api.app import create_app
        import uvicorn
        engine = GatewayEngine(settings)
        receive = engine.receive
        fault = {'crash_commit': args.crash_commit, 'transient': 0}
        def wrapped(envelope):
            if envelope.kind in {'event', 'event_update'} and fault['transient']:
                fault['transient'] -= 1
                write_json(root / 'transient-nack.json', {'message_id': envelope.message_id, 'error': 'OperationalError'})
                raise sqlite3.OperationalError('Acceptance transient storage failure')
            result = receive(envelope)
            if fault['crash_commit'] and envelope.kind in {'event', 'event_update'}:
                write_json(root / 'commit-before-ack.json', {'message_id': envelope.message_id,
                           'created': result.created, 'changed': result.changed})
                os._exit(73)
            callbacks.record(envelope, result)
            return result
        engine.receive = wrapped
        server = uvicorn.Server(uvicorn.Config(create_app(settings, engine), host='127.0.0.1', port=args.port,
                                               log_config=None, access_log=False, ws='websockets'))
        server_thread = threading.Thread(target=server.run, daemon=True)
        server_thread.start()
        wait_for(lambda: server.started, 10, 'HTTP server startup')
    def respond(value):
        print(json.dumps(value), flush=True)
    respond({'ready': True, 'pid': os.getpid()})
    try:
        for line in sys.stdin:
            try:
                command = json.loads(line)
                op = command['op']
                if op == 'status':
                    value = {'health': engine.health(), 'published': engine.bus.published, 'subscribers': engine.bus.subscribers,
                             'durable_callbacks': callbacks.snapshot()}
                elif op == 'add':
                    image = synthetic_jpeg('red')
                    path = settings.media_dir / 'source-red.jpg'
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(image)
                    ev = Event(id=args.event_id, camera_id='synthetic', site='Acceptance', rule_id='zone', type='zone_intrusion',
                               severity='high', started_at=100, confirmed_at=102, bbox=(1, 1, 40, 30),
                               snapshot_path=path.relative_to(settings.data_dir).as_posix(),
                               details={'origin': {'edge_id': args.edge_id, 'run_id': engine.run_id}})
                    engine.journal.add(ev)
                    value = {'event_id': ev.id, 'snapshot_sha256': digest(image)}
                elif op == 'close':
                    image = synthetic_jpeg('blue')
                    path = settings.media_dir / 'source-blue.jpg'
                    path.write_bytes(image)
                    engine.journal.update(args.event_id, ended_at=110, snapshot_path=path.relative_to(settings.data_dir).as_posix())
                    value = {'snapshot_sha256': digest(image)}
                elif op == 'arm_confirm_crash':
                    publisher.gate.clear()
                    publisher.crash = True
                    value = {'armed': True}
                elif op == 'release':
                    respond({'released': True})
                    publisher.gate.set()
                    continue
                elif op == 'replay':
                    # Separate connection owned by this RPC thread, not the sender.
                    replay = RabbitPublisher(config)
                    try:
                        replay.publish(decode(Path(command['path']).read_bytes(), expected_edge_id=args.edge_id))
                    finally:
                        replay.close()
                    value = {'confirmed': True}
                elif op == 'arm_transient':
                    fault['transient'] = 1
                    value = {'armed': True}
                elif op == 'barrier':
                    engine.bus.publish({'kind': 'acceptance_barrier', 'id': command['id']})
                    value = {'barrier': command['id']}
                elif op == 'stop':
                    respond({'stopping': True})
                    break
                else:
                    raise ValueError('Unknown actor command')
                respond(value)
            except Exception as exc:
                respond({'error': type(exc).__name__})
    finally:
        if server:
            server.should_exit = True
            server_thread.join(timeout=10)
            check(not server_thread.is_alive(), 'HTTP server failed to stop')
            check(engine._closed, 'Gateway resources failed to close')
        else:
            engine.stop()
    return 0


class Actor:
    def __init__(self, mode, directory, edge_id, event_id, port=0, crash_commit=False):
        argv = [sys.executable, str(Path(__file__).resolve()), '--actor', mode, '--directory', str(directory),
                '--edge-id', edge_id, '--event-id', event_id, '--port', str(port)]
        if crash_commit:
            argv.append('--crash-commit')
        self.messages = queue.Queue()
        self.verified_exit = None
        self.log = (directory.parent / (mode + '-' + uuid4().hex[:8] + '.log')).open('wb')
        self.process = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
                                        text=True, env=os.environ.copy())
        def reader():
            for line in self.process.stdout:
                try:
                    self.messages.put(json.loads(line))
                except ValueError:
                    self.messages.put({'error': 'InvalidActorResponse'})
        self.reader = threading.Thread(target=reader, daemon=True)
        self.reader.start()
        if not crash_commit:
            try:
                check(self.response(15).get('ready'), 'Actor not ready')
            except BaseException:
                self.process.kill()
                self.process.wait(timeout=5)
                self.process.stdin.close()
                self.process.stdout.close()
                self.log.close()
                raise

    def response(self, timeout=10):
        try:
            result = self.messages.get(timeout=timeout)
        except queue.Empty:
            raise AcceptanceError('Actor response timeout') from None
        check('error' not in result, 'Actor operation failed: ' + str(result.get('error')))
        return result

    def rpc(self, op, **kwargs):
        check(self.process.poll() is None, 'Actor exited unexpectedly')
        self.process.stdin.write(json.dumps({'op': op, **kwargs}) + '\n')
        self.process.stdin.flush()
        return self.response()

    def crashed(self, expected):
        check(self.process.wait(timeout=15) == expected, 'Fault barrier exit code mismatch')
        self.verified_exit = expected

    def stop(self):
        try:
            if self.process.poll() is None:
                self.rpc('stop')
                self.process.wait(timeout=15)
                check(self.process.returncode == 0, 'Actor graceful stop failed')
                self.verified_exit = 0
            else:
                check(self.process.returncode == (self.verified_exit if self.verified_exit is not None else 0),
                      'Actor exited unexpectedly before cleanup')
        finally:
            if self.process.poll() is None:
                self.process.kill()
                self.process.wait(timeout=5)
            self.process.stdin.close()
            self.process.stdout.close()
            self.log.close()


@contextlib.contextmanager
def admin(url, edge_id):
    from guard.transport.rabbit import RabbitConfig, _connection, _topology
    config = RabbitConfig(url, edge_id)
    connection = _connection(config)
    try:
        channel = connection.channel()
        _topology(channel, config)
        yield channel, config, connection
    finally:
        if connection.is_open:
            connection.close()


def queue_count(url, edge_id):
    with admin(url, edge_id) as (channel, config, _):
        return channel.queue_declare(queue=config.event_queue, passive=True).method.message_count


def db_counts(path):
    if not path.exists():
        return None
    with sqlite3.connect(path.as_uri() + '?mode=ro', uri=True) as connection:
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        return {name: connection.execute('SELECT count(*) FROM ' + name).fetchone()[0]
                for name in ('events', 'transport_outbox', 'transport_receipts') if name in tables}


def http(base, path, method='GET', body=None):
    request = Request(base + path, method=method, data=json.dumps(body).encode() if body is not None else None,
                      headers={'Content-Type': 'application/json'})
    with urlopen(request, timeout=3) as response:
        data = response.read()
        return data if path.endswith('.jpg') else json.loads(data)


def websocket_barrier(actor, ws):
    marker = uuid4().hex
    actor.rpc('barrier', id=marker)
    result = []
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        ws.settimeout(max(0.01, deadline - time.monotonic()))
        item = json.loads(ws.recv())
        if item == {'kind': 'acceptance_barrier', 'id': marker}:
            return result
        if item.get('kind') != 'metrics':
            result.append(item)
    raise AcceptanceError('WebSocket barrier timeout')


def wait_event_ws(ws, kind):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        ws.settimeout(max(0.01, deadline - time.monotonic()))
        item = json.loads(ws.recv())
        if item.get('kind') == kind:
            return item
    raise AcceptanceError('Expected WebSocket event missing')


def run_acceptance(args):
    import pika
    import websocket
    from PIL import Image
    url = os.environ.get('GUARD_RABBIT_URL')
    check(bool(url), 'GUARD_RABBIT_URL is required; no broker means unavailable')
    host, port = broker_address(url)
    control = BrokerControl(args.broker_control, url) if args.broker_control else None
    directory = args.output.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    edge_id, event_id = 'accept-' + uuid4().hex, 'event-' + uuid4().hex
    edge_dir, gateway_dir = directory / 'edge', directory / 'gateway'
    edge_db, gateway_db = edge_dir / 'data' / 'guard.db', gateway_dir / 'data' / 'guard.db'
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        api_port = sock.getsockname()[1]
    base = f'http://127.0.0.1:{api_port}'
    report = {'schema': 'guard-rabbit-acceptance-v1', 'scope': 'broker-and-process-crash' if control else 'process-crash-only',
              'status': 'running', 'edge_id': edge_id, 'event_id': event_id, 'ports': {'amqp': port, 'http': api_port},
              'versions': {'python': sys.version.split()[0], 'pika': pika.__version__}, 'phases': {}, 'processes': [],
              'git_sha': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()}
    actors, sockets = [], []
    previous_alarm = signal.getsignal(signal.SIGALRM)
    def deadline_expired(*_):
        raise AcceptanceError('Global acceptance deadline exceeded')
    signal.signal(signal.SIGALRM, deadline_expired)
    signal.setitimer(signal.ITIMER_REAL, args.deadline)
    def spawn(mode, **kw):
        actor = Actor(mode, edge_dir if mode == 'edge' else gateway_dir, edge_id, event_id, api_port, **kw)
        actors.append(actor)
        report['processes'].append({'role': mode, 'pid': actor.process.pid, 'crash_commit': bool(kw.get('crash_commit'))})
        return actor
    def phase(name):
        report['active_phase'] = name
        print(json.dumps({'phase': name}), flush=True)
    def ready_gateway(actor):
        wait_for(lambda: actor.rpc('status')['health']['transport']['connected'], args.timeout, 'gateway AMQP ready')
        http(base, '/api/events')
    def connect_ws(actor):
        ws = websocket.create_connection(f'ws://127.0.0.1:{api_port}/ws/events', timeout=5, http_proxy_host=None)
        sockets.append(ws)
        wait_for(lambda: actor.rpc('status')['subscribers'] > 0, 5, 'WS subscribed')
        return ws
    try:
        phase('initial_connection')
        try:
            with admin(url, edge_id) as (_, _, connection):
                report['versions']['rabbitmq'] = str(connection._impl.server_properties.get('version', 'unknown'))
        except Exception as exc:
            report['connection_error_type'] = type(exc).__name__
            if hasattr(exc, 'reply_code'):
                report['connection_reply_code'] = exc.reply_code
            raise AcceptanceUnavailable('Loopback broker did not complete AMQP handshake') from None
        phase('http_ws_snapshot_operator')
        gateway, edge = spawn('gateway'), spawn('edge')
        ready_gateway(gateway)
        ws = connect_ws(gateway)
        first = edge.rpc('add')
        created = wait_event_ws(ws, 'event')['event']
        check(created['id'] == event_id, 'Wrong new event')
        wait_for(lambda: db_counts(edge_db)['transport_outbox'] == 0, args.timeout, 'initial publisher confirm')
        page = http(base, '/api/events')
        check(page['total'] == 1, 'Expected exactly one HTTP event')
        image = http(base, '/api/events/' + event_id + '/snapshot.jpg')
        check(digest(image) == first['snapshot_sha256'], 'Original JPEG hash mismatch')
        with Image.open(io.BytesIO(image)) as decoded:
            check(decoded.format == 'JPEG' and decoded.size == (48, 32), 'Snapshot decode mismatch')
        with sqlite3.connect(edge_db.as_uri() + '?mode=ro', uri=True) as connection:
            check(connection.execute('SELECT count(*) FROM events').fetchone()[0] == 1, 'Edge row count')
        first_confirm = json.loads((edge_dir / 'last-confirm.json').read_text())
        old_wire = edge_dir / ('wire-' + first_confirm['sha256'] + '.json')
        patch = {'status': 'confirmed', 'operator_note': 'Acceptance operator decision'}
        http(base, '/api/events/' + event_id, 'PATCH', patch)
        updated = wait_event_ws(ws, 'event_update')['event']
        check(all(updated[key] == value for key, value in patch.items()), 'Operator WS update missing')
        report['phases']['http_ws_snapshot_operator'] = {'status': 'passed', 'snapshot_sha256': digest(image)}
        # Stop the real consumer before publishing the closure, proving confirms
        # are independent of gateway ACK and that the broker holds durable work.
        ws.close()
        gateway.stop()
        edge.rpc('arm_confirm_crash')
        phase('broker_offline_outbox')
        if control:
            control.run('stop')
        second = edge.rpc('close')
        edge.rpc('release')
        if control:
            wait_for(lambda: edge.rpc('status')['health']['transport']['last_error'], args.timeout, 'offline sender error visible')
            check(db_counts(edge_db)['transport_outbox'] == 1, 'Offline outbox was lost')
            report['phases']['broker_offline_outbox'] = {'status': 'passed', 'pending': 1}
            control.run('start')
        else:
            report['phases']['broker_offline_outbox'] = {'status': 'not_verified', 'reason': 'No explicit owned broker control'}
        edge.crashed(72)
        phase('edge_crash_after_confirm')
        check(db_counts(edge_db)['transport_outbox'] == 1, 'Crash-before-mark must retain outbox')
        confirmed = json.loads((edge_dir / 'last-confirm.json').read_text())
        with sqlite3.connect(edge_db.as_uri() + '?mode=ro', uri=True) as connection:
            pending_wire = bytes(connection.execute('SELECT envelope FROM transport_outbox').fetchone()[0])
        check(digest(pending_wire) == confirmed['sha256'], 'Confirmed wire differs from frozen outbox')
        wait_for(lambda: queue_count(url, edge_id) == 1, args.timeout, 'durable closure queued without consumer')
        report['phases']['edge_crash_after_confirm'] = {'status': 'passed', 'pending': 1, 'wire_sha256': confirmed['sha256']}
        if control:
            phase('confirmed_queue_survives_broker_restart')
            control.run('stop')
            control.run('start')
            check(queue_count(url, edge_id) == 1, 'Confirmed event did not survive broker restart')
            report['phases']['confirmed_queue_survives_broker_restart'] = {'status': 'passed', 'ready_messages': 1}
        else:
            report['phases']['confirmed_queue_survives_broker_restart'] = {'status': 'not_verified', 'reason': 'No explicit owned broker control'}
        phase('gateway_crash_after_commit_before_ack')
        gateway = spawn('gateway', crash_commit=True)
        gateway.crashed(73)
        committed = json.loads((gateway_dir / 'commit-before-ack.json').read_text())
        check(committed['changed'] and not committed['created'], 'Closure commit barrier not reached')
        check(db_counts(gateway_db)['events'] == 1 and db_counts(gateway_db)['transport_receipts'] == 2, 'Gateway receipt was not committed')
        wait_for(lambda: queue_count(url, edge_id) == 1, args.timeout, 'unacked delivery requeued after process crash')
        report['phases']['gateway_crash_after_commit_before_ack'] = {'status': 'passed', 'events': 1, 'receipts': 2}
        phase('restart_exact_replay_no_duplicates')
        gateway = spawn('gateway')
        ready_gateway(gateway)
        ws = connect_ws(gateway)
        wait_for(lambda: queue_count(url, edge_id) == 0, args.timeout, 'redelivery ACK')
        check(websocket_barrier(gateway, ws) == [], 'Duplicate commit produced a second WS change')
        edge = spawn('edge')
        wait_for(lambda: db_counts(edge_db)['transport_outbox'] == 0, args.timeout, 'edge restart replay confirm')
        replay_confirm = json.loads((edge_dir / 'last-confirm.json').read_text())
        check(replay_confirm == confirmed, 'Restart replay changed frozen wire identity or JPEG')
        edge.rpc('replay', path=str(old_wire))
        wait_for(lambda: queue_count(url, edge_id) == 0, args.timeout, 'old revision ACK')
        # Telemetry is deliberately excluded: generic adapter.received counts
        # cameras/metrics too and cannot establish a durable replay barrier.
        def exact_replays_completed():
            outcomes = gateway.rpc('status')['durable_callbacks']
            return (outcomes.get(confirmed['message_id'], {}).get('completed', 0) >= 2
                    and outcomes.get(first_confirm['message_id'], {}).get('completed', 0) >= 1)
        wait_for(exact_replays_completed, args.timeout, 'exact durable replay callbacks completed')
        check(websocket_barrier(gateway, ws) == [], 'Replay emitted a duplicate WS notification')
        current = http(base, '/api/events')['items'][0]
        check(current['ended_at'] == 110 and all(current[key] == value for key, value in patch.items()), 'Replay rolled back closure/operator decision')
        image = http(base, '/api/events/' + event_id + '/snapshot.jpg')
        check(digest(image) == second['snapshot_sha256'], 'Old replay downgraded latest JPEG')
        check(current['clip_path'] is None, 'Edge clip must not be exposed on gateway')
        report['phases']['restart_exact_replay_no_duplicates'] = {'status': 'passed', 'events': 1, 'snapshot_sha256': digest(image)}
        # Actual basic_reject for an unsupported wire version, then actual NACK
        # and reconnect for a transient callback/storage error.
        phase('reject_and_nack_recovery')
        gateway.rpc('arm_transient')
        before = gateway.rpc('status')['health']['transport']
        with admin(url, edge_id) as (channel, config, _):
            channel.confirm_delivery()
            channel.basic_publish(exchange=config.exchange, routing_key=config.routing_key(True), body=b'{"version":999}',
                                  properties=pika.BasicProperties(delivery_mode=2), mandatory=True)
        wait_for(lambda: gateway.rpc('status')['health']['transport']['rejected'] == before['rejected'] + 1, args.timeout, 'poison message rejected')
        edge.rpc('replay', path=str(old_wire))
        wait_for(lambda: (gateway_dir / 'transient-nack.json').exists(), args.timeout, 'transient NACK barrier')
        wait_for(lambda: gateway.rpc('status')['health']['transport']['last_error'] == 'OperationalError', args.timeout, 'transient consumer error visible')
        wait_for(lambda: gateway.rpc('status')['durable_callbacks'].get(first_confirm['message_id'], {}).get('completed', 0) == 2,
                 args.timeout, 'exact transient durable callback recovery')
        check(websocket_barrier(gateway, ws) == [], 'Transient replay emitted duplicate WS')
        check(http(base, '/api/events')['total'] == 1 and db_counts(gateway_db)['transport_receipts'] == 2, 'Final event/receipt count mismatch')
        report['phases']['reject_and_nack_recovery'] = {'status': 'passed', 'rejected': 1, 'transient_recovered': True}
        report['durable_callbacks'] = gateway.rpc('status')['durable_callbacks']
        check(all(item['changed'] == 0 for item in report['durable_callbacks'].values()), 'Replay callback unexpectedly changed committed state')
        report['readback'] = {'edge': db_counts(edge_db), 'gateway': db_counts(gateway_db), 'status': current['status'],
                              'operator_note_preserved': current['operator_note'] == patch['operator_note'], 'ended_at': current['ended_at']}
        report['status'] = 'passed' if control else 'passed_with_unverified_phases'
        return report
    except Exception as exc:
        report['status'] = 'unavailable' if isinstance(exc, AcceptanceUnavailable) else 'failed'
        # Types and a fixed phase label are safe; raw network exceptions are not.
        report['failure_type'] = type(exc).__name__
        if isinstance(exc, AcceptanceError):
            report['failure'] = str(exc)
        raise
    finally:
        # Cleanup has its own bounded actor/control waits and must still run
        # after the work deadline, so it cannot leave an owned broker offline.
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_alarm)
        for ws in sockets:
            with contextlib.suppress(Exception):
                ws.close()
        errors = []
        for actor in reversed(actors):
            try:
                actor.stop()
            except Exception as exc:
                errors.append(type(exc).__name__)
        if control and control.stopped:
            try:
                control.run('start')
            except Exception as exc:
                errors.append(type(exc).__name__)
        try:
            with admin(url, edge_id) as (channel, config, _):
                channel.queue_delete(queue=config.event_queue)
                channel.queue_delete(queue=config.telemetry_queue)
            report['cleanup'] = {'queues_deleted': True, 'shared_exchange_retained': True}
        except Exception as exc:
            errors.append(type(exc).__name__)
            report['cleanup'] = {'queues_deleted': False}
        if control and not control.stopped:
            try:
                control.run('stop')
            except Exception as exc:
                errors.append(type(exc).__name__)
        if control:
            report['cleanup']['broker_stopped'] = control.stopped
        if errors:
            report['cleanup']['errors'] = errors
            report['status'] = 'failed'
        write_json(directory / 'report.json', report)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--run', action='store_true', help='explicit opt-in to real local broker/process fault tests')
    result.add_argument('--output', type=Path, default=ROOT / 'data' / ('rabbit-acceptance-' + uuid4().hex[:12]))
    result.add_argument('--broker-control', type=Path, help='explicit owned loopback broker start/stop JSON; enables broker restart checks')
    result.add_argument('--timeout', type=float, default=20)
    result.add_argument('--deadline', type=float, default=600, help='global work deadline in seconds; cleanup has separate bounded waits (POSIX)')
    result.add_argument('--actor', choices=['edge', 'gateway'], help=argparse.SUPPRESS)
    result.add_argument('--directory', type=Path, help=argparse.SUPPRESS)
    result.add_argument('--edge-id', help=argparse.SUPPRESS)
    result.add_argument('--event-id', help=argparse.SUPPRESS)
    result.add_argument('--port', type=int, default=0, help=argparse.SUPPRESS)
    result.add_argument('--crash-commit', action='store_true', help=argparse.SUPPRESS)
    return result


def main(argv=None):
    logging.disable(logging.CRITICAL)
    args = parser().parse_args(argv)
    try:
        check(0 < args.timeout <= 120, 'timeout must be in (0, 120]')
        check(0 < args.deadline <= 1800, 'deadline must be in (0, 1800]')
        if args.actor:
            return run_actor(args)
        check(args.run, 'Explicit --run is required; acceptance was not executed')
        report = run_acceptance(args)
        print(json.dumps({'status': report['status'], 'scope': report['scope'], 'report': str(args.output / 'report.json')}))
        return 0 if report['status'] in {'passed', 'passed_with_unverified_phases'} else 1
    except Exception as exc:
        print(json.dumps({'status': 'unavailable' if isinstance(exc, AcceptanceUnavailable) or not os.environ.get('GUARD_RABBIT_URL') else 'failed',
                          'error_type': type(exc).__name__}), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
