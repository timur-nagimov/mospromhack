// Ролевая модель интерфейса: нормализация ответов /api/auth/*, стартовая страница,
// мок-сервер с проверкой прав и журналом действий. Запуск: npm run test:auth (Node 20).
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import test from 'node:test';
import ts from 'typescript';

const require = createRequire(import.meta.url);
for (const extension of ['.ts', '.tsx']) {
  require.extensions[extension] = (module, filename) => {
    const source = readFileSync(filename, 'utf8');
    const result = ts.transpileModule(source, {
      fileName: filename,
      compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020, jsx: ts.JsxEmit.ReactJSX, esModuleInterop: true },
    });
    module._compile(result.outputText, filename);
  };
}

// Мок отвечает через window.setTimeout: в тесте — сразу
globalThis.window = { setTimeout: (callback) => { callback(); return 1; } };

const auth = require('../src/domain/auth.ts');
const { authSignals } = require('../src/api/authSignals.ts');
const { ApiError } = require('../src/api/types.ts');
const { sectionFor, startPath, safeNextPath } = require('../src/components/layout/sections.ts');
const { createMockApi } = require('../src/api/mock/server.ts');

const rejects = async (promise, status, permission) => {
  await assert.rejects(promise, (err) => {
    assert.ok(err instanceof ApiError, `ожидали ApiError, получили ${err}`);
    assert.equal(err.status, status);
    if (permission) assert.equal(err.permission, permission);
    return true;
  });
};

test('матрица по guard/auth/roles.py: вложенность ролей и права сотрудников', () => {
  const { ROLE_PERMISSIONS: R, ALL_PERMISSIONS } = auth;
  const sub = (a, b) => a.every((p) => b.includes(p));
  assert.equal(ALL_PERMISSIONS.length, 15);
  assert.deepEqual(R.worker, ['analytics.view', 'events.view', 'employees.view']);
  assert.ok(sub(R.worker, R.operator));
  // виновников записывают диспетчер и администратор, инженер по ОТ — нет
  assert.ok(R.operator.includes('events.assign') && !R.safety.includes('events.assign'));
  assert.ok(sub(R.operator.filter((p) => p !== 'events.assign'), R.safety));
  assert.ok(R.safety.includes('zones.edit') && !R.safety.includes('cameras.manage'));
  assert.ok(!R.operator.includes('zones.edit') && !R.safety.includes('employees.manage'));
  assert.deepEqual([...R.admin].sort(), [...ALL_PERMISSIONS].sort());
  assert.equal(auth.needRoleText('events.assign', auth.fallbackRolesCatalog().roles), 'Нужна роль: Диспетчер смены или Администратор');
  assert.equal(auth.needRoleText('zones.edit', auth.fallbackRolesCatalog().roles), 'Нужна роль: Инженер по ОТ или Администратор');
  assert.equal(auth.localAccessUser('worker').permissions.includes('video.view'), false);
  assert.equal(auth.localAccessUser('admin', true).permissions.includes('users.manage'), false);
});

test('каталог ролей: массив, {roles, permissions}, словарь по id', () => {
  const asArray = auth.normalizeRolesCatalog([
    { id: 'admin', title: 'Администратор', permissions: ['users.manage', 'analytics.view'] },
    { id: 'worker', title: 'Сотрудник', description: 'смотрит', permissions: ['analytics.view'] },
  ]);
  assert.deepEqual(asArray.roles.map((r) => r.id), ['worker', 'admin']);
  assert.deepEqual(asArray.roles[1].permissions, ['analytics.view', 'users.manage']);
  assert.deepEqual(asArray.permissions.map((p) => p.id), ['analytics.view', 'users.manage']);
  assert.equal(asArray.permissions[0].title, 'Аналитика');

  const asObject = auth.normalizeRolesCatalog({
    roles: { operator: { title: 'Диспетчер', permissions: [{ id: 'video.view' }] }, worker: ['events.view'] },
    permissions: { 'video.view': { title: 'Видео', description: 'кадры' }, 'custom.perm': 'Своё право' },
  });
  assert.deepEqual(asObject.roles.map((r) => [r.id, r.title]), [['worker', 'Сотрудник'], ['operator', 'Диспетчер']]);
  assert.deepEqual(asObject.permissions.map((p) => p.id), ['events.view', 'video.view', 'custom.perm']);
  assert.equal(asObject.permissions.find((p) => p.id === 'custom.perm').title, 'Своё право');
  assert.deepEqual(auth.normalizeRolesCatalog(null), { roles: [], permissions: [] });
});

test('config, me, пользователи и журнал: поля по контракту и терпимость к форме', () => {
  const config = auth.normalizeConfig({
    auth_enabled: true,
    demo_login: true,
    demo_users: [{ username: 'admin', display_name: 'A', role: 'admin' }, { username: 'worker', role: 'worker' }],
  });
  assert.deepEqual(config.demo_users.map((u) => u.username), ['worker', 'admin']);
  assert.equal(auth.normalizeConfig({ demo_login: false }).auth_enabled, true);
  assert.equal(auth.normalizeConfig({ auth_enabled: false }).auth_enabled, false);

  const me = auth.normalizeMe({ username: 'safety', role: 'safety', permissions: ['zones.edit'] });
  assert.equal(me.display_name, 'safety');
  assert.equal(me.role_title, 'Инженер по ОТ');

  const user = auth.normalizeUser({ username: 'u', role: 'worker', active: 0, created_at: '2026-10-24T09:00:00Z', last_login_at: 1761296400000 });
  assert.equal(user.active, false);
  assert.equal(user.created_at, Date.parse('2026-10-24T09:00:00Z') / 1000);
  assert.equal(user.last_login_at, 1761296400);

  const page = auth.normalizeAuditPage({ items: [{ ts: '1761296400', action: 'auth.login', details: '{"a":1}' }], total: 9 });
  assert.equal(page.total, 9);
  assert.equal(page.items[0].ts, 1761296400);
  assert.deepEqual(page.items[0].details, { a: 1 });
  assert.equal(auth.normalizeAuditPage([{ action: 'x' }]).total, 1);
});

test('стартовая страница: Мониторинг, иначе Аналитика; разделы и ?next=', () => {
  const has = (perms) => (p) => perms.includes(p);
  assert.equal(startPath(has(auth.ROLE_PERMISSIONS.operator)), '/');
  assert.equal(startPath(has(auth.ROLE_PERMISSIONS.worker)), '/analytics');
  assert.equal(startPath(has(['events.view'])), '/journal');
  assert.equal(startPath(has([])), null);
  assert.equal(sectionFor('/cameras/cam-01').permission, 'cameras.view');
  assert.equal(sectionFor('/').permission, 'video.view');
  assert.equal(sectionFor('/users').permission, 'users.manage');
  assert.equal(sectionFor('/nope'), undefined);
  assert.equal(safeNextPath('/cameras/cam-01?x=1'), '/cameras/cam-01?x=1');
  for (const bad of [null, '', 'https://evil', '//evil', '/login?next=/'] ) assert.equal(safeNextPath(bad), null);
  assert.equal(auth.forbiddenText('zones.edit'), 'Недостаточно прав: Зоны и правила (zones.edit)');
  assert.equal(auth.forbiddenText('x.y'), 'Недостаточно прав: x.y');
});

test('сигналы: 401 на вход не уводит на /login, 403 называет право', () => {
  const seen = [];
  const stop = authSignals.listen({ unauthorized: () => seen.push('401'), forbidden: (p) => seen.push(p) });
  authSignals.report(new ApiError(401, 'x'), '/api/auth/login');
  authSignals.report(new ApiError(401, 'x'), '/api/events');
  authSignals.report(new ApiError(403, 'x', 'video.view'), '/api/cameras/c/frame.jpg');
  authSignals.report(new ApiError(403, 'обучение выключено'), '/api/flywheel/jobs');
  stop();
  authSignals.report(new ApiError(401, 'x'), '/api/events');
  assert.deepEqual(seen, ['401', 'video.view']);
});

test('мок: без входа 401, вход демо-учёткой, права роли и 403 с названием права', async () => {
  const api = createMockApi({ auth: true, demoLogin: true });
  const config = await api.auth.config();
  assert.equal(config.auth_enabled, true);
  assert.deepEqual(config.demo_users.map((u) => u.role), ['worker', 'operator', 'safety', 'admin']);
  await rejects(api.listEvents({ limit: 1 }), 401);
  await rejects(api.auth.me(), 401);
  await rejects(api.auth.login('worker', 'nope'), 401);

  const me = await api.auth.login('worker', 'worker');
  assert.equal(me.role, 'worker');
  assert.deepEqual(me.permissions, ['analytics.view', 'events.view', 'employees.view']);
  assert.ok((await api.listEvents({ limit: 1 })).total > 0);
  assert.ok((await api.getAnalytics({})).types.length > 0);
  await rejects(api.listCameras(), 403, 'cameras.view');
  await rejects(api.getMetrics(), 403, 'system.view');
  const [event] = (await api.listEvents({ limit: 1 })).items;
  await rejects(api.patchEvent(event.id, { status: 'confirmed' }), 403, 'events.handle');
  await rejects(api.flywheel.summary(), 403, 'flywheel.review');
  await rejects(api.audit.list({}), 403, 'audit.view');
  await rejects(api.users.list(), 403, 'users.manage');
  await api.auth.logout();
  await rejects(api.listEvents({ limit: 1 }), 401);
});

test('мок: инженер по ОТ меняет зоны и правила, но не источник камеры', async () => {
  const api = createMockApi({ auth: true });
  await api.auth.login('safety', 'safety');
  const [camera] = await api.listCameras();
  const zonesOnly = structuredClone(camera);
  zonesOnly.zones[0].name = 'Проход у стеллажей (новая граница)';
  await api.updateCamera(camera.camera_id, zonesOnly);
  const source = structuredClone(zonesOnly);
  source.source.uri = 'rtsp://10.0.0.9/stream';
  await rejects(api.updateCamera(camera.camera_id, source), 403, 'cameras.manage');
  await rejects(api.createCamera({ ...camera, camera_id: 'cam-99' }), 403, 'cameras.manage');
  await rejects(api.deleteCamera(camera.camera_id), 403, 'cameras.manage');
  await rejects(api.flywheel.startJob('train', {}), 403, 'flywheel.train');
  assert.ok((await api.flywheel.summary()) != null);
  const [event] = (await api.listEvents({ limit: 1 })).items;
  assert.ok((await api.employees.list()).length > 0);
  await rejects(api.employees.assign(event.id, []), 403, 'events.assign');
  await rejects(api.employees.create({ full_name: 'Тест' }), 403, 'employees.manage');
  await rejects(api.probeSource(camera.source), 403, 'cameras.manage');

  const audit = await api.audit.list({ limit: 200 });
  const mine = audit.items.filter((a) => a.username === 'safety');
  assert.equal(mine[0].action, 'zones.update');
  assert.equal(mine[0].target, camera.camera_id);
  assert.equal(mine[1].action, 'auth.login');
  assert.ok(audit.items.some((a) => a.action === 'auth.login_failed'), 'в истории есть неудачный вход');
  const filtered = await api.audit.list({ action: 'zones.update', username: 'safety' });
  assert.ok(filtered.items.every((a) => a.action === 'zones.update' && a.username === 'safety'));
});

test('мок: администратор ведёт учётные записи, последнего администратора не понизить', async () => {
  const api = createMockApi({ auth: true });
  await api.auth.login('admin', 'admin');
  const users = await api.users.list();
  assert.ok(users.some((u) => u.username === 'sidorov' && !u.active));
  const created = await api.users.create({ username: 'ivanov', display_name: 'Иван Иванов', role: 'operator', password: 'secret-123' });
  assert.equal(created.role, 'operator');
  await rejects(api.users.create({ username: 'ivanov', display_name: 'x', role: 'worker', password: 'secret-123' }), 409);
  await rejects(api.users.create({ username: 'Bad Name', display_name: 'x', role: 'worker', password: 'secret-123' }), 422);
  assert.equal((await api.users.update('ivanov', { role: 'safety' })).role, 'safety');
  await rejects(api.users.update('admin', { role: 'worker' }), 409);
  await rejects(api.users.update('admin', { active: false }), 409);
  const log = await api.audit.list({ username: 'admin', limit: 5 });
  assert.deepEqual(log.items.slice(0, 2).map((a) => a.action), ['user.update', 'user.create']);
  assert.equal(JSON.stringify(log.items).includes('secret-123'), false, 'пароль не попадает в журнал');

  // отключённый пользователь войти не может
  await api.auth.logout();
  await api.auth.login('ivanov', 'secret-123');
  await api.auth.logout();
  await api.auth.login('admin', 'admin');
  await api.users.update('ivanov', { active: false });
  await rejects(api.auth.login('ivanov', 'secret-123'), 401);
});

test('мок с выключенным входом (GUARD_AUTH=0): всё как раньше, роль admin', async () => {
  const api = createMockApi({ auth: false });
  const config = await api.auth.config();
  assert.equal(config.auth_enabled, false);
  assert.deepEqual(config.demo_users, []);
  assert.equal((await api.auth.me()).role, 'admin');
  assert.ok((await api.listCameras()).length > 0);
  assert.ok((await api.users.list()).length >= 4);
});
