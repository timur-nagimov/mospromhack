import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import test from 'node:test';
import ts from 'typescript';

// Проверяем настоящие TS-модули без дополнительного раннера и зависимостей.
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
const { analyticsRange, criticalCount, fillDays, statusByType, sumStatuses, weekdayHourMatrix } =
  require('../src/features/analytics/aggregate.ts');
const { moscowDayKey, moscowHour, moscowWeekdayHour } = require('../src/domain/moscow.ts');
const epoch = (value) => Date.parse(value) / 1000;

test('Выбранный сотрудник: все агрегаты, список и CSV содержат только его подтверждённые нарушения', async () => {
  const { createMockApi } = require('../src/api/mock/server.ts');
  const previous = globalThis.window;
  globalThis.window = { setTimeout: (callback) => { callback(); return 1; } };
  try {
    const api = createMockApi({ auth: false }); // вход выключен, как GUARD_AUTH=0: роли здесь не проверяются
    const seeded = await api.employees.list();
    assert.equal(seeded.length, 10);
    assert.equal(new Set(seeded.map((employee) => employee.personnel_number)).size, 10);
    assert.ok(seeded.every((employee) => employee.personnel_number.startsWith('DEMO-')));
    const first = await api.employees.create({ full_name: 'Первый сотрудник' });
    const second = await api.employees.create({ full_name: 'Другой сотрудник' });
    const page = await api.listEvents({ limit: 100000 });
    const [confirmed, falseAlarm, pending, forwarded, other] = page.items;
    for (const [event, status] of [[confirmed, 'confirmed'], [falseAlarm, 'false'], [pending, 'new'], [forwarded, 'forwarded']]) {
      await api.patchEvent(event.id, { status });
      await api.employees.assign(event.id, [first.id]);
    }
    await api.employees.assign(confirmed.id, [first.id, second.id]);
    await api.patchEvent(other.id, { status: 'confirmed' });
    await api.employees.assign(other.id, [second.id]);
    const analytics = await api.getAnalytics({ employee_id: first.id });
    for (const group of ['types', 'sites', 'cameras', 'days', 'weekdayHours'])
      assert.equal(analytics[group].reduce((sum, row) => sum + row.count, 0), 1);
    assert.deepEqual(analytics.statusTypes.new, []);
    assert.deepEqual(analytics.statusTypes.false, []);
    assert.deepEqual(analytics.statusTypes.forwarded, []);
    assert.deepEqual(analytics.statusTypes.confirmed, analytics.types);
    const report = await api.employees.report(first.id, {});
    assert.deepEqual(report.items.map((event) => event.id), [confirmed.id]);
    const csv = await api.employees.exportReport(first.id, {});
    assert.ok(csv.includes(confirmed.id));
    for (const event of [falseAlarm, pending, forwarded, other]) assert.ok(!csv.includes(event.id));
    await api.patchEvent(confirmed.id, { status: 'false' });
    assert.deepEqual((await api.getAnalytics({ employee_id: first.id })).types, []);
    assert.equal((await api.employees.report(first.id, {})).total, 0);
    assert.equal((await api.getAnalytics({})).types.reduce((sum, row) => sum + row.count, 0), page.total);
  } finally { globalThis.window = previous; }
});

test('МСК и границы периода не зависят от часового пояса браузера', () => {
  const previous = process.env.TZ;
  try {
    for (const tz of ['UTC', 'America/Los_Angeles', 'Asia/Tokyo']) {
      process.env.TZ = tz;
      const now = epoch('2026-10-18T21:05:12Z');
      assert.equal(moscowDayKey(now), '2026-10-19');
      assert.equal(moscowHour(now), 0);
      assert.equal(moscowWeekdayHour(now), '0:0');
      assert.deepEqual(analyticsRange('today', now), {
        from: epoch('2026-10-18T21:00:00Z'), to: epoch('2026-10-18T21:05:00Z'),
      });
      assert.equal(analyticsRange('7d', now).from, epoch('2026-10-12T21:00:00Z'));
      assert.equal(analyticsRange('30d', now).from, epoch('2026-09-19T21:00:00Z'));
    }
  } finally {
    if (previous == null) delete process.env.TZ;
    else process.env.TZ = previous;
  }
});

test('Тепловая карта сохраняет все 5205 событий из агрегатов', () => {
  const matrix = weekdayHourMatrix([{ key: '0:0', count: 3600 }, { key: '0:1', count: 1605 }]);
  assert.equal(matrix.length, 7);
  assert.equal(matrix[0].length, 24);
  assert.equal(matrix[0][0], 3600);
  assert.equal(matrix.flat().reduce((a, b) => a + b), 5205);
});

test('Объединение critical не считает падения и технику дважды', () => {
  assert.equal(criticalCount(
    [{ key: 'ppe_missing', count: 1735 }, { key: 'fall_immobile', count: 1735 }, { key: 'vehicle_proximity', count: 1735 }],
    [{ key: 'ppe_missing', count: 347 }, { key: 'fall_immobile', count: 347 }, { key: 'vehicle_proximity', count: 347 }],
  ), 3817);
});

test('Статусы и знаменатель точности восстанавливаются из полных агрегатов', () => {
  const rows = statusByType({
    new: [{ key: 'ppe_missing', count: 1302 }],
    confirmed: [{ key: 'ppe_missing', count: 1301 }],
    forwarded: [{ key: 'ppe_missing', count: 1301 }],
    false: [{ key: 'ppe_missing', count: 1301 }],
  });
  assert.equal(rows[0].total, 5205);
  assert.equal(rows[0].reviewed, 3903);
  assert.equal(rows[0].falseRate, 1 / 3);
  assert.deepEqual(sumStatuses(rows), { new: 1302, confirmed: 1301, forwarded: 1301, false: 1301 });
});

test('Дни за всё время не обрезаются до первых 400 и не смещаются при смене TZ', () => {
  const result = fillDays([{ key: '2025-01-01', count: 2 }, { key: '2026-10-19', count: 3 }], undefined, epoch('2026-10-18T21:05:00Z'));
  assert.equal(result.length, 657);
  assert.deepEqual(result[0], { key: '2025-01-01', count: 2 });
  assert.deepEqual(result.at(-1), { key: '2026-10-19', count: 3 });
  assert.equal(result.reduce((sum, day) => sum + day.count, 0), 5);
});

test('Mock поддерживает новые группировки и фильтры в том же календаре МСК', async () => {
  const { createMockApi } = require('../src/api/mock/server.ts');
  const previous = globalThis.window;
  globalThis.window = { setTimeout: (callback) => { callback(); return 1; } };
  try {
    const api = createMockApi({ auth: false }); // вход выключен, как GUARD_AUTH=0: роли здесь не проверяются
    const page = await api.listEvents({ limit: 100000 });
    for (const [group, keyFor] of [
      ['site', (event) => event.site],
      ['weekday_hour', (event) => moscowWeekdayHour(event.confirmed_at)],
      ['day', (event) => moscowDayKey(event.confirmed_at)],
    ]) {
      const expected = new Map();
      for (const event of page.items) {
        const key = keyFor(event);
        expected.set(key, (expected.get(key) ?? 0) + 1);
      }
      const result = await api.getStats({ group_by: group });
      assert.deepEqual(new Map(result.items.map((item) => [item.key, item.count])), expected);
    }
    const result = await api.getStats({ group_by: 'type', status: 'confirmed,false', severity: 'critical' });
    const expected = page.items.filter((event) => ['confirmed', 'false'].includes(event.status) && event.severity === 'critical').length;
    assert.equal(result.items.reduce((sum, item) => sum + item.count, 0), expected);

    const period = { from: page.items[0].confirmed_at - 86400, to: page.items[0].confirmed_at };
    const summary = await api.getAnalytics(period);
    for (const [field, group] of [
      ['types', 'type'], ['cameras', 'camera'], ['sites', 'site'], ['days', 'day'], ['weekdayHours', 'weekday_hour'],
    ]) {
      assert.deepEqual(summary[field], (await api.getStats({ ...period, group_by: group })).items);
    }
    assert.deepEqual(summary.criticalTypes, (await api.getStats({ ...period, group_by: 'type', severity: 'critical' })).items);
    for (const status of ['new', 'confirmed', 'false', 'forwarded']) {
      assert.deepEqual(summary.statusTypes[status], (await api.getStats({ ...period, group_by: 'type', status })).items);
    }
  } finally {
    globalThis.window = previous;
  }
});

test('Mock summary сохраняет снимок при PATCH до доставки ответа', async () => {
  const { createMockApi } = require('../src/api/mock/server.ts');
  const previous = globalThis.window;
  const callbacks = [];
  globalThis.window = { setTimeout: (callback) => { callbacks.push(callback); return callbacks.length; } };
  const flush = () => { while (callbacks.length) callbacks.shift()(); };
  const total = (items) => items.reduce((sum, item) => sum + item.count, 0);
  try {
    const api = createMockApi({ auth: false }); // вход выключен, как GUARD_AUTH=0: роли здесь не проверяются
    const eventPromise = api.listEvents({ status: 'new', limit: 1 });
    flush();
    const event = (await eventPromise).items[0];
    assert.ok(event);
    const period = { from: event.confirmed_at, to: event.confirmed_at };
    const snapshotPromise = api.getAnalytics(period);
    const patchPromise = api.patchEvent(event.id, { status: 'confirmed' });
    flush();
    const [snapshot] = await Promise.all([snapshotPromise, patchPromise]);
    const freshPromise = api.getAnalytics(period);
    flush();
    const fresh = await freshPromise;
    assert.equal(total(snapshot.types), total(fresh.types));
    assert.equal(total(snapshot.statusTypes.new), total(fresh.statusTypes.new) + 1);
    assert.equal(total(snapshot.statusTypes.confirmed) + 1, total(fresh.statusTypes.confirmed));
    for (const response of [snapshot, fresh]) {
      assert.equal(Object.values(response.statusTypes).reduce((sum, items) => sum + total(items), 0), total(response.types));
    }
  } finally {
    globalThis.window = previous;
  }
});
