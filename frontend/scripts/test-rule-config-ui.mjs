// Run through tools/heavy.sh: actual rendered components, isolated temporary browser profile.
import assert from 'node:assert/strict';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, resolve } from 'node:path';
import { realpathSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import test, { after } from 'node:test';
import { chromium } from 'playwright';
import { createServer } from 'vite';
import react from '@vitejs/plugin-react';

const frontend = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const temporary = await mkdtemp(resolve(tmpdir(), 'guard-rule-config-test-'));
const harness = { name: 'rule-config-test-harness', configureServer(vite) {
  vite.middlewares.use('/__rule_ui_test__', async (req, res) => {
    res.setHeader('Content-Type', 'text/html');
    res.end(await vite.transformIndexHtml(req.originalUrl,
      '<div id="root"></div><script type="module" src="/scripts/rule-config-ui-fixture.tsx"></script>'));
  });
} };
const server = await createServer({ configFile: false, envFile: false, root: frontend,
  cacheDir: resolve(temporary, 'vite'), plugins: [react(), harness], server: { host: '127.0.0.1', port: 0,
    fs: { allow: [frontend, realpathSync(resolve(frontend, 'node_modules'))] } } });
await server.listen();
const base = `http://127.0.0.1:${server.httpServer.address().port}/__rule_ui_test__?mock=0`;
let context;
after(async () => {
  await context?.close();
  await server.close();
  await rm(temporary, { recursive: true, force: true });
});
context = await chromium.launchPersistentContext(resolve(temporary, 'profile'), {
  ...(process.env.UI_BROWSER_PATH ? { executablePath: process.env.UI_BROWSER_PATH } : {}),
  headless: true, args: ['--disable-gpu', '--disable-dev-shm-usage'],
});

const direction = { type: 'array', title: 'Направление', minItems: 2, maxItems: 2,
  items: { type: 'array', minItems: 2, maxItems: 2, items: { type: 'number', minimum: 0, maximum: 1 } } };
const matrixMeta = { type: 'new_generic_type', title: 'Направление движения', required_models: ['det'], default_severity: 'high',
  zone_schema: { type: 'array', minItems: 1, maxItems: 1 },
  params_schema: { type: 'object', required: ['direction'], properties: { direction }, additionalProperties: false } };
const zone = (id) => ({ zone_id: id, name: id, kind: 'danger', polygon: [[0, 0], [1, 0], [1, 1]], active_hours: null });
const rule = (type = matrixMeta.type, params = { direction: [[.1, .2], [.7, .8]] }) => ({
  rule_id: 'rule1', type, enabled: true, zones: ['a'], params, min_duration_s: 2, min_ratio: .7,
  close_after_s: 3, cooldown_s: 30, severity: 'high',
});
const camera = (rules = [rule()]) => ({ camera_id: 'cam1', name: 'Камера', site: 'Тест', enabled: false, sample_fps: 5,
  source: { type: 'file', uri: 'unused', loop: false, realtime: true }, zones: [zone('a'), zone('b')], rules });

async function render(input) {
  const page = await context.newPage();
  page.setDefaultTimeout(5000);
  const writes = [], errors = [];
  page.on('pageerror', (error) => errors.push(error.message));
  page.on('console', (message) => { if (message.type() === 'error') errors.push(message.text()); });
  await page.addInitScript((fixture) => { window.__fixture = fixture; }, input);
  await page.route('**/api/**', async (route) => {
    const request = route.request();
    // Vite also serves /src/api/*.ts: only intercept application HTTP endpoints.
    if (!new URL(request.url()).pathname.startsWith('/api/')) return route.continue();
    if (request.method() === 'PUT' || request.method() === 'POST') {
      const body = request.postDataJSON();
      writes.push(body);
      await route.fulfill({ json: body });
    } else if (request.url().endsWith('/rules/types')) await route.fulfill({ json: input.types });
    else await route.fulfill({ status: 204 });
  });
  await page.goto(base);
  try {
    await page.waitForFunction(() => typeof window.updateOriginal === 'function', null, { timeout: 5000 });
  } catch (error) {
    await page.close();
    throw new Error(`${error.message}; page errors: ${errors.join('; ')}`);
  }
  return { page, writes, errors };
}
const save = (page) => page.getByRole('button', { name: 'Сохранить', exact: true });
const undo = (page) => page.getByRole('button', { name: 'Отменить', exact: true });
const rulesTab = (page) => page.getByRole('tab', { name: /Правила/ }).click();
const cell = (page, point, axis) => page.getByRole('spinbutton', { name: `Направление: точка ${point}, ${axis}`, exact: true });

test('shared draft validation enforces finite bounds, required scalars, arrays and old metadata', async () => {
  const { validateSchema, schemaDefaults } = await server.ssrLoadModule('/src/features/cameras/SchemaForm.tsx');
  const { validateRule, zoneRemovalError } = await server.ssrLoadModule('/src/features/cameras/RulesEditor.tsx');
  for (const value of [undefined, 0, -1, NaN, Infinity, '2'])
    assert.ok(validateSchema({ type: 'number', exclusiveMinimum: 0 }, value).length);
  assert.deepEqual(validateSchema(direction, [[0, 1], [.2, .3]]), []);
  for (const value of [[[0, 1]], [[0, 1], [2, .3]], [[0, 1], [.2, Infinity]], [[0, 1], [.2]]])
    assert.ok(validateSchema(direction, value).length);
  assert.deepEqual(schemaDefaults(matrixMeta.params_schema), {}); // Never invent an arrow.
  assert.ok(validateRule(rule(), camera(), matrixMeta).length === 0);
  assert.deepEqual(validateRule({ ...rule(), min_duration_s: 70, cooldown_s: 5000, close_after_s: 200 }, camera(), matrixMeta), []);
  assert.ok(zoneRemovalError(camera(), 'a', [matrixMeta]));
  const old = { ...matrixMeta }; delete old.zone_schema;
  assert.equal(zoneRemovalError(camera(), 'a', [old]), null);
  assert.deepEqual(validateRule({ ...rule(), zones: null }, camera(), old), []);
});

test('four controlled direction fields block invalid Save and submit exactly the edited matrix', async () => {
  const configured = camera([{ ...rule(), min_duration_s: 70, cooldown_s: 5000, close_after_s: 200 }]);
  const { page, writes, errors } = await render({ camera: configured, types: [matrixMeta] });
  try {
    await rulesTab(page);
    assert.equal(await page.locator('fieldset[aria-label="Направление"] input[type="number"]').count(), 4);
    await cell(page, 1, 'X').fill('1.2');
    assert.equal(await save(page).isDisabled(), true);
    assert.ok(await page.getByRole('alert').filter({ hasText: 'максимум 1' }).count());
    assert.equal(writes.length, 0);
    await cell(page, 1, 'X').fill('0.3');
    assert.equal(await save(page).isDisabled(), false);
    await cell(page, 2, 'Y').fill('');
    assert.equal(await save(page).isDisabled(), true);
    await cell(page, 2, 'Y').fill('0.9');
    await save(page).click();
    await page.waitForFunction(() => document.querySelector('.dirty-mark') === null);
    assert.deepEqual(writes[0].rules[0].params.direction, [[.3, .2], [.7, .9]]);
    assert.equal(writes[0].rules[0].min_duration_s, 70);
    assert.equal(writes[0].rules[0].cooldown_s, 5000);
    assert.equal(writes[0].rules[0].close_after_s, 200);
    assert.deepEqual(errors, []);
  } finally { await page.close(); }
});

test('invalid JSON stays visible, blocks Save, and Undo/refetch re-sync controlled fields', async () => {
  const meta = { ...matrixMeta, params_schema: { type: 'object', required: ['custom'], properties: {
    custom: { type: 'array', title: 'JSON список', minItems: 1, items: { type: 'number' } },
    gate: { type: 'boolean', title: 'Учитывать условие', default: true },
    window: { type: 'number', title: 'Окно', minimum: 0, default: 2 },
  } } };
  const initial = camera([rule(meta.type, { custom: [1] })]);
  const { page, writes, errors } = await render({ camera: initial, types: [meta] });
  try {
    await rulesTab(page);
    const field = page.getByLabel('JSON список', { exact: true });
    const optionalNumber = page.getByRole('spinbutton', { name: 'Окно', exact: true });
    const optionalFlag = page.getByRole('checkbox', { name: 'Учитывать условие', exact: true });
    assert.equal(await optionalFlag.isChecked(), true);
    assert.equal(await optionalNumber.inputValue(), '2');
    await field.fill('[broken');
    assert.equal(await field.inputValue(), '[broken');
    assert.equal(await save(page).isDisabled(), true);
    assert.ok(await page.getByRole('alert').filter({ hasText: 'некорректный JSON' }).count());
    await page.getByRole('tab', { name: /Источник/ }).click();
    assert.equal(await save(page).isDisabled(), true); // Unmounting rules cannot drop invalidity.
    await rulesTab(page);
    assert.equal(await field.inputValue(), '[broken');
    const updated = structuredClone(initial); updated.rules[0].params.custom = [9];
    await page.evaluate((value) => window.updateOriginal(value), updated);
    assert.equal(await field.inputValue(), '[broken'); // Refetch preserves the unsaved draft.
    await undo(page).click();
    assert.equal(await field.inputValue(), '[9]');
    assert.equal(await save(page).isDisabled(), true);
    updated.rules[0].params.custom = [8];
    await page.evaluate((value) => window.updateOriginal(value), updated);
    await page.waitForFunction(() => document.querySelector('textarea').value === '[8]');
    await optionalNumber.fill('');
    assert.equal(await optionalNumber.inputValue(), '');
    assert.equal(await save(page).isDisabled(), true);
    await optionalNumber.fill('3');
    await optionalFlag.uncheck();
    await field.fill('[2,3]');
    await save(page).click();
    await page.waitForFunction(() => document.querySelector('.dirty-mark') === null);
    assert.deepEqual(writes[0].rules[0].params.custom, [2, 3]);
    assert.equal(writes[0].rules[0].params.window, 3);
    assert.equal(writes[0].rules[0].params.gate, false);
    assert.deepEqual(errors, []);
  } finally { await page.close(); }
});

test('required capacity and dwell remain empty/invalid without stale defaults', async () => {
  for (const [key, schema] of [['capacity', { type: 'integer', minimum: 1 }], ['dwell_s', { type: 'number', exclusiveMinimum: 0 }]]) {
    const meta = { ...matrixMeta, params_schema: { type: 'object', required: [key], properties: { [key]: { ...schema, title: key } } } };
    const { page, writes } = await render({ camera: camera([rule(meta.type, { [key]: 2 })]), types: [meta] });
    try {
      await rulesTab(page);
      const field = page.getByRole('spinbutton', { name: key, exact: true });
      await field.fill('');
      assert.equal(await field.inputValue(), '');
      assert.equal(await save(page).isDisabled(), true);
      assert.ok(await page.getByRole('alert').filter({ hasText: 'обязательное' }).count());
      await field.fill('0'); assert.equal(await save(page).isDisabled(), true);
      await field.fill('3'); assert.equal(await save(page).isDisabled(), false);
      assert.equal(writes.length, 0);
      await undo(page).click(); assert.equal(await field.inputValue(), '2');
    } finally { await page.close(); }
  }
});

test('strict linked zone deletion is blocked until reassignment or dependent rule removal; old types remain compatible', async () => {
  const { page, writes, errors } = await render({ camera: camera(), types: [matrixMeta] });
  try {
    await page.getByRole('button', { name: 'Удалить зону', exact: true }).click();
    assert.ok(await page.getByRole('alert').filter({ hasText: 'переназначьте' }).count());
    assert.equal(writes.length, 0);
    await rulesTab(page);
    assert.equal(await page.getByRole('button', { name: 'Весь кадр', exact: true }).count(), 0);
    await page.getByRole('button', { name: 'b', exact: true }).click();
    await page.getByRole('tab', { name: /Зоны/ }).click();
    await page.getByRole('button', { name: 'Удалить зону', exact: true }).click();
    await save(page).click();
    await page.waitForFunction(() => document.querySelector('.dirty-mark') === null);
    assert.deepEqual(writes[0].zones.map((z) => z.zone_id), ['b']);
    assert.deepEqual(writes[0].rules[0].zones, ['b']);
    await rulesTab(page);
    await page.getByRole('button', { name: 'Удалить правило', exact: true }).click();
    await page.getByRole('tab', { name: /Зоны/ }).click();
    await page.getByRole('button', { name: 'Удалить зону', exact: true }).click();
    await save(page).click();
    await page.waitForFunction(() => document.querySelector('.dirty-mark') === null);
    assert.deepEqual(writes[1].rules, []); assert.deepEqual(writes[1].zones, []);
    assert.deepEqual(errors, []);
  } finally { await page.close(); }
  const old = { ...matrixMeta }; delete old.zone_schema;
  const legacy = await render({ camera: camera(), types: [old] });
  try {
    await legacy.page.getByRole('button', { name: 'Удалить зону', exact: true }).click();
    await rulesTab(legacy.page);
    assert.equal(await legacy.page.getByRole('button', { name: 'Весь кадр', exact: true }).getAttribute('aria-pressed'), 'true');
    await save(legacy.page).click();
    await legacy.page.waitForFunction(() => document.querySelector('.dirty-mark') === null);
    assert.equal(legacy.writes[0].rules[0].enabled, false);
    assert.equal(legacy.writes[0].rules[0].zones, null);
  } finally { await legacy.page.close(); }
});
