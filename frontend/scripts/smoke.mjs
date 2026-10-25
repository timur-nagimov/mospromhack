// Сквозная проверка ключевых сценариев UI на мок-данных (один браузер, один контекст).
// Запуск: tools/heavy.sh frontend/scripts/shoot.sh smoke.mjs
import { chromium } from 'playwright';
import { readFileSync } from 'node:fs';

const BASE = process.env.UI_URL ?? 'http://localhost:5173/';
const results = [];
const check = (name, ok, extra = '') => {
  results.push({ name, ok });
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${extra ? ` — ${extra}` : ''}`);
};

const browser = await chromium.launch({ args: ['--disable-gpu', '--disable-dev-shm-usage'] });
const errors = [];
try {
  const context = await browser.newContext({ viewport: { width: 1600, height: 1000 }, acceptDownloads: true });
  const page = await context.newPage();
  page.on('pageerror', (e) => errors.push(e.message));

  await page.goto(`${BASE}?mock=1&auth=0#/`, { waitUntil: 'networkidle' });
  await page.waitForTimeout(4500);
  const kpi = async () => Number((await page.locator('.kpi--accent .kpi__value').innerText()).trim());
  const before = await kpi();
  check('мониторинг: плитки камер', (await page.locator('.tile').count()) === 6);
  check('мониторинг: критическая тревога пришла по WS', (await page.locator('.alarm').count()) === 1);

  await page.getByRole('button', { name: 'Открыть карточку' }).click();
  await page.waitForTimeout(800);
  check('карточка: открыта по ?event=', page.url().includes('event='));
  await page.keyboard.press('1');
  await page.waitForTimeout(900);
  const status = await page.locator('.ev-head__badges').innerText();
  check('карточка: хоткей 1 → «Подтверждено» (PATCH)', status.includes('Подтверждено'), status.replace(/\n/g, ' '));
  await page.keyboard.press('Escape');
  await page.waitForTimeout(600);
  const after = await kpi();
  check('счётчик неразобранных уменьшился', after === before - 1, `${before} → ${after}`);

  await page.evaluate(() => (window.location.hash = '#/journal'));
  await page.waitForTimeout(1500);
  await page.locator('label:has-text("Статус") select').selectOption('false');
  await page.waitForTimeout(1200);
  const statuses = await page.locator('tbody .badge--false').count();
  const rows = await page.locator('tbody tr').count();
  check('журнал: фильтр по статусу «Ложное»', rows > 0 && statuses === rows, `${statuses}/${rows}`);
  check('журнал: фильтр сохранён в адресе', page.url().includes('status=false'));
  const [download] = await Promise.all([page.waitForEvent('download'), page.getByRole('button', { name: /Экспорт CSV/ }).click()]);
  const csv = readFileSync(await download.path(), 'utf8');
  const lines = csv.trim().split('\r\n');
  check('журнал: экспорт CSV', lines[0].includes('Тип нарушения') && lines.length > 1, `${download.suggestedFilename()}, строк ${lines.length - 1}`);

  await page.evaluate(() => (window.location.hash = '#/cameras/cam-02'));
  await page.waitForTimeout(1500);
  await page.locator('.zone-props input.input').first().fill('Зона СИЗ, линия 1');
  await page.waitForTimeout(200);
  check('камеры: признак несохранённых изменений', (await page.locator('.dirty-mark').count()) === 1);
  await page.getByRole('button', { name: 'Сохранить' }).click();
  await page.waitForTimeout(1200);
  check('камеры: PUT сохранён', (await page.locator('.toast').last().innerText()).includes('сохранена'));

  await page.getByRole('tab', { name: /Правила/ }).click();
  const rulesBefore = await page.locator('.rule-card').count();
  await page.locator('.rules-editor__add select').selectOption('running');
  await page.getByRole('button', { name: 'Добавить правило' }).click();
  await page.waitForTimeout(300);
  const rulesAfter = await page.locator('.rule-card').count();
  check('правила: добавлено правило с формой по params_schema', rulesAfter === rulesBefore + 1 && (await page.locator('.rule-card').last().locator('.schema-form').count()) === 1);

  await page.evaluate(() => (window.location.hash = '#/cameras/new'));
  await page.waitForTimeout(800);
  await page.locator('label:has-text("Название") input').fill('Цех №2, пресс');
  await page.locator('label:has-text("Участок") input').fill('Цех №2');
  await page.locator('label:has-text("Адрес источника") input').fill('rtsp://10.0.12.60:8554/press');
  await page.getByRole('button', { name: 'Создать камеру' }).click();
  await page.waitForTimeout(1500);
  check('камеры: POST новой камеры', /#\/cameras\/cam-07/.test(page.url()), page.url());

  await page.goto(`${BASE}?mock=0#/`, { waitUntil: 'domcontentloaded' });
  await page.waitForTimeout(3500);
  const pill = await page.locator('.topbar__status').innerText();
  check('http-режим (?mock=0): /api и /ws через прокси, статус связи виден', /Поток событий|Нет связи|Подключение/.test(pill), pill.replace(/\n/g, ' · '));
} finally {
  await browser.close();
}
const failed = results.filter((r) => !r.ok).length;
console.log(`\nИтого: ${results.length - failed}/${results.length} PASS`);
if (errors.length) console.log('Ошибки страницы:', [...new Set(errors)].slice(0, 5));
process.exit(failed ? 1 : 0);
