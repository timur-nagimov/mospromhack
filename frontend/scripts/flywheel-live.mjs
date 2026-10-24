// Страница «Данные» на живом бэкенде: сводка, сбор по WS, очередь, кадр с рамками, решение 1, датасеты, модели.
// Запуск — через scripts/flywheel-live.sh (поднимает сервер с отдельным журналом и гасит его).
import { chromium } from 'playwright';
import { mkdirSync } from 'node:fs';
import { resolve } from 'node:path';

const BASE = process.env.UI_URL ?? 'http://127.0.0.1:8011/';
const OUT = resolve(process.env.SHOTS_DIR ?? '/Users/nagimov-timur/Desktop/mosprom/data/media/checks');
const WAIT_S = Number(process.env.COLLECT_S ?? 35);
mkdirSync(OUT, { recursive: true });

const results = [];
const check = (name, ok, extra = '') => {
  results.push({ name, ok });
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${extra ? ` — ${extra}` : ''}`);
};

const browser = await chromium.launch({ args: ['--disable-gpu', '--disable-dev-shm-usage'] });
const errors = [];
try {
  const context = await browser.newContext({ viewport: { width: 1600, height: 1000 }, locale: 'ru-RU', timezoneId: 'Europe/Moscow' });
  const page = await context.newPage();
  page.on('pageerror', (e) => errors.push(`pageerror: ${e.message}`));
  const wsFlywheel = [];
  page.on('websocket', (ws) => ws.on('framereceived', (f) => typeof f.payload === 'string' && f.payload.includes('flywheel_sample') && wsFlywheel.push(f.payload)));
  const shot = async (name) => {
    await page.waitForTimeout(300);
    const path = `${OUT}/flywheel_ui_live_${name}.png`;
    await page.screenshot({ path });
    console.log('saved', path);
  };
  const total = async () =>
    Number((await page.locator('.fw-step').first().locator('.fw-step__note').innerText()).match(/всего\s([\d\s ]+)/)?.[1]?.replace(/\D/g, '') ?? -1);

  await page.goto(`${BASE}?mock=0#/data`, { waitUntil: 'domcontentloaded' });
  await page.waitForTimeout(3000);
  check('живой API: нет плашки «API маховика на сервере нет»', (await page.locator('.fw-missing').count()) === 0);
  check('полоса цикла', (await page.locator('.fw-step').count()) === 6);
  const t0 = await total();
  await page.waitForTimeout(WAIT_S * 1000);
  const t1 = await total();
  check('сбор: счётчик «всего» растёт без перезагрузки', t1 > t0, `${t0} → ${t1}`);
  check('WS: сообщения flywheel_sample приходят', wsFlywheel.length > 0, `${wsFlywheel.length} шт.`);
  const items = await page.locator('.fw-qitem').count();
  check('очередь: сэмплы с сервера', items > 0, `${items}`);
  if (items > 0) {
    await page.locator('.fw-qitem').first().click();
    await page.waitForTimeout(1200);
    const href = await page.locator('.fw-card__frame svg image').getAttribute('href');
    const status = href ? await page.evaluate(async (u) => (await fetch(u)).status, href) : 0;
    check('кадр сэмпла: frame.jpg отдаётся', status === 200, `${href} → ${status}`);
    const vb = await page.locator('.fw-card__frame svg').getAttribute('viewBox');
    check('кадр: viewBox по width × height сэмпла', /^0 0 \d+ \d+$/.test(vb ?? ''), vb ?? '');
    await page.getByRole('button', { name: 'Обе' }).click();
    await page.waitForTimeout(300);
    const boxes = await page.locator('.fw-card__frame .fw-box').count();
    check('рамки детекций поверх кадра', boxes > 0, `${boxes}`);
    await shot('01-review');
    await page.locator('body').click({ position: { x: 5, y: 5 } });
    await page.keyboard.press('1');
    await page.waitForTimeout(1200);
    const toast = await page.locator('.toast').last().innerText().catch(() => '');
    check('клавиша 1: PATCH на сервере', toast.includes('принят'), toast);
  }

  await page.getByRole('tab', { name: /Датасеты/ }).click();
  await page.waitForTimeout(800);
  await page.getByRole('button', { name: 'Собрать версию' }).click();
  await page.waitForTimeout(2500);
  const toastDs = await page.locator('.toast').last().innerText().catch(() => '');
  check('«Собрать версию»: ответ сервера показан', toastDs.includes('Версия') || toastDs.includes('не собрана'), toastDs);
  await shot('02-datasets');

  await page.getByRole('tab', { name: /Модели/ }).click();
  await page.waitForTimeout(800);
  const train = page.getByRole('button', { name: 'Дообучить' });
  check('«Дообучить» неактивна без GUARD_FLYWHEEL_ALLOW_TRAIN', await train.isDisabled());
  await shot('03-models');

  // карточка события: бейдж, если у события есть сэмпл
  const evSample = await page.evaluate(async () => {
    const r = await fetch('/api/flywheel/samples?source=event&limit=1');
    const page = await r.json();
    return page.items?.[0]?.event_id ?? null;
  });
  if (evSample) {
    await page.evaluate((id) => (window.location.hash = `#/journal?event=${id}`), evSample);
    await page.waitForTimeout(2000);
    check('карточка события: бейдж «в выборке маховика»', (await page.locator('.ev-flywheel').count()) === 1, evSample);
    await shot('04-event-badge');
  } else {
    console.log('SKIP  карточка события: за время прогона событий не было');
  }
} finally {
  await browser.close();
}
const uniq = [...new Set(errors)];
for (const e of uniq) console.log(e);
const failed = results.filter((r) => !r.ok).length;
console.log(`\nИтого: ${results.length - failed} из ${results.length}, ошибок страницы: ${uniq.length}`);
process.exit(failed || uniq.length ? 1 : 0);
