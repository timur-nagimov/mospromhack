// Скриншоты всех экранов на мок-данных: один процесс браузера, один контекст, по очереди.
// Запуск (dev-сервер в режиме мока уже поднят): node scripts/screenshots.mjs
//   UI_URL=http://localhost:5173/  SHOTS_DIR=../data/media/ui
import { chromium } from 'playwright';
import { mkdirSync } from 'node:fs';
import { resolve } from 'node:path';

const BASE = process.env.UI_URL ?? 'http://localhost:5173/';
const OUT = resolve(process.env.SHOTS_DIR ?? '/Users/nagimov-timur/Desktop/mosprom/data/media/ui');
const ONLY = process.env.ONLY ? process.env.ONLY.split(',') : null;
mkdirSync(OUT, { recursive: true });

const W = Number(process.env.SHOT_W ?? 1600);
const H = Number(process.env.SHOT_H ?? 1000);

const browser = await chromium.launch({ args: ['--disable-gpu', '--disable-dev-shm-usage'] });
const errors = [];
try {
  const context = await browser.newContext({ viewport: { width: W, height: H }, deviceScaleFactor: 1, locale: 'ru-RU', timezoneId: 'Europe/Moscow' });
  const page = await context.newPage();
  page.on('pageerror', (e) => errors.push(`pageerror: ${e.message}`));
  page.on('console', (m) => m.type() === 'error' && errors.push(`console: ${m.text()}`));

  const shot = async (name, height = H) => {
    if (ONLY && !ONLY.includes(name)) return;
    await page.setViewportSize({ width: W, height });
    await page.waitForTimeout(400);
    const path = `${OUT}/${process.env.SHOT_PREFIX ?? ''}${name}.png`;
    await page.screenshot({ path });
    console.log('saved', path);
  };
  const go = async (hash, wait = 1500) => {
    await page.evaluate((h) => (window.location.hash = h), hash);
    await page.waitForTimeout(wait);
  };

  // ?mock=1 — на случай, если сервер поднят без VITE_MOCK
  await page.goto(`${BASE}?mock=1#/`, { waitUntil: 'networkidle' });
  // первое «живое» событие (критическое падение) приходит через ~2,5 с
  await page.waitForTimeout(5500);
  await shot('01-monitoring');

  // карточка критического события из баннера
  const open = page.getByRole('button', { name: 'Открыть карточку' });
  if (await open.count()) await open.first().click();
  else await page.locator('.feed-item').first().click();
  await page.waitForTimeout(1500);
  await shot('02-event-card');
  // клип (в демо — реконструкция)
  await page.getByRole('tab', { name: /Клип/ }).click();
  await page.waitForTimeout(1200);
  await shot('02b-event-clip');
  await page.keyboard.press('Escape');
  await page.waitForTimeout(400);

  await go('#/journal', 2500);
  await shot('03-journal');

  await go('#/analytics', 2500);
  await shot('04-analytics', 1650);

  await go('#/cameras/cam-01', 2000);
  await shot('05-cameras-zones');
  await page.getByRole('tab', { name: /Правила/ }).click();
  await page.waitForTimeout(800);
  await shot('06-cameras-rules', 1400);
  await page.getByRole('tab', { name: /Зоны/ }).click();
  await page.waitForTimeout(500);
  // режим рисования новой зоны: три вершины
  await page.getByRole('button', { name: 'Нарисовать зону' }).click();
  const stage = page.locator('.zone-stage');
  const box = await stage.boundingBox();
  if (box) {
    for (const [x, y] of [
      [0.08, 0.2],
      [0.3, 0.16],
      [0.34, 0.42],
      [0.1, 0.46],
    ]) {
      await page.mouse.click(box.x + box.width * x, box.y + box.height * y);
      await page.waitForTimeout(150);
    }
    await page.mouse.move(box.x + box.width * 0.2, box.y + box.height * 0.55);
  }
  await page.waitForTimeout(300);
  await shot('05b-zone-drawing');
  await page.keyboard.press('Escape');

  await go('#/performance', 6000);
  await shot('07-performance', 1600);
} finally {
  await browser.close();
}
if (errors.length) {
  console.log('--- ошибки страницы ---');
  for (const e of [...new Set(errors)]) console.log(e);
}
