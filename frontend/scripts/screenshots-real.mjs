// Скриншоты против живого бэкенда (через dev-прокси Vite на :8000). Только чтение:
// ничего не сохраняет и не меняет статусы событий.
// Запуск: VITE_MODE=development tools/heavy.sh frontend/scripts/shoot.sh screenshots-real.mjs
import { chromium } from 'playwright';
import { mkdirSync } from 'node:fs';
import { resolve } from 'node:path';

const BASE = process.env.UI_URL ?? 'http://localhost:5173/';
const OUT = resolve(process.env.SHOTS_DIR ?? '/Users/nagimov-timur/Desktop/mosprom/data/media/ui');
mkdirSync(OUT, { recursive: true });

const browser = await chromium.launch({ args: ['--disable-gpu', '--disable-dev-shm-usage'] });
const errors = [];
try {
  const context = await browser.newContext({ viewport: { width: 1600, height: 1000 }, deviceScaleFactor: 1 });
  const page = await context.newPage();
  page.on('pageerror', (e) => errors.push(`pageerror: ${e.message}`));
  const shot = async (name, height = 1000) => {
    await page.setViewportSize({ width: 1600, height });
    await page.waitForTimeout(400);
    await page.screenshot({ path: `${OUT}/${name}.png` });
    console.log('saved', `${OUT}/${name}.png`);
  };
  const go = async (hash, wait = 2500) => {
    await page.evaluate((h) => (window.location.hash = h), hash);
    await page.waitForTimeout(wait);
  };

  await page.goto(`${BASE}?mock=0#/`, { waitUntil: 'domcontentloaded' });
  await page.waitForTimeout(6000);
  await shot('real-01-monitoring');

  const item = page.locator('.feed-item').first();
  if (await item.count()) {
    await item.click();
    await page.waitForTimeout(2500);
    await shot('real-02-event-card');
    await page.keyboard.press('Escape');
  }
  await go('#/journal');
  await shot('real-03-journal');
  await go('#/analytics', 3000);
  await shot('real-04-analytics', 1650);
  await go('#/cameras', 3000);
  await shot('real-05-cameras-zones');
  await go('#/performance', 5000);
  await shot('real-07-performance', 1400);
} finally {
  await browser.close();
}
if (errors.length) console.log([...new Set(errors)].join('\n'));
