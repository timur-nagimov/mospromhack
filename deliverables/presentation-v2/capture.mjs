#!/usr/bin/env node
// Скриншоты интерфейса для презентации v2. Только чтение: ничего не сохраняет,
// статусы событий и зоны не меняет (редактор зон открывается без сохранения).
//
//   tools/heavy.sh node deliverables/presentation-v2/capture.mjs                 все экраны
//   tools/heavy.sh node deliverables/presentation-v2/capture.mjs --only event,zones
//   tools/heavy.sh node deliverables/presentation-v2/capture.mjs --dump zones     + тексты кнопок на экране
//
// Окружение: UI_URL (по умолчанию http://127.0.0.1:8000/), UI_USER / UI_PASSWORD (admin / admin),
// PLAYWRIGHT_DIR (по умолчанию frontend/node_modules/playwright основного репозитория), SHOTS_DIR.
import fs from 'node:fs';
import path from 'node:path';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const require = createRequire(import.meta.url);
const PW = process.env.PLAYWRIGHT_DIR ?? '/Users/nagimov-timur/Desktop/mosprom/frontend/node_modules/playwright';
const { chromium } = require(PW);

const BASE = (process.env.UI_URL ?? 'http://127.0.0.1:8000/').replace(/\/?$/, '/');
const OUT = path.resolve(process.env.SHOTS_DIR ?? path.join(HERE, 'screens'));
const USER = process.env.UI_USER ?? 'admin';
const PASSWORD = process.env.UI_PASSWORD ?? 'admin';
const W = 1600;
const H = 900;

const args = process.argv.slice(2);
const opt = (name) => {
  const i = args.indexOf(name);
  return i >= 0 ? args[i + 1] ?? '' : null;
};
const only = opt('--only') ? new Set(opt('--only').split(',')) : null;
const dump = opt('--dump') ? new Set(opt('--dump').split(',')) : new Set();
fs.mkdirSync(OUT, { recursive: true });

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function isLoginPage(page) {
  if (/login/i.test(page.url())) return true;
  return (await page.locator('input[type="password"]').count()) > 0;
}

async function ensureLogin(page) {
  if (!(await isLoginPage(page))) return;
  console.log('вход:', page.url());
  const user = page.locator('input[name="username"], input[name="login"], input[autocomplete="username"], input[type="text"]').first();
  if (await user.count()) await user.fill(USER);
  await page.locator('input[type="password"]').first().fill(PASSWORD);
  await page.locator('input[type="password"]').first().press('Enter');
  await sleep(2500);
  if (await isLoginPage(page)) {
    // Запасной путь: кнопка демо-входа под ролью администратора.
    const demo = page.getByRole('button', { name: /админ|admin/i }).first();
    if (await demo.count()) await demo.click();
    await sleep(2500);
  }
  if (await isLoginPage(page)) throw new Error('не удалось войти');
}

async function waitImages(page, timeout = 8000) {
  await page
    .waitForFunction(() => [...document.images].every((i) => i.complete), null, { timeout })
    .catch(() => console.log('  картинки догружаются дольше', timeout, 'мс'));
}

async function go(page, hash, wait = 3000) {
  await page.evaluate((h) => {
    window.location.hash = h;
  }, hash);
  await sleep(wait);
  await ensureLogin(page);
  await waitImages(page);
}

async function dumpPage(page, name) {
  const info = await page.evaluate(() => {
    const txt = (el) => (el.innerText || el.getAttribute('aria-label') || '').trim().replace(/\s+/g, ' ').slice(0, 60);
    return {
      url: location.href,
      buttons: [...document.querySelectorAll('button, [role="tab"], a')].map(txt).filter(Boolean).slice(0, 80),
      inputs: [...document.querySelectorAll('input, select')].map((e) => `${e.tagName}:${e.type}:${e.name}:${e.value}`).slice(0, 30),
    };
  });
  fs.writeFileSync(path.join(OUT, `${name}.dump.json`), JSON.stringify(info, null, 2));
  console.log('  dump', name, info.url, info.buttons.length, 'кнопок');
}

async function shot(page, name) {
  const file = path.join(OUT, `${name}.jpg`);
  await page.screenshot({ path: file, type: 'jpeg', quality: Number(process.env.QUALITY ?? 85) });
  console.log('saved', file);
  if (dump.has(name)) await dumpPage(page, name);
}

async function apiJson(page, url) {
  return page.evaluate(async (u) => {
    const r = await fetch(u, { credentials: 'include', headers: { 'X-Guard-Role': 'admin' } });
    return r.ok ? r.json() : { error: r.status };
  }, url);
}

function pickEvent(items) {
  const withSnap = items.filter((e) => e.snapshot_path);
  const pref = (pred) => withSnap.find(pred);
  return (
    pref((e) => e.type === 'zone_intrusion' && e.clip_path && e.zone_id) ||
    pref((e) => e.type === 'zone_intrusion' && e.zone_id) ||
    pref((e) => e.clip_path) ||
    withSnap[0] ||
    items[0]
  );
}

const SHOTS = {
  async monitoring(page) {
    await go(page, '#/', 2000);
    await page.waitForSelector('.feed-item', { timeout: 30000 }).catch(() => console.log('  лента пуста'));
    await sleep(6000);
    await waitImages(page);
    await shot(page, 'monitoring');
  },

  async event(page) {
    const data = await apiJson(page, '/api/events?limit=300');
    const items = Array.isArray(data) ? data : data.items ?? [];
    const ev = pickEvent(items);
    if (!ev) {
      console.log('  событий нет');
      return;
    }
    console.log('  событие', ev.id, ev.type, ev.camera_id, 'zone', ev.zone_id, 'clip', Boolean(ev.clip_path));
    await go(page, `#/journal?event=${ev.id}`, 4000);
    await sleep(1500);
    await shot(page, 'event');
  },

  async journal(page) {
    await go(page, '#/journal', 4000);
    await shot(page, 'journal');
  },

  async zones(page) {
    await go(page, '#/cameras/cam_s1', 4000);
    // Выбираем первую зону, чтобы показать её тип и расписание. Ничего не сохраняем.
    const zone = page.locator('.zone-item, .zone-list button, [data-zone-id]').first();
    if (await zone.count()) {
      await zone.click().catch(() => {});
      await sleep(1200);
    }
    await shot(page, 'zones');
  },

  async rules(page) {
    await go(page, '#/cameras/cam_s1', 3000);
    const tab = page.getByRole('tab', { name: /Правила/ }).first();
    if (await tab.count()) {
      await tab.click();
      await sleep(1500);
    }
    await shot(page, 'rules');
  },

  async analytics(page) {
    await go(page, '#/analytics', 5000);
    await shot(page, 'analytics');
  },

  async employees(page) {
    await go(page, '#/employees', 4000);
    await shot(page, 'employees');
  },

  async data(page) {
    await go(page, '#/data', 5000);
    await shot(page, 'data');
  },

  async performance(page) {
    await go(page, '#/performance', 6000);
    await shot(page, 'performance');
  },
};

const browser = await chromium.launch({ args: ['--disable-gpu', '--disable-dev-shm-usage'] });
const errors = [];
try {
  const context = await browser.newContext({ viewport: { width: W, height: H }, deviceScaleFactor: Number(process.env.SCALE ?? 1.5), locale: 'ru-RU', timezoneId: 'Europe/Moscow' });
  const page = await context.newPage();
  page.on('pageerror', (e) => errors.push(`pageerror: ${e.message}`));
  await page.goto(`${BASE}#/`, { waitUntil: 'domcontentloaded' });
  await sleep(2500);
  await ensureLogin(page);
  // Баннер критической тревоги в демо всплывает каждые несколько секунд (ролики зациклены)
  // и закрывает полстраницы. Для скриншотов прячем его стилем, состояние приложения не меняется.
  if (!process.env.KEEP_ALARM) await page.addStyleTag({ content: '.alarm { display: none !important; }' });
  for (const [name, fn] of Object.entries(SHOTS)) {
    if (only && !only.has(name)) continue;
    console.log('экран', name);
    try {
      await fn(page);
    } catch (e) {
      console.log('  ошибка', name, e.message);
    }
    await page.keyboard.press('Escape').catch(() => {});
  }
} finally {
  await browser.close();
}
if (errors.length) console.log([...new Set(errors)].join('\n'));
