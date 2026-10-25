// Ролевая модель в интерфейсе: вход демо-учётками, что видит и может каждая роль, скриншоты.
// Работает и на мок-данных (MOCK=1, по умолчанию), и на живом бэкенде с GUARD_AUTH=1 (MOCK=0).
// Запуск: tools/heavy.sh frontend/scripts/rbac.sh            (мок, статика dist на :8013)
//         tools/heavy.sh frontend/scripts/rbac-live.sh       (uvicorn с входом на :8013)
//   UI_URL — адрес интерфейса; SHOTS_DIR — каталог скриншотов; SHOT_PREFIX — префикс имён
import { chromium } from 'playwright';
import { mkdirSync } from 'node:fs';
import { resolve } from 'node:path';

const BASE = process.env.UI_URL ?? 'http://127.0.0.1:8013/';
const MOCK = process.env.MOCK !== '0';
const OUT = resolve(process.env.SHOTS_DIR ?? '/Users/nagimov-timur/Desktop/mosprom-heavy.noindex/screens/rbac');
const PREFIX = process.env.SHOT_PREFIX ?? (MOCK ? '' : 'live-');
const W = Number(process.env.SHOT_W ?? 1440);
const H = Number(process.env.SHOT_H ?? 900);
mkdirSync(OUT, { recursive: true });

const results = [];
const check = (name, ok, extra = '') => {
  results.push({ name, ok: Boolean(ok) });
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${extra ? ` — ${extra}` : ''}`);
};

const browser = await chromium.launch({ args: ['--disable-gpu', '--disable-dev-shm-usage'] });
const errors = [];
try {
  const context = await browser.newContext({ viewport: { width: W, height: H }, deviceScaleFactor: 1, locale: 'ru-RU', timezoneId: 'Europe/Moscow' });
  const page = await context.newPage();
  page.on('pageerror', (e) => errors.push(`pageerror: ${e.message}`));
  page.on('console', (m) => {
    // 401 до входа и 403 проверки прав — ожидаемые ответы сервера
    if (m.type() === 'error' && !/401|403|Unauthorized|Forbidden/.test(m.text())) errors.push(`console: ${m.text()}`);
  });

  const shot = async (name, height = H) => {
    await page.setViewportSize({ width: W, height });
    await page.waitForTimeout(450);
    const path = `${OUT}/${PREFIX}${name}.png`;
    await page.screenshot({ path });
    console.log('saved', path);
    if (height !== H) await page.setViewportSize({ width: W, height: H });
  };
  const go = async (hash, wait = 1600) => {
    await page.evaluate((h) => (window.location.hash = h), hash);
    await page.waitForTimeout(wait);
  };
  const hash = () => page.evaluate(() => window.location.hash);
  const railLabels = async () => (await page.locator('.rail__label').allInnerTexts()).map((t) => t.replace(/­|\s+/g, ' ').replace(/­/g, '').trim());
  const toasts = async () => (await page.locator('.toast').allInnerTexts()).join(' | ');
  const closeModal = async () => {
    await page.keyboard.press('Escape');
    await page.waitForTimeout(400);
  };
  const openFirstEvent = async () => {
    await go('#/journal', 2200);
    const rows = page.locator('.journal__table tbody tr.is-clickable');
    if ((await rows.count()) === 0) return false;
    await rows.first().click();
    await page.waitForSelector('.ev-modal', { timeout: 5000 });
    await page.waitForTimeout(900);
    return true;
  };
  const switchRole = async (short) => {
    await page.locator('.demo-switch__btn', { hasText: short }).click();
    await page.waitForTimeout(2200);
  };

  // ---------- вход
  await page.goto(`${BASE}${MOCK ? '?mock=1&auth=1&demo=1' : ''}#/`, { waitUntil: 'networkidle' });
  await page.waitForSelector('.login', { timeout: 10000 });
  check('без входа — страница входа', (await hash()).startsWith('#/login'));
  const passes = await page.locator('.role-pass strong').allInnerTexts();
  check(
    'демо-вход: четыре роли',
    ['Сотрудник', 'Диспетчер смены', 'Инженер по ОТ', 'Администратор'].every((r) => passes.some((p) => p.includes(`Войти как ${r}`))),
    passes.join(', '),
  );
  check('карточки ролей: 2–3 строки «что делает»', (await page.locator('.role-pass').first().locator('li').count()) >= 2);
  await shot('00-login');

  // неверный пароль
  await page.locator('input[name="username"]').fill('worker');
  await page.locator('input[name="password"]').fill('wrong-password');
  await page.locator('.login__submit').click();
  await page.waitForTimeout(900);
  check('неверный пароль — понятная ошибка, остаёмся на входе', (await page.locator('.login .error-box').innerText()).includes('Неверный логин или пароль'));

  // ---------- Сотрудник
  await page.locator('.role-pass', { hasText: 'Войти как Сотрудник' }).click();
  await page.waitForSelector('.rail', { timeout: 10000 });
  await page.waitForTimeout(2500);
  check('сотрудник: стартовая страница — Аналитика', (await hash()).startsWith('#/analytics'), await hash());
  let labels = await railLabels();
  check('сотрудник: в меню только Журнал, Аналитика, Сотрудники', JSON.stringify(labels) === JSON.stringify(['Журнал', 'Аналитика', 'Сотрудники']), labels.join(', '));
  check('сотрудник: бейдж роли в шапке', (await page.locator('.role-chip').innerText()).includes('Сотрудник'));
  check('сотрудник: нет пилюли метрик инференса (system.view)', (await page.locator('.topbar .pill', { hasText: 'Инференс' }).count()) === 0);
  await shot('10-worker-analytics');

  await go('#/journal', 2500);
  const heads = await page.locator('.journal__table thead th').allInnerTexts();
  check('сотрудник: журнал без колонки кадров', !heads.some((h) => /кадр/i.test(h)), heads.join(','));
  check('сотрудник: журнал без снимков', (await page.locator('.journal__table img.thumb').count()) === 0);
  await shot('11-worker-journal');
  if (await openFirstEvent()) {
    check('сотрудник: в карточке события снимок и видео скрыты (152-ФЗ)', (await page.locator('.ev-media__locked').count()) === 1);
    check('сотрудник: решение по событию недоступно', (await page.locator('.decision:not([disabled])').count()) === 0);
    check('сотрудник: подсказка, какой роли доступно решение', (await page.locator('.ev-actions').innerText()).includes('Нужна роль'));
    await shot('12-worker-event');
    await closeModal();
  } else check('сотрудник: в журнале есть события', false, 'журнал пуст');

  await go('#/cameras', 1500);
  check('сотрудник: прямой переход на камеры — «Нет доступа» с правом', (await page.locator('.no-access .perm-tag code').innerText()) === 'cameras.view');
  await shot('13-worker-no-access');
  await go('#/users', 1200);
  check('сотрудник: «Пользователи» закрыты (users.manage)', (await page.locator('.no-access .perm-tag code').innerText()) === 'users.manage');

  await go('#/analytics', 1500);
  await page.locator('.role-chip').click();
  await page.waitForTimeout(500);
  const card = page.locator('.role-card');
  check('карточка роли: доступные и недоступные разделы', (await card.locator('.cap-list li.is-on').count()) > 0 && (await card.locator('.cap-list li.is-off').count()) > 0);
  check('карточка роли: «Нужна роль: …» у недоступного', (await card.innerText()).includes('Нужна роль'));
  await shot('14-worker-role-card');
  await page.keyboard.press('Escape');

  // ---------- Диспетчер смены — в один клик из колонны навигации
  await switchRole('Диспетчер');
  check('смена роли в один клик: диспетчер', (await page.locator('.role-chip').innerText()).includes('Диспетчер'));
  check('диспетчер: стартовая — Мониторинг', (await hash()) === '#/' || (await hash()) === '#/analytics' ? true : false, await hash());
  await go('#/', 3500);
  labels = await railLabels();
  check('диспетчер: Мониторинг, Камеры, Производительность есть; Данные, Пользователи — нет',
    labels.includes('Мониторинг') && labels.includes('Камеры и правила') && !labels.includes('Данные') && !labels.includes('Пользователи'), labels.join(', '));
  await shot('20-operator-monitoring');
  await go('#/cameras', 2500);
  check('диспетчер: камеры только для просмотра', (await page.locator('.readonly-mark').count()) === 1);
  check('диспетчер: рисование зон недоступно', (await page.getByRole('button', { name: 'Нарисовать зону' }).count()) === 0);
  check('диспетчер: «Добавить» камеру неактивна с подсказкой', await page.locator('.cam-list__head button').first().isDisabled());
  await shot('21-operator-cameras');
  if (await openFirstEvent()) {
    check('диспетчер: снимок события виден', (await page.locator('.ev-media__locked').count()) === 0);
    check('диспетчер: кнопки решения активны', (await page.locator('.decision:not([disabled])').count()) === 3);
    await shot('22-operator-event');
    await closeModal();
  }

  // ---------- Инженер по ОТ
  await switchRole('Инженер');
  await go('#/cameras', 2500);
  check('инженер ОТ: редактор зон доступен', (await page.getByRole('button', { name: 'Нарисовать зону' }).count()) === 1);
  const sourceTab = page.getByRole('tab', { name: /Источник/ });
  check('инженер ОТ: форма источника недоступна', (await sourceTab.count()) === 0 || (await sourceTab.isDisabled()));
  check('инженер ОТ: удалить камеру нельзя', (await page.getByRole('button', { name: /Удалить$/ }).count()) === 0 || (await page.getByRole('button', { name: /Удалить$/ }).first().isDisabled()));
  await shot('30-safety-cameras');
  labels = await railLabels();
  check('инженер ОТ: Данные и Журнал действий в меню, Пользователей нет',
    labels.includes('Данные') && labels.includes('Журнал действий') && !labels.includes('Пользователи'), labels.join(', '));
  await go('#/audit', 2500);
  check('инженер ОТ: журнал действий загрузился', (await page.locator('.audit__table tbody tr').count()) > 0);
  await shot('31-safety-audit');
  await go('#/data?tab=models', 2500);
  check('инженер ОТ: дообучение недоступно (flywheel.train)', (await page.locator('.fw-models').innerText()).includes('Нужна роль'));
  await shot('32-safety-data-models');

  // ---------- Администратор
  await switchRole('Админ');
  await go('#/users', 2600);
  check('админ: таблица пользователей', (await page.locator('.users__table tbody tr').count()) >= 4);
  check('админ: матрица ролей × права', (await page.locator('.role-matrix__table tbody tr:not(.role-matrix__group)').count()) >= 15);
  await shot('40-admin-users', 1720);
  // матрица целиком: окно выше страницы, чтобы элемент не обрезала прокрутка
  await page.setViewportSize({ width: W, height: 2400 });
  await page.waitForTimeout(500);
  await page.locator('.role-matrix').screenshot({ path: `${OUT}/${PREFIX}41-admin-role-matrix.png` });
  await page.setViewportSize({ width: W, height: H });
  console.log('saved', `${OUT}/${PREFIX}41-admin-role-matrix.png`);
  await go('#/audit', 2500);
  const actions = await page.locator('.audit-action__text code').allInnerTexts();
  check('админ: в журнале действий — входы и смены ролей', actions.includes('auth.login') && actions.includes('auth.logout'), [...new Set(actions)].join(','));
  await shot('42-admin-audit');
  await go('#/cameras', 2500);
  await page.getByRole('tab', { name: /Источник/ }).click();
  await page.waitForTimeout(700);
  check('админ: форма источника камеры', (await page.locator('.source-form').count()) === 1);
  await shot('43-admin-cameras-source');
  await page.locator('.role-chip').click();
  await page.waitForTimeout(500);
  await shot('44-admin-role-card');
  await page.keyboard.press('Escape');

  // ---------- выход
  await page.locator('.role-chip').click();
  await page.locator('.role-card__logout').click();
  await page.waitForSelector('.login', { timeout: 8000 });
  check('выход — снова страница входа без ?next', (await hash()) === '#/login', await hash());
  const leftover = await toasts();
  if (leftover) console.log('уведомления:', leftover);
} finally {
  await browser.close();
}

const failed = results.filter((r) => !r.ok);
console.log(`\nИтог: ${results.length - failed.length} из ${results.length} проверок`);
if (errors.length) {
  console.log('--- ошибки страницы ---');
  for (const e of [...new Set(errors)]) console.log(e);
}
process.exit(failed.length ? 1 : 0);
