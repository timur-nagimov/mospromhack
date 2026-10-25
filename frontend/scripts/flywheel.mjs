// Страница «Данные» (маховик данных) на мок-данных: проверки сценариев и скриншоты.
// Запуск: tools/heavy.sh frontend/scripts/shoot.sh flywheel.mjs
//   SHOTS_DIR — куда класть скриншоты (по умолчанию data/media/checks основного репозитория)
import { chromium } from 'playwright';
import { mkdirSync } from 'node:fs';
import { resolve } from 'node:path';

const BASE = process.env.UI_URL ?? 'http://localhost:5173/';
const OUT = resolve(process.env.SHOTS_DIR ?? '/Users/nagimov-timur/Desktop/mosprom/data/media/checks');
mkdirSync(OUT, { recursive: true });
const W = Number(process.env.SHOT_W ?? 1600);
const H = Number(process.env.SHOT_H ?? 1000);

const results = [];
const check = (name, ok, extra = '') => {
  results.push({ name, ok });
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${extra ? ` — ${extra}` : ''}`);
};

const browser = await chromium.launch({ args: ['--disable-gpu', '--disable-dev-shm-usage'] });
const errors = [];
try {
  const context = await browser.newContext({ viewport: { width: W, height: H }, deviceScaleFactor: 1, locale: 'ru-RU', timezoneId: 'Europe/Moscow' });
  const page = await context.newPage();
  page.on('pageerror', (e) => errors.push(`pageerror: ${e.message}`));
  page.on('console', (m) => m.type() === 'error' && errors.push(`console: ${m.text()}`));

  const shot = async (name, height = H) => {
    await page.setViewportSize({ width: W, height });
    await page.waitForTimeout(350);
    const path = `${OUT}/flywheel_ui_${name}.png`;
    await page.screenshot({ path });
    console.log('saved', path);
  };
  const selectedId = async () => (await page.locator('.fw-card__kicker .mono').getAttribute('title')) ?? '';

  await page.goto(`${BASE}?mock=1&auth=0#/data`, { waitUntil: 'networkidle' });
  await page.waitForTimeout(2500);

  // ---------- верх страницы
  check('навигация: пункт «Данные» активен', (await page.locator('.rail__link.active', { hasText: 'Данные' }).count()) === 1);
  check('полоса цикла: шесть шагов', (await page.locator('.fw-step').count()) === 6);
  const legend = await page.locator('.fw-sources__legend').innerText();
  check(
    'счётчики: источники событие, неуверенность, мигание, расхождение, оператор',
    ['Событие', 'Неуверенность', 'Мигание трека', 'Расхождение TPU и CPU', 'Оператор'].every((s) => legend.includes(s)),
  );
  const items = await page.locator('.fw-qitem').count();
  check('очередь: сэмплы загружены', items > 10, `${items} на странице`);
  const frameRects = await page.locator('.fw-card__frame svg .fw-box').count();
  check('карточка: рамки поверх кадра', frameRects > 0, `${frameRects} рамок`);
  const vb = await page.locator('.fw-card__frame svg').getAttribute('viewBox');
  check('карточка: viewBox в пикселях кадра', /^0 0 (960 540|1280 720)$/.test(vb ?? ''), vb ?? '');
  await shot('01-review');

  // ---------- клавиши ← →
  const first = await selectedId();
  await page.keyboard.press('ArrowRight');
  await page.waitForTimeout(400);
  const second = await selectedId();
  check('клавиша →: следующий сэмпл', second && second !== first);
  await page.keyboard.press('ArrowLeft');
  await page.waitForTimeout(400);
  check('клавиша ←: предыдущий сэмпл', (await selectedId()) === first);

  // ---------- правка меток: смена класса и удаление рамки
  await page.locator('.fw-qitem', { has: page.locator('.fw-st--auto') }).first().click();
  await page.waitForTimeout(500);
  const labelsBefore = await page.locator('.fw-boxes li').count();
  if (labelsBefore > 0) {
    const sel = page.locator('.fw-boxes li select').first();
    const current = await sel.inputValue();
    await sel.selectOption(current === 'forklift' ? 'person' : 'forklift');
    await page.waitForTimeout(200);
    check('смена класса рамки: черновик помечен', (await page.locator('.fw-dirty').count()) === 1);
    await page.locator('.fw-boxes li button[title="Удалить рамку"]').last().click();
    await page.waitForTimeout(200);
    check('удаление рамки', (await page.locator('.fw-boxes li').count()) === labelsBefore - 1, `${labelsBefore} → ${labelsBefore - 1}`);
    await page.locator('.fw-card__frame').hover();
    await page.locator('.fw-boxes li').first().hover();
    await page.waitForTimeout(200);
    await shot('02-edit');
  } else {
    check('смена класса рамки', false, 'у сэмпла нет меток');
  }

  // ---------- клавиша 1: принять (с правками) и переход к следующему
  const decided = await selectedId();
  await page.locator('body').click({ position: { x: 5, y: 5 } });
  await page.keyboard.press('1');
  await page.waitForTimeout(900);
  const toastText = await page.locator('.toast').last().innerText().catch(() => '');
  check('клавиша 1: PATCH status=accepted', toastText.includes('принят'), toastText);
  check('после решения — следующий сэмпл', (await selectedId()) !== decided);

  // клавиша 2: отклонить
  const toReject = await selectedId();
  await page.keyboard.press('2');
  await page.waitForTimeout(900);
  const toast2 = await page.locator('.toast').last().innerText().catch(() => '');
  check('клавиша 2: PATCH status=rejected', toast2.includes('отклонён'), toast2);
  check('после отклонения — следующий сэмпл', (await selectedId()) !== toReject);

  // ---------- переключатель «метки / детекции»
  await page.getByRole('button', { name: /^Детекции/ }).click();
  await page.waitForTimeout(300);
  const dashed = await page.locator('.fw-card__frame .fw-box--det').count();
  const solid = await page.locator('.fw-card__frame .fw-box:not(.fw-box--det)').count();
  check('режим «Детекции»: только детекции', dashed > 0 && solid === 0, `${dashed} пунктирных`);
  await page.getByRole('button', { name: 'Обе' }).click();
  await page.waitForTimeout(300);
  await shot('03-both');

  // ---------- фильтр по источнику и ссылка на событие
  await page.locator('label:has-text("Источник") select').selectOption('event');
  await page.waitForTimeout(900);
  const tags = await page.locator('.fw-qitem .fw-src').allInnerTexts();
  check('фильтр «Источник: Событие»', tags.length > 0 && tags.every((t) => t.includes('Событие')), `${tags.length} сэмплов`);
  check('фильтр сохранён в адресе', page.url().includes('source=event'));
  const eventLink = page.locator('.fw-facts .fw-link');
  if (await eventLink.count()) {
    await eventLink.click();
    await page.waitForTimeout(1200);
    check('ссылка на событие открывает карточку', page.url().includes('event='));
    const badge = page.locator('.ev-flywheel');
    await badge.waitFor({ timeout: 4000 }).catch(() => undefined);
    check('карточка события: бейдж «в выборке маховика»', (await badge.count()) === 1);
    // клавиша 1 в карточке события не должна трогать сэмпл под ней
    await shot('04-event-badge');
    if (await badge.count()) {
      await badge.click();
      await page.waitForTimeout(1200);
      check('бейдж ведёт на сэмпл, карточка события закрыта', page.url().includes('sample=') && !page.url().includes('event='));
    }
  } else {
    check('ссылка на событие', false, 'нет сэмпла-события');
  }

  // ---------- живой сэмпл по WS
  await page.locator('label:has-text("Источник") select').selectOption('');
  await page.waitForTimeout(600);
  const totalBefore = Number((await page.locator('.fw-step').first().locator('.fw-step__note').innerText()).match(/всего\s([\d\s ]+)/)?.[1]?.replace(/\D/g, '') ?? 0);
  await page.waitForTimeout(13000);
  const totalAfter = Number((await page.locator('.fw-step').first().locator('.fw-step__note').innerText()).match(/всего\s([\d\s ]+)/)?.[1]?.replace(/\D/g, '') ?? 0);
  check('WS flywheel_sample: счётчик «всего» вырос', totalAfter > totalBefore, `${totalBefore} → ${totalAfter}`);
  check('WS flywheel_sample: новый сэмпл в очереди', (await page.locator('.fw-qitem.is-fresh').count()) > 0);

  // ---------- датасеты
  await page.getByRole('tab', { name: /Датасеты/ }).click();
  await page.waitForTimeout(800);
  const versionsBefore = await page.locator('.fw-versions__table tbody tr').count();
  await page.getByRole('button', { name: 'Собрать версию' }).click();
  await page.waitForTimeout(2200);
  const versionsAfter = await page.locator('.fw-versions__table tbody tr').count();
  check('«Собрать версию»: новая версия в списке', versionsAfter === versionsBefore + 1, `${versionsBefore} → ${versionsAfter}`);
  await shot('05-datasets');

  // ---------- модели и 403 на «Дообучить»
  await page.getByRole('tab', { name: /Модели/ }).click();
  await page.waitForTimeout(900);
  check('модели: карточки', (await page.locator('.fw-model').count()) >= 3);
  check('модели: решение гейта с причинами', (await page.locator('.fw-gate--fail li').count()) > 0 && (await page.locator('.fw-gate--pass').count()) > 0);
  await shot('06-models', 1500);
  const train = page.getByRole('button', { name: 'Дообучить' });
  await train.click();
  await page.waitForTimeout(900);
  check('«Дообучить» после 403 неактивна', await train.isDisabled());
  check('«Дообучить»: подсказка с причиной', (await page.locator('.fw-note--lock').innerText()).includes('GUARD_FLYWHEEL_ALLOW_TRAIN'));
  await page.evaluate(() => document.querySelector('.page')?.scrollTo(0, 0));
  await shot('07-train-forbidden');

  // ---------- остальные страницы живы
  for (const hash of ['#/', '#/journal', '#/analytics', '#/performance']) {
    await page.evaluate((h) => (window.location.hash = h), hash);
    await page.waitForTimeout(1200);
    check(`страница ${hash} открывается`, (await page.locator('.page').count()) > 0);
  }
} finally {
  await browser.close();
}

const uniq = [...new Set(errors)];
if (uniq.length) {
  console.log('--- ошибки страницы ---');
  for (const e of uniq) console.log(e);
}
const failed = results.filter((r) => !r.ok).length;
console.log(`\nИтого: ${results.length - failed} из ${results.length} проверок, ошибок страницы: ${uniq.length}`);
process.exit(failed || uniq.length ? 1 : 0);
