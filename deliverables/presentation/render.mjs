#!/usr/bin/env node
// Сборка презентации: index.html + data.json → zazerkalye.pdf (16:9, 1920×1080, слайд на страницу).
//
//   node render.mjs                 PDF в deliverables/presentation/zazerkalye.pdf
//   node render.mjs --png DIR       плюс PNG каждого слайда в DIR (для проверки вёрстки)
//   node render.mjs --out FILE      другой путь PDF
//   node render.mjs --serve         только локальный сервер для просмотра в браузере
//   node render.mjs --data FILE     подставить другой data.json (проверить, как лягут новые цифры)
//
// Редактируемый PPTX из той же колоды — to_pptx.mjs (сервер и загрузка общие, deck_page.mjs).
// Playwright берётся из frontend/node_modules (cd frontend && npm ci) или из PLAYWRIGHT_DIR.
// На ноутбуке команды запускать через tools/heavy.sh: браузер — тяжёлый процесс.
import fs from "node:fs";
import path from "node:path";
import { HERE, DECK_PATH, cliOption, startServer, loadPlaywright, openDeck } from "./deck_page.mjs";

const args = process.argv.slice(2);
const opt = name => cliOption(args, name);
const OUT = path.resolve(opt("--out") || path.join(HERE, "zazerkalye.pdf"));
const PNG_DIR = opt("--png");
const DATA_FILE = opt("--data") ? path.resolve(opt("--data")) : null;

// Проверка вёрстки: что вылезло за .body слайда или обрезано overflow:hidden.
function layoutCheck() {
  const out = [];
  const label = n => (n.id ? "#" + n.id : n.tagName.toLowerCase() + (n.className && typeof n.className === "string" ? "." + n.className.split(" ")[0] : "")) +
    " «" + (n.textContent || "").trim().replace(/\s+/g, " ").slice(0, 50) + "»";
  document.querySelectorAll(".slide").forEach((sl, i) => {
    const sid = sl.id || "slide" + (i + 1);
    const src = sl.querySelector(".foot .src");
    if (src && src.scrollHeight > src.clientHeight + 1) out.push(sid + ": источник в подвале не влез в 2 строки");
    const S = sl.getBoundingClientRect();
    const body = sl.querySelector(".body");
    const B = body ? body.getBoundingClientRect() : null;
    const limitBottom = B ? B.bottom : S.bottom - 110;
    const limitRight = B ? B.right : S.right - 96;
    sl.querySelectorAll("*").forEach(n => {
      if (n.closest(".chrome") || n.closest(".foot") || n.closest("svg.pixels")) return;
      const r = n.getBoundingClientRect();
      if (!r.width || !r.height) return;
      if (r.right > limitRight + 1) out.push(sid + ": за правым краем контента на " + Math.round(r.right - limitRight) + "px — " + label(n));
      if (body && body.contains(n) && r.bottom > limitBottom + 1) out.push(sid + ": ниже области контента на " + Math.round(r.bottom - limitBottom) + "px — " + label(n));
      const cs = getComputedStyle(n);
      if ((cs.overflow === "hidden" || cs.overflowX === "hidden") && n.tagName !== "svg" && !n.classList.contains("slide") && !n.classList.contains("win") && !n.classList.contains("pane") && !n.classList.contains("th") && !n.classList.contains("feed")) {
        if (n.scrollWidth > n.clientWidth + 1) out.push(sid + ": обрезан текст по ширине — " + label(n));
      }
      if (n.tagName === "PRE" && n.parentElement && n.scrollWidth > n.parentElement.clientWidth) out.push(sid + ": код шире окна — " + label(n));
    });
  });
  return [...new Set(out)];
}

async function main() {
  const server = await startServer({ dataFile: DATA_FILE });
  const base = "http://127.0.0.1:" + server.address().port + DECK_PATH;
  if (args.includes("--serve")) {
    console.log("Слайды: " + base + "  (Ctrl+C — остановить)");
    return;
  }
  const { chromium } = loadPlaywright();
  const browser = await chromium.launch({ headless: true });
  let code = 0;
  try {
    const { page, badImages: bad } = await openDeck(browser, base, { onPageError: e => { console.error("Ошибка JS:", e.message); code = 1; } });
    if (bad.length) { console.error("Не загрузились картинки:", bad.join(", ")); code = 1; }

    const issues = await page.evaluate(layoutCheck);
    if (issues.length) { console.warn("Вёрстка — проверьте:\n  " + issues.join("\n  ")); }
    else console.log("Вёрстка: переполнений не найдено");

    const n = await page.locator(".slide").count();
    if (PNG_DIR) {
      fs.mkdirSync(PNG_DIR, { recursive: true });
      for (let i = 0; i < n; i++) {
        const f = path.join(PNG_DIR, "slide-" + String(i + 1).padStart(2, "0") + ".png");
        await page.locator(".slide").nth(i).screenshot({ path: f });
      }
      console.log("PNG: " + n + " шт. в " + PNG_DIR);
    }
    await page.emulateMedia({ media: "print" });
    await page.pdf({ path: OUT, width: "1920px", height: "1080px", printBackground: true, preferCSSPageSize: true });
    console.log("PDF: " + OUT + " (" + n + " слайдов)");
  } finally {
    await browser.close();
    server.close();
  }
  process.exitCode = code;
}

main().catch(e => { console.error(e.message || e); process.exit(1); });
