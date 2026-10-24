#!/usr/bin/env node
// Сборка презентации: index.html + data.json → zazerkalye.pdf (16:9, 1920×1080, слайд на страницу).
//
//   node render.mjs                 PDF в deliverables/presentation/zazerkalye.pdf
//   node render.mjs --png DIR       плюс PNG каждого слайда в DIR (для проверки вёрстки)
//   node render.mjs --out FILE      другой путь PDF
//   node render.mjs --serve         только локальный сервер для просмотра в браузере
//
// Playwright берётся из frontend/node_modules (cd frontend && npm ci) или из PLAYWRIGHT_DIR.
// На ноутбуке команды запускать через tools/heavy.sh: браузер — тяжёлый процесс.
import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const SERVE_ROOT = path.resolve(HERE, "..");            // deliverables/: нужны ../screens
const REPO = path.resolve(HERE, "..", "..");
const args = process.argv.slice(2);
const opt = name => { const i = args.indexOf(name); return i >= 0 ? (args[i + 1] || true) : null; };
const OUT = path.resolve(opt("--out") || path.join(HERE, "zazerkalye.pdf"));
const PNG_DIR = opt("--png");

const MIME = {
  ".html": "text/html; charset=utf-8", ".css": "text/css; charset=utf-8", ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8", ".json": "application/json; charset=utf-8", ".woff2": "font/woff2",
  ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".svg": "image/svg+xml",
};

function startServer() {
  const server = http.createServer((req, res) => {
    const url = decodeURIComponent((req.url || "/").split("?")[0]);
    const file = path.resolve(SERVE_ROOT, "." + url);
    if (!file.startsWith(SERVE_ROOT + path.sep)) { res.writeHead(403); res.end(); return; }
    fs.readFile(file, (err, buf) => {
      if (err) { res.writeHead(404); res.end("not found"); return; }
      res.writeHead(200, { "content-type": MIME[path.extname(file).toLowerCase()] || "application/octet-stream", "cache-control": "no-store" });
      res.end(buf);
    });
  });
  return new Promise(resolve => server.listen(0, "127.0.0.1", () => resolve(server)));
}

function loadPlaywright() {
  const tries = [];
  if (process.env.PLAYWRIGHT_DIR) tries.push(path.resolve(process.env.PLAYWRIGHT_DIR));
  tries.push("playwright");
  const req = createRequire(path.join(REPO, "frontend", "package.json"));
  for (const t of tries) {
    try { return req(t); } catch { /* следующий вариант */ }
  }
  throw new Error("playwright не найден: выполните `cd frontend && npm ci` или задайте PLAYWRIGHT_DIR=/путь/к/node_modules/playwright");
}

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
  const server = await startServer();
  const base = "http://127.0.0.1:" + server.address().port + "/presentation/index.html";
  if (args.includes("--serve")) {
    console.log("Слайды: " + base + "  (Ctrl+C — остановить)");
    return;
  }
  const { chromium } = loadPlaywright();
  const browser = await chromium.launch({ headless: true });
  let code = 0;
  try {
    const page = await browser.newPage({ viewport: { width: 1920, height: 1080 }, deviceScaleFactor: 1 });
    page.on("pageerror", e => { console.error("Ошибка JS:", e.message); code = 1; });
    await page.goto(base, { waitUntil: "load" });
    await page.waitForSelector("body[data-ready='1']", { timeout: 30000 });
    const err = await page.getAttribute("body", "data-error");
    if (err) throw new Error("страница не собралась: " + err);
    await page.evaluate(() => Promise.all([...document.images].map(im => im.complete ? 0 : new Promise(r => { im.onload = im.onerror = r; }))));
    const bad = await page.evaluate(() => [...document.images].filter(im => !im.naturalWidth).map(im => im.getAttribute("src")));
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
