// Общий код сборки презентации (render.mjs → PDF, to_pptx.mjs → PPTX):
// локальный HTTP-сервер над deliverables/, поиск Playwright и загрузка колоды в Chromium.
import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";

export const HERE = path.dirname(fileURLToPath(import.meta.url));
export const SERVE_ROOT = path.resolve(HERE, "..");            // deliverables/: нужны ../screens
export const REPO = path.resolve(HERE, "..", "..");
export const DECK_PATH = "/presentation/index.html";
export const SLIDE_W = 1920;
export const SLIDE_H = 1080;

const MIME = {
  ".html": "text/html; charset=utf-8", ".css": "text/css; charset=utf-8", ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8", ".json": "application/json; charset=utf-8", ".woff2": "font/woff2",
  ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".svg": "image/svg+xml",
};

/** Значение опции командной строки: `--out FILE` → "FILE", флаг без значения → true, нет опции → null. */
export function cliOption(args, name) {
  const i = args.indexOf(name);
  if (i < 0) return null;
  const v = args[i + 1];
  return v && !v.startsWith("--") ? v : true;
}

/**
 * Статический сервер на 127.0.0.1 со случайным портом.
 * dataFile подменяет /presentation/data.json (проверить, как лягут новые цифры).
 */
export function startServer({ root = SERVE_ROOT, dataFile = null } = {}) {
  const server = http.createServer((req, res) => {
    const url = decodeURIComponent((req.url || "/").split("?")[0]);
    const file = (dataFile && url === "/presentation/data.json") ? dataFile : path.resolve(root, "." + url);
    if (file !== dataFile && !file.startsWith(root + path.sep)) { res.writeHead(403); res.end(); return; }
    fs.readFile(file, (err, buf) => {
      if (err) { res.writeHead(404); res.end("not found"); return; }
      res.writeHead(200, { "content-type": MIME[path.extname(file).toLowerCase()] || "application/octet-stream", "cache-control": "no-store" });
      res.end(buf);
    });
  });
  return new Promise(resolve => server.listen(0, "127.0.0.1", () => resolve(server)));
}

/** URL на локальном сервере → путь к файлу на диске (или null, если URL чужой). */
export function urlToFile(server, url, { root = SERVE_ROOT, dataFile = null } = {}) {
  let u;
  try { u = new URL(url); } catch { return null; }
  if (u.hostname !== "127.0.0.1" || Number(u.port) !== server.address().port) return null;
  const p = decodeURIComponent(u.pathname);
  if (dataFile && p === "/presentation/data.json") return dataFile;
  const file = path.resolve(root, "." + p);
  return file.startsWith(root + path.sep) && fs.existsSync(file) ? file : null;
}

/** Playwright из PLAYWRIGHT_DIR или из frontend/node_modules. */
export function loadPlaywright() {
  const tries = [];
  if (process.env.PLAYWRIGHT_DIR) tries.push(path.resolve(process.env.PLAYWRIGHT_DIR));
  tries.push("playwright");
  const req = createRequire(path.join(REPO, "frontend", "package.json"));
  for (const t of tries) {
    try { return req(t); } catch { /* следующий вариант */ }
  }
  throw new Error("playwright не найден: выполните `cd frontend && npm ci` или задайте PLAYWRIGHT_DIR=/путь/к/node_modules/playwright");
}

/**
 * Открывает колоду и ждёт, пока deck.js подставит данные (body[data-ready='1']), шрифты и картинки.
 * readySelector=null — для произвольной страницы без deck.js.
 * Возвращает { page, badImages }; ошибки JS на странице передаются в onPageError.
 */
export async function openDeck(browser, url, { deviceScaleFactor = 1, readySelector = "body[data-ready='1']", onPageError = null } = {}) {
  const page = await browser.newPage({ viewport: { width: SLIDE_W, height: SLIDE_H }, deviceScaleFactor });
  if (onPageError) page.on("pageerror", onPageError);
  await page.goto(url, { waitUntil: "load" });
  if (readySelector) {
    await page.waitForSelector(readySelector, { timeout: 30000 });
    const err = await page.getAttribute("body", "data-error");
    if (err) throw new Error("страница не собралась: " + err);
  }
  await page.evaluate(() => document.fonts.ready);
  await page.evaluate(() => Promise.all([...document.images].map(im => im.complete ? 0 : new Promise(r => { im.onload = im.onerror = r; }))));
  const badImages = await page.evaluate(() => [...document.images].filter(im => !im.naturalWidth).map(im => im.getAttribute("src")));
  return { page, badImages };
}
