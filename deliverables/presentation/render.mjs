#!/usr/bin/env node
// Сборка презентации: index.html + data.json → zazerkalye.pdf (16:9, 1920×1080, слайд на страницу).
//
//   node render.mjs                 PDF в deliverables/presentation/zazerkalye.pdf
//   node render.mjs --png DIR       плюс PNG каждого слайда в DIR (для проверки вёрстки)
//   node render.mjs --out FILE      другой путь PDF
//   node render.mjs --serve         только локальный сервер для просмотра в браузере
//   node render.mjs --data FILE     подставить другой data.json (проверить, как лягут новые цифры)
//   node render.mjs --check         только проверки вёрстки и наложений, без PDF
//   node render.mjs --png DIR --only s10,b9   PNG только этих слайдов
//   --strict                        код выхода 1, если проверки что-то нашли
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
const DATA_FILE = opt("--data") ? path.resolve(opt("--data")) : null;

const MIME = {
  ".html": "text/html; charset=utf-8", ".css": "text/css; charset=utf-8", ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8", ".json": "application/json; charset=utf-8", ".woff2": "font/woff2",
  ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".svg": "image/svg+xml",
};

function startServer() {
  const server = http.createServer((req, res) => {
    const url = decodeURIComponent((req.url || "/").split("?")[0]);
    const file = (DATA_FILE && url === "/presentation/data.json") ? DATA_FILE : path.resolve(SERVE_ROOT, "." + url);
    if (file !== DATA_FILE && !file.startsWith(SERVE_ROOT + path.sep)) { res.writeHead(403); res.end(); return; }
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

// Проверка наложений: текст на текст, текст на линиях и стрелках SVG, текст за рамкой своей карточки,
// соседние блоки друг на друге, выход за слайд. Плюс то, что конвертер в PPTX не перенесёт: текст в ::before/::after.
function overlapCheck() {
  const out = [];
  const T = 2;
  const label = n => {
    const t = (n.textContent || "").trim().replace(/\s+/g, " ").slice(0, 40);
    const cls = typeof n.className === "string" ? n.className : (n.className && n.className.baseVal) || "";
    return (n.id ? "#" + n.id : n.tagName.toLowerCase() + (cls ? "." + cls.split(" ")[0] : "")) + (t ? " «" + t + "»" : "");
  };
  const area = (a, b) => Math.max(0, Math.min(a.right, b.right) - Math.max(a.left, b.left)) * Math.max(0, Math.min(a.bottom, b.bottom) - Math.max(a.top, b.top));
  const ovX = (a, b) => Math.min(a.right, b.right) - Math.max(a.left, b.left);
  const ovY = (a, b) => Math.min(a.bottom, b.bottom) - Math.max(a.top, b.top);
  const shown = n => { for (let e = n; e && e.nodeType === 1; e = e.parentElement) { const cs = getComputedStyle(e); if (cs.display === "none" || cs.visibility === "hidden" || cs.opacity === "0") return false; } return true; };
  const skip = n => !!n.closest(".chrome, .foot, svg.pixels, defs, marker, title");
  const isBox = n => {
    if (n instanceof SVGElement) return false;
    const cs = getComputedStyle(n);
    const bg = cs.backgroundColor, img = cs.backgroundImage;
    const hasBg = (bg && bg !== "transparent" && !/rgba\(.*,\s*0\)$/.test(bg)) || (img && img !== "none");
    const sides = ["Top", "Right", "Bottom", "Left"].filter(s => parseFloat(cs["border" + s + "Width"]) > 0 && cs["border" + s + "Style"] !== "none").length;
    return hasBg || sides >= 3;
  };
  document.querySelectorAll(".slide").forEach((sl, i) => {
    const sid = sl.id || "slide" + (i + 1);
    const S = sl.getBoundingClientRect();
    // 1. Прямоугольники строк текста
    const texts = [];
    const tw = document.createTreeWalker(sl, NodeFilter.SHOW_TEXT);
    for (let t; (t = tw.nextNode());) {
      const el = t.parentElement;
      if (!t.nodeValue.trim() || !el || skip(el) || !shown(el)) continue;
      let rects = [];
      if (el instanceof SVGElement) { const te = el.closest("text"); if (te) rects = [te.getBoundingClientRect()]; }
      else { const rg = document.createRange(); rg.selectNodeContents(t); rects = [...rg.getClientRects()]; }
      rects.filter(r => r.width > 1 && r.height > 1).forEach(r => texts.push({ el: el instanceof SVGElement ? el.closest("text") : el, r }));
    }
    // 1а. Текст на текст (разные элементы, не вложенные друг в друга)
    for (let a = 0; a < texts.length; a++) for (let b = a + 1; b < texts.length; b++) {
      const A = texts[a], B = texts[b];
      if (A.el === B.el || A.el.contains(B.el) || B.el.contains(A.el)) continue;
      const h = Math.min(A.r.height, B.r.height);
      if (ovX(A.r, B.r) > 4 && ovY(A.r, B.r) > 0.3 * h) out.push(sid + ": текст на тексте — " + label(A.el) + " × " + label(B.el));
    }
    // 1б. Текст за рамкой своей карточки (HTML-блок с фоном или рамкой; в SVG — rect своей группы)
    texts.forEach(({ el, r }) => {
      let box = null;
      if (el instanceof SVGElement) {
        const g = el.closest("g"); const rc = g && g.querySelector(":scope > rect");
        if (rc) box = rc.getBoundingClientRect();
      } else {
        for (let e = el; e && e !== sl; e = e.parentElement) if (isBox(e)) { box = e.getBoundingClientRect(); break; }
      }
      if (!box) return;
      const d = Math.max(box.left - r.left, r.right - box.right, box.top - r.top, r.bottom - box.bottom);
      if (d > T) out.push(sid + ": текст выходит за рамку карточки на " + Math.round(d) + "px — " + label(el));
    });
    // 2. Текст на линиях и стрелках SVG
    sl.querySelectorAll("svg").forEach(svg => {
      if (skip(svg)) return;
      const ctm = svg.getScreenCTM(); if (!ctm) return;
      svg.querySelectorAll("line, path, polyline").forEach(sh => {
        if (skip(sh)) return;
        const cs = getComputedStyle(sh);
        if (cs.stroke === "none" || !(parseFloat(cs.strokeWidth) > 0)) return;
        const m = sh.getScreenCTM() || ctm;
        const len = sh.getTotalLength ? sh.getTotalLength() : 0; if (!len) return;
        const arrow = sh.getAttribute("marker-end") || cs.markerEnd !== "none";
        const sw = parseFloat(cs.strokeWidth) * (m.a || 1);
        const pts = [];
        for (let s = 0; s <= len; s += 3) {
          const p = sh.getPointAtLength(s); const q = new DOMPoint(p.x, p.y).matrixTransform(m);
          pts.push({ x: q.x, y: q.y, pad: sw / 2 + (arrow && len - s < 8 * sw / 2 + 4 ? 4 * sw / 2 : 0) });
        }
        for (const { el, r } of texts) {
          if (!svg.contains(el) && area(r, svg.getBoundingClientRect()) === 0) continue;
          if (pts.some(p => p.x > r.left - p.pad + 1 && p.x < r.right + p.pad - 1 && p.y > r.top - p.pad + 1 && p.y < r.bottom + p.pad - 1)) {
            out.push(sid + ": текст на линии или стрелке — " + label(el));
          }
        }
      });
    });
    // 3. Соседние блоки друг на друге и выход за слайд
    sl.querySelectorAll("*").forEach(n => {
      if (skip(n) || !shown(n)) return;
      const r = n.getBoundingClientRect();
      if (r.width && r.height && !n.closest("svg") && (r.left < S.left - 1 || r.top < S.top - 1 || r.right > S.right + 1 || r.bottom > S.bottom + 1)) out.push(sid + ": за краем слайда — " + label(n));
      if (n instanceof SVGElement) return;
      const kids = [...n.children].filter(c => {
        if (c instanceof SVGElement && c.tagName !== "svg") return false;
        const cs = getComputedStyle(c);
        if (cs.position === "absolute" || cs.position === "fixed" || cs.display === "none" || cs.display === "contents") return false;
        if (cs.display.startsWith("inline") && !cs.display.includes("block") && !cs.display.includes("flex") && !cs.display.includes("grid")) return false;
        const b = c.getBoundingClientRect(); return b.width > 2 && b.height > 2;
      });
      for (let a = 0; a < kids.length; a++) for (let b = a + 1; b < kids.length; b++) {
        const A = kids[a].getBoundingClientRect(), B = kids[b].getBoundingClientRect();
        if (ovX(A, B) > T && ovY(A, B) > T) out.push(sid + ": блоки наложились — " + label(kids[a]) + " × " + label(kids[b]));
      }
      for (const p of ["::before", "::after"]) {
        const c = getComputedStyle(n, p).content;
        if (c && c !== "none" && c !== "normal" && c !== '""' && c !== "''") out.push(sid + ": текст в " + p + " (в PPTX не перенесётся) — " + label(n));
      }
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
    const over = await page.evaluate(overlapCheck);
    if (over.length) { console.warn("Наложения — проверьте:\n  " + over.join("\n  ")); }
    else console.log("Наложения: не найдено");
    if ((issues.length || over.length) && args.includes("--strict")) code = 1;

    const n = await page.locator(".slide").count();
    const ids = await page.locator(".slide").evaluateAll(els => els.map(e => e.id));
    if (PNG_DIR) {
      fs.mkdirSync(PNG_DIR, { recursive: true });
      const only = typeof opt("--only") === "string" ? opt("--only").split(",") : null;
      let k = 0;
      for (let i = 0; i < n; i++) {
        if (only && !only.includes(ids[i])) continue;
        const f = path.join(PNG_DIR, "slide-" + String(i + 1).padStart(2, "0") + "-" + ids[i] + ".png");
        await page.locator(".slide").nth(i).screenshot({ path: f });
        k++;
      }
      console.log("PNG: " + k + " шт. в " + PNG_DIR);
    }
    if (args.includes("--check")) { console.log("Слайдов: " + n + " (" + ids.join(" ") + ")"); return; }
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
