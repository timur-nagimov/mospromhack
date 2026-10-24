#!/usr/bin/env node
// HTML-колода → редактируемый PPTX (16:9, 13,333 × 7,5 дюйма; 1920 px = 13,333 дюйма, 1 px = 0,5 pt).
//
//   node to_pptx.mjs                  → deliverables/presentation/zazerkalye.pptx
//   node to_pptx.mjs --out FILE       другой путь
//   node to_pptx.mjs --data FILE      подставить другой data.json
//   node to_pptx.mjs --slides 1,3-5   только эти слайды (отладка)
//   node to_pptx.mjs --report FILE    JSON-отчёт: видимый текст HTML по слайдам, число фигур, предупреждения
//   node to_pptx.mjs --dump DIR       модели слайдов и снимки для отладки
//   node to_pptx.mjs --no-tables      таблицы фигурами и текстом поверх, без нативных таблиц PPTX
//
// Каждый слайд открывается в Chromium (1920×1080) и обходится в порядке отрисовки CSS (pptx/dom_extract.js):
// текст — нативные текстовые фигуры с переносами как в браузере; плашки и рамки — фигуры; таблицы — таблицы;
// <img> — картинки. Градиенты, тени, маски, SVG-графики — снимком элемента со скрытым текстом, текст сверху.
// Нужны Node 20, pptxgenjs (npm ci в этой папке), Playwright с Chromium (как для render.mjs)
// и шрифты из fonts-ttf/ (см. README.md). На ноутбуке запускать через tools/heavy.sh.
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { createRequire } from "node:module";
import { HERE, DECK_PATH, SLIDE_W, cliOption, startServer, loadPlaywright, openDeck, urlToFile } from "./deck_page.mjs";
import { makeFontResolver, opXml, replacePlaceholders } from "./pptx/ooxml.mjs";

const args = process.argv.slice(2);
const opt = name => cliOption(args, name);
const OUT = path.resolve(opt("--out") || path.join(HERE, "zazerkalye.pptx"));
const DATA_FILE = opt("--data") ? path.resolve(opt("--data")) : null;
const REPORT = opt("--report") ? path.resolve(opt("--report")) : null;
const DUMP = opt("--dump") ? path.resolve(opt("--dump")) : null;
const SLIDES = opt("--slides");
const NATIVE_TABLES = !args.includes("--no-tables");
const SCALE = 2;                       // снимки элементов — в 2× (чёткие на проекторе), фон слайда — 1×
const PX_PER_IN = SLIDE_W / 13.333333; // 144

// семейства CSS → семейства в PPTX (fonts-ttf/fonts.json); остальные имена — как в CSS
const FONT_ALIASES = { JBMono: "JetBrains Mono", "sans-serif": "Onest", monospace: "JetBrains Mono", serif: "Onest" };

const require = createRequire(import.meta.url);
function need(mod) {
  try { return require(mod); } catch {
    throw new Error(`${mod} не найден: выполните \`cd deliverables/presentation && npm ci\``);
  }
}

function parseSlides(spec, n) {
  if (!spec || spec === true) return [...Array(n).keys()];
  const out = new Set();
  for (const part of String(spec).split(",")) {
    const [a, b] = part.split("-").map(s => parseInt(s, 10));
    for (let i = a; i <= (b || a); i++) if (i >= 1 && i <= n) out.add(i - 1);
  }
  return [...out].sort((x, y) => x - y);
}

function fontsInstalled(manifest) {
  const dirs = [path.join(os.homedir(), "Library", "Fonts"), "/Library/Fonts", path.join(os.homedir(), ".local", "share", "fonts"), "/usr/share/fonts"];
  const files = Object.values(manifest).flatMap(w => Object.values(w).map(x => x.file));
  return files.filter(f => !dirs.some(d => fs.existsSync(path.join(d, f))));
}

const inch = px => px / PX_PER_IN;

async function main() {
  const PptxGenJS = need("pptxgenjs");
  const JSZip = need("jszip");
  const manifest = JSON.parse(fs.readFileSync(path.join(HERE, "fonts-ttf", "fonts.json"), "utf8"));
  const fonts = makeFontResolver(manifest, FONT_ALIASES);
  const missing = fontsInstalled(manifest);
  if (missing.length) console.warn(`Шрифты не установлены (${missing.length}): PowerPoint подставит другие. См. README.md, «Шрифты».`);

  const server = await startServer({ dataFile: DATA_FILE });
  const url = "http://127.0.0.1:" + server.address().port + DECK_PATH;
  const { chromium } = loadPlaywright();
  const browser = await chromium.launch({ headless: true });
  let code = 0;
  const t0 = Date.now();
  try {
    const { page, badImages } = await openDeck(browser, url, {
      deviceScaleFactor: SCALE, onPageError: e => { console.error("Ошибка JS:", e.message); code = 1; },
    });
    if (badImages.length) { console.error("Не загрузились картинки:", badImages.join(", ")); code = 1; }
    await page.emulateMedia({ media: "print" });   // как PDF из render.mjs
    await page.addScriptTag({ path: path.join(HERE, "pptx", "dom_extract.js") });
    const total = await page.evaluate(() => window.__pptx.count());
    const indexes = parseSlides(SLIDES, total);
    if (DUMP) fs.mkdirSync(DUMP, { recursive: true });

    const pptx = new PptxGenJS();
    pptx.layout = "LAYOUT_WIDE";
    pptx.title = await page.title();
    pptx.author = "Команда «Зазеркалье»";
    pptx.company = "Зазеркалье";
    const marksBySlide = [];
    const report = { slides: [], total, out: OUT };
    let seq = 0;

    for (const i of indexes) {
      const model = await page.evaluate(([idx, o]) => window.__pptx.extractSlide(idx, o), [i, { nativeTables: NATIVE_TABLES }]);
      const shots = new Map();
      for (const cap of model.captures) {
        await page.evaluate(([id, mode]) => window.__pptx.isolate(id, mode), [cap.id, cap.mode]);
        const clip = { x: cap.clip.x, y: cap.clip.y, width: cap.clip.w, height: cap.clip.h };
        const shot = { clip, type: cap.format, scale: cap.scale, animations: "disabled", caret: "hide" };
        if (cap.format === "jpeg") shot.quality = 90; else shot.omitBackground = true;
        shots.set(cap.id, await page.screenshot(shot));
        await page.evaluate(() => window.__pptx.unisolate());
      }
      if (DUMP) {
        fs.writeFileSync(path.join(DUMP, `slide-${String(i + 1).padStart(2, "0")}.json`), JSON.stringify(model, null, 1));
        for (const [id, buf] of shots) fs.writeFileSync(path.join(DUMP, `s${i + 1}-cap${id}.${model.captures.find(c => c.id === id).format === "jpeg" ? "jpg" : "png"}`), buf);
      }

      const slide = pptx.addSlide();
      const marks = new Map();
      const counts = { shape: 0, line: 0, text: 0, table: 0, image: 0, raster: 0 };
      if (model.bg && shots.has(model.bg.cap)) slide.background = { data: "image/jpeg;base64," + shots.get(model.bg.cap).toString("base64") };
      else if (model.bg) slide.background = { color: model.bg.color };
      for (const op of model.ops) {
        if (op.kind === "bg") continue;
        counts[op.kind] = (counts[op.kind] || 0) + 1;
        const pos = { x: inch(op.x), y: inch(op.y), w: inch(op.w || 1), h: inch(op.h || 1) };
        if (op.kind === "raster") {
          const buf = shots.get(op.cap);
          if (!buf) continue;
          const mime = model.captures.find(c => c.id === op.cap).format === "jpeg" ? "image/jpeg" : "image/png";
          slide.addImage({ data: `${mime};base64,${buf.toString("base64")}`, ...pos, altText: op.name, objectName: op.name });
        } else if (op.kind === "image") {
          const file = urlToFile(server, op.src, { dataFile: DATA_FILE });
          if (file) slide.addImage({ path: file, ...pos, altText: op.name, objectName: op.name });
          else {
            const b64 = await page.evaluate(async src => {
              const r = await fetch(src); const b = new Uint8Array(await r.arrayBuffer());
              let s = ""; for (let k = 0; k < b.length; k += 0x8000) s += String.fromCharCode(...b.subarray(k, k + 0x8000));
              return { type: r.headers.get("content-type") || "image/png", data: btoa(s) };
            }, op.src);
            slide.addImage({ data: `${b64.type};base64,${b64.data}`, ...pos, altText: op.name, objectName: op.name });
          }
        } else {
          const mark = `PPTXMARK${++seq}Z`;
          marks.set(mark, opXml(op, fonts));
          slide.addText(mark, { x: 0, y: 0, w: 1, h: 1 });
        }
      }
      marksBySlide.push(marks);
      report.slides.push({ n: i + 1, id: model.id, innerText: model.innerText, textChars: model.textChars, counts, warnings: model.warnings, ms: model.ms });
      const w = model.warnings.length ? "  ⚠ " + model.warnings.slice(0, 3).join("; ") : "";
      console.log(`слайд ${String(i + 1).padStart(2)} ${model.id.padEnd(4)} текст ${counts.text} табл ${counts.table} фигур ${counts.shape + counts.line} картинок ${counts.image} снимков ${counts.raster}${w}`);
    }

    // pptxgenjs собирает пакет, затем заглушки заменяются нашим XML
    const raw = await pptx.write({ outputType: "nodebuffer" });
    const zip = await JSZip.loadAsync(raw);
    for (let k = 0; k < marksBySlide.length; k++) {
      const name = `ppt/slides/slide${k + 1}.xml`;
      const xml = await zip.file(name).async("string");
      zip.file(name, replacePlaceholders(xml, marksBySlide[k]));
    }
    const buf = await zip.generateAsync({ type: "nodebuffer", compression: "DEFLATE", compressionOptions: { level: 6 } });
    fs.mkdirSync(path.dirname(OUT), { recursive: true });
    fs.writeFileSync(OUT, buf);
    if (REPORT) fs.writeFileSync(REPORT, JSON.stringify(report, null, 1));
    console.log(`PPTX: ${OUT} (${indexes.length} слайдов, ${(buf.length / 1048576).toFixed(1)} МБ, ${((Date.now() - t0) / 1000).toFixed(0)} с)`);
  } finally {
    await browser.close();
    server.close();
  }
  process.exitCode = code;
}

main().catch(e => { console.error(e.stack || e.message || e); process.exit(1); });

