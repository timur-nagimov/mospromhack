// DrawingML для операций из dom_extract.js: фигуры, линии, текстовые блоки, таблицы.
// pptxgenjs собирает пакет (слайды, картинки, фон), а фигуры, текст и таблицы вставляются этим XML
// вместо меток-заглушек: так под контролем каждый атрибут (поля 0, точный межстрочный, интервал, alpha).
//
// Единицы: CSS px слайда 1920×1080 → EMU (1 px = 6350 EMU, ширина 12 192 000 EMU = 13,333 дюйма);
// 1 px = 0,5 pt, поэтому кегль sz (сотые пункта) = px × 50, интервалы spcPts и spc — тоже px × 50.

export const EMU_PER_PX = 6350;
const emu = px => Math.round(px * EMU_PER_PX);
const pt100 = px => Math.round(px * 50);

export const esc = s => String(s)
  .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;")
  // управляющие символы недопустимы в XML 1.0
  .replace(/[\u0000-\u0008\u000B\u000C\u000E-\u001F]/g, "");

/** Выбор гарнитуры по списку семейств CSS и толщине: ближайшее статическое начертание из fonts-ttf/fonts.json. */
export function makeFontResolver(manifest, aliases = {}) {
  const cache = new Map();
  return (families, weight) => {
    const key = families.join("|") + "/" + weight;
    if (cache.has(key)) return cache.get(key);
    let res = null;
    for (const raw of families) {
      const fam = aliases[raw] || aliases[raw.toLowerCase()] || raw;
      const w = manifest[fam];
      if (!w) continue;
      const best = Object.keys(w).map(Number).reduce((a, b) => {
        const da = Math.abs(a - weight), db = Math.abs(b - weight);
        return db < da || (db === da && b > a) ? b : a;
      });
      res = { typeface: w[best].typeface, bold: !!w[best].bold };
      break;
    }
    if (!res) {
      const raw = families[0] || "Onest";
      res = { typeface: aliases[raw] || raw, bold: weight >= 600 };
    }
    cache.set(key, res);
    return res;
  };
}

function colorXml({ color, alpha }) {
  const a = alpha == null || alpha >= 0.999 ? "" : `<a:alpha val="${Math.round(alpha * 100000)}"/>`;
  return `<a:srgbClr val="${color}">${a}</a:srgbClr>`;
}

const fillXml = p => (p ? `<a:solidFill>${colorXml(p)}</a:solidFill>` : "<a:noFill/>");

function lnXml(line, tag = "a:ln") {
  if (!line) return `<${tag}><a:noFill/></${tag}>`;
  return `<${tag} w="${emu(line.width)}" cap="flat" cmpd="sng" algn="ctr">${fillXml(line)}<a:prstDash val="${line.dash || "solid"}"/><a:miter lim="800000"/></${tag}>`;
}

const xfrm = (x, y, w, h) => `<a:xfrm><a:off x="${emu(x)}" y="${emu(y)}"/><a:ext cx="${Math.max(1, emu(w))}" cy="${Math.max(1, emu(h))}"/></a:xfrm>`;

/** Прямоугольник с отдельным радиусом каждого угла (CSS border-radius у части углов). */
function cornersGeom(w, h, radii) {
  const [tl, tr, br, bl] = radii.map(r => emu(Math.min(r, w / 2, h / 2)));
  const W = emu(w), H = emu(h);
  const arc = (r, st) => (r ? `<a:arcTo wR="${r}" hR="${r}" stAng="${st}" swAng="5400000"/>` : "");
  const d = `<a:moveTo><a:pt x="0" y="${tl}"/></a:moveTo>${arc(tl, 10800000)}` +
    `<a:lnTo><a:pt x="${W - tr}" y="0"/></a:lnTo>${arc(tr, 16200000)}` +
    `<a:lnTo><a:pt x="${W}" y="${H - br}"/></a:lnTo>${arc(br, 0)}` +
    `<a:lnTo><a:pt x="${bl}" y="${H}"/></a:lnTo>${arc(bl, 5400000)}<a:close/>`;
  return `<a:custGeom><a:avLst/><a:gdLst/><a:ahLst/><a:cxnLst/><a:rect l="l" t="t" r="r" b="b"/><a:pathLst><a:path w="${W}" h="${H}">${d}</a:path></a:pathLst></a:custGeom>`;
}

function geomXml(op) {
  if (op.geom === "corners") return cornersGeom(op.w, op.h, op.radii);
  if (op.geom === "roundRect" && op.radius > 0) {
    const adj = Math.min(50000, Math.round(op.radius / Math.max(1e-6, Math.min(op.w, op.h)) * 100000));
    return `<a:prstGeom prst="roundRect"><a:avLst><a:gd name="adj" fmla="val ${adj}"/></a:avLst></a:prstGeom>`;
  }
  return `<a:prstGeom prst="rect"><a:avLst/></a:prstGeom>`;
}

export function shapeXml(op, id) {
  return `<p:sp><p:nvSpPr><p:cNvPr id="${id}" name="${esc(op.name || "Фигура")}"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr>` +
    `<p:spPr>${xfrm(op.x, op.y, op.w, op.h)}${geomXml(op)}${fillXml(op.fill)}${lnXml(op.line)}</p:spPr></p:sp>`;
}

export function lineShapeXml(op, id) {
  const x = Math.min(op.x1, op.x2), y = Math.min(op.y1, op.y2);
  const w = Math.abs(op.x2 - op.x1), h = Math.abs(op.y2 - op.y1);
  return `<p:cxnSp><p:nvCxnSpPr><p:cNvPr id="${id}" name="${esc(op.name || "Линия")}"/><p:cNvCxnSpPr/><p:nvPr/></p:nvCxnSpPr>` +
    `<p:spPr>${xfrm(x, y, w, h)}<a:prstGeom prst="line"><a:avLst/></a:prstGeom>${lnXml(op)}</p:spPr></p:cxnSp>`;
}

// ---------------------------------------------------------------- текст
function rPrXml(run, fonts, tag = "a:rPr") {
  const f = fonts(run.families, run.weight);
  let a = `lang="ru-RU" altLang="en-US" sz="${Math.max(100, pt100(run.size))}" b="${f.bold ? 1 : 0}" i="${run.italic ? 1 : 0}" kern="100"`;
  if (run.underline) a += ` u="sng"`;
  if (run.strike) a += ` strike="sngStrike"`;
  if (run.spacing) a += ` spc="${pt100(run.spacing)}"`;
  if (run.baseline) a += ` baseline="${run.baseline}"`;
  const fill = run.color ? fillXml({ color: run.color, alpha: run.alpha }) : "";
  const tf = esc(f.typeface);
  return `<${tag} ${a} dirty="0">${fill}<a:latin typeface="${tf}"/><a:ea typeface="${tf}"/><a:cs typeface="${tf}"/></${tag}>`;
}

const ALIGN = { l: "l", ctr: "ctr", r: "r", just: "just" };

function paragraphXml(p, fonts) {
  let x = `<a:p><a:pPr marL="0" indent="0" algn="${ALIGN[p.align] || "l"}">` +
    `<a:lnSpc><a:spcPts val="${Math.max(1, pt100(p.lineSpacing))}"/></a:lnSpc>` +
    `<a:spcBef><a:spcPts val="${pt100(p.spaceBefore || 0)}"/></a:spcBef><a:spcAft><a:spcPts val="0"/></a:spcAft><a:buNone/></a:pPr>`;
  let last = null;
  p.lines.forEach((runs, i) => {
    if (i > 0) x += `<a:br>${rPrXml(runs[0] || last, fonts)}</a:br>`;
    for (const r of runs) { x += `<a:r>${rPrXml(r, fonts)}<a:t>${esc(r.text)}</a:t></a:r>`; last = r; }
  });
  x += last ? rPrXml({ ...last, spacing: 0, baseline: 0 }, fonts, "a:endParaRPr") : `<a:endParaRPr lang="ru-RU" sz="100" dirty="0"/>`;
  return x + "</a:p>";
}

export function paragraphsXml(paragraphs, fonts) {
  if (!paragraphs || !paragraphs.length) {
    return `<a:p><a:pPr marL="0" indent="0"><a:lnSpc><a:spcPts val="100"/></a:lnSpc><a:buNone/></a:pPr><a:endParaRPr lang="ru-RU" sz="100" dirty="0"/></a:p>`;
  }
  return paragraphs.map(p => paragraphXml(p, fonts)).join("");
}

export function textShapeXml(op, id, fonts) {
  const body = `<p:txBody><a:bodyPr wrap="square" lIns="0" tIns="0" rIns="0" bIns="0" rtlCol="0" anchor="t" anchorCtr="0"><a:noAutofit/></a:bodyPr>` +
    `<a:lstStyle/>${paragraphsXml(op.paragraphs, fonts)}</p:txBody>`;
  return `<p:sp><p:nvSpPr><p:cNvPr id="${id}" name="${esc(op.name || "Текст")}"/><p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr>` +
    `<p:spPr>${xfrm(op.x, op.y, op.w, op.h)}<a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:noFill/></p:spPr>${body}</p:sp>`;
}

// ---------------------------------------------------------------- таблицы
function cellLn(tag, b) {
  if (!b) return `<${tag} w="0" cap="flat" cmpd="sng" algn="ctr"><a:noFill/></${tag}>`;
  return `<${tag} w="${emu(b.width)}" cap="flat" cmpd="sng" algn="ctr">${fillXml(b)}<a:prstDash val="${b.dash || "solid"}"/><a:round/>` +
    `<a:headEnd type="none" w="med" len="med"/><a:tailEnd type="none" w="med" len="med"/></${tag}>`;
}

function cellXml(cell, fonts, attrs = "") {
  const m = cell ? cell.margins : { T: 0, R: 0, B: 0, L: 0 };
  const anchor = cell ? ({ t: "t", ctr: "ctr", b: "b" }[cell.anchor] || "t") : "t";
  const b = cell ? cell.borders : [null, null, null, null];    // [T, R, B, L]
  const tcPr = `<a:tcPr marL="${emu(m.L)}" marR="${emu(m.R)}" marT="${emu(m.T)}" marB="${emu(m.B)}" anchor="${anchor}">` +
    cellLn("a:lnL", b[3]) + cellLn("a:lnR", b[1]) + cellLn("a:lnT", b[0]) + cellLn("a:lnB", b[2]) +
    fillXml(cell && cell.fill) + `</a:tcPr>`;
  const body = `<a:txBody><a:bodyPr/><a:lstStyle/>${paragraphsXml(cell && cell.paragraphs, fonts)}</a:txBody>`;
  return `<a:tc${attrs}>${body}${tcPr}</a:tc>`;
}

export function tableXml(op, id, fonts) {
  const nr = op.rows.length, nc = op.cols.length;
  const grid = Array.from({ length: nr }, () => new Array(nc).fill(null));
  for (const c of op.cells) {
    for (let r = c.r; r < c.r + c.rs; r++) for (let k = c.c; k < c.c + c.cs; k++) {
      if (r < nr && k < nc) grid[r][k] = { owner: c, h: k > c.c, v: r > c.r };
    }
  }
  let rows = "";
  for (let r = 0; r < nr; r++) {
    rows += `<a:tr h="${emu(op.rows[r])}">`;
    for (let k = 0; k < nc; k++) {
      const g = grid[r][k];
      if (!g) { rows += cellXml(null, fonts); continue; }
      if (!g.h && !g.v) {
        const c = g.owner;
        const attrs = (c.cs > 1 ? ` gridSpan="${c.cs}"` : "") + (c.rs > 1 ? ` rowSpan="${c.rs}"` : "");
        rows += cellXml(c, fonts, attrs);
      } else {
        rows += cellXml({ ...g.owner, paragraphs: null }, fonts, (g.h ? ` hMerge="1"` : "") + (g.v ? ` vMerge="1"` : ""));
      }
    }
    rows += `</a:tr>`;
  }
  const gridXml = op.cols.map(w => `<a:gridCol w="${emu(w)}"/>`).join("");
  return `<p:graphicFrame><p:nvGraphicFramePr><p:cNvPr id="${id}" name="${esc(op.name || "Таблица")}"/>` +
    `<p:cNvGraphicFramePr><a:graphicFrameLocks noGrp="1"/></p:cNvGraphicFramePr><p:nvPr/></p:nvGraphicFramePr>` +
    `<p:xfrm><a:off x="${emu(op.x)}" y="${emu(op.y)}"/><a:ext cx="${emu(op.w)}" cy="${emu(op.h)}"/></p:xfrm>` +
    `<a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/table">` +
    `<a:tbl><a:tblPr firstRow="0" bandRow="0"/><a:tblGrid>${gridXml}</a:tblGrid>${rows}</a:tbl></a:graphicData></a:graphic></p:graphicFrame>`;
}

/** XML операции (shape/line/text/table); id потом перенумеровывается. */
export function opXml(op, fonts) {
  switch (op.kind) {
    case "shape": return shapeXml(op, 0);
    case "line": return lineShapeXml(op, 0);
    case "text": return textShapeXml(op, 0, fonts);
    case "table": return tableXml(op, 0, fonts);
    default: throw new Error("неизвестная операция " + op.kind);
  }
}

/** Заменяет фигуры-заглушки (с меткой в тексте) готовым XML и перенумеровывает id фигур слайда. */
export function replacePlaceholders(xml, marks) {
  let out = xml;
  for (const [mark, frag] of marks) {
    const i = out.indexOf(mark);
    if (i < 0) throw new Error("метка не найдена: " + mark);
    const a = out.lastIndexOf("<p:sp>", i);
    const b = out.indexOf("</p:sp>", i);
    if (a < 0 || b < 0) throw new Error("не найдена фигура-заглушка: " + mark);
    out = out.slice(0, a) + frag + out.slice(b + "</p:sp>".length);
  }
  let id = 1;
  return out.replace(/<p:cNvPr id="\d+"/g, () => `<p:cNvPr id="${id++}"`);
}
