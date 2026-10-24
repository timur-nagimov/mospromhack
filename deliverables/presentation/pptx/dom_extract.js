/* Разбор слайда HTML-колоды в модель для PPTX (to_pptx.mjs). Выполняется в Chromium через addScriptTag.
 *
 * Координаты — CSS px относительно левого верхнего угла слайда. Операции идут в порядке отрисовки CSS
 * (контексты наложения, z-index, позиционированные слои), поэтому слои в PPTX совпадают с браузером.
 *   shape — прямоугольник или скруглённый прямоугольник со сплошной заливкой и/или рамкой;
 *   text  — текстовый блок: абзацы, строки, прогоны со шрифтом, кеглем, цветом, интервалами;
 *   table — нативная таблица (сетка, заливки, рамки, отступы, текст простых ячеек);
 *   image — <img> целиком из файла;
 *   raster — то, что нативно не нарисовать (градиенты, тени, маски, SVG): снимок элемента
 *            со скрытым текстом, текст ложится сверху отдельными операциями text.
 * Перед разбором стили элементов слайда «замораживаются» в inline-стили, а ::before/::after
 * превращаются в настоящие span: так у псевдоэлементов появляются размеры и текст для Range.
 */
(() => {
  "use strict";

  const SVG_NS = "http://www.w3.org/2000/svg";
  const REPLACED = new Set(["img", "svg", "canvas", "video", "iframe", "object", "embed", "input", "textarea", "select", "math"]);
  const SKIP_FREEZE = /^(transition|animation|will-change|content$|counter-|view-transition|anchor-|position-try|scroll-timeline|view-timeline|timeline-scope)/;
  const ISO_CSS = `
html.pp-iso, html.pp-iso body { background: transparent !important; }
html.pp-iso *:not([data-pp-show]) { visibility: hidden !important; }
html.pp-iso [data-pp-show] { visibility: visible !important; }
html.pp-iso [data-pp-show="tree"] *, html.pp-iso [data-pp-show="svg"] * { visibility: visible !important; }
html.pp-iso [data-pp-show]:not(svg), html.pp-iso [data-pp-show="tree"] *:not(svg):not(svg *) {
  color: transparent !important; -webkit-text-fill-color: transparent !important; text-shadow: none !important;
  text-decoration-color: transparent !important; text-emphasis-color: transparent !important; caret-color: transparent !important; }
html.pp-iso [data-pp-show] [data-pp-text], html.pp-iso [data-pp-show][data-pp-text] { visibility: hidden !important; }
[data-pp-nb]::before { content: none !important; display: none !important; }
[data-pp-na]::after { content: none !important; display: none !important; }
.pp-off { display: none !important; }`;

  const S = {};               // состояние текущего слайда
  let canvasCtx = null;
  const range = document.createRange();

  // ---------------------------------------------------------------- утилиты
  const num = v => { const n = parseFloat(v); return Number.isFinite(n) ? n : 0; };
  const round2 = v => Math.round(v * 100) / 100;
  const ctx2d = () => (canvasCtx = canvasCtx || document.createElement("canvas").getContext("2d"));

  function style(el) {
    let cs = S.styles.get(el);
    if (!cs) { cs = getComputedStyle(el); S.styles.set(el, cs); }
    return cs;
  }

  const COLOR_RE = /^rgba?\(\s*([\d.]+)[,\s]+([\d.]+)[,\s]+([\d.]+)(?:\s*[,/]\s*([\d.]+%?))?\s*\)$/;

  function parseColor(str) {
    if (!str || str === "transparent" || str === "none") return null;
    let m = str.match(COLOR_RE);
    if (!m) {
      const c = ctx2d();
      c.fillStyle = "#000";
      c.fillStyle = str;
      const v = c.fillStyle;
      if (v[0] === "#") return { hex: v.slice(1).toUpperCase(), a: 1 };
      m = v.match(COLOR_RE);
      if (!m) return null;
    }
    const a = m[4] == null ? 1 : (m[4].endsWith("%") ? parseFloat(m[4]) / 100 : parseFloat(m[4]));
    const hex = [m[1], m[2], m[3]].map(x => Math.round(+x).toString(16).padStart(2, "0")).join("").toUpperCase();
    return { hex, a };
  }

  /** Цвет с учётом общей непрозрачности; null, если не виден. */
  function paint(str, opacity) {
    const c = parseColor(str);
    if (!c) return null;
    const a = c.a * opacity;
    return a < 0.004 ? null : { color: c.hex, alpha: round2(a) };
  }

  function opacityOf(el) {
    if (S.opacity.has(el)) return S.opacity.get(el);
    let o = num(style(el).opacity);
    const p = el.parentElement;
    if (p && el !== document.documentElement) o *= opacityOf(p);
    S.opacity.set(el, o);
    return o;
  }

  function rel(r) { return { x: r.left - S.ox, y: r.top - S.oy, w: r.width, h: r.height }; }

  function label(el) {
    if (!el || !el.localName) return "?";
    const raw = typeof el.className === "string" ? el.className : (el.className && el.className.baseVal) || "";
    const cls = raw.trim().split(/\s+/)[0];
    return el.localName + (el.id ? "#" + el.id : "") + (cls ? "." + cls : "");
  }

  function warn(msg) { if (S.warns.length < 200) S.warns.push(msg); }

  // ---------------------------------------------------------------- шрифтовые метрики
  function fontMetrics(cs, size) {
    const italic = cs.fontStyle !== "normal" ? "italic " : "";
    const font = `${italic}${cs.fontWeight} ${size}px ${cs.fontFamily}`;
    let m = S.metrics.get(font);
    if (!m) {
      const c = ctx2d();
      c.font = font;
      const t = c.measureText("Hgй");
      m = { asc: t.fontBoundingBoxAscent, desc: t.fontBoundingBoxDescent, space: c.measureText(" ").width, font };
      S.metrics.set(font, m);
    }
    return m;
  }

  function families(cs) {
    return cs.fontFamily.split(",").map(s => s.trim().replace(/^["']|["']$/g, "")).filter(Boolean);
  }

  // ---------------------------------------------------------------- подготовка слайда
  function cssText(cs) {
    let t = "";
    for (let i = 0; i < cs.length; i++) {
      const p = cs[i];
      if (SKIP_FREEZE.test(p)) continue;
      t += p + ":" + cs.getPropertyValue(p) + ";";
    }
    return t;
  }

  function freeze(root) {
    const els = [root, ...root.querySelectorAll("*")].filter(el => !(el.namespaceURI === SVG_NS && el.localName !== "svg"));
    const decls = els.map(el => cssText(getComputedStyle(el)));
    els.forEach((el, i) => el.setAttribute("style", decls[i]));
  }

  function resolveContent(content, el) {
    if (!content || content === "none" || content === "normal") return null;
    let out = "", ok = true;
    const re = /"((?:[^"\\]|\\.)*)"|'((?:[^'\\]|\\.)*)'|attr\(\s*([\w-]+)\s*\)|(counters?\(|url\(|image-set\(|(?:no-)?(?:open|close)-quote)/g;
    for (const m of content.matchAll(re)) {
      if (m[1] != null || m[2] != null) {
        out += (m[1] != null ? m[1] : m[2])
          .replace(/\\([0-9a-fA-F]{1,6})\s?/g, (_, h) => String.fromCodePoint(parseInt(h, 16)))
          .replace(/\\(.)/g, "$1");
      } else if (m[3]) out += el.getAttribute(m[3]) || "";
      else ok = false;
    }
    if (!ok) warn(label(el) + ": content «" + content.slice(0, 40) + "» (счётчик/картинка) — пропущено");
    return out;
  }

  function materializePseudos(root) {
    const todo = [];
    for (const el of [root, ...root.querySelectorAll("*")]) {
      if (el.namespaceURI === SVG_NS || REPLACED.has(el.localName)) continue;
      for (const which of ["before", "after"]) {
        const cs = getComputedStyle(el, "::" + which);
        if (!cs.content || cs.content === "none" || cs.content === "normal" || cs.display === "none") continue;
        todo.push({ el, which, style: cssText(cs), text: resolveContent(cs.content, el) });
      }
    }
    for (const { el, which, style: st, text } of todo) {
      const span = document.createElement("span");
      span.setAttribute("style", st);
      span.setAttribute("data-pp-pseudo", which);
      if (text) span.textContent = text;
      if (which === "before") { el.insertBefore(span, el.firstChild); el.setAttribute("data-pp-nb", ""); }
      else { el.appendChild(span); el.setAttribute("data-pp-na", ""); }
    }
    return todo.length;
  }

  function snapshotRects(root) {
    return [root, ...root.querySelectorAll("*")].map(el => [el, el.getBoundingClientRect()]);
  }

  // ---------------------------------------------------------------- классификация элементов
  const isReplaced = el => REPLACED.has(el.localName);
  const isPositioned = cs => cs.position !== "static";
  const zOf = cs => (cs.zIndex === "auto" ? 0 : parseInt(cs.zIndex, 10) || 0);
  const notNone = v => !!v && v !== "none";

  function createsSC(el, cs) {
    if (el === S.slide) return true;
    if (isPositioned(cs) && cs.zIndex !== "auto") return true;
    if (cs.position === "fixed" || cs.position === "sticky") return true;
    const p = el.parentElement;
    if (p && cs.zIndex !== "auto" && /flex|grid/.test(style(p).display)) return true;
    if (num(cs.opacity) < 1) return true;
    if (notNone(cs.transform) || notNone(cs.filter) || cs.mixBlendMode !== "normal" || cs.isolation === "isolate") return true;
    if (notNone(cs.maskImage) || notNone(cs.webkitMaskImage) || notNone(cs.clipPath)) return true;
    if (notNone(cs.backdropFilter) || /paint|layout|strict|content/.test(cs.contain)) return true;
    return false;
  }

  /** Элемент рисуется своим слоем (позиционирован или образует контекст наложения). */
  const isLayered = (el, cs) => isPositioned(cs) || createsSC(el, cs);
  const isFloat = cs => cs.float !== "none";
  const isAtomicInline = (el, cs) => cs.display.startsWith("inline-") || (cs.display === "inline" && isReplaced(el));
  const isInlineBox = cs => cs.display === "inline";

  function* kids(el) {
    for (const c of el.children) {
      if (style(c).display === "contents") yield* kids(c);
      else yield c;
    }
  }

  // ---------------------------------------------------------------- клип предками
  function cbParent(el) {
    const cs = style(el);
    if (cs.position === "fixed") return null;
    if (cs.position === "absolute") {
      let p = el.parentElement;
      while (p && p !== S.slide && style(p).position === "static" && !notNone(style(p).transform)) p = p.parentElement;
      return p;
    }
    return el.parentElement;
  }

  function radiiOf(cs, w, h) {
    const one = v => {
      const parts = (v || "0px").split(" ");
      const val = (s, base) => (s.endsWith("%") ? num(s) / 100 * base : num(s));
      return [val(parts[0], w), val(parts[1] || parts[0], h)];
    };
    const r = [one(cs.borderTopLeftRadius), one(cs.borderTopRightRadius), one(cs.borderBottomRightRadius), one(cs.borderBottomLeftRadius)];
    // как в CSS: если сумма радиусов больше стороны — все уменьшаются пропорционально
    const f = Math.min(1,
      w / Math.max(1e-6, r[0][0] + r[1][0]), w / Math.max(1e-6, r[3][0] + r[2][0]),
      h / Math.max(1e-6, r[0][1] + r[3][1]), h / Math.max(1e-6, r[1][1] + r[2][1]));
    return r.map(([x, y]) => [x * f, y * f]);
  }

  /** Прямоугольный клип (координаты страницы) и скруглённые клипы предков. */
  function clipOf(el) {
    if (S.clips.has(el)) return S.clips.get(el);
    let l = -Infinity, t = -Infinity, r = Infinity, b = Infinity;
    const rounded = [];
    for (let n = cbParent(el); n; n = cbParent(n)) {
      const cs = style(n);
      if (cs.overflowX !== "visible" || cs.overflowY !== "visible" || /paint|strict|content/.test(cs.contain)) {
        const R = n.getBoundingClientRect();
        const bw = [num(cs.borderLeftWidth), num(cs.borderTopWidth), num(cs.borderRightWidth), num(cs.borderBottomWidth)];
        const pl = R.left + bw[0], pt = R.top + bw[1], pr = R.right - bw[2], pb = R.bottom - bw[3];
        if (cs.overflowX !== "visible") { l = Math.max(l, pl); r = Math.min(r, pr); }
        if (cs.overflowY !== "visible") { t = Math.max(t, pt); b = Math.min(b, pb); }
        const outer = radiiOf(cs, R.width, R.height);
        if (outer.some(([x, y]) => x > 0.5 || y > 0.5)) {
          const inner = [
            [outer[0][0] - bw[0], outer[0][1] - bw[1]], [outer[1][0] - bw[2], outer[1][1] - bw[1]],
            [outer[2][0] - bw[2], outer[2][1] - bw[3]], [outer[3][0] - bw[0], outer[3][1] - bw[3]],
          ].map(([x, y]) => [Math.max(0, x), Math.max(0, y)]);
          rounded.push({ l: pl, t: pt, r: pr, b: pb, radii: inner });
        }
      }
      if (n === S.slide) break;
    }
    const res = { l, t, r, b, rounded };
    S.clips.set(el, res);
    return res;
  }

  /** Как скруглённые клипы предков задевают прямоугольник r (координаты страницы):
   *  null — не задевают; {radii} — углы r совпадают со скруглёнными углами клипа; false — сложный случай. */
  function roundedCut(r, clip) {
    const radii = [0, 0, 0, 0];
    let touched = false;
    for (const c of clip.rounded) {
      const corners = [
        [c.l, c.t, c.radii[0], r.left, r.top, 1, 1],
        [c.r, c.t, c.radii[1], r.right, r.top, -1, 1],
        [c.r, c.b, c.radii[2], r.right, r.bottom, -1, -1],
        [c.l, c.b, c.radii[3], r.left, r.bottom, 1, -1],
      ];
      for (let i = 0; i < 4; i++) {
        const [cx, cy, [rx, ry], px, py, sx, sy] = corners[i];
        if (rx < 0.5 || ry < 0.5) continue;
        const qx = sx > 0 ? Math.max(px, cx) : Math.min(px, cx);
        const qy = sy > 0 ? Math.max(py, cy) : Math.min(py, cy);
        const ox = cx + sx * rx, oy = cy + sy * ry;               // центр дуги
        const inZone = (sx > 0 ? qx < ox : qx > ox) && (sy > 0 ? qy < oy : qy > oy);
        if (!inZone) continue;
        const dx = (qx - ox) / rx, dy = (qy - oy) / ry;
        if (dx * dx + dy * dy <= 1.0001) continue;                // угол внутри дуги — не задет
        touched = true;
        if (Math.abs(qx - cx) < 1 && Math.abs(qy - cy) < 1 && Math.abs(rx - ry) < 0.5) radii[i] = Math.max(radii[i], rx);
        else return false;
      }
    }
    return touched ? { radii } : null;
  }

  // ---------------------------------------------------------------- снимки (растр)
  function addCapture(els, mode, rect, format, scale) {
    const id = ++S.capSeq;
    for (const e of els) e.setAttribute("data-pp-cap", String(id));
    const x = Math.floor(rect.l), y = Math.floor(rect.t);
    const w = Math.ceil(rect.r) - x, h = Math.ceil(rect.b) - y;
    const cap = { id, mode, format, scale, clip: { x, y, w, h } };
    S.caps.push(cap);
    return cap;
  }

  function inkRect(el, cs) {
    const R = el.getBoundingClientRect();
    let l = R.left, t = R.top, r = R.right, b = R.bottom;
    if (notNone(cs.boxShadow)) {
      for (const sh of cs.boxShadow.split(/,(?![^(]*\))/)) {
        if (/inset/.test(sh)) continue;
        const lens = (sh.replace(/[a-z-]+\([^)]*\)|#[0-9a-f]+/gi, "").match(/-?[\d.]+px/g) || []).map(num);
        const [dx = 0, dy = 0, blur = 0, spread = 0] = lens;
        const e = blur + spread;
        l = Math.min(l, R.left + dx - e); r = Math.max(r, R.right + dx + e);
        t = Math.min(t, R.top + dy - e); b = Math.max(b, R.bottom + dy + e);
      }
    }
    if (cs.outlineStyle !== "none" && num(cs.outlineWidth) > 0) {
      const o = num(cs.outlineWidth) + Math.max(0, num(cs.outlineOffset));
      l -= o; t -= o; r += o; b += o;
    }
    if (notNone(cs.filter)) { l -= 20; t -= 20; r += 20; b += 20; }
    return { l, t, r, b };
  }

  function clampToClip(rect, clip) {
    const s = S.slideRect;
    const l = Math.max(rect.l, clip.l, s.left), t = Math.max(rect.t, clip.t, s.top);
    const r = Math.min(rect.r, clip.r, s.right), b = Math.min(rect.b, clip.b, s.bottom);
    return r - l < 0.5 || b - t < 0.5 ? null : { l, t, r, b };
  }

  function pushRaster(el, mode, rect, format, name) {
    const clipped = clampToClip(rect, clipOf(el));
    if (!clipped) return null;
    const cap = addCapture([el], mode, clipped, format, "device");
    const op = { kind: "raster", cap: cap.id, x: cap.clip.x - S.ox, y: cap.clip.y - S.oy, w: cap.clip.w, h: cap.clip.h, name, el };
    S.ops.push(op);
    return op;
  }

  // ---------------------------------------------------------------- фон и рамка элемента
  function borderSides(cs) {
    return ["Top", "Right", "Bottom", "Left"].map(side => ({
      st: cs["border" + side + "Style"], w: num(cs["border" + side + "Width"]), c: cs["border" + side + "Color"],
    }));
  }

  const dashOf = st => (st === "dashed" ? "dash" : st === "dotted" ? "sysDot" : "solid");

  function emitBox(el) {
    if (el === S.slide) { emitSlideBackground(); return; }
    if (S.tableParts.has(el)) return;
    const cs = style(el);
    if (cs.visibility !== "visible") return;
    const op = opacityOf(el);
    if (op < 0.004) return;
    if (isInlineBox(cs) && !isReplaced(el)) return;     // фон строчных элементов рисуется вместе с текстом
    const R = el.getBoundingClientRect();
    if (R.width < 0.5 || R.height < 0.5) return;

    const fill = paint(cs.backgroundColor, op);
    const sides = borderSides(cs).map(s => ({ ...s, p: (s.w > 0 && s.st !== "none" && s.st !== "hidden") ? paint(s.c, op) : null }));
    const hasBorder = sides.some(s => s.p);
    const bgImage = notNone(cs.backgroundImage);
    const shadow = notNone(cs.boxShadow);
    const outline = cs.outlineStyle !== "none" && num(cs.outlineWidth) > 0 && !!paint(cs.outlineColor, op);
    const effects = notNone(cs.filter) || notNone(cs.backdropFilter) || notNone(cs.clipPath) ||
      notNone(cs.maskImage) || notNone(cs.webkitMaskImage) || cs.mixBlendMode !== "normal";
    if (!fill && !hasBorder && !bgImage && !shadow && !outline && !effects) return;

    const radii = radiiOf(cs, R.width, R.height);
    const elliptic = radii.some(([x, y]) => Math.abs(x - y) > 0.5);
    const uniformR = radii.every(([x]) => Math.abs(x - radii[0][0]) < 0.5);
    const rounded = radii.some(([x]) => x > 0.5);
    const bordersPlain = sides.every(s => !s.p || s.st === "solid" || s.st === "dashed" || s.st === "dotted");
    const vis = sides.filter(s => s.p);
    const uniformBorder = vis.length === 4 && vis.every(s => Math.abs(s.w - vis[0].w) < 0.1 && s.st === vis[0].st && s.p.color === vis[0].p.color && s.p.alpha === vis[0].p.alpha);
    const clip = clipOf(el);
    const cut = roundedCut(R, clip);
    const fullyInside = R.left >= clip.l - 0.5 && R.top >= clip.t - 0.5 && R.right <= clip.r + 0.5 && R.bottom <= clip.b + 0.5;

    let raster = bgImage || shadow || outline || effects || elliptic || !bordersPlain || cut === false || !fullyInside;
    if (!raster && hasBorder && !uniformBorder && rounded) raster = true;       // разные стороны рамки + скругление
    if (!raster && !uniformR && (hasBorder || !fill)) raster = true;             // разные радиусы углов с рамкой
    if (!raster && cut && (rounded || (hasBorder && uniformBorder))) raster = true;
    if (raster) { pushRaster(el, "box", inkRect(el, cs), "png", "Фон " + label(el)); return; }

    const r = rel(R);
    const name = label(el);
    if (cut || !uniformR) {
      // углы обрезаны скруглением родителя (полоса заголовка окна) или разные радиусы углов
      const cr = cut ? cut.radii : radii.map(([x]) => x);
      if (fill) S.ops.push({ kind: "shape", geom: "corners", radii: cr, ...r, fill, line: null, name });
      pushSides(sides, r, name);
      return;
    }
    const rad = Math.min(radii[0][0], r.w / 2, r.h / 2);
    if (uniformBorder) {
      const bw = vis[0].w;
      S.ops.push({
        kind: "shape", geom: rad > 0.5 ? "roundRect" : "rect",
        x: r.x + bw / 2, y: r.y + bw / 2, w: Math.max(0.5, r.w - bw), h: Math.max(0.5, r.h - bw), radius: Math.max(0, rad - bw / 2),
        fill, line: { ...vis[0].p, width: bw, dash: dashOf(vis[0].st) }, name,
      });
      return;
    }
    if (fill) S.ops.push({ kind: "shape", geom: rad > 0.5 ? "roundRect" : "rect", ...r, radius: rad, fill, line: null, name });
    pushSides(sides, r, name);
  }

  /** Отдельные стороны рамки: сплошные — тонкими прямоугольниками, пунктир — линиями. */
  function pushSides(sides, r, name) {
    const [T, Rt, B, L] = sides;
    const geo = [
      [T, { x: r.x, y: r.y, w: r.w, h: T.w }],
      [B, { x: r.x, y: r.y + r.h - B.w, w: r.w, h: B.w }],
      [L, { x: r.x, y: r.y, w: L.w, h: r.h }],
      [Rt, { x: r.x + r.w - Rt.w, y: r.y, w: Rt.w, h: r.h }],
    ];
    for (const [s, g] of geo) {
      if (!s.p) continue;
      if (s.st === "solid") { S.ops.push({ kind: "shape", geom: "rect", ...g, fill: s.p, line: null, name: name + " (рамка)" }); continue; }
      const horiz = g.w >= g.h;
      S.ops.push({
        kind: "line", name: name + " (рамка)", width: s.w, dash: dashOf(s.st), color: s.p.color, alpha: s.p.alpha,
        x1: horiz ? g.x : g.x + g.w / 2, y1: horiz ? g.y + g.h / 2 : g.y,
        x2: horiz ? g.x + g.w : g.x + g.w / 2, y2: horiz ? g.y + g.h / 2 : g.y + g.h,
      });
    }
  }

  function emitSlideBackground() {
    const cs = style(S.slide);
    const R = S.slideRect;
    const cap = addCapture([S.slide], "bg", { l: R.left, t: R.top, r: R.right, b: R.bottom }, "jpeg", "css");
    S.bg = { cap: cap.id, color: (parseColor(cs.backgroundColor) || { hex: "000000" }).hex, els: [S.slide] };
    S.ops.push({ kind: "bg", cap: cap.id });
  }

  // ---------------------------------------------------------------- замещаемые элементы
  function emitReplaced(el) {
    const cs = style(el);
    if (cs.visibility !== "visible" && el.localName !== "svg") return;
    const R = el.getBoundingClientRect();
    if (R.width < 0.5 || R.height < 0.5) return;
    const name = label(el);
    if (el.localName === "img") {
      if (!el.naturalWidth) { warn(name + ": картинка не загрузилась"); return; }
      const clip = clipOf(el);
      const inside = R.left >= clip.l - 0.5 && R.top >= clip.t - 0.5 && R.right <= clip.r + 0.5 && R.bottom <= clip.b + 0.5;
      const radii = radiiOf(cs, R.width, R.height);
      const cut = roundedCut(R, clip);
      const aspectSame = Math.abs(el.naturalWidth / el.naturalHeight - R.width / R.height) < 0.01;
      const fileOk = /^https?:/.test(el.currentSrc) && !/\.svg(\?|$)/i.test(el.currentSrc);
      const plain = inside && !cut && radii.every(([x]) => x < 0.5) && opacityOf(el) > 0.999 && !notNone(cs.filter) &&
        (cs.objectFit === "fill" || aspectSame) && fileOk && num(cs.paddingLeft) === 0 && num(cs.borderLeftWidth) === 0;
      if (plain) { S.ops.push({ kind: "image", src: el.currentSrc, ...rel(R), name: "Картинка " + name }); return; }
      const opaque = !cut && radii.every(([x]) => x < 0.5) && opacityOf(el) > 0.999 && !/\.(png|gif|webp)(\?|$)/i.test(el.currentSrc);
      pushRaster(el, "box", inkRect(el, cs), opaque ? "jpeg" : "png", "Картинка " + name);
      return;
    }
    if (el.localName === "svg") {
      const texts = svgTexts(el);
      const ink = { l: R.left, t: R.top, r: R.right, b: R.bottom };
      if (cs.overflowX === "visible" || cs.overflowY === "visible") {
        for (const d of el.querySelectorAll("*")) {
          const q = d.getBoundingClientRect();
          if (q.width || q.height) { ink.l = Math.min(ink.l, q.left); ink.t = Math.min(ink.t, q.top); ink.r = Math.max(ink.r, q.right); ink.b = Math.max(ink.b, q.bottom); }
        }
      }
      if (hasSvgPaint(el)) pushRaster(el, "svg", ink, "png", "Графика " + name);
      for (const t of texts) S.ops.push(t);
      return;
    }
    pushRaster(el, "tree", inkRect(el, cs), "png", "Растр " + name);
  }

  function hasSvgPaint(svg) {
    for (const d of svg.querySelectorAll("*")) {
      if (d.closest("defs,clipPath,mask,marker,pattern,symbol,linearGradient,radialGradient,filter")) continue;
      if (d.closest("[data-pp-text]")) continue;
      if (/^(rect|circle|ellipse|line|polyline|polygon|path|image|use|foreignObject|text)$/.test(d.localName)) return true;
    }
    const cs = style(svg);
    return !!(paint(cs.backgroundColor, 1) || num(cs.borderTopWidth));
  }

  // ---------------------------------------------------------------- текст: символы → строки → блоки
  function runStyle(el, scale) {
    const key = el.namespaceURI === SVG_NS ? null : el;
    if (key && S.runStyles.has(key)) return S.runStyles.get(key);
    const cs = style(el);
    const op = opacityOf(el);
    const isSvg = el.namespaceURI === SVG_NS;
    let col;
    if (isSvg) col = paint(cs.fill, op * (cs.fillOpacity === "" ? 1 : num(cs.fillOpacity)));
    else col = paint(cs.webkitTextFillColor && cs.webkitTextFillColor !== cs.color ? cs.webkitTextFillColor : cs.color, op);
    const size = num(cs.fontSize) * scale;
    const fm = fontMetrics(cs, size);
    let deco = "";
    for (let n = el; n && n !== S.slide; n = n.parentElement) {
      const d = style(n).textDecorationLine || "";
      if (d.includes("underline")) deco += " u";
      if (d.includes("line-through")) deco += " s";
      if (n !== el && style(n).display !== "inline") break;
    }
    const ls = cs.letterSpacing === "normal" ? 0 : num(cs.letterSpacing) * scale;
    const ws = cs.whiteSpaceCollapse || cs.whiteSpace;
    const st = {
      families: families(cs), weight: parseInt(cs.fontWeight, 10) || 400, italic: cs.fontStyle !== "normal",
      size, color: col ? col.color : null, alpha: col ? col.alpha : 0, ls,
      underline: deco.includes("u"), strike: deco.includes("s"),
      transform: cs.textTransform, preserve: /^(pre|pre-wrap|break-spaces|preserve)/.test(ws),
      asc: fm.asc, desc: fm.desc, space: fm.space, font: fm.font,
      visible: cs.visibility === "visible" && !!col && size > 0.5,
    };
    st.key = [st.families.join("|"), st.weight, st.italic, round2(st.size), st.color, st.alpha, round2(st.ls), st.underline, st.strike].join("/");
    if (key) S.runStyles.set(key, st);
    return st;
  }

  function transformChar(ch, st, prevCh) {
    switch (st.transform) {
      case "uppercase": return ch.toLocaleUpperCase("ru-RU");
      case "lowercase": return ch.toLocaleLowerCase("ru-RU");
      case "capitalize": return (!prevCh || /[\s(«"„-]/.test(prevCh)) ? ch.toLocaleUpperCase("ru-RU") : ch;
      default: return ch;
    }
  }

  const WS = new Set([" ", "\n", "\t", "\r", "\f"]);
  const INVISIBLE = /[​­‌‍﻿]/;

  /** Символы текстовых узлов с прямоугольниками (координаты страницы). */
  function measureChars(nodes, scaleOf) {
    const chars = [];
    for (const node of nodes) {
      const owner = node.parentElement;
      const scale = scaleOf ? scaleOf(owner) : 1;
      const st = runStyle(owner, scale);
      if (!st.visible) continue;
      const clip = clipOf(owner.namespaceURI === SVG_NS ? owner.ownerSVGElement || owner : owner);
      const data = node.data;
      let prev = "";
      for (let i = 0; i < data.length;) {
        const cp = data.codePointAt(i);
        const len = cp > 0xffff ? 2 : 1;
        const raw = data.slice(i, i + len);
        range.setStart(node, i); range.setEnd(node, i + len);
        i += len;
        let r = null;
        for (const q of range.getClientRects()) { if (q.width > 0.01 || q.height > 0.01) { r = q; break; } }
        const prevRaw = prev;
        prev = raw;
        if (!r) continue;
        const ws = WS.has(raw);
        let ch = raw;
        if (ws) {
          if ((raw === "\n" || raw === "\r") && st.preserve) continue;    // перевод строки в pre — по геометрии
          if (r.width < 0.3) continue;                                     // схлопнутый пробел
          ch = raw === "\t" && st.preserve ? " ".repeat(Math.max(1, Math.round(r.width / Math.max(1, st.space + st.ls)))) : " ";
        } else if (r.width < 0.01 && INVISIBLE.test(raw)) continue;
        const cx = r.left + r.width / 2, cy = r.top + r.height / 2;
        if (cx < clip.l || cx > clip.r || cy < clip.t || cy > clip.b) continue;
        chars.push({ ch: ws ? ch : transformChar(ch, st, prevRaw), space: ws, l: r.left, r: r.right, t: r.top, b: r.bottom, st, owner });
      }
    }
    return chars;
  }

  function splitLines(chars) {
    const lines = [];
    let cur = null;
    for (const c of chars) {
      if (cur && cur.last) {
        const back = !c.space && c.l < cur.last.r - 2;
        const below = c.t >= cur.maxB - 1;
        if (back || below) cur = null;
      }
      if (!cur) { cur = { chars: [], maxB: -Infinity, last: null }; lines.push(cur); }
      cur.chars.push(c);
      if (!c.space) { cur.maxB = Math.max(cur.maxB, c.b); cur.last = c; }
    }
    const out = [];
    for (const ln of lines) {
      const cs = ln.chars;
      let a = 0, z = cs.length - 1;
      while (a <= z && cs[a].space) a++;
      while (z >= a && cs[z].space) z--;
      if (a > z) continue;
      const chars = cs.slice(a, z + 1).filter((c, i, arr) => !(c.space && i > 0 && arr[i - 1].space));
      const bases = chars.filter(c => !c.space).map(c => c.t + c.st.asc).sort((p, q) => p - q);
      const baseline = bases[Math.floor(bases.length / 2)];
      let D = 0, A = 0;
      for (const c of chars) {
        const shift = c.space ? 0 : baseline - (c.t + c.st.asc);
        c.shift = Math.abs(shift) > 1 ? shift : 0;
        if (!c.shift) { D = Math.max(D, c.st.desc); A = Math.max(A, c.st.asc); }
      }
      out.push({ chars, baseline, D, A, x0: chars[0].l, x1: chars[chars.length - 1].r });
    }
    return out;
  }

  /** Строка → прогоны: стиль меняется или после символа есть зазор (отступ строчного элемента, inline-block). */
  function lineRuns(line) {
    const runs = [];
    const cs = line.chars;
    for (let i = 0; i < cs.length; i++) {
      const c = cs[i];
      let extra = 0;
      if (i + 1 < cs.length) {
        const gap = cs[i + 1].l - c.r;
        if (Math.abs(gap) > 0.75) extra += gap;
      }
      if (c.space) {
        const over = (c.r - c.l) - c.st.space - c.st.ls;
        if (Math.abs(over) > 0.75) extra += over;
      }
      const spacing = round2(c.st.ls + extra);
      const baseline = c.shift ? Math.round(c.shift / c.st.size * 100000) : 0;
      const last = runs[runs.length - 1];
      if (last && last.st.key === c.st.key && last.spacing === spacing && last.baseline === baseline && !extra && !last.hasExtra) last.text += c.ch;
      else runs.push({ text: c.ch, st: c.st, spacing, baseline, hasExtra: !!extra });
    }
    return runs.map(r => ({
      text: r.text, families: r.st.families, weight: r.st.weight, italic: r.st.italic, size: round2(r.st.size),
      color: r.st.color, alpha: r.st.alpha, spacing: r.spacing, underline: r.st.underline, strike: r.st.strike, baseline: r.baseline,
    }));
  }

  function lineHeightOf(cs, st) {
    if (cs.lineHeight === "normal") return st.asc + st.desc;
    return num(cs.lineHeight);
  }

  function lineFits(ln, boxL, boxR, align) {
    if (align === "r") return Math.abs(ln.x1 - boxR) <= 1.5;
    if (align === "ctr") return Math.abs((ln.x0 + ln.x1) / 2 - (boxL + boxR) / 2) <= 1.5;
    return Math.abs(ln.x0 - boxL) <= 1.5;
  }

  /** Строки одного блока → текстовые операции. boxL/boxR — границы блока (координаты страницы). */
  function emitLines(lines, opts) {
    const { boxL, boxR, align, lhCss, name } = opts;
    const chunks = [];
    let cur = null;
    for (const ln of lines) {
      if (!lineFits(ln, boxL, boxR, align)) { chunks.push({ lines: [ln], loose: true, pitch: null }); cur = null; continue; }
      if (cur) {
        const prev = cur.lines[cur.lines.length - 1];
        const pitch = ln.baseline - prev.baseline;
        const okPitch = pitch > 0 && (cur.pitch == null || Math.abs(pitch - cur.pitch) <= 1);
        if (okPitch && Math.abs(ln.D - cur.lines[0].D) <= 0.6) { cur.lines.push(ln); if (cur.pitch == null) cur.pitch = pitch; continue; }
      }
      cur = { lines: [ln], pitch: null, loose: false };
      chunks.push(cur);
    }
    for (const ch of chunks) {
      const first = ch.lines[0];
      const L = ch.pitch != null ? ch.pitch : Math.max(lhCss || 0, 1);
      let l, r;
      if (ch.loose) { l = first.x0; r = first.x1; }
      else { l = Math.min(boxL, ...ch.lines.map(x => x.x0)); r = Math.max(boxR, ...ch.lines.map(x => x.x1)); }
      const maxW = Math.max(...ch.lines.map(x => x.x1 - x.x0));
      const slack = Math.max(6, 0.06 * maxW);
      if (align === "r") l -= slack;
      else if (align === "ctr") { l -= slack / 2; r += slack / 2; }
      else r += slack;
      const top = first.baseline - L + first.D;
      S.ops.push({
        kind: "text", name, x: l - S.ox, y: top - S.oy, w: r - l, h: L * ch.lines.length,
        paragraphs: [{ align, lineSpacing: round2(L), spaceBefore: 0, lines: ch.lines.map(lineRuns) }],
      });
      S.textChars += ch.lines.reduce((n, x) => n + x.chars.length, 0);
    }
  }

  function alignOf(cs) {
    const a = cs.textAlign;
    if (a === "right" || a === "end" || a === "-webkit-right") return "r";
    if (a === "center" || a === "-webkit-center") return "ctr";
    return "l";
  }

  function contentBox(el) {
    const R = el.getBoundingClientRect(), cs = style(el);
    return {
      l: R.left + num(cs.borderLeftWidth) + num(cs.paddingLeft),
      r: R.right - num(cs.borderRightWidth) - num(cs.paddingRight),
      t: R.top + num(cs.borderTopWidth) + num(cs.paddingTop),
      b: R.bottom - num(cs.borderBottomWidth) - num(cs.paddingBottom),
    };
  }

  /** Текст группы (подряд идущее строчное содержимое блока) → строки; учитывает text-overflow: ellipsis. */
  function groupLines(group) {
    let chars = measureChars(group.nodes);
    const B = group.block;
    const cs = style(B);
    if (cs.textOverflow === "ellipsis" && cs.overflowX !== "visible" && B.scrollWidth > B.clientWidth + 1 && chars.length) {
      const cb = contentBox(B);
      const lastSt = chars[chars.length - 1].st;
      const c = ctx2d();
      c.font = lastSt.font;
      const ew = c.measureText("…").width;
      const kept = chars.filter(ch => ch.r <= cb.r - ew + 0.5);
      if (kept.length < chars.length) {
        while (kept.length && kept[kept.length - 1].space) kept.pop();
        const last = kept[kept.length - 1];
        if (last) kept.push({ ...last, ch: "…", space: false, l: last.r, r: last.r + ew });
        chars = kept;
      }
    }
    return splitLines(chars);
  }

  function emitInlineBackgrounds(group) {
    for (const el of group.inlines) {
      const ics = style(el);
      if (ics.visibility !== "visible") continue;
      const op = opacityOf(el);
      const fill = paint(ics.backgroundColor, op);
      const sides = borderSides(ics).filter(s => s.w > 0 && s.st !== "none" && s.st !== "hidden" && paint(s.c, op));
      const bgImg = notNone(ics.backgroundImage);
      if (!fill && !sides.length && !bgImg) continue;
      if (bgImg || sides.length || radiiOf(ics, 10, 10).some(([x]) => x > 0.5)) { pushRaster(el, "box", inkRect(el, ics), "png", "Фон " + label(el)); continue; }
      for (const q of el.getClientRects()) S.ops.push({ kind: "shape", geom: "rect", ...rel(q), fill, line: null, name: "Фон " + label(el) });
    }
  }

  function emitGroup(group) {
    emitInlineBackgrounds(group);
    const B = group.block;
    const cs = style(B);
    const lines = groupLines(group);
    if (!lines.length) return;
    let boxL, boxR;
    if (/flex|grid/.test(cs.display)) {        // текст прямо во flex/grid — анонимный элемент своей ширины
      boxL = Math.min(...lines.map(l => l.x0)); boxR = Math.max(...lines.map(l => l.x1));
    } else { const cb = contentBox(B); boxL = cb.l; boxR = cb.r; }
    const st0 = lines[0].chars[0].st;
    const text = lines.map(l => l.chars.map(c => c.ch).join("")).join(" ");
    emitLines(lines, { boxL, boxR, align: alignOf(cs), lhCss: lineHeightOf(cs, st0), name: "Текст «" + text.slice(0, 40) + "»" });
  }

  // ---------------------------------------------------------------- SVG-текст
  function svgTexts(svg) {
    const saved = S.ops;
    const out = [];
    for (const t of svg.querySelectorAll("text")) {
      if (t.closest("defs,clipPath,mask,marker,pattern,symbol")) continue;
      const cs = style(t);
      if (cs.display === "none" || cs.visibility !== "visible") continue;
      const m = t.getScreenCTM();
      if (!m || Math.abs(m.b) > 1e-3 || Math.abs(m.c) > 1e-3 || m.a <= 0 || Math.abs(m.a - m.d) > 1e-3) { warn(label(t) + ": повёрнутый SVG-текст оставлен в картинке"); continue; }
      const nodes = [];
      const walker = document.createTreeWalker(t, NodeFilter.SHOW_TEXT);
      for (let n = walker.nextNode(); n; n = walker.nextNode()) nodes.push(n);
      const chars = measureChars(nodes, () => m.a);
      if (!chars.length) continue;
      t.setAttribute("data-pp-text", "1");
      const anchor = cs.textAnchor === "middle" ? "ctr" : cs.textAnchor === "end" ? "r" : "l";
      S.ops = [];
      for (const ln of splitLines(chars)) {
        emitLines([ln], { boxL: ln.x0, boxR: ln.x1, align: anchor, lhCss: ln.A + ln.D, name: "Текст «" + ln.chars.map(c => c.ch).join("").slice(0, 40) + "»" });
      }
      out.push(...S.ops);
      S.ops = saved;
    }
    S.ops = saved;
    return out;
  }

  // ---------------------------------------------------------------- таблицы
  function tryTable(table) {
    const cs = style(table);
    const spacing = cs.borderSpacing.split(" ").map(num);
    if (cs.borderCollapse !== "collapse" && spacing.some(v => v > 0.5)) return false;
    if (createsSC(table, cs) || isPositioned(cs)) return false;
    const rows = [...table.rows].filter(tr => style(tr).display !== "none");
    const cells = [];
    for (const tr of rows) for (const td of tr.cells) if (style(td).display !== "none") cells.push(td);
    if (!cells.length) return false;
    const xs = [], ys = [];
    const pushEdge = (arr, v) => { if (!arr.some(e => Math.abs(e - v) < 1)) arr.push(v); };
    for (const td of cells) { const R = td.getBoundingClientRect(); pushEdge(xs, R.left); pushEdge(xs, R.right); }
    for (const tr of rows) { const R = tr.getBoundingClientRect(); pushEdge(ys, R.top); pushEdge(ys, R.bottom); }
    xs.sort((a, b) => a - b); ys.sort((a, b) => a - b);
    const idx = (arr, v) => arr.findIndex(e => Math.abs(e - v) < 1);
    const model = [];
    for (const td of cells) {
      const R = td.getBoundingClientRect();
      const c0 = idx(xs, R.left), c1 = idx(xs, R.right), r0 = idx(ys, R.top), r1 = idx(ys, R.bottom);
      if (c0 < 0 || c1 <= c0 || r0 < 0 || r1 <= r0) return false;
      const tcs = style(td);
      if (notNone(tcs.backgroundImage) || notNone(tcs.boxShadow) || num(tcs.opacity) < 1) return false;
      if (radiiOf(tcs, R.width, R.height).some(([x]) => x > 0.5)) return false;
      let fill = paint(tcs.backgroundColor, opacityOf(td));
      for (let n = td.parentElement; !fill && n && n !== table; n = n.parentElement) fill = paint(style(n).backgroundColor, opacityOf(n));
      const sides = borderSides(tcs).map(s => {
        if (!(s.w > 0 && s.st !== "none" && s.st !== "hidden")) return null;
        const p = paint(s.c, opacityOf(td));
        return p ? { ...p, width: s.w, dash: dashOf(s.st) } : null;
      });
      const pad = { T: num(tcs.paddingTop), R: num(tcs.paddingRight), B: num(tcs.paddingBottom), L: num(tcs.paddingLeft) };
      model.push({ td, r: r0, c: c0, rs: r1 - r0, cs: c1 - c0, rect: R, fill, sides, pad, valign: tcs.verticalAlign });
    }
    // общие грани ячеек: побеждает более толстая рамка, при равенстве — верхняя/левая ячейка
    const at = new Map(model.map(m => [m.r + ":" + m.c, m]));
    const pick = (a, b) => (!a ? b : !b ? a : (b.width > a.width + 0.1 ? b : a));
    for (const m of model) {
      const right = at.get(m.r + ":" + (m.c + m.cs));
      if (right && right.rs === m.rs) { const w = pick(m.sides[1], right.sides[3]); m.sides[1] = w; right.sides[3] = w; }
      const below = at.get((m.r + m.rs) + ":" + m.c);
      if (below && below.cs === m.cs) { const w = pick(m.sides[2], below.sides[0]); m.sides[2] = w; below.sides[0] = w; }
    }
    // простые ячейки — текст в таблице; сложные (бейджи, картинки, вложенные блоки с фоном) — поверх
    let chars = 0;
    for (const m of model) {
      const paras = cellIsSimple(m.td) ? cellParagraphs(m) : null;
      if (paras) {
        Object.assign(m, paras, { simple: true });
        chars += paras.paragraphs.reduce((k, p) => k + p.lines.reduce((q, l) => q + l.reduce((z, r) => z + r.text.length, 0), 0), 0);
      } else Object.assign(m, { simple: false, paragraphs: null, margins: { T: 0, R: 0, B: 0, L: 0 }, anchor: "t" });
    }
    S.tableParts.add(table);
    for (const tr of rows) { S.tableParts.add(tr); if (tr.parentElement !== table) S.tableParts.add(tr.parentElement); }
    for (const m of model) { S.tableParts.add(m.td); if (m.simple) S.tableSimpleCells.add(m.td); }
    const tfill = paint(cs.backgroundColor, opacityOf(table));
    if (tfill) S.ops.push({ kind: "shape", geom: "rect", ...rel(table.getBoundingClientRect()), fill: tfill, line: null, name: "Фон " + label(table) });
    S.ops.push({
      kind: "table", name: "Таблица " + label(table), x: xs[0] - S.ox, y: ys[0] - S.oy, w: xs[xs.length - 1] - xs[0], h: ys[ys.length - 1] - ys[0],
      cols: xs.slice(1).map((x, i) => round2(x - xs[i])), rows: ys.slice(1).map((y, i) => round2(y - ys[i])),
      cells: model.map(m => ({ r: m.r, c: m.c, rs: m.rs, cs: m.cs, fill: m.fill, borders: m.sides, margins: m.margins, anchor: m.anchor, paragraphs: m.paragraphs })),
    });
    S.textChars += chars;
    return true;
  }

  function cellIsSimple(td) {
    for (const d of td.querySelectorAll("*")) {
      const cs = style(d);
      if (cs.display === "none") continue;
      if (isReplaced(d) || isAtomicInline(d, cs) || isLayered(d, cs) || isFloat(cs)) return false;
      if (paint(cs.backgroundColor, 1) || notNone(cs.backgroundImage)) return false;
      if (borderSides(cs).some(s => s.w > 0 && s.st !== "none" && s.st !== "hidden" && paint(s.c, 1))) return false;
      if (cs.display !== "inline" && (num(cs.paddingLeft) + num(cs.marginLeft) > 0.5 || num(cs.paddingRight) + num(cs.marginRight) > 0.5)) return false;
    }
    return true;
  }

  /** Абзацы ячейки и поля так, чтобы базовые линии легли как в браузере (модель фиксированного интервала). */
  function cellParagraphs(m) {
    const cb = contentBox(m.td);
    const paras = [];
    for (const g of collectGroups(m.td)) {
      const lines = groupLines(g);
      if (!lines.length) continue;
      const cs = style(g.block);
      const align = alignOf(cs);
      if (!lines.every(ln => lineFits(ln, cb.l, cb.r, align))) return null;
      let pitch = null;
      for (let i = 1; i < lines.length; i++) {
        const p = lines[i].baseline - lines[i - 1].baseline;
        if (pitch == null) pitch = p;
        else if (Math.abs(p - pitch) > 1) return null;
        if (Math.abs(lines[i].D - lines[0].D) > 0.6) return null;
      }
      paras.push({ align, L: pitch != null ? pitch : lineHeightOf(cs, lines[0].chars[0].st), lines });
    }
    const R = m.rect;
    const lr = { L: round2(Math.max(0, cb.l - R.left)), R: round2(Math.max(0, R.right - cb.r - 2)) };
    if (!paras.length) return { paragraphs: [], margins: { T: 0, B: 0, ...lr }, anchor: "t" };
    const out = [];
    let prevBottom = null, textH = 0;
    const firstTop = paras[0].lines[0].baseline - paras[0].L + paras[0].lines[0].D;
    for (const p of paras) {
      const top = p.lines[0].baseline - p.L + p.lines[0].D;
      const before = prevBottom == null ? 0 : Math.max(0, top - prevBottom);
      out.push({ align: p.align, lineSpacing: round2(p.L), spaceBefore: round2(before), lines: p.lines.map(lineRuns) });
      textH += before + p.L * p.lines.length;
      prevBottom = top + p.L * p.lines.length;
    }
    const H = R.height;
    const off = firstTop - R.top;
    let anchor = m.valign === "middle" ? "ctr" : m.valign === "bottom" ? "b" : "t";
    let T, B;
    if (anchor === "t") { T = off; B = H - T - textH; }
    else if (anchor === "b") { B = H - textH - off; T = H - B - textH; }
    else {
      T = Math.min(m.pad.T, off);
      B = H - textH + T - 2 * off;
      if (B < 0) { B = 0; T = 2 * off - H + textH; }
    }
    if (T < -0.01 || B < -0.01) { anchor = "t"; T = off; B = H - T - textH; }
    return { paragraphs: out, margins: { T: round2(Math.max(0, T)), B: round2(Math.max(0, B)), ...lr }, anchor };
  }

  // ---------------------------------------------------------------- обход в порядке отрисовки
  function newGroup(block) { return { block, nodes: [], inlines: [] }; }

  /** Группы строчного содержимого блока и его блочных потомков (для ячеек таблиц). */
  function collectGroups(B) {
    const groups = [];
    let g = null;
    const walk = parent => {
      for (const node of parent.childNodes) {
        if (node.nodeType === 3) { if (node.data.length) (g = g || newGroup(B)).nodes.push(node); continue; }
        if (node.nodeType !== 1) continue;
        const cs = style(node);
        if (cs.display === "none") continue;
        if (cs.display === "contents") { walk(node); continue; }
        if (isLayered(node, cs) || isFloat(cs) || isReplaced(node) || isAtomicInline(node, cs)) continue;
        if (cs.display === "inline") { (g = g || newGroup(B)).inlines.push(node); walk(node); continue; }
        if (g) { groups.push(g); g = null; }
        groups.push(...collectGroups(node));
      }
    };
    walk(B);
    if (g) groups.push(g);
    return groups.filter(x => x.nodes.length);
  }

  function paintSC(el, pseudo) {
    emitBox(el);
    if (isReplaced(el)) { emitReplaced(el); return; }
    const neg = [], zero = [], pos = [];
    if (!pseudo) collectLayers(el, neg, zero, pos);
    const byZ = (a, b) => zOf(style(a)) - zOf(style(b));
    neg.sort(byZ); pos.sort(byZ);
    for (const c of neg) paintSC(c, false);
    paintBlocks(el);
    paintInline(el);
    for (const c of zero) paintSC(c, !createsSC(c, style(c)));
    for (const c of pos) paintSC(c, false);
  }

  function collectLayers(root, neg, zero, pos) {
    for (const c of kids(root)) {
      const cs = style(c);
      if (cs.display === "none") continue;
      if (createsSC(c, cs)) {
        const z = zOf(cs);
        (z < 0 ? neg : z > 0 ? pos : zero).push(c);
        continue;
      }
      if (isPositioned(cs)) zero.push(c);
      if (!isReplaced(c)) collectLayers(c, neg, zero, pos);
    }
  }

  /** Шаг 4: фоны и рамки блочных потомков в потоке. */
  function paintBlocks(root) {
    for (const c of kids(root)) {
      const cs = style(c);
      if (cs.display === "none" || isLayered(c, cs) || isFloat(cs) || isReplaced(c) || isAtomicInline(c, cs)) continue;
      if (cs.display === "inline") { paintBlocks(c); continue; }
      if (cs.display === "table" && S.nativeTables && !S.tableParts.has(c)) tryTable(c);
      emitBox(c);
      if (!S.tableSimpleCells.has(c)) paintBlocks(c);
    }
  }

  /** Шаги 5 и 7: плавающие, строчное содержимое (текст, inline-block, картинки) в порядке дерева. */
  function paintInline(B) {
    if (S.tableSimpleCells.has(B)) return;
    let g = null;
    const flush = () => { if (g) { emitGroup(g); g = null; } };
    const walk = parent => {
      for (const node of parent.childNodes) {
        if (node.nodeType === 3) { if (node.data.length) (g = g || newGroup(B)).nodes.push(node); continue; }
        if (node.nodeType !== 1) continue;
        const cs = style(node);
        if (cs.display === "none") continue;
        if (cs.display === "contents") { walk(node); continue; }
        if (isLayered(node, cs)) continue;
        if (isFloat(cs)) { paintSC(node, true); continue; }
        if (isReplaced(node)) { emitBox(node); emitReplaced(node); continue; }
        if (isAtomicInline(node, cs)) { paintSC(node, true); continue; }
        if (cs.display === "inline") { (g = g || newGroup(B)).inlines.push(node); walk(node); continue; }
        flush();
        paintInline(node);
      }
    };
    walk(B);
    flush();
  }

  // ---------------------------------------------------------------- фон слайда: декор без текста сливается с фоном
  function isDecor(el) {
    if (REPLACED.has(el.localName)) return false;
    if ((el.textContent || "").trim()) return false;
    return !el.querySelector("img,svg,canvas,video,iframe");
  }

  function mergeBackground() {
    const ops = S.ops;
    const i = ops.findIndex(o => o.kind === "bg");
    if (i < 0) return;
    while (i + 1 < ops.length && ops[i + 1].kind === "raster" && ops[i + 1].el && isDecor(ops[i + 1].el)) {
      const { el, cap } = ops[i + 1];
      S.bg.els.push(el);
      S.caps = S.caps.filter(c => c.id !== cap);
      el.setAttribute("data-pp-cap", String(S.bg.cap));
      ops.splice(i + 1, 1);
    }
  }

  // ---------------------------------------------------------------- API для to_pptx.mjs
  function setup() {
    if (document.getElementById("pp-iso-style")) return;
    const st = document.createElement("style");
    st.id = "pp-iso-style";
    st.textContent = ISO_CSS;
    document.head.appendChild(st);
  }

  const slides = selector => [...document.querySelectorAll(selector || ".slide")];

  function extractSlide(index, opts) {
    opts = opts || {};
    setup();
    const all = slides(opts.selector);
    all.forEach((s, j) => s.classList.toggle("pp-off", j !== index));
    window.scrollTo(0, 0);
    const slide = all[index];
    Object.assign(S, {
      slide, ops: [], caps: [], warns: [], capSeq: S.capSeq || 0, bg: null, textChars: 0,
      styles: new Map(), opacity: new Map(), metrics: new Map(), clips: new Map(), runStyles: new Map(),
      tableParts: new Set(), tableSimpleCells: new Set(), nativeTables: opts.nativeTables !== false,
    });
    const innerText = slide.innerText;
    const before = snapshotRects(slide);
    const t0 = performance.now();
    freeze(slide);
    const pseudo = materializePseudos(slide);
    // заморозка стилей и псевдоэлементы не должны сдвигать вёрстку
    let worst = 0, worstEl = null;
    for (const [el, r] of before) {
      const q = el.getBoundingClientRect();
      const d = Math.max(Math.abs(q.left - r.left), Math.abs(q.top - r.top), Math.abs(q.width - r.width), Math.abs(q.height - r.height));
      if (d > worst) { worst = d; worstEl = el; }
    }
    if (worst > 0.5) warn(`вёрстка сдвинулась после заморозки стилей на ${worst.toFixed(1)}px — ${label(worstEl)}`);
    S.styles = new Map();
    const R = slide.getBoundingClientRect();
    S.slideRect = R; S.ox = R.left; S.oy = R.top;
    paintSC(slide, false);
    mergeBackground();
    return {
      index, id: slide.id || "", width: R.width, height: R.height,
      ops: S.ops.map(({ el, ...rest }) => rest), captures: S.caps, bg: S.bg && { cap: S.bg.cap, color: S.bg.color },
      warnings: S.warns, innerText, textChars: S.textChars, pseudo, ms: Math.round(performance.now() - t0),
    };
  }

  function isolate(capId, mode) {
    document.documentElement.classList.add("pp-iso");
    for (const el of document.querySelectorAll(`[data-pp-cap="${capId}"]`)) {
      el.setAttribute("data-pp-show", mode === "bg" ? (el === S.slide ? "box" : "tree") : mode);
    }
  }

  function unisolate() {
    document.documentElement.classList.remove("pp-iso");
    for (const el of document.querySelectorAll("[data-pp-show]")) el.removeAttribute("data-pp-show");
  }

  window.__pptx = { extractSlide, isolate, unisolate, count: sel => slides(sel).length };
})();
