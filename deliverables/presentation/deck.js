/* Зазеркалье — сборка слайдов: хром, подвал, данные из data.json, графики.
   Только DOM-методы (без innerHTML). Цифры — в data.json, вёрстка — в index.html. */
(function () {
  "use strict";
  const NS = "http://www.w3.org/2000/svg";
  const C = { cpu: "#3f86c4", lime: "#a8d42c", limeSoft: "rgba(168,212,44,.42)", ink: "#eef4f1", ink2: "#b9cbd2", ink3: "#7f9cab", surface: "#0a2031" };
  const get = (o, path) => path.split(".").reduce((a, k) => (a == null ? undefined : a[k]), o);
  // округление «половина вверх» без артефактов float: 22.65 → 22.7
  const round = (v, d) => (Math.round((v + Number.EPSILON) * 10 ** d) / 10 ** d).toFixed(d);
  const fmt = (v, d) => (v == null ? "—" : (typeof v === "number" ? (d != null ? round(v, d) : String(v)) : String(v)));

  /* 95 % доверительный интервал Уилсона для доли k из n */
  function wilson(k, n) {
    if (!n) return [0, 1];
    const z = 1.959964, p = k / n, den = 1 + z * z / n;
    const c = (p + z * z / (2 * n)) / den;
    const hw = z * Math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den;
    return [Math.max(0, c - hw), Math.min(1, c + hw)];
  }

  /* HTML-узел: h("div", {class: "x"}, "текст", h("b", null, "жирный")) */
  function h(tag, attrs, ...kids) {
    const n = document.createElement(tag);
    if (attrs) for (const k in attrs) {
      if (k === "class") n.className = attrs[k];
      else if (k === "style") n.style.cssText = attrs[k];
      else n.setAttribute(k, attrs[k]);
    }
    for (const c of kids.flat()) {
      if (c == null || c === false) continue;
      n.append(c.nodeType ? c : document.createTextNode(String(c)));
    }
    return n;
  }
  /* SVG-узел */
  function s(tag, attrs, parent, text) {
    const n = document.createElementNS(NS, tag);
    for (const k in attrs) n.setAttribute(k, attrs[k]);
    if (text != null) n.textContent = text;
    if (parent) parent.appendChild(n);
    return n;
  }
  const swatch = (color, label) => h("span", null, h("span", { class: "sw", style: "background:" + color }), label);
  /* полоса со скруглённым концом справа и прямым основанием слева */
  const barPath = (x, y, w, hh, r) => {
    r = Math.min(r, w / 2, hh / 2);
    return "M" + x + "," + y + " h" + (w - r) + " q" + r + ",0 " + r + "," + r + " v" + (hh - 2 * r) + " q0," + r + " -" + r + "," + r + " h-" + (w - r) + " z";
  };

  /* ---- хром окна и подвал на каждом слайде ---- */
  function chrome() {
    const slides = [...document.querySelectorAll(".slide")];
    const main = slides.filter(x => !x.dataset.backup);
    const back = slides.filter(x => x.dataset.backup);
    slides.forEach(sl => {
      const isB = !!sl.dataset.backup;
      const idx = isB ? back.indexOf(sl) + 1 : main.indexOf(sl) + 1;
      const total = isB ? back.length : main.length;
      sl.prepend(h("header", { class: "chrome" },
        h("span", { class: "sqs" }, h("i"), h("i"), h("i")),
        h("span", { class: "brand" }, "Зазеркалье ", h("span", null, "· видеоаналитика ТБ на LinQ TPU")),
        h("span", { class: "ev" }, "Открой#Моспром · кейс LinQ"),
        h("span", { class: "btns" }, h("i", null, "–"), h("i", null, "□"), h("i", null, "×"))));
      const prog = h("div", { class: "prog" + (isB ? " bk" : "") });
      for (let i = 1; i <= total; i++) prog.append(h("i", { class: i <= idx ? "on" : "" }));
      const pad = n => String(n).padStart(2, "0");
      const num = isB
        ? h("div", { class: "num" }, "Б" + idx + " ", h("span", null, "/ Б" + total))
        : h("div", { class: "num" }, pad(idx) + " ", h("span", null, "/ " + pad(total)));
      sl.append(h("footer", { class: "foot" },
        sl.dataset.src ? h("div", { class: "src" }, "Источник: " + sl.dataset.src) : null, prog, num));
    });
  }

  /* ---- data-bind="путь.в.json" → текст ---- */
  function bind(data) {
    document.querySelectorAll("[data-bind]").forEach(n => {
      const v = get(data, n.dataset.bind);
      if (v == null) return;
      n.textContent = fmt(v) + (n.dataset.suffix || "");
    });
  }

  /* ---- плейсхолдеры «будет обновлено к защите» ---- */
  const isDone = b => !!b && b.status === "done";
  function ptags(data) {
    document.querySelectorAll("[data-ptag]").forEach(n => {
      const b = get(data, n.dataset.ptag);
      if (isDone(b)) {
        n.classList.add("done");
        n.textContent = "замер" + (b.measured_at ? " " + b.measured_at : "");
        const box = n.closest(".pending-box");
        if (box) box.classList.remove("pending-box");
      } else {
        n.textContent = data.pending_label;
      }
    });
  }

  /* ---- строки статуса (data-status="pending.xxx"): текст из pending_text / done_text ---- */
  function statusLines(data) {
    document.querySelectorAll("[data-status]").forEach(n => {
      const b = get(data, n.dataset.status);
      if (!b) return;
      const t = n.querySelector("[data-status-text]");
      if (t) t.textContent = isDone(b) ? b.done_text + (b.path ? ": " + b.path : "") : b.pending_text;
    });
  }

  /* ---- скриншот, который появится позже (data-shot="pending.xxx", поле src) ---- */
  function shots(data) {
    document.querySelectorAll("[data-shot]").forEach(n => {
      const b = get(data, n.dataset.shot);
      if (!isDone(b) || !b.src) return;
      n.querySelector(".pane").replaceChildren(h("img", { src: b.src, alt: b.title || "" }));
      n.classList.remove("pending-box");
    });
  }

  /* ---- слайд 8: плитки качества ---- */
  function qualityTiles(data) {
    const Q = data.quality, host = document.getElementById("q-tiles");
    if (!host || !Q) return;
    const o = Q.overall, b = Q.base || {};
    const nP = o.tp + o.fp, nR = o.tp + o.fn;
    const [plo, phi] = wilson(o.tp, nP), [rlo, rhi] = wilson(o.tp, nR);
    const tiles = [
      { k: "Precision", v: fmt(o.tp / nP, 2), s: o.tp + " из " + nP + " · 95 % ДИ " + fmt(plo, 2) + "–" + fmt(phi, 2), hl: true },
      { k: "Recall", v: fmt(o.tp / nR, 2), s: o.tp + " из " + nR + " · 95 % ДИ " + fmt(rlo, 2) + "–" + fmt(rhi, 2), hl: true },
      { k: "Дубли", v: String(o.duplicates), s: "в базовой настройке — " + b.duplicates },
      { k: "Ложные тревоги", v: String(o.fp), s: "на " + o.duration_s + " с: скорее всего меньше ~" + o.fa_per_hour_upper + " в час" },
      { k: "Задержка, p50", v: fmt(o.latency_p50_s, 2), unit: "с", s: "p95 " + fmt(o.latency_p95_s, 2) + " с — из-за СИЗ" },
    ];
    tiles.forEach(t => host.append(h("div", { class: "panel kt" + (t.hl ? " hl" : "") },
      h("div", { class: "k" }, t.k),
      h("div", { class: "v" }, t.v, t.unit ? h("small", null, t.unit) : null),
      h("div", { class: "s" }, t.s))));
  }

  /* ---- слайд 8: recall по типам с интервалами Уилсона ---- */
  function forest(data) {
    const Q = data.quality, svg = document.getElementById("forest");
    if (!svg || !Q) return;
    const W = +svg.getAttribute("width"), x0 = 430, x1 = W - 30, sx = v => x0 + v * (x1 - x0);
    const rows = Q.by_type.concat([{ id: "Σ", title: "Все типы", k: Q.overall.tp, n: Q.overall.tp + Q.overall.fn, total: true }]);
    const top = 34, rowH = 50, gapTotal = 16;
    const plotB = top + rows.length * rowH + gapTotal;
    const H = plotB + 32;
    svg.setAttribute("height", H);
    svg.setAttribute("viewBox", "0 0 " + W + " " + H);
    s("text", { x: x0, y: 16, class: "tick" }, svg, "доля найденных событий (recall)");
    const sepY = top + (rows.length - 1) * rowH + gapTotal / 2;   // над строкой «Все типы» сетки нет: там подпись ДИ
    [0, 0.25, 0.5, 0.75, 1].forEach(t => {
      s("line", { x1: sx(t), x2: sx(t), y1: top - 8, y2: t ? sepY : plotB, class: t ? "grid" : "base" }, svg);
      s("text", { x: sx(t), y: plotB + 24, "text-anchor": t === 1 ? "end" : (t ? "middle" : "start"), class: "tick" }, svg, t === 0 ? "0" : String(t));
    });
    rows.forEach((r, i) => {
      const cy = top + i * rowH + rowH / 2 + (r.total ? gapTotal : 0);
      if (r.total) s("line", { x1: 0, x2: W, y1: cy - rowH / 2 - gapTotal / 2, y2: cy - rowH / 2 - gapTotal / 2, class: "base" }, svg);
      s("text", { x: 0, y: cy + 6, "font-family": "JBMono", "font-size": 16, fill: r.total ? C.ink : C.lime }, svg, r.id);
      const t = s("text", { x: 44, y: cy + 7, "font-size": r.total ? 23 : 21, "font-weight": r.total ? 700 : 500, fill: r.not_claimed ? C.ink3 : C.ink }, svg, r.title);
      if (r.not_claimed) s("tspan", { fill: C.ink3, "font-size": 17 }, t, " · не заявляем");
      s("text", { x: x0 - 26, y: cy + 6, "text-anchor": "end", "font-family": "JBMono", "font-size": 18, fill: r.not_claimed ? C.ink3 : C.ink2 }, svg, r.k + " из " + r.n);
      const [lo, hi] = wilson(r.k, r.n), p = r.n ? r.k / r.n : 0;
      const g = s("g", {}, svg);
      s("title", {}, g, r.title + ": " + r.k + " из " + r.n + ", доля " + fmt(p, 2) + ", 95 % ДИ " + fmt(lo, 2) + "–" + fmt(hi, 2));
      const stroke = r.total ? C.lime : (r.not_claimed ? C.ink3 : C.ink2), sw = r.total ? 4 : 2.5;
      s("line", { x1: sx(lo), x2: sx(hi), y1: cy, y2: cy, stroke: stroke, "stroke-width": sw, "stroke-linecap": "round" }, g);
      [lo, hi].forEach(v => s("line", { x1: sx(v), x2: sx(v), y1: cy - 7, y2: cy + 7, stroke: stroke, "stroke-width": 2 }, g));
      s("circle", { cx: sx(p), cy: cy, r: r.total ? 9 : 7, fill: r.not_claimed ? C.ink3 : C.lime, stroke: C.surface, "stroke-width": 2 }, g);
      if (r.total) s("text", { x: sx(lo) - 16, y: cy + 7, "text-anchor": "end", "font-family": "JBMono", "font-size": 18, "font-weight": 600, fill: C.ink }, svg,
        fmt(p, 2) + " · ДИ " + fmt(lo, 2) + "–" + fmt(hi, 2));
    });
  }

  /* ---- слайд 8: качество на TPU (плейсхолдер) ---- */
  function evalTpu(data) {
    const e = data.pending.eval_tpu, host = document.getElementById("eval-tpu");
    if (!host || !e) return;
    e.metrics.forEach(m => {
      const has = m.value != null;
      host.append(h("div", { class: "tile", style: has ? "border-color:var(--lime)" : "" },
        h("div", { class: "k" }, m.title),
        h("div", { class: "pval" + (has ? " ok" : "") }, has ? fmt(m.value) + (m.unit ? " " + m.unit : "") : "—")));
    });
  }

  /* ---- слайд 9: дообучение по слабым местам ---- */
  function finetune(data) {
    const F = data.pending.finetune, host = document.getElementById("ft-rows");
    if (!host || !F) return;
    F.rows.forEach(r => {
      const done = r.status === "done" && r.after != null;
      const after = h("td", { class: "after" + (done ? " done" : "") },
        done ? [h("span", { class: "val" }, String(r.after)), r.card ? h("span", { class: "cardref" }, r.card) : null]
             : h("span", { class: "ptag" }, data.pending_label));
      host.append(h("tr", null,
        h("td", null, h("span", { class: "id" }, r.id), h("span", { class: "tt" }, r.title)),
        h("td", { class: "muted" }, r.weak),
        h("td", { class: "now" }, r.now),
        h("td", { class: "muted" }, r.datasets),
        after));
    });
  }

  /* ---- слайд 6: из чего складывается задержка оповещения ---- */
  function latencyChart(data) {
    const L = data.latency, svg = document.getElementById("lat-chart");
    if (!svg || !L) return;
    const parts = [
      { key: "window", color: C.cpu, label: "окно подтверждения правила", short: "окно" },
      { key: "edge", color: C.lime, label: "edge: от подтверждения до отправки", short: "edge" },
      { key: "channel", color: C.limeSoft, label: "канал стенд → ноутбук через VPN", short: "канал" },
    ];
    const lg = document.getElementById("lat-legend");
    if (lg) lg.append(...parts.map(p => swatch(p.color, p.label)));
    const W = +svg.getAttribute("width"), x0 = 220, x1 = W - 200, max = 3.5, sx = v => x0 + (v / max) * (x1 - x0);
    const top = 8, rowH = 84, bh = 24;
    const plotB = top + L.runs.length * rowH;
    const H = plotB + 28;
    svg.setAttribute("height", H);
    svg.setAttribute("viewBox", "0 0 " + W + " " + H);
    // сетки нет: у каждого сегмента прямая подпись; ось — линия 0 и короткие деления внизу
    s("line", { x1: sx(0), x2: sx(0), y1: top, y2: plotB, class: "base" }, svg);
    s("line", { x1: sx(0), x2: sx(3), y1: plotB, y2: plotB, class: "grid" }, svg);
    for (let t = 0; t <= 3; t += 1) {
      if (t) s("line", { x1: sx(t), x2: sx(t), y1: plotB - 6, y2: plotB, class: "base" }, svg);
      s("text", { x: sx(t), y: plotB + 22, "text-anchor": t ? "middle" : "start", class: "tick" }, svg, t ? t + " с" : "0");
    }
    s("text", { x: W - 186, y: top + 2, "font-family": "JBMono", "font-size": 14, fill: C.ink3, "dominant-baseline": "hanging" }, svg, "p50 / p95");
    L.runs.forEach((r, i) => {
      const y = top + i * rowH + 18;
      s("text", { x: 0, y: y + 18, "font-family": "Tektur", "font-size": 22, "font-weight": 600, fill: C.ink }, svg, r.label);
      String(r.sub).split(" · ").forEach((line, j) => s("text", { x: 0, y: y + 42 + j * 19, "font-size": 15, fill: C.ink3 }, svg, line));
      let acc = 0;
      const total = parts.reduce((a, p) => a + r[p.key], 0);
      parts.forEach((p, j) => {
        const v = r[p.key];
        const xa = sx(acc) + (j ? 1 : 0), xb = sx(acc + v) - (j < parts.length - 1 ? 1 : 0);
        const w = Math.max(2, xb - xa);
        const el = j === parts.length - 1 ? s("path", { d: barPath(xa, y, w, bh, 4), fill: p.color }, svg)
                                          : s("rect", { x: xa, y: y, width: w, height: bh, fill: p.color }, svg);
        s("title", {}, el, r.label + " — " + p.label + ": " + fmt(v, 2) + " с (среднее)");
        acc += v;
      });
      s("text", { x: sx(total) + 10, y: y + bh - 5, "font-family": "JBMono", "font-size": 18, "font-weight": 600, fill: C.ink }, svg, "= " + fmt(total, 2));
      s("text", { x: x0, y: y + bh + 22, "font-family": "JBMono", "font-size": 15, fill: C.ink2 }, svg,
        parts.map(p => p.short + " " + fmt(r[p.key], 2)).join(" + ") + " — средние, с");
      s("text", { x: W - 186, y: y + bh - 2, "font-family": "Tektur", "font-size": 26, "font-weight": 700, fill: C.ink }, svg, fmt(r.p50, 2));
      s("text", { x: W - 100, y: y + bh - 2, "font-family": "JBMono", "font-size": 18, fill: C.ink2 }, svg, "/ " + fmt(r.p95, 2));
    });
  }

  /* ---- график: TPU против CPU (горизонтальные сгруппированные бары) ---- */
  function tpuChart(data) {
    const D = data.tpu_vs_cpu, svg = document.getElementById("tpu-chart");
    if (!svg) return;
    document.getElementById("tpu-legend").append(swatch(C.cpu, D.cpu_label), swatch(C.lime, D.tpu_label));
    const W = +svg.getAttribute("width"), x0 = 150, x1 = W - 150, top = 14, band = 78, bh = 22, gap = 2;
    const max = 55, sx = v => x0 + (v / max) * (x1 - x0);
    const plotB = top + D.rows.length * band - 8;
    const H = plotB + 34;
    svg.setAttribute("height", H);
    svg.setAttribute("viewBox", "0 0 " + W + " " + H);
    // сетки нет: у каждого бара прямая подпись; внизу ось с короткими делениями
    s("line", { x1: x0, x2: sx(50), y1: plotB, y2: plotB, class: "grid" }, svg);
    for (let t = 0; t <= 50; t += 10) {
      s("line", { x1: sx(t), x2: sx(t), y1: plotB - 6, y2: plotB, class: "base" }, svg);
      s("text", { x: sx(t), y: plotB + 26, "text-anchor": "middle", class: "tick" }, svg, t);
    }
    s("text", { x: sx(55), y: plotB + 26, "text-anchor": "middle", class: "tick" }, svg, "к/с");
    D.rows.forEach((r, i) => {
      const y = top + i * band + 6;
      s("text", { x: 0, y: y + bh + 2, "font-family": "Tektur", "font-size": 24, "font-weight": 600, fill: C.ink }, svg, r.model);
      s("text", { x: 0, y: y + bh + 26, "font-size": 16, fill: C.ink3 }, svg, r.model.indexOf("pose") === 0 ? "YOLOv8n-pose" : "YOLOv8n");
      const bar = (v, yy, color, label) => {
        const w = Math.max(6, sx(v) - x0);
        const p = s("path", { d: barPath(x0, yy, w, bh, 4), fill: color }, svg);
        s("title", {}, p, r.full + " — " + label + ": " + v + " к/с");
        s("text", { x: x0 + w + 10, y: yy + bh - 4, class: "val", "font-size": 20 }, svg, fmt(v, 1));
      };
      bar(r.cpu, y, C.cpu, "CPU");
      bar(r.tpu, y + bh + gap, C.lime, "TPU");
      const g = s("g", {}, svg);
      s("rect", { x: W - 92, y: y + 2, width: 90, height: 44, rx: 6, fill: "rgba(168,212,44,.12)", stroke: C.lime, "stroke-width": 1.5 }, g);
      s("text", { x: W - 47, y: y + 33, "text-anchor": "middle", "font-family": "Tektur", "font-size": 25, "font-weight": 700, fill: C.ink }, g, "×" + round(r.tpu / r.cpu, 1));
    });
    s("line", { x1: x0, x2: x0, y1: top - 6, y2: plotB, class: "base" }, svg);
  }

  /* ---- график: det 480 по окнам (сгруппированные столбцы) ---- */
  function winChart(data) {
    const D = data.det480_windows, svg = document.getElementById("win-chart");
    if (!svg) return;
    const colors = [C.cpu, C.lime];
    document.getElementById("win-legend").append(...D.series.map((x, i) => swatch(colors[i], x.label)));
    const W = 704, y0 = 410, y1 = 30, x0 = 60, max = 50, sy = v => y0 - (v / max) * (y0 - y1);
    // сетки нет: у каждого столбца прямая подпись; слева ось с короткими делениями
    s("line", { x1: x0, x2: W, y1: sy(0), y2: sy(0), class: "base" }, svg);
    s("line", { x1: x0, x2: x0, y1: sy(50), y2: sy(0), class: "grid" }, svg);
    for (let t = 0; t <= 50; t += 10) {
      s("line", { x1: x0, x2: x0 + 6, y1: sy(t), y2: sy(t), class: "base" }, svg);
      s("text", { x: x0 - 12, y: sy(t) + 6, "text-anchor": "end", class: "tick" }, svg, t);
    }
    const gw = (W - x0) / D.windows.length, bw = 64;
    D.windows.forEach((w, i) => {
      const cx = x0 + gw * i + gw / 2;
      s("text", { x: cx, y: y0 + 36, "text-anchor": "middle", "font-size": 21, fill: C.ink2 }, svg, w);
      D.series.forEach((x, j) => {
        const v = x.values[i], bx = cx - bw - 1 + j * (bw + 2), y = sy(v), hh = y0 - y;
        const d = "M" + bx + "," + y0 + " v-" + (hh - 4) + " q0,-4 4,-4 h" + (bw - 8) + " q4,0 4,4 v" + (hh - 4) + " z";
        const p = s("path", { d: d, fill: colors[j] }, svg);
        s("title", {}, p, x.label + ", " + w + ": " + v + " к/с");
        s("text", { x: bx + bw / 2, y: y - 10, "text-anchor": "middle", class: "val" }, svg, fmt(v, 1));
      });
    });
  }

  /* ---- камеры на один TPU: пиксельные ряды (data-rows="main" — только основные наборы) ---- */
  function cams(data) {
    const CP = data.cameras_per_tpu, host = document.getElementById("cams");
    if (!host) return;
    const only = host.dataset.rows === "main";
    const slots = 10;
    CP.rows.filter(r => !only || r.main).forEach(r => {
      const px = h("div", { class: "px" });
      for (let i = 0; i < slots; i++) px.append(h("i", { class: i < r.cams ? "on" : "" }));
      host.append(h("div", { class: "cam" },
        h("div", { class: "nm" }, r.set, h("small", null, r.use)),
        h("div", null, px, h("div", { class: "src-in", style: "margin-top:5px" },
          fmt(r.fps) + " / " + CP.sample_fps + " = " + round(r.fps / CP.sample_fps, 1))),
        h("div", { class: "cn" }, String(r.cams), h("small", null, r.cams < 5 ? "камеры" : "камер"))));
    });
  }

  /* ---- окно подтверждения по кадрам (слайд 6, схема) ---- */
  function frames() {
    const svg = document.getElementById("frames");
    if (!svg) return;
    const lit = new Set([3, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 26]);
    const n = 34, pitch = 44, sz = 34, x0 = 236, y = 56;
    s("text", { x: 0, y: y + 12, "font-size": 20, fill: C.ink2 }, svg, "Кадры анализа,");
    s("text", { x: 0, y: y + 36, "font-size": 20, fill: C.ink2 }, svg, "5 к/с · схема");
    for (let i = 0; i < n; i++) {
      const on = lit.has(i);
      s("rect", { x: x0 + i * pitch, y: y, width: sz, height: sz, fill: on ? C.lime : "none",
        stroke: on ? C.lime : "rgba(132,180,200,.35)", "stroke-width": 1.5, opacity: (i > 17 && on) ? 0.5 : 1 }, svg);
    }
    const cx3 = x0 + 3 * pitch + sz / 2;
    s("text", { x: cx3, y: y + sz + 26, "text-anchor": "middle", "font-size": 19, fill: C.ink2 }, svg, "одиночный кадр —");
    s("text", { x: cx3, y: y + sz + 48, "text-anchor": "middle", "font-size": 19, fill: C.ink2 }, svg, "не тревога");
    const wx = x0 + 8 * pitch - 4, ww = 10 * pitch;
    s("path", { d: "M" + wx + "," + (y - 8) + " v-10 h" + ww + " v10", fill: "none", stroke: C.ink, "stroke-width": 2 }, svg);
    s("text", { x: wx + ww / 2, y: y - 26, "text-anchor": "middle", "font-size": 20, fill: C.ink }, svg, "окно: нарушение на большей части кадров");
    const ex = x0 + 17 * pitch + sz + 4;
    s("rect", { x: ex - 2, y: y - 6, width: 4, height: sz + 12, fill: "#ff5d52" }, svg);
    s("text", { x: ex - 8, y: y + sz + 26, "text-anchor": "end", "font-size": 20, "font-weight": 650, fill: C.ink }, svg, "одно событие → оператору");
    const rx = x0 + 18 * pitch, rw = 9 * pitch - 8;
    s("path", { d: "M" + rx + "," + (y - 8) + " v-10 h" + rw + " v10", fill: "none", stroke: C.ink3, "stroke-width": 2, "stroke-dasharray": "5 4" }, svg);
    s("text", { x: rx + rw / 2, y: y - 26, "text-anchor": "middle", "font-size": 20, fill: C.ink2 }, svg, "тот же трек — повторов нет");
    const cx = x0 + 30.5 * pitch;
    s("text", { x: cx, y: y - 30, "text-anchor": "middle", "font-size": 19, fill: C.ink2 }, svg, "нарушения нет —");
    s("text", { x: cx, y: y - 9, "text-anchor": "middle", "font-size": 19, fill: C.ink2 }, svg, "событие закрыто");
  }

  /* ---- ссылки на документы (слайд 16) ---- */
  function docLinks(data) {
    const host = document.getElementById("doc-links");
    if (!host || !data.links) return;
    (data.links.docs || []).forEach(d => host.append(h("div", { class: "dl" }, h("b", null, d.path), h("span", null, d.what))));
  }

  /* ---- сценарии S1–S10 (резерв) ---- */
  function scen(data) {
    const host = document.getElementById("scen-rows");
    if (!host) return;
    data.scenarios.rows.forEach(r => {
      host.append(h("tr", null,
        h("td", { class: "id" }, r.id),
        h("td", null, r.title),
        h("td", { class: "mono", style: "font-size:17px;color:var(--ink-2)" }, r.plugin),
        h("td", { class: "muted" }, r.models),
        h("td", null, h("span", { class: "st " + r.state }), h("span", { class: "muted" }, r.status))));
    });
  }

  /* ---- метки статуса: data-sst="путь.в.json" → done | part | wip (подписи — data.json → state_labels) ---- */
  function states(data) {
    const names = data.state_labels || { done: "работает", part: "частично", wip: "в работе" };
    document.querySelectorAll("[data-sst]").forEach(n => {
      const st = get(data, n.dataset.sst) || "wip";
      n.textContent = names[st] || st;
      n.classList.add(st);
    });
  }

  /* ---- вопросы экспертам (резерв) ---- */

  function questions(data) {
    const host = document.getElementById("questions"), Q = data.questions;
    if (!host || !Q) return;
    host.style.setProperty("--qcols", String(Q.groups.length));
    let n = 0;
    Q.groups.forEach(g => {
      const ol = h("ol", { start: String(n + 1) });
      g.items.forEach(t => { n += 1; ol.append(h("li", null, h("span", { class: "qn" }, String(n)), t)); });
      host.append(h("div", { class: "panel" }, h("div", { class: "lbl" }, g.title), ol));
    });
  }

  function fail(msg) {
    document.body.dataset.error = msg;
    document.body.prepend(h("div", { class: "data-error" },
      "data.json не загружен (" + msg + "). Откройте через локальный сервер: node render.mjs --serve, или смотрите PDF."));
  }

  async function main() {
    chrome();
    frames();
    let data;
    try {
      const r = await fetch("data.json", { cache: "no-store" });
      if (!r.ok) throw new Error("HTTP " + r.status);
      data = await r.json();
    } catch (e) {
      fail(e.message);
      document.body.dataset.ready = "1";
      return;
    }
    try {
      bind(data); ptags(data); statusLines(data); shots(data);
      qualityTiles(data); forest(data); evalTpu(data); finetune(data); latencyChart(data);
      tpuChart(data); winChart(data); cams(data); docLinks(data); scen(data); questions(data); states(data);
    } catch (e) {
      fail("ошибка сборки: " + e.message);
    }
    await document.fonts.ready;
    document.body.dataset.ready = "1";
  }
  main();
})();
