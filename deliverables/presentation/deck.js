/* Зазеркалье — сборка слайдов: хром, подвал, данные из data.json, графики.
   Только DOM-методы (без innerHTML). Цифры — в data.json, вёрстка — в index.html. */
(function () {
  "use strict";
  const NS = "http://www.w3.org/2000/svg";
  const get = (o, path) => path.split(".").reduce((a, k) => (a == null ? undefined : a[k]), o);
  // округление «половина вверх» без артефактов float: 22.65 → 22.7
  const round = (v, d) => (Math.round((v + Number.EPSILON) * 10 ** d) / 10 ** d).toFixed(d);
  const fmt = (v, d) => (v == null ? "—" : (typeof v === "number" ? (d != null ? round(v, d) : String(v)) : String(v)));

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
  function evalTiles(data) {
    const e = data.pending.eval, host = document.getElementById("eval-tiles");
    if (!host) return;
    e.metrics.forEach(m => {
      const has = m.value != null;
      host.append(h("div", { class: "tile", style: has ? "border-color:var(--lime)" : "" },
        h("div", { class: "k" }, m.title, m.hint ? h("i", null, " · " + m.hint) : null),
        h("div", { class: "pval" + (has ? " ok" : "") }, has ? fmt(m.value) + (m.unit ? " " + m.unit : "") : "—")));
    });
  }
  function latency(data) {
    const L = data.pending.edge_latency, done = isDone(L) && L.p50_s != null;
    const val = done ? fmt(L.p50_s) + " с" : "—";
    const big = document.querySelector("[data-latency]");
    if (big) {
      big.classList.toggle("pending-box", !done);
      big.querySelector(".k").textContent = "Сквозная задержка, edge + TPU" + (done ? " — замер" : "");
      const v = big.querySelector(".v");
      v.textContent = val;
      v.classList.add(done ? "big" : "pval");
      v.classList.toggle("ok", done);
      const n = big.querySelector(".n");
      n.textContent = "";
      if (done) n.append("p95 " + fmt(L.p95_s) + " с · " + (L.measured_at || ""));
      else n.append(h("span", { class: "ptag" }, data.pending_label),
        h("div", { style: "margin-top:8px" }, L.estimate_note + " ≈ " + L.estimate_s + " с"));
    }
    const cmp = document.querySelector("[data-latency-compact]");
    if (cmp) {
      const v = cmp.querySelector(".pval");
      v.textContent = val;
      v.classList.toggle("ok", done);
      cmp.querySelector(".k").textContent = "Сквозная задержка, edge + TPU" + (done ? ", p50" : "") + ": " + L.what;
    }
  }

  /* ---- строки статуса (data-status="pending.xxx"): текст из pending_text / done_text ---- */
  function statusLines(data) {
    document.querySelectorAll("[data-status]").forEach(n => {
      const b = get(data, n.dataset.status);
      if (!b) return;
      const t = n.querySelector("[data-status-text]");
      if (t) t.textContent = isDone(b) ? b.done_text : b.pending_text;
    });
  }

  /* ---- график: TPU против CPU (горизонтальные сгруппированные бары) ---- */
  function tpuChart(data) {
    const D = data.tpu_vs_cpu, svg = document.getElementById("tpu-chart");
    if (!svg) return;
    document.getElementById("tpu-legend").append(swatch("var(--cpu)", D.cpu_label), swatch("var(--lime)", D.tpu_label));
    const W = 956, x0 = 150, x1 = 800, top = 26, band = 122, bh = 26, gap = 2;
    const max = 55, sx = v => x0 + (v / max) * (x1 - x0);
    const plotB = top + D.rows.length * band - 14;
    const H = plotB + 40;
    svg.setAttribute("height", H);
    svg.setAttribute("viewBox", "0 0 " + W + " " + H);
    for (let t = 0; t <= 50; t += 10) {
      s("line", { x1: sx(t), x2: sx(t), y1: top - 10, y2: plotB, class: "grid" }, svg);
      s("text", { x: sx(t), y: plotB + 28, "text-anchor": "middle", class: "tick" }, svg, t);
    }
    s("text", { x: x1 + 40, y: plotB + 28, "text-anchor": "start", class: "tick" }, svg, "к/с");
    D.rows.forEach((r, i) => {
      const y = top + i * band + 8;
      s("text", { x: 0, y: y + bh + 4, "font-family": "Tektur", "font-size": 26, "font-weight": 600, fill: "#eef4f1" }, svg, r.model);
      s("text", { x: 0, y: y + bh + 30, "font-size": 17, fill: "#7f9cab" }, svg, r.model.indexOf("pose") === 0 ? "YOLOv8n-pose" : "YOLOv8n");
      const bar = (v, yy, color, label) => {
        const w = Math.max(6, sx(v) - x0);
        const d = "M" + x0 + "," + yy + " h" + (w - 4) + " q4,0 4,4 v" + (bh - 8) + " q0,4 -4,4 h-" + (w - 4) + " z";
        const p = s("path", { d: d, fill: color }, svg);
        s("title", {}, p, r.full + " — " + label + ": " + v + " к/с");
        s("text", { x: x0 + w + 10, y: yy + bh - 5, class: "val" }, svg, fmt(v, 1));
      };
      bar(r.cpu, y, "#3f86c4", "CPU");
      bar(r.tpu, y + bh + gap, "#a8d42c", "TPU");
      const g = s("g", {}, svg);
      s("rect", { x: 862, y: y + 5, width: 92, height: 46, rx: 6, fill: "rgba(168,212,44,.12)", stroke: "#a8d42c", "stroke-width": 1.5 }, g);
      s("text", { x: 908, y: y + 37, "text-anchor": "middle", "font-family": "Tektur", "font-size": 26, "font-weight": 700, fill: "#eef4f1" }, g, "×" + round(r.tpu / r.cpu, 1));
    });
    s("line", { x1: x0, x2: x0, y1: top - 10, y2: plotB, class: "base" }, svg);
  }

  /* ---- график: det 480 по окнам (сгруппированные столбцы) ---- */
  function winChart(data) {
    const D = data.det480_windows, svg = document.getElementById("win-chart");
    if (!svg) return;
    const colors = ["#3f86c4", "#a8d42c"];
    document.getElementById("win-legend").append(...D.series.map((x, i) => swatch(colors[i], x.label)));
    const W = 704, y0 = 410, y1 = 30, x0 = 60, max = 50, sy = v => y0 - (v / max) * (y0 - y1);
    for (let t = 0; t <= 50; t += 10) {
      s("line", { x1: x0, x2: W, y1: sy(t), y2: sy(t), class: t ? "grid" : "base" }, svg);
      s("text", { x: x0 - 12, y: sy(t) + 6, "text-anchor": "end", class: "tick" }, svg, t);
    }
    const gw = (W - x0) / D.windows.length, bw = 64;
    D.windows.forEach((w, i) => {
      const cx = x0 + gw * i + gw / 2;
      s("text", { x: cx, y: y0 + 36, "text-anchor": "middle", "font-size": 21, fill: "#b9cbd2" }, svg, w);
      D.series.forEach((x, j) => {
        const v = x.values[i], bx = cx - bw - 1 + j * (bw + 2), y = sy(v), hh = y0 - y;
        const d = "M" + bx + "," + y0 + " v-" + (hh - 4) + " q0,-4 4,-4 h" + (bw - 8) + " q4,0 4,4 v" + (hh - 4) + " z";
        const p = s("path", { d: d, fill: colors[j] }, svg);
        s("title", {}, p, x.label + ", " + w + ": " + v + " к/с");
        s("text", { x: bx + bw / 2, y: y - 10, "text-anchor": "middle", class: "val" }, svg, fmt(v, 1));
      });
    });
  }

  /* ---- камеры на один TPU: пиксельные ряды ---- */
  function cams(data) {
    const C = data.cameras_per_tpu, host = document.getElementById("cams");
    if (!host) return;
    const slots = 10;
    C.rows.forEach(r => {
      const px = h("div", { class: "px" });
      for (let i = 0; i < slots; i++) px.append(h("i", { class: i < r.cams ? "on" : "" }));
      host.append(h("div", { class: "cam" },
        h("div", { class: "nm" }, r.set, h("small", null, r.use)),
        h("div", null, px, h("div", { class: "src-in", style: "margin-top:6px" },
          fmt(r.fps) + " / " + C.sample_fps + " = " + round(r.fps / C.sample_fps, 1))),
        h("div", { class: "cn" }, String(r.cams), h("small", null, r.cams < 5 ? "камеры" : "камер"))));
    });
  }

  /* ---- окно подтверждения по кадрам (слайд 6) ---- */
  function frames() {
    const svg = document.getElementById("frames");
    if (!svg) return;
    const lit = new Set([3, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 26]);
    const n = 34, pitch = 44, sz = 36, x0 = 236, y = 64;
    s("text", { x: 0, y: y + 14, "font-size": 20, fill: "#b9cbd2" }, svg, "Кадры анализа,");
    s("text", { x: 0, y: y + 38, "font-size": 20, fill: "#b9cbd2" }, svg, "5 к/с · схема");
    for (let i = 0; i < n; i++) {
      const on = lit.has(i);
      s("rect", { x: x0 + i * pitch, y: y, width: sz, height: sz, fill: on ? "#a8d42c" : "none",
        stroke: on ? "#a8d42c" : "rgba(132,180,200,.35)", "stroke-width": 1.5, opacity: (i > 17 && on) ? 0.5 : 1 }, svg);
    }
    const cx3 = x0 + 3 * pitch + sz / 2;
    s("text", { x: cx3, y: y + sz + 30, "text-anchor": "middle", "font-size": 19, fill: "#b9cbd2" }, svg, "одиночный кадр —");
    s("text", { x: cx3, y: y + sz + 54, "text-anchor": "middle", "font-size": 19, fill: "#b9cbd2" }, svg, "не тревога");
    const wx = x0 + 8 * pitch - 4, ww = 10 * pitch;
    s("path", { d: "M" + wx + "," + (y - 8) + " v-12 h" + ww + " v12", fill: "none", stroke: "#eef4f1", "stroke-width": 2 }, svg);
    s("text", { x: wx + ww / 2, y: y - 30, "text-anchor": "middle", "font-size": 20, fill: "#eef4f1" }, svg, "окно 2 с: нарушение на 9 из 10 кадров");
    const ex = x0 + 17 * pitch + sz + 4;
    s("rect", { x: ex - 2, y: y - 6, width: 4, height: sz + 12, fill: "#ff5d52" }, svg);
    s("text", { x: ex - 8, y: y + sz + 30, "text-anchor": "end", "font-size": 20, "font-weight": 650, fill: "#eef4f1" }, svg, "одно событие → оператору");
    const rx = x0 + 18 * pitch, rw = 9 * pitch - 8;
    s("path", { d: "M" + rx + "," + (y - 8) + " v-12 h" + rw + " v12", fill: "none", stroke: "#7f9cab", "stroke-width": 2, "stroke-dasharray": "5 4" }, svg);
    s("text", { x: rx + rw / 2, y: y - 30, "text-anchor": "middle", "font-size": 20, fill: "#b9cbd2" }, svg, "тот же трек — повторов нет");
    const cx = x0 + 30.5 * pitch;
    s("text", { x: cx, y: y - 30, "text-anchor": "middle", "font-size": 20, fill: "#b9cbd2" }, svg, "нарушения нет —");
    s("text", { x: cx, y: y - 6, "text-anchor": "middle", "font-size": 20, fill: "#b9cbd2" }, svg, "событие закрыто");
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
      bind(data); ptags(data); statusLines(data); evalTiles(data); latency(data);
      tpuChart(data); winChart(data); cams(data); scen(data);
    } catch (e) {
      fail("ошибка сборки: " + e.message);
    }
    await document.fonts.ready;
    document.body.dataset.ready = "1";
  }
  main();
})();
