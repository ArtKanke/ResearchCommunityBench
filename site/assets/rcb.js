// ResearchCommunityBench: data loading and charts shared by the site and the deck.
// Plain SVG and HTML, no libraries. Colors come from CSS custom properties so the page theme owns them.

export const DOMAIN_ORDER = ["materials", "compbio", "aiml"];
export const DOMAIN_ORDER_UI = ["aiml", "compbio", "materials"];
export const DOMAIN_SHORT = { materials: "Materials science", compbio: "Protein biology", aiml: "Machine learning" };
export const DOMAIN_ABBR = { materials: "materials", compbio: "protein", aiml: "ML" };
export const MODEL_ORDER = ["gpt-6-1-sol", "gpt-6-astra", "claude-opus-5-5", "gpt-6-luna", "claude-haiku-5-5"];
export const LETTER = { "gpt-6-1-sol": "S", "gpt-6-astra": "A", "claude-opus-5-5": "O", "gpt-6-luna": "L", "claude-haiku-5-5": "H" };

const NS = "http://www.w3.org/2000/svg";
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
export const short = (name) => name.replace(/^(GPT-6\.1 |GPT-6 |Claude )/, "").replace(/ 5\.5$/, "");
export const modelColor = (key) => css(`--m-${key}`) || css("--ink");
export const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
export const mtile = (m) => `<span class="sw" style="background:${modelColor(m.key)}"></span>`;

// ---------- data ----------

export async function loadResults(base = "") {
  const doc = await (await fetch(base + "data/results.json", { cache: "no-store" })).json();
  doc.modelsByKey = Object.fromEntries(doc.models.map((m) => [m.key, m]));
  doc.models.sort((a, b) => MODEL_ORDER.indexOf(a.key) - MODEL_ORDER.indexOf(b.key));
  return doc;
}

export const pickDataset = (doc, id) => doc.datasets.find((d) => d.id === id) || doc.datasets[0];
export const row = (ds, model, domain, mode) => ds.rows.find((r) => r.model === model && r.domain === domain && r.mode === mode);
export const val = (r, k) => (r && r.m[k] ? r.m[k].mean : null);
export const se = (r, k) => (r && r.m[k] ? r.m[k].se : null);
export const modelsIn = (doc, ds) => doc.models.filter((m) => ds.rows.some((r) => r.model === m.key));
export const cell = (ds, model, domain, mode) => (domain === "all" ? pooled(ds, model, mode) : row(ds, model, domain, mode));

// Pool a model's rows across domains: sum of per-domain means, SE combined in quadrature.
export function pooled(ds, model, mode) {
  const rs = DOMAIN_ORDER.map((d) => row(ds, model, d, mode)).filter(Boolean);
  if (!rs.length) return null;
  const m = {};
  for (const k of Object.keys(rs[0].m)) {
    const parts = rs.map((r) => r.m[k]).filter(Boolean);
    const mean = parts.reduce((s, p) => s + p.mean, 0);
    const v = parts.every((p) => p.se != null) ? Math.sqrt(parts.reduce((s, p) => s + p.se * p.se, 0)) : null;
    m[k] = { mean, se: v };
  }
  return { model, domain: "all", mode, n: Math.min(...rs.map((r) => r.n)), m, domains: rs.length };
}

// Ratio a/b with a delta-method SE on the log scale.
export function ratio(a, b, k) {
  const A = val(a, k), B = val(b, k);
  if (A == null || B == null || B <= 0) return { v: null, A, B };
  const v = A / B;
  const ra = A > 0 && se(a, k) != null ? se(a, k) / A : null;
  const rb = se(b, k) != null ? se(b, k) / B : null;
  const sl = ra != null && rb != null ? Math.sqrt(ra * ra + rb * rb) : null;
  return { v, A, B, lo: sl != null && v > 0 ? v / Math.exp(sl) : null, hi: sl != null && v > 0 ? v * Math.exp(sl) : null };
}

// ---------- formatting ----------

export const fx = (v) => (v == null ? "–" : v >= 10 ? v.toFixed(0) + "×" : v >= 1 ? v.toFixed(1) + "×" : v.toFixed(2) + "×");
export const fusd = (v) => (v == null ? "–" : v >= 100 ? "$" + v.toFixed(0) : v >= 10 ? "$" + v.toFixed(1) : v >= 0.1 ? "$" + v.toFixed(2) : "$" + v.toFixed(3));
export const fnum = (v, d = 1) => (v == null ? "–" : Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(d));
const usdCell = (v) => fusd(v);
const pmCell = (r, k, d = 1) => (r && r.m[k] ? fnum(r.m[k].mean, r.m[k].se == null && Number.isInteger(r.m[k].mean) ? 0 : d) + (r.m[k].se != null ? ` <span class="se">±${fnum(r.m[k].se, d)}</span>` : "") : "–");
export const pm = pmCell;

// ---------- svg helpers ----------

export function el(tag, attrs = {}, parent) {
  const n = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) if (v != null) n.setAttribute(k, v);
  if (parent) parent.appendChild(n);
  return n;
}
function text(parent, x, y, s, attrs = {}) { const t = el("text", { x, y, ...attrs }, parent); t.textContent = s; return t; }
export function frame(host, w, h, label) {
  host.innerHTML = "";
  if (tipEl) tipEl.style.opacity = 0;  // a redraw removes the hovered mark without a pointerleave
  return el("svg", { viewBox: `0 0 ${w} ${h}`, role: "img", "aria-label": label, class: "chart" }, host);
}
const logScale = (d0, d1, r0, r1) => { const a = Math.log(d0), b = Math.log(d1); return (v) => r0 + ((Math.log(v) - a) / (b - a)) * (r1 - r0); };
const linScale = (d0, d1, r0, r1) => (v) => r0 + ((v - d0) / (d1 - d0)) * (r1 - r0);
const pow2Range = (vals, pad = 1.6) => {
  const xs = vals.filter((v) => v > 0);
  return [Math.pow(2, Math.floor(Math.log2(Math.min(...xs) / pad))), Math.pow(2, Math.ceil(Math.log2(Math.max(...xs) * pad)))];
};
const ratioTick = (v) => (v >= 1 ? v + "×" : "1/" + Math.round(1 / v) + "×");

// A model marker: a small rounded tile with the model's letter in its colour.
// one marker shape per field: circle materials, square protein biology, triangle machine learning
export function domainMark(parent, dom, x, y, col, hollow) {
  const g = el("g", { class: "tile", transform: `translate(${x},${y})` }, parent);
  const st = `fill:${hollow ? "var(--bg)" : col};stroke:${hollow ? col : "var(--bg)"};stroke-width:${hollow ? 1.6 : 1.5}`;
  if (dom === "compbio") el("rect", { x: -6, y: -6, width: 12, height: 12, rx: 1.5, style: st }, g);
  else if (dom === "aiml") el("path", { d: "M0,-7.5 L7,5.5 L-7,5.5 Z", style: st }, g);
  else el("circle", { r: 6.5, style: st }, g);
  return g;
}

export function tileMark(parent, mod, x, y, opts = {}) {
  // a plain dot in the model's colour; hollow for solo runs
  const s = opts.size || 20, col = modelColor(mod.key);
  const g = el("g", { class: "tile", transform: `translate(${x},${y})` }, parent);
  el("circle", { r: Math.max(4, s / 3.4), style: `fill:${opts.hollow ? "var(--bg)" : col};stroke:${col};stroke-width:1.6` }, g);
  return g;
}

// ---------- tooltip ----------

let tipEl;
export function tip(evt, html) {
  if (!tipEl) { tipEl = document.createElement("div"); tipEl.className = "tip"; document.body.appendChild(tipEl); }
  if (!html) { tipEl.style.opacity = 0; return; }
  tipEl.innerHTML = html;
  tipEl.style.opacity = 1;
  const pad = 14, r = tipEl.getBoundingClientRect();
  let x = evt.clientX + pad, y = evt.clientY + pad;
  if (x + r.width > innerWidth - 8) x = evt.clientX - r.width - pad;
  if (y + r.height > innerHeight - 8) y = evt.clientY - r.height - pad;
  tipEl.style.transform = `translate(${x}px, ${y}px)`;
}
function hover(node, html) {
  node.addEventListener("pointermove", (e) => tip(e, html));
  node.addEventListener("pointerleave", () => tip(null, null));
}

// ---------- 1. headline: team / solo output against team / solo cost ----------

export function headline(host, doc, ds, opts = {}) {
  const domain = opts.domain || "all", key = opts.metric || "R";
  const W = opts.w || 1280, H = opts.h || 520, m = { l: 64, r: 120, t: 24, b: 50 };
  const pts = [];
  const doms = opts.byDomain ? (domain === "all" ? DOMAIN_ORDER : [domain]) : [domain];
  for (const mod of modelsIn(doc, ds)) for (const d of doms) {
    const t = cell(ds, mod.key, d, "team"), s = cell(ds, mod.key, d, "solo");
    if (!t || !s) continue;
    pts.push({ model: mod, dom: d, out: ratio(t, s, key), cost: ratio(t, s, "cost"), t, s });
  }
  const ok = pts.filter((p) => p.out.v > 0 && p.cost.v > 0);
  const zero = pts.filter((p) => p.out.v === 0 && p.cost.v > 0);
  const undef = pts.filter((p) => p.out.v == null);
  const bars = opts.errorBars !== false;
  const xr = pow2Range(pts.map((p) => p.cost.v).filter((v) => v > 0).concat([1, ...ok.map((p) => (bars && p.cost.hi) || 1)]), 1.3);
  const yr = pow2Range(ok.map((p) => p.out.v).concat([1, ...ok.map((p) => (bars && p.out.hi) || 1), ...ok.map((p) => (bars && p.out.lo) || 1)]), 1.15);
  if (zero.length) yr[0] = Math.min(yr[0], 1 / 8);
  xr[0] = Math.min(xr[0], 1 / 2); yr[1] = Math.max(yr[1], 4);
  const X = logScale(xr[0], xr[1], m.l, W - m.r), Y = logScale(yr[0], yr[1], H - m.b, m.t);
  const svg = frame(host, W, H, "Team over solo verified output against team over solo cost, one tile per model");

  // zones: above both lines the community wins outright; below no-gain it loses
  const d1 = Math.min(xr[1], yr[1]);
  el("path", { d: `M${X(xr[0])},${Y(yr[1])} L${X(xr[0])},${Y(1)} L${X(Math.max(1, xr[0]))},${Y(1)} L${X(d1)},${Y(d1)} L${X(d1)},${Y(yr[1])} Z`, class: "zone-good" }, svg);
  el("rect", { x: m.l, y: Y(1), width: W - m.r - m.l, height: Y(yr[0]) - Y(1), class: "zone-bad" }, svg);
  text(svg, m.l + 10, m.t + 16, "MORE SCIENCE PER DOLLAR THAN SOLO", { class: "zone-label" });
  text(svg, W - m.r - 10, Y(yr[0]) - 10, "LESS SCIENCE THAN SOLO", { class: "zone-label bad", "text-anchor": "end" });

  const g = el("g", {}, svg);
  for (let v = yr[0]; v <= yr[1] + 1e-9; v *= 2) {
    el("line", { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v), class: "grid" }, g);
    text(g, m.l - 10, Y(v) + 4, v === yr[0] && zero.length ? "0" : ratioTick(v), { class: "tick", "text-anchor": "end" });
  }
  for (let v = xr[0]; v <= xr[1] + 1e-9; v *= 2) {
    el("line", { x1: X(v), x2: X(v), y1: m.t, y2: H - m.b, class: "grid" }, g);
    text(g, X(v), H - m.b + 18, ratioTick(v), { class: "tick", "text-anchor": "middle" });
  }
  el("line", { x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b, class: "axisline" }, g);
  text(g, (m.l + W - m.r) / 2, H - 10, "Cost, team ÷ solo", { class: "axis-title", "text-anchor": "middle" });
  text(g, 16, (m.t + H - m.b) / 2, key === "best_gain" ? "Best gain, team ÷ solo" : "Verified claims, team ÷ solo", { class: "axis-title", "text-anchor": "middle", transform: `rotate(-90 16 ${(m.t + H - m.b) / 2})` });

  const ref = el("g", { class: "ref" }, svg);
  el("line", { x1: m.l, x2: W - m.r, y1: Y(1), y2: Y(1), class: "ref-line" }, ref);
  text(ref, W - m.r + 8, Y(1) + 3.5, "NO GAIN", { class: "ref-label" });
  const d0 = Math.max(xr[0], yr[0]);
  el("line", { x1: X(d0), y1: Y(d0), x2: X(d1), y2: Y(d1), class: "ref-line diag" }, ref);
  text(ref, X(d1) + 8, Y(d1) + 3.5, "BREAK-EVEN", { class: "ref-label diag" });
  text(ref, X(d1) + 8, Y(d1) + 15, "PER DOLLAR", { class: "ref-label diag" });

  const P = el("g", { class: "marks" }, svg);
  const draw = (p, yv) => {
    const x = X(p.cost.v), y = Y(yv), col = modelColor(p.model.key);
    if (bars && p.out.v > 0 && p.out.lo && p.out.hi) el("line", { x1: x, x2: x, y1: Y(Math.max(p.out.lo, yr[0])), y2: Y(Math.min(p.out.hi, yr[1])), stroke: col, class: "whisker" }, P);
    if (bars && p.cost.lo && p.cost.hi) el("line", { x1: X(Math.max(p.cost.lo, xr[0])), x2: X(Math.min(p.cost.hi, xr[1])), y1: y, y2: y, stroke: col, class: "whisker" }, P);
    const ts = opts.tile || 20;
    const node = opts.byDomain ? domainMark(P, p.dom, x, y, modelColor(p.model.key), p.out.v === 0) : tileMark(P, p.model, x, y, { hollow: p.out.v === 0, size: ts });
    labels.push({ x, y, ts: opts.byDomain ? 14 : ts, name: opts.byDomain ? `${short(p.model.name)} · ${DOMAIN_ABBR[p.dom]}` : p.model.name });
    const dn = p.dom === "all" ? "all fields" : DOMAIN_SHORT[p.dom];
    const fmtA = (v) => (key === "best_gain" ? (v * 100).toFixed(1) + "%" : fnum(v));
    hover(node, `<b>${esc(p.model.name)}</b> <span class="muted">· ${dn}</span><br>${key === "best_gain" ? "Best gain" : "Verified claims"} <span class="n">${fx(p.out.v)}</span> <span class="muted">(${fmtA(p.out.A)} team vs ${fmtA(p.out.B)} solo)</span><br>Cost <span class="n">${fx(p.cost.v)}</span> <span class="muted">(${fusd(p.cost.A)} vs ${fusd(p.cost.B)})</span><br><span class="muted">n = ${p.t.n} team, ${p.s.n} solo per domain</span>`);
  };
  const labels = [];
  ok.forEach((p) => draw(p, p.out.v));
  zero.forEach((p) => draw(p, yr[0]));
  // labels: right of the tile, else left, else above; avoid other tiles and labels placed so far
  const px = opts.labelPx || 11, boxes = labels.map((l) => [l.x - l.ts / 2, l.y - l.ts / 2, l.x + l.ts / 2, l.y + l.ts / 2]);
  const hit = (b) => boxes.some((o) => b[0] < o[2] && b[2] > o[0] && b[1] < o[3] && b[3] > o[1]);
  for (const l of labels) {
    const w = l.name.length * px * 0.6, h = px * 1.2, g = l.ts * 0.75;
    const tries = [[l.x + g, l.y, "start"], [l.x - g, l.y, "end"], [l.x, l.y - l.ts * 0.9, "middle"], [l.x, l.y + l.ts * 1.3, "middle"]];
    let pick = tries[0];
    for (const t of tries) {
      const x0 = t[2] === "start" ? t[0] : t[2] === "end" ? t[0] - w : t[0] - w / 2;
      const b = [x0, t[1] - h / 2, x0 + w, t[1] + h / 2];
      if (!hit(b)) { pick = t; boxes.push(b); break; }
    }
    text(P, pick[0], pick[1] + px * 0.35, l.name, { class: "pt-label", "text-anchor": pick[2] });
  }
  return { points: pts, undefined: undef, zero };
}

// ---------- 2. cost against verified output, solo -> team ----------

export function costScatter(host, doc, ds, opts = {}) {
  const W = opts.w || 1280, H = opts.h || 460, m = { l: 64, r: 120, t: 20, b: 50 };
  const domains = opts.domain && opts.domain !== "all" ? [opts.domain] : DOMAIN_ORDER;
  const pairs = [];
  for (const mod of modelsIn(doc, ds)) for (const d of domains) {
    const t = row(ds, mod.key, d, "team"), s = row(ds, mod.key, d, "solo");
    if (t || s) pairs.push({ model: mod, domain: d, t, s });
  }
  const costs = pairs.flatMap((p) => [val(p.t, "cost"), val(p.s, "cost")]).filter((v) => v > 0);
  const outs = pairs.flatMap((p) => [val(p.t, "R") + (se(p.t, "R") || 0), val(p.s, "R") + (se(p.s, "R") || 0)]).filter((v) => v != null);
  const c0 = Math.pow(10, Math.floor(Math.log10(Math.min(...costs)))), c1 = Math.pow(10, Math.ceil(Math.log10(Math.max(...costs))));
  const ymax = Math.max(5, Math.ceil(Math.max(...outs) / 5) * 5);
  const X = logScale(c0, c1, m.l, W - m.r), Y = linScale(0, ymax, H - m.b, m.t);
  const svg = frame(host, W, H, "Cost per run against verified claims, solo and team");
  const g = el("g", {}, svg);
  for (let v = c0; v <= c1 * 1.0001; v *= 10) {
    el("line", { x1: X(v), x2: X(v), y1: m.t, y2: H - m.b, class: "grid" }, g);
    text(g, X(v), H - m.b + 18, "$" + (v < 1 ? v.toFixed(Math.round(-Math.log10(v))) : v.toFixed(0)), { class: "tick", "text-anchor": "middle" });
  }
  const step = ymax > 20 ? 10 : 5;
  for (let v = 0; v <= ymax; v += step) {
    el("line", { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v), class: v ? "grid" : "axisline" }, g);
    text(g, m.l - 10, Y(v) + 4, v, { class: "tick", "text-anchor": "end" });
  }
  text(g, (m.l + W - m.r) / 2, H - 10, "Cost per run, log scale", { class: "axis-title", "text-anchor": "middle" });
  text(g, 16, (m.t + H - m.b) / 2, "Verified claims per run", { class: "axis-title", "text-anchor": "middle", transform: `rotate(-90 16 ${(m.t + H - m.b) / 2})` });
  const P = el("g", {}, svg);
  const at = (r) => (r && val(r, "cost") > 0 ? [X(val(r, "cost")), Y(val(r, "R"))] : null);
  for (const p of pairs) {
    const a = at(p.s), b = at(p.t), col = modelColor(p.model.key);
    if (a && b) el("line", { x1: a[0], y1: a[1], x2: b[0], y2: b[1], stroke: col, class: "pair-link" }, P);
  }
  for (const p of pairs) {
    const col = modelColor(p.model.key);
    for (const [r, xy, solo] of [[p.s, at(p.s), true], [p.t, at(p.t), false]]) {
      if (!xy) continue;
      if (se(r, "R")) el("line", { x1: xy[0], x2: xy[0], y1: Y(Math.max(0, val(r, "R") - se(r, "R"))), y2: Y(val(r, "R") + se(r, "R")), stroke: col, class: "whisker" }, P);
      const node = tileMark(P, p.model, xy[0], xy[1], { hollow: solo, fill: !solo, size: 16 });
      hover(node, `<b>${esc(p.model.name)}</b> <span class="muted">· ${DOMAIN_SHORT[p.domain]} · ${solo ? "solo" : "team"}</span><br><span class="n">${fnum(val(r, "R"))}</span> verified claims${se(r, "R") != null ? ` <span class="muted">±${fnum(se(r, "R"))}</span>` : ""}<br><span class="n">${fusd(val(r, "cost"))}</span> per run · <span class="n">${fnum(val(r, "agents"))}</span> researchers<br><span class="muted">n = ${r.n}</span>`);
    }
  }
}

// ---------- 3. leaderboard ----------

export const LB_COLS = [
  { k: "rank", label: "" },
  { k: "model", label: "Model" },
  { k: "bestTeam", label: "Best gain, team", sort: true, hint: "How far the team's best verified result lowers held-out error below the reference model, averaged over each field's datasets, then over fields" },
  { k: "bestSolo", label: "Best gain, solo", sort: true, hint: "The same for the model working alone (single-agent baseline)" },
  { k: "gainBest", label: "Team ÷ solo", sort: true, hint: "Best gain of the team divided by best gain of the model alone; above 1× the team did better" },
  { k: "teamR", label: "Verified, team", sort: true, hint: "Verified claims of the team runs, summed over the fields: the experiment re-runs within 10% and the claim's test beats its comparator on the held-out test set" },
  { k: "soloR", label: "Verified, solo", sort: true, hint: "Verified claims per solo run" },
  { k: "gain", label: "Verified ×", sort: true, hint: "Community ÷ solo verified claims" },
  { k: "frontier", label: "New bests", sort: true, hint: "Verified claims that set a new best held-out result on their dataset, per community run" },
  { k: "bestGain", label: "Gain", sort: true, hint: "Best relative loss reduction over the baseline, averaged over the domain's datasets, community runs" },
  { k: "costx", label: "Cost ×", sort: true, hint: "Team ÷ solo cost per run" },
  { k: "C", label: "Collaborative", sort: true, hint: "Verified claims that depend on a teammate's: a correction whose target fails verification while the correction passes, or an extension that gets materially worse when the teammate's data it used is shuffled" },
  { k: "teamCost", label: "Cost, team", sort: true, hint: "Model spend of the team runs, summed over the fields, US dollars" },
  { k: "agents", label: "Researchers", sort: true, hint: "Researchers recruited per community run" },
  { k: "time", label: "Time used", sort: true, hint: "Share of the session in use, until the last researcher stopped" },
];

export function leaderboardRows(doc, ds, domain) {
  const rows = [];
  for (const mod of modelsIn(doc, ds)) {
    const t = cell(ds, mod.key, domain, "team"), s = cell(ds, mod.key, domain, "solo");
    if (!t && !s) continue;
    rows.push({
      model: mod, t, s, teamR: val(t, "R"), soloR: val(s, "R"), gain: ratio(t, s, "R").v, costx: ratio(t, s, "cost").v, C: val(t, "C"),
      acc: val(t, "submitted") ? val(t, "R") / val(t, "submitted") : null,
      cpc: val(t, "R") ? val(t, "cost") / val(t, "R") : null, teamCost: val(t, "cost"),
      agents: perRun(t, "agents"), time: perRun(t, "session_share"), frontier: val(t, "frontier"), bestGain: perRun(t, "best_gain"),
      bestTeam: perRun(t, "best_gain"), bestSolo: perRun(s, "best_gain"),
      gainBest: perRun(t, "best_gain") != null && perRun(s, "best_gain") > 0 ? perRun(t, "best_gain") / perRun(s, "best_gain") : null,
    });
  }
  return rows;
}

export function leaderboard(host, doc, ds, domain = "all", sortKey = "teamR", desc = true, only = null) {
  const COLS = only ? LB_COLS.filter((c) => only.includes(c.k)) : LB_COLS;
  const rows = leaderboardRows(doc, ds, domain).sort((a, b) => ((a[sortKey] ?? -1) - (b[sortKey] ?? -1)) * (desc ? -1 : 1));
  const n = rows.length;
  const cellHtml = (r, k) => {
    switch (k) {
      case "rank": return r.rank;
      case "model": return `<div class="who">${mtile(r.model)}${esc(r.model.name)}</div>`;
      case "teamR": return pmCell(r.t, "R");
      case "soloR": return pmCell(r.s, "R");
      case "gain": return fx(r.gain);
      case "costx": return fx(r.costx);
      case "C": return pmCell(r.t, "C");
      case "acc": return r.acc == null ? "–" : `${(r.acc * 100).toFixed(0)}<span class="u">%</span>`;
      case "frontier": return r.frontier == null ? "–" : pmCell(r.t, "frontier");
      case "bestTeam": return r.bestTeam == null ? "–" : `${(r.bestTeam * 100).toFixed(1)}<span class="u">%</span>`;
      case "bestSolo": return r.bestSolo == null ? "–" : `${(r.bestSolo * 100).toFixed(1)}<span class="u">%</span>`;
      case "gainBest": return fx(r.gainBest);
      case "bestGain": return r.bestGain == null ? "–" : `${(r.bestGain * 100).toFixed(1)}<span class="u">%</span>`;
      case "agents": return fnum(r.agents);
      case "time": return r.time == null ? "–" : `${Math.round(r.time * 100)}<span class="u">%</span>`;
      case "cpc": return usdCell(r.cpc);
      case "teamCost": return usdCell(r.teamCost);
    }
  };
  rows.forEach((r, i) => (r.rank = i + 1));
  host.innerHTML = `<table class="lb"><thead><tr>${COLS.map((c) => `<th class="${c.sort ? "sortable" : ""}${c.k === sortKey ? " on" : ""}" data-k="${c.k}" title="${esc(c.hint || "")}">${c.label}${c.sort ? `<span class="ar">${c.k === sortKey ? (desc ? "↓" : "↑") : "↕"}</span>` : ""}</th>`).join("")}</tr></thead>
  <tbody>${rows.map((r, i) => { const a = 18 * (1 - i / Math.max(1, n)); return `<tr>${COLS.map((c) => `<td class="c-${c.k}" style="background:color-mix(in srgb, var(--theme-600) ${a.toFixed(1)}%, transparent)">${cellHtml(r, c.k)}</td>`).join("")}</tr>`; }).join("")}</tbody></table>`;
  host.querySelectorAll("th.sortable").forEach((th) => th.addEventListener("click", () => {
    const k = th.dataset.k; leaderboard(host, doc, ds, domain, k, k === sortKey ? !desc : true, only);
  }));
  return rows;
}

// ---------- 4. five failure modes, as a heatmap ----------

const median = (xs) => { if (!xs.length) return null; const a = xs.slice().sort((p, q) => p - q), k = Math.floor(a.length / 2); return a.length % 2 ? a[k] : (a[k - 1] + a[k]) / 2; };
// a pooled row sums per-domain means; divide by the domain count to get a per-run figure
const perRun = (t, k) => (val(t, k) == null ? null : val(t, k) / (t.domains || 1));

export const FAILURE_MODES = [
  { id: "overhead", name: "Coordination overhead", def: "The community pays for organizing, not for science.",
    metric: "Cost per verified claim, team ÷ solo", better: "low",
    f: (t, s) => (val(t, "R") > 0 && val(s, "R") > 0 ? (val(t, "cost") / val(t, "R")) / (val(s, "cost") / val(s, "R")) : null), fmt: fx },
  { id: "transfer", name: "Knowledge transfer", def: "A correction of a teammate's claim that then fails verification, or an extension whose gain disappears when the teammate's input is removed.",
    metric: "Confirmed collaborative claims per run", better: "high", f: (t) => perRun(t, "C"), fmt: (v) => fnum(v) },
  { id: "propagation", name: "Invalid propagation", def: "Work built on a result that later failed verification. A few runs dominate the mean, so this is the median.",
    metric: "Built on failed work, median per run", better: "low", f: (t, s, runs) => median(runs.map((r) => r.built_on_failed).filter((v) => v != null)), fmt: (v) => fnum(v) },
  { id: "allocation", name: "Poor allocation", def: "Researchers recruited who add no verified output.",
    metric: "Verified claims per researcher", better: "high", f: (t) => (val(t, "agents") ? val(t, "R") / val(t, "agents") : null), fmt: (v) => fnum(v, 2) },
  { id: "review", name: "Ineffective review", def: "Reviewers endorse claims that do not hold up.",
    metric: "Endorsed failed work per run", better: "low", f: (t) => perRun(t, "replicated_failed"), fmt: (v) => fnum(v) },
];

export function failureHeatmap(host, doc, ds, domain = "all") {
  const models = modelsIn(doc, ds);
  const runsOf = (mk) => (ds.runs || []).filter((r) => r.model === mk && r.mode === "team" && (domain === "all" || r.domain === domain));
  const vals = FAILURE_MODES.map((fm) => models.map((mod) => fm.f(cell(ds, mod.key, domain, "team"), cell(ds, mod.key, domain, "solo"), runsOf(mod.key))));
  // severity 0..1 per column: how bad this model is relative to the others on that mode
  const sev = vals.map((col, j) => {
    const xs = col.filter((v) => v != null), lo = Math.min(...xs), hi = Math.max(...xs);
    return col.map((v) => (v == null || hi === lo ? 0 : FAILURE_MODES[j].better === "low" ? (v - lo) / (hi - lo) : (hi - v) / (hi - lo)));
  });
  const runs = (mod) => (domain === "all" ? DOMAIN_ORDER : [domain]).reduce((s, d) => s + (row(ds, mod.key, d, "team")?.n || 0), 0);
  host.innerHTML = `<table class="heat"><thead><tr><th class="rowh"></th>${FAILURE_MODES.map((fm, j) => `<th><span class="n">${String(j + 1).padStart(2, "0")}</span><span class="h">${fm.name}</span><span class="m">${fm.metric}<br><i>${fm.better === "low" ? "lower is better" : "higher is better"}</i></span></th>`).join("")}</tr></thead>
  <tbody>${models.map((mod, i) => `<tr><td class="rowh"><div class="who">${mtile(mod)}${esc(mod.name)}</div></td>${FAILURE_MODES.map((fm, j) => {
    const v = vals[j][i], s = sev[j][i];
    return `<td style="background:color-mix(in srgb, var(--rejected) ${(5 + s * 30).toFixed(0)}%, var(--bg))" title="${esc(fm.def)}">${v == null ? "–" : fm.fmt(v)}</td>`;
  }).join("")}</tr>`).join("")}</tbody></table>`;
}

// ---------- 5. organization profile ----------

export const ORG_COLS = [
  { k: "agents", label: "Researchers recruited", fmt: (v) => fnum(v) },
  { k: "messages", label: "Board messages", fmt: (v) => fnum(v, 0) },
  { k: "calls", label: "Model calls", fmt: (v) => fnum(v, 0) },
  { k: "session_share", label: "Share of session used", fmt: (v) => Math.round(v * 100) + "%", max: 1 },
];

export function orgProfile(host, doc, ds, domain = "all") {
  const models = modelsIn(doc, ds);
  const get = (mod, k) => {
    const rs = (domain === "all" ? DOMAIN_ORDER : [domain]).map((d) => row(ds, mod.key, d, "team")).filter(Boolean);
    const xs = rs.map((r) => val(r, k)).filter((v) => v != null);
    return xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : null;
  };
  const maxes = Object.fromEntries(ORG_COLS.map((c) => [c.k, c.max || Math.max(1e-9, ...models.map((m) => get(m, c.k) ?? 0))]));
  host.innerHTML = `<table class="org"><thead><tr><th class="rowh">Per community run</th>${ORG_COLS.map((c) => `<th>${c.label}</th>`).join("")}</tr></thead>
  <tbody>${models.map((mod) => `<tr><td><div class="who">${mtile(mod)}${esc(mod.name)}</div></td>${ORG_COLS.map((c) => {
    const v = get(mod, c.k);
    return v == null ? `<td><span class="ph">Pending main run</span></td>`
      : `<td><div class="bar"><span class="tr"><span class="fl" style="width:${Math.max(1, (v / maxes[c.k]) * 100)}%;background:${modelColor(mod.key)}"></span></span><span class="v">${c.fmt(v)}</span></div></td>`;
  }).join("")}</tr>`).join("")}</tbody></table>`;
  return get;
}

// ---------- 6. timeline (swimlanes from a replay file) ----------

export function timeline(host, R, opts = {}) {
  const agents = R.agents.slice(), order = [];
  const kids = (id) => agents.filter((a) => a.parent === id).sort((a, b) => a.t_born - b.t_born);
  const walk = (a, depth) => { order.push({ a, depth }); kids(a.id).forEach((k) => walk(k, depth + 1)); };
  agents.filter((a) => !a.parent).forEach((a) => walk(a, 0));
  const dur = R.meta.duration || Math.max(...R.events.map((e) => e.t));
  const verifyEnd = dur * 1.14;
  const lane = opts.lane || 22, W = opts.w || 1280, m = { l: 200, r: 16, t: 30, b: 34 };
  const H = m.t + order.length * lane + m.b;
  const X = linScale(0, verifyEnd, m.l, W - m.r);
  const svg = frame(host, W, H, "Timeline of the community, one lane per researcher");
  const g = el("g", {}, svg);
  el("rect", { x: X(dur), y: m.t - 10, width: X(verifyEnd) - X(dur), height: H - m.b - m.t + 10, class: "verify-zone" }, g);
  const tickStep = dur > 7200 ? 3600 : dur > 2400 ? 600 : dur > 900 ? 300 : 60;
  for (let t = 0; t <= dur; t += tickStep) {
    el("line", { x1: X(t), x2: X(t), y1: m.t - 10, y2: H - m.b, class: "grid" }, g);
    text(g, X(t), H - m.b + 18, tickStep >= 3600 ? t / 3600 + " h" : Math.round(t / 60) + " min", { class: "tick", "text-anchor": "middle" });
  }
  text(g, X(0), m.t - 16, "SESSION", { class: "zone-label", style: "fill:var(--ink2)" });
  text(g, X(dur) + 8, m.t - 16, "VERIFICATION", { class: "zone-label" });
  if (R.meta.verdicts === "aggregate-only" || R.meta.verdicts === "pending") ["PER-CLAIM", "VERDICTS", "PENDING"].forEach((s, i) => text(g, (X(dur) + X(verifyEnd)) / 2, m.t + 12 + i * 13, s, { class: "zone-label", "text-anchor": "middle" }));
  const idx = Object.fromEntries(order.map((o, i) => [o.a.id, i]));
  const yOf = (id) => m.t + (idx[id] ?? 0) * lane + lane / 2;
  const L = el("g", {}, svg);
  order.forEach(({ a, depth }, i) => {
    const y = m.t + i * lane + lane / 2;
    text(L, 16 + depth * 12, y + 4, (a.role || a.id).replace(/_/g, " ").slice(0, 26 - depth), { class: "lane-label" });
    el("line", { x1: X(a.t_born || 0), x2: X(a.t_end ?? dur), y1: y, y2: y, class: "lane" }, L);
    if (a.parent && idx[a.parent] != null) el("path", { d: `M${X(a.t_born)},${yOf(a.parent)} V${y}`, class: "spawn-link" }, L);
  });
  const E = el("g", {}, svg);
  for (const e of R.events) {
    if (idx[e.agent] == null) continue;
    const x = X(e.t), y = yOf(e.agent);
    if (e.type === "message") { const n = el("line", { x1: x, x2: x, y1: y - 6, y2: y + 6, class: "ev-msg" }, E); hover(n, `<span class="muted n">${fmtT(e.t)} · message</span><br>${esc((e.text || "").slice(0, 280))}`); }
    else if (e.type === "exec") { const n = el("circle", { cx: x, cy: y, r: 2.4, class: e.ok ? "ev-exec" : "ev-exec fail" }, E); hover(n, `<span class="muted n">${fmtT(e.t)} · experiment</span><br>${esc(e.label || "")}`); }
  }
  for (const a of R.artifacts) {
    if (idx[a.agent] == null) continue;
    const credited = ["finding", "null_result", "method", "hypothesis"].includes(a.kind);
    const y = yOf(a.agent), x = X(a.t), st = credited ? a.status : "note";
    if (credited && (a.status === "accepted" || a.status === "rejected")) {
      const vx = X(dur + (verifyEnd - dur) * (0.2 + 0.6 * ((a.t / dur) % 1)));
      el("line", { x1: x + 4, x2: vx - 3, y1: y, y2: y, class: `verdict-link ${a.status}` }, E);
      el("circle", { cx: vx, cy: y, r: 3.6, class: `ev-verdict ${a.status}` }, E);
    }
    const n = el("rect", { x: x - 4.5, y: y - 4.5, width: 9, height: 9, rx: 2, class: `ev-art ${st}` }, E);
    hover(n, `<span class="muted n">${fmtT(a.t)} · ${esc(a.kind.replace("_", " "))} · ${st}</span><br>${esc(a.title || a.id)}${a.reason ? `<br><span class="muted">${esc(a.reason.slice(0, 200))}</span>` : ""}`);
  }
  el("line", { x1: X(dur), x2: X(dur), y1: m.t - 10, y2: H - m.b, class: "end-line" }, g);
}
const fmtT = (t) => `T+${String(Math.floor(t / 60)).padStart(2, "0")}:${String(Math.floor(t % 60)).padStart(2, "0")}`;

// ---------- 7. discovery flow: claims over time, and the claims they build on ----------

export const FLOW_CLAIMS = new Set(["finding", "null_result", "method", "hypothesis", "preregistration"]);
export const BUILD_RELS = ["extends", "replicates", "corrects", "refutes"];
export const REL_LABEL = { extends: "extends", replicates: "replicates", corrects: "corrects", refutes: "refutes" };

function claimShape(parent, a, x, y, s) {
  const k = a.kind;
  let n;
  if (k === "hypothesis" || k === "preregistration") n = el("circle", { cx: x, cy: y, r: s * 0.8 }, parent);
  else if (k === "method") n = el("path", { d: `M${x},${y - s * 1.2} L${x + s * 1.2},${y} L${x},${y + s * 1.2} L${x - s * 1.2},${y} Z` }, parent);
  else n = el("rect", { x: x - s, y: y - s, width: 2 * s, height: 2 * s, rx: 1.5 }, parent);
  n.setAttribute("class", `fc ${k} ${a.status === "accepted" ? "ok" : a.status === "rejected" ? "bad" : a.status === "superseded" ? "dup" : "pend"}`);
  return n;
}

export function flowView(host, R, opts = {}) {
  const agents = R.agents.slice(), order = [];
  const kids = (id) => agents.filter((a) => a.parent === id).sort((a, b) => a.t_born - b.t_born);
  const walk = (a, d) => { order.push({ a, d }); kids(a.id).forEach((k) => walk(k, d + 1)); };
  agents.filter((a) => !a.parent).forEach((a) => walk(a, 0));
  const dur = R.meta.duration || Math.max(...R.events.map((e) => e.t));
  const byId = Object.fromEntries(R.artifacts.map((a) => [a.id, a]));
  const claims = R.artifacts.filter((a) => FLOW_CLAIMS.has(a.kind)).sort((a, b) => a.t - b.t);
  const edges = [];
  for (const a of claims) for (const r of a.relations || []) {
    const s = byId[r.target];
    if (s && FLOW_CLAIMS.has(s.kind) && BUILD_RELS.includes(r.relation)) edges.push({ s, d: a, rel: r.relation, cross: s.agent !== a.agent, ok: !!r.verified });
  }
  const parents = {}, children = {}, crossOut = {};
  for (const e of edges) {
    (parents[e.d.id] ||= []).push(e.s.id); (children[e.s.id] ||= []).push(e.d.id);
    if (e.cross) crossOut[e.s.id] = (crossOut[e.s.id] || 0) + 1;
  }
  const depthMemo = {};
  const depth = (id, seen = new Set()) => {
    if (depthMemo[id] != null) return depthMemo[id];
    if (seen.has(id)) return 0;
    seen.add(id);
    return (depthMemo[id] = 1 + Math.max(0, ...(parents[id] || []).map((p) => depth(p, seen))));
  };
  claims.forEach((c) => depth(c.id));

  const W = opts.w || 1280, m = { l: opts.ml || 200, r: 28, t: 30, b: 36 };
  const laneH = Math.max(24, Math.min(46, ((opts.h || 460) - m.t - m.b) / order.length));
  const H = m.t + order.length * laneH + m.b;
  const X = linScale(0, dur, m.l, W - m.r);
  const svg = frame(host, W, H, "Claims over the session, with arcs to the earlier claims they build on");
  const idx = Object.fromEntries(order.map((o, i) => [o.a.id, i]));
  const yOf = (id) => m.t + (idx[id] ?? 0) * laneH + laneH / 2;
  const g = el("g", {}, svg);
  const step = dur > 2400 ? 600 : dur > 900 ? 300 : 120;
  for (let t = 0; t <= dur + 1; t += step) {
    el("line", { x1: X(t), x2: X(t), y1: m.t - 8, y2: H - m.b, class: "grid" }, g);
    text(g, X(t), H - m.b + 18, Math.round(t / 60) + " min", { class: "tick", "text-anchor": "middle" });
  }
  // lanes: who, and when they were running experiments
  const execs = {};
  for (const e of R.events) if (e.type === "exec" && idx[e.agent] != null) (execs[e.agent] ||= []).push(e.t);
  const nClaims = {};
  claims.forEach((c) => (nClaims[c.agent] = (nClaims[c.agent] || 0) + 1));
  order.forEach(({ a, d }, i) => {
    const y = m.t + i * laneH + laneH / 2;
    if (i % 2 === 0) el("rect", { x: m.l - 6, y: y - laneH / 2, width: W - m.r - m.l + 6, height: laneH, class: "flow-band" }, g);
    const lbl = (a.role || a.id).replace(/_/g, " ");
    text(g, 14 + d * 12, y + 4, lbl.length > 24 - d ? lbl.slice(0, 23 - d) + "…" : lbl, { class: "lane-label" });
    text(g, m.l - 14, y + 4, nClaims[a.id] || 0, { class: "tick", "text-anchor": "end", style: "fill:var(--muted)" });
    el("line", { x1: X(a.t_born || 0), x2: X(a.t_end ?? dur), y1: y, y2: y, class: "flow-life" }, g);
  });
  const testing = el("g", { class: "flow-exec" }, svg);
  const execEls = [];
  for (const [ag, ts] of Object.entries(execs)) for (const t of ts) execEls.push({ t, el: el("line", { x1: X(t), x2: X(t), y1: yOf(ag) - 3, y2: yOf(ag) + 3 }, testing) });
  // arcs from the earlier claim to the claim that builds on it
  const arcs = el("g", {}, svg);
  for (const e of edges) {
    const x1 = X(e.s.t), y1 = yOf(e.s.agent), x2 = X(e.d.t), y2 = yOf(e.d.agent);
    const d = y1 === y2 ? `M${x1},${y1} Q${(x1 + x2) / 2},${y1 - Math.min(26, 8 + (x2 - x1) * 0.12)} ${x2},${y2}`
      : `M${x1},${y1} C${x1 + (x2 - x1) * 0.55},${y1} ${x2 - (x2 - x1) * 0.55},${y2} ${x2},${y2}`;
    e.el = el("path", { d, class: `fa ${e.rel}${e.cross ? " cross" : ""}${e.ok ? " vok" : ""}` }, arcs);
  }
  const nodes = el("g", {}, svg);
  for (const c of claims) {
    const s = 4 + 1.8 * Math.sqrt(crossOut[c.id] || 0);
    c._el = claimShape(nodes, c, X(c.t), yOf(c.agent), s);
    if (c.frontier) { c._ring = el("circle", { cx: X(c.t), cy: yOf(c.agent), r: s + 4, class: "fring" }, nodes); c._ring.style.pointerEvents = "none"; }
    const who = (R.agents.find((a) => a.id === c.agent) || {}).role || c.agent;
    const verdict = { accepted: "verified", rejected: "rejected", superseded: "duplicate", uncredited: "not credited" }[c.status] || "not yet verified";
    hover(c._el, `<span class="muted n">${fmtT(c.t)} · ${esc(who)} · ${esc(c.kind.replace("_", " "))} · ${verdict}${c.frontier ? " · new best" : ""}</span><br>${esc(c.title || c.id)}<br><span class="muted">builds on ${(parents[c.id] || []).length} · built on by ${(children[c.id] || []).length}</span>`);
    c._el.addEventListener("click", (ev) => { ev.stopPropagation(); select(c.id); });
  }
  svg.addEventListener("click", () => select(null));
  const head = el("line", { x1: m.l, x2: m.l, y1: m.t - 10, y2: H - m.b, class: "flow-head" }, svg);

  let selected = null;
  function lineage(id) {
    const up = new Set(), down = new Set();
    const goUp = (i) => (parents[i] || []).forEach((p) => { if (!up.has(p)) { up.add(p); goUp(p); } });
    const goDown = (i) => (children[i] || []).forEach((c) => { if (!down.has(c)) { down.add(c); goDown(c); } });
    goUp(id); goDown(id);
    return { up, down };
  }
  function select(id) {
    selected = id;
    svg.classList.toggle("dim", !!id);
    const L = id ? lineage(id) : null, on = id ? new Set([id, ...L.up, ...L.down]) : null;
    for (const c of claims) { c._el.classList.toggle("hl", !!on && on.has(c.id)); if (c._ring) c._ring.classList.toggle("hl", !!on && on.has(c.id)); }
    for (const e of edges) e.el.classList.toggle("hl", !!on && on.has(e.s.id) && on.has(e.d.id));
    if (opts.onSelect) opts.onSelect(id ? byId[id] : null, L, { parents, children, agents: R.agents });
  }

  let cur = -1;
  function render(t) {
    if (t === cur) return;
    cur = t;
    for (const c of claims) { c._el.style.display = c.t <= t ? "" : "none"; if (c._ring) c._ring.style.display = c.t <= t ? "" : "none"; }
    for (const e of edges) e.el.style.display = e.d.t <= t ? "" : "none";
    for (const x of execEls) x.el.style.display = x.t <= t ? "" : "none";
    head.setAttribute("x1", X(Math.min(t, dur))); head.setAttribute("x2", X(Math.min(t, dur)));
    head.style.display = t >= dur ? "none" : "";
    const shown = claims.filter((c) => c.t <= t);
    if (opts.onTick) opts.onTick({
      t, dur, claims: shown.length, cross: edges.filter((e) => e.cross && e.d.t <= t).length,
      onColleague: new Set(edges.filter((e) => e.cross && e.d.t <= t).map((e) => e.d.id)).size,
      chain: Math.max(0, ...shown.map((c) => depthMemo[c.id] || 1)),
      active: R.agents.filter((a) => (a.t_born || 0) <= t && (a.t_end == null || a.t_end >= t)).length,
    });
  }
  render(dur);
  return { render, select, dur, claims, edges, totals: { claims: claims.length, cross: edges.filter((e) => e.cross).length, researchers: R.agents.length, chain: Math.max(0, ...claims.map((c) => depthMemo[c.id] || 1)) } };
}

// ---------- 8. productivity over the hour ----------

export function curves(host, flow, doc, ds, domain, metric = "claims", opts = {}) {
  const models = doc.models.filter((m) => flow.runs.some((r) => r.model === m.key));
  const doms = domain === "all" ? DOMAIN_ORDER : [domain];
  const key = metric === "cross" ? "curve_cross" : "curve_claims";
  const mean = (arrs) => arrs[0].map((_, i) => arrs.reduce((s, a) => s + a[i], 0) / arrs.length);
  const series = (mk, mode) => {
    const per = doms.map((d) => flow.runs.filter((r) => r.model === mk && r.domain === d && r.mode === mode && r.verified != null).map((r) => r[key])).filter((a) => a.length);
    if (!per.length) return null;
    const means = per.map(mean);  // mean over runs within a domain, then added up over domains
    return means[0].map((_, i) => means.reduce((s, a) => s + a[i], 0));
  };
  const verified = (mk, mode) => { const r = cell(ds, mk, domain, mode); return r ? val(r, "R") : null; };
  const vseries = (mk, mode) => {
    const groups = doms.map((d) => flow.runs.filter((r) => r.model === mk && r.domain === d && r.mode === mode && r.verified != null));
    if (groups.some((g) => !g.length || g.some((r) => !r.curve_verified))) return null;
    const means = groups.map((g) => mean(g.map((r) => r.curve_verified)));
    return means[0].map((_, i) => means.reduce((s, a) => s + a[i], 0));
  };
  const data = models.map((m) => ({ m, team: series(m.key, "community"), solo: metric === "cross" ? null : series(m.key, "solo"), vt: verified(m.key, "team"), vs: verified(m.key, "solo"),
    vteam: metric === "cross" ? null : vseries(m.key, "community"), vsolo: metric === "cross" ? null : vseries(m.key, "solo") }));
  const ymax = Math.max(5, ...data.flatMap((d) => [...(d.team || []), ...(d.solo || []), d.vt || 0]));
  const nice = Math.pow(10, Math.floor(Math.log10(ymax))), Ymax = Math.ceil(ymax / nice) * nice;
  const cols = opts.cols || 4, PW = opts.pw || 300, PH = opts.ph || 220, gap = opts.gap || 26, W = cols * PW + (cols - 1) * gap;
  const rows = Math.ceil(data.length / cols), H = rows * (PH + 30);
  const svg = frame(host, W, H, "Cumulative claims over the session, team and solo, per model");
  const minutes = flow.minutes || 70, m = { l: 34, r: 66, t: 44, b: 26 };
  data.forEach((d, i) => {
    const ox = (i % cols) * (PW + gap), oy = Math.floor(i / cols) * (PH + 30);
    const X = linScale(0, minutes, ox + m.l, ox + PW - m.r), Y = linScale(0, Ymax, oy + PH - m.b, oy + m.t);
    const g = el("g", {}, svg);
    tileMark(g, d.m, ox + 10, oy + 14, { size: 18 });
    text(g, ox + 26, oy + 22, d.m.name, { class: "pt-label cv-name" });
    const t = cell(ds, d.m.key, domain, "team"), so = cell(ds, d.m.key, domain, "solo");
    const ro = ratio(t, so, "R").v, rc = ratio(t, so, "cost").v;
    if (ro != null && rc != null) text(g, ox + PW - m.r + 56, oy + 22, `${fx(ro)} verified claims · ${fx(rc)} cost`, { class: "cv-ratio", "text-anchor": "end" });
    for (let k = 0; k <= 2; k++) { const v = (Ymax * k) / 2; el("line", { x1: X(0), x2: X(minutes), y1: Y(v), y2: Y(v), class: k ? "grid" : "axisline" }, g); text(g, X(0) - 6, Y(v) + 4, Math.round(v), { class: "tick", "text-anchor": "end" }); }
    const session = flow.session || 60, hours = session > 120;
    for (const t of hours ? [0, 60, 120, 180, 240, 300, 360] : [0, 30, 60]) text(g, X(t), oy + PH - m.b + 16, hours ? t / 60 + " h" : t + " min", { class: "tick", "text-anchor": "middle" });
    el("line", { x1: X(session), x2: X(session), y1: Y(Ymax), y2: Y(0), class: "ref-line" }, g);
    text(g, (X(session) + X(minutes)) / 2, Y(Ymax) + 10, "wrap-up", { class: "tick", "text-anchor": "middle", style: "font-size:9px" });
    const line = (arr, cls) => arr && el("path", { d: arr.map((v, k) => `${k ? "L" : "M"}${X(k).toFixed(1)},${Y(v).toFixed(1)}`).join(""), class: cls, stroke: modelColor(d.m.key) }, g);
    if (d.vteam) el("path", { d: d.vteam.map((v, k) => `${k ? "L" : "M"}${X(k).toFixed(1)},${Y(v).toFixed(1)}`).join("") + `L${X(minutes)},${Y(0)}L${X(0)},${Y(0)}Z`, class: "curve-area", fill: modelColor(d.m.key) }, g);
    line(d.solo, "curve solo"); line(d.team, "curve team");
    if (d.vsolo && opts.vsolo) line(d.vsolo, "curve vsolo");
    if (metric !== "cross") {
      const dot = (v, solo) => { if (v == null) return; const y = Y(v); el("circle", { cx: X(minutes) + 10, cy: y, r: 4.5, class: solo ? "vdot solo" : "vdot", stroke: modelColor(d.m.key), fill: solo ? "var(--bg)" : modelColor(d.m.key) }, g); };
      dot(d.vs, true); dot(d.vt, false);
      if (d.team) text(g, X(minutes) + 18, Y(d.vt ?? 0) + 4, `${fnum(d.vt, 0)} ✓`, { class: "tick" });
    }
    if (d.team) text(g, X(minutes) - 2, Y(d.team[d.team.length - 1]) - 7, fnum(d.team[d.team.length - 1], 0), { class: "tick", "text-anchor": "end" });
    // hover: a vertical guide and the values at that minute
    const guide = el("line", { x1: 0, x2: 0, y1: Y(Ymax), y2: Y(0), class: "cguide", style: "display:none" }, g);
    const dots = [d.team, d.vteam, d.solo].map((arr) => (arr ? el("circle", { r: 3.5, class: "cdot", style: `display:none;fill:${modelColor(d.m.key)}` }, g) : null));
    const hit = el("rect", { x: X(0), y: Y(Ymax), width: X(minutes) - X(0), height: Y(0) - Y(Ymax), fill: "transparent", style: "cursor:crosshair" }, g);
    hit.addEventListener("pointermove", (ev) => {
      const p = svg.createSVGPoint(); p.x = ev.clientX; p.y = ev.clientY;
      const loc = p.matrixTransform(svg.getScreenCTM().inverse());
      const k = Math.max(0, Math.min(minutes, Math.round(((loc.x - X(0)) / (X(minutes) - X(0))) * minutes)));
      guide.setAttribute("x1", X(k)); guide.setAttribute("x2", X(k)); guide.style.display = "";
      [d.team, d.vteam, d.solo].forEach((arr, j) => { if (arr && dots[j]) { dots[j].setAttribute("cx", X(k)); dots[j].setAttribute("cy", Y(arr[k])); dots[j].style.display = ""; } });
      tip(ev, `<b>${esc(d.m.name)}</b> <span class="muted">· ${minutes > 120 ? `${Math.floor(k / 60)} h ${String(k % 60).padStart(2, "0")} min` : `minute ${k}`}</span><br>Team: <span class="n">${d.team ? fnum(d.team[k], 0) : "–"}</span> claims${d.vteam ? `, <span class="n">${fnum(d.vteam[k], 0)}</span> verified` : ""}<br>Solo: <span class="n">${d.solo ? fnum(d.solo[k], 0) : "–"}</span> claims${d.vsolo ? `, <span class="n">${fnum(d.vsolo[k], 0)}</span> verified` : ""}`);
    });
    hit.addEventListener("pointerleave", () => { guide.style.display = "none"; dots.forEach((x) => x && (x.style.display = "none")); tip(null, null); });
  });
  return data;
}

// ---------- 9. the four questions, answered (shared by the site and the deck) ----------

export function answers(doc, ds, flow, domain = "all") {
  const inDomain = (r) => domain === "all" || r.domain === domain;
  const dn = domain === "all" ? "all domains" : DOMAIN_SHORT[domain];
  const comm = flow.runs.filter((r) => r.mode === "community" && r.verified != null && inDomain(r));
  const teamRows = ds.rows.filter((r) => r.mode === "team" && inDomain(r));
  const tot = (k) => teamRows.reduce((s, r) => s + (val(r, k) || 0) * r.n, 0);
  const res = comm.map((r) => r.recruited ?? r.researchers - 1).sort((a, b) => a - b);
  const roles = [...new Set(comm.flatMap((r) => r.roles || []))].filter((r) => /audit|critic|review|replic|stack|extend/i.test(r)).slice(0, 4);
  const lb = leaderboardRows(doc, ds, domain);
  const gm = (xs) => Math.exp(xs.reduce((s, x) => s + Math.log(x), 0) / xs.length);
  const g = gm(lb.filter((r) => r.gain > 0).map((r) => r.gain)), c = gm(lb.filter((r) => r.costx > 0).map((r) => r.costx));
  return { dn, comm, tot, lb, items: [
    { q: "Can models organize a community?", v: `${res[0]}–${res[res.length - 1]}`, c: "researchers recruited per community", href: "#inside" },
    { q: "Does the science hold up?", v: `${Math.round((tot("R") / tot("submitted")) * 100)}%`, c: "of community claims verified", href: "#productivity" },
    { q: "Do they build on each other?", v: `${Math.round(tot("C"))}`, c: `confirmed collaborations in ${comm.length} runs, all corrections or refutations`, href: "#cases" },
    { q: "Is it worth the cost?", v: `${Math.min(...lb.filter((r) => r.gain > 0).map((r) => r.gain)).toFixed(1)}–${fx(Math.max(...lb.map((r) => r.gain || 0)))}`, c: "verified claims vs solo, by model", href: "#gain" },
  ] };
}

// ---------- 10. community map: researchers, their claims, and who built on whom ----------
// Researchers sit on a ring around the founder in the order they were recruited. Each researcher's claims are
// small dots around them: hollow during the session, green or red once the verifier has run. One arc per pair of
// researchers, as thick as the number of claims that built on the other's work: green where the dependence was
// confirmed by ablation, red dashed where it rests on a claim that failed, grey where it was declared only.

export function communityMap(host, R, opts = {}) {
  const W = opts.w || 1000, H = opts.h || 560;
  const agents = R.agents.slice().sort((a, b) => (a.id === "founder" ? -1 : b.id === "founder" ? 1 : (a.t_born || 0) - (b.t_born || 0)));
  const dur = R.meta.duration || Math.max(...R.events.map((e) => e.t));
  const judged = R.meta.verdicts === "per-claim";
  const end = judged ? dur * 1.12 : dur;
  const cx = W / 2, cy = H / 2 + 4, ringX = W * 0.37, ringY = H * 0.36;
  const others = agents.filter((a) => a.id !== "founder");
  const pos = { founder: { x: cx, y: cy } };
  others.forEach((a, i) => { const ang = -Math.PI / 2 + (i * 2 * Math.PI) / Math.max(1, others.length); pos[a.id] = { x: cx + Math.cos(ang) * ringX, y: cy + Math.sin(ang) * ringY, ang }; });
  if (!others.length) pos.founder = { x: cx, y: cy };
  const byId = Object.fromEntries(R.artifacts.map((a) => [a.id, a]));
  const claims = R.artifacts.filter((a) => FLOW_CLAIMS.has(a.kind) && pos[a.agent]).sort((a, b) => a.t - b.t);
  const roleOf = (id) => ((agents.find((a) => a.id === id) || {}).role || id).replace(/_/g, " ");

  // aggregate build-ons between researchers: from the researcher whose claim was built on, to the builder
  const agg = {};
  for (const d of claims) for (const r of d.relations || []) {
    const s = byId[r.target];
    if (!s || !BUILD_RELS.includes(r.relation) || s.agent === d.agent || !pos[s.agent]) continue;
    const k = s.agent + ">" + d.agent;
    const e = (agg[k] ||= { from: s.agent, to: d.agent, items: [] });
    e.items.push({ t: d.t, rel: r.relation, ok: !!r.verified, failedTarget: s.status === "rejected", s, d });
  }
  const edges = Object.values(agg);

  const svg = frame(host, W, H, "Community map: researchers, their claims, and who built on whose work");
  const defs = el("defs", {}, svg);
  for (const [id, col] of [["cm-ar", "var(--ink2)"], ["cm-ar-ok", "var(--accepted)"], ["cm-ar-bad", "var(--rejected)"]]) {
    const mk = el("marker", { id, viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 5, markerHeight: 5, orient: "auto-start-reverse" }, defs);
    el("path", { d: "M0,0 L10,5 L0,10 z", fill: col }, mk);
  }
  const gSpawn = el("g", {}, svg), gEdges = el("g", {}, svg), gClaims = el("g", {}, svg), gNodes = el("g", {}, svg), gLabels = el("g", {}, svg);
  if (opts.spawnLines) for (const a of others) el("line", { x1: cx, y1: cy, x2: pos[a.id].x, y2: pos[a.id].y, class: "cm-spawn", "data-a": a.id }, gSpawn);

  // edges
  for (const e of edges) {
    const A = pos[e.from], B = pos[e.to];
    const dx = B.x - A.x, dy = B.y - A.y, L = Math.hypot(dx, dy) || 1;
    const mx = (A.x + B.x) / 2 - (dy / L) * L * 0.16, my = (A.y + B.y) / 2 + (dx / L) * L * 0.16;
    const shrink = (P, Q, r) => { const ux = Q.x - P.x, uy = Q.y - P.y, ul = Math.hypot(ux, uy) || 1; return { x: P.x + (ux / ul) * r, y: P.y + (uy / ul) * r }; };
    const a0 = shrink(A, { x: mx, y: my }, 20), b0 = shrink(B, { x: mx, y: my }, 22);
    e.el = el("path", { d: `M${a0.x.toFixed(1)},${a0.y.toFixed(1)} Q${mx.toFixed(1)},${my.toFixed(1)} ${b0.x.toFixed(1)},${b0.y.toFixed(1)}`, class: "cm-edge" }, gEdges);
    e.mid = { x: 0.25 * a0.x + 0.5 * mx + 0.25 * b0.x, y: 0.25 * a0.y + 0.5 * my + 0.25 * b0.y };
    e.label = text(gLabels, e.mid.x, e.mid.y - 6, "", { class: "cm-elabel", "text-anchor": "middle" });
    const rels = {};
    e.items.forEach((x) => (rels[x.rel] = (rels[x.rel] || 0) + 1));
    hover(e.el, `<b>${esc(roleOf(e.to))}</b> built on <b>${esc(roleOf(e.from))}</b><br><span class="n">${Object.entries(rels).map(([k, v]) => `${v} ${k}`).join(" · ")}</span>` +
      (judged ? `<br><span class="muted">${e.items.filter((x) => x.ok).length} confirmed by ablation · ${e.items.filter((x) => x.failedTarget).length} on a claim that failed</span>` : "") +
      `<br><span class="muted">${esc(e.items[0].d.title.slice(0, 120))}</span>`);
    e.el.addEventListener("click", (ev) => { ev.stopPropagation(); select(e.to); });
  }

  // claims: a sunflower of dots around their author, opening away from the centre
  const perAgent = {};
  for (const c of claims) {
    const k = (perAgent[c.agent] = (perAgent[c.agent] || 0) + 1) - 1;
    const P = pos[c.agent], out = c.agent === "founder" ? 0 : P.ang;
    const r = (c.agent === "founder" ? 26 : 22) + 4.6 * Math.sqrt(k + 1), th = (c.agent === "founder" ? 0 : out) + k * 2.39996;
    c._x = P.x + Math.cos(th) * r; c._y = P.y + Math.sin(th) * r;
    c._el = el("circle", { cx: c._x.toFixed(1), cy: c._y.toFixed(1), r: 3.6, class: "cm-claim pend" }, gClaims);
    if (c.frontier && opts.rings) c._ring = el("circle", { cx: c._x.toFixed(1), cy: c._y.toFixed(1), r: 5.6, class: "cm-ring" }, gClaims);
    const verdict = { accepted: "verified", rejected: "rejected", superseded: "duplicate", uncredited: "not credited" }[c.status] || "not verified";
    hover(c._el, `<span class="muted n">${fmtT(c.t)} · ${esc(roleOf(c.agent))} · ${judged ? verdict : "claim"}${c.frontier ? " · new best" : ""}</span><br>${esc(c.title || c.id)}`);
  }
  const vOrder = claims.slice();  // verdicts land in the order claims were made

  // researcher nodes
  const nodes = {};
  for (const a of agents) {
    const P = pos[a.id], founder = a.id === "founder";
    const g = el("g", { class: "cm-node" + (founder ? " founder" : ""), transform: `translate(${P.x.toFixed(1)},${P.y.toFixed(1)})` }, gNodes);
    el("circle", { r: founder ? 15 : 12, class: "cm-body" }, g);
    el("circle", { r: founder ? 4.5 : 3.5, class: "cm-core" }, g);
    const nClaims = claims.filter((c) => c.agent === a.id).length, petal = (founder ? 26 : 22) + 4.6 * Math.sqrt(nClaims) + 8;
    const below = founder ? -petal - 16 : (P.y > cy + 5 ? petal + 6 : -petal - 16), anchor = "middle";
    const lbl = founder ? "FOUNDER" : roleOf(a.id).toUpperCase();
    text(g, 0, below, lbl.length > 26 ? lbl.slice(0, 25) + "…" : lbl, { class: "cm-role", "text-anchor": anchor });
    const sub = text(g, 0, below + 13, "", { class: "cm-sub", "text-anchor": anchor });
    g.addEventListener("click", (ev) => { ev.stopPropagation(); select(a.id); });
    nodes[a.id] = { g, sub, a };
  }
  svg.addEventListener("click", () => select(null));

  let selected = null;
  function select(id) {
    selected = id;
    svg.classList.toggle("dim", !!id);
    for (const [k, n] of Object.entries(nodes)) n.g.classList.toggle("hl", k === id);
    for (const c of claims) c._el.classList.toggle("hl", c.agent === id);
    for (const e of edges) e.el.classList.toggle("hl", e.from === id || e.to === id);
    if (opts.onSelect) opts.onSelect(id ? { agent: agents.find((a) => a.id === id), claims: claims.filter((c) => c.agent === id),
      into: edges.filter((e) => e.to === id), out: edges.filter((e) => e.from === id), judged, roleOf } : null);
  }

  let cur = -1;
  function render(t) {
    if (t === cur) return;
    cur = t;
    const verifying = judged && t >= dur;
    const vcut = verifying ? Math.floor(((t - dur) / (end - dur)) * vOrder.length) : -1;
    const flipped = new Set(verifying ? vOrder.slice(0, Math.max(0, vcut)).map((c) => c.id) : []);
    for (const a of agents) {
      const on = (a.t_born || 0) <= t;
      nodes[a.id].g.style.display = on ? "" : "none";
      const mine = claims.filter((c) => c.agent === a.id && c.t <= t);
      const ok = mine.filter((c) => flipped.has(c.id) && c.status === "accepted").length;
      nodes[a.id].sub.textContent = mine.length ? `${mine.length} claim${mine.length === 1 ? "" : "s"}${verifying ? ` · ${ok} verified` : ""}` : "";
    }
    gSpawn.querySelectorAll("line").forEach((l) => (l.style.display = (agents.find((a) => a.id === l.dataset.a).t_born || 0) <= t ? "" : "none"));
    for (const c of claims) {
      const on = c.t <= t;
      c._el.style.display = on ? "" : "none";
      const st = flipped.has(c.id) ? (c.status === "accepted" ? "ok" : c.status === "rejected" ? "bad" : c.status === "superseded" ? "dup" : "pend") : "pend";
      c._el.setAttribute("class", `cm-claim ${st}${c.agent === selected ? " hl" : ""}`);
      if (c._ring) c._ring.style.display = on && flipped.has(c.id) ? "" : "none";
    }
    for (const e of edges) {
      const vis = e.items.filter((x) => x.t <= t);
      e.el.style.display = vis.length ? "" : "none";
      e.el.setAttribute("stroke-width", (1 + 1.5 * Math.sqrt(vis.length)).toFixed(2));
      const ok = verifying ? vis.filter((x) => x.ok && flipped.has(x.d.id)).length : 0;
      const bad = verifying ? vis.filter((x) => x.failedTarget && flipped.has(x.s.id)).length : 0;
      const cls = ok ? "ok" : bad && bad >= vis.length / 2 ? "bad" : "dec";
      e.el.setAttribute("class", `cm-edge ${cls}${selected && (e.from === selected || e.to === selected) ? " hl" : ""}`);
      if (opts.arrows) e.el.setAttribute("marker-end", `url(#cm-ar${cls === "ok" ? "-ok" : cls === "bad" ? "-bad" : ""})`);
      const show = vis.length && (ok || (cls === "bad" && bad >= 2));
      e.label.style.display = show ? "" : "none";
      if (show) {
        const rels = {};
        vis.forEach((x) => (rels[x.rel] = (rels[x.rel] || 0) + 1));
        const top = Object.entries(rels).sort((a, b) => b[1] - a[1])[0];
        e.label.textContent = ok ? `${ok} confirmed` : `${vis.filter((x) => x.failedTarget).length} on failed work`;
        e.label.setAttribute("class", `cm-elabel ${cls}`);
      }
    }
    if (opts.onTick) opts.onTick({ t, dur, end, verifying });
  }
  render(end);
  return { render, select, dur, end, judged, totals: { researchers: agents.length, claims: claims.length } };
}

// ---------- 11. compare models (after vals.ai "compare task results"): one track per field, one dot per model ----------

export const COMPARE_METRICS = {
  verified: { label: "Verified claims", unit: "per community run", log: false, get: (t, s) => perRun(t, "R") },
  output: { label: "Output vs solo", unit: "community ÷ solo verified claims", log: true, ref: 1, refLabel: "SOLO", get: (t, s) => ratio(t, s, "R").v, fmt: (v) => fx(v) },
  frontier: { label: "New bests", unit: "per community run", log: false, get: (t) => perRun(t, "frontier") },
  cpc: { label: "Cost per verified claim", unit: "community, lower is better", log: true, low: true, get: (t) => (val(t, "R") > 0 ? val(t, "cost") / val(t, "R") : null), fmt: (v) => fusd(v) },
};

export function compareModels(host, doc, ds, metric = "verified", only = null) {
  const M = COMPARE_METRICS[metric];
  const models = modelsIn(doc, ds).filter((m) => !only || only.includes(m.key));
  const rows = [["all", "Overall"], ...DOMAIN_ORDER_UI.map((d) => [d, DOMAIN_SHORT[d]])].map(([d, label]) => ({
    d, label, pts: models.map((m) => ({ m, v: M.get(cell(ds, m.key, d, "team"), cell(ds, m.key, d, "solo")) })).filter((p) => p.v != null && p.v > (M.log ? 0 : -1)),
  }));
  const all = rows.flatMap((r) => r.pts.map((p) => p.v)).concat(M.ref ? [M.ref] : []);
  let lo = M.log ? Math.min(...all) / 1.6 : 0, hi = Math.max(...all) * (M.log ? 1.6 : 1.1);
  const pos = (v) => (M.log ? (Math.log(v) - Math.log(lo)) / (Math.log(hi) - Math.log(lo)) : (v - lo) / (hi - lo)) * 100;
  const f = M.fmt || ((v) => fnum(v));
  host.innerHTML = `<div class="cmp">
    <div class="cmp-scale"><span>${f(lo)}</span>${M.ref ? `<span style="left:${pos(M.ref)}%">${M.refLabel}</span>` : ""}<span style="left:100%">${f(hi)}</span></div>
    ${rows.map((r) => {
      const best = r.pts.slice().sort((a, b) => (M.low ? a.v - b.v : b.v - a.v))[0];
      return `<div class="cmp-row"><div class="cmp-l">${r.label}</div><div class="cmp-t">
        ${best ? `<span class="cmp-band" style="left:0;width:${pos(best.v)}%"></span>` : ""}
        ${M.ref ? `<span class="cmp-ref" style="left:${pos(M.ref)}%"></span>` : ""}
        ${r.pts.map((p) => `<span class="cmp-dot" style="left:${pos(p.v)}%;background:${modelColor(p.m.key)}" title="${esc(p.m.name)} · ${r.label}: ${f(p.v)}"></span>`).join("")}
        ${best ? `<span class="cmp-name" style="left:${pos(best.v)}%">${esc(short(best.m.name))} ${f(best.v)}</span>` : ""}
      </div></div>`; }).join("")}
  </div>`;
}

// ---------- 11. the build-on funnel: what agents claimed about each other's work, and what the verifier confirmed ----------
// Two groups on one linear scale (the largest stage): build-ons (extends) and challenges (corrects, refutes). Counts
// come from site/data/flow.json `funnel`, computed by build_flow.py from the verdicts of the community runs.
export function funnel(host, F, opts = {}) {
  if (!F) { host.innerHTML = ""; return; }
  const k = opts.scale || 1, W = opts.w || 960, lw = 330 * k, bh = 22 * k, gap = 12 * k, gh = 34 * k, right = 70 * k;
  const groups = [
    ["Building on a teammate's result", [
      ["Claimed to extend a teammate's result", F.claimed, "Cross-researcher “extends” relations declared on submitted claims"],
      ["Both results verified", F.both_verified, "The claim and the teammate's claim it extends both passed verification"],
      ["Testable by ablation", F.testable, "The claim consumed a teammate's data table, which the verifier can shuffle; the rest reused code or models"],
      ["Confirmed", F.confirmed, "Shuffling the teammate's table removed the gain"],
    ]],
    ["Catching a teammate's error", [
      ["Claimed to correct or refute a teammate", F.challenges, "Cross-researcher “corrects” or “refutes” relations"],
      ["Confirmed", F.challenges_confirmed, "The challenger passed verification and the claim it challenged failed"],
    ]],
  ];
  const rows = groups.reduce((s, g) => s + g[1].length, 0);
  const H = groups.length * gh + rows * (bh + gap) + 6 * k;
  const svg = frame(host, W, H, "Build-on funnel");
  const x = linScale(0, Math.max(F.claimed, F.challenges), 0, W - lw - right);
  let y = 0;
  for (const [title, items] of groups) {
    text(svg, 0, y + gh - 12 * k, title, { class: "tick", style: `font-size:${11 * k}px;letter-spacing:.06em;text-transform:uppercase;fill:var(--muted)` });
    el("line", { x1: 0, x2: W, y1: y + gh - 4 * k, y2: y + gh - 4 * k, stroke: "var(--line)" }, svg);
    y += gh;
    items.forEach(([label, v, hint], i) => {
      const last = i === items.length - 1, g = el("g", {}, svg);
      el("title", {}, g).textContent = `${label}: ${v}. ${hint}`;
      text(g, 0, y + bh * 0.68, label, { style: `font:${14 * k}px var(--sans);fill:var(--ink2)` });
      const bw = Math.max(v ? 3 : 0, x(v));
      if (bw) el("rect", { x: lw, y, width: bw, height: bh, rx: Math.min(3, bw / 2), fill: last ? "var(--accepted)" : "var(--ink)", opacity: last ? 1 : 0.18 + 0.12 * (items.length - 1 - i) / items.length }, g);
      else el("line", { x1: lw, x2: lw, y1: y, y2: y + bh, stroke: "var(--muted)", "stroke-width": 1 }, g);
      text(g, lw + bw + 8 * k, y + bh * 0.7, String(v), { style: `font:${15 * k}px var(--mono);fill:var(--ink)` });
      y += bh + gap;
    });
  }
}

// ---------- 12. claim outcomes: where each model's claims went, community and solo ----------
// 100% bars per model and mode, totals over runs (mean × n): verified, failed on hidden data (not_replicated, which
// includes tests that crashed), duplicates, and every other rejection or pending check.
export function outcomeTotals(ds, model, domain, mode) {
  const rs = ds.rows.filter((r) => r.model === model && r.mode === mode && (domain === "all" || r.domain === domain));
  const sum = (k) => rs.reduce((s, r) => s + (val(r, k) || 0) * r.n, 0);
  const tot = sum("submitted"), ok = sum("R"), hid = sum("not_replicated"), dup = sum("duplicates");
  return { tot, ok, hid, dup, other: Math.max(0, tot - ok - hid - dup), fab: sum("fabrication"), n: rs.reduce((s, r) => s + r.n, 0) };
}
export const OUTCOME_SEGS = [
  ["ok", "Verified", "var(--accepted)"],
  ["hid", "Failed on held-out data", "var(--rejected)"],
  ["dup", "Duplicate", "#8a857f"],
  ["other", "Other rejection", "#d8d4cf"],
];
export function outcomes(host, doc, ds, domain, opts = {}) {
  const W = opts.w || 1100, lw = 230, rw = 96, bh = 20, gap = 8, sep = 14;
  const rows = [];
  for (const m of modelsIn(doc, ds)) for (const mode of ["team", "solo"]) {
    const o = outcomeTotals(ds, m.key, domain, mode);
    if (o.tot) rows.push({ m, mode, o });
  }
  const models = new Set(rows.map((r) => r.m.key)).size;
  const H = rows.length * (bh + gap) + (models - 1) * sep + 4;
  const svg = frame(host, W, H, "Claim outcomes by model, team and solo");
  const x = linScale(0, 1, lw, W - rw);
  let y = 0, prev = null;
  for (const r of rows) {
    if (prev && prev !== r.m.key) y += sep;
    prev = r.m.key;
    const g = el("g", {}, svg);
    if (r.mode === "team") {
      el("circle", { cx: 5, cy: y + bh / 2, r: 4, fill: modelColor(r.m.key) }, g);
      text(g, 16, y + bh * 0.72, short(r.m.name), { style: "font:13.5px var(--sans);fill:var(--ink)" });
    }
    text(g, 120, y + bh * 0.72, r.mode === "team" ? "Team" : "Solo", { style: "font:11px var(--mono);letter-spacing:.05em;text-transform:uppercase;fill:var(--muted)" });
    let x0 = 0;
    OUTCOME_SEGS.forEach(([k, label, col]) => {
      const f = r.o[k] / r.o.tot;
      if (!f) return;
      const s = el("rect", { x: x(x0) + (x0 ? 1 : 0), y, width: Math.max(0, x(x0 + f) - x(x0) - (x0 ? 1 : 0)), height: bh, fill: col }, g);
      el("title", {}, s).textContent = `${r.m.name}, ${r.mode === "team" ? "team" : "solo"}: ${label.toLowerCase()} ${Math.round(r.o[k])} of ${Math.round(r.o.tot)} (${Math.round(f * 100)}%)`;
      x0 += f;
    });
    text(g, W - rw + 10, y + bh * 0.72, `${Math.round(r.o.ok)} / ${Math.round(r.o.tot)}`, { style: "font:13px var(--mono);fill:var(--ink)" });
    y += bh + gap;
  }
  return rows;
}
