/* F1 Race Replay Web — FastF1 静的データによるリプレイ＋分析 UI
 *
 * データは build_data.py が書き出す JSON（track / series / laps / stints /
 * trackStatus / raceControl）だけを使う。外部ライブラリなし。
 *
 * 座標系: F1 タイミングシステムの X/Y（1/10 m 単位）。track.bbox は同じ単位。
 */
"use strict";

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

const clamp = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);
const lerp = (a, b, u) => a + (b - a) * u;

function fmtClock(sec) {
  if (!Number.isFinite(sec)) return "--:--";
  const s = Math.max(0, Math.floor(sec));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const ss = s % 60;
  return h > 0
    ? `${h}:${String(m).padStart(2, "0")}:${String(ss).padStart(2, "0")}`
    : `${String(m).padStart(2, "0")}:${String(ss).padStart(2, "0")}`;
}

// Signed variant: race control messages exist before the formation lap (t < 0)
// and must not be collapsed onto 00:00.
function fmtRelClock(sec) {
  if (!Number.isFinite(sec)) return "--:--";
  const sign = sec < 0 ? "-" : "";
  const body = fmtClock(Math.abs(sec));
  return body === "--:--" ? body : `${sign}${body}`;
}

function fmtLapTime(ms) {
  if (!Number.isFinite(ms) || ms <= 0) return "—";
  const total = ms / 1000;
  const m = Math.floor(total / 60);
  const s = total - m * 60;
  return `${m}:${s.toFixed(3).padStart(6, "0")}`;
}

function fmtGap(sec) {
  if (!Number.isFinite(sec)) return "—";
  if (sec < 10) return `+${sec.toFixed(1)}`;
  if (sec < 60) return `+${sec.toFixed(1)}`;
  return `+${Math.floor(sec / 60)}:${String(Math.round(sec % 60)).padStart(2, "0")}`;
}

const COMPOUND_COLORS = {
  SOFT: "#ff0000",
  MEDIUM: "#ffd12e",
  HARD: "#f2f2f2",
  INTERMEDIATE: "#3fbf3f",
  WET: "#2f7bff",
  UNKNOWN: "#8b95a7",
};
const compoundColor = (c) => COMPOUND_COLORS[c] || COMPOUND_COLORS.UNKNOWN;

const STATUS_STYLE = {
  green: { label: "GREEN", color: "#22c55e" },
  allclear: { label: "GREEN", color: "#22c55e" },
  yellow: { label: "YELLOW", color: "#f5c400" },
  double_yellow: { label: "DOUBLE YELLOW", color: "#e8930c" },
  safety_car: { label: "SAFETY CAR", color: "#f5c400" },
  virtual_safety_car: { label: "VSC", color: "#c9a227" },
  red: { label: "RED", color: "#e10600" },
  chequered_flag: { label: "CHEQUERED", color: "#d0d0d0" },
};
/* FastF1 exposes TrackStatus as a numeric code; `name` carries the meaning. */
const STATUS_BY_CODE = {
  "1": "green",
  "2": "yellow",
  "3": "double_yellow",
  "4": "safety_car",
  "5": "red",
  "6": "virtual_safety_car",
};

function statusStyle(seg) {
  if (!seg) return { key: "none", label: "—", color: "#5b6572" };
  const key = STATUS_STYLE[seg.name] ? seg.name : STATUS_BY_CODE[String(seg.code)];
  if (STATUS_STYLE[key]) return { ...STATUS_STYLE[key], key };
  const label = seg.label || seg.name || String(seg.code);
  return { key: "none", label: String(label).toUpperCase(), color: "#5b6572" };
}

const FLAG_COLORS = {
  CLEAR: "#22c55e",
  GREEN: "#22c55e",
  YELLOW: "#f5c400",
  "DOUBLE YELLOW": "#e8930c",
  RED: "#e10600",
  BLUE: "#3b82f6",
  BLACK: "#111111",
  "BLACK AND WHITE": "#9aa3b2",
  WHITE: "#e5e7eb",
  CHEQUERED: "#d0d0d0",
};

/* ---------------------------------------------------------------- state */

const state = {
  meta: null,
  track: null,
  drivers: [],
  driverByAbbr: new Map(),
  series: null,
  laps: null,
  stints: null,
  trackStatus: [],
  raceControl: [],
  sessions: [],
  catalog: [],
  filters: { year: "", type: "" },
  currentSlug: "",
  pitStops: [],
  weather: [],
  schema: null,
  data: null,

  duration: 0,
  t: 0,
  playing: false,
  speed: 4,

  selected: new Set(),
  tab: "lapTimes",

  lapMarkers: [],          // 先頭マシンのラップ開始時刻
  standingsCache: null,
  standingsT: NaN,

  chartDirty: true,
  chartCanvas: null,
  chartKey: "",
  trackCanvasCache: null,
  trackCacheKey: "",
};

/* ------------------------------------------------------------- helpers */

function sampleSeries(s, t) {
  const dt = state.meta.dtMain;
  const i = clamp(Math.round((t - s.t0) / dt), 0, s.x.length - 1);
  return {
    i,
    x: s.x[i],
    y: s.y[i],
    lap: s.lap[i],
    rel: s.rel[i],
    st: s.st[i],
  };
}

function sampleDetail(s, t) {
  const dt = state.meta.dtDetail;
  const i = clamp(Math.round((t - s.t0) / dt), 0, s.speed.length - 1);
  return {
    speed: s.speed[i],
    gear: s.gear[i],
    drs: s.drs[i],
    throttle: s.throttle[i],
    brake: s.brake[i],
  };
}

const progressOf = (s, t) => {
  const p = sampleSeries(s, t);
  return { progress: p.lap - 1 + p.rel / state.meta.relScale, ...p };
};

function statusAt(t) {
  let cur = null;
  for (const seg of state.trackStatus) {
    if (seg.t > t) break;
    cur = seg;
  }
  return cur;
}

function tyreFor(abbr, lap) {
  const rows = state.laps[abbr];
  if (!rows) return null;
  let best = null;
  for (const r of rows) {
    if (r.lap === lap) best = r;
    else if (r.lap < lap && (!best || r.lap > best.lap)) best = r;
  }
  return best ? best.compound : null;
}

/* --------------------------------------------------- standings + board */

function computeStandings(t) {
  const live = [];
  const done = [];
  for (const d of state.drivers) {
    const s = state.series[d.abbr];
    if (!s) continue;
    const p = progressOf(s, t);
    const retired = d.status && d.status !== "Finished" && t > s.t1 + 1;
    const row = {
      d,
      s,
      progress: p.progress,
      lap: p.lap,
      st: p.st,
      present: p.st !== 0 && t <= s.t1 + 1,
      retired,
      compound: tyreFor(d.abbr, p.lap),
      inPit: p.st === 2 || p.st === 3,
    };
    if (row.present) live.push(row);
    else done.push(row);
  }
  live.sort((a, b) => b.progress - a.progress);
  done.sort((a, b) => {
    if (a.retired !== b.retired) return a.retired ? 1 : -1;
    if (a.d.finish && b.d.finish) return a.d.finish - b.d.finish;
    return b.progress - a.progress;
  });

  const rows = live.concat(done);
  let leader = live.length ? live[0] : null;
  rows.forEach((r, i) => {
    r.pos = i + 1;
    r.gap = "—";
    if (!leader || r === leader) return;
    const dp = leader.progress - r.progress;
    if (dp >= 1) {
      r.gap = `+${Math.floor(dp)}L`;
    } else if (dp > 0.0005) {
      const lapMeters = (state.meta && state.meta.trackLengthMeters) ||
        (state.track ? state.track.lengthMeters : null);
      if (!lapMeters) {
        // 1 周距離が分からないセッション（位置データなし）は周数の割合で表示
        r.gap = `+${dp.toFixed(2)}L`;
        return;
      }
      const det = sampleDetail(r.s, t);
      const v = Math.max(15, det.speed / 3.6);
      r.gap = fmtGap((dp * lapMeters) / v);
    } else {
      r.gap = "+0.0";
    }
  });
  return rows;
}

function renderBoard(rows) {
  const body = $("#leaderBody");
  const sel = state.selected;
  body.innerHTML = rows
    .map((r) => {
      const cls = [
        r.d.abbr && sel.has(r.d.abbr) ? "sel" : "",
        r.retired ? "dnf" : "",
      ].join(" ");
      const tire = r.compound
        ? `<span class="tire-dot" style="background:${compoundColor(r.compound)}" title="${r.compound}"></span>`
        : "";
      const pit = r.inPit ? '<span class="pit-flag"> PIT</span>' : "";
      const grid = r.d.grid != null ? r.d.grid : "—";
      return `<tr class="${cls}" data-abbr="${r.d.abbr}">
        <td class="pos-cell">${r.pos}</td>
        <td class="driver-cell"><span class="num">${r.d.number ?? ""}</span>${r.d.shortName || r.d.name}</td>
        <td><span class="team-bar" style="background:${r.d.teamColor}"></span>${r.d.team}</td>
        <td class="gap-cell">${r.gap}</td>
        <td>${r.retired ? "DNF" : r.lap}</td>
        <td>${tire}${pit}</td>
      </tr>`;
    })
    .join("");
}

/* ------------------------------------------------------- track drawing */

function computeTransform(w, h) {
  const bb = state.track.bbox;
  const pad = 0.06;
  const wUnits = bb.maxX - bb.minX;
  const hUnits = bb.maxY - bb.minY;
  const scale = Math.min((w * (1 - 2 * pad)) / wUnits, (h * (1 - 2 * pad)) / hUnits);
  const cx = (bb.minX + bb.maxX) / 2;
  const cy = (bb.minY + bb.maxY) / 2;
  return {
    scale,
    cx,
    cy,
    w,
    h,
    px: (x) => w / 2 + (x - cx) * scale,
    py: (y) => h / 2 - (y - cy) * scale, // Y 反転（タイミング系は Y 上方向）
  };
}

function outlinePath(tf) {
  const pts = state.track.outline;
  const p = new Path2D();
  p.moveTo(tf.px(pts[0][0]), tf.py(pts[0][1]));
  for (let i = 1; i < pts.length; i++) p.lineTo(tf.px(pts[i][0]), tf.py(pts[i][1]));
  p.closePath();
  return p;
}

function pitPath(tf) {
  const pts = state.track.pitlane || [];
  if (pts.length < 2) return null;
  const p = new Path2D();
  p.moveTo(tf.px(pts[0][0]), tf.py(pts[0][1]));
  for (let i = 1; i < pts.length; i++) p.lineTo(tf.px(pts[i][0]), tf.py(pts[i][1]));
  return p;
}

function drawTrackLayer(tf) {
  const off = document.createElement("canvas");
  off.width = tf.w;
  off.height = tf.h;
  const c = off.getContext("2d");
  const road = Math.max(6, 130 * tf.scale);

  const outline = outlinePath(tf);
  c.lineJoin = "round";
  c.lineCap = "round";

  // 縁取り → アスファルト → 中心線
  c.strokeStyle = "#2b3444";
  c.lineWidth = road + 6;
  c.stroke(outline);
  c.strokeStyle = "#4c5666";
  c.lineWidth = road;
  c.stroke(outline);
  c.strokeStyle = "rgba(255,255,255,0.16)";
  c.lineWidth = Math.max(1, road * 0.06);
  c.setLineDash([Math.max(3, road * 0.5), Math.max(4, road * 0.6)]);
  c.stroke(outline);
  c.setLineDash([]);

  const pit = pitPath(tf);
  if (pit) {
    c.strokeStyle = "#2b3444";
    c.lineWidth = road * 0.75 + 5;
    c.stroke(pit);
    c.strokeStyle = "#39424f";
    c.lineWidth = road * 0.75;
    c.stroke(pit);
    c.strokeStyle = "rgba(225,6,0,0.55)";
    c.lineWidth = Math.max(1, road * 0.12);
    c.setLineDash([6, 6]);
    c.stroke(pit);
    c.setLineDash([]);
  }

  // スタート／フィニッシュ線
  const pts = state.track.outline;
  const a = pts[0];
  const b = pts[3] || pts[1];
  const dx = b[0] - a[0];
  const dy = b[1] - a[1];
  const n = Math.hypot(dx, dy) || 1;
  const nx = -dy / n;
  const ny = dx / n;
  const half = road * 0.75;
  const x0 = tf.px(a[0]), y0 = tf.py(a[1]);
  const x1 = tf.px(a[0] + nx * half * 2), y1 = tf.py(a[1] + ny * half * 2);
  const cells = 6;
  for (let i = 0; i < cells; i++) {
    const u0 = i / cells, u1 = (i + 1) / cells;
    c.strokeStyle = i % 2 ? "#f5f5f5" : "#1b1f27";
    c.lineWidth = Math.max(2, (Math.hypot(x1 - x0, y1 - y0) / cells) * 0.9);
    c.beginPath();
    c.moveTo(lerp(x0, x1, u0), lerp(y0, y1, u0));
    c.lineTo(lerp(x0, x1, u1), lerp(y0, y1, u1));
    c.stroke();
  }

  return off;
}

function speedColor(v, min, max) {
  const u = clamp((v - min) / Math.max(1, max - min), 0, 1);
  const stops = [
    [40, 90, 200],
    [40, 190, 120],
    [230, 200, 40],
    [235, 90, 30],
    [225, 6, 0],
  ];
  const f = u * (stops.length - 1);
  const i = Math.min(stops.length - 2, Math.floor(f));
  const t = f - i;
  const a = stops[i], b = stops[i + 1];
  return `rgb(${Math.round(lerp(a[0], b[0], t))},${Math.round(lerp(a[1], b[1], t))},${Math.round(lerp(a[2], b[2], t))})`;
}

function drawSpeedOverlay(c, tf, abbr) {
  const s = state.series[abbr];
  if (!s) return;
  const lap = sampleSeries(s, state.t).lap;
  const dt = state.meta.dtMain;
  const i0 = clamp(Math.round((state.t - s.t0) / dt), 0, s.x.length - 1);
  // 現在のラップ範囲を探す
  let a = i0;
  while (a > 0 && s.lap[a - 1] === lap) a--;
  let b = i0;
  while (b < s.x.length - 1 && s.lap[b + 1] === lap) b++;
  if (b - a < 3) return;

  let min = Infinity, max = -Infinity;
  for (let i = a; i <= b; i++) {
    const v = sampleDetail(s, s.t0 + i * dt).speed;
    if (v < min) min = v;
    if (v > max) max = v;
  }
  c.lineCap = "round";
  c.lineWidth = Math.max(3, 90 * tf.scale);
  for (let i = a; i < b; i++) {
    const v = sampleDetail(s, s.t0 + i * dt).speed;
    c.strokeStyle = speedColor(v, min, max);
    c.beginPath();
    c.moveTo(tf.px(s.x[i]), tf.py(s.y[i]));
    c.lineTo(tf.px(s.x[i + 1]), tf.py(s.y[i + 1]));
    c.stroke();
  }
}

function drawCars(c, tf) {
  const rows = state.standingsCache || [];
  const r = Math.max(4.5, Math.min(9, 120 * tf.scale));
  for (const row of rows) {
    if (!row.present) continue;
    const s = row.s;
    const p = sampleSeries(s, state.t);
    const px = tf.px(p.x), py = tf.py(p.y);
    const sel = state.selected.has(row.d.abbr);

    // 進行方向のしりとり（選択マシンのみ）
    if (sel) {
      c.strokeStyle = row.d.teamColor;
      c.globalAlpha = 0.35;
      c.lineWidth = Math.max(2, r * 0.7);
      c.beginPath();
      c.moveTo(px, py);
      for (let k = 1; k <= 6; k++) {
        const q = sampleSeries(s, state.t - k * 0.5);
        c.lineTo(tf.px(q.x), tf.py(q.y));
      }
      c.stroke();
      c.globalAlpha = 1;
    }

    c.beginPath();
    c.arc(px, py, sel ? r + 2 : r, 0, Math.PI * 2);
    c.fillStyle = row.d.teamColor || "#8b95a7";
    c.fill();
    c.lineWidth = sel ? 2.5 : 1.2;
    c.strokeStyle = sel ? "#ffffff" : "rgba(0,0,0,0.55)";
    c.stroke();

    if (sel) {
      c.font = "700 12px ui-sans-serif, system-ui, sans-serif";
      c.fillStyle = "#ffffff";
      c.strokeStyle = "rgba(0,0,0,0.75)";
      c.lineWidth = 3;
      c.lineJoin = "round";
      const label = row.d.abbr;
      c.strokeText(label, px + r + 4, py - r - 2);
      c.fillText(label, px + r + 4, py - r - 2);
    }
  }
}

function renderTrack() {
  const canvas = $("#trackCanvas");
  if (!state.track) {
    // 位置データを持たないセッション（2018年 R1-R2）は地図を描かない
    if (canvas.width || canvas.height) { canvas.width = 1; canvas.height = 1; }
    state.tf = null;
    state.trackCacheKey = "";
    return;
  }
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const w = Math.max(1, Math.floor(rect.width * dpr));
  const h = Math.max(1, Math.floor(rect.height * dpr));
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
  }
  const c = canvas.getContext("2d");
  const tf = computeTransform(w, h);

  const key = `${w}x${h}`;
  if (state.trackCacheKey !== key) {
    state.trackCanvasCache = drawTrackLayer(tf);
    state.trackCacheKey = key;
  }
  c.clearRect(0, 0, w, h);
  c.drawImage(state.trackCanvasCache, 0, 0);

  if (state.tab === "telemetry" && state.selected.size === 1) {
    drawSpeedOverlay(c, tf, Array.from(state.selected)[0]);
  }
  drawCars(c, tf);
  state.tf = tf;
}

/* ------------------------------------------------------------- charts */

const CHART_PAD = { l: 58, r: 14, t: 16, b: 30 };

function chartSize() {
  const wrap = $("#chartWrap");
  const rect = wrap.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  return {
    w: Math.max(200, Math.floor(rect.width * dpr)),
    h: Math.max(160, Math.floor((rect.height || 360) * dpr)),
    dpr,
  };
}

function buildChart() {
  const { w, h } = chartSize();
  const canvas = state.chartCanvas || document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  const c = canvas.getContext("2d");
  c.clearRect(0, 0, w, h);
  c.fillStyle = "#12161d";
  c.fillRect(0, 0, w, h);

  const dpr = window.devicePixelRatio || 1;
  const abbrs = Array.from(state.selected);
  const note = $("#chartNote");
  const plotW = w - CHART_PAD.l - CHART_PAD.r;
  const plotH = h - CHART_PAD.t - CHART_PAD.b;

  const mkScale = (xMax, yMin, yMax, xMap, yMap) => ({
    x: (v) => CHART_PAD.l + (v / xMax) * plotW,
    y: (v) => CHART_PAD.t + plotH - ((v - yMin) / (yMax - yMin)) * plotH,
    xMax,
    yMin,
    yMax,
    xMap,
    yMap,
  });

  if (state.tab === "lapTimes") {
    const laps = [];
    let best = Infinity;
    for (const a of abbrs) {
      for (const r of state.laps[a] || []) {
        if (r.del || r.gen || !r.time) continue;
        laps.push({ a, r });
        if (r.time < best) best = r.time;
      }
    }
    if (!laps.length) {
      note.textContent = "このマシン群には計測ラップがありません。";
      state.chartCanvas = canvas;
      return;
    }
    const maxLap = state.meta.totalLaps;
    let tMin = Infinity, tMax = -Infinity;
    for (const { r } of laps) {
      tMin = Math.min(tMin, r.time);
      tMax = Math.max(tMax, r.time);
    }
    tMin = Math.max(0, tMin * 0.985);
    tMax *= 1.01;
    const sc = mkScale(maxLap, tMin / 1000, tMax / 1000);
    const yTicks = [];
    const stepT = niceStep((tMax - tMin) / 1000 / 6);
    for (let v = Math.ceil(tMin / 1000 / stepT) * stepT; v <= tMax / 1000; v += stepT) yTicks.push(v);
    drawAxes(c, w, h, sc, yTicks, (v) => fmtLapTime(v * 1000), {
      step: Math.max(1, Math.round(maxLap / 11)),
      map: (v) => sc.x(v),
      f: (v) => String(v),
    });

    // 全体最速ラップのガイド
    let fastest = null;
    for (const a of Object.keys(state.laps)) {
      for (const r of state.laps[a]) {
        if (r.del || r.gen || !r.time) continue;
        if (!fastest || r.time < fastest.time) fastest = { ...r, abbr: a };
      }
    }
    if (fastest) {
      c.strokeStyle = "rgba(225,6,0,0.7)";
      c.setLineDash([6, 5]);
      c.lineWidth = 1.5 * dpr;
      c.beginPath();
      c.moveTo(CHART_PAD.l, sc.y(fastest.time / 1000));
      c.lineTo(w - CHART_PAD.r, sc.y(fastest.time / 1000));
      c.stroke();
      c.setLineDash([]);
    }

    for (const a of abbrs) {
      const d = state.driverByAbbr.get(a);
      c.strokeStyle = d.teamColor || "#8b95a7";
      c.lineWidth = 1.6 * dpr;
      c.beginPath();
      let started = false;
      for (const r of state.laps[a] || []) {
        if (r.del || r.gen || !r.time) continue;
        const px = sc.x(r.lap), py = sc.y(r.time / 1000);
        if (!started) { c.moveTo(px, py); started = true; } else c.lineTo(px, py);
      }
      c.stroke();
      for (const r of state.laps[a] || []) {
        if (r.del || r.gen || !r.time) continue;
        c.fillStyle = compoundColor(r.compound);
        c.beginPath();
        c.arc(sc.x(r.lap), sc.y(r.time / 1000), 2.6 * dpr, 0, Math.PI * 2);
        c.fill();
      }
    }
    note.innerHTML =
      "各ラップのタイム（縦軸が下ほど速い）。点の色＝コンパウンド、赤破線＝全セッション最速ラップ。" +
      `<span style="color:#8b95a7"> 削除/補完ラップは除外。</span>`;
  } else if (state.tab === "positions") {
    const maxLap = state.meta.totalLaps;
    const n = state.drivers.length;
    const sc = mkScale(maxLap, 1, n);
    const yTicks = [];
    for (let v = 1; v <= n; v += Math.max(1, Math.round(n / 11))) yTicks.push(v);
    drawAxes(c, w, h, sc, yTicks, (v) => `P${v}`, {
      step: Math.max(1, Math.round(maxLap / 11)),
      map: (v) => sc.x(v),
      f: (v) => String(v),
    });
    for (const a of abbrs) {
      const d = state.driverByAbbr.get(a);
      c.strokeStyle = d.teamColor || "#8b95a7";
      c.lineWidth = 2 * dpr;
      c.beginPath();
      let prev = null;
      for (const r of state.laps[a] || []) {
        if (r.gen) continue;
        const px = sc.x(r.lap), py = sc.y(clamp(r.pos || n, 1, n));
        if (prev === null) c.moveTo(px, py);
        else { c.lineTo(px, prev[1]); c.lineTo(px, py); }
        prev = [px, py];
      }
      c.stroke();
    }
    note.textContent = "ラップごとの順位（上=P1）。ラインの色＝コンストラクター。";
  } else if (state.tab === "stints") {
    const maxLap = state.meta.totalLaps;
    const rows = abbrs.length;
    const sc = { x: (v) => CHART_PAD.l + (v / maxLap) * plotW };
    const rowH = plotH / Math.max(1, rows);
    c.font = `${11 * dpr}px ui-sans-serif, system-ui, sans-serif`;
    c.fillStyle = "#8b95a7";
    c.textAlign = "right";
    c.textBaseline = "middle";
    const step = Math.max(1, Math.round(maxLap / 11));
    c.strokeStyle = "#232a35";
    for (let v = 0; v <= maxLap; v += step) {
      c.beginPath();
      c.moveTo(sc.x(v), CHART_PAD.t);
      c.lineTo(sc.x(v), h - CHART_PAD.b);
      c.stroke();
      c.textAlign = "center";
      c.textBaseline = "top";
      c.fillText(String(v), sc.x(v), h - CHART_PAD.b + 6);
      c.textAlign = "right";
      c.textBaseline = "middle";
    }
    abbrs.forEach((a, i) => {
      const d = state.driverByAbbr.get(a);
      const cy = CHART_PAD.t + rowH * (i + 0.5);
      c.fillStyle = "#cfd6e2";
      c.fillText(d.shortName || d.abbr, CHART_PAD.l - 6, cy);
      for (const st of state.stints[a] || []) {
        const x0 = sc.x(st.start), x1 = sc.x(st.end);
        c.fillStyle = compoundColor(st.compound);
        const hgt = Math.min(rowH * 0.62, 22 * dpr);
        c.fillRect(x0, cy - hgt / 2, Math.max(2, x1 - x0), hgt);
        c.fillStyle = "rgba(0,0,0,0.65)";
        c.font = `${10 * dpr}px ui-monospace, monospace`;
        c.textAlign = "left";
        if (x1 - x0 > 40 * dpr) {
          c.fillText(`${st.end - st.start + 1}L ${st.compound}`, x0 + 4 * dpr, cy);
        }
        c.font = `${11 * dpr}px ui-sans-serif, system-ui, sans-serif`;
        c.textAlign = "right";
      }
    });
    note.textContent = "タイヤスティント（横軸＝ラップ）。色＝コンパウンド。";
  } else if (state.tab === "telemetry") {
    const xMax = state.duration;
    let vMax = 0;
    for (const a of abbrs) {
      const s = state.series[a];
      if (!s) continue;
      for (let i = 0; i < s.speed.length; i += 4) vMax = Math.max(vMax, s.speed[i]);
    }
    vMax = Math.max(50, Math.ceil(vMax / 50) * 50);
    const sc = mkScale(xMax, 0, vMax);
    const yTicks = [];
    for (let v = 0; v <= vMax; v += vMax / 5) yTicks.push(v);
    drawAxes(c, w, h, sc, yTicks, (v) => `${Math.round(v)}`, {
      step: Math.max(60, niceStep(xMax / 10)),
      map: (v) => sc.x(v),
      f: (v) => fmtClock(v),
    });
    let drsSeen = 0;
    const dtMain = state.meta.dtMain;
    for (const a of abbrs) {
      const s = state.series[a];
      if (!s) continue;
      const d = state.driverByAbbr.get(a);
      c.strokeStyle = d.teamColor || "#8b95a7";
      c.lineWidth = 1.5 * dpr;
      c.beginPath();
      let started = false;
      for (let i = 0; i < s.speed.length; i++) {
        const t = s.t0 + i * state.meta.dtDetail;
        // Break the line where the car is off the grid (retired, or not yet on
        // track) instead of drawing a flat tail at its last sampled value.
        const j = Math.min(s.st.length - 1, Math.round((t - s.t0) / dtMain));
        if (s.st[j] === 0) {
          started = false;
          continue;
        }
        const px = sc.x(t), py = sc.y(s.speed[i]);
        if (!started) {
          c.moveTo(px, py);
          started = true;
        } else {
          c.lineTo(px, py);
        }
        if (s.drs[i] >= 8) drsSeen++;
      }
      c.stroke();
    }
    note.innerHTML =
      "速度（km/h）のタイム推移。" +
      (drsSeen
        ? "DRS 検知あり。"
        : `<span style="color:#f5c400">このレースは DRS 使用が記録されていません（ウェットコンディション）。</span>`) +
      (abbrs.length > 1 ? " 複数選択中は重なるため、1台だけ選ぶ見やすくなります。" : "");
  } else if (state.tab === "control") {
    drawControlChart(c, w, h, dpr);
    note.textContent = "コース上のセッション状態（上部バー）とレースコントロールメッセージ。バーをクリックするとその時刻へ移動します。";
  }

  state.chartCanvas = canvas;
}

function niceStep(raw) {
  if (!(raw > 0)) return 1;
  const p = Math.pow(10, Math.floor(Math.log10(raw)));
  for (const m of [1, 2, 5, 10]) if (p * m >= raw) return p * m;
  return 10 * p;
}

function drawAxes(c, w, h, sc, yTicks, yFmt, xFmt) {
  const dpr = window.devicePixelRatio || 1;
  c.strokeStyle = "#232a35";
  c.lineWidth = 1;
  c.fillStyle = "#8b95a7";
  c.font = `${11 * dpr}px ui-monospace, monospace`;
  c.textAlign = "right";
  c.textBaseline = "middle";
  for (const v of yTicks) {
    const yy = sc.y(v);
    c.beginPath();
    c.moveTo(CHART_PAD.l, yy);
    c.lineTo(w - CHART_PAD.r, yy);
    c.stroke();
    c.fillText(yFmt(v), CHART_PAD.l - 6, yy);
  }
  c.textAlign = "center";
  c.textBaseline = "top";
  const xMax = sc.xMax ?? xFmt.xMax ?? 0;
  for (let v = 0; v <= xMax + 1e-9; v += xFmt.step) {
    c.fillText(xFmt.f(v), xFmt.map(v), h - CHART_PAD.b + 6);
  }
}

function drawControlChart(c, w, h, dpr) {
  const padL = CHART_PAD.l, padR = CHART_PAD.r;
  const plotW = w - padL - padR;
  const barY = CHART_PAD.t + 10 * dpr;
  const barH = 26 * dpr;
  const dur = state.duration;
  const xOf = (t) => padL + (clamp(t, 0, dur) / dur) * plotW;

  c.fillStyle = "#1b212a";
  c.fillRect(padL, barY, plotW, barH);
  for (const seg of state.trackStatus) {
    const t0 = clamp(seg.t, 0, dur);
    const t1 = clamp(seg.t + (seg.dur || 0), 0, dur);
    if (t1 <= t0) continue;
    const st = statusStyle(seg);
    c.fillStyle = st.color;
    c.globalAlpha = 0.85;
    c.fillRect(xOf(t0), barY, Math.max(1, xOf(t1) - xOf(t0)), barH);
    c.globalAlpha = 1;
  }
  c.strokeStyle = "#2b3444";
  c.strokeRect(padL, barY, plotW, barH);

  c.font = `${11 * dpr}px ui-monospace, monospace`;
  c.fillStyle = "#8b95a7";
  c.textAlign = "center";
  c.textBaseline = "top";
  const step = Math.max(60, niceStep(dur / 10));
  for (let v = 0; v <= dur; v += step) c.fillText(fmtClock(v), xOf(v), barY + barH + 6);

  // メッセージ
  const listY = barY + barH + 34 * dpr;
  const rowH = 20 * dpr;
  const msgs = state.raceControl.filter((m) => m.t >= -60 && m.t <= dur + 60);
  const rows = Math.floor((h - listY - 10 * dpr) / rowH);
  const near = msgs
    .map((m) => ({ m, d: Math.abs(m.t - state.t) }))
    .sort((a, b) => a.d - b.d)
    .slice(0, Math.max(1, rows))
    .sort((a, b) => a.m.t - b.m.t);
  c.textAlign = "left";
  c.textBaseline = "middle";
  for (let i = 0; i < near.length; i++) {
    const m = near[i].m;
    const y = listY + i * rowH;
    c.fillStyle = "#6d7787";
    c.fillText(fmtClock(m.t), padL, y);
    c.fillStyle = FLAG_COLORS[m.flag] || "#cfd6e2";
    c.fillText(m.flag || "INFO", padL + 62 * dpr, y);
    c.fillStyle = "#cfd6e2";
    c.fillText(clipText(c, m.text || m.category || "", plotW - 170 * dpr), padL + 170 * dpr, y);
  }
}

function clipText(c, text, maxW) {
  if (c.measureText(text).width <= maxW) return text;
  let s = text;
  while (s.length > 1 && c.measureText(s + "…").width > maxW) s = s.slice(0, -1);
  return s + "…";
}

function blitChart() {
  const canvas = $("#chartCanvas");
  if (!canvas) return;
  const src = state.chartCanvas;
  if (!src) return;
  canvas.width = src.width;
  canvas.height = src.height;
  const c = canvas.getContext("2d");
  c.drawImage(src, 0, 0);

  if (state.tab === "telemetry" || state.tab === "control") {
    const dpr = window.devicePixelRatio || 1;
    const padL = CHART_PAD.l, padR = CHART_PAD.r;
    const plotW = src.width - padL - padR;
    const dur = state.duration;
    const x = padL + (clamp(state.t, 0, dur) / dur) * plotW;
    c.strokeStyle = "#e10600";
    c.lineWidth = 1.5 * dpr;
    c.beginPath();
    c.moveTo(x, CHART_PAD.t);
    c.lineTo(x, src.height - CHART_PAD.b);
    c.stroke();
  }
}

/* ------------------------------------------------------------- panels */

function renderControlLog() {
  const ul = $("#controlLog");
  const items = state.raceControl.map((m) => {
    const flag = m.flag || m.category || "INFO";
    const color = FLAG_COLORS[flag] || "#8b95a7";
    return `<li class="future" data-t="${m.t.toFixed(2)}">
      <span class="t">${fmtRelClock(m.t)}</span>
      <span class="flag" style="color:${color}">${flag}</span>
      <span>${escapeHtml(m.text || m.category || "")}</span>
    </li>`;
  });
  ul.innerHTML = items.join("");
  logBoundary = 0;
  updateControlLogHighlight();
}

// The log rows are sorted by time, so the boundary between "already happened"
// and "not yet" moves monotonically with the playhead.  Only the rows between
// the old and the new boundary need a class change.
let logBoundary = 0;

function updateControlLogHighlight() {
  if (state.tab !== "control") return;
  const items = $("#controlLog").children;
  const n = items.length;
  if (!n) return;
  let lo = 0;
  let hi = n;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (Number(items[mid].dataset.t) <= state.t) lo = mid + 1;
    else hi = mid;
  }
  if (lo === logBoundary) return;
  const from = Math.min(lo, logBoundary);
  const to = Math.max(lo, logBoundary);
  for (let i = from; i < to; i++) items[i].classList.toggle("future", i >= lo);
  logBoundary = lo;
}

function renderStatusTimeline() {
  const el = $("#statusTimeline");
  const segs = state.trackStatus.filter((s) => s.t + (s.dur || 0) >= 0);
  const total = segs.reduce((a, s) => a + (s.dur || 0), 0) || 1;
  el.innerHTML = segs
    .map((s) => {
      const st = statusStyle(s);
      const w = Math.max(0.15, ((s.dur || 0) / total) * 100);
      return `<div style="width:${w.toFixed(3)}%;background:${st.color}" title="${fmtClock(s.t)} ${st.label}"></div>`;
    })
    .join("");
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"]/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[ch]));
}

function renderChips() {
  const wrap = $("#driverChips");
  wrap.innerHTML = state.drivers
    .map((d) => {
      const on = state.selected.has(d.abbr);
      return `<button class="chip ${on ? "on" : ""}" data-abbr="${d.abbr}" style="--team:${d.teamColor}">
        <span class="dot" style="background:${d.teamColor}"></span>${d.abbr}
      </button>`;
    })
    .join("");
}

/* ------------------------------------------------------------- HUD */

function updateHud() {
  const rows = state.standingsCache || [];
  // the race lap is the highest lap number any car has reached.  Sampling
  // clamps to the last known keyframe, so this keeps working after the last
  // car has crossed the line and nothing is "present" any more.
  let lap = 0;
  for (const r of rows) {
    if (r.st === 0) continue;
    if (r.lap > lap) lap = r.lap;
  }
  $("#clockLap").textContent = `LAP ${lap} / ${state.meta.totalLaps}`;
  $("#clockTime").textContent = fmtClock(state.t);
  $("#clockTotal").textContent = fmtClock(state.duration);

  const sc = $("#scrub");
  if (!state.scrubDragging) sc.value = String((state.t / state.duration) * 100);

  const seg = statusAt(state.t);
  const badge = $("#statusBadge");
  if (seg) {
    const st = statusStyle(seg);
    badge.textContent = st.label;
    badge.dataset.state = st.key;
  } else {
    badge.textContent = "—";
    badge.dataset.state = "none";
  }

  const key = seg ? statusStyle(seg).key : "none";
  const scActive = key === "safety_car" || key === "virtual_safety_car";
  $("#scIndicator").classList.toggle("on", scActive);
  $("#scIndicator").textContent = key === "virtual_safety_car" ? "VSC" : "SC";
}

/* --------------------------------------------------------- animation */

let lastFrame = 0;
let boardAt = 0;

function frame(now) {
  if (!state.data) {
    requestAnimationFrame(frame);
    return;
  }
  const dt = Math.min(0.1, (now - lastFrame) / 1000 || 0);
  lastFrame = now;

  if (state.playing) {
    state.t += dt * state.speed;
    if (state.t >= state.duration) {
      state.t = state.duration;
      setPlaying(false);
    }
  }

  state.standingsCache = computeStandings(state.t);
  renderTrack();
  updateHud();

  if (now - boardAt > 120) {
    boardAt = now;
    renderBoard(state.standingsCache);
    updateControlLogHighlight();
  }
  // Charts are pre-rendered offscreen and only rebuilt when marked dirty.  The
  // telemetry and control charts carry a moving time cursor, so they are
  // re-blitted every frame.
  if (state.chartDirty) {
    buildChart();
    state.chartDirty = false;
    blitChart();
  } else if (state.tab === "telemetry" || state.tab === "control") {
    blitChart();
  }

  requestAnimationFrame(frame);
}

/* ------------------------------------------------------------ actions */

function setPlaying(on) {
  state.playing = on;
  $("#btnPlay").textContent = on ? "❚❚ 一時停止" : "▶ 再生";
}

function seek(t) {
  state.t = clamp(t, 0, state.duration);
  state.standingsCache = computeStandings(state.t);
  renderBoard(state.standingsCache);
  updateHud();
  updateControlLogHighlight();
}

function jumpLap(dir) {
  const marks = state.lapMarkers;
  if (!marks.length) return;
  if (dir < 0) {
    for (let i = marks.length - 1; i >= 0; i--) {
      if (marks[i] < state.t - 0.5) return seek(marks[i]);
    }
    return seek(0);
  }
  for (const m of marks) if (m > state.t + 0.5) return seek(m);
}

function selectDriver(abbr, on) {
  if (on) state.selected.add(abbr);
  else state.selected.delete(abbr);
  state.chartDirty = true;
  renderChips();
  renderBoard(state.standingsCache || computeStandings(state.t));
}

/* ------------------------------------------------------------- exports */

const SESSION_EXPORTS = [
  { key: "meta", label: "セッション情報", build: () => [state.meta] },
  { key: "drivers", label: "出走ドライバー / 公式結果", build: () => state.drivers },
  { key: "laps", label: "ラップ（周回ごと）", build: () => flattenByDriver(state.laps) },
  { key: "stints", label: "スティント（連続走行）", build: () => flattenByDriver(state.stints) },
  { key: "pitStops", label: "ピットストップ", build: () => flattenByDriver(state.pitStops) },
  { key: "trackStatus", label: "トラックステータス", build: () => state.trackStatus },
  { key: "raceControl", label: "レースコントロール", build: () => state.raceControl },
  { key: "weather", label: "天気（気温 / 路面 / 風 / 降雨）", build: () => state.weather },
  { key: "telemetry", label: "テレメトリ（選択中ドライバー）", build: () => telemetryRows() },
];

function flattenByDriver(obj) {
  const rows = [];
  for (const [abbr, list] of Object.entries(obj || {})) {
    for (const r of list || []) rows.push(Object.assign({ driver: abbr }, r));
  }
  return rows;
}

function telemetryRows() {
  const dtMain = state.meta.dtMain;
  const dtDetail = state.meta.dtDetail;
  const rows = [];
  for (const abbr of state.selected) {
    const s = state.series[abbr];
    if (!s || !s.speed) continue;
    const nMain = s.x.length;
    for (let i = 0; i < s.speed.length; i++) {
      const t = Number((i * dtDetail).toFixed(2));
      const m = clamp(Math.round(t / dtMain), 0, Math.max(0, nMain - 1));
      rows.push({
        driver: abbr, t, lap: s.lap[m], rel: s.rel[m], x: s.x[m], y: s.y[m], status: s.st[m],
        speed: s.speed[i], gear: s.gear[i], drs: s.drs[i], throttle: s.throttle[i], brake: s.brake[i],
      });
    }
  }
  return rows;
}

function csvText(rows) {
  if (!rows.length) return "";
  const cols = [];
  for (const r of rows) {
    for (const k of Object.keys(r)) if (!cols.includes(k)) cols.push(k);
  }
  const cell = (v) => {
    if (v === null || v === undefined) return "";
    const s = typeof v === "object" ? JSON.stringify(v) : String(v);
    return /[",\r\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  };
  const lines = [cols.join(",")];
  for (const r of rows) lines.push(cols.map((c) => cell(r[c])).join(","));
  return "\uFEFF" + lines.join("\r\n");   // BOM 付き = Excel で文字化けしない
}

function downloadText(filename, text) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([text], { type: "text/csv;charset=utf-8" }));
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 5000);
}

function renderSessionExports() {
  const box = $("#sessionExportButtons");
  if (!box || !state.meta) return;
  const entry = state.sessions.find((x) => x.slug === state.currentSlug) || {};
  $("#exportSessionFile").textContent = `data/${state.currentSlug}.json.gz`;
  box.innerHTML = SESSION_EXPORTS.map((spec) => {
    let n = 0;
    try { n = (spec.build() || []).length; } catch (err) { n = 0; }
    return `<button class="export-btn" data-key="${spec.key}"${n ? "" : " disabled"}>` +
      `<b>${spec.label}</b><small>${n.toLocaleString()} 行 → ${state.currentSlug}-${spec.key}.csv</small></button>`;
  }).join("") +
    `<a class="export-btn raw" href="./data/${entry.featuresFile || `features/raw/${state.currentSlug}.json.gz`}" download="features-${state.currentSlug}.json.gz">` +
    `<b>このセッションの特徴量テーブル</b><small>JSON (gzip) → 全10卓の生データ</small></a>`;
}

async function renderFeatureTables() {
  const box = $("#featureTables");
  if (!box || box.dataset.loaded === "1") return;
  let schema;
  try {
    schema = await fetchJsonAny("./data/features/schema.json");
  } catch (err) {
    $("#featuresSummary").textContent =
      "特徴量テーブルはまだ生成されていません。build_data.py --bulk の後に build_features.py を実行してください。";
    return;
  }
  state.schema = schema;
  const tables = Object.entries(schema.tables || {});
  const rows = tables.reduce((n, [, t]) => n + (t.rowCount || 0), 0);
  const bytes = tables.reduce((n, [, t]) => n + (t.csvBytes || 0) + (t.parquetBytes || 0), 0);
  $("#featuresSummary").innerHTML =
    `${tables.length} 卓 / ${rows.toLocaleString()} 行 / ${(bytes / 1e6).toFixed(1)} MB（CSV + Parquet 合計）` +
    `<br><code>schema.json</code> には上の表がそのまま書き込まれています。` +
    `<a href="./data/features/schema.json" download="f1-features-schema.json">スキーマをダウンロード</a> · ` +
    `<a href="./data/index.json" download="f1-session-index.json">セッション一覧 (index.json)</a>`;

  box.innerHTML = tables.map(([name, t]) => {
    const links = [
      t.csv ? `<a href="./data/features/${t.csv}" download="${t.csv}">CSV ${(t.csvBytes / 1024).toFixed(0)} KB</a>` : "",
      t.parquet ? `<a href="./data/features/${t.parquet}" download="${t.parquet}">Parquet ${(t.parquetBytes / 1024).toFixed(0)} KB</a>` : "",
    ].filter(Boolean).join(" ");
    const cols = (t.columns || []).map((c) =>
      `<tr><td><code>${c.name}</code></td><td>${c.type}</td><td>${escapeHtml(
        [c.label, c.unit].filter(Boolean).join("（") + (c.unit ? "）" : ""))}</td></tr>`).join("");
    return `<details class="feature-card"><summary>` +
      `<b>${escapeHtml(t.label || name)}</b> <code>${name}</code>` +
      `<span class="feature-meta">${(t.rowCount || 0).toLocaleString()} 行 · primary key: ${(t.primaryKey || []).join(", ")}</span>` +
      `<span class="feature-links">${links}</span></summary>` +
      `<p class="feature-desc">${escapeHtml(t.description || "")}</p>` +
      `<table class="schema-table"><thead><tr><th>列</th><th>型</th><th>意味 / 単位</th></tr></thead>` +
      `<tbody>${cols}</tbody></table></details>`;
  }).join("");
  box.dataset.loaded = "1";
}

function setTab(tab) {
  state.tab = tab;
  $$("#analysisTabs button").forEach((b) => b.classList.toggle("active", b.dataset.tab === tab));
  $("#logWrap").hidden = tab !== "control";
  $("#exportWrap").hidden = tab !== "export";
  $("#chartWrap").hidden = tab === "export";
  if (tab === "control") renderControlLog();
  if (tab === "export") { renderSessionExports(); renderFeatureTables(); }
  state.chartDirty = true;
}

function buildLapMarkers() {
  //  race 終盤で先頭のマシンを基準にラップ境界を作る
  let best = null;
  const tRef = state.duration * 0.9;
  for (const d of state.drivers) {
    const s = state.series[d.abbr];
    if (!s || s.t1 < tRef) continue;
    const p = progressOf(s, tRef);
    if (!best || p.progress > best.progress) best = { abbr: d.abbr, progress: p.progress };
  }
  if (!best) return;
  const rows = state.laps[best.abbr] || [];
  state.lapMarkers = rows.map((r) => r.t0).filter((v) => Number.isFinite(v)).sort((a, b) => a - b);
}

/* ------------------------------------------------------------- catalog */

const TYPE_LABELS = { R: "決勝", Q: "予選", S: "スプリント", SQ: "スプリント予選" };

function filteredSessions() {
  const { year, type } = state.filters;
  return state.sessions.filter((s) =>
    (!year || String(s.year) === year) && (!type || s.sessionType === type));
}

function renderCatalog() {
  const list = filteredSessions();
  state.catalog = list;
  const sel = $("#sessionSelect");

  const groups = [];
  const byKey = new Map();
  for (const s of list) {
    const key = `${s.year}-${s.round}`;
    let g = byKey.get(key);
    if (!g) { g = { head: s, rows: [] }; byKey.set(key, g); groups.push(g); }
    g.rows.push(s);
  }

  sel.innerHTML = groups.map((g) => {
    const h = g.head;
    const label = `${h.year} R${String(h.round).padStart(2, "0")}  ${h.country}（${h.location}） ${h.date}`;
    const opts = g.rows.map((s) => {
      const flags = [`${s.drivers}台`, `${s.totalLaps}周`, `${s.sizeKB} KB`];
      if (!s.hasPosition) flags.push("位置データなし");
      return `<option value="${s.slug}">${TYPE_LABELS[s.sessionType] || s.sessionType} — ${flags.join(" / ")}</option>`;
    }).join("");
    return `<optgroup label="${escapeHtml(label)}">${opts}</optgroup>`;
  }).join("");

  const years = [...new Set(state.sessions.map((s) => s.year))].sort((a, b) => b - a);
  const yearSel = $("#yearFilter");
  if (yearSel.dataset.ready !== "1") {
    yearSel.innerHTML = `<option value="">全シーズン</option>` +
      years.map((y) => `<option value="${y}">${y}年</option>`).join("");
    yearSel.dataset.ready = "1";
  }
  yearSel.value = state.filters.year;
  $("#typeFilter").value = state.filters.type;

  const sizeMB = list.reduce((n, s) => n + (s.sizeKB || 0), 0) / 1024;
  $("#catalogCount").textContent =
    `${list.length} / ${state.sessions.length} セッション · 転送量 ${sizeMB.toFixed(0)} MB`;

  if (state.currentSlug && list.some((s) => s.slug === state.currentSlug)) {
    sel.value = state.currentSlug;
  } else if (list.length) {
    sel.value = list[list.length - 1].slug;   // 既定は最新のセッション
  }
  return list;
}

function stepSession(dir) {
  const list = state.catalog.length ? state.catalog : filteredSessions();
  if (!list.length) return;
  let i = list.findIndex((s) => s.slug === state.currentSlug);
  if (i < 0) i = list.length - 1;
  const next = list[clamp(i + dir, 0, list.length - 1)];
  if (next && next.slug !== state.currentSlug) loadSession(next.slug).catch(showError);
}

/* gzip 配信（.json.gz）を優先し、無い場合は素の .json にフォールバックする */
async function fetchJsonAny(url) {
  let res = await fetch(url);
  if (!res.ok && url.endsWith(".json.gz")) {
    res = await fetch(url.slice(0, -3));
  }
  if (!res.ok) throw new Error(`データが読み込めません (${res.status})`);
  if (!url.endsWith(".gz")) return await res.json();
  if (typeof DecompressionStream === "undefined") {
    throw new Error("このブラウザは gzip 展開（DecompressionStream）に対応していません。Chrome 100 以降 / Safari 16.4 以降で開いてください。");
  }
  const buf = new Uint8Array(await res.arrayBuffer());
  const stream = new Blob([buf]).stream().pipeThrough(new DecompressionStream("gzip"));
  return JSON.parse(await new Response(stream).text());
}

async function loadIndex() {
  const idx = await fetchJsonAny("./data/index.json");
  state.sessions = idx.sessions || [];
  if (!state.sessions.length) {
    $("#eventName").textContent = "データがありません";
    $("#eventSub").textContent = "build_data.py --bulk でデータを生成してください。";
    return null;
  }
  const list = renderCatalog();
  return list[list.length - 1];
}

async function loadSession(slug) {
  const s = state.sessions.find((x) => x.slug === slug);
  if (!s) throw new Error(`セッション ${slug} が見つかりません`);
  const data = await fetchJsonAny(`./data/${s.file}`);
  state.currentSlug = s.slug;
  $("#sessionSelect").value = s.slug;

  state.data = data;
  state.meta = data.meta;
  state.track = data.track || null;
  state.drivers = data.drivers;
  state.driverByAbbr = new Map(data.drivers.map((d) => [d.abbr, d]));
  state.series = data.series;
  state.laps = data.laps;
  state.stints = data.stints;
  state.pitStops = data.pitStops || {};
  state.weather = data.weather || [];
  state.duration = data.meta.duration;

  const hasPos = Boolean(state.track);
  $("#noPositionNotice").hidden = hasPos;
  $("#trackCanvas").hidden = !hasPos;
  $("#trackHint").hidden = !hasPos;
  // the canvas keeps the wrapper's height; without it the notice would collapse
  // the box and overflow onto the transport controls below
  $("#trackCanvas").parentElement.classList.toggle("no-track", !hasPos);
  state.tf = null;
  state.trackCacheKey = "";
  state.trackCanvasCache = null;

  // trackStatus に終了時刻がない行は次の開始まで
  const ts = (data.trackStatus || []).slice().sort((a, b) => a.t - b.t);
  for (let i = 0; i < ts.length; i++) {
    const next = ts[i + 1] ? ts[i + 1].t : state.duration;
    ts[i].dur = Math.max(0, Math.min(next, state.duration) - Math.max(ts[i].t, 0));
  }
  state.trackStatus = ts;
  state.raceControl = (data.raceControl || []).slice().sort((a, b) => a.t - b.t);

  const label = TYPE_LABELS[data.meta.sessionType] || data.meta.sessionType;
  $("#eventName").textContent = data.meta.eventName;
  $("#eventSub").textContent =
    `${data.meta.year} R${String(data.meta.round).padStart(2, "0")} · ${label} · ` +
    `${data.meta.location}（${data.meta.country}） · ${data.meta.date} · ${data.meta.totalLaps} Laps`;
  const pit = data.meta.pitLaneMeters ? ` / pit ${data.meta.pitLaneMeters.toFixed(0)} m` : "";
  $("#buildInfo").textContent =
    `exported ${data.meta.generatedAt} · ${s.sizeKB} KB (gzip) / ${(JSON.stringify(data).length / 1e6).toFixed(2)} MB · ` +
    (hasPos ? `track ${data.track.lengthMeters.toFixed(0)} m${pit}` : "位置データなし") +
    ` · ${data.drivers.length} cars · ${data.meta.source || "FastF1"}`;
  $("#clockTotal").textContent = fmtClock(state.duration);

  buildLapMarkers();
  // 上位3台をデフォルト選択
  state.selected = new Set(
    state.drivers
      .filter((d) => d.finish && d.finish <= 3)
      .sort((a, b) => a.finish - b.finish)
      .map((d) => d.abbr)
  );
  if (!state.selected.size) state.selected = new Set([state.drivers[0].abbr]);

  renderChips();
  renderStatusTimeline();
  renderSessionExports();
  setTab(state.tab === "export" ? "export" : "lapTimes");
  state.standingsCache = computeStandings(0);
  renderBoard(state.standingsCache);
  state.chartDirty = true;
  seek(0);
}

function wireEvents() {
  $("#btnPlay").addEventListener("click", () => setPlaying(!state.playing));
  $("#btnRestart").addEventListener("click", () => { seek(0); setPlaying(true); });
  $("#btnPrevLap").addEventListener("click", () => jumpLap(-1));
  $("#btnNextLap").addEventListener("click", () => jumpLap(1));
  $("#speedButtons").addEventListener("click", (e) => {
    const b = e.target.closest("button[data-speed]");
    if (!b) return;
    state.speed = Number(b.dataset.speed);
    $$("#speedButtons button").forEach((x) => x.classList.toggle("active", x === b));
  });

  const scrub = $("#scrub");
  scrub.addEventListener("pointerdown", () => { state.scrubDragging = true; });
  window.addEventListener("pointerup", () => { state.scrubDragging = false; });
  scrub.addEventListener("input", () => seek((Number(scrub.value) / 100) * state.duration));

  $("#sessionSelect").addEventListener("change", (e) => loadSession(e.target.value).catch(showError));
  $("#btnPrevSession").addEventListener("click", () => stepSession(-1));
  $("#btnNextSession").addEventListener("click", () => stepSession(1));
  $("#yearFilter").addEventListener("change", (e) => {
    state.filters.year = e.target.value;
    const list = renderCatalog();
    if (list.length) loadSession(list[list.length - 1].slug).catch(showError);
  });
  $("#typeFilter").addEventListener("change", (e) => {
    state.filters.type = e.target.value;
    const list = renderCatalog();
    if (list.length) loadSession(list[list.length - 1].slug).catch(showError);
  });
  $("#sessionExportButtons").addEventListener("click", (e) => {
    const b = e.target.closest("button[data-key]");
    if (!b || b.disabled) return;
    const spec = SESSION_EXPORTS.find((x) => x.key === b.dataset.key);
    if (!spec) return;
    let rows = [];
    try { rows = spec.build() || []; } catch (err) { showError(err); return; }
    if (!rows.length) return;
    downloadText(`${state.currentSlug}-${spec.key}.csv`, csvText(rows));
    b.querySelector("small").textContent = `${rows.length.toLocaleString()} 行を書き出しました`;
  });
  $("#analysisTabs").addEventListener("click", (e) => {
    const b = e.target.closest("button[data-tab]");
    if (b) setTab(b.dataset.tab);
  });
  $("#driverChips").addEventListener("click", (e) => {
    const b = e.target.closest(".chip");
    if (b) selectDriver(b.dataset.abbr, !state.selected.has(b.dataset.abbr));
  });
  $("#chipTop").addEventListener("click", () => {
    state.selected = new Set(
      state.drivers.filter((d) => d.finish && d.finish <= 3).map((d) => d.abbr)
    );
    state.chartDirty = true;
    renderChips();
  });
  $("#chipAll").addEventListener("click", () => {
    state.selected = new Set(state.drivers.map((d) => d.abbr));
    state.chartDirty = true;
    renderChips();
  });
  $("#chipNone").addEventListener("click", () => {
    state.selected = new Set();
    state.chartDirty = true;
    renderChips();
  });

  $("#leaderTable").addEventListener("click", (e) => {
    const tr = e.target.closest("tr[data-abbr]");
    if (tr) selectDriver(tr.dataset.abbr, !state.selected.has(tr.dataset.abbr));
  });

  $("#controlLog").addEventListener("click", (e) => {
    const li = e.target.closest("li[data-t]");
    if (li) seek(Number(li.dataset.t));
  });

  $("#trackCanvas").addEventListener("click", (e) => {
    if (!state.tf) return;
    const dpr = window.devicePixelRatio || 1;
    const rect = e.currentTarget.getBoundingClientRect();
    const px = (e.clientX - rect.left) * dpr;
    const py = (e.clientY - rect.top) * dpr;
    let best = null, bestD = 26 * dpr;
    for (const row of state.standingsCache || []) {
      if (!row.present) continue;
      const p = sampleSeries(row.s, state.t);
      const d = Math.hypot(state.tf.px(p.x) - px, state.tf.py(p.y) - py);
      if (d < bestD) { bestD = d; best = row.d.abbr; }
    }
    if (best) selectDriver(best, !state.selected.has(best));
  });

  $("#statusTimeline").addEventListener("click", (e) => {
    const el = e.target.closest("div");
    if (!el) return;
    const m = (el.title || "").split(" ")[0];
    const parts = m.split(":").map(Number);
    const sec = parts.length === 3 ? parts[0] * 3600 + parts[1] * 60 + parts[2] : parts[0] * 60 + parts[1];
    seek(sec);
  });

  window.addEventListener("keydown", (e) => {
    if (e.target.tagName === "INPUT" && e.target.type === "text") return;
    if (e.code === "Space") { e.preventDefault(); setPlaying(!state.playing); }
    else if (e.code === "ArrowRight") { e.preventDefault(); jumpLap(1); }
    else if (e.code === "ArrowLeft") { e.preventDefault(); jumpLap(-1); }
    else if (e.key === "Escape") { state.selected.clear(); state.chartDirty = true; renderChips(); }
  });

  let resizeTimer = 0;
  window.addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => {
      state.trackCacheKey = "";
      state.chartDirty = true;
    }, 150);
  });
}

function showError(err) {
  console.error(err);
  $("#eventName").textContent = "読み込みエラー";
  $("#eventSub").textContent = String(err && err.message ? err.message : err);
}

async function main() {
  wireEvents();
  try {
    const first = await loadIndex();
    if (!first) return;
    await loadSession(first.slug);
    requestAnimationFrame(frame);
  } catch (err) {
    showError(err);
  }
}

main();
