// 教学视频的 canvas 动画引擎。
//
// 每个视频是一个场景文件（tools/videos/<名字>.js），里面定义 draw(t)：给定时间 t（秒）画出一帧。
// 旁白分成若干段，render.py 用 TTS 生成每段的音频、量出时长，通过 window.SEG_DUR 传进来；
// 场景代码用 seg(i) 得到第 i 段内的进度（0～1），用 after(i) 判断是否已经到了第 i 段，
// 于是"之前出现的元素一直保留、当前段的元素逐渐出现"写起来很自然。字幕由引擎统一画在底部。

const W = 1280, H = 720;
const C = {
  bg: "#fafafa", fg: "#1d1d1f", muted: "#6e6e73", line: "#d2d2d7", surface: "#ffffff",
  blue: "#007aff", green: "#28a745", orange: "#ff9500", purple: "#af52de", pink: "#ff2d55",
  teal: "#30b0c7", red: "#ff3b30", yellow: "#e0a800", gray: "#8e8e93",
};
const FONT = '"PingFang SC", "Noto Sans CJK SC", "Noto Sans SC", "Microsoft YaHei", sans-serif';
const MONO = '"JetBrains Mono", "DejaVu Sans Mono", monospace';

let ctx, T0 = [], SEGS = [], TOTAL = 0, NOW = 0;

function setupTimeline() {
  const dur = window.SEG_DUR || SEGMENTS.map(() => 4);
  T0 = []; let t = 0;
  for (const d of dur) { T0.push(t); t += d; }
  T0.push(t); TOTAL = t; SEGS = dur;
}
const clamp = (x, a = 0, b = 1) => Math.max(a, Math.min(b, x));
const ease = (p) => (p < 0.5 ? 2 * p * p : 1 - Math.pow(-2 * p + 2, 2) / 2);
// 第 i 段内的进度；a、b 是段内的起止比例（例如 seg(3, 0.2, 0.6) 在该段 20%～60% 的时间内从 0 走到 1）
function seg(i, a = 0, b = 1) {
  if (i >= SEGS.length) return 0;
  const p = (NOW - T0[i]) / SEGS[i];
  return ease(clamp((p - a) / (b - a)));
}
const after = (i) => i < SEGS.length && NOW >= T0[i];
const during = (i) => after(i) && !after(i + 1);
const mix = (a, b, p) => a + (b - a) * p;

function alpha(color, a) {
  const n = parseInt(color.slice(1), 16);
  return `rgba(${n >> 16},${(n >> 8) & 255},${n & 255},${a})`;
}

function rr(x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y); ctx.arcTo(x + w, y, x + w, y + h, r); ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r); ctx.arcTo(x, y, x + w, y, r); ctx.closePath();
}

// 圆角框：o = {sub, a（透明度）, solid, fs, r, dash, glow}
function box(x, y, w, h, label, color = C.gray, o = {}) {
  const a = o.a ?? 1; if (a <= 0) return;
  ctx.save(); ctx.globalAlpha = a;
  rr(x, y, w, h, o.r ?? 12);
  if (o.glow) { ctx.shadowColor = alpha(color, 0.55); ctx.shadowBlur = 24 * o.glow; }
  ctx.fillStyle = o.solid ? color : mixWhite(color, 0.9);
  ctx.fill(); ctx.shadowBlur = 0;
  ctx.lineWidth = o.lw ?? 2; ctx.strokeStyle = color;
  if (o.dash) ctx.setLineDash([8, 6]);
  ctx.stroke(); ctx.setLineDash([]);
  if (label) {
    const fs = o.fs ?? 22;
    let cy = y + h / 2 + (o.sub ? -fs * 0.35 : fs * 0.35);
    if (o.top) cy = y + fs + 12;
    text(x + w / 2, cy, label, fs, o.solid ? "#fff" : C.fg, "center", 600, o.mono);
    if (o.sub) text(x + w / 2, cy + fs * 1.15, o.sub, fs * 0.72, o.solid ? "rgba(255,255,255,.9)" : C.muted, "center", 400, o.subMono);
  }
  ctx.restore();
}
function mixWhite(color, k) {
  const n = parseInt(color.slice(1), 16);
  const r = Math.round((n >> 16) * (1 - k) + 255 * k), g = Math.round(((n >> 8) & 255) * (1 - k) + 255 * k),
        b = Math.round((n & 255) * (1 - k) + 255 * k);
  return `rgb(${r},${g},${b})`;
}

function text(x, y, s, fs = 20, color = C.fg, align = "center", weight = 400, mono = false, a = 1) {
  if (a <= 0) return;
  ctx.save(); ctx.globalAlpha *= a;
  ctx.font = `${weight} ${fs}px ${mono ? MONO : FONT}`;
  ctx.fillStyle = color; ctx.textAlign = align; ctx.textBaseline = "alphabetic";
  String(s).split("\n").forEach((line, i) => ctx.fillText(line, x, y + i * fs * 1.35));
  ctx.restore();
}

// 箭头，按进度 p 从起点画到终点
function arrow(x1, y1, x2, y2, color = C.gray, p = 1, o = {}) {
  if (p <= 0) return;
  const x = mix(x1, x2, p), y = mix(y1, y2, p);
  ctx.save(); ctx.globalAlpha = o.a ?? 1;
  ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = o.lw ?? 3;
  if (o.dash) ctx.setLineDash([9, 7]);
  ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x, y); ctx.stroke(); ctx.setLineDash([]);
  const ang = Math.atan2(y2 - y1, x2 - x1), L = 13;
  ctx.beginPath(); ctx.moveTo(x, y);
  ctx.lineTo(x - L * Math.cos(ang - 0.4), y - L * Math.sin(ang - 0.4));
  ctx.lineTo(x - L * Math.cos(ang + 0.4), y - L * Math.sin(ang + 0.4)); ctx.closePath(); ctx.fill();
  if (o.label && p > 0.6) text((x1 + x2) / 2 + (o.lx ?? 0), (y1 + y2) / 2 + (o.ly ?? -10), o.label, o.fs ?? 17, color, "center", 600, o.mono, (p - 0.6) / 0.4);
  ctx.restore();
}

// 小方块（token、KV 格子等）
function chip(x, y, s, color = C.blue, o = {}) {
  const w = o.w ?? 46, h = o.h ?? 34, a = o.a ?? 1;
  if (a <= 0) return;
  ctx.save(); ctx.globalAlpha = a;
  rr(x - w / 2, y - h / 2, w, h, o.r ?? 7);
  ctx.fillStyle = o.solid ? color : (o.empty ? "#fff" : mixWhite(color, 0.82)); ctx.fill();
  ctx.lineWidth = 1.6; ctx.strokeStyle = o.empty ? C.line : color;
  if (o.dash) ctx.setLineDash([5, 4]);
  ctx.stroke(); ctx.setLineDash([]);
  if (s !== undefined && s !== "") text(x, y + (o.fs ?? 16) * 0.36, s, o.fs ?? 16, o.solid ? "#fff" : C.fg, "center", 600, o.mono ?? true);
  ctx.restore();
}

// 沿折线移动：返回进度 p 处的坐标
function along(points, p) {
  const segs = [];
  let total = 0;
  for (let i = 1; i < points.length; i++) {
    const d = Math.hypot(points[i][0] - points[i - 1][0], points[i][1] - points[i - 1][1]);
    segs.push(d); total += d;
  }
  let dist = clamp(p) * total;
  for (let i = 0; i < segs.length; i++) {
    if (dist <= segs[i] || i === segs.length - 1) {
      const k = segs[i] ? dist / segs[i] : 0;
      return [mix(points[i][0], points[i + 1][0], k), mix(points[i][1], points[i + 1][1], k)];
    }
    dist -= segs[i];
  }
}

function title(s, sub, a = 1) {
  text(W / 2, 318, s, 50, C.fg, "center", 800, false, a);
  if (sub) text(W / 2, 382, sub, 26, C.muted, "center", 400, false, a);
}

function heading(s, a = 1) {
  text(48, 64, s, 30, C.fg, "left", 700, false, a);
}

// 字幕：当前段的旁白按标点切句，按字数分配时间
function subtitle() {
  let i = SEGS.length - 1;
  while (i > 0 && NOW < T0[i]) i--;
  const narration = SEGMENTS[i] || "";
  ctx.save();
  ctx.font = `500 26px ${FONT}`;
  // 太长的句子在逗号、冒号处再切开，保证一行放得下
  const sentences = [];
  for (const s of narration.split(/(?<=[。！？；])/).map((s) => s.trim()).filter(Boolean)) {
    if (ctx.measureText(s).width <= W - 150) { sentences.push(s); continue; }
    let cur = "";
    for (const part of s.split(/(?<=[，：、])/)) {
      if (cur && ctx.measureText(cur + part).width > W - 150) { sentences.push(cur); cur = part; }
      else cur += part;
    }
    if (cur) sentences.push(cur);
  }
  ctx.restore();
  if (!sentences.length) return;
  const total = sentences.reduce((a, s) => a + s.length, 0);
  let t = T0[i];
  let cur = sentences[sentences.length - 1];
  for (const s of sentences) {
    const d = SEGS[i] * s.length / total;
    if (NOW < t + d) { cur = s; break; }
    t += d;
  }
  cur = cur.replace(/[。；，、]$/, "");
  ctx.save();
  ctx.font = `500 26px ${FONT}`;
  const w = Math.min(ctx.measureText(cur).width + 44, W - 80);
  ctx.fillStyle = "rgba(0,0,0,0.72)";
  rr(W / 2 - w / 2, H - 70, w, 46, 10); ctx.fill();
  ctx.fillStyle = "#fff"; ctx.textAlign = "center";
  ctx.fillText(cur, W / 2, H - 38);
  ctx.restore();
}

function progressBar() {
  ctx.fillStyle = C.line; ctx.fillRect(0, H - 4, W, 4);
  ctx.fillStyle = C.blue; ctx.fillRect(0, H - 4, W * NOW / TOTAL, 4);
}

window.renderFrame = function (t) {
  NOW = t;
  ctx.fillStyle = C.bg; ctx.fillRect(0, 0, W, H);
  ctx.save(); draw(t); ctx.restore();
  subtitle();
  progressBar();
};

window.addEventListener("load", () => {
  const canvas = document.getElementById("c");
  canvas.width = W; canvas.height = H;
  ctx = canvas.getContext("2d");
  setupTimeline();
  window.TOTAL = TOTAL;
  window.sceneReady = true;
  renderFrame(0);
});
