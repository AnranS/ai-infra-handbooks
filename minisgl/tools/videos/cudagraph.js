// 视频六：CUDA Graph（第 18 章）
const SEGMENTS = [
  "CUDA Graph：把一整次 decode 前向录制下来，之后一次 replay，就能发射全部 kernel。",
  "28 层的模型 decode 一步，要启动几百个 kernel。小批量时，每个 kernel 只算几微秒，而 CPU 发射一个 kernel 也要几微秒。GPU 大部分时间在等 CPU。",
  "有了 CUDA Graph，CPU 只需发射一次 replay。GPU 上的 kernel 一个接一个，中间没有空隙。",
  "代价是：录下来的 kernel，读写的地址全都固定了。所以每一轮的输入和注意力元数据，都要先拷进固定的缓冲区，再 replay。",
  "每个批大小录一个 graph：1、2、4，然后是 8 的倍数。实际的 batch 向上补齐：3 个请求，就补一个 dummy 请求，凑成 4 个。",
  "dummy 请求在 page table 里有专门的一行，指向 KV 池里专门的一页。它算出的 KV 写进这一页，它的 logits 直接丢掉。",
  "录制从最大的批大小开始，后面的小 graph 复用同一个内存池，显存由最大的那个决定。",
  "最常见的 bug 是漏拷某个输入。比如 replay 前忘了拷 positions，graph 里的 RoPE 读到的，还是缓冲区里的旧位置。",
  "结果从第二个 token 开始就错了，而且没有任何报错。本书用 EmulatedGraph 在 CPU 上仿真这种行为，测试能抓住它。",
  "录制、补齐、拷入缓冲区、replay，这就是 GraphRunner。第 18 章逐行实现。",
];
const SEG_MIN = [4, 0, 0, 0, 0, 0, 0, 0, 0, 7];

// ---------- 发射开销 ----------
function launchLane(x0, y, n, play, graph) {
  const k = 18, gapCpu = 26, kw = 14;  // 每个 kernel 在 GPU 上 14px；CPU 发射一个 26px
  text(x0 - 16, y + 30, "CPU", 18, C.muted, "right", 700);
  text(x0 - 16, y + 88, "GPU", 18, C.muted, "right", 700);
  ctx.save(); ctx.fillStyle = "#eeeef0"; ctx.fillRect(x0, y + 8, 1000, 36); ctx.fillRect(x0, y + 66, 1000, 36); ctx.restore();
  if (!graph) {
    for (let i = 0; i < n; i++) {
      const s = i * gapCpu; if (s > play) break;
      box(x0 + s + 1, y + 8, gapCpu - 3, 36, "", C.blue, { r: 4, solid: true, a: 0.85 });
      const gs = s + gapCpu; if (gs > play) continue;
      box(x0 + gs + 2, y + 66, Math.min(kw, play - gs), 36, "", C.purple, { r: 3, solid: true });
    }
  } else {
    const rw = 60;
    if (play > 0) box(x0 + 1, y + 8, Math.min(rw, play) - 2, 36, "", C.blue, { r: 4, solid: true });
    if (play > 20) text(x0 + 30, y + 32, "replay", 13, "#fff", "center", 700, true);
    for (let i = 0; i < n * 2; i++) {
      const s = rw + i * (kw + 1); if (s > play || s > 1000 - kw) break;
      box(x0 + s, y + 66, Math.min(kw, play - s), 36, "", C.purple, { r: 3, solid: true });
    }
  }
}
function overhead() {
  heading("小批量 decode：GPU 在等 CPU 发射 kernel");
  const x0 = 170;
  text(x0, 130, "逐个发射 kernel", 20, C.fg, "left", 700, false, seg(1, 0, 0.1));
  launchLane(x0, 145, 38, seg(1, 0.05, 0.6) * 1000, false);
  const a = seg(1, 0.55, 0.7);
  text(x0 + 500, 280, "蓝：CPU 发射一个 kernel（几微秒）   紫：GPU 执行（也只有几微秒）", 17, C.muted, "center", 600, false, a);
  if (after(2)) {
    text(x0, 350, "CUDA Graph：一次 replay", 20, C.fg, "left", 700, false, seg(2, 0, 0.1));
    launchLane(x0, 365, 38, seg(2, 0.05, 0.5) * 1000, true);
    text(x0 + 500, 520, "同样的 kernel，GPU 几乎没有空隙", 20, C.green, "center", 700, false, seg(2, 0.55, 0.7));
  }
}

// ---------- 固定缓冲区 ----------
function buffers() {
  heading("录下来的地址是固定的");
  const a = seg(3, 0, 0.12);
  box(700, 120, 500, 420, "录好的 graph", C.purple, { a, top: true, fs: 20 });
  ["embedding", "RoPE", "attention", "…× 28 层", "lm_head"].forEach((s, i) =>
    box(740, 180 + i * 66, 180, 50, s, C.purple, { a: seg(3, 0.05 + i * 0.04, 0.15 + i * 0.04), solid: true, fs: 16 }));
  const bufs = [["input_ids", "0x7f3a…00"], ["positions", "0x7f3a…40"], ["out_loc", "0x7f3a…80"], ["page table", "0x7f3b…00"], ["cache_seqlens", "0x7f3b…80"]];
  const b = seg(3, 0.25, 0.4);
  text(380, 150, "固定缓冲区（地址不变）", 18, C.fg, "center", 700, false, b);
  bufs.forEach(([n, addr], i) => {
    const y = 170 + i * 66;
    box(290, y, 180, 50, n, C.teal, { a: b, fs: 16, sub: addr, subMono: true });
    arrow(470, y + 25, 738, [205, 271, 337, 337, 337][i], C.teal, b, { lw: 1.5, dash: true, a: 0.6 * b });
  });
  const c = seg(3, 0.5, 0.7);
  text(120, 150, "本轮的 batch", 18, C.fg, "center", 700, false, c);
  bufs.forEach((_, i) => {
    const y = 170 + i * 66;
    chip(120, y + 25, "新值", C.orange, { a: c, w: 110, h: 40, mono: false, fs: 15 });
    arrow(176, y + 25, 286, y + 25, C.orange, seg(3, 0.6 + i * 0.05, 0.75 + i * 0.05), { lw: 2 });
  });
  text(230, 150, "① 拷入", 18, C.orange, "center", 700, false, seg(3, 0.7, 0.8));
  text(585, 150, "② replay", 18, C.purple, "center", 700, false, seg(3, 0.85, 0.95));
}

// ---------- 补齐 ----------
function padding() {
  heading("每个批大小一个 graph，batch 向上补齐");
  const bss = [1, 2, 4, 8, 16, 24, 32, "…", 160];
  const a = seg(4, 0, 0.3);
  text(100, 150, "已录制的 graph：", 19, C.fg, "left", 700, false, a);
  const pick = seg(4, 0.6, 0.72) > 0.5;
  bss.forEach((b, i) => {
    const hit = pick && b === 4;
    chip(310 + i * 86, 142, String(b), hit ? C.green : C.purple, { a: seg(4, 0.03 * i, 0.1 + 0.03 * i), w: 70, h: 44, solid: hit, fs: 18, glow: hit });
  });
  const r = seg(4, 0.35, 0.5);
  text(100, 280, "本轮 batch：", 19, C.fg, "left", 700, false, r);
  ["r0", "r1", "r2"].forEach((s, i) => chip(300 + i * 90, 272, s, C.blue, { a: r, w: 76, h: 48, fs: 18 }));
  const d = seg(4, 0.5, 0.62);
  chip(300 + 3 * 90, 272, "dummy", C.gray, { a: d, w: 76, h: 48, fs: 15, dash: true });
  text(690, 280, "3 → 补齐到 4", 20, C.green, "left", 700, false, d);
  const e = seg(4, 0.72, 0.85);
  arrow(435, 244, 480, 170, C.green, e, { lw: 2.5, dash: true });
  text(640, 380, "间隔小的档位多（1、2、4、8），大批量时 kernel 已较饱，补齐浪费比例也小", 17, C.muted, "center", 400, false, seg(4, 0.8, 0.95));
  if (!after(5)) return;
  // dummy 的去向
  const f = seg(5, 0.05, 0.25);
  text(100, 450, "page table", 17, C.fg, "left", 700, false, f);
  ["r0", "r1", "r2", "dummy"].forEach((s, i) => {
    chip(130 + 0, 480 + i * 34, s, i === 3 ? C.gray : C.blue, { a: f, w: 64, h: 28, fs: 13 });
    for (let j = 0; j < 5; j++) chip(190 + j * 34, 480 + i * 34, i === 3 ? "D" : String(10 * i + j + 3), i === 3 ? C.gray : C.blue, { a: f, w: 30, h: 28, fs: 12, dash: i === 3 });
  });
  const g = seg(5, 0.3, 0.5);
  box(420, 560, 160, 44, "KV 池的 dummy 页", C.gray, { a: g, fs: 15, dash: true });
  arrow(360, 582, 418, 582, C.gray, g, { lw: 2 });
  const h = seg(5, 0.55, 0.8);
  text(720, 470, "logits [4, vocab]", 18, C.fg, "left", 700, true, h);
  [0, 1, 2, 3].forEach((i) => chip(740 + i * 60, 510, i === 3 ? "丢弃" : `r${i}`, i === 3 ? C.red : C.blue, { a: h, w: 54, h: 34, fs: 14, mono: i !== 3, dash: i === 3 }));
  text(720, 575, "logits[:batch.size] → 只保留 3 行", 17, C.muted, "left", 600, true, h);
}

// ---------- 录制顺序 ----------
function captureOrder() {
  heading("从最大的批大小开始录");
  const list = [160, 152, 144, "…", 16, 8, 4, 2, 1];
  const p = seg(6, 0.05, 0.7) * list.length;
  box(120, 360, 1040, 70, "", C.purple, { a: seg(6, 0, 0.1), r: 8 });
  text(140, 345, "同一个内存池（pool）", 18, C.fg, "left", 700, false, seg(6, 0, 0.1));
  list.forEach((b, i) => {
    const a = clamp(p - i);
    chip(170 + i * 115, 200, `bs=${b}`, C.purple, { a, w: 96, h: 44, solid: i === 0, fs: 15 });
    const w = typeof b === "number" ? 1000 * b / 160 : 0;
    if (a > 0 && w) box(140 + 0, 370, w, 50, "", C.purple, { a: a * (i === 0 ? 0.85 : 0.18), solid: true, r: 6 });
  });
  text(640, 300, "先录最大的，占住显存；后面更小的 graph 都放得进这块内存", 19, C.fg, "center", 600, false, seg(6, 0.6, 0.8));
  text(640, 490, "默认上限：显存 80 GB 以上的卡 256，否则 160", 17, C.muted, "center", 400, false, seg(6, 0.75, 0.9));
}

// ---------- 漏拷的 bug ----------
function bug() {
  heading("漏拷一个输入：没有报错，结果却错了");
  const a = seg(7, 0, 0.15);
  text(120, 150, "本轮真实的 positions", 18, C.fg, "left", 700, false, a);
  ["6", "9", "12", "0"].forEach((s, i) => chip(150 + i * 70, 190, s, C.orange, { a, w: 60, h: 40, fs: 18 }));
  const b = seg(7, 0.15, 0.3);
  text(620, 150, "graph 读的 positions 缓冲区", 18, C.fg, "left", 700, false, b);
  ["0", "0", "0", "0"].forEach((s, i) => chip(650 + i * 70, 190, s, C.teal, { a: b, w: 60, h: 40, fs: 18 }));
  const x = seg(7, 0.35, 0.5);
  arrow(440, 190, 620, 190, C.red, x, { lw: 2.5, dash: true, label: "忘了拷", ly: -14 });
  if (x > 0.5) text(530, 205, "✗", 30, C.red, "center", 800, false, x);
  text(640, 290, "RoPE 用的是旧位置 → 注意力算错", 20, C.red, "center", 700, false, seg(7, 0.6, 0.8));
  if (!after(8)) return;
  const good = [12095, 13, 576, 6722, 315, 15344], bad = [12095, 1103, 9736, 6686, 374, 264];
  const c = seg(8, 0, 0.15);
  text(120, 380, "正确", 18, C.green, "left", 700, false, c);
  text(120, 450, "漏拷", 18, C.red, "left", 700, false, c);
  good.forEach((t, i) => {
    const s = seg(8, 0.05 + i * 0.05, 0.12 + i * 0.05);
    chip(230 + i * 118, 372, String(t), C.green, { a: s, w: 100, h: 40, fs: 16 });
    chip(230 + i * 118, 442, String(bad[i]), i === 0 ? C.green : C.red, { a: s, w: 100, h: 40, fs: 16, solid: i > 0 });
  });
  text(230, 500, "↑ prefill 不走 graph，第一个 token 仍然对", 15, C.muted, "left", 400, false, seg(8, 0.35, 0.45));
  const d = seg(8, 0.55, 0.7);
  box(700, 480, 470, 90, "", C.green, { a: d });
  text(935, 518, "EmulatedGraph：CPU 上仿真", 19, C.fg, "center", 700, false, d);
  text(935, 550, "graph 只能看到缓冲区里的东西", 16, C.muted, "center", 400, false, d);
}

function outro() {
  const a = seg(9, 0, 0.2);
  heading("GraphRunner 的四步", a);
  [["录制", "_capture_graphs", C.purple], ["补齐", "pad_batch", C.gray], ["拷入缓冲区", "prepare_for_replay", C.teal], ["replay", "replay", C.green]].forEach(([n, f, c], i) => {
    box(100 + i * 275, 250, 245, 150, n, c, { a: seg(9, 0.05 + i * 0.1, 0.2 + i * 0.1), sub: f, fs: 30, subMono: true });
    if (i < 3) arrow(345 + i * 275, 325, 375 + i * 275, 325, C.line, seg(9, 0.15 + i * 0.1, 0.25 + i * 0.1));
  });
  text(W / 2, 480, "engine/graph.py · 第 18 章", 22, C.muted, "center", 600, false, seg(9, 0.5, 0.7));
}

function draw() {
  if (!after(1)) { title("CUDA Graph", "录一次，replay 千万次", 1 - seg(0, 0.88, 1)); return; }
  if (!after(3)) { overhead(); return; }
  if (!after(4)) { buffers(); return; }
  if (!after(6)) { padding(); return; }
  if (!after(7)) { captureOrder(); return; }
  if (!after(9)) { bug(); return; }
  outro();
}
