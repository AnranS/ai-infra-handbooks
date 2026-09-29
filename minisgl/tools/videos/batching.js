// 视频二：连续批处理与准入控制（第 7、8 章）
const SEGMENTS = [
  "连续批处理与准入控制：调度器每一轮算哪些请求，又凭什么接纳一个新请求。",
  "静态批处理要等一整批请求都生成完，才能接下一批。短的请求早早结束，它的位置就一直空着。",
  "连续批处理每一轮都重新组 batch：结束的请求立刻离开，等待的请求立刻补上，GPU 始终是满的。",
  "mini-sglang 的调度器一轮只做一种事：有等待的请求就做 prefill，否则做 decode。",
  "接纳新请求之前，调度器要确认 KV 缓存放得下。它按最坏情况估算：剩余的提示词，加上 max_tokens。",
  "假设 KV 池只有 64 个位置。前四个请求最坏情况一共需要 61 个，放得下；第五个还需要 13 个，只能排队等待。",
  "运行中的请求逐渐生成，真实用量在预留的范围里增长。因为预留的是最坏情况，运行中永远不会缺 KV，也就不需要抢占。",
  "代价是保守：此刻实际只用了 41 个位置，还空着 23 个，却接纳不了只需要 13 个的第五个请求。直到第一个请求结束、归还空间，它才被接纳。",
  "vLLM 选择了相反的取舍：乐观地接纳，空间不够时抢占重算。第 8 章对比了这两种做法。",
];
const SEG_MIN = [4, 0, 0, 0, 0, 0, 0, 0, 0];
const COL = (i) => [C.blue, C.green, C.orange, C.purple, C.teal, C.pink, C.yellow][i % 7];

function lanes(x0, y0, rows, steps, cellW, schedule, p, label) {
  text(x0, y0 - 18, label, 22, C.fg, "left", 700);
  const shown = Math.floor(p * steps + 0.0001);
  for (let r = 0; r < rows; r++) {
    text(x0 - 12, y0 + r * 44 + 28, `槽 ${r + 1}`, 16, C.muted, "right", 600);
    for (let s = 0; s < steps; s++) {
      const who = schedule[r][s];
      const on = s < shown;
      if (!on) continue;
      if (who === null) chip(x0 + s * cellW + cellW / 2, y0 + r * 44 + 22, "", C.gray, { w: cellW - 4, h: 34, r: 5, empty: true, dash: true });
      else chip(x0 + s * cellW + cellW / 2, y0 + r * 44 + 22, "", COL(who), { w: cellW - 4, h: 34, r: 5 });
    }
  }
}

function draw() {
  if (!after(1)) { title("连续批处理与准入控制", "调度器每一轮算什么、凭什么接纳新请求", 1 - seg(0, 0.85, 1)); return; }
  // ①② 静态 vs 连续
  if (!after(3)) {
    heading("静态批处理 vs 连续批处理");
    const len = [4, 12, 6, 9];
    const stat = len.map((n, r) => Array.from({ length: 24 }, (_, s) => (s < 12 ? (s < n ? r : null) : (s - 12 < [7, 5, 10, 3][r] ? r + 4 : null))));
    lanes(160, 150, 4, 24, 42, stat, seg(1, 0, 0.9), "静态批处理：一批结束才换下一批（灰色 = 空转）");
    if (after(2)) {
      // 连续：结束立刻补
      const queue = [4, 5, 6, 0, 1, 2, 3, 4, 5, 6];
      const cont = [[], [], [], []];
      const remain = [4, 12, 6, 9];
      const owner = [0, 1, 2, 3];
      let q = 0;
      for (let s = 0; s < 24; s++) {
        for (let r = 0; r < 4; r++) {
          cont[r].push(owner[r] % 7);
          remain[r]--;
          if (remain[r] === 0) { owner[r] = queue[q % queue.length]; remain[r] = [7, 5, 10, 3, 8, 6, 4][q % 7]; q++; }
        }
      }
      lanes(160, 400, 4, 24, 42, cont, seg(2, 0, 0.9), "连续批处理：每轮重新组 batch，结束即离开、等待即补上");
    }
    return;
  }
  // ③ 一轮只做一种
  if (!after(4)) {
    heading("一轮只做一种：prefill 优先");
    const seq = ["P", "D", "D", "D", "P", "D", "D", "P", "D", "D", "D", "D"];
    seq.forEach((k, i) => {
      const a = seg(3, i / seq.length * 0.8, i / seq.length * 0.8 + 0.1);
      box(120 + i * 88, 260, 76, 76, k === "P" ? "prefill" : "decode", k === "P" ? C.orange : C.purple, { a, fs: 16 });
    });
    text(W / 2, 400, "有等待的请求 → prefill（新请求尽快拿到第一个 token）；否则 → decode", 22, C.fg, "center", 600, false, seg(3, 0.5, 0.7));
    text(W / 2, 440, "一轮不混合两种：decode batch 的形状简单，可以用 CUDA Graph", 20, C.muted, "center", 400, false, seg(3, 0.7, 0.9));
    return;
  }
  // ④～⑧ KV 池与准入
  heading("准入控制：按最坏情况预留 KV");
  const px = 120, py = 190, cw = 16, N = 64;
  text(px, py - 16, "KV 池：64 个位置", 20, C.fg, "left", 700);
  const reqs = [[5, 8], [5, 8], [4, 8], [15, 8], [5, 8]];  // [提示词, max_tokens]
  // 每个请求在池中的预留区间
  let off = 0; const spans = [];
  for (let i = 0; i < 4; i++) { const need = reqs[i][0] + reqs[i][1]; spans.push([off, need]); off += need; }
  const gen = after(6) ? Math.round(3 * seg(6, 0.1, 0.8)) : 0;   // 每个请求已生成的 token 数
  const r0done = after(7) && seg(7, 0.72, 0.78) > 0.5;             // 第一个请求结束
  const r4in = after(7) && seg(7, 0.86, 0.92) > 0.5;              // 第五个请求被接纳
  for (let k = 0; k < N; k++) {
    let color = null, solid = false;
    spans.forEach(([o, n], i) => {
      if (!after(5) || k < o || k >= o + n) return;
      if (i === 0 && r0done) { if (r4in) { color = COL(4); solid = k - o < reqs[4][0]; } }
      else { color = COL(i); solid = k - o < reqs[i][0] + gen; }
    });
    const x = px + k * cw + cw / 2;
    if (!color) chip(x, py + 26, "", C.gray, { w: cw - 2, h: 40, r: 3, empty: true });
    else if (solid) chip(x, py + 26, "", color, { w: cw - 2, h: 40, r: 3, solid: true, a: 0.85 });
    else chip(x, py + 26, "", color, { w: cw - 2, h: 40, r: 3, dash: true });
  }
  if (after(4)) {
    const a = seg(4, 0, 0.3) * (after(5) ? 0 : 1);
    box(px, 300, 460, 150, "", C.blue, { a });
    text(px + 24, 340, "一个请求最坏需要：", 20, C.fg, "left", 600, false, a);
    text(px + 24, 380, "剩余提示词 5 + max_tokens 8 = 13 个位置", 20, C.blue, "left", 700, false, a);
    text(px + 24, 420, "（命中前缀缓存的部分不算）", 18, C.muted, "left", 400, false, a);
  }
  if (after(5)) {
    const a = seg(5, 0.1, 0.3);
    const lx = 120;
    reqs.forEach(([pl, mt], i) => {
      const y = 300 + i * 52;
      let st = "已接纳", sc = C.green;
      if (i === 0 && r0done) { st = "已结束，归还 13 个位置"; sc = C.muted; }
      if (i === 4 && !r4in) { st = "排队：13 + 61 > 64"; sc = C.red; }
      chip(lx + 30, y + 16, `r${i}`, COL(i), { a, w: 50 });
      text(lx + 70, y + 24, `${pl} + ${mt} = ${pl + mt}`, 19, C.fg, "left", 600, true, a);
      text(lx + 230, y + 24, st, 19, sc, "left", 700, false, a);
    });
    const a2 = seg(5, 0.5, 0.8) * (1 - seg(7, 0, 0.08));
    text(760, 330, "预留合计：61 ≤ 64 ✓", 22, C.green, "left", 700, false, a2);
    text(760, 370, "再加 r4：74 > 64 ✗", 22, C.red, "left", 700, false, a2);
  }
  if (after(6)) {
    const a = seg(6, 0.1, 0.3) * (after(8) ? 0 : 1);
    chip(790, 452, "", C.blue, { a, w: 30, h: 26, solid: true }); text(815, 460, "已使用", 18, C.fg, "left", 600, false, a);
    chip(930, 452, "", C.blue, { a, w: 30, h: 26, dash: true }); text(955, 460, "预留、尚未使用", 18, C.fg, "left", 600, false, a);
    text(760, 520, "运行中不会缺 KV → 不需要抢占", 22, C.fg, "left", 700, false, seg(6, 0.5, 0.7) * (after(8) ? 0 : 1));
  }
  if (after(7)) {
    const a = seg(7, 0.08, 0.2) * (after(8) ? 0 : 1);
    text(760, 330, `实际只用了 41 个，空着 23 个`, 22, C.orange, "left", 700, false, a);
    text(760, 370, "却接纳不了只需 13 个的 r4", 22, C.orange, "left", 700, false, a);
    text(760, 570, "r0 结束 → r4 才被接纳", 22, C.green, "left", 700, false, seg(7, 0.86, 0.95) * (after(8) ? 0 : 1));
  }
  if (after(8)) {
    const a = seg(8, 0.1, 0.4);
    ctx.save(); ctx.globalAlpha = a; ctx.fillStyle = C.bg; ctx.fillRect(0, 280, W, 360); ctx.restore();
    box(160, 330, 440, 180, "mini-sglang", C.green, { a, sub: "按最坏情况预留 · 永不抢占 · 实现简单", fs: 28 });
    box(680, 330, 440, 180, "vLLM", C.orange, { a, sub: "乐观接纳 · 不够就抢占并重算 · 并发更高", fs: 28 });
  }
}
