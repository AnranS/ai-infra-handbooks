// 视频一：一个请求的一生（导读）
const SEGMENTS = [
  "这段视频跟着一个请求，把 mini-sglang 的所有进程走一遍。",
  "mini-sglang 由几类进程组成：API Server 接收 HTTP 请求；tokenizer 负责分词和反分词；每张 GPU 上有一个调度器进程，里面是调度器和引擎。",
  "客户端发来一个对话请求。API Server 给它分配一个 uid，把文本交给 tokenizer；tokenizer 套上对话模板、分词，再把 token 发给调度器。",
  "调度器先把请求放进等待队列。轮到它时，在 Radix Cache 里找最长的已缓存前缀，确认显存够用，然后分配一个请求槽和若干 KV 页。",
  "引擎做一次 prefill，算出第一个 token。这个 token 直接在 GPU 上写进 token pool，作为下一轮的输入，同时异步拷回 CPU。",
  "调度器把新 token 发给 detokenizer，它把 token 增量地还原成文本，再由 API Server 通过 SSE 流式推给客户端。",
  "之后每一轮 decode，这个请求和其他正在生成的请求组成一个 batch，每个请求算一个 token。请求随时加入、随时离开，这就是连续批处理。",
  "遇到结束符或者达到最大长度，请求就结束了：请求槽归还，它的 KV 留在 Radix Cache 里，等待前缀相同的请求来复用。",
  "如果开了张量并行，rank 0 把收到的消息原样广播给其他 rank，所有 rank 做出完全相同的调度决策；模型计算中间用 all-reduce 汇总。",
  "这条路径上的每一步，都对应手册里的一章。",
];
const SEG_MIN = [4, 0, 0, 0, 0, 0, 7, 0, 0, 8];

const P = {
  client: [40, 300, 150, 92], api: [250, 280, 200, 132], tok: [520, 280, 200, 132],
  r0: [800, 104, 440, 250], sch: [818, 150, 190, 186], eng: [1030, 150, 192, 186],
  r1: [800, 390, 440, 190],
};
const cx = (b) => b[0] + b[2] / 2, cy = (b) => b[1] + b[3] / 2;

function processes(a) {
  const p1 = seg(1, 0.0, 0.25), p2 = seg(1, 0.25, 0.5), p3 = seg(1, 0.5, 0.85);
  box(...P.client, "客户端", C.gray, { a: a * p1, sub: "OpenAI SDK" });
  box(...P.api, "API Server", C.blue, { a: a * p1, sub: "FastAPI · 主进程" });
  box(...P.tok, "tokenizer", C.orange, { a: a * p2, sub: "分词 / 反分词" });
  box(...P.r0, "", C.green, { a: a * p3 * 0.9, r: 16 });
  text(cx(P.r0), P.r0[1] + 32, "调度器进程 rank 0（GPU 0）", 20, C.green, "center", 700, false, a * p3);
  box(...P.sch, "Scheduler", C.green, { a: a * p3, sub: "调度 · KV 分配", top: true });
  box(...P.eng, "Engine", C.purple, { a: a * p3, sub: "模型 · KV 池", top: true });
}

function tokenChips(x, y, a, n = 4) {
  const ids = [151644, 872, 198, 108386];
  for (let i = 0; i < n; i++) chip(x + i * 50 - (n - 1) * 25, y, ids[i] % 1000, C.orange, { a, w: 46, fs: 14 });
}

function draw() {
  if (!after(1)) { title("一个请求的一生", "mini-sglang 的进程与数据流", 1 - seg(0, 0.85, 1)); return; }
  heading("一个请求的一生", 1);
  const dimOut = after(9) ? 1 - 0.75 * seg(9, 0, 0.3) : 1;
  ctx.globalAlpha = 1;
  processes(dimOut);

  // ② 请求从客户端到调度器
  if (after(2)) {
    const p = seg(2);
    const path = [[cx(P.client), cy(P.client)], [cx(P.api), cy(P.api)]];
    if (p < 0.33) {
      const [x, y] = along(path, p / 0.33);
      chip(x, y - 40, "你好", C.blue, { w: 70, mono: false });
    }
    if (p >= 0.25) text(cx(P.api), P.api[1] + P.api[3] + 30, "uid = 7", 18, C.blue, "center", 700, true, dimOut * seg(2, 0.25, 0.35));
    if (p >= 0.33 && p < 0.62) {
      const [x, y] = along([[cx(P.api), cy(P.api)], [cx(P.tok), cy(P.tok)]], (p - 0.33) / 0.29);
      chip(x, y - 40, "你好", C.blue, { w: 70, mono: false });
    }
    if (p >= 0.62 && p < 0.78) tokenChips(cx(P.tok), P.tok[1] - 34, 1);
    if (p >= 0.78 && !after(3)) {
      const [x, y] = along([[cx(P.tok), P.tok[1] - 34], [cx(P.sch), P.sch[1] + 120]], (p - 0.78) / 0.22);
      tokenChips(x, y, 1);
    }
    arrow(P.api[0] + P.api[2], cy(P.api) - 20, P.tok[0], cy(P.tok) - 20, C.blue, seg(2, 0.3, 0.45), { label: "TokenizeMsg", fs: 15, a: dimOut });
    arrow(P.tok[0] + P.tok[2], P.tok[1] + 30, P.sch[0], P.sch[1] + 60, C.orange, seg(2, 0.75, 0.9), { label: "UserMsg", fs: 15, a: dimOut, ly: -16 });
  }

  // ③ 调度：等待队列 → 前缀匹配 → 分配
  if (after(3)) {
    const a = dimOut;
    const qx = P.sch[0] + 20, qy = P.sch[1] + 92;
    text(qx, qy - 8, "等待队列", 15, C.muted, "left", 600, false, a * seg(3, 0, 0.15));
    for (let i = 0; i < 3; i++) chip(qx + 20 + i * 44, qy + 22, i === 2 ? "7" : "", i === 2 ? C.blue : C.gray, { a: a * seg(3, 0, 0.15) * (after(4) ? 0.35 : 1), w: 38, h: 28, fs: 14 });
    text(qx, qy + 70, "Radix 前缀匹配", 15, C.green, "left", 600, false, a * seg(3, 0.3, 0.45) * (after(4) ? 0.4 : 1));
    text(qx, qy + 90, "→ 分配请求槽、KV 页", 15, C.green, "left", 600, false, a * seg(3, 0.55, 0.7) * (after(4) ? 0.4 : 1));
    // KV 池的格子（画在引擎里）
    const kx = P.eng[0] + 12, ky = P.eng[1] + 78;
    for (let r = 0; r < 3; r++) for (let c = 0; c < 8; c++) {
      const idx = r * 8 + c;
      const mine = idx >= 9 && idx < 13, other = [1, 2, 3, 17, 18].includes(idx);
      let color = mine ? C.blue : other ? C.green : null;
      if (after(7) && mine) color = mix(0, 1, seg(7, 0.3, 0.6)) > 0.5 ? C.gray : C.blue;
      const show = mine ? seg(3, 0.6, 0.9) : 1;
      chip(kx + 10 + c * 20, ky + 10 + r * 24, "", color || C.gray, { w: 18, h: 20, r: 3, empty: !color || show < 0.5, a: a * seg(3, 0, 0.2) });
    }
    text(kx, ky - 8, "KV 池", 14, C.muted, "left", 600, false, a * seg(3, 0, 0.2));
  }

  // ④ prefill → 第一个 token
  if (after(4)) {
    const a = dimOut;
    const p = seg(4);
    if (p < 0.4) tokenChips(mix(cx(P.sch), cx(P.eng), p / 0.4), P.eng[1] + 166, 1);
    const flash = p > 0.35 && p < 0.65 ? Math.sin((p - 0.35) / 0.3 * Math.PI) : 0;
    box(...P.eng, "", C.purple, { a: flash * 0.9 * a, glow: flash, lw: 3 });
    if (p > 0.6) {
      chip(P.eng[0] + 150, P.eng[1] + 166, "你", C.pink, { w: 44, mono: false, a: a * seg(4, 0.6, 0.7) * (after(5) ? 0.4 : 1) });
      text(P.eng[0] + 60, P.eng[1] + 172, "新 token", 15, C.pink, "center", 600, false, a * seg(4, 0.6, 0.7) * (after(5) ? 0.4 : 1));
    }
    text(cx(P.eng), P.eng[1] + P.eng[3] + 26, "写回 token pool（GPU 上）", 15, C.pink, "center", 600, false, a * seg(4, 0.7, 0.85) * (after(6) ? 0 : 1));
  }

  // ⑤ 回程：detokenize → SSE
  if (after(5)) {
    const a = dimOut;
    const p = seg(5);
    arrow(P.sch[0], P.sch[1] + 170, P.tok[0] + P.tok[2], P.tok[1] + 100, C.green, seg(5, 0, 0.25), { label: "DetokenizeMsg", fs: 15, a, ly: 30, lx: 20 });
    arrow(P.tok[0], cy(P.tok) + 30, P.api[0] + P.api[2], cy(P.api) + 30, C.orange, seg(5, 0.3, 0.5), { label: "UserReply", fs: 15, a, ly: 26 });
    arrow(P.api[0], cy(P.api) + 30, P.client[0] + P.client[2], cy(P.client) + 30, C.blue, seg(5, 0.55, 0.7), { label: "SSE", fs: 15, a, ly: 24 });
  }
  // 客户端收到的文本
  if (after(5)) {
    const full = "你好！有什么可以帮你的吗？";
    let n = seg(5, 0.7, 0.8) > 0 ? 1 : 0;
    if (after(6)) n = 1 + Math.floor(seg(6, 0.05, 0.95) * (full.length - 1));
    const bubble = full.slice(0, n);
    if (bubble) {
      box(20, 440, 220, 110, "", C.blue, { a: dimOut, r: 14 });
      text(36, 472, "客户端收到：", 16, C.muted, "left", 600, false, dimOut);
      text(36, 506, bubble.slice(0, 8), 21, C.fg, "left", 600, false, dimOut);
      if (bubble.length > 8) text(36, 536, bubble.slice(8), 21, C.fg, "left", 600, false, dimOut);
    }
  }

  // ⑥ decode 轮次：batch 里的请求来来去去
  if (after(6) && !after(8)) {
    const a = dimOut * (1 - seg(8, 0, 0.2));
    const rounds = 6, r = Math.min(rounds - 1, Math.floor(seg(6, 0, 1) * rounds));
    const members = [["7", C.blue], ["3", C.green], ["5", C.orange], ["9", C.teal]];
    const present = [[1, 1, 1, 0], [1, 1, 1, 0], [1, 0, 1, 1], [1, 0, 1, 1], [1, 0, 0, 1], [1, 0, 0, 1]][r];
    const bx = P.sch[0], by = P.r0[1] + P.r0[3] + 22;
    text(bx, by + 24, `decode 第 ${r + 1} 轮 batch：`, 18, C.fg, "left", 700, false, a);
    let k = 0;
    members.forEach(([id, c], i) => {
      if (present[i]) { chip(bx + 210 + k * 56, by + 18, id, c, { a, w: 48 }); k++; }
    });
    text(bx, by + 64, r >= 2 ? "uid 3 结束离开，uid 9 加入" : "", 16, C.muted, "left", 400, false, a);
    text(bx, by + 88, r >= 4 ? "uid 5 结束离开" : "", 16, C.muted, "left", 400, false, a);
  }

  // ⑦ 结束
  if (after(7)) {
    const a = dimOut * (after(8) ? 1 - seg(8, 0, 0.3) : 1);
    chip(P.eng[0] + 150, P.eng[1] + 166, "EOS", C.red, { w: 56, a: a * seg(7, 0, 0.2) });
    text(cx(P.r0), P.r0[1] + P.r0[3] + 40, "请求槽归还 · KV 进入 Radix Cache（灰色 = 可复用、可淘汰）", 17, C.fg, "center", 600, false, a * seg(7, 0.3, 0.5));
  }

  // ⑧ 张量并行
  if (after(8)) {
    const a = dimOut * seg(8, 0.1, 0.4);
    box(...P.r1, "", C.green, { a: a * 0.9, r: 16 });
    text(cx(P.r1), P.r1[1] + 32, "调度器进程 rank 1（GPU 1）", 20, C.green, "center", 700, false, a);
    box(P.r1[0] + 18, P.r1[1] + 50, 190, 120, "Scheduler", C.green, { a, sub: "完全相同的决策" });
    box(P.r1[0] + 230, P.r1[1] + 50, 192, 120, "Engine", C.purple, { a, sub: "另一半权重" });
    arrow(P.sch[0] + 40, P.sch[1] + P.sch[3], P.r1[0] + 58, P.r1[1] + 50, C.green, seg(8, 0.35, 0.55), { dash: true, label: "原样广播消息", fs: 15, lx: 72, ly: 8, a: dimOut });
    arrow(P.eng[0] + 150, P.eng[1] + P.eng[3], P.r1[0] + 380, P.r1[1] + 50, C.purple, seg(8, 0.6, 0.8), { label: "all-reduce", fs: 15, lx: -86, ly: 8, a: dimOut });
    arrow(P.r1[0] + 350, P.r1[1] + 50, P.eng[0] + 120, P.eng[1] + P.eng[3], C.purple, seg(8, 0.6, 0.8), { a: dimOut });
  }

  // ⑨ 收尾
  if (after(9)) {
    const a = seg(9, 0.2, 0.5);
    ctx.save(); ctx.globalAlpha = a * 0.92; ctx.fillStyle = C.bg; ctx.fillRect(0, 90, W, 540); ctx.restore();
    const items = [["进程与消息", "第 12～15 章"], ["调度与 KV 分配", "第 7～10 章"], ["prefill / decode", "第 1～6 章"],
                   ["重叠调度", "第 11 章"], ["张量并行", "第 16 章"], ["CUDA Graph", "第 18 章"]];
    items.forEach(([t, ch], i) => {
      const x = 180 + (i % 3) * 320, y = 220 + Math.floor(i / 3) * 150;
      box(x, y, 280, 110, t, [C.orange, C.green, C.purple, C.blue, C.teal, C.pink][i], { a: a * seg(9, 0.3 + i * 0.08, 0.5 + i * 0.08), sub: ch, fs: 24 });
    });
  }
}
