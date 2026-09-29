// 视频三：Radix Cache（第 9 章）
const SEGMENTS = [
  "Radix Cache：很多请求的开头是相同的，比如同一段系统提示词、多轮对话的历史。Radix Cache 让它们共享同一份 KV。",
  "四个请求带着同一段系统提示词。没有前缀缓存时，每个请求都要把这段系统提示词重新算一遍。",
  "有了前缀缓存，第一个请求算完之后，系统提示词的 KV 留在缓存里，后面三个请求直接复用。prefill 的计算量从 231 个 token 降到 73 个。",
  "缓存是一棵基数树。插入 1 2 3 4，得到一个节点：它保存这段 token，以及它们的 KV 在池中的位置。没有分叉的一串，只占一个节点。",
  "再插入 1 2 5 6。它和已有的节点共享前缀 1 2，于是节点在这里分裂：公共部分成为父节点，下面分出两支。再插入 7 8，它从根上另起一支。",
  "匹配 1 2 3 9：从根往下走，命中 1 2，再命中 3。3 4 这个节点只匹配了一半，于是再分裂一次，让句柄正好指向 3。",
  "请求运行期间，要给这条路径加锁：路径上每个节点的引用计数加一。被锁住的节点，不能淘汰。",
  "KV 不够时就淘汰：只淘汰没有被锁的叶子，最久没访问的先走。要求释放 2 个 token，先淘汰 4，不够，再淘汰 5 6，实际释放了 3 个。",
  "再要 1 个 token，就淘汰 7 8。被锁住的 1 2 和 3 留了下来，请求结束、解锁之后，它们才能被淘汰。",
  "匹配、插入、加锁、淘汰，这就是 Radix Cache 的全部。mini-sglang 用两百多行实现了它，第 9 章逐行讲解。",
];
const SEG_MIN = [5, 0, 0, 0, 0, 0, 0, 0, 0, 8];

// ---------- 前两段：四个请求共享系统提示词 ----------
function sharing() {
  heading("四个请求，同一段系统提示词");
  const q = [70, 110, 60, 90];
  const x0 = 170, sw = 560;
  const reuse = seg(2, 0.15, 0.45);
  q.forEach((qw, i) => {
    const y = 170 + i * 88, a = seg(1, i * 0.1, i * 0.1 + 0.2);
    text(x0 - 20, y + 32, `请求 ${i + 1}`, 20, C.muted, "right", 600, false, a);
    const shared = i > 0 && reuse > 0;
    box(x0, y, sw, 48, "", shared ? C.green : C.orange, { a, r: 8, solid: !shared || reuse < 0.5, dash: shared && reuse >= 0.5 });
    text(x0 + sw / 2, y + 31, shared && reuse >= 0.5 ? "系统提示词：复用缓存里的 KV" : "系统提示词", 19,
         shared && reuse >= 0.5 ? C.green : "#fff", "center", 700, false, a);
    box(x0 + sw + 8, y, qw, 48, "", C.orange, { a, r: 8, solid: true });
    text(x0 + sw + 8 + qw / 2, y + 31, "问题", 17, "#fff", "center", 700, false, a);
  });
  const a1 = seg(1, 0.5, 0.7);
  chip(1010, 205, "", C.orange, { a: a1, w: 30, h: 24, solid: true }); text(1035, 213, "要算", 18, C.fg, "left", 600, false, a1);
  chip(1010, 245, "", C.green, { a: seg(2, 0.3, 0.5), w: 30, h: 24, dash: true }); text(1035, 253, "复用", 18, C.fg, "left", 600, false, seg(2, 0.3, 0.5));
  const n = Math.round(mix(231, 73, seg(2, 0.45, 0.8)));
  text(1010, 360, "prefill 计算量", 20, C.muted, "left", 600, false, a1);
  text(1010, 420, `${n}`, 56, after(2) && seg(2, 0.45, 0.8) > 0 ? C.green : C.orange, "left", 800, true, a1);
  text(1010, 452, "个 token", 20, C.muted, "left", 600, false, a1);
}

// ---------- 树：以 token 为单位做动画 ----------
const TOK = { 1: 10, 2: 11, 3: 12, 4: 13, 5: 20, 6: 21, 7: 30, 8: 31 };  // token → KV 位置
const L1 = 250, L2 = 395, L3 = 540, RX = 400, RY = 145, CW = 52;
const STATES = [
  [],
  [{ id: "A", toks: [1, 2, 3, 4], x: RX, y: L1, par: "root" }],
  [{ id: "P", toks: [1, 2], x: 290, y: L1, par: "root" }, { id: "A", toks: [3, 4], x: 170, y: L2, par: "P" },
   { id: "B", toks: [5, 6], x: 410, y: L2, par: "P" }],
  [{ id: "P", toks: [1, 2], x: 290, y: L1, par: "root" }, { id: "A", toks: [3, 4], x: 170, y: L2, par: "P" },
   { id: "B", toks: [5, 6], x: 410, y: L2, par: "P" }, { id: "C", toks: [7, 8], x: 620, y: L1, par: "root" }],
  [{ id: "P", toks: [1, 2], x: 290, y: L1, par: "root" }, { id: "S", toks: [3], x: 170, y: L2, par: "P" },
   { id: "A", toks: [4], x: 170, y: L3, par: "S" }, { id: "B", toks: [5, 6], x: 410, y: L2, par: "P" },
   { id: "C", toks: [7, 8], x: 620, y: L1, par: "root" }],
  [{ id: "P", toks: [1, 2], x: 290, y: L1, par: "root" }, { id: "S", toks: [3], x: 170, y: L2, par: "P" },
   { id: "B", toks: [5, 6], x: 410, y: L2, par: "P" }, { id: "C", toks: [7, 8], x: 620, y: L1, par: "root" }],
  [{ id: "P", toks: [1, 2], x: 290, y: L1, par: "root" }, { id: "S", toks: [3], x: 170, y: L2, par: "P" },
   { id: "C", toks: [7, 8], x: 620, y: L1, par: "root" }],
  [{ id: "P", toks: [1, 2], x: 290, y: L1, par: "root" }, { id: "S", toks: [3], x: 170, y: L2, par: "P" }],
];
// 第 k 次转换把 STATES[k] 变成 STATES[k+1]：[段, 段内起点, 段内终点]
const TRANS = [[3, 0.25, 0.5], [4, 0.2, 0.45], [4, 0.72, 0.9], [5, 0.62, 0.85], [7, 0.4, 0.55], [7, 0.62, 0.78], [8, 0.1, 0.35]];

function tokPos(state, t) {
  for (const n of state) {
    const j = n.toks.indexOf(t);
    if (j >= 0) return [n.x + (j - (n.toks.length - 1) / 2) * CW, n.y];
  }
  return null;
}
const same = (a, b) => a && b && a.par === b.par && a.toks.join() === b.toks.join();

function tree() {
  let k = 0;
  while (k < TRANS.length && seg(...TRANS[k]) >= 1) k++;
  const prev = STATES[Math.min(k, STATES.length - 1)], next = STATES[Math.min(k + 1, STATES.length - 1)];
  const p = k < TRANS.length ? seg(...TRANS[k]) : 0;
  // 每个 token 当前的位置与透明度
  const cur = {};
  for (const t of Object.keys(TOK).map(Number)) {
    const a = tokPos(prev, t), b = tokPos(next, t);
    if (a && b) cur[t] = { x: mix(a[0], b[0], p), y: mix(a[1], b[1], p), a: 1 };
    else if (b) cur[t] = { x: b[0], y: b[1], a: p };
    else if (a) cur[t] = { x: a[0], y: a[1], a: 1 - p };
  }
  const geom = (n) => {
    const xs = n.toks.map((t) => cur[t].x), ys = n.toks.map((t) => cur[t].y);
    const x1 = Math.min(...xs) - 34, x2 = Math.max(...xs) + 34, y = ys.reduce((s, v) => s + v, 0) / ys.length;
    return { x1, x2, cx: (x1 + x2) / 2, top: y - 32, bot: y + 46 };
  };
  // 要画的节点：两个状态里相同的画一次，不同的交叉淡入淡出
  const items = [];
  for (const n of prev) items.push({ n, st: prev, a: same(n, next.find((m) => m.id === n.id)) ? 1 : 1 - p });
  for (const n of next) if (!same(n, prev.find((m) => m.id === n.id))) items.push({ n, st: next, a: p });
  // 根
  const rootA = after(3) ? seg(3, 0, 0.15) : 0;
  box(RX - 40, RY - 20, 80, 40, "root", C.gray, { a: rootA, fs: 18, mono: true });
  // 加锁状态
  const lockA = after(6) ? seg(6, 0.2, 0.45) : 0;
  const locked = (n) => lockA > 0 && (n.id === "P" || n.id === "S");
  // 即将被淘汰的节点先变成橙色
  const doomed = (id) => (id === "A" && after(7) && seg(7, 0.28, 0.34) > 0.5) || (id === "B" && after(7) && seg(7, 0.55, 0.6) > 0.5) ||
                         (id === "C" && after(8) && seg(8, 0.0, 0.06) > 0.5);
  const owner = (t) => ({ 4: "A", 5: "B", 6: "B", 7: "C", 8: "C" })[t];
  for (const { n, st, a } of items) {
    if (a <= 0) continue;
    const g = geom(n);
    const pg = n.par === "root" ? { cx: RX, bot: RY + 20 } : geom(st.find((m) => m.id === n.par));
    arrow(pg.cx, pg.bot, g.cx, g.top - 2, C.line, 1, { a, lw: 2.5 });
    const lk = locked(n);
    box(g.x1, g.top, g.x2 - g.x1, g.bot - g.top, "", lk ? C.green : doomed(n.id) ? C.orange : C.blue, { a, r: 10, lw: lk ? 2 + lockA : 2, glow: lk ? lockA * 0.6 : 0 });
    if (after(6) && ["P", "S", "A", "B", "C"].includes(n.id)) {
      const ref = lk ? 1 : 0;
      chip(g.x2 - 4, g.top, `ref ${ref}`, ref ? C.green : C.gray, { a: a * seg(6, 0.1, 0.3), w: 56, h: 22, fs: 13, solid: !!ref });
    }
  }
  // 匹配高亮
  const hit = (t) => (t <= 2 ? seg(5, 0.18, 0.3) : t === 3 ? seg(5, 0.36, 0.46) : 0) * (after(7) ? 1 : 1);
  for (const [t, c] of Object.entries(cur)) {
    if (c.a <= 0) continue;
    const h = Number(t) <= 3 && after(5) ? hit(Number(t)) : 0;
    chip(c.x, c.y, String(t), h > 0.5 ? C.green : doomed(owner(Number(t))) ? C.orange : C.blue, { a: c.a, w: 44, h: 38, solid: h > 0.5, fs: 18 });
    text(c.x, c.y + 38, String(TOK[t]), 14, C.muted, "center", 600, true, c.a);
  }
  if (after(3)) text(RX + 60, RY + 6, "（数字下方：KV 在池中的位置）", 16, C.muted, "left", 400, false, seg(3, 0.5, 0.7));
  // 句柄
  if (after(5)) {
    const s = cur[3];
    const a = seg(5, 0.85, 0.95);
    if (s) arrow(s.x - 110, s.y, s.x - 38, s.y, C.green, a, { a, label: "句柄", ly: -12 });
  }
}

// 右侧的操作说明
function panel() {
  const x = 860;
  const line = (i, s, color, a, y) => text(x, y, s, 21, color, "left", 700, i === 1, a);
  if (during(3)) {
    line(1, "insert [1,2,3,4]", C.blue, seg(3, 0.2, 0.35), 200);
    text(x, 240, "→ 一个节点，KV 在 10～13", 19, C.fg, "left", 600, false, seg(3, 0.4, 0.55));
    text(x, 300, "基数树：没有分叉的一串 = 一个节点", 18, C.muted, "left", 400, false, seg(3, 0.6, 0.75));
    text(x, 330, "（前缀树 trie 则是每个 token 一个节点）", 18, C.muted, "left", 400, false, seg(3, 0.6, 0.75));
  }
  if (during(4)) {
    line(1, "insert [1,2,5,6]", C.blue, seg(4, 0, 0.12), 200);
    text(x, 240, "共享前缀 [1,2] → 分裂", 19, C.fg, "left", 600, false, seg(4, 0.2, 0.35));
    line(1, "insert [7,8]", C.blue, seg(4, 0.62, 0.72), 310);
    text(x, 350, "没有共享前缀 → 根上另起一支", 19, C.fg, "left", 600, false, seg(4, 0.75, 0.88));
  }
  if (during(5)) {
    line(1, "match [1,2,3,9]", C.green, seg(5, 0, 0.12), 200);
    [1, 2, 3, 9].forEach((t, i) => {
      const h = t <= 2 ? seg(5, 0.18, 0.3) : t === 3 ? seg(5, 0.36, 0.46) : 0;
      const miss = t === 9 && seg(5, 0.48, 0.55) > 0.5;
      chip(x + 24 + i * 54, 250, String(t), miss ? C.red : C.green, { a: seg(5, 0.05, 0.15), solid: h > 0.5, w: 44, h: 38, fs: 18 });
    });
    text(x, 310, "命中 3 个 → KV [10, 11, 12]", 19, C.fg, "left", 600, false, seg(5, 0.5, 0.6));
    text(x, 350, "[3,4] 只匹配一半 → 分裂成 [3] → [4]", 19, C.fg, "left", 600, false, seg(5, 0.62, 0.75));
    text(x, 390, "匹配会改变树的形状，但不增删内容", 18, C.muted, "left", 400, false, seg(5, 0.85, 0.95));
  }
  if (during(6)) {
    line(1, "lock(handle)", C.green, seg(6, 0, 0.12), 200);
    text(x, 240, "从句柄走到根，路径上 ref + 1", 19, C.fg, "left", 600, false, seg(6, 0.15, 0.3));
    text(x, 300, "受保护：3 个 token", 20, C.green, "left", 700, false, seg(6, 0.5, 0.65));
    text(x, 336, "可淘汰：5 个 token", 20, C.blue, "left", 700, false, seg(6, 0.5, 0.65));
    text(x, 390, "可用空间 = 空闲页 + 可淘汰", 18, C.muted, "left", 400, false, seg(6, 0.75, 0.9));
  }
  if (during(7)) {
    line(1, "evict(2)", C.orange, seg(7, 0, 0.1), 200);
    text(x, 240, "候选：没被锁的叶子，按访问时间", 19, C.fg, "left", 600, false, seg(7, 0.1, 0.22));
    text(x, 280, "[4] 最久没访问（匹配没刷新它）", 18, C.muted, "left", 400, false, seg(7, 0.2, 0.32));
    text(x, 330, "释放 13", 20, C.orange, "left", 700, true, seg(7, 0.45, 0.55));
    text(x, 370, "释放 20 21", 20, C.orange, "left", 700, true, seg(7, 0.7, 0.8));
    text(x, 420, "按节点淘汰：要 2 个，实际释放 3 个", 18, C.muted, "left", 400, false, seg(7, 0.82, 0.92));
  }
  if (during(8)) {
    line(1, "evict(1)", C.orange, seg(8, 0, 0.1), 200);
    text(x, 250, "释放 30 31", 20, C.orange, "left", 700, true, seg(8, 0.3, 0.4));
    text(x, 310, "被锁的 [1,2]、[3] 留下", 19, C.green, "left", 700, false, seg(8, 0.45, 0.6));
    text(x, 350, "请求结束、解锁后才能淘汰", 19, C.fg, "left", 600, false, seg(8, 0.65, 0.8));
  }
}

function outro() {
  const a = seg(9, 0, 0.2);
  heading("Radix Cache 的四个操作", a);
  const ops = [["匹配", "match_prefix", C.green], ["插入", "insert_prefix", C.blue], ["加锁", "lock_handle", C.purple], ["淘汰", "evict", C.orange]];
  ops.forEach(([n, f, c], i) => {
    box(110 + i * 270, 250, 240, 150, n, c, { a: seg(9, 0.05 + i * 0.1, 0.2 + i * 0.1), sub: f, fs: 34, subMono: true });
  });
  text(W / 2, 490, "kvcache/radix_cache.py · 第 9 章", 22, C.muted, "center", 600, false, seg(9, 0.5, 0.7));
}

function draw() {
  if (!after(1)) { title("Radix Cache", "跨请求复用前缀的 KV", 1 - seg(0, 0.88, 1)); return; }
  if (!after(3)) { sharing(); return; }
  if (after(9)) { outro(); return; }
  heading("一棵基数树");
  tree();
  panel();
}
