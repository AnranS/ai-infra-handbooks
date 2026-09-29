// 视频五：张量并行（第 16 章）
const SEGMENTS = [
  "张量并行：把每一层的权重切开，分到多张卡上。以两张卡为例，看一个 decoder 层是怎么算的。",
  "第一个矩阵按输出维切，叫列并行。每张卡拿到一半的列，用完整的输入 x，算出一半的输出通道。",
  "第二个矩阵按输入维切，叫行并行。每张卡用自己那一半中间结果，乘自己那一半的行，得到的是完整形状的部分和。",
  "最后做一次 all-reduce，把两份部分和加起来，两张卡都拿到完整的结果。先列后行，中间不需要任何通信。",
  "一个 decoder 层就是两组先列后行：注意力按头切，MLP 按中间维切。所以每层只有两次 all-reduce，在 o_proj 和 down_proj 之后。",
  "Qwen3 有 16 个 q 头、8 个 KV 头。两张卡时，每张卡 8 个 q 头、4 个 KV 头，KV 池也只存自己的 4 个头。",
  "嵌入层和输出层按词表切。查嵌入时，不在本段的 token 填零，再 all-reduce 相加；输出层各算一段 logits，用 all-gather 拼成完整的分布。",
  "数一数：28 层乘以 2，再加上嵌入层的 1 次，一次前向是 57 次 all-reduce，加上 1 次 all-gather。示例里测到的正是这个数。",
  "在 CPU 上用 gloo 通信，两卡、四卡的输出与单卡完全一致。第 16 章给出完整的代码和测试。",
];
const SEG_MIN = [4, 0, 0, 0, 0, 0, 0, 0, 7];

const R0 = 135, R1 = 405;  // 两个 rank 的行
function mat(x, y, w, h, label, color, o = {}) {
  box(x, y, w, h, "", color, { r: 6, a: o.a ?? 1, solid: o.solid, lw: 2 });
  text(x + w / 2, y + h / 2 + 7, label, o.fs ?? 18, o.solid ? "#fff" : C.fg, "center", 700, o.mono ?? true, o.a ?? 1);
  if (o.dim) text(x + w / 2, y + h + 22, o.dim, 14, C.muted, "center", 600, true, o.a ?? 1);
}

function matmul() {
  heading("先列后行：一次 all-reduce");
  const cols = [C.blue, C.orange];
  const a0 = seg(1, 0, 0.1);
  [R0, R1].forEach((y, r) => {
    text(60, y + 62, `rank ${r}`, 18, cols[r], "left", 800, false, a0);
    mat(130, y, 80, 110, "x", C.gray, { a: a0, dim: "[n, h]" });
    text(232, y + 64, "×", 26, C.muted, "center", 700, false, a0);
  });
  // W1 从中间一分为二
  const sp = seg(1, 0.25, 0.55);
  [R0, R1].forEach((y, r) => {
    const x = 255, yy = mix(270, y, sp), xx = mix(255 + r * 70, x, sp);
    mat(xx, yy, 70, 110, `W₁${sp > 0.5 ? (r ? "右" : "左") : ""}`, cols[r], { a: seg(1, 0.1, 0.2), fs: 16, mono: false, dim: sp > 0.9 ? "[h, d/2]" : "" });
  });
  text(325, 262, "W₁ 按列切", 16, C.muted, "center", 600, false, seg(1, 0.1, 0.2) * (1 - sp));
  const ya = seg(1, 0.6, 0.8);
  [R0, R1].forEach((y, r) => {
    text(348, y + 64, "=", 26, C.muted, "center", 700, false, ya);
    mat(370, y, 70, 110, `y${r}`, cols[r], { a: ya, solid: true, dim: "[n, d/2]" });
  });
  if (!after(2)) return;
  // W2 按行切
  const sp2 = seg(2, 0.2, 0.45);
  [R0, R1].forEach((y, r) => {
    text(462, y + 64, "×", 26, C.muted, "center", 700, false, seg(2, 0, 0.1));
    const yy = mix(255 + r * 70, y + 20, sp2);
    mat(485, yy, 110, 70, `W₂${sp2 > 0.5 ? (r ? "下" : "上") : ""}`, cols[r], { a: seg(2, 0.05, 0.15), fs: 16, mono: false, dim: sp2 > 0.9 ? "[d/2, h]" : "" });
  });
  text(540, 250, "W₂ 按行切", 16, C.muted, "center", 600, false, seg(2, 0.05, 0.15) * (1 - sp2));
  const za = seg(2, 0.55, 0.75);
  [R0, R1].forEach((y, r) => {
    text(618, y + 64, "=", 26, C.muted, "center", 700, false, za);
    mat(640, y, 110, 110, `z${r}`, cols[r], { a: za, dim: "[n, h] 部分和" });
  });
  if (!after(3)) return;
  const ar = seg(3, 0.05, 0.3);
  box(800, 280, 150, 70, "all-reduce", C.green, { a: ar, solid: true, fs: 20 });
  arrow(750, R0 + 55, 800, 300, C.green, ar, { lw: 2.5 });
  arrow(750, R1 + 55, 800, 330, C.green, ar, { lw: 2.5 });
  const fa = seg(3, 0.3, 0.5);
  arrow(950, 300, 1000, R0 + 55, C.green, fa, { lw: 2.5 });
  arrow(950, 330, 1000, R1 + 55, C.green, fa, { lw: 2.5 });
  [R0, R1].forEach((y) => mat(1000, y, 110, 110, "z₀+z₁", C.green, { a: fa, solid: true, dim: "[n, h] 完整", fs: 17 }));
  text(640, 600, "y 的切分方式正好是 W₂ 需要的：中间不需要通信", 20, C.fg, "center", 700, false, seg(3, 0.6, 0.8));
}

function layer() {
  heading("一个 decoder 层：两次 all-reduce");
  const blocks = [
    ["RMSNorm", "", C.gray], ["qkv_proj", "列并行", C.blue], ["注意力", "HEADS", C.purple], ["o_proj", "行并行", C.blue], ["AR", "", C.green],
    ["RMSNorm", "", C.gray], ["gate_up", "列并行", C.orange], ["SiLU × up", "逐元素", C.orange], ["down_proj", "行并行", C.orange], ["AR", "", C.green],
  ];
  const x0 = 50, bw = 104, gap = 14, flow = seg(4, 0.08, 0.7) * blocks.length;
  [165, 385].forEach((y, r) => text(x0, y - 16, `rank ${r}`, 17, r ? C.orange : C.blue, "left", 800, false, seg(4, 0, 0.1)));
  blocks.forEach(([n, sub, col], i) => {
    const x = x0 + i * (bw + gap), a = clamp(flow - i + 1);
    const hot = flow > i && flow < i + 1.2;
    if (n === "AR") {
      box(x, 165, bw, 310, "all-reduce", col, { a, solid: true, fs: 17, glow: hot ? 0.8 : 0 });
      return;
    }
    [165, 385].forEach((y, r) => {
      let s = sub;
      if (sub === "HEADS") s = r ? "头 8–15" : "头 0–7";
      box(x, y, bw, 90, n, col, { a, sub: s, fs: 16, glow: hot ? 0.6 : 0, lw: hot ? 3 : 2 });
    });
    if (i < blocks.length - 1) [210, 430].forEach((y) => arrow(x + bw, y, x + bw + gap, y, C.line, a, { lw: 2 }));
  });
  const b = seg(4, 0.75, 0.9);
  text(W / 2, 540, "注意力按头切、MLP 按中间维切：头之间、通道之间互不相关，本地就能算完", 19, C.fg, "center", 600, false, b);
  text(W / 2, 575, "（残差相加和 RMSNorm 在每张卡上用完整的 hidden 重复计算）", 17, C.muted, "center", 400, false, b);
}

function heads() {
  heading("GQA 的头怎么分");
  const a = seg(5, 0, 0.15);
  text(120, 170, "q 头 × 16", 20, C.fg, "left", 700, false, a);
  text(120, 330, "KV 头 × 8", 20, C.fg, "left", 700, false, a);
  const s = seg(5, 0.25, 0.45);
  for (let i = 0; i < 16; i++) {
    const r = i >= 8 ? 1 : 0;
    chip(140 + i * 58 + r * s * 40, 215, `q${i}`, s > 0.5 ? (r ? C.orange : C.blue) : C.gray, { a, w: 50, h: 40, fs: 14 });
  }
  for (let i = 0; i < 8; i++) {
    const r = i >= 4 ? 1 : 0;
    chip(140 + i * 116 + 29 + r * s * 40, 375, `kv${i}`, s > 0.5 ? (r ? C.orange : C.blue) : C.gray, { a, w: 100, h: 40, fs: 14 });
  }
  const b = seg(5, 0.45, 0.6);
  text(140 + 3.5 * 58, 270, "rank 0：8 个 q 头", 18, C.blue, "center", 700, false, b);
  text(140 + 11.5 * 58 + 40, 270, "rank 1：8 个 q 头", 18, C.orange, "center", 700, false, b);
  text(140 + 1.5 * 116 + 29, 430, "rank 0：4 个 KV 头", 18, C.blue, "center", 700, false, b);
  text(140 + 5.5 * 116 + 29 + 40, 430, "rank 1：4 个 KV 头", 18, C.orange, "center", 700, false, b);
  const c = seg(5, 0.65, 0.85);
  text(W / 2, 510, "每张卡的 KV 池：[页数, 1, 4, 128]——显存减半", 21, C.fg, "center", 700, false, c);
  text(W / 2, 550, "TP=16 时 KV 头不够分：每两张卡共用一个 KV 头，各存一份副本", 17, C.muted, "center", 400, false, c);
}

function vocab() {
  heading("词表并行：嵌入 all-reduce，输出 all-gather");
  const a = seg(6, 0, 0.12);
  // 左：嵌入
  text(80, 130, "嵌入层", 22, C.fg, "left", 800, false, a);
  box(80, 150, 60, 150, "", C.blue, { a, solid: true, r: 4 });
  box(80, 300, 60, 150, "", C.orange, { a, solid: true, r: 4 });
  text(150, 230, "rank 0：词表 0 ~ 75967", 15, C.blue, "left", 600, false, a);
  text(150, 380, "rank 1：词表 75968 ~ 151935", 15, C.orange, "left", 600, false, a);
  const q = seg(6, 0.12, 0.3);
  chip(110, 490, "token 100000", C.gray, { a: q, w: 140, fs: 15 });
  const v = seg(6, 0.25, 0.4);
  text(400, 230, "→ 0 0 0 0（不在本段）", 17, C.muted, "left", 600, true, v);
  text(400, 380, "→ 0.3 -1.2 0.8 …（查到）", 17, C.orange, "left", 600, true, v);
  const s = seg(6, 0.35, 0.5);
  box(400, 430, 200, 56, "all-reduce 相加", C.green, { a: s, solid: true, fs: 17 });
  text(500, 520, "每个 token 恰好一个 rank 查到", 15, C.muted, "center", 400, false, s);
  // 右：输出
  const b = seg(6, 0.55, 0.65);
  text(700, 130, "输出层", 22, C.fg, "left", 800, false, b);
  text(700, 190, "rank 0 的 logits", 16, C.blue, "left", 600, false, b);
  box(700, 205, 250, 44, "0 ~ 75967", C.blue, { a: b, solid: true, fs: 15 });
  text(700, 290, "rank 1 的 logits", 16, C.orange, "left", 600, false, b);
  box(700, 305, 250, 44, "75968 ~ 151935", C.orange, { a: b, solid: true, fs: 15 });
  const g = seg(6, 0.7, 0.85);
  box(700, 400, 150, 50, "all-gather", C.green, { a: g, solid: true, fs: 17 });
  box(700, 480, 250, 44, "", C.blue, { a: g, solid: true, r: 4 });
  box(950, 480, 250, 44, "", C.orange, { a: g, solid: true, r: 4 });
  text(950, 560, "拼接成完整的 151936 维分布 → 采样", 16, C.fg, "center", 600, false, g);
}

function count() {
  heading("数一数通信次数");
  const n = Math.floor(seg(7, 0.05, 0.55) * 28 + 0.0001);
  for (let i = 0; i < 28; i++) {
    const x = 100 + (i % 14) * 76, y = 150 + Math.floor(i / 14) * 110;
    const on = i < n;
    box(x, y, 64, 70, `L${i}`, on ? C.purple : C.gray, { a: 0.3 + 0.7 * seg(7, 0, 0.05), fs: 15, r: 8 });
    chip(x + 20, y + 58, "", C.green, { a: on ? 1 : 0, w: 14, h: 10, solid: true, r: 3 });
    chip(x + 44, y + 58, "", C.green, { a: on ? 1 : 0, w: 14, h: 10, solid: true, r: 3 });
  }
  const ar = n * 2 + (seg(7, 0.55, 0.62) > 0.5 ? 1 : 0);
  text(100, 440, "all-reduce", 22, C.muted, "left", 700, false, 1);
  text(100, 510, `${ar}`, 60, C.green, "left", 800, true, 1);
  text(260, 510, "= 28 × 2 + 1（嵌入）", 22, C.fg, "left", 600, false, seg(7, 0.55, 0.65));
  text(700, 440, "all-gather", 22, C.muted, "left", 700, false, seg(7, 0.65, 0.75));
  text(700, 510, "1", 60, C.green, "left", 800, true, seg(7, 0.65, 0.75));
  text(760, 510, "（输出层）", 22, C.fg, "left", 600, false, seg(7, 0.65, 0.75));
  text(100, 575, "示例输出：一次前向 all_reduce 57 次、all_gather 1 次 ✓", 19, C.muted, "left", 600, false, seg(7, 0.8, 0.92));
}

function outro() {
  const a = seg(8, 0, 0.2);
  heading("结果与单卡一致", a);
  ["TP=1", "TP=2", "TP=4"].forEach((s, i) => {
    box(170 + i * 330, 230, 280, 150, s, [C.gray, C.blue, C.orange][i], { a: seg(8, 0.05 + i * 0.1, 0.2 + i * 0.1), sub: "同样的 token", fs: 34 });
  });
  text(W / 2, 470, "CPU 上 gloo · GPU 上 NCCL · 第 16 章", 22, C.muted, "center", 600, false, seg(8, 0.45, 0.65));
}

function draw() {
  if (!after(1)) { title("张量并行", "把每一层切开，分到多张卡上", 1 - seg(0, 0.88, 1)); return; }
  if (!after(4)) { matmul(); return; }
  if (!after(5)) { layer(); return; }
  if (!after(6)) { heads(); return; }
  if (!after(7)) { vocab(); return; }
  if (!after(8)) { count(); return; }
  outro();
}
