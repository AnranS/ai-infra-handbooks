// 视频四：重叠调度（第 11 章）
const SEGMENTS = [
  "重叠调度：小模型 decode 一步，GPU 只要几毫秒，而调度器在 CPU 上做的事也要几毫秒。重叠调度让这两者同时进行。",
  "普通循环里，CPU 先调度、发射一轮，然后等 GPU 算完，处理结果，再调度下一轮。GPU 有将近一半的时间在等 CPU。",
  "重叠循环先发射下一轮，再处理上一轮的结果。处理结果的时候，GPU 已经在算下一轮了，CPU 的开销被藏了起来。同样的时间，多算了两轮。",
  "可是发射下一轮时，CPU 还不知道这一轮采样出了什么。办法是：采样结果直接在 GPU 上写进 token pool，下一轮从那里读输入，完全不经过 CPU。",
  "调度需要的长度，在发射时就在 CPU 上推进一步。唯一不知道的，是请求有没有在这一轮遇到 EOS。",
  "所以一个在第 N 轮遇到 EOS 的请求，会被多调度一轮：处理第 N 轮、发现 EOS 的时候，第 N 加 1 轮已经带着它在 GPU 上跑了。",
  "这让处理结果变得微妙。复刻时我们发现了四个问题，每一个都有测试可以复现。",
  "第一，max_tokens 的结束标记提前了一个 token，客户端少收一个。改为按已经收到的 token 数判断。",
  "第二，遇到 EOS 之后，又发出一条过期的消息，detokenizer 为它留下永远不会结束的状态。改为跳过上一轮已经结束的请求。",
  "第三，请求槽被过早复用，新请求的提示词可能被在途的旧写入覆盖。改为推迟到下一次处理结果时再归还。",
  "第四，prefill 在途时收到 abort，同一批页既在空闲列表里，又在缓存树里。abort 时把请求记为已结束即可。",
  "把任何一处修正改回官方写法，对应的测试就会失败。第 11 章逐一讲解。",
];
const SEG_MIN = [5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7];

// ---------- 时间线 ----------
const SCH = 0.9, PRO = 0.7, FWD = 2.0, SPAN = 11;
function normalPlan() {
  const cpu = [], gpu = []; let t = 0;
  for (let i = 1; t < SPAN; i++) {
    cpu.push([t, t + SCH, `调度 ${i}`, C.blue]); t += SCH;
    gpu.push([t, t + FWD, `前向 ${i}`, C.purple]); t += FWD;
    cpu.push([t, t + PRO, `处理 ${i}`, C.teal]); t += PRO;
  }
  return { cpu, gpu };
}
function overlapPlan() {
  const cpu = [], gpu = []; let c = 0, g = 0; const done = [];
  cpu.push([0, SCH, "调度 1", C.blue]); c = SCH; g = c; gpu.push([g, g + FWD, "前向 1", C.purple]); g += FWD; done.push(g);
  for (let i = 2; c < SPAN; i++) {
    cpu.push([c, c + SCH, `调度 ${i}`, C.blue]); c += SCH;
    const st = Math.max(g, c); gpu.push([st, st + FWD, `前向 ${i}`, C.purple]); g = st + FWD; done.push(g);
    c = Math.max(c, done[i - 2]); cpu.push([c, c + PRO, `处理 ${i - 1}`, C.teal]); c += PRO;
  }
  return { cpu, gpu };
}
function lane(x0, y, sc, items, play, label) {
  text(x0 - 16, y + 30, label, 18, C.muted, "right", 700);
  ctx.save(); ctx.fillStyle = "#eeeef0"; ctx.fillRect(x0, y + 4, SPAN * sc, 42); ctx.restore();
  for (const [s, e, name, col] of items) {
    if (s >= play || s >= SPAN) continue;
    const e2 = Math.min(e, play, SPAN);
    box(x0 + s * sc + 1, y + 4, (e2 - s) * sc - 2, 42, "", col, { r: 6, solid: true, a: 0.9 });
    if (e2 - s > 0.55) text(x0 + (s + e2) / 2 * sc, y + 31, name, 15, "#fff", "center", 700);
  }
}
function timelines() {
  heading("普通循环 vs 重叠循环");
  const x0 = 150, sc = 98;
  const n = normalPlan(), o = overlapPlan();
  const p1 = seg(1, 0.05, 0.85) * SPAN, p2 = seg(2, 0.02, 0.6) * SPAN;
  text(x0, 128, "普通循环：发射 → 等待 → 处理 → 调度下一轮", 20, C.fg, "left", 700);
  lane(x0, 145, sc, n.cpu, p1, "CPU");
  lane(x0, 200, sc, n.gpu, p1, "GPU");
  if (p1 > 3.2) text(x0 + 3.2 * sc, 262, "↑ GPU 空等 CPU", 17, C.red, "center", 700, false, seg(1, 0.4, 0.55));
  if (after(2)) {
    text(x0, 348, "重叠循环：先发射下一轮，再处理上一轮", 20, C.fg, "left", 700);
    lane(x0, 365, sc, o.cpu, p2, "CPU");
    lane(x0, 420, sc, o.gpu, p2, "GPU");
    const a = seg(2, 0.65, 0.8);
    text(x0, 530, "同样的时间：普通循环 3 轮，重叠循环 5 轮", 22, C.green, "left", 700, false, a);
    text(x0, 564, "GPU 前向一轮接一轮，CPU 的调度和处理都藏在前向的时间里", 18, C.muted, "left", 400, false, a);
  }
  const a = seg(1, 0.6, 0.75);
  chip(x0 + 60, 612, "", C.blue, { a, w: 28, h: 20, solid: true }); text(x0 + 80, 618, "调度 + 发射", 16, C.fg, "left", 600, false, a);
  chip(x0 + 230, 612, "", C.purple, { a, w: 28, h: 20, solid: true }); text(x0 + 250, 618, "GPU 前向", 16, C.fg, "left", 600, false, a);
  chip(x0 + 380, 612, "", C.teal, { a, w: 28, h: 20, solid: true }); text(x0 + 400, 618, "处理结果（要等 GPU）", 16, C.fg, "left", 600, false, a);
}

// ---------- 两个前提 ----------
function premises() {
  heading("为什么能提前发射");
  // 前提一：输入不经过 CPU
  const a1 = seg(3, 0, 0.15);
  box(60, 100, 760, 240, "GPU · 引擎 stream", C.purple, { a: a1, top: true, fs: 18 });
  box(90, 170, 150, 60, "第 N 轮前向", C.purple, { a: a1, solid: true, fs: 17 });
  box(290, 170, 100, 60, "采样", C.purple, { a: seg(3, 0.15, 0.3), solid: true, fs: 17 });
  arrow(240, 200, 288, 200, C.purple, seg(3, 0.15, 0.3));
  text(450, 162, "token pool 的一行", 16, C.muted, "left", 600, false, seg(3, 0.3, 0.4));
  const w = seg(3, 0.4, 0.55);
  arrow(390, 200, 662, 200, C.orange, w, { a: w > 0 ? 1 : 0, lw: 2 });
  ["12", "7", "33", "5"].forEach((s, i) => chip(470 + i * 54, 200, s, C.gray, { a: seg(3, 0.3, 0.4), w: 46 }));
  chip(686, 200, "88", C.orange, { a: w, solid: true, w: 46 });
  text(686, 250, "写进下一格", 15, C.orange, "center", 600, false, w);
  const r = seg(3, 0.55, 0.7);
  box(560, 272, 180, 50, "第 N+1 轮输入", C.purple, { a: r, fs: 16 });
  arrow(686, 220, 686, 270, C.orange, r, { lw: 2 });
  box(880, 100, 340, 240, "CPU · 调度器", C.blue, { a: seg(3, 0.2, 0.35), top: true, fs: 18 });
  text(1050, 205, "第 N 轮的 token = ?", 22, C.blue, "center", 700, true, seg(3, 0.25, 0.4));
  text(1050, 250, "不需要知道", 20, C.green, "center", 700, false, seg(3, 0.7, 0.85));
  text(1050, 285, "照样能发射第 N+1 轮", 17, C.muted, "center", 400, false, seg(3, 0.7, 0.85));
  // 前提二：长度提前推进
  if (!after(4)) return;
  const b = seg(4, 0, 0.15);
  box(60, 370, 1160, 220, "", C.blue, { a: b });
  text(90, 410, "CPU 上的请求状态", 19, C.fg, "left", 700, false, b);
  const adv = seg(4, 0.2, 0.35) > 0.5;
  text(90, 460, "发射第 N 轮时立即 complete_one()：", 18, C.muted, "left", 400, false, b);
  text(90, 505, `device_len = ${adv ? 8 : 7}`, 24, adv ? C.green : C.fg, "left", 700, true, b);
  text(90, 545, adv ? "已是第 N 轮之后的值 → 能算位置、分配 KV" : "", 17, C.green, "left", 600, false, seg(4, 0.25, 0.4));
  box(700, 430, 470, 120, "", C.red, { a: seg(4, 0.5, 0.65), dash: true });
  text(935, 478, "第 N 轮采样出 EOS 了吗？", 22, C.red, "center", 700, false, seg(4, 0.5, 0.65));
  text(935, 520, "要等处理第 N 轮时才知道", 18, C.muted, "center", 400, false, seg(4, 0.6, 0.75));
}

// ---------- 多调度一轮 ----------
function extraRound() {
  heading("遇到 EOS 的请求会被多调度一轮");
  const x0 = 180, sc = 160, a0 = seg(5, 0, 0.1);
  const cpu = [[0, SCH, "发射 N+1", C.blue], [2, 2 + PRO, "处理 N", C.teal], [2 + PRO, 2 + PRO + SCH, "发射 N+2", C.blue], [4, 4 + PRO, "处理 N+1", C.teal]];
  const gpu = [[0, 2, "前向 N", C.purple], [2, 4, "前向 N+1", C.purple], [4, 6, "前向 N+2", C.purple]];
  const play = seg(5, 0.05, 0.75) * 6;
  const L = (y, items, label) => {
    text(x0 - 16, y + 30, label, 18, C.muted, "right", 700, false, a0);
    ctx.save(); ctx.globalAlpha = a0; ctx.fillStyle = "#eeeef0"; ctx.fillRect(x0, y + 4, 6 * sc, 42); ctx.restore();
    for (const [s, e, name, col] of items) {
      if (s >= play) continue;
      const e2 = Math.min(e, play);
      box(x0 + s * sc + 1, y + 4, (e2 - s) * sc - 2, 42, "", col, { r: 6, solid: true, a: 0.9 });
      if (e2 - s > 0.5) text(x0 + (s + e2) / 2 * sc, y + 31, name, 16, "#fff", "center", 700);
    }
  };
  L(150, cpu, "CPU");
  L(260, gpu, "GPU");
  // 请求 r 在各轮里的 token
  const rowY = 350;
  text(x0 - 16, rowY + 6, "请求 r", 18, C.muted, "right", 700, false, a0);
  const t1 = play > 1.0 ? 1 : 0, t2 = play > 3.0 ? 1 : 0, t3 = play > 5 ? 1 : 0;
  chip(x0 + 1 * sc, rowY, "EOS", C.red, { a: t1, solid: true, w: 70 });
  chip(x0 + 3 * sc, rowY, "多算的", C.gray, { a: t2, w: 84, dash: true, fs: 15, mono: false });
  text(x0 + 5 * sc, rowY + 6, "（已不在 batch 里）", 16, C.muted, "center", 400, false, t3);
  // 注释
  const a1 = seg(5, 0.35, 0.5);
  arrow(x0 + 2.35 * sc, 205, x0 + 2.35 * sc, 440, C.red, a1, { lw: 2, dash: true });
  text(x0 + 2.35 * sc, 470, "处理 N 时才发现 EOS，释放 r", 18, C.red, "center", 700, false, a1);
  text(x0 + 2.35 * sc, 498, "可此时前向 N+1 已经带着 r 在跑", 17, C.fg, "center", 600, false, seg(5, 0.5, 0.65));
  const a2 = seg(5, 0.78, 0.92);
  text(x0 + 4.35 * sc, 560, "处理 N+1：r 的结果已过期", 18, C.orange, "center", 700, false, a2);
  arrow(x0 + 4.35 * sc, 540, x0 + 4.35 * sc, 205, C.orange, a2, { lw: 2, dash: true });
}

// ---------- 四个问题 ----------
const ISSUES = [
  ["结束标记提前一个 token", "max_tokens=3 只收到 2 个", "len(input_ids) >= max_device_len"],
  ["EOS 之后的过期消息", "detokenizer 状态泄漏", "跳过上一轮已结束的请求"],
  ["请求槽过早复用", "新提示词被在途写入覆盖", "推迟到下一次处理结果再归还"],
  ["prefill 在途时 abort", "页既空闲又在缓存树里", "abort 时加入 finished_reqs"],
];
function issues() {
  heading("复刻时发现的四个问题");
  ISSUES.forEach(([t, bad, fix], i) => {
    const x = 70 + (i % 2) * 590, y = 105 + Math.floor(i / 2) * 250;
    const a = seg(6, 0.2 + i * 0.12, 0.35 + i * 0.12);
    const active = during(7 + i), done = after(8 + i);
    const d = after(7 + i) ? seg(7 + i, 0.1, 0.3) : 0;
    box(x, y, 550, 220, "", active ? C.orange : C.gray, { a: a * (after(7) && !active && !done ? 0.55 : 1), lw: active ? 3 : 2, glow: active ? 0.5 : 0 });
    text(x + 26, y + 46, `问题 ${i + 1}`, 18, C.muted, "left", 700, false, a);
    text(x + 26, y + 84, t, 26, C.fg, "left", 800, false, a);
    text(x + 26, y + 136, "✗ " + bad, 19, C.red, "left", 700, false, d);
    text(x + 26, y + 180, "✓ " + fix, 19, C.green, "left", 700, i === 0 || i === 3 ? true : false, seg(7 + i, 0.55, 0.75));
  });
}

function outro() {
  const a = seg(11, 0, 0.2);
  heading("每个修正都有测试守着", a);
  ["test_overlap_respects_max_tokens_exactly", "test_stale_result_after_eos_is_dropped",
   "test_request_slot_is_not_reused_while_a_batch_using_it_is_in_flight", "test_abort_while_prefill_is_in_flight"]
    .forEach((s, i) => {
      const b = seg(11, 0.05 + i * 0.1, 0.2 + i * 0.1);
      chip(170, 190 + i * 70, "✓", C.green, { a: b, solid: true, w: 40, mono: false });
      text(210, 198 + i * 70, s, 22, C.fg, "left", 600, true, b);
    });
  text(170, 510, "tests/test_ch11_overlap.py · 第 11 章", 20, C.muted, "left", 600, false, seg(11, 0.5, 0.7));
}

function draw() {
  if (!after(1)) { title("重叠调度", "让 CPU 的调度开销藏进 GPU 的计算时间", 1 - seg(0, 0.88, 1)); return; }
  if (!after(3)) { timelines(); return; }
  if (!after(5)) { premises(); return; }
  if (!after(6)) { extraRound(); return; }
  if (!after(11)) { issues(); return; }
  outro();
}
