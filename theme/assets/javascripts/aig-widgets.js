// 章节里的交互小工具。页面里放一个占位：<div class="aig-widget" data-widget="kv-calc"></div>
//   kv-calc   KV Cache 与显存：一张卡能同时服务多少个请求（大模型原理 · KV Cache）
//   roofline  矩阵乘的屋顶线：batch 变大，瓶颈从访存变成算力（大模型原理 · 性能与服务中的数学）
//   mask      注意力掩码：因果、带历史、滑动窗口、序列打包、前缀双向，以及按块跳过（大模型原理 · 注意力机制）
//   pipeline  流水线的调度与气泡：GPipe 与 1F1B（分布式训练 · 流水线并行）
// 字节数按 1024 进位（和正文里"每个 token 112 KB"的算法一致）。
(function () {
  var GPUS = {                      // 显存 GB、带宽 TB/s、BF16 稠密 TFLOPS、FP8 稠密 TFLOPS（与推理系统手册的硬件速查一致）
    "H100 SXM": [80, 3.35, 989, 1979], "H200": [141, 4.8, 989, 1979], "A100 80GB": [80, 2.0, 312, 0],
    "H20": [96, 4.0, 148, 296], "L40S": [48, 0.86, 362, 733], "B200": [180, 8.0, 2250, 4500]
  };
  var KB = 1024, MB = KB * KB, GB = MB * KB;

  function fmtBytes(b) {
    if (b >= 1024 * GB) return (b / 1024 / GB).toFixed(2) + " TB";
    if (b >= GB) return (b / GB).toFixed(b >= 100 * GB ? 0 : 1) + " GB";
    if (b >= MB) return (b / MB).toFixed(1) + " MB";
    return (b / KB).toFixed(1) + " KB";
  }
  function fmtInt(n) { return Math.round(n).toLocaleString("zh-CN"); }
  function row(label, input, wide) { return '<label class="aw-row' + (wide ? " aw-wide" : "") + '"><span>' + label + "</span>" + input + "</label>"; }
  function select(id, opts, v) {
    return '<select data-k="' + id + '">' + opts.map(function (o) {
      return '<option value="' + o + '"' + (o === v ? " selected" : "") + ">" + o + "</option>";
    }).join("") + "</select>";
  }
  function num(id, v, min, max, step) {
    return '<input type="number" data-k="' + id + '" value="' + v + '" min="' + min + '" max="' + max + '" step="' + (step || 1) + '">';
  }
  function range(id, v, min, max) {
    return '<input type="range" data-k="' + id + '" value="' + v + '" min="' + min + '" max="' + max + '" step="1"><b data-v="' + id + '"></b>';
  }
  function bind(box, update) {
    [].forEach.call(box.querySelectorAll("[data-k]"), function (i) { i.addEventListener("input", update); });
    update();
  }
  function input(box, k) { return box.querySelector('[data-k="' + k + '"]'); }
  function val(box, k) {
    var e = input(box, k);
    if (e.tagName === "SELECT") return e.value;
    var v = +e.value, lo = +e.min, hi = +e.max;
    return isFinite(v) && e.value !== "" ? Math.min(hi, Math.max(lo, v)) : lo;    // 输入框清空或越界时按边界算
  }
  function show(box, k, text) { box.querySelector('[data-v="' + k + '"]').textContent = text; }
  function svgText(x, y, s, anchor) { return '<text x="' + x + '" y="' + y + '" class="aw-t"' + (anchor ? ' text-anchor="' + anchor + '"' : "") + ">" + s + "</text>"; }

  // ---------------------------------------------------------------- KV Cache 与显存
  var MODELS = {                    // 层数、KV 头数、头维、总参数量、每个 token 激活的参数量（十亿）；KV 头数为 0 表示 MLA：每层只缓存 512 维潜向量 + 64 维 RoPE 键
    "Qwen3-0.6B": [28, 8, 128, 0.6, 0.6], "Qwen3-8B": [36, 8, 128, 8.2, 8.2], "LLaMA-3-8B": [32, 8, 128, 8.0, 8.0],
    "Qwen2.5-7B": [28, 4, 128, 7.6, 7.6], "LLaMA-3-70B": [80, 8, 128, 70.6, 70.6], "DeepSeek-V3（MLA）": [61, 0, 0, 671, 37]
  };
  function kvCalc(box) {
    box.innerHTML = '<div class="aw-title">KV Cache 与显存：一张卡能同时服务多少个请求</div><div class="aw-grid">' +
      row("模型", select("model", Object.keys(MODELS), "LLaMA-3-8B")) + row("GPU", select("gpu", Object.keys(GPUS), "H100 SXM")) +
      row("卡数（张量并行）", select("tp", ["1", "2", "4", "8"], "1")) + row("权重精度", select("wdt", ["BF16", "FP8", "INT4"], "BF16")) +
      row("KV 精度", select("kdt", ["BF16", "FP8"], "BF16")) + row("平均上下文（token）", num("ctx", 4096, 1, 1048576, 256)) +
      row("每张卡预留（GB）", num("reserve", 8, 0, 64)) + '</div><div class="aw-out"></div>';
    var out = box.querySelector(".aw-out");
    bind(box, function () {
      var m = MODELS[val(box, "model")], g = GPUS[val(box, "gpu")], tp = +val(box, "tp");
      var kb = val(box, "kdt") === "FP8" ? 1 : 2, wb = { BF16: 2, FP8: 1, INT4: 0.5 }[val(box, "wdt")];
      var mla = !m[1], ctx = val(box, "ctx");
      // 张量并行按 KV 头切分；KV 头比卡少时每张卡至少放一个头（多出来的卡上是副本）。MLA 的潜向量所有头共用，每张卡都要存完整的一份
      var headsPerCard = mla ? 0 : Math.max(1, m[1] / tp);
      var perTok = mla ? m[0] * 576 * kb : 2 * m[0] * headsPerCard * m[2] * kb;
      var weights = m[3] * 1e9 * wb / tp, free = g[0] * GB - weights - val(box, "reserve") * GB;
      var perReq = perTok * ctx;
      var html = "<p>每张卡上每个 token 的 KV：<b>" + fmtBytes(perTok) + "</b>（" + (mla ? m[0] + " 层 × 576 维 × " + kb + " 字节，MLA 只缓存潜向量" :
        "2 × " + m[0] + " 层 × " + (headsPerCard % 1 ? headsPerCard.toFixed(1) : headsPerCard) + " 个 KV 头 × " + m[2] + " 维 × " + kb + " 字节") +
        "）；一个 " + fmtInt(ctx) + " token 的请求占 " + fmtBytes(perReq) + "</p>";
      if (!mla && tp > m[1]) html += '<p class="aw-note">卡数超过了 KV 头数（' + m[1] + "），多出来的卡只能存 KV 头的副本，每张卡的 KV 不再减少。</p>";
      if (mla && tp > 1) html += '<p class="aw-note">MLA 的潜向量是所有头共用的，张量并行切不开它，每张卡都存完整的一份——所以部署 MLA 模型常把注意力改成数据并行（DP attention）：不同的卡服务不同的请求。</p>';
      html += "<p>每张卡上的权重 " + fmtBytes(weights) + "，留给 KV 的空间：<b>" + (free > 0 ? fmtBytes(free) : "放不下权重") + "</b></p>";
      if (free > 0) {
        var n = Math.floor(free / perReq), tokens = Math.floor(free / perTok);
        html += "<p>能同时放下 <b>" + fmtInt(n) + "</b> 个这样长的请求（共 " + fmtInt(tokens) + " 个 token 的 KV）" +
          (n < 1 ? "——一个请求都放不下，要么增加卡数，要么缩短上下文" : "") + "</p>";
        if (n >= 1) {
          // decode 一步：每张卡读一遍自己的权重和所有请求的 KV；计算量约 2 × 激活参数 × 请求数（忽略注意力本身的计算）
          var bytes = weights + perReq * n, flops = 2 * m[4] * 1e9 * n / tp;
          var peak = (val(box, "wdt") === "FP8" && g[3] ? g[3] : g[2]) * 1e12;
          var tMem = bytes / (g[1] * 1e12) * 1e3, tCmp = flops / peak * 1e3, ms = Math.max(tMem, tCmp);
          html += '<p class="aw-note">满载 decode 时每一步每张卡要读 ' + fmtBytes(bytes) + "（权重加上所有请求的 KV），按带宽至少 " + tMem.toFixed(1) +
            " ms；要算 " + (flops / 1e12).toFixed(1) + " TFLOP，按算力至少 " + tCmp.toFixed(1) + " ms。所以一步至少 " + ms.toFixed(1) + " ms（" +
            (tMem >= tCmp ? "访存受限" : "计算受限") + "），整个实例的吞吐上限约 " + fmtInt(n / ms * 1e3) + " token/s。KV 越大，能并发的请求越少，每一步也越慢。</p>";
        }
      } else {
        html += '<p class="aw-note">权重放不下：增加卡数，或者换更低的权重精度。</p>';
      }
      out.innerHTML = html;
    });
  }

  // ---------------------------------------------------------------- 屋顶线
  function roofline(box) {
    box.innerHTML = '<div class="aw-title">矩阵乘的屋顶线：[m, k] × [k, n]，m 是一个 batch 里的 token 数</div><div class="aw-grid">' +
      row("GPU", select("gpu", Object.keys(GPUS), "H100 SXM")) + row("精度", select("dt", ["BF16", "FP8"], "BF16")) +
      row("k（输入维）", num("k", 4096, 64, 65536, 64)) + row("n（输出维）", num("n", 4096, 64, 65536, 64)) +
      row("m", range("lm", 0, 0, 14), true) + '</div><svg class="aw-chart" viewBox="0 0 560 228"></svg><div class="aw-out"></div>';
    var out = box.querySelector(".aw-out"), svg = box.querySelector("svg");
    // 横轴：算术强度 1～10⁴ FLOP/字节；纵轴：1～10⁴ TFLOPS；都取对数
    var X = function (a) { return 52 + Math.log10(Math.min(Math.max(a, 1), 1e4)) / 4 * 490; };
    var Y = function (tf) { return 196 - Math.log10(Math.min(Math.max(tf, 1), 1e4)) / 4 * 184; };
    bind(box, function () {
      var g = GPUS[val(box, "gpu")], fp8 = val(box, "dt") === "FP8";
      var m = Math.pow(2, val(box, "lm")), k = val(box, "k"), n = val(box, "n");
      show(box, "lm", String(m));
      var peak = (fp8 ? g[3] : g[2]) * 1e12, bw = g[1] * 1e12, b = fp8 ? 1 : 2;
      if (!peak) { out.innerHTML = "<p>这张卡没有 FP8 Tensor Core，换一张卡或者选 BF16。</p>"; svg.innerHTML = ""; return; }
      var flops = 2 * m * k * n, bytes = b * (m * k + k * n) + 2 * m * n;   // 输出按 BF16 写回
      var ai = flops / bytes, ridge = peak / bw, t = Math.max(flops / peak, bytes / bw), got = flops / t;
      var s = "";
      for (var e = 0; e <= 4; e++) {
        var v = Math.pow(10, e);
        s += '<line x1="' + X(v) + '" y1="12" x2="' + X(v) + '" y2="196" class="aw-gl"/><line x1="52" y1="' + Y(v) + '" x2="542" y2="' + Y(v) + '" class="aw-gl"/>' +
          svgText(X(v), 210, v, "middle") + svgText(46, Y(v) + 4, v, "end");
      }
      s += '<line x1="52" y1="196" x2="542" y2="196" class="aw-axis"/><line x1="52" y1="196" x2="52" y2="12" class="aw-axis"/>' +
        '<path class="aw-roof" d="M ' + X(1) + " " + Y(bw / 1e12) + " L " + X(ridge) + " " + Y(peak / 1e12) + " L " + X(1e4) + " " + Y(peak / 1e12) + '"/>' +
        '<line x1="' + X(ridge) + '" y1="' + Y(peak / 1e12) + '" x2="' + X(ridge) + '" y2="196" class="aw-dash"/>' +
        svgText(X(ridge) + 5, 190, "屋脊点 " + ridge.toFixed(0)) + svgText(X(1e4) - 4, Y(peak / 1e12) - 7, "峰值 " + (peak / 1e12) + " TFLOPS", "end") +
        '<circle cx="' + X(ai) + '" cy="' + Y(got / 1e12) + '" r="6" class="aw-dot"/>' +
        svgText(300, 225, "算术强度（FLOP/字节）", "middle") + '<text x="12" y="104" class="aw-t" transform="rotate(-90 12 104)" text-anchor="middle">TFLOPS</text>';
      svg.innerHTML = s;
      var bound = ai < ridge;
      var work = flops < 1e9 ? (flops / 1e6).toFixed(1) + " MFLOP" : flops < 1e12 ? (flops / 1e9).toFixed(flops < 1e10 ? 2 : 0) + " GFLOP" : (flops / 1e12).toFixed(2) + " TFLOP";
      out.innerHTML = "<p>计算量 " + work + "，读写 " + fmtBytes(bytes) + "，算术强度 <b>" + ai.toFixed(1) +
        "</b> FLOP/字节（这张卡的屋脊点是 " + ridge.toFixed(0) + "）</p><p><b>" + (bound ? "访存受限" : "计算受限") + "</b>：至少 " +
        (t * 1e6 < 1000 ? (t * 1e6).toFixed(1) + " μs" : (t * 1e3).toFixed(2) + " ms") + "，最多发挥 " + (got / 1e12).toFixed(got < 1e13 ? 1 : 0) +
        " TFLOPS（峰值的 " + (100 * got / peak).toFixed(got / peak < 0.1 ? 1 : 0) + "%）</p>" +
        '<p class="aw-note">' + (bound ? "m 远小于屋脊点时，耗时几乎全花在读权重上，m 从 1 加到几十，时间几乎不变——这就是 decode 要攒 batch 的原因。" :
          "越过屋脊点之后，再增大 m 只会按比例增加时间：prefill、大 batch 的 decode 都在这一侧，这时要比的是峰值算力，FP8 能再快一倍。") + "</p>";
    });
  }

  // ---------------------------------------------------------------- 注意力掩码
  var MASKS = {
    "因果": ["", "训练和 prefill 的默认掩码：第 i 个 token 只能看到自己和之前的 token。"],
    "带历史的 prefill": ["已缓存的 token 数", "有 KV Cache（分块 prefill、多轮对话）时，新 token 能看到全部历史：对角线右移了历史长度，也就是 tril(diagonal=S−T)。decode 是只有一个新 token 的特例，那一行全部可见。"],
    "滑动窗口": ["窗口 W", "每个 token 只看最近 W 个 token（包括自己）。窗口之外的 KV 可以直接丢掉，这些层的 KV Cache 大小固定为 W。"],
    "序列打包": ["文档数", "把几篇文档拼成一个训练样本时，文档之间必须互不可见，否则模型会学到跨文档的虚假关联。FlashAttention 的变长接口用 cu_seqlens 标出每篇文档的边界，直接按文档分开算。"],
    "前缀双向": ["前缀长度", "前缀内部互相可见（比如 Prefix-LM 的输入部分、一些多模态模型的图像 token），前缀之后仍然是因果的。"]
  };
  function maskFn(kind, T, a) {
    if (kind === "因果") return function (i, j) { return j <= i; };
    if (kind === "带历史的 prefill") return function (i, j) { return i >= a ? j <= i : null; };   // null：这一行不是 query
    if (kind === "滑动窗口") return function (i, j) { return j <= i && i - j < a; };
    if (kind === "前缀双向") return function (i, j) { return j <= i || (i < a && j < a); };
    var w = [0.45, 0.3, 0.25, 0.2].slice(0, a), sum = 0, doc = [], cut = [], acc = 0;   // 序列打包：a 篇长短不一的文档
    w.forEach(function (x) { sum += x; });
    w.forEach(function (x) { acc += x; cut.push(Math.round(acc / sum * T)); });
    for (var i = 0, d = 0; i < T; i++) { while (i >= cut[d]) d++; doc.push(d); }
    return function (i, j) { return j <= i && doc[i] === doc[j]; };
  }
  function mask(box) {
    box.innerHTML = '<div class="aw-title">注意力掩码：哪个 query（行）能看到哪个 key（列）</div><div class="aw-grid">' +
      row("掩码", select("kind", Object.keys(MASKS), "因果")) + row("块大小", select("blk", ["2", "4", "8"], "4")) +
      row("序列长度", range("T", 16, 8, 32), true) + row("", range("a", 4, 1, 16), true) +
      '</div><svg class="aw-chart aw-mask"></svg><div class="aw-out"></div>';
    var out = box.querySelector(".aw-out"), svg = box.querySelector("svg"), param = input(box, "a").parentNode;
    var HI = { "带历史的 prefill": function (T) { return T - 1; }, "滑动窗口": function (T) { return T; },
               "序列打包": function () { return 4; }, "前缀双向": function (T) { return T; } };
    bind(box, function () {
      var kind = val(box, "kind"), T = val(box, "T"), blk = +val(box, "blk"), ra = input(box, "a");
      ra.max = HI[kind] ? HI[kind](T) : 1;
      var a = val(box, "a");
      param.hidden = !MASKS[kind][0];
      param.firstChild.textContent = MASKS[kind][0];
      show(box, "T", T);
      show(box, "a", a);
      // 行是 query：带历史时只有新输入的 token 是 query（行号是它们的绝对位置），列是全部 key
      var f = maskFn(kind, T, a), hist = kind === "带历史的 prefill", rows = [], i, j;
      for (i = hist ? a : 0; i < T; i++) rows.push(i);
      var q = rows.length, c = Math.min(18, 420 / T), x0 = 26, y0 = hist ? 40 : 22, S = "";
      var on = 0, full = 0, part = 0, skip = 0;
      rows.forEach(function (i, r) {
        for (var j = 0; j < T; j++) {
          var m = f(i, j);
          if (m) on++;
          S += '<rect x="' + (x0 + j * c + 0.5) + '" y="' + (y0 + r * c + 0.5) + '" width="' + (c - 1) + '" height="' + (c - 1) + '" rx="1.5" class="' + (m ? "aw-on" : "aw-off") + '"/>';
        }
      });
      for (var br = 0; br < q; br += blk) {                  // 按块统计：整块看不见的跳过，整块可见的直接算，其余要逐元素加掩码
        for (var bj = 0; bj < T; bj += blk) {
          var seen = 0, cells = 0;
          for (var r = br; r < Math.min(q, br + blk); r++) for (j = bj; j < Math.min(T, bj + blk); j++) { cells++; if (f(rows[r], j)) seen++; }
          if (!seen) skip++; else if (seen === cells) full++; else part++;
          S += '<rect x="' + (x0 + bj * c) + '" y="' + (y0 + br * c) + '" width="' + (Math.min(blk, T - bj) * c) + '" height="' + (Math.min(blk, q - br) * c) +
            '" class="aw-blk' + (seen ? "" : " skip") + '"/>';
        }
      }
      var tick = T > 16 ? 4 : 2;
      for (j = 0; j < T; j += tick) S += svgText(x0 + j * c + c / 2, y0 - 7, j, "middle");
      rows.forEach(function (i, r) { if (i % tick === 0 || hist && r === 0) S += svgText(x0 - 5, y0 + r * c + c / 2 + 4, i, "end"); });
      if (hist) {                                            // 列的上方标出哪些 key 来自 KV Cache
        var hw = a * c, nw = (T - a) * c;
        S += '<path class="aw-brace" d="M ' + (x0 + 1) + " 18 V 12 H " + (x0 + hw - 2) + ' V 18"/>' +
          '<path class="aw-brace" d="M ' + (x0 + hw + 2) + " 18 V 12 H " + (x0 + T * c - 1) + ' V 18"/>';
        if (hw >= 24) S += svgText(x0 + hw / 2, 8, hw >= 100 ? "KV Cache 里的历史" : "历史", "middle");
        if (nw >= 24) S += svgText(x0 + hw + nw / 2, 8, nw >= 56 ? "本次输入" : "新", "middle");
      }
      var vw = x0 + T * c + 4, vh = y0 + q * c + 4;
      svg.setAttribute("viewBox", "0 0 " + vw + " " + vh);
      svg.setAttribute("width", Math.round(vw * 1.3));
      svg.innerHTML = S;
      out.innerHTML = "<p>" + q + " 个 query × " + T + " 个 key，可见的有 <b>" + on + "</b> 对（" + (100 * on / (q * T)).toFixed(0) + "%）。按 " + blk + "×" + blk +
        " 的块看：<b>" + full + "</b> 块整块计算，<b>" + part + "</b> 块要逐元素加掩码，<b>" + skip + "</b> 块整块跳过（虚线框）</p>" +
        '<p class="aw-note">' + MASKS[kind][1] + "FlashAttention、FlexAttention 都按块处理：整块看不见的直接跳过，掩码越稀疏，省下的计算越多。</p>";
    });
  }

  // ---------------------------------------------------------------- 流水线
  function schedule(p, m, oneFOneB) {
    var order = [];
    for (var s = 0; s < p; s++) {
      var seq = [], i;
      if (!oneFOneB) {
        for (i = 0; i < m; i++) seq.push(["F", i]);
        for (i = 0; i < m; i++) seq.push(["B", i]);
      } else {                                               // 先做 p-s-1 个前向"预热"，之后一前一后交替
        var warm = Math.min(p - s - 1, m), f = warm, b = 0;
        for (i = 0; i < warm; i++) seq.push(["F", i]);
        while (b < m) { if (f < m) seq.push(["F", f++]); seq.push(["B", b++]); }
      }
      order.push(seq);
    }
    // 按依赖排时间：F(s,i) 等 F(s-1,i)，B(s,i) 等 B(s+1,i)，最后一个 stage 的 B 等自己的 F；前向耗时 1、反向耗时 2
    var done = {}, free = [], ops = [], live = [], peak = [];
    for (s = 0; s < p; s++) { free.push(0); ops.push([]); live.push(0); peak.push(0); }
    for (var guard = 0; order.some(function (o) { return o.length; }) && guard < 100000; guard++) {
      for (s = 0; s < p; s++) {
        if (!order[s].length) continue;
        var kind = order[s][0][0], mb = order[s][0][1];
        var dep = kind === "F" ? (s > 0 ? "F" + (s - 1) + "," + mb : null) : (s < p - 1 ? "B" + (s + 1) + "," + mb : "F" + s + "," + mb);
        if (dep && !(dep in done)) continue;
        var start = Math.max(free[s], dep ? done[dep] : 0), end = start + (kind === "F" ? 1 : 2);
        done[kind + s + "," + mb] = free[s] = end;
        ops[s].push([kind, mb, start, end]);
        live[s] += kind === "F" ? 1 : -1;                    // 做完前向、还没做反向的 micro-batch 要保存激活
        peak[s] = Math.max(peak[s], live[s]);
        order[s].shift();
      }
    }
    return { ops: ops, total: Math.max.apply(null, free), peak: peak };
  }
  function pipeline(box) {
    box.innerHTML = '<div class="aw-title">流水线的调度与气泡（前向耗时 1，反向耗时 2）</div><div class="aw-grid">' +
      row("stage 数 p", range("p", 4, 2, 8), true) + row("micro-batch 数 m", range("m", 8, 1, 16), true) +
      row("调度", select("kind", ["1F1B", "GPipe"], "1F1B")) + '</div><svg class="aw-chart"></svg><div class="aw-out"></div>';
    var svg = box.querySelector("svg"), out = box.querySelector(".aw-out");
    bind(box, function () {
      var p = val(box, "p"), m = val(box, "m"), one = val(box, "kind") === "1F1B", r = schedule(p, m, one);
      show(box, "p", p);
      show(box, "m", m);
      var unit = Math.min(24, 530 / r.total), h = 20, S = "";
      for (var s = 0; s < p; s++) {
        var y = 2 + s * (h + 5);
        S += svgText(0, y + 14, "S" + s) + '<rect x="28" y="' + y + '" width="' + r.total * unit + '" height="' + h + '" rx="3" class="aw-idle"/>';
        r.ops[s].forEach(function (o) {
          S += '<rect x="' + (28 + o[2] * unit + 0.5) + '" y="' + y + '" width="' + ((o[3] - o[2]) * unit - 1) + '" height="' + h + '" rx="3" class="aw-' + o[0].toLowerCase() + '"/>';
          if (unit >= 13) S += svgText(28 + (o[2] + o[3]) / 2 * unit, y + 14, o[0] + o[1], "middle");
        });
      }
      svg.setAttribute("viewBox", "0 0 560 " + (p * (h + 5) + 2));
      svg.innerHTML = S;
      out.innerHTML = "<p>总时间 " + r.total + " 个单位，每个 stage 真正在算的是 " + 3 * m + " 个：气泡 <b>" + (100 * (1 - 3 * m / r.total)).toFixed(1) +
        "%</b>，和公式 (p−1)/(m+p−1) = " + (100 * (p - 1) / (m + p - 1)).toFixed(1) + "% 一致</p><p>stage 0 最多同时保存 <b>" + r.peak[0] + "</b> 份激活" +
        (one ? "：1F1B 不超过 stage 数，与 m 无关" : "：GPipe 等于 micro-batch 数，m 越大越占显存") + "</p>" +
        '<p class="aw-note">两种调度的气泡一样大。想让气泡变小，只能增大 m（受全局 batch 限制），或者用交错调度、零气泡调度把空闲填上。</p>';
    });
  }

  var WIDGETS = { "kv-calc": kvCalc, roofline: roofline, mask: mask, pipeline: pipeline };
  function init() {
    [].forEach.call(document.querySelectorAll(".aig-widget[data-widget]:not([data-ready])"), function (box) {
      var fn = WIDGETS[box.dataset.widget];
      if (!fn) return;
      box.setAttribute("data-ready", "");
      fn(box);
    });
  }
  init();
  if (window.document$ && window.document$.subscribe) window.document$.subscribe(init);   // 开了即时导航时，换页后再初始化
})();
