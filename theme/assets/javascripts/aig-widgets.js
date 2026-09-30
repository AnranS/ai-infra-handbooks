// 章节里的交互小工具。页面里放一个占位：<div class="aig-widget" data-widget="kv-calc"></div>
//   kv-calc   KV Cache 与显存：一张卡能同时服务多少个请求（大模型原理 · KV Cache）
//   roofline  矩阵乘的屋顶线：batch 变大，瓶颈从访存变成算力（大模型原理 · 性能与服务中的数学）
//   mask      注意力掩码：因果、带历史、滑动窗口、序列打包、前缀双向，以及按块跳过（大模型原理 · 注意力机制）
//   pipeline  流水线的调度与气泡：GPipe 与 1F1B（分布式训练 · 流水线并行）
//   linmap    线性变换：2×2 矩阵把网格变成什么样（大模型原理 · 线性代数）
//   lowrank   低秩近似：保留几个奇异值就够（大模型原理 · 线性代数）
//   softmax   softmax 与采样：温度、top-k、top-p（大模型原理 · 概率与采样）
//   graddesc  梯度下降：学习率、动量与条件数（大模型原理 · 微积分与反向传播）
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
  function range2(id, v, min, max) {      // 和 range 一样，但值由各个小工具自己格式化
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

  // ---------------------------------------------------------------- 线性变换：矩阵把网格变成什么样
  function linmap(box) {
    var PRESETS = {
      "拉伸（对角阵）": [1.6, 0, 0, 0.6], "旋转 30°": [0.866, -0.5, 0.5, 0.866],
      "剪切": [1, 1, 0, 1], "投影到一条线（秩 1）": [1, 1, 0.5, 0.5],
      "翻折（行列式为负）": [0, 1, 1, 0], "单位阵": [1, 0, 0, 1]
    };
    box.innerHTML = '<div class="aw-title">线性变换：一个 2×2 矩阵把平面变成什么样</div><div class="aw-grid">' +
      row("常见变换", select("preset", Object.keys(PRESETS), "剪切")) +
      row("动画进度 t", range2("t", 100, 0, 100) + '<button type="button" data-k="play" class="aw-btn">播放</button>') +
      row("a（i 的 x）", num("a", 1, -3, 3, 0.1)) + row("b（j 的 x）", num("b", 1, -3, 3, 0.1)) +
      row("c（i 的 y）", num("c", 0, -3, 3, 0.1)) + row("d（j 的 y）", num("d", 1, -3, 3, 0.1)) +
      '</div><svg class="aw-chart aw-lin" viewBox="0 0 560 300"></svg><div class="aw-out"></div>';
    var svg = box.querySelector("svg"), out = box.querySelector(".aw-out"), timer = null;

    function draw() {
      var t = val(box, "t") / 100;
      var a = 1 + (val(box, "a") - 1) * t, b = val(box, "b") * t,
          c = val(box, "c") * t, d = 1 + (val(box, "d") - 1) * t;
      var cx = 280, cy = 150, u = 46;                       // 画布中心与单位长度
      function P(x, y) { return [cx + (a * x + b * y) * u, cy - (c * x + d * y) * u]; }
      var S = "", n = 9;
      for (var i = -n; i <= n; i++) {                        // 变换后的网格
        var p1 = P(i, -n), p2 = P(i, n), p3 = P(-n, i), p4 = P(n, i);
        var cls = i === 0 ? "aw-axis" : "aw-gl";
        S += '<line x1="' + p1[0].toFixed(1) + '" y1="' + p1[1].toFixed(1) + '" x2="' + p2[0].toFixed(1) +
             '" y2="' + p2[1].toFixed(1) + '" class="' + cls + '"/>';
        S += '<line x1="' + p3[0].toFixed(1) + '" y1="' + p3[1].toFixed(1) + '" x2="' + p4[0].toFixed(1) +
             '" y2="' + p4[1].toFixed(1) + '" class="' + cls + '"/>';
      }
      var o = P(0, 0), ei = P(1, 0), ej = P(0, 1), ij = P(1, 1);
      S += '<polygon points="' + [o, ei, ij, ej].map(function (p) { return p[0].toFixed(1) + "," + p[1].toFixed(1); }).join(" ") +
           '" class="aw-area"/>';                            // 单位正方形被变成的平行四边形
      S += '<line x1="' + o[0] + '" y1="' + o[1] + '" x2="' + ei[0].toFixed(1) + '" y2="' + ei[1].toFixed(1) + '" class="aw-i"/>';
      S += '<line x1="' + o[0] + '" y1="' + o[1] + '" x2="' + ej[0].toFixed(1) + '" y2="' + ej[1].toFixed(1) + '" class="aw-j"/>';
      S += '<circle cx="' + ei[0].toFixed(1) + '" cy="' + ei[1].toFixed(1) + '" r="4" class="aw-idot"/>';
      S += '<circle cx="' + ej[0].toFixed(1) + '" cy="' + ej[1].toFixed(1) + '" r="4" class="aw-jdot"/>';
      S += svgText(ei[0] + 10, ei[1] + 4, "i → (" + a.toFixed(2) + ", " + c.toFixed(2) + ")", "start");
      S += svgText(ej[0] + 10, ej[1] + 4, "j → (" + b.toFixed(2) + ", " + d.toFixed(2) + ")", "start");
      svg.innerHTML = S;

      var det = a * d - b * c;
      var tr = a + d, disc = tr * tr - 4 * det;
      var eig = disc >= 0
        ? "特征值 " + ((tr + Math.sqrt(disc)) / 2).toFixed(2) + " 和 " + ((tr - Math.sqrt(disc)) / 2).toFixed(2) +
          "：有两条方向不变的直线，向量只被拉伸"
        : "特征值是复数：没有方向不变的实向量，这个变换里有旋转成分";
      out.innerHTML = "<p>行列式 <b>" + det.toFixed(3) + "</b>：单位正方形的面积变成了它的 " + Math.abs(det).toFixed(2) + " 倍" +
        (det < 0 ? "，并且被<b>翻折</b>了（左右手性反过来）" : det === 0 ? "，整个平面被压扁到一条线上（<b>秩 1</b>，不可逆）" : "") + "</p>" +
        "<p>" + eig + "</p>" +
        '<p class="aw-note">矩阵乘以一个向量，就是把它按 i、j 的新位置重新组合：Wx = x₁·(列 1) + x₂·(列 2)。' +
        '线性层做的就是这件事，只不过维度是几千而不是二。行列式为 0 对应秩亏，正是 LoRA 假设"增量只占几个方向"的几何含义。</p>';
    }
    function stop() { if (timer) { clearInterval(timer); timer = null; box.querySelector('[data-k="play"]').textContent = "播放"; } }
    box.querySelector('[data-k="preset"]').addEventListener("change", function () {
      var m = PRESETS[val(box, "preset")];
      ["a", "b", "c", "d"].forEach(function (k, i) { input(box, k).value = m[i]; });
      draw();
    });
    box.querySelector('[data-k="play"]').addEventListener("click", function () {
      if (timer) return stop();
      this.textContent = "暂停";
      input(box, "t").value = 0;
      timer = setInterval(function () {
        if (!box.isConnected) return stop();                   // 即时导航换页后别再动
        var v = +input(box, "t").value + 2;
        input(box, "t").value = Math.min(100, v);
        show(box, "t", (v / 100).toFixed(2));
        draw();
        if (v >= 100) stop();
      }, 30);
    });
    bind(box, function () { show(box, "t", (val(box, "t") / 100).toFixed(2)); draw(); });
  }

  // ---------------------------------------------------------------- 低秩近似：保留几个奇异值就够
  function lowrank(box) {
    var N = 24;
    function noise(i, j) {                                   // 确定性伪随机：每个格子独立，矩阵是满秩的
      var x = Math.sin(i * 127.1 + j * 311.7) * 43758.5453;
      return 2 * (x - Math.floor(x)) - 1;
    }
    function build(kind) {                                   // 造一个 N×N 的"图案"矩阵
      var M = [];
      for (var i = 0; i < N; i++) {
        M[i] = [];
        for (var j = 0; j < N; j++) {
          var low = Math.sin(i / 3) * Math.cos(j / 4) + 0.6 * Math.sin(i / 7) * Math.cos(j / 5);
          if (kind === "低秩图案（两个方向）") M[i][j] = low;
          else if (kind === "接近低秩 + 噪声") M[i][j] = low + 0.22 * noise(i, j);
          else M[i][j] = noise(i, j);                          // 满秩噪声
        }
      }
      return M;
    }
    // 用 Jacobi 旋转求 A^T A 的特征分解，得到奇异值与右奇异向量
    function svd(A) {
      var n = A[0].length, B = [], i, j, k, s;
      for (i = 0; i < n; i++) { B[i] = []; for (j = 0; j < n; j++) { s = 0; for (k = 0; k < A.length; k++) s += A[k][i] * A[k][j]; B[i][j] = s; } }
      var V = [];
      for (i = 0; i < n; i++) { V[i] = []; for (j = 0; j < n; j++) V[i][j] = i === j ? 1 : 0; }
      for (var sweep = 0; sweep < 30; sweep++) {
        var off = 0;
        for (i = 0; i < n; i++) for (j = i + 1; j < n; j++) off += B[i][j] * B[i][j];
        if (off < 1e-12) break;
        for (i = 0; i < n; i++) for (j = i + 1; j < n; j++) {
          if (Math.abs(B[i][j]) < 1e-14) continue;
          var theta = (B[j][j] - B[i][i]) / (2 * B[i][j]);
          var tt = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
          var cc = 1 / Math.sqrt(tt * tt + 1), ss = tt * cc;
          for (k = 0; k < n; k++) {
            var bik = B[k][i], bjk = B[k][j];
            B[k][i] = cc * bik - ss * bjk; B[k][j] = ss * bik + cc * bjk;
          }
          for (k = 0; k < n; k++) {
            var bki = B[i][k], bkj = B[j][k];
            B[i][k] = cc * bki - ss * bkj; B[j][k] = ss * bki + cc * bkj;
            var vki = V[k][i], vkj = V[k][j];
            V[k][i] = cc * vki - ss * vkj; V[k][j] = ss * vki + cc * vkj;
          }
        }
      }
      var idx = [];
      for (i = 0; i < n; i++) idx.push(i);
      idx.sort(function (p, q) { return B[q][q] - B[p][p]; });
      var sv = idx.map(function (p) { return Math.sqrt(Math.max(0, B[p][p])); });
      var Vs = idx.map(function (p) { return V.map(function (rowv) { return rowv[p]; }); });   // Vs[r] 是第 r 个右奇异向量
      return { s: sv, V: Vs };
    }
    function approx(A, dec, r) {                              // 用前 r 个方向重建：A ≈ Σ (A v_r) v_r^T
      var out = [], i, j, k;
      for (i = 0; i < A.length; i++) { out[i] = []; for (j = 0; j < A[0].length; j++) out[i][j] = 0; }
      for (k = 0; k < r; k++) {
        var v = dec.V[k], Av = A.map(function (rowa) { var s = 0; for (var t = 0; t < rowa.length; t++) s += rowa[t] * v[t]; return s; });
        for (i = 0; i < A.length; i++) for (j = 0; j < A[0].length; j++) out[i][j] += Av[i] * v[j];
      }
      return out;
    }
    box.innerHTML = '<div class="aw-title">低秩近似：保留几个奇异值，矩阵还剩多少信息</div><div class="aw-grid">' +
      row("矩阵", select("kind", ["低秩图案（两个方向）", "接近低秩 + 噪声", "满秩噪声"], "接近低秩 + 噪声")) +
      row("保留的秩 r", range2("r", 3, 1, 24) + '<button type="button" data-k="play" class="aw-btn">播放</button>') + '</div>' +
      '<svg class="aw-chart aw-lr" viewBox="0 0 560 210"></svg><div class="aw-out"></div>';
    var svg = box.querySelector("svg"), out = box.querySelector(".aw-out"), cache = {}, timer = null;
    function stop() { if (timer) { clearInterval(timer); timer = null; box.querySelector('[data-k="play"]').textContent = "播放"; } }
    box.querySelector('[data-k="play"]').addEventListener("click", function () {
      if (timer) return stop();
      this.textContent = "暂停";
      var r = 0;
      timer = setInterval(function () {                       // 逐个加回奇异值，看图案一点点还原
        if (!box.isConnected) return stop();                   // 即时导航换页后别再动
        r += 1;
        input(box, "r").value = r;
        input(box, "r").dispatchEvent(new Event("input", { bubbles: true }));
        if (r >= N) stop();
      }, 180);
    });

    bind(box, function () {
      var kind = val(box, "kind"), r = val(box, "r");
      show(box, "r", String(r));
      if (!cache[kind]) { var A = build(kind); cache[kind] = { A: A, dec: svd(A) }; }
      var A = cache[kind].A, dec = cache[kind].dec, R = approx(A, dec, r);
      var cell = 6, x0 = 40, x1 = 230, i, j, err = 0, tot = 0;
      var S = svgText(x0 + N * cell / 2, 16, "原矩阵", "middle") + svgText(x1 + N * cell / 2, 16, "秩 " + r + " 的近似", "middle");
      for (i = 0; i < N; i++) for (j = 0; j < N; j++) {
        err += (A[i][j] - R[i][j]) * (A[i][j] - R[i][j]);
        tot += A[i][j] * A[i][j];
        S += '<rect x="' + (x0 + j * cell) + '" y="' + (26 + i * cell) + '" width="' + cell + '" height="' + cell +
             '" fill="' + heat(A[i][j]) + '"/>';
        S += '<rect x="' + (x1 + j * cell) + '" y="' + (26 + i * cell) + '" width="' + cell + '" height="' + cell +
             '" fill="' + heat(R[i][j]) + '"/>';
      }
      var bx = 400, bw = 130, bh = 120;                       // 奇异值柱状图
      S += svgText(bx + bw / 2, 16, "奇异值", "middle");
      var smax = dec.s[0] || 1;
      for (i = 0; i < N; i++) {
        var h = Math.max(1, dec.s[i] / smax * bh);
        S += '<rect x="' + (bx + i * (bw / N)) + '" y="' + (26 + bh - h) + '" width="' + (bw / N - 1) + '" height="' + h +
             '" class="' + (i < r ? "aw-on" : "aw-off") + '"/>';
      }
      S += '<line x1="' + (bx + r * (bw / N)) + '" y1="20" x2="' + (bx + r * (bw / N)) + '" y2="' + (26 + bh) + '" class="aw-dash"/>';
      S += svgText(x0 + N * cell / 2, 26 + N * cell + 16, "24 × 24 = 576 个数", "middle");
      S += svgText(x1 + N * cell / 2, 26 + N * cell + 16, "2 × 24 × " + r + " = " + (2 * N * r) + " 个数", "middle");
      svg.innerHTML = S;
      var rel = Math.sqrt(err / tot);
      out.innerHTML = "<p>相对误差 <b>" + (100 * rel).toFixed(1) + "%</b>，参数量 <b>" +
        (100 * 2 * N * r / (N * N)).toFixed(0) + "%</b>（" + (2 * N * r) + " / " + (N * N) + "）</p>" +
        '<p class="aw-note">奇异值掉得快的矩阵，几个方向就能还原大部分信息——LoRA 赌的是"微调增量"属于这一类，' +
        'MLA 赌的是"KV 激活"属于这一类。换成满秩噪声，你会看到 r 必须接近满秩才像样。</p>';
    });
  }
  function heat(v) {
    var t = Math.max(-1, Math.min(1, v / 1.6));
    return t >= 0 ? "rgba(0, 122, 255, " + (0.12 + 0.8 * t).toFixed(2) + ")"
                  : "rgba(240, 140, 0, " + (0.12 + 0.8 * -t).toFixed(2) + ")";
  }

  // ---------------------------------------------------------------- softmax 与采样
  function softmaxw(box) {
    var LOGITS = [4.2, 3.6, 3.1, 2.4, 2.0, 1.6, 1.1, 0.6, 0.1, -0.4, -1.0, -1.8];
    var WORDS = ["的", "是", "了", "在", "和", "有", "人", "我", "他", "这", "中", "大"];
    box.innerHTML = '<div class="aw-title">softmax 与采样：温度、top-k、top-p 各自在做什么</div><div class="aw-grid">' +
      row("温度 T", range2("temp", 100, 10, 200)) + row("top-k（0 = 不限）", range2("k", 0, 0, 12)) +
      row("top-p", range2("p", 90, 10, 100) + '<button type="button" data-k="play" class="aw-btn">播放</button>') + '</div>' +
      '<svg class="aw-chart aw-sm" viewBox="0 0 560 202"></svg><div class="aw-out"></div>';
    var svg = box.querySelector("svg"), out = box.querySelector(".aw-out"), timer = null;
    function stop() { if (timer) { clearInterval(timer); timer = null; box.querySelector('[data-k="play"]').textContent = "播放"; } }
    box.querySelector('[data-k="play"]').addEventListener("click", function () {
      if (timer) return stop();
      this.textContent = "暂停";
      var t = 10, dir = 1;                                    // 温度从 0.1 升到 2.0 再降回来
      timer = setInterval(function () {
        if (!box.isConnected) return stop();
        t += dir * 5;
        if (t >= 200) dir = -1;
        if (t <= 10 && dir < 0) return stop();
        input(box, "temp").value = t;
        input(box, "temp").dispatchEvent(new Event("input", { bubbles: true }));
      }, 60);
    });
    bind(box, function () {
      var T = val(box, "temp") / 100, k = val(box, "k"), p = val(box, "p") / 100;
      show(box, "temp", T.toFixed(2)); show(box, "k", k ? String(k) : "不限"); show(box, "p", p.toFixed(2));
      var ex = LOGITS.map(function (l) { return Math.exp(l / T); });
      var sum = ex.reduce(function (s, v) { return s + v; }, 0);
      var prob = ex.map(function (v) { return v / sum; });
      var keep = prob.map(function () { return true; }), acc = 0, i;
      for (i = 0; i < prob.length; i++) {
        if (k && i >= k) keep[i] = false;                     // top-k：只留前 k 个
        else if (acc >= p) keep[i] = false;                   // top-p：累计概率够了就截断
        if (keep[i]) acc += prob[i];
      }
      var kept = prob.filter(function (_, j) { return keep[j]; });
      var ksum = kept.reduce(function (s, v) { return s + v; }, 0);
      var entropy = -prob.reduce(function (s, v) { return s + (v > 0 ? v * Math.log2(v) : 0); }, 0);
      var W = 40, gap = 4, base = 150, h = 120;
      var S = "";
      for (i = 0; i < prob.length; i++) {
        var x = 20 + i * (W + gap), bh = Math.max(1, prob[i] / Math.max.apply(null, prob) * h);
        S += '<rect x="' + x + '" y="' + (base - bh) + '" width="' + W + '" height="' + bh +
             '" class="' + (keep[i] ? "aw-on" : "aw-off") + '" rx="2"/>';
        S += svgText(x + W / 2, base + 14, WORDS[i], "middle");
        if (prob[i] > 0.02) S += svgText(x + W / 2, base - bh - 6, (100 * prob[i]).toFixed(0) + "%", "middle");
      }
      S += svgText(20, 190, "蓝色 = 可能被采到，灰色 = 被 top-k / top-p 截掉", "start");
      svg.innerHTML = S;
      out.innerHTML = "<p>熵 <b>" + entropy.toFixed(2) + " 比特</b>（越大越犹豫）；候选 <b>" + kept.length +
        "</b> 个，占总概率 <b>" + (100 * ksum).toFixed(1) + "%</b>；最大概率 <b>" + (100 * prob[0]).toFixed(1) + "%</b></p>" +
        '<p class="aw-note">温度除在 logits 上：T &lt; 1 放大差距（更确定、更容易重复），T &gt; 1 拉平（更发散）。' +
        'T → 0 就是贪心解码。top-k 固定留几个，top-p 按累计概率动态留——分布尖锐时只留一两个，平坦时留更多。</p>';
    });
  }

  // ---------------------------------------------------------------- 梯度下降
  function graddesc(box) {
    box.innerHTML = '<div class="aw-title">梯度下降：学习率与动量怎么影响轨迹</div><div class="aw-grid">' +
      row("学习率", range2("lr", 20, 1, 100)) + row("动量", range2("mom", 0, 0, 95)) +
      row("曲面的拉伸（条件数）", range2("cond", 8, 1, 20)) +
      row("", '<button type="button" data-k="play" class="aw-btn">播放</button>', true) +
      '</div><svg class="aw-chart aw-gd" viewBox="0 0 560 240"></svg><div class="aw-out"></div>';
    var svg = box.querySelector("svg"), out = box.querySelector(".aw-out"), shown = 999, timer = null;

    function run() {
      var lr = val(box, "lr") / 100, mom = val(box, "mom") / 100, cond = val(box, "cond");
      show(box, "lr", lr.toFixed(2)); show(box, "mom", mom.toFixed(2)); show(box, "cond", String(cond));
      // f(x, y) = (x² + cond·y²) / 2，梯度 (x, cond·y)
      var x = 2.6, y = 0.9, vx = 0, vy = 0, path = [[x, y]], i, diverged = false;
      for (i = 0; i < 60; i++) {
        vx = mom * vx - lr * x; vy = mom * vy - lr * (cond * y);
        x += vx; y += vy;
        if (!isFinite(x) || Math.abs(x) > 12 || Math.abs(y) > 12) { diverged = true; break; }
        path.push([x, y]);
      }
      var mx = 2.9, my = 1.15;                                // 自动缩放：发散时也要看得见整条轨迹
      for (i = 0; i < path.length; i++) { mx = Math.max(mx, Math.abs(path[i][0])); my = Math.max(my, Math.abs(path[i][1])); }
      var cx = 280, cy = 120, u = Math.min(80, 255 / mx, 105 / my);
      function P(px, py) { return [cx + px * u, cy - py * u]; }
      var S = "";
      var rmax = Math.min(250, 104 * Math.sqrt(cond));        // 等高线撑满画面，形状仍由条件数决定
      for (var lvl = 1; lvl <= 6; lvl++) {                    // 等高线：椭圆
        var rx = lvl / 6 * rmax, ry = rx / Math.sqrt(cond);
        S += '<ellipse cx="' + cx + '" cy="' + cy + '" rx="' + rx.toFixed(1) + '" ry="' + ry.toFixed(1) + '" class="aw-blk"/>';
      }
      S += '<line x1="20" y1="' + cy + '" x2="540" y2="' + cy + '" class="aw-axis"/>';
      S += '<line x1="' + cx + '" y1="10" x2="' + cx + '" y2="230" class="aw-axis"/>';
      var n = Math.min(shown, path.length);
      var d = path.slice(0, n).map(function (pt, j) { var q = P(pt[0], pt[1]); return (j ? "L " : "M ") + q[0].toFixed(1) + " " + q[1].toFixed(1); }).join(" ");
      S += '<path d="' + d + '" class="aw-roof"/>';
      for (i = 0; i < n; i++) {
        var q = P(path[i][0], path[i][1]);
        if (Math.abs(q[0]) < 1e4) S += '<circle cx="' + q[0].toFixed(1) + '" cy="' + q[1].toFixed(1) + '" r="3" class="aw-dot"/>';
      }
      var st = P(path[0][0], path[0][1]);
      S += svgText(st[0] + 8, st[1] - 8, "起点", "start");
      S += svgText(20, 18, "椭圆 = 损失相同的点，正中心是最小值", "start");
      svg.innerHTML = S;
      var last = path[n - 1];
      var loss = (last[0] * last[0] + cond * last[1] * last[1]) / 2;
      out.innerHTML = diverged
        ? '<p><b>发散了</b>：学习率超过了 2 / 最大曲率（这里约 ' + (2 / cond).toFixed(2) + '），每一步都被放大。</p>'
        : "<p>" + (n - 1) + " 步之后，损失 <b>" + loss.toExponential(1) + "</b>" +
          (loss < 1e-3 ? "（收敛）" : loss < 1 ? "（还在往下走）" : "（几乎没动）") + "</p>";
      out.innerHTML += '<p class="aw-note">椭圆越扁（条件数越大），沿陡方向容易震荡、沿平方向走得慢——这就是为什么要做归一化' +
        '（把曲面拉圆）、用动量（把震荡抵消）、以及 Adam 那样按维度调步长。学习率的上界由最大曲率决定：超过 2 / L 必然发散。</p>';
    }
    function stop() { if (timer) { clearInterval(timer); timer = null; box.querySelector('[data-k="play"]').textContent = "播放"; } }
    box.querySelector('[data-k="play"]').addEventListener("click", function () {
      if (timer) { stop(); shown = 999; run(); return; }
      this.textContent = "暂停";
      shown = 1;
      timer = setInterval(function () { if (!box.isConnected) return stop(); shown += 1; run(); if (shown > 60) { stop(); } }, 80);
    });
    bind(box, function () { shown = 999; stop(); run(); });
  }

  var WIDGETS = { "kv-calc": kvCalc, roofline: roofline, mask: mask, pipeline: pipeline,
                  linmap: linmap, lowrank: lowrank, softmax: softmaxw, graddesc: graddesc };
  function init() {
    [].forEach.call(document.querySelectorAll(".aig-widget[data-widget]:not([data-ready])"), function (box) {
      var fn = WIDGETS[box.dataset.widget];
      if (!fn) return;
      box.setAttribute("data-ready", "");
      fn(box);
      [].forEach.call(box.querySelectorAll("svg.aw-chart"), function (svg) {   // 窄屏下图表可横向滚动，文字不至于缩成一团
        if (svg.parentNode.classList.contains("aw-scroll")) return;
        var wrap = document.createElement("div");
        wrap.className = "aw-scroll";
        svg.parentNode.insertBefore(wrap, svg);
        wrap.appendChild(svg);
      });
    });
  }
  init();
  if (window.document$ && window.document$.subscribe) window.document$.subscribe(init);   // 开了即时导航时，换页后再初始化
})();
