// 章节里的交互小工具。页面里放一个占位：<div class="aig-widget" data-widget="kv-calc"></div>
//   kv-calc   KV Cache 与显存：一张卡能同时服务多少个请求（大模型原理 · KV Cache）
//   roofline  矩阵乘的屋顶线：batch 变大，瓶颈从访存变成算力（大模型原理 · 性能与服务中的数学）
//   mask      注意力掩码：因果、带历史、滑动窗口、序列打包、前缀双向，以及按块跳过（大模型原理 · 注意力机制）
//   pipeline  流水线的调度与气泡：GPipe 与 1F1B（分布式训练 · 流水线并行）
//   linmap    线性变换：2×2 矩阵把网格变成什么样（大模型原理 · 线性代数）
//   lowrank   低秩近似：保留几个奇异值就够（大模型原理 · 线性代数）
//   softmax   softmax 与采样：温度、top-k、top-p（大模型原理 · 概率与采样）
//   graddesc  梯度下降：学习率、动量与条件数（大模型原理 · 微积分与反向传播）
//   coalesce  合并访问：一个 warp 要搬多少字节（CUDA · 内存层次与访存优化）
//   bankconf  bank 冲突：32 个线程撞在几个 bank 上（CUDA · 内存层次与访存优化）
//   scanviz   并行前缀和：Hillis-Steele 与 Blelloch 分步演示（CUDA · 前缀和）
//   occupancy 占用率：哪一项资源先卡住你（CUDA · 执行模型与性能基础）
//   pagewalk  地址翻译：多级页表怎么把虚拟地址变成物理地址（计算机基础 · 虚拟内存）
//   cachemap  组相联缓存：地址落进哪一组、什么时候开始冲突（计算机基础 · CPU 体系结构）
//   hashring  一致性哈希：加一台机器要搬多少数据（计算机基础 · 一致性哈希与分片）
//   pagedkv   分页 KV Cache：块大小怎么影响浪费（推理系统 · 分页 KV Cache）
//   radixcache 前缀缓存：基数树是怎么长出来的（推理系统 · 前缀缓存）
//   contbatch 调度：静态批、连续批与分块 prefill（推理系统 · 调度器）
//   spectree  投机解码：草稿树的宽度、深度与加速比（推理系统 · 投机解码进阶）
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

  // ---------------------------------------------------------------- 全局内存：合并访问
  function coalesce(box) {
    var MODES = {                                            // 模式 → [跨步（float 个数）, 起始偏移]
      "连续且对齐 a[k]": [1, 0], "连续但偏移一个 float a[k+1]": [1, 1], "跨步 2：a[2k]": [2, 0],
      "跨步 8：a[8k]": [8, 0], "跨步 32：按列读二维数组": [32, 0], "所有线程读同一个地址 a[0]": [0, 0]
    };
    box.innerHTML = '<div class="aw-title">合并访问：一个 warp 要把多少字节搬过总线</div><div class="aw-grid">' +
      row("访问模式", select("mode", Object.keys(MODES), "跨步 2：a[2k]"), true) +
      row("跨步（float 个数）", range2("stride", 2, 0, 32)) + row("起始偏移", range2("off", 0, 0, 8)) + '</div>' +
      '<svg class="aw-chart aw-co" viewBox="0 0 560 168"></svg><div class="aw-out"></div>';
    var svg = box.querySelector("svg"), out = box.querySelector(".aw-out");
    box.querySelector('[data-k="mode"]').addEventListener("change", function () {
      var m = MODES[val(box, "mode")];
      input(box, "stride").value = m[0]; input(box, "off").value = m[1];
      draw();
    });
    function draw() {
      var stride = val(box, "stride"), off = val(box, "off");
      show(box, "stride", stride === 0 ? "同一地址" : String(stride)); show(box, "off", String(off));
      var addr = [], used = {}, sec = {}, i;
      for (i = 0; i < 32; i++) { var a = (off + (stride ? i * stride : 0)) * 4; addr.push(a); used[a / 4] = 1; sec[Math.floor(a / 32)] = 1; }
      var secs = Object.keys(sec).map(Number).sort(function (p, q) { return p - q; });
      var cw = 6, sw = 8 * cw, pad = 6, perRow = 10;         // 只画真正被触及的扇区
      var list = secs.slice(0, 20), rows = Math.ceil(list.length / perRow);
      var S = svgText(20, 14, "每个格子 = 4 字节的 float，每个框 = 一次 32 字节的扇区传输", "start");
      for (i = 0; i < list.length; i++) {
        var s = list[i];
        var x = 20 + (i % perRow) * (sw + pad), y = 32 + Math.floor(i / perRow) * 50;
        S += '<rect x="' + x + '" y="' + y + '" width="' + sw + '" height="20" rx="2" class="aw-blk"/>';
        for (var j = 0; j < 8; j++) {
          S += '<rect x="' + (x + j * cw + 0.6) + '" y="' + (y + 2.6) + '" width="' + (cw - 1.2) + '" height="14.8" class="' +
               (used[s * 8 + j] ? "aw-on" : "aw-b") + '"/>';
        }
        S += svgText(x + sw / 2, y + 32, "#" + s, "middle");
      }
      if (secs.length > list.length) S += svgText(20 + perRow * (sw + pad) - 4, 32 + rows * 50 + 12, "… 共 " + secs.length + " 个扇区", "end");
      var ly = 32 + rows * 50 + 22;
      S += '<rect x="20" y="' + ly + '" width="10" height="10" class="aw-on"/>' + svgText(36, ly + 6, "线程真正要的字节", "start");
      S += '<rect x="170" y="' + ly + '" width="10" height="10" class="aw-b"/>' + svgText(186, ly + 6, "被一起搬上来、但没人用", "start");
      svg.setAttribute("viewBox", "0 0 560 " + (ly + 22));
      svg.innerHTML = S;
      var moved = secs.length * 32, useful = stride === 0 ? 4 : 128;
      out.innerHTML = "<p>触及 <b>" + secs.length + "</b> 个扇区 → 总线上搬了 <b>" + moved + " 字节</b>，真正用上 " + useful +
        " 字节，带宽利用率 <b>" + (100 * useful / moved).toFixed(1) + "%</b>" +
        (stride === 0 ? "（硬件会广播，只有一次传输，不算浪费）" : "") + "</p>" +
        '<p class="aw-note">合并访问的全部内容就是这张图：让相邻线程读相邻地址，32 个线程正好铺满 4 个扇区。' +
        '跨步越大，每个扇区里被用上的字节越少，实测带宽就按同样的比例掉下去。二维数组里让 threadIdx.x 沿列走，就是跨步 = 行宽。</p>';
    }
    bind(box, draw);
  }

  // ---------------------------------------------------------------- 共享内存：bank 冲突
  function bankconf(box) {
    var MODES = { "按行访问 tile[0][tid]": 1, "按列访问 tile[tid][0]，行宽 32": 32,
                  "按列访问 tile[tid][0]，行宽 33（padding）": 33, "跨步 2": 2, "跨步 4": 4,
                  "所有线程读同一个字（广播）": 0 };
    box.innerHTML = '<div class="aw-title">bank 冲突：32 个线程撞在几个 bank 上</div><div class="aw-grid">' +
      row("访问模式", select("mode", Object.keys(MODES), "按列访问 tile[tid][0]，行宽 32"), true) +
      row("相邻线程的字间隔", range2("stride", 32, 0, 36)) + '</div>' +
      '<svg class="aw-chart aw-bk" viewBox="0 0 560 220"></svg><div class="aw-out"></div>';
    var svg = box.querySelector("svg"), out = box.querySelector(".aw-out");
    box.querySelector('[data-k="mode"]').addEventListener("change", function () {
      input(box, "stride").value = MODES[val(box, "mode")]; draw();
    });
    function draw() {
      var stride = val(box, "stride");
      show(box, "stride", stride === 0 ? "同一个字" : String(stride));
      var banks = [], i, k;
      for (i = 0; i < 32; i++) banks.push([]);
      for (k = 0; k < 32; k++) { var word = stride ? k * stride : 0; banks[word % 32].push(word); }
      var worst = 1, busiest = 0;
      for (i = 0; i < 32; i++) {
        var d = {}, n = 0;
        banks[i].forEach(function (w) { if (!d[w]) { d[w] = 1; n++; } });
        if (n > worst) { worst = n; busiest = i; }
      }
      var x0 = 22, bw = 16, CAP = 7, base = 168, step = 15;
      var S = svgText(x0, 14, "横轴是 32 个 bank（每个 4 字节宽），往上堆的每个格子 = 落在这个 bank 上的一个线程", "start");
      for (i = 0; i < 32; i++) {
        var x = x0 + i * bw, list = banks[i];
        S += '<rect x="' + (x + 1) + '" y="' + base + '" width="' + (bw - 2) + '" height="13" rx="2" class="aw-off"/>';
        for (k = 0; k < Math.min(list.length, CAP); k++) {
          S += '<rect x="' + (x + 1) + '" y="' + (base - (k + 1) * step) + '" width="' + (bw - 2) + '" height="13" rx="2" class="' +
               (list.length > 1 && stride ? "aw-b" : "aw-on") + '"/>';
        }
        if (list.length > CAP) S += svgText(x + bw / 2, base - CAP * step - 14, "×" + list.length, "middle");
        if (i % 4 === 0) S += svgText(x + bw / 2, base + 26, String(i), "middle");
      }
      S += svgText(x0, base + 44, "bank 编号", "start");
      svg.innerHTML = S;
      out.innerHTML = "<p>最忙的 bank 上有 <b>" + worst + "</b> 个不同的字" +
        (stride === 0 ? "：所有线程读同一个地址，硬件广播，<b>不算冲突</b>" :
         worst === 1 ? "：32 个线程落在 32 个不同的 bank，<b>一个周期就能完成</b>" :
         "：<b>" + worst + " 路冲突</b>，这条访存指令要拆成 " + worst + " 次，耗时 " + worst + " 倍") + "</p>" +
        '<p class="aw-note">bank 编号就是"字编号 % 32"。所以只要相邻线程的字间隔和 32 互质（间隔为奇数），就一定铺满 32 个 bank——' +
        '这正是 <code>__shared__ float tile[32][33]</code> 那个多出来的一列在做的事：把间隔从 32 变成 33。</p>';
    }
    bind(box, draw);
  }

  // ---------------------------------------------------------------- 前缀和：两种并行算法
  function scanviz(box) {
    var A = [3, 1, 7, 0, 4, 1, 6, 3, 2, 5, 1, 2, 0, 4, 3, 1], N = 16;
    function hillis() {                                      // Hillis-Steele：步数少、加法多
      var st = [{ v: A.slice(), e: [], t: "初始值" }], cur = A.slice(), adds = 0;
      for (var d = 1; d < N; d *= 2) {
        var nx = cur.slice(), e = [];
        for (var i = N - 1; i >= d; i--) { nx[i] = cur[i] + cur[i - d]; e.push([i - d, i]); adds++; }
        cur = nx; st.push({ v: cur.slice(), e: e, t: "步 " + st.length + "：x[i] += x[i−" + d + "]" });
      }
      return { steps: st, adds: adds, note: "包含扫描（inclusive），" + (Math.log(N) / Math.log(2)) + " 步、" + adds + " 次加法" };
    }
    function blelloch() {                                    // Blelloch：两趟、加法少
      var cur = A.slice(), st = [{ v: cur.slice(), e: [], t: "初始值" }], adds = 0, d, i, e;
      for (d = 1; d < N; d *= 2) {                           // 上扫：求部分和
        var nx = cur.slice(); e = [];
        for (i = d * 2 - 1; i < N; i += d * 2) { nx[i] = cur[i] + cur[i - d]; e.push([i - d, i]); adds++; }
        cur = nx; st.push({ v: cur.slice(), e: e, t: "上扫 " + Math.round(Math.log(d * 2) / Math.log(2)) + "：间隔 " + d });
      }
      cur = cur.slice(); cur[N - 1] = 0;
      st.push({ v: cur.slice(), e: [], t: "把最后一个换成 0" });
      for (d = N / 2; d >= 1; d /= 2) {                      // 下扫：把部分和散回去
        var n2 = cur.slice(); e = [];
        for (i = d * 2 - 1; i < N; i += d * 2) { n2[i - d] = cur[i]; n2[i] = cur[i] + cur[i - d]; e.push([i - d, i]); adds++; }
        cur = n2; st.push({ v: cur.slice(), e: e, t: "下扫：间隔 " + d });
      }
      return { steps: st, adds: adds, note: "排除扫描（exclusive），2×log₂N 步、" + adds + " 次加法，工作量是 O(N)" };
    }
    var ALGO = { "Hillis-Steele（步数少，加法多）": hillis(), "Blelloch（两趟扫，加法只有 O(N)）": blelloch() };
    box.innerHTML = '<div class="aw-title">并行前缀和：两种经典算法一步一步看</div><div class="aw-grid">' +
      row("算法", select("algo", Object.keys(ALGO), "Hillis-Steele（步数少，加法多）"), true) +
      row("第几步", range2("step", 0, 0, 4) + '<button type="button" data-k="play" class="aw-btn">播放</button>') + '</div>' +
      '<svg class="aw-chart aw-sc" viewBox="0 0 560 140"></svg><div class="aw-out"></div>';
    var svg = box.querySelector("svg"), out = box.querySelector(".aw-out"), timer = null;
    function stop() { if (timer) { clearInterval(timer); timer = null; box.querySelector('[data-k="play"]').textContent = "播放"; } }
    box.querySelector('[data-k="play"]').addEventListener("click", function () {
      if (timer) return stop();
      this.textContent = "暂停";
      input(box, "step").value = 0; draw();
      timer = setInterval(function () {
        if (!box.isConnected) return stop();
        var s = val(box, "step") + 1, mx = +input(box, "step").max;
        input(box, "step").value = Math.min(mx, s); draw();
        if (s >= mx) stop();
      }, 700);
    });
    function draw() {
      var a = ALGO[val(box, "algo")], sl = input(box, "step");
      sl.max = a.steps.length - 1;
      var si = Math.min(val(box, "step"), a.steps.length - 1);
      show(box, "step", si + " / " + (a.steps.length - 1));
      var st = a.steps[si], prev = a.steps[Math.max(0, si - 1)];
      var x0 = 22, cw = 32, gap = 2, top = 74;
      var S = svgText(x0, 16, st.t, "start");
      var touched = {}; st.e.forEach(function (p) { touched[p[1]] = 1; });
      for (var i = 0; i < N; i++) {
        var x = x0 + i * (cw + gap);
        S += '<rect x="' + x + '" y="' + top + '" width="' + cw + '" height="26" rx="3" class="' +
             (touched[i] ? "aw-on" : "aw-off") + '"/>';
        S += svgText(x + cw / 2, top + 13, String(st.v[i]), "middle");
        S += svgText(x + cw / 2, top + 40, String(i), "middle");
      }
      st.e.forEach(function (p) {                             // 这一步谁加到了谁
        var xs = x0 + p[0] * (cw + gap) + cw / 2, xd = x0 + p[1] * (cw + gap) + cw / 2, h = 26 + (p[1] - p[0]) * 1.6;
        S += '<path d="M ' + xs + ' ' + top + ' C ' + xs + ' ' + (top - h) + ' ' + xd + ' ' + (top - h) + ' ' + xd + ' ' + top +
             '" class="aw-brace"/>';
      });
      svg.innerHTML = S;
      out.innerHTML = "<p>第 <b>" + si + "</b> 步，这一步做了 <b>" + st.e.length + "</b> 次加法（全部 " + a.adds + " 次）。" + a.note + "</p>" +
        '<p class="aw-note">Hillis-Steele 每一步所有线程都在干活，步数少但总加法是 O(N log N)；' +
        'Blelloch 先把部分和往上收、再往下散，总加法只有 O(N)，代价是要跑两趟、且活跃线程越来越稀疏。' +
        'GPU 上 warp 内用前者（shuffle 就够），block 和设备级用后者。</p>';
    }
    bind(box, function () { stop(); draw(); });
  }

  // ---------------------------------------------------------------- 占用率
  function occupancy(box) {
    var ARCH = {                                             // 每 SM：最大线程、最大 block、寄存器数、共享内存 KB、每 SM 最大 warp
      "A100（sm_80）": [2048, 32, 65536, 164, 64], "H100（sm_90）": [2048, 32, 65536, 228, 64],
      "RTX 4090（sm_89）": [1536, 24, 65536, 100, 48], "T4（sm_75）": [1024, 16, 65536, 64, 32]
    };
    box.innerHTML = '<div class="aw-title">占用率：哪一项资源先卡住你</div><div class="aw-grid">' +
      row("GPU", select("arch", Object.keys(ARCH), "A100（sm_80）")) +
      row("block 大小", range2("bs", 8, 1, 32)) +
      row("每线程寄存器", range2("regs", 40, 16, 128)) +
      row("每 block 共享内存 KB", range2("smem", 16, 0, 164)) + '</div>' +
      '<svg class="aw-chart aw-oc" viewBox="0 0 560 150"></svg><div class="aw-out"></div>';
    var svg = box.querySelector("svg"), out = box.querySelector(".aw-out");
    bind(box, function () {
      var a = ARCH[val(box, "arch")], maxT = a[0], maxB = a[1], maxR = a[2], maxS = a[3], maxW = a[4];
      var bs = val(box, "bs") * 32, regs = val(box, "regs"), smem = Math.min(val(box, "smem"), maxS);
      show(box, "bs", bs + " 线程"); show(box, "regs", String(regs)); show(box, "smem", smem + " KB");
      var wpb = bs / 32;
      var regPerWarp = Math.ceil(regs * 32 / 256) * 256;      // 寄存器按 warp、每 256 个为一档分配
      var byThread = Math.floor(maxT / bs), byBlock = maxB;
      var byReg = Math.floor(maxR / (regPerWarp * wpb));
      var bySmem = smem > 0 ? Math.floor(maxS / smem) : 99;
      var limits = [["线程数上限", byThread], ["block 数上限", byBlock], ["寄存器", byReg], ["共享内存", bySmem]];
      var blocks = Math.min(byThread, byBlock, byReg, bySmem);
      var warps = Math.min(blocks * wpb, maxW), occ = warps / maxW;
      var x0 = 118, W = 380, S = "", top = 16;
      var mx = Math.max(4, Math.min(12, Math.max.apply(null, limits.map(function (l) { return Math.min(l[1], 12); }))));
      limits.forEach(function (l, i) {
        var y = top + i * 26, n = Math.min(l[1], mx), bind_ = l[1] === blocks;
        S += svgText(x0 - 10, y + 9, l[0], "end");
        S += '<rect x="' + x0 + '" y="' + y + '" width="' + (n / mx * W) + '" height="18" rx="3" class="' + (bind_ ? "aw-b" : "aw-on") + '"/>';
        S += svgText(x0 + n / mx * W + 8, y + 9, (l[1] > 90 ? "不限" : l[1] + " 个 block") + (bind_ ? "  ← 瓶颈" : ""), "start");
      });
      var yb = top + 4 * 26 + 12;
      S += svgText(x0 - 10, yb + 10, "实际占用率", "end");
      S += '<rect x="' + x0 + '" y="' + yb + '" width="' + W + '" height="20" rx="3" class="aw-off"/>';
      S += '<rect x="' + x0 + '" y="' + yb + '" width="' + (occ * W) + '" height="20" rx="3" class="aw-on"/>';
      S += svgText(x0 + W + 8, yb + 10, (100 * occ).toFixed(0) + "%", "start");
      svg.innerHTML = S;
      out.innerHTML = "<p>每个 SM 能同时驻留 <b>" + blocks + "</b> 个 block、<b>" + warps + "</b> 个 warp，占用率 <b>" +
        (100 * occ).toFixed(0) + "%</b>（上限 " + maxW + " 个 warp）。每个 block 要 " +
        (regPerWarp * wpb / 1024).toFixed(1) + "K 个寄存器、" + smem + " KB 共享内存。</p>" +
        '<p class="aw-note">占用率不是越高越好：它买的是"用别的 warp 盖住访存延迟"的能力。' +
        '访存密集的 kernel 需要高占用率；而计算密集的 kernel 往往宁可多用寄存器做寄存器分块，占用率 25% 反而更快。' +
        '先看 Nsight Compute 报告里到底是延迟没盖住还是算力没喂饱，再决定要不要调这几个数。</p>';
    });
  }

  // ---------------------------------------------------------------- 多级页表的地址翻译
  function pagewalk(box) {
    var PAGES = { "4 KiB（四级页表）": 4096, "2 MiB（大页，三级）": 2 * 1024 * 1024, "1 GiB（大页，两级）": 1024 * 1024 * 1024 };
    var LV = ["PML4（第 4 级）", "PDPT（第 3 级）", "PD（第 2 级）", "PT（第 1 级）"];
    box.innerHTML = '<div class="aw-title">地址翻译：一个虚拟地址是怎么变成物理地址的</div><div class="aw-grid">' +
      row("虚拟地址", '<input type="text" data-k="va" value="0x7F3A1C2D5E6F" spellcheck="false">' +
          '<button type="button" data-k="rand" class="aw-btn">换一个</button>', true) +
      row("页大小", select("page", Object.keys(PAGES), "4 KiB（四级页表）"), true) + '</div>' +
      '<svg class="aw-chart aw-pw" viewBox="0 0 560 240"></svg><div class="aw-out"></div>';
    var svg = box.querySelector("svg"), out = box.querySelector(".aw-out");
    box.querySelector('[data-k="rand"]').addEventListener("click", function () {
      var hi = Math.floor(Math.random() * 0x8000), lo = Math.floor(Math.random() * 0x100000000);
      input(box, "va").value = "0x" + hi.toString(16).toUpperCase() + ("00000000" + lo.toString(16).toUpperCase()).slice(-8);
      draw();
    });
    function draw() {
      var raw = input(box, "va").value.trim(), page = PAGES[val(box, "page")];
      var va = Number(raw);
      if (!isFinite(va) || va < 0) va = 0;
      va = Math.floor(va) % Math.pow(2, 48);
      var offBits = Math.round(Math.log(page) / Math.log(2));  // 12 / 21 / 30
      var levels = Math.round((48 - offBits) / 9);             // 4 / 3 / 2
      var idx = [], i;
      for (i = 0; i < levels; i++) idx.push(Math.floor(va / Math.pow(2, 39 - i * 9)) % 512);
      var off = va % page;
      // 位段条
      var x0 = 22, W = 516, S = "", segs = [], bits = 0;
      for (i = 0; i < levels; i++) { segs.push([9, LV[i].split("（")[0] + " 下标", idx[i], "aw-on"]); bits += 9; }
      segs.push([offBits, "页内偏移", "0x" + off.toString(16), "aw-b"]);
      var total = bits + offBits, cx = x0;
      S += svgText(x0, 14, "48 位虚拟地址 0x" + va.toString(16).toUpperCase() + " 的拆法", "start");
      segs.forEach(function (sg) {
        var w = sg[0] / total * W;
        S += '<rect x="' + cx.toFixed(1) + '" y="24" width="' + (w - 2).toFixed(1) + '" height="26" rx="3" class="' + sg[3] + '"/>';
        S += svgText(cx + w / 2 - 1, 37, sg[0] + " 位", "middle");
        S += svgText(cx + w / 2 - 1, 62, sg[2], "middle");
        cx += w;
      });
      // 查表链路
      var by = 100, bw = 96, gapx = (W - (levels + 1) * bw) / levels;
      S += svgText(x0, 86, "页表遍历：从 CR3 出发，每一级用一个 9 位下标找下一级", "start");
      for (i = 0; i <= levels; i++) {
        var x = x0 + i * (bw + gapx), last = i === levels;
        S += '<rect x="' + x.toFixed(1) + '" y="' + by + '" width="' + bw + '" height="46" rx="5" class="' + (last ? "aw-b" : "aw-blk") + '"/>';
        S += svgText(x + bw / 2, by + 17, last ? "物理页" : LV[i].split("（")[0], "middle");
        S += svgText(x + bw / 2, by + 33, last ? "+ 偏移" : "[" + idx[i] + "]", "middle");
        if (i < levels) {
          var xa = x + bw, xb = x + bw + gapx;
          S += '<line x1="' + xa + '" y1="' + (by + 23) + '" x2="' + (xb - 5) + '" y2="' + (by + 23) + '" class="aw-axis"/>';
          S += '<polygon points="' + xb + ',' + (by + 23) + ' ' + (xb - 7) + ',' + (by + 19) + ' ' + (xb - 7) + ',' + (by + 27) + '" class="aw-dot"/>';
        }
      }
      var cov = [512 * 512 * 512 * 4096, 512 * 512 * 4096, 512 * 4096, 4096];
      S += svgText(x0, 176, "每一级表项管多大地址：", "start");
      for (i = 0; i < levels; i++) {
        var x2 = x0 + i * (bw + gapx), c = cov[4 - levels + i];
        S += svgText(x2 + bw / 2, 196, c >= 1e9 ? (c / 1024 / 1024 / 1024) + " GiB" : c >= 1e6 ? (c / 1024 / 1024) + " MiB" : (c / 1024) + " KiB", "middle");
      }
      S += svgText(x0, 222, "一张页表 512 项 × 8 字节 = 4096 字节，正好一页", "start");
      svg.innerHTML = S;
      out.innerHTML = "<p>TLB 未命中时，这一次访问要先做 <b>" + levels + "</b> 次查表、再取数据，一共 <b>" + (levels + 1) +
        "</b> 次内存访问；TLB 命中时只要 1 次。</p>" +
        '<p class="aw-note">换成大页就是把低几级砍掉：2 MiB 页少查一级、偏移从 12 位变成 21 位，一个 TLB 表项覆盖的范围从 4 KiB 涨到 2 MiB，' +
        'TLB 命中率跟着涨——这就是 GPU 显存固定、RDMA 注册大块内存时都偏爱大页的原因。PagedAttention 用的是同一个思路：' +
        '把"连续的逻辑地址"和"散落的物理块"用一张表隔开。</p>';
    }
    box.querySelector('[data-k="va"]').addEventListener("input", draw);
    bind(box, draw);
  }

  // ---------------------------------------------------------------- 组相联缓存的映射与冲突
  function cachemap(box) {
    box.innerHTML = '<div class="aw-title">组相联缓存：地址落进哪一组，什么时候开始互相踢</div><div class="aw-grid">' +
      row("缓存容量 KB", range2("cap", 32, 1, 64)) + row("相联度（路数）", range2("ways", 8, 1, 16)) +
      row("访问跨步（字节）", range2("stride", 11, 6, 16)) + '</div>' +
      '<svg class="aw-chart aw-cm" viewBox="0 0 560 190"></svg><div class="aw-out"></div>';
    var svg = box.querySelector("svg"), out = box.querySelector(".aw-out");
    bind(box, function () {
      var LINE = 64, cap = val(box, "cap") * 1024, ways = val(box, "ways"), stride = Math.pow(2, val(box, "stride"));
      var nsets = Math.max(1, Math.round(cap / (ways * LINE)));
      show(box, "cap", (cap / 1024) + " KB"); show(box, "ways", ways + " 路");
      show(box, "stride", stride >= 1024 ? (stride / 1024) + " KB" : stride + " B");
      var idxBits = Math.round(Math.log(nsets) / Math.log(2)), offBits = 6, tagBits = 48 - idxBits - offBits;
      var x0 = 22, W = 516, segs = [[tagBits, "tag", "aw-off"], [idxBits, "组号", "aw-on"], [offBits, "偏移", "aw-b"]];
      var S = svgText(x0, 14, "地址怎么拆：低 6 位选行内字节，中间 " + idxBits + " 位选组，剩下的是 tag", "start"), cx = x0;
      segs.forEach(function (sg) {
        var w = sg[0] / 48 * W;
        S += '<rect x="' + cx.toFixed(1) + '" y="22" width="' + (w - 2).toFixed(1) + '" height="30" rx="3" class="' + sg[2] + '"/>';
        S += svgText(cx + w / 2 - 1, 32, sg[1], "middle");
        S += svgText(cx + w / 2 - 1, 46, sg[0] + " 位", "middle");
        cx += w;
      });
      // 连续按这个跨步访问 64 次，看落进哪些组
      var hit = {}, k, touched = 0, perSet = {};
      for (k = 0; k < 64; k++) {
        var line = Math.floor(k * stride / LINE), st = line % nsets;
        if (!hit[st]) { hit[st] = 0; touched++; }
        hit[st] += 1;
        perSet[st] = (perSet[st] || {}); perSet[st][line] = 1;
      }
      var worst = 0;
      Object.keys(perSet).forEach(function (s2) { worst = Math.max(worst, Object.keys(perSet[s2]).length); });
      var SHOWN = Math.min(nsets, 64), cw = W / SHOWN;
      S += svgText(x0, 74, "按这个跨步连续访问 64 次，落进了哪些组（只画前 " + SHOWN + " 组）", "start");
      for (k = 0; k < SHOWN; k++) {
        var n = hit[k] || 0, h = n ? Math.min(ways, n) / ways * 56 : 0;
        S += '<rect x="' + (x0 + k * cw).toFixed(1) + '" y="88" width="' + Math.max(1, cw - 1).toFixed(1) + '" height="56" class="aw-off"/>';
        if (n) S += '<rect x="' + (x0 + k * cw).toFixed(1) + '" y="' + (144 - h).toFixed(1) + '" width="' + Math.max(1, cw - 1).toFixed(1) +
                    '" height="' + h.toFixed(1) + '" class="' + (n > ways ? "aw-b" : "aw-on") + '"/>';
      }
      S += svgText(x0, 162, "组 0", "start") + svgText(x0 + W, 162, "组 " + (SHOWN - 1), "end");
      S += svgText(x0 + W / 2, 182, "柱子高度 = 这一组里挤了几条不同的缓存行（满格 = " + ways + " 路）", "middle");
      svg.innerHTML = S;
      out.innerHTML = "<p>" + (cap / 1024) + " KB、" + ways + " 路 → <b>" + nsets + " 组</b>，每组 " + ways +
        " 行。相距 <b>" + (nsets * LINE >= 1024 ? (nsets * LINE / 1024) + " KB" : nsets * LINE + " B") +
        "</b> 整数倍的地址一定落进同一组。这 64 次访问只用到 <b>" + touched + "</b> 组，最挤的一组里有 <b>" + worst + "</b> 条不同的行" +
        (worst > ways ? "，<b>超过 " + ways + " 路，开始互相踢（冲突缺失）</b>" : "，还放得下") + "。</p>" +
        '<p class="aw-note">这就是"按列遍历矩阵很慢"的机制：行距是 2 的幂时，一列上的元素全落进同一组，' +
        '读进来的行还没被第二次用到就被挤掉了。每行末尾补一个缓存行（行距改成奇数倍），地址就散到所有组里——' +
        '和共享内存加一列 padding 躲 bank 冲突是同一招。</p>';
    });
  }

  // ---------------------------------------------------------------- 一致性哈希
  function hashring(box) {
    function h32(s) {                                        // FNV-1a + murmur3 收尾，保证相似字符串也散得开
      var x = 2166136261;
      for (var i = 0; i < s.length; i++) { x ^= s.charCodeAt(i); x = Math.imul(x, 16777619); }
      x ^= x >>> 16; x = Math.imul(x, 2246822507);
      x ^= x >>> 13; x = Math.imul(x, 3266489909); x ^= x >>> 16;
      return (x >>> 0) / 4294967296;
    }
    var KEYS = [];
    for (var i = 0; i < 2000; i++) KEYS.push(h32("key-" + i));
    function assign(nodes, vn) {                             // 返回 {环上的点, 每个 key 落到哪个节点}
      var ring = [];
      nodes.forEach(function (n) { for (var j = 0; j < vn; j++) ring.push([h32("node-" + n + "#" + j), n]); });
      ring.sort(function (a, b) { return a[0] - b[0]; });
      var owner = KEYS.map(function (k) {
        var lo = 0, hi = ring.length - 1, ans = 0;           // 顺时针找第一个 ≥ k 的点
        if (k > ring[hi][0]) return ring[0][1];
        while (lo <= hi) { var m = (lo + hi) >> 1; if (ring[m][0] >= k) { ans = m; hi = m - 1; } else lo = m + 1; }
        return ring[ans][1];
      });
      return { ring: ring, owner: owner };
    }
    box.innerHTML = '<div class="aw-title">一致性哈希：加一台机器，要搬多少数据</div><div class="aw-grid">' +
      row("节点数", range2("n", 6, 2, 12)) + row("每节点虚拟节点数", range2("vn", 60, 1, 200)) +
      row("动作", select("act", ["不动", "加一台新机器", "挂掉一台机器"], "加一台新机器"), true) + '</div>' +
      '<svg class="aw-chart aw-hr" viewBox="0 0 560 230"></svg><div class="aw-out"></div>';
    var svg = box.querySelector("svg"), out = box.querySelector(".aw-out");
    var PAL = ["#007aff", "#f08c00", "#34c759", "#af52de", "#ff3b30", "#00b8d4", "#ffb300", "#8e8e93",
               "#5856d6", "#ff2d55", "#30b0c7", "#a2845e", "#66bb6a"];
    bind(box, function () {
      var n = val(box, "n"), vn = val(box, "vn"), act = val(box, "act");
      show(box, "n", n + " 台"); show(box, "vn", String(vn));
      var before = [], k;
      for (k = 0; k < n; k++) before.push(k);
      var after = before.slice();
      if (act === "加一台新机器") after.push(n);
      else if (act === "挂掉一台机器" && n > 1) after = before.slice(0, n - 1);
      var A = assign(before, vn), B = assign(after, vn);
      var moved = 0, cnt = {}, i;
      for (i = 0; i < KEYS.length; i++) { if (A.owner[i] !== B.owner[i]) moved++; cnt[B.owner[i]] = (cnt[B.owner[i]] || 0) + 1; }
      // 左边：环
      var cx = 118, cy = 112, R = 84, S = "";
      S += '<circle cx="' + cx + '" cy="' + cy + '" r="' + R + '" fill="none" class="aw-blk"/>';
      B.ring.forEach(function (p) {
        var a = p[0] * Math.PI * 2 - Math.PI / 2, col = PAL[p[1] % PAL.length];
        S += '<line x1="' + (cx + Math.cos(a) * (R - 9)).toFixed(1) + '" y1="' + (cy + Math.sin(a) * (R - 9)).toFixed(1) +
             '" x2="' + (cx + Math.cos(a) * (R + 9)).toFixed(1) + '" y2="' + (cy + Math.sin(a) * (R + 9)).toFixed(1) +
             '" stroke="' + col + '" stroke-width="' + (vn > 40 ? 1.2 : 2.6) + '"/>';
      });
      S += svgText(cx, cy - 8, after.length + " 台机器", "middle");
      S += svgText(cx, cy + 10, B.ring.length + " 个环上的点", "middle");
      // 右边：每台机器分到多少 key
      var x0 = 250, W = 280, bh = 15, ideal = KEYS.length / after.length, mx = 0;
      after.forEach(function (m) { mx = Math.max(mx, cnt[m] || 0); });
      S += svgText(x0, 16, "每台机器分到的 key 数（虚线 = 完全均匀）", "start");
      after.forEach(function (m, r) {
        var y = 28 + r * (bh + 5), w = (cnt[m] || 0) / mx * W;
        S += '<rect x="' + x0 + '" y="' + y + '" width="' + w.toFixed(1) + '" height="' + bh + '" rx="2" fill="' + PAL[m % PAL.length] + '" opacity="0.75"/>';
        S += svgText(x0 + w + 6, y + bh / 2, String(cnt[m] || 0), "start");
      });
      var xi = x0 + ideal / mx * W;
      S += '<line x1="' + xi.toFixed(1) + '" y1="24" x2="' + xi.toFixed(1) + '" y2="' + (28 + after.length * (bh + 5)) + '" class="aw-dash"/>';
      svg.innerHTML = S;
      var lo = Infinity, hi2 = 0;
      after.forEach(function (m) { lo = Math.min(lo, cnt[m] || 0); hi2 = Math.max(hi2, cnt[m] || 0); });
      out.innerHTML = "<p>" + (act === "不动" ? "没有变动，" : act + "之后，") + "有 <b>" + (100 * moved / KEYS.length).toFixed(1) +
        "%</b> 的 key 换了归属" + (act === "不动" ? "" : "（理论值约 " + (100 / Math.max(after.length, before.length)).toFixed(1) + "%）") +
        "；最重的机器是最轻的 <b>" + (hi2 / Math.max(1, lo)).toFixed(2) + "</b> 倍。</p>" +
        '<p class="aw-note">取模分片下加一台机器要搬掉几乎全部数据；一致性哈希只搬 1/N。' +
        '但每台机器只放一个点时，环上的间隔很不均匀（试试把虚拟节点数拉到 1），' +
        '所以实践中每台机器放几十到几百个虚拟节点，负载就平了。推理系统里按前缀哈希路由到同一台机器，用的就是这套。</p>';
    });
  }

  // ---------------------------------------------------------------- 分页 KV Cache
  function pagedkv(box) {
    var SIZES = [1, 8, 16, 32, 64, 128];
    function lens(n, seed) {                                 // 固定的伪随机长度，拖滑块时画面稳定
      var out = [], x = seed;
      for (var i = 0; i < n; i++) { x = (x * 1103515245 + 12345) % 2147483648; out.push(400 + x % 2800); }
      return out;
    }
    box.innerHTML = '<div class="aw-title">分页 KV Cache：块大小怎么影响浪费</div><div class="aw-grid">' +
      row("块大小（token）", range2("bs", 2, 0, 5)) + row("请求数", range2("n", 10, 2, 12)) +
      row("场景", select("mode", ["各自独立", "共享同一段长前缀（多轮对话）"], "各自独立"), true) + '</div>' +
      '<svg class="aw-chart aw-pk" viewBox="0 0 560 230"></svg><div class="aw-out"></div>';
    var svg = box.querySelector("svg"), out = box.querySelector(".aw-out");
    bind(box, function () {
      var bs = SIZES[val(box, "bs")], n = val(box, "n"), shared = val(box, "mode").indexOf("共享") === 0;
      show(box, "bs", String(bs)); show(box, "n", n + " 个");
      var L = lens(n, 7), pre = shared ? 900 : 0, i;
      var maxLen = 0, useful = 0, blocks = 0;
      var rows = L.map(function (l) {
        var total = pre + l; maxLen = Math.max(maxLen, total); useful += total;
        return total;
      });
      var sharedBlocks = shared ? Math.floor(pre / bs) : 0;   // 共享前缀的整块只分配一次
      var blocksNoShare = 0;
      rows.forEach(function (t) { blocks += Math.ceil(t / bs) - sharedBlocks; blocksNoShare += Math.ceil(t / bs); });
      blocks += sharedBlocks;
      var x0 = 20, W = 470, cw = Math.max(0.3, Math.min(16, W / Math.ceil(maxLen / bs))), rh = 15;
      var SHOW = Math.min(n, 12), S = svgText(x0, 12, "每个小格 = 一个 " + bs + " token 的物理块；浅色 = 块里没用满的部分", "start");
      for (i = 0; i < SHOW; i++) {
        var t = rows[i], nb = Math.ceil(t / bs), y = 24 + i * rh, tail = t % bs || bs;
        for (var b = 0; b < nb; b++) {
          var isPre = shared && b < sharedBlocks;
          var full = b < nb - 1;
          S += '<rect x="' + (x0 + b * cw).toFixed(1) + '" y="' + y + '" width="' + (cw - 1).toFixed(1) + '" height="' + (rh - 3) +
               '" class="' + (isPre ? "aw-f" : full ? "aw-on" : "aw-b") + '"/>';
        }
        S += svgText(x0 + nb * cw + 6, y + 6, t + " tok", "start");
      }
      var ly = 24 + SHOW * rh + 12;
      S += '<rect x="' + x0 + '" y="' + ly + '" width="10" height="10" class="aw-on"/>' + svgText(x0 + 16, ly + 6, "用满的块", "start");
      S += '<rect x="' + (x0 + 110) + '" y="' + ly + '" width="10" height="10" class="aw-b"/>' + svgText(x0 + 126, ly + 6, "最后一个块（平均浪费半块）", "start");
      if (shared) S += '<rect x="' + (x0 + 330) + '" y="' + ly + '" width="10" height="10" class="aw-f"/>' + svgText(x0 + 346, ly + 6, "共享前缀，只分配一份", "start");
      svg.setAttribute("viewBox", "0 0 560 " + (ly + 26));
      svg.innerHTML = S;
      var alloc = blocks * bs, allocNoShare = blocksNoShare * bs, reserve = n * 4096;
      out.innerHTML = "<p>这些请求逻辑上一共 <b>" + fmtInt(useful) + "</b> 个 token，分页之后占 <b>" + fmtInt(allocNoShare) +
        "</b> 个槽位，<b>" + (100 * useful / allocNoShare).toFixed(1) + "%</b> 是有效的（差的那点就是每个请求最后一个块没用满）。" +
        "同样这些请求按最大长度 4096 预留要 " + fmtInt(reserve) + " 个槽位，有效率只有 " +
        (100 * useful / reserve).toFixed(1) + "%。</p>" +
        (shared ? "<p>共享前缀只存一份之后，实际只占 <b>" + fmtInt(alloc) + "</b> 个槽位（" + blocks + " 个块），比每人存一份再省 <b>" +
          (100 * (1 - alloc / allocNoShare)).toFixed(0) + "%</b>。</p>" : "") +
        '<p class="aw-note">块越小越省（浪费只剩最后半个块），但块表更长、kernel 每次要处理的块更碎；' +
        '块越大越省调度开销，却把内部碎片放大。实践中 16 是个常见折中。' +
        '切到"共享长前缀"看另一半故事：多轮对话里系统提示 + 历史是所有请求共用的，分页让它们指向同一批物理块，' + 
        '这就是前缀缓存能省下大量显存和 prefill 的前提。</p>';
    });
  }

  // ---------------------------------------------------------------- 前缀缓存：基数树
  function radixcache(box) {
    var SCEN = {
      "多轮对话（越聊前缀越长）": [
        [["系统提示", 40], ["问题 A", 25]],
        [["系统提示", 40], ["问题 A", 25], ["回答 A", 80], ["问题 B", 20]],
        [["系统提示", 40], ["问题 A", 25], ["回答 A", 80], ["问题 B", 20], ["回答 B", 70], ["问题 C", 22]],
        [["系统提示", 40], ["问题 D", 30]],
        [["系统提示", 40], ["问题 D", 30], ["回答 D", 60], ["问题 E", 18]]
      ],
      "few-shot：同一段长示例 + 不同问题": [
        [["长示例", 320], ["问题 1", 25]], [["长示例", 320], ["问题 2", 30]],
        [["长示例", 320], ["问题 3", 22]], [["长示例", 320], ["问题 4", 28]],
        [["长示例", 320], ["问题 5", 26]]
      ],
      "互不相关的请求": [
        [["请求 1", 120]], [["请求 2", 150]], [["请求 3", 90]], [["请求 4", 200]], [["请求 5", 110]]
      ]
    };
    box.innerHTML = '<div class="aw-title">前缀缓存：基数树是怎么长出来的</div><div class="aw-grid">' +
      row("请求序列", select("scen", Object.keys(SCEN), "多轮对话（越聊前缀越长）"), true) +
      row("已经来了几个请求", range2("k", 3, 1, 5) + '<button type="button" data-k="play" class="aw-btn">播放</button>') + '</div>' +
      '<svg class="aw-chart aw-rx" viewBox="0 0 560 240"></svg><div class="aw-out"></div>';
    var svg = box.querySelector("svg"), out = box.querySelector(".aw-out"), timer = null;
    function stop() { if (timer) { clearInterval(timer); timer = null; box.querySelector('[data-k="play"]').textContent = "播放"; } }
    box.querySelector('[data-k="play"]').addEventListener("click", function () {
      if (timer) return stop();
      this.textContent = "暂停";
      input(box, "k").value = 1; draw();
      timer = setInterval(function () {
        if (!box.isConnected) return stop();
        var k = val(box, "k") + 1;
        input(box, "k").value = k; draw();
        if (k >= 5) stop();
      }, 900);
    });
    function draw() {
      var reqs = SCEN[val(box, "scen")], k = val(box, "k");
      show(box, "k", k + " / 5");
      var root = { ch: {}, depth: 0 }, totTok = 0, hitTok = 0, lastPath = [];
      for (var i = 0; i < k; i++) {
        var node = root, path = [];
        reqs[i].forEach(function (seg) {
          totTok += seg[1];
          var key = seg[0];
          var isHit = !!node.ch[key];
          if (isHit) hitTok += seg[1];
          if (!node.ch[key]) node.ch[key] = { name: key, tok: seg[1], ch: {}, depth: node.depth + 1, added: i };
          node = node.ch[key];
          path.push(node);
        });
        if (i === k - 1) lastPath = path;
      }
      var leaves = 0, rowsY = {};
      (function count(nd) { var kk = Object.keys(nd.ch); if (!kk.length) { leaves++; return; } kk.forEach(function (c) { count(nd.ch[c]); }); })(root);
      var y = 0, rh = Math.min(34, 190 / Math.max(1, leaves)), S = "";
      var NW = 70, GX = 86, x0 = 22, H = 26 + leaves * rh + 36;
      function place(nd) {
        var kk = Object.keys(nd.ch);
        if (!kk.length) { nd.y = 22 + y * rh + rh / 2; y++; return nd.y; }
        var ys = kk.map(function (c) { return place(nd.ch[c]); });
        nd.y = (Math.min.apply(null, ys) + Math.max.apply(null, ys)) / 2;
        return nd.y;
      }
      place(root);
      root.x = x0;
      (function drawn(nd) {
        Object.keys(nd.ch).forEach(function (c) {
          var ch = nd.ch[c];
          ch.x = x0 + ch.depth * GX;
          var onPath = lastPath.indexOf(ch) >= 0, isNew = ch.added === k - 1;
          var sx = nd.x + (nd === root ? 9 : NW / 2), ex = ch.x - NW / 2, mid = (sx + ex) / 2;
          S += '<path d="M ' + sx.toFixed(1) + ' ' + nd.y.toFixed(1) + ' C ' + mid.toFixed(1) + ' ' + nd.y.toFixed(1) + ' ' +
               mid.toFixed(1) + ' ' + ch.y.toFixed(1) + ' ' + ex.toFixed(1) + ' ' + ch.y.toFixed(1) +
               '" class="' + (onPath ? "aw-roof" : "aw-brace") + '" fill="none"/>';
          S += '<rect x="' + (ch.x - NW / 2) + '" y="' + (ch.y - 12) + '" width="' + NW + '" height="24" rx="4" class="' +
               (onPath ? (isNew ? "aw-b" : "aw-f") : "aw-off") + '"/>';
          S += svgText(ch.x, ch.y, ch.name + " " + ch.tok, "middle");
          drawn(ch);
        });
      })(root);
      S += '<circle cx="' + x0 + '" cy="' + root.y + '" r="8" class="aw-dot"/>' + svgText(x0, root.y + 22, "根", "middle");
      S += '<rect x="330" y="' + (H - 22) + '" width="10" height="10" class="aw-f"/>' + svgText(346, H - 16, "这次命中的", "start");
      S += '<rect x="440" y="' + (H - 22) + '" width="10" height="10" class="aw-b"/>' + svgText(456, H - 16, "这次新增的", "start");
      svg.setAttribute("viewBox", "0 0 560 " + H);
      svg.innerHTML = S;
      var last = reqs[k - 1], lastTot = 0, lastHit = 0, cur = root, stillHit = true;
      last.forEach(function (sg) {                            // 最后一个请求：前缀匹配到哪里为止
        lastTot += sg[1];
        var c = cur.ch[sg[0]];
        if (stillHit && c && c.added < k - 1) lastHit += sg[1]; else stillHit = false;
        cur = c || cur;
      });
      out.innerHTML = "<p>第 " + k + " 个请求一共 <b>" + lastTot + "</b> 个 token，其中 <b>" + lastHit +
        "</b> 个在缓存里命中（" + (100 * lastHit / lastTot).toFixed(0) + "%），只有剩下的 " + (lastTot - lastHit) +
        " 个需要真的做 prefill。前 " + k + " 个请求累计命中率 <b>" + (100 * hitTok / totTok).toFixed(0) + "%</b>。</p>" +
        '<p class="aw-note">树的每条边是一段 token 和它们的 KV 槽位，从根到某个节点的路径就是一个缓存过的前缀。' +
        '新请求沿树往下匹配，匹配不上的地方把边劈开、长出新枝。多轮对话和 few-shot 是最划算的两种形态：' +
        '前者越聊共享前缀越长，后者几百个 token 的示例被所有请求共用。互不相关的请求则完全没得省。</p>';
    }
    bind(box, function () { stop(); draw(); });
  }

  // ---------------------------------------------------------------- 调度：静态批 / 连续批 / 分块 prefill
  function contbatch(box) {
    var REQS = [                                             // [到达步, 提示词长度, 输出 token 数]
      [0, 900, 12], [0, 220, 18], [1, 1600, 8], [3, 380, 22],
      [5, 2400, 10], [7, 300, 16], [9, 1100, 14], [12, 260, 20]
    ];
    var MODES = ["静态批处理（攒满一批再跑）", "连续批处理", "连续批处理 + 分块 prefill"];
    box.innerHTML = '<div class="aw-title">调度：静态批、连续批与分块 prefill 的差别</div><div class="aw-grid">' +
      row("调度方式", select("mode", MODES, "连续批处理 + 分块 prefill"), true) +
      row("每步 token 预算", range2("budget", 2048, 256, 4096)) + '</div>' +
      '<svg class="aw-chart aw-cb" viewBox="0 0 560 250"></svg><div class="aw-out"></div>';
    var svg = box.querySelector("svg"), out = box.querySelector(".aw-out");
    bind(box, function () {
      var mode = val(box, "mode"), budget = Math.round(val(box, "budget") / 256) * 256;
      show(box, "budget", fmtInt(budget));
      var chunked = mode.indexOf("分块") > 0, isStatic = mode.indexOf("静态") === 0;
      var st = REQS.map(function (r) { return { arr: r[0], p: r[1], o: r[2], done: 0, gen: 0, ttft: -1, fin: -1 }; });
      var trace = [], step = 0, MAXSTEP = 160;
      while (step < MAXSTEP) {
        var running = st.filter(function (r) { return r.arr <= step && r.fin < 0; });
        if (!running.length) { if (st.every(function (r) { return r.fin >= 0; })) break; trace.push([]); step++; continue; }
        var batch = running;
        if (isStatic) {                                       // 静态批：攒满 4 个再跑，一批全部结束才换下一批
          var active = st.filter(function (r) { return r.arr <= step && r.fin < 0 && r.started; });
          if (active.length) batch = active;
          else {
            var waiting = running.filter(function (r) { return !r.started; });
            var allArrived = st.every(function (r) { return r.arr <= step; });
            if (waiting.length < 4 && !allArrived) { trace.push([]); step++; continue; }
            batch = waiting.slice(0, 4);
            batch.forEach(function (r) { r.started = 1; });
          }
        }
        var used = 0, cells = [];
        batch.forEach(function (r) {
          if (r.done < r.p) {                                 // 还在 prefill
            var want = r.p - r.done, take = chunked ? Math.min(want, Math.max(0, budget - used)) : want;
            if (!chunked && used > 0 && used + want > budget) return;   // 不分块时 prefill 独占
            if (take <= 0) return;
            r.done += take; used += take;
            cells.push([r, "p", take]);
            if (r.done >= r.p) r.ttft = step + 1;
          } else if (used + 1 <= budget) {                    // decode，一步一个 token
            r.gen += 1; used += 1;
            cells.push([r, "d", 1]);
            if (r.gen >= r.o) r.fin = step + 1;
          }
        });
        trace.push(cells);
        step++;
      }
      var steps = trace.length, x0 = 74, W = 470, cw = W / Math.max(1, steps), rh = 22;
      var S = svgText(x0, 12, "横轴 = 第几次前向，每一格是这一步里这个请求干的事", "start");
      st.forEach(function (r, i) {
        var y = 22 + i * rh;
        S += svgText(x0 - 8, y + 9, "请求 " + (i + 1), "end");
        S += '<rect x="' + x0 + '" y="' + y + '" width="' + W + '" height="' + (rh - 4) + '" class="aw-off"/>';
      });
      trace.forEach(function (cells, t) {
        cells.forEach(function (c) {
          var i = st.indexOf(c[0]), y = 22 + i * rh;
          S += '<rect x="' + (x0 + t * cw).toFixed(1) + '" y="' + y + '" width="' + Math.max(1.4, cw - 0.6).toFixed(1) +
               '" height="' + (rh - 4) + '" class="' + (c[1] === "p" ? "aw-b" : "aw-on") + '"/>';
        });
      });
      var ly = 22 + st.length * rh + 10;
      S += '<rect x="' + x0 + '" y="' + ly + '" width="10" height="10" class="aw-b"/>' + svgText(x0 + 16, ly + 6, "prefill", "start");
      S += '<rect x="' + (x0 + 90) + '" y="' + ly + '" width="10" height="10" class="aw-on"/>' + svgText(x0 + 106, ly + 6, "decode（一步一个 token）", "start");
      S += '<rect x="' + (x0 + 290) + '" y="' + ly + '" width="10" height="10" class="aw-off"/>' + svgText(x0 + 306, ly + 6, "没在这一批里", "start");
      svg.setAttribute("viewBox", "0 0 560 " + (ly + 26));
      svg.innerHTML = S;
      var ttfts = st.map(function (r) { return r.ttft < 0 ? steps : r.ttft - r.arr; });
      var maxT = Math.max.apply(null, ttfts), avgT = ttfts.reduce(function (a, b) { return a + b; }, 0) / ttfts.length;
      var busy = trace.reduce(function (a, c) { return a + (c.length ? 1 : 0); }, 0);
      out.innerHTML = "<p>" + st.length + " 个请求全部跑完用了 <b>" + steps + "</b> 步；首 token 延迟平均 <b>" +
        avgT.toFixed(1) + "</b> 步、最差 <b>" + maxT + "</b> 步。</p>" +
        '<p class="aw-note">静态批处理要等一批全部结束才换人，短请求陪着最长的那个一起等（看请求 2 后面那一长条空白）；' +
        '连续批处理让完成的请求立刻退出、新请求立刻补位。但只要 prefill 还独占一步，' +
        'decode 中的请求就会被一个长 prompt 卡住（TPOT 抖动）——分块 prefill 把长 prompt 切成几块，' +
        '和 decode 混在同一步里，代价是 prefill 自己变慢一点。token 预算就是这两者之间的旋钮。</p>';
    });
  }

  // ---------------------------------------------------------------- 投机解码：树形草稿
  function spectree(box) {
    box.innerHTML = '<div class="aw-title">投机解码：草稿树的宽度、深度与加速比</div><div class="aw-grid">' +
      row("每层候选数（宽度）", range2("k", 2, 1, 4)) + row("草稿深度", range2("d", 4, 1, 6)) +
      row("草稿 top-1 命中率", range2("a", 70, 30, 95)) + row("草稿一步 ÷ 目标一步", range2("c", 15, 2, 40)) + '</div>' +
      '<svg class="aw-chart aw-sp" viewBox="0 0 560 210"></svg><div class="aw-out"></div>';
    var svg = box.querySelector("svg"), out = box.querySelector(".aw-out");
    bind(box, function () {
      var k = val(box, "k"), d = val(box, "d"), a = val(box, "a") / 100, c = val(box, "c") / 100;
      show(box, "k", k + " 个"); show(box, "d", d + " 层"); show(box, "a", (100 * a).toFixed(0) + "%");
      show(box, "c", (100 * c).toFixed(0) + "%");
      var p = 1 - Math.pow(1 - a, k);                         // 简化模型：k 个候选独立，命中率 1-(1-α)^k
      var exp = 1, acc = 0, i;
      for (i = 1; i <= d; i++) { acc += Math.pow(p, i); }
      exp = 1 + acc;                                          // 接受 a 个就吐出 a+1 个 token
      var nodes = 0;
      for (i = 1; i <= d; i++) nodes += Math.pow(k, i);
      var cost = 1 + c * d, speed = exp / cost;
      // 画树
      var x0 = 30, W = 330, S = svgText(x0, 12, "草稿树：每层取 top-" + k + "，整棵树在一次前向里验证完", "start");
      var levelY = function (l) { return 34 + l * Math.min(30, 150 / d); };
      var prev = [[x0 + W / 2, levelY(0)]];
      S += '<circle cx="' + (x0 + W / 2) + '" cy="' + levelY(0) + '" r="6" class="aw-dot"/>';
      for (i = 1; i <= d; i++) {
        var cnt = Math.min(Math.pow(k, i), 16), cur = [], j;
        var pAcc = Math.pow(p, i);
        for (j = 0; j < cnt; j++) {
          var x = x0 + (cnt === 1 ? W / 2 : (j + 0.5) / cnt * W), y = levelY(i);
          var par = prev[Math.min(prev.length - 1, Math.floor(j / Math.max(1, cnt / prev.length)))];
          S += '<line x1="' + par[0].toFixed(1) + '" y1="' + (par[1] + 5) + '" x2="' + x.toFixed(1) + '" y2="' + (y - 5) + '" class="aw-brace"/>';
          S += '<circle cx="' + x.toFixed(1) + '" cy="' + y + '" r="5" class="' + (j === 0 ? "aw-on" : "aw-off") + '" fill-opacity="' +
               (0.2 + 0.75 * pAcc).toFixed(2) + '"/>';
          cur.push([x, y]);
        }
        S += svgText(x0 + W + 14, levelY(i), "第 " + i + " 层：走到这里的概率 " + (100 * pAcc).toFixed(0) + "%", "start");
        prev = cur;
      }
      svg.setAttribute("viewBox", "0 0 560 " + (levelY(d) + 30));
      svg.innerHTML = S;
      var chainP = a, chainExp = 1;
      for (i = 1; i <= d; i++) chainExp += Math.pow(chainP, i);
      out.innerHTML = "<p>每层命中率从 top-1 的 " + (100 * a).toFixed(0) + "% 提到 <b>" + (100 * p).toFixed(0) +
        "%</b>（k 个候选里中一个就行）。一次验证平均吐出 <b>" + exp.toFixed(2) + "</b> 个 token" +
        "（链式草稿只有 " + chainExp.toFixed(2) + " 个），验证的节点数 <b>" + nodes + "</b>，" +
        "算上草稿开销后的加速比 <b>" + speed.toFixed(2) + "×</b>" + (speed < 1 ? "（<b>反而变慢了</b>）" : "") + "。</p>" +
        '<p class="aw-note">三个旋钮的方向相反：深度越大期望越长，但越往后越走不到（概率是连乘），而草稿开销线性增长；' +
        '宽度提高每层命中率，代价是验证的 token 数按 k 的幂次涨——batch 大的时候目标模型本来就把算力吃满了，' +
        '多验证的 token 不再免费，投机解码就失效。所以它在低并发、长输出、草稿准的场景收益最大。' +
        '（这里用的是"k 个候选独立"的简化模型，真实的树会剪枝，命中率也不会这么理想。）</p>';
    });
  }

  var WIDGETS = { "kv-calc": kvCalc, roofline: roofline, mask: mask, pipeline: pipeline,
                  linmap: linmap, lowrank: lowrank, softmax: softmaxw, graddesc: graddesc,
                  coalesce: coalesce, bankconf: bankconf, scanviz: scanviz, occupancy: occupancy,
                  pagewalk: pagewalk, cachemap: cachemap, hashring: hashring,
                  pagedkv: pagedkv, radixcache: radixcache, contbatch: contbatch, spectree: spectree };
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
