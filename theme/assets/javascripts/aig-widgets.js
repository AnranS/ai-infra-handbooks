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
//   ringreduce 环形 all-reduce 的每一步（分布式训练 · 集合通信原语）
//   zeromem   ZeRO 各级每卡显存（分布式训练 · ZeRO 与 FSDP）
//   structlayout 结构体布局：字段顺序与填充（C++ · 对象布局、对齐与缓存）
//   embed3d   词向量空间：真实嵌入的三维投影，可拖动旋转（大模型原理 · 嵌入层与输出层）
//   rope-helix RoPE：每一对维度随位置旋转画成螺旋（大模型原理 · 位置编码与 RoPE）
//   swiglu3d  门控激活的曲面：门(a) × 内容 b（大模型原理 · 前馈网络与 SwiGLU）
//   scaling3d Scaling Law 的损失曲面与等算力线（大模型原理 · 预训练与 Scaling Law）
//   nextword  语言模型：每个位置的条件概率与交叉熵，真实模型的数据（大模型原理 · 语言模型）
//   broadcast 广播：两个形状怎么对齐、哪一维被拉伸（大模型原理 · 数学与 PyTorch 预备）
//   bpe       BPE：一步一步合并，再用学到的规则编码新文本（大模型原理 · 分词）
//   float-bits 浮点数拆成符号、指数、尾数，看舍入误差和 ulp（大模型原理 · 浮点与数值计算）
//   attention2d 注意力 = 软查表：拖动 query 看权重和加权平均（大模型原理 · 注意力机制）
//   norm      LayerNorm 与 RMSNorm 对同一个向量各做了什么（大模型原理 · 归一化与残差流）
//   param-share 一层和整个模型的参数怎么分，注意力分数的 T² 项何时成为大头（大模型原理 · 前馈网络与 SwiGLU）
//   moe-route MoE 路由：top-k、负载不均衡、容量与丢弃（大模型原理 · 混合专家 MoE）
// 后四个用文件中段的 view3d 小引擎：SVG 里的画家算法 + 拖动旋转，不依赖任何 3D 库。
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

  // ---------------------------------------------------------------- 环形 all-reduce 分步演示
  function ringreduce(box) {
    box.innerHTML = '<div class="aw-title">环形 all-reduce：2(n−1) 步之后每张卡都拿到全量</div><div class="aw-grid">' +
      row("卡数 n", range2("n", 4, 3, 6)) +
      row("第几步", range2("step", 0, 0, 6) + '<button type="button" data-k="play" class="aw-btn">播放</button>') +
      row("每张卡的梯度大小", range2("mb", 256, 16, 2048), true) + '</div>' +
      '<svg class="aw-chart aw-rr" viewBox="0 0 560 230"></svg><div class="aw-out"></div>';
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
      }, 900);
    });
    function draw() {
      var n = val(box, "n"), sl = input(box, "step"), mb = val(box, "mb");
      sl.max = 2 * (n - 1);
      var t = Math.min(val(box, "step"), 2 * (n - 1));
      show(box, "n", n + " 张"); show(box, "step", t + " / " + (2 * (n - 1)));
      show(box, "mb", mb + " MB");
      // own[r][c] = 第 r 张卡手里第 c 块包含了哪些 rank 的贡献
      var own = [], r, c, i;
      for (r = 0; r < n; r++) { own[r] = []; for (c = 0; c < n; c++) own[r][c] = (c === r) ? [r] : [r]; }
      for (r = 0; r < n; r++) for (c = 0; c < n; c++) own[r][c] = [r];
      var phase = t < n ? "reduce-scatter" : "all-gather";
      for (i = 0; i < Math.min(t, n - 1); i++) {              // reduce-scatter：把块往右传并累加
        var nx = own.map(function (row) { return row.map(function (v) { return v.slice(); }); });
        for (r = 0; r < n; r++) {
          var src = (r - 1 + n) % n, cc = (r - i + n) % n;
          nx[r][cc] = own[src][cc].concat(own[r][cc]).filter(function (v, k, arr) { return arr.indexOf(v) === k; });
        }
        own = nx;
      }
      for (i = 0; i < Math.max(0, t - (n - 1)); i++) {         // all-gather：把完整块再传一圈
        var nx2 = own.map(function (row) { return row.map(function (v) { return v.slice(); }); });
        for (r = 0; r < n; r++) {
          var src2 = (r - 1 + n) % n, cc2 = (r - i + 1 + n) % n;
          nx2[r][cc2] = own[src2][cc2].slice();
        }
        own = nx2;
      }
      var x0 = 74, cw = Math.min(56, 460 / n), rh = 30, S = "";
      S += svgText(x0, 14, t === 0 ? "开始：每张卡只有自己的那份梯度" :
        (t <= n - 1 ? "reduce-scatter 第 " + t + " 步：每张卡把一块发给右边、收一块累加"
                    : "all-gather 第 " + (t - n + 1) + " 步：把累加好的块沿环再传一圈"), "start");
      for (c = 0; c < n; c++) S += svgText(x0 + c * cw + cw / 2, 34, "块 " + c, "middle");
      for (r = 0; r < n; r++) {
        var y = 44 + r * rh;
        S += svgText(x0 - 10, y + 12, "卡 " + r, "end");
        for (c = 0; c < n; c++) {
          var full = own[r][c].length === n;
          S += '<rect x="' + (x0 + c * cw + 1).toFixed(1) + '" y="' + y + '" width="' + (cw - 3).toFixed(1) + '" height="' + (rh - 6) +
               '" rx="3" class="' + (full ? "aw-on" : own[r][c].length > 1 ? "aw-b" : "aw-off") + '"/>';
          S += svgText(x0 + c * cw + cw / 2, y + 12, own[r][c].length === n ? "全" : own[r][c].length + " 份", "middle");
        }
      }
      var ly = 44 + n * rh + 8;
      S += '<rect x="' + x0 + '" y="' + ly + '" width="10" height="10" class="aw-off"/>' + svgText(x0 + 16, ly + 6, "只有自己的", "start");
      S += '<rect x="' + (x0 + 110) + '" y="' + ly + '" width="10" height="10" class="aw-b"/>' + svgText(x0 + 126, ly + 6, "累加了一部分", "start");
      S += '<rect x="' + (x0 + 230) + '" y="' + ly + '" width="10" height="10" class="aw-on"/>' + svgText(x0 + 246, ly + 6, "已经是全量的和", "start");
      svg.setAttribute("viewBox", "0 0 560 " + (ly + 26));
      svg.innerHTML = S;
      var perStep = mb / n, totalSent = 2 * (n - 1) * perStep;
      out.innerHTML = "<p>每一步每张卡只发一块（" + perStep.toFixed(1) + " MB），一共 " + (2 * (n - 1)) + " 步，" +
        "每张卡总共发送 <b>" + totalSent.toFixed(0) + " MB</b> ≈ 2 × " + mb + " MB ×(n−1)/n。" +
        "<b>和卡数几乎无关</b>——这正是环形算法能扩展的原因。</p>" +
        '<p class="aw-note">朴素做法（所有卡把梯度发给 rank 0 再广播回来）会让 rank 0 的网卡成为瓶颈，' +
        '通信量随卡数线性增长。环形算法把带宽压力均分到每条链路上，代价是延迟随卡数线性增长（2(n−1) 次握手），' +
        '所以小消息上 NCCL 改用树形算法：延迟 O(log n)，带宽差一点。</p>';
    }
    bind(box, function () { stop(); draw(); });
  }

  // ---------------------------------------------------------------- ZeRO 各级的每卡显存
  function zeromem(box) {
    box.innerHTML = '<div class="aw-title">ZeRO：切掉哪些状态，每张卡还剩多少</div><div class="aw-grid">' +
      row("参数量（十亿）", range2("psi", 7, 1, 70)) + row("数据并行度 N", range2("n", 8, 1, 64)) +
      row("激活 + 其他（GB/卡）", range2("act", 12, 0, 40)) + row("单卡显存 GB", range2("cap", 80, 24, 192)) + '</div>' +
      '<svg class="aw-chart aw-zr" viewBox="0 0 560 210"></svg><div class="aw-out"></div>';
    var svg = box.querySelector("svg"), out = box.querySelector(".aw-out");
    bind(box, function () {
      var psi = val(box, "psi") * 1e9, N = val(box, "n"), act = val(box, "act"), cap = val(box, "cap");
      show(box, "psi", val(box, "psi") + " B"); show(box, "n", N + " 卡");
      show(box, "act", act + " GB"); show(box, "cap", cap + " GB");
      var G = 1024 * 1024 * 1024;
      // bf16 参数 2Ψ + bf16 梯度 2Ψ + fp32 参数 4Ψ + Adam m/v 8Ψ = 16Ψ
      var LV = [
        ["数据并行", 2, 2, 12, 1, 1, 1], ["ZeRO-1", 2, 2, 12, 1, 1, N],
        ["ZeRO-2", 2, 2, 12, 1, N, N], ["ZeRO-3", 2, 2, 12, N, N, N]
      ];
      var bars = LV.map(function (l) {
        var p = l[1] * psi / l[4], g = l[2] * psi / l[5], o = l[3] * psi / l[6];
        return { name: l[0], parts: [["参数", p], ["梯度", g], ["优化器状态", o]], sum: (p + g + o) / G + act };
      });
      var mx = Math.max(cap, bars[0].sum), x0 = 92, W = 380, S = "";
      S += svgText(x0, 14, "每张卡的显存占用（GB）", "start");
      bars.forEach(function (b, i) {
        var y = 26 + i * 40, cx = x0;
        S += svgText(x0 - 10, y + 13, b.name, "end");
        var cls = ["aw-on", "aw-b", "aw-f"];
        b.parts.forEach(function (pt, j) {
          var w = pt[1] / G / mx * W;
          S += '<rect x="' + cx.toFixed(1) + '" y="' + y + '" width="' + w.toFixed(1) + '" height="26" class="' + cls[j] + '"/>';
          cx += w;
        });
        var aw = act / mx * W;
        S += '<rect x="' + cx.toFixed(1) + '" y="' + y + '" width="' + aw.toFixed(1) + '" height="26" class="aw-off"/>';
        S += svgText(cx + aw + 8, y + 13, b.sum.toFixed(1) + (b.sum > cap ? " ✗" : " ✓"), "start");
      });
      var xc = x0 + cap / mx * W;
      S += '<line x1="' + xc.toFixed(1) + '" y1="20" x2="' + xc.toFixed(1) + '" y2="' + (26 + 4 * 40) + '" class="aw-dash"/>';
      S += svgText(xc, 186, "单卡 " + cap + " GB", "middle");
      var lx = x0, names = [["参数", "aw-on"], ["梯度", "aw-b"], ["优化器状态", "aw-f"], ["激活等", "aw-off"]];
      names.forEach(function (nm) {
        S += '<rect x="' + lx + '" y="196" width="10" height="10" class="' + nm[1] + '"/>' + svgText(lx + 16, 202, nm[0], "start");
        lx += nm[0].length * 13 + 34;
      });
      svg.innerHTML = S;
      var fit = bars.filter(function (b) { return b.sum <= cap; });
      out.innerHTML = "<p>" + (fit.length ? "最省事的可行方案是 <b>" + fit[0].name + "</b>（每卡 " + fit[0].sum.toFixed(1) + " GB）。"
        : "<b>四级都放不下</b>，得再叠张量并行 / 流水线并行，或者做激活重计算。") +
        " 数据并行要 " + bars[0].sum.toFixed(1) + " GB，ZeRO-3 只要 " + bars[3].sum.toFixed(1) + " GB。</p>" +
        '<p class="aw-note">混合精度 Adam 下模型状态是 16Ψ 字节：bf16 参数 2Ψ + bf16 梯度 2Ψ + fp32 参数副本 4Ψ + 一阶二阶动量 8Ψ。' +
        'ZeRO-1、2 不增加通信量（all-reduce 本来就是 reduce-scatter + all-gather）；' +
        'ZeRO-3 连参数都要临时 all-gather，通信量约 1.5 倍，所以能用低一级就别上高一级。' +
        '注意激活那一段是不随 N 变小的，长序列训练里它往往才是大头。</p>';
    });
  }

  // ---------------------------------------------------------------- 结构体布局与填充
  function structlayout(box) {
    var F = [                                                // 名字、类型、大小、对齐
      ["done", "bool", 1, 1], ["temperature", "double", 8, 8], ["len", "int32_t", 4, 4],
      ["stream", "bool", 1, 1], ["id", "int64_t", 8, 8]
    ];
    var COL = ["aw-on", "aw-b", "aw-f", "aw-i2", "aw-j2"];
    var order = [0, 1, 2, 3, 4];
    box.innerHTML = '<div class="aw-title">结构体布局：换个字段顺序，sizeof 就变了</div>' +
      '<div class="aw-row aw-wide"><span>字段顺序</span><span class="aw-chips"></span></div>' +
      '<div class="aw-row aw-wide"><span></span>' +
      '<button type="button" data-k="sort" class="aw-btn">按对齐从大到小排</button>' +
      '<button type="button" data-k="reset" class="aw-btn">还原成随手写的顺序</button></div>' +
      '<svg class="aw-chart aw-sl" viewBox="0 0 560 200"></svg><div class="aw-out"></div>';
    var svg = box.querySelector("svg"), out = box.querySelector(".aw-out"), chips = box.querySelector(".aw-chips");
    box.querySelector('[data-k="sort"]').addEventListener("click", function () {
      order = order.slice().sort(function (a, b) { return F[b][3] - F[a][3]; }); draw();
    });
    box.querySelector('[data-k="reset"]').addEventListener("click", function () { order = [0, 1, 2, 3, 4]; draw(); });
    chips.addEventListener("click", function (e) {
      var b = e.target.closest("button[data-i]");
      if (!b) return;
      var i = +b.dataset.i;
      if (i > 0) { var t = order[i - 1]; order[i - 1] = order[i]; order[i] = t; draw(); }
      else { order.push(order.shift()); draw(); }             // 第一个再点一下就转到末尾
    });
    function draw() {
      chips.innerHTML = order.map(function (f, i) {
        return '<button type="button" class="aw-chip" data-i="' + i + '">' + F[f][1] + " " + F[f][0] + "</button>";
      }).join("");
      var off = 0, seg = [], align = 1;
      order.forEach(function (f) {
        var sz = F[f][2], al = F[f][3];
        align = Math.max(align, al);
        var pad = (al - off % al) % al;
        if (pad) seg.push([null, off, pad]);
        off += pad;
        seg.push([f, off, sz]);
        off += sz;
      });
      var tailPad = (align - off % align) % align;
      if (tailPad) seg.push([null, off, tailPad]);
      var total = off + tailPad, useful = F.reduce(function (a, x) { return a + x[2]; }, 0);
      var perRow = 8, bw = 52, rows = Math.ceil(total / perRow), x0 = 46;
      var S = svgText(x0, 14, "每一行 8 个字节（一个 64 位字）；灰色斜纹是编译器插进去的填充", "start");
      var i, r;
      for (r = 0; r < rows; r++) S += svgText(x0 - 10, 42 + r * 40, "+" + (r * 8), "end");
      seg.forEach(function (sg) {
        var f = sg[0], start = sg[1], len = sg[2];
        for (i = 0; i < len; i++) {
          var b = start + i, x = x0 + (b % perRow) * bw, y = 30 + Math.floor(b / perRow) * 40;
          S += '<rect x="' + (x + 1) + '" y="' + y + '" width="' + (bw - 2) + '" height="24" rx="2" class="' +
               (f === null ? "aw-pad" : COL[f % COL.length]) + '"/>';
        }
        if (f !== null) {                                    // 名字画在它占的第一段上
          var firstLen = Math.min(len, perRow - start % perRow);
          S += svgText(x0 + (start % perRow) * bw + firstLen * bw / 2,
                       30 + Math.floor(start / perRow) * 40 + 12, F[f][0], "middle");
        }
      });
      var ly = 30 + rows * 40 + 6;
      S += '<rect x="' + x0 + '" y="' + ly + '" width="10" height="10" class="aw-pad"/>' + svgText(x0 + 16, ly + 6, "填充字节（白占地方）", "start");
      svg.setAttribute("viewBox", "0 0 560 " + (ly + 26));
      svg.innerHTML = S;
      out.innerHTML = "<p><code>sizeof</code> = <b>" + total + "</b>，<code>alignof</code> = <b>" + align +
        "</b>；真正的数据只有 " + useful + " 字节，填充占了 <b>" + (total - useful) + "</b> 字节（" +
        (100 * (total - useful) / total).toFixed(0) + "%）。</p>" +
        '<p class="aw-note">规则只有两条：每个成员的偏移必须是它对齐要求的倍数，结构体总大小要补齐到最大对齐的倍数。' +
        '所以把大对齐的字段放前面，小的挤在后面，填充就最少。一个 32 字节的结构体缩到 24 字节，' +
        '一条 64 字节缓存行能多装一个——批量遍历时这直接变成带宽。' +
        '（点字段可以把它往前挪一位，点第一个会把它转到末尾。）</p>';
    }
    draw();
  }

  // ---------------------------------------------------------------- 3D 小引擎：在 SVG 里画可拖动旋转的点、线、面
  // view3d(svg, opts) → { set(items), draw(), stop() }。items 的每一项是：
  //   { t: "pt", p: [x, y, z], r, fill, label }        { t: "seg", a, b, cls, sw, dash }       { t: "arrow", a, b, cls, sw, fill }
  //   { t: "poly", pts: [[x, y, z], ...], fill, op }   { t: "text", p, s, anchor }
  // 坐标：y 朝上，z 朝向观察者。先绕 y 轴转 ay、再绕 x 轴转 ax；拖动改变两个角，auto 为真时慢慢自转，直到用户碰它。
  // 画法是画家算法：按旋转后的深度排序，远的先画，所以面不要互相穿插。
  function view3d(svg, opts) {
    var W = opts.w || 560, H = opts.h || 340, cx = opts.cx || W / 2, cy = opts.cy || H / 2, scale = opts.scale || 90;
    var ax = opts.ax == null ? 0.42 : opts.ax, ay = opts.ay == null ? -0.65 : opts.ay;
    var items = [], auto = opts.auto !== false, timer = null, drag = null;
    svg.setAttribute("viewBox", "0 0 " + W + " " + H);
    svg.classList.add("aw-3d");
    function rot(p) {
      var c1 = Math.cos(ay), s1 = Math.sin(ay), c2 = Math.cos(ax), s2 = Math.sin(ax);
      var x = p[0] * c1 + p[2] * s1, z = -p[0] * s1 + p[2] * c1, y = p[1];
      return [x, y * c2 - z * s2, y * s2 + z * c2];
    }
    function P(p) {                                  // 投影到画布：[sx, sy, 深度]，深度越大越靠近观察者
      var q = rot(p), f = opts.persp === false ? 1 : 1 / (1 - q[2] / (opts.dist || 9));
      return [cx + q[0] * scale * f, cy - q[1] * scale * f, q[2]];
    }
    function f1(v) { return (Math.round(v * 10) / 10).toString(); }
    function draw() {
      var ordered = items.map(function (it) {
        var z, k;
        if (it.t === "poly") { z = 0; for (k = 0; k < it.pts.length; k++) z += rot(it.pts[k])[2]; z /= it.pts.length; }
        else if (it.t === "seg" || it.t === "arrow") z = (rot(it.a)[2] + rot(it.b)[2]) / 2;
        else z = rot(it.p)[2];
        return [z + (it.z || 0), it];
      }).sort(function (a, b) { return a[0] - b[0]; });                     // 远的先画
      var S = "", i;
      for (i = 0; i < ordered.length; i++) {
        var it = ordered[i][1];
        if (it.t === "poly") {
          var pts = it.pts.map(P), q0 = rot(it.pts[0]), q1 = rot(it.pts[1]), q2 = rot(it.pts[2]);
          var ux = q1[0] - q0[0], uy = q1[1] - q0[1], uz = q1[2] - q0[2], vx = q2[0] - q0[0], vy = q2[1] - q0[1], vz = q2[2] - q0[2];
          var nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx, nl = Math.sqrt(nx * nx + ny * ny + nz * nz) || 1;
          var light = Math.abs(nz / nl) * 0.7 + Math.abs(ny / nl) * 0.3;      // 正对观察者的面最亮，侧着的略暗
          S += '<polygon points="' + pts.map(function (p) { return f1(p[0]) + "," + f1(p[1]); }).join(" ") + '" fill="' + it.fill +
            '" fill-opacity="' + (0.25 + 0.6 * light * (it.op == null ? 1 : it.op)).toFixed(2) + '" class="' + (it.cls || "aw-mesh") + '"/>';
        } else if (it.t === "seg" || it.t === "arrow") {
          var a = P(it.a), b = P(it.b);
          S += '<line x1="' + f1(a[0]) + '" y1="' + f1(a[1]) + '" x2="' + f1(b[0]) + '" y2="' + f1(b[1]) + '" class="' + (it.cls || "aw-ax3") + '"' +
            (it.sw ? ' stroke-width="' + it.sw + '"' : "") + (it.dash ? ' stroke-dasharray="' + it.dash + '"' : "") +
            (it.stroke ? ' stroke="' + it.stroke + '"' : "") + "/>";
          if (it.t === "arrow") {
            var ang = Math.atan2(b[1] - a[1], b[0] - a[0]), hs = it.hs || 7;
            S += '<polygon points="' + f1(b[0]) + "," + f1(b[1]) + " " + f1(b[0] - hs * Math.cos(ang - 0.45)) + "," + f1(b[1] - hs * Math.sin(ang - 0.45)) +
              " " + f1(b[0] - hs * Math.cos(ang + 0.45)) + "," + f1(b[1] - hs * Math.sin(ang + 0.45)) + '" fill="' + (it.fill || it.stroke || "currentColor") + '"/>';
          }
        } else if (it.t === "pt") {
          var c = P(it.p);
          S += '<circle cx="' + f1(c[0]) + '" cy="' + f1(c[1]) + '" r="' + (it.r || 4) + '" fill="' + it.fill + '" class="' + (it.cls || "aw-p3") + '"/>';
          if (it.label) S += '<text x="' + f1(c[0] + (it.lx == null ? (it.r || 4) + 2 : it.lx)) + '" y="' + f1(c[1] + (it.ly == null ? 3.5 : it.ly)) + '" class="aw-t ' + (it.lcls || "aw-ptl") + '"' +
            (it.anchor ? ' text-anchor="' + it.anchor + '"' : "") + ">" + it.label + "</text>";
        } else if (it.t === "text") {
          var tp = P(it.p);
          S += svgText(f1(tp[0]), f1(tp[1]), it.s, it.anchor || "middle");
        }
      }
      svg.innerHTML = S;
    }
    function stop() { if (timer) { clearInterval(timer); timer = null; } }
    function visible() { var r = svg.getBoundingClientRect(); return r.bottom > 0 && r.top < (window.innerHeight || 800); }
    if (auto) timer = setInterval(function () { if (!svg.isConnected) return stop(); if (drag || !visible()) return; ay += 0.006; draw(); }, 50);
    svg.addEventListener("pointerdown", function (e) { drag = [e.clientX, e.clientY]; stop(); svg.setPointerCapture(e.pointerId); e.preventDefault(); });
    svg.addEventListener("pointermove", function (e) {
      if (!drag) return;
      var k = 2 * Math.PI / Math.max(200, svg.clientWidth || W);               // 拖过整个宽度正好转一圈
      ay += (e.clientX - drag[0]) * k; ax = Math.max(-1.5, Math.min(1.5, ax + (e.clientY - drag[1]) * k));
      drag = [e.clientX, e.clientY]; draw();
    });
    svg.addEventListener("pointerup", function () { drag = null; });
    svg.addEventListener("pointercancel", function () { drag = null; });
    return { set: function (list) { items = list; draw(); }, draw: draw, stop: stop };
  }
  function axes3(len, names) {                       // 三根坐标轴和端点的名字，总是画在最底层
    var L = [], dirs = [[len, 0, 0], [0, len, 0], [0, 0, len]];
    for (var i = 0; i < 3; i++) {
      L.push({ t: "arrow", a: [0, 0, 0], b: i === 1 ? [0, len * 0.72, 0] : dirs[i], cls: "aw-ax3", z: -99, hs: 6 });
      if (names[i]) L.push({ t: "text", p: dirs[i].map(function (v) { return v * 1.13; }), s: names[i], z: 99 });
    }
    return L;
  }
  function surface3(fn, x0, x1, z0, z1, n, color) {  // y = fn(x, z) 的网格面，color(y) 给每个小面的颜色
    var L = [], g = [], i, j;
    for (i = 0; i <= n; i++) { g.push([]); for (j = 0; j <= n; j++) { var x = x0 + (x1 - x0) * i / n, z = z0 + (z1 - z0) * j / n; g[i].push([x, fn(x, z), z]); } }
    for (i = 0; i < n; i++) for (j = 0; j < n; j++) {
      var q = [g[i][j], g[i + 1][j], g[i + 1][j + 1], g[i][j + 1]];
      L.push({ t: "poly", pts: q, fill: color((q[0][1] + q[1][1] + q[2][1] + q[3][1]) / 4) });
    }
    return L;
  }
  function heat(t) { t = Math.max(0, Math.min(1, t)); return "hsl(" + Math.round(215 - 190 * t) + ", 78%, 50%)"; }   // 蓝（低）→ 橙（高）
  function gelu(x) { return 0.5 * x * (1 + Math.tanh(0.7978845608 * (x + 0.044715 * x * x * x))); }

  // ---------------------------------------------------------------- 词向量的三维投影（真实模型的嵌入做 PCA）
  // 数据来自 Qwen3-0.6B 的输入嵌入（1024 维）：挑 59 个单 token 的词，在这些词上做 PCA 取前三维（保留 24% 的方差）；
  // 对照组是随机挑的三个方向。每一项：[词, 组, PCA 三维, 随机三维]
  var EMBED3D = [["中国","国家",[0.278,-0.205,0.113],[-0.003,0.026,0.648]],["美国","国家",[0.482,-0.235,0.071],[-0.403,-0.102,-0.421]],["日本","国家",[0.58,-0.395,0.009],[0.409,-0.197,-0.107]],["法国","国家",[0.665,-0.362,0.007],[0.044,0.194,-0.063]],["英国","国家",[0.595,-0.257,0.005],[0.067,0.399,-0.193]],["德国","国家",[0.592,-0.298,0.024],[-0.571,0.379,0.2]],["印度","国家",[0.453,-0.174,0.137],[-0.083,-0.298,-0.061]],["韩国","国家",[0.539,-0.241,0.003],[0.105,0.133,-0.043]],["俄罗斯","国家",[0.518,-0.284,0.043],[0.055,-0.466,0.439]],["意大利","国家",[0.569,-0.249,-0.004],[0.325,0.135,0.658]],["北京","城市",[0.392,-0.229,0.142],[-0.089,0.095,0.132]],["上海","城市",[0.406,-0.219,0.13],[0.223,-0.152,0.295]],["东京","城市",[0.668,-0.451,-0.029],[-0.203,-0.568,-0.42]],["巴黎","城市",[0.692,-0.438,-0.027],[-0.176,0.159,0.226]],["伦敦","城市",[0.671,-0.367,-0.011],[0.152,0.247,-0.111]],["柏林","城市",[0.501,-0.195,0.058],[-0.025,-0.569,0.165]],["纽约","城市",[0.494,-0.218,0.041],[-0.103,-0.632,0.23]],["深圳","城市",[0.349,-0.15,0.09],[0.496,-0.001,-0.016]],["杭州","城市",[0.469,-0.242,0.061],[0.063,-0.361,-0.013]],["一","数字",[-0.5,-0.109,0.153],[-0.123,-0.286,-0.146]],["二","数字",[-0.671,-0.212,0.078],[0.128,0.31,-0.026]],["三","数字",[-0.809,-0.357,0.039],[-0.148,-0.183,-0.639]],["四","数字",[-0.906,-0.509,-0.008],[-0.238,0.051,0.125]],["五","数字",[-0.934,-0.571,-0.049],[-0.046,0.017,0.022]],["六","数字",[-0.97,-0.605,-0.033],[-0.489,0.396,0.234]],["七","数字",[-1.0,-0.608,-0.117],[-0.502,0.195,0.256]],["八","数字",[-0.99,-0.594,-0.138],[-0.626,0.299,-0.137]],["九","数字",[-0.868,-0.53,-0.13],[-0.481,0.332,0.242]],["十","数字",[-0.673,-0.351,-0.094],[-0.54,0.424,0.47]],["猫","动物",[0.017,0.661,-0.66],[0.621,-0.32,-0.615]],["狗","动物",[-0.045,0.618,-0.64],[-0.441,0.269,-1.0]],["马","动物",[-0.182,0.535,-0.208],[0.02,-0.569,-0.346]],["牛","动物",[-0.126,0.688,-0.353],[-0.059,-0.75,0.207]],["羊","动物",[-0.042,0.662,-0.374],[0.286,0.101,-0.339]],["鱼","动物",[-0.082,0.571,-0.516],[-0.205,0.235,0.33]],["鸟","动物",[-0.004,0.532,-0.477],[-0.321,-0.487,-0.38]],["虎","动物",[-0.129,0.518,-0.37],[0.132,-0.335,0.119]],["猪","动物",[-0.05,0.547,-0.5],[-0.079,0.015,-0.339]],["鸡","动物",[-0.004,0.593,-0.518],[0.481,-0.868,0.111]],["红","颜色",[-0.273,0.575,0.894],[0.319,-0.074,0.279]],["黄","颜色",[-0.244,0.503,0.848],[-0.032,0.489,-0.394]],["蓝","颜色",[-0.172,0.583,0.945],[0.271,-0.018,-0.289]],["绿","颜色",[-0.149,0.486,0.871],[0.197,0.012,0.064]],["黑","颜色",[-0.155,0.555,0.751],[0.365,0.517,-0.687]],["白","颜色",[-0.268,0.559,0.884],[0.471,0.123,-0.385]],["紫","颜色",[-0.129,0.372,0.655],[0.315,0.587,0.362]],["灰","颜色",[-0.103,0.512,0.512],[0.349,0.48,-0.287]],["king","英文",[0.043,0.197,-0.214],[0.203,0.252,0.059]],["queen","英文",[0.106,0.201,-0.211],[0.808,0.115,0.285]],["man","英文",[-0.162,0.338,-0.128],[-0.334,-0.399,0.079]],["woman","英文",[-0.015,0.246,-0.185],[-0.119,-0.042,0.061]],["Paris","英文",[0.546,-0.276,-0.058],[-0.549,0.047,0.665]],["France","英文",[0.679,-0.358,0.039],[0.343,0.13,-0.137]],["Tokyo","英文",[0.58,-0.394,-0.053],[0.297,0.125,0.277]],["Japan","英文",[0.629,-0.386,-0.011],[0.676,0.135,-0.237]],["cat","英文",[-0.041,0.593,-0.544],[0.412,-0.576,0.109]],["dog","英文",[-0.08,0.62,-0.624],[-0.467,-0.079,-0.09]],["seven","英文",[-0.885,-0.594,-0.134],[-0.597,0.481,0.489]],["eight","英文",[-0.854,-0.598,-0.181],[-0.579,0.428,0.079]]];
  function embed3d(box) {
    var COL = { "国家": "#007aff", "城市": "#34c759", "数字": "#f08c00", "动物": "#af52de", "颜色": "#ff3b30", "英文": "#8e8e93" };
    var ANALOGY = {
      "不画": [], "国家 → 城市（中国→北京、法国→巴黎、日本→东京）": [["中国", "北京"], ["法国", "巴黎"], ["日本", "东京"]],
      "man → woman 与 king → queen": [["man", "woman"], ["king", "queen"]], "数字的顺序（一 → 十）": [["一", "二"], ["二", "三"], ["三", "四"], ["四", "五"], ["五", "六"], ["六", "七"], ["七", "八"], ["八", "九"], ["九", "十"]]
    };
    box.innerHTML = '<div class="aw-title">词向量空间：Qwen3-0.6B 的真实嵌入投影到三维（拖动旋转）</div><div class="aw-grid">' +
      row("投影方式", select("proj", ["PCA 前三维（保留最多方差）", "随机挑三个方向"], "PCA 前三维（保留最多方差）"), true) +
      row("画出关系", select("ana", Object.keys(ANALOGY), "不画"), true) + row("标签", select("lab", ["每个词", "只标组名"], "每个词")) +
      '</div><svg class="aw-chart"></svg><div class="aw-out"></div>';
    var out = box.querySelector(".aw-out"), v = view3d(box.querySelector("svg"), { h: 420, scale: 190, ax: 0.35, ay: -0.5, dist: 12 });
    bind(box, function () {
      var pca = val(box, "proj").indexOf("PCA") === 0, k = pca ? 2 : 3, each = val(box, "lab") === "每个词", pos = {}, cent = {}, items = axes3(1.25, ["", "", ""]);
      var LP = [[6, 3.5, "start"], [-6, 3.5, "end"], [0, -6, "middle"], [0, 12, "middle"]];
      EMBED3D.forEach(function (e, idx) {
        pos[e[0]] = e[k];
        var lp = LP[idx % 4];
        if (e[1] === "英文") lp = LP[idx % 2];
        if (!cent[e[1]]) cent[e[1]] = [0, 0, 0, 0];
        cent[e[1]][0] += e[k][0]; cent[e[1]][1] += e[k][1]; cent[e[1]][2] += e[k][2]; cent[e[1]][3] += 1;
        items.push({ t: "pt", p: e[k], r: 3.5, fill: COL[e[1]], label: each ? e[0] : "", lx: lp[0], ly: lp[1], anchor: lp[2] });
      });
      if (!each) Object.keys(cent).forEach(function (g) {
        var c = cent[g]; items.push({ t: "pt", p: [c[0] / c[3], c[1] / c[3], c[2] / c[3]], r: 7, fill: COL[g], label: g, lcls: "aw-ptl aw-ptb", z: 50 });
      });
      ANALOGY[val(box, "ana")].forEach(function (pr) {
        if (pos[pr[0]] && pos[pr[1]]) items.push({ t: "arrow", a: pos[pr[0]], b: pos[pr[1]], cls: "aw-arr3", stroke: "#f08c00", sw: 2, z: 60 });
      });
      v.set(items);
      out.innerHTML = '<p>' + Object.keys(COL).map(function (g) { return '<span class="aw-leg" style="background:' + COL[g] + '"></span>' + g; }).join("　") + "</p>" +
        (pca ? '<p>PCA 把 1024 维压到 3 维，只保留了 24% 的方差，但同类的词已经聚在一起：数字一堆、动物一堆、颜色一堆；国家和城市几乎重叠——模型把它们都当"地名"。英文词分散在中间，和中文的对应词（king / 国王、cat / 猫）并不在同一个簇里，跨语言的对应要在完整的 1024 维里用余弦相似度才看得清。</p>' :
          '<p>随机挑三个方向看，什么结构都没有：每个方向上同类的词和异类的词混在一起。结构不是没有，而是分散在 1024 维里，PCA 的作用正是找到方差最大的几个方向。</p>') +
        '<p class="aw-note">在完整的 1024 维里：同组词的平均余弦相似度 0.28，异组 0.08；向量算术 king − man + woman 最近的词是 queen（余弦 0.58），北京 − 中国 + 法国 最近的是 巴黎（0.52），东京 − 日本 + 法国 也是 巴黎（0.57）。三维投影里这些箭头未必平行——丢掉的 76% 方差里有它们的大部分信息。</p>';
    });
  }

  // ---------------------------------------------------------------- RoPE：每一对维度随位置旋转，画成螺旋
  function ropeHelix(box) {
    var D = 128, L = 64;
    box.innerHTML = '<div class="aw-title">RoPE：第 i 对维度的 (cos mθ<sub>i</sub>, sin mθ<sub>i</sub>) 随位置 m 画成一条螺旋（拖动旋转）</div><div class="aw-grid">' +
      row("维度对 i（共 64 对）", range2("i", 2, 0, 63)) + row("base", select("base", ["10000", "1000000"], "10000")) +
      row("q 的位置 m", range2("m", 20, 0, 63)) + row("k 的位置 n", range2("n", 14, 0, 63)) + '</div><svg class="aw-chart"></svg><div class="aw-out"></div>';
    var out = box.querySelector(".aw-out"), v = view3d(box.querySelector("svg"), { h: 320, scale: 72, ax: 0.3, ay: -1.0, dist: 14 });
    bind(box, function () {
      var i = val(box, "i"), base = +val(box, "base"), m = val(box, "m"), n = val(box, "n");
      show(box, "i", String(i)); show(box, "m", String(m)); show(box, "n", String(n));
      var th = Math.pow(base, -2 * i / D), r = 0.95;
      function pos(p) { return [p / (L - 1) * 5.2 - 2.6, r * Math.sin(p * th), r * Math.cos(p * th)]; }   // 位置沿 x 轴；(cos, sin) 画在垂直于它的平面里
      var items = [{ t: "arrow", a: [-2.8, 0, 0], b: [3.0, 0, 0], cls: "aw-ax3", z: -99, hs: 6 }, { t: "text", p: [3.25, -0.05, 0], s: "位置 m", anchor: "start", z: 99 },
                   { t: "text", p: [-2.6, -1.2, 0], s: "0", z: 99 }, { t: "text", p: [2.6, -1.2, 0], s: String(L - 1), z: 99 }];
      var steps = 480, prev = null, s;
      for (s = 0; s <= steps; s++) { var p = pos(s / steps * (L - 1)); if (prev) items.push({ t: "seg", a: prev, b: p, cls: "aw-hel", sw: 1.6 }); prev = p; }
      [[m, "q", "#007aff"], [n, "k", "#f08c00"]].forEach(function (c) {
        var x = pos(c[0])[0], pv = null, t;
        for (t = 0; t <= 48; t++) { var q = [x, r * Math.sin(t / 48 * 2 * Math.PI), r * Math.cos(t / 48 * 2 * Math.PI)]; if (pv) items.push({ t: "seg", a: pv, b: q, cls: "aw-dash" }); pv = q; }
        items.push({ t: "arrow", a: [x, 0, 0], b: pos(c[0]), cls: "aw-arr3", stroke: c[2], sw: 2.4, z: 60 });
        items.push({ t: "pt", p: pos(c[0]), r: 4, fill: c[2], label: c[1] + "（" + (c[1] === "q" ? "m" : "n") + "=" + c[0] + "）", z: 61 });
      });
      v.set(items);
      var dl = (m - n) * th, wav = 2 * Math.PI / th;
      out.innerHTML = "<p>θ<sub>" + i + "</sub> = " + base + "<sup>−2·" + i + "/128</sup> = <b>" + th.toExponential(2) + "</b> 弧度/位置，转一整圈要 <b>" +
        (wav >= 1e5 ? wav.toExponential(2) : Math.round(wav).toLocaleString("zh-CN")) + "</b> 个位置（波长）。q 在 m = " + m + " 处转了 " + (m * th).toFixed(3) +
        " 弧度，k 在 n = " + n + " 处转了 " + (n * th).toFixed(3) + "，夹角 = (m − n)·θ<sub>" + i + "</sub> = <b>" + dl.toFixed(3) + "</b>；这一对维度对 q·k 的贡献是 |q||k|·cos(夹角) = |q||k| × <b>" +
        Math.cos(dl).toFixed(3) + "</b>——只和 m − n 有关，与绝对位置无关。</p>" +
        '<p class="aw-note">i 小的维度对转得快，像秒针，分辨近处的相对位置；i 大的几乎不转，像时针，区分远距离。128 维 = 64 根快慢不同的表针合在一起。把 base 调大，所有表针都变慢，能分辨更长的距离——这就是长上下文扩展里改 base 的原因。</p>';
    });
  }

  // ---------------------------------------------------------------- SwiGLU：门控激活的曲面
  function swiglu3d(box) {
    var FN = {
      "SwiGLU：SiLU(a) · b": [function (a, b) { return a / (1 + Math.exp(-a)) * b; }, "门是 SiLU(a)：a 很负时接近 0（关），a 很正时约等于 a（开，而且随 a 放大），中间平滑过渡；沿 b 方向是直线——b 只提供\"内容\"，开多大由 a 决定。负区不是硬 0，有一点负值，梯度能过。"],
      "GEGLU：GELU(a) · b": [function (a, b) { return gelu(a) * b; }, "GELU 和 SiLU 的形状几乎一样（GELU 的负区更浅），曲面也几乎一样；两者的差别在实践里很小，选哪个主要看 kernel 的实现。"],
      "ReGLU：ReLU(a) · b": [function (a, b) { return Math.max(0, a) * b; }, "a < 0 的半边被硬切成一个平面：输出恒为 0、梯度也为 0，这半边的神经元在这些输入上学不到东西（\"死神经元\"）。a > 0 的半边和 SwiGLU 一样是双线性的。"],
      "双线性：a · b（门不加非线性）": [function (a, b) { return a * b; }, "一个马鞍面：没有\"关\"的区域，a 和 b 完全对称，谁是门谁是内容分不出来。门之所以要加非线性，就是为了造出\"关\"和\"开\"的两种状态。"],
      "不带门的 MLP：GELU(a)，与 b 无关": [function (a, b) { return gelu(a) * 1.6; }, "经典两层 MLP 的激活只看一个输入：曲面沿 b 方向完全是平的，表达能力少了一个乘法。GLU 系列在同样参数量下效果更好，差别就在这个乘法上。"]
    };
    box.innerHTML = '<div class="aw-title">门控激活的曲面：输出 = 门(a) × 内容 b（拖动旋转）</div><div class="aw-grid">' +
      row("函数", select("fn", Object.keys(FN), "SwiGLU：SiLU(a) · b"), true) + '</div><svg class="aw-chart"></svg><div class="aw-out"></div>';
    var out = box.querySelector(".aw-out"), v = view3d(box.querySelector("svg"), { h: 340, scale: 50, ax: 0.5, ay: -0.6, dist: 10 });
    bind(box, function () {
      var f = FN[val(box, "fn")];
      var items = axes3(3.4, ["a = x·W_gate", "", "b = x·W_up"]);
      items.push({ t: "text", p: [0, 2.75, 0], s: "输出", z: 99 });
      items = items.concat(surface3(function (a, b) { return Math.max(-2.4, Math.min(2.4, f[0](a, b) / 3.2)); }, -3, 3, -3, 3, 20, function (y) { return heat((y + 2.4) / 4.8); }));
      v.set(items);
      out.innerHTML = "<p>" + f[1] + "</p>" + '<p class="aw-note">a、b 都是同一个输入 x 经过不同矩阵得到的（a = x·W<sub>gate</sub>，b = x·W<sub>up</sub>），所以这个曲面描述的是中间层每一个神经元的行为；d<sub>ff</sub> 个这样的神经元并排，再被 W<sub>down</sub> 加权求和。高度按 1/3.2 缩放并截断在 ±2.4，颜色也表示高度。</p>';
    });
  }

  // ---------------------------------------------------------------- Scaling Law：损失曲面与等算力线
  function scaling3d(box) {
    // L(N, D) = E + A / N^α + B / D^β，系数取自对 Chinchilla 数据的复现拟合（Besiroglu 等，2024）；它给出的最优 D/N 约 20
    var E = 1.8172, A = 482.01, B = 2085.43, al = 0.3478, be = 0.3658;
    function loss(lN, lD) { return E + A / Math.pow(10, lN * al) + B / Math.pow(10, lD * be); }
    var MODELS3 = [["GPT-3", 11.24, 11.48], ["Chinchilla", 10.85, 12.15], ["LLaMA-3-8B", 9.9, 13.18], ["Qwen3-0.6B", 8.78, 13.56], ["DeepSeek-V3", 11.83, 13.17]];
    box.innerHTML = '<div class="aw-title">Scaling Law 的损失曲面：L(N, D) = E + A/N<sup>α</sup> + B/D<sup>β</sup>，橙线是算力预算 C = 6ND 下所有的 (N, D) 组合（拖动旋转）</div><div class="aw-grid">' +
      row("算力预算 C", range2("c", 210, 170, 260)) + row("GPU", select("gpu", Object.keys(GPUS), "H100 SXM")) + row("MFU", range2("mfu", 40, 10, 70)) +
      '</div><svg class="aw-chart"></svg><div class="aw-out"></div>';
    var out = box.querySelector(".aw-out"), v = view3d(box.querySelector("svg"), { h: 380, scale: 60, ax: 0.45, ay: -0.75, dist: 11 });
    function X(lN) { return (lN - 9.5) * 0.9; }
    function Z(lD) { return (lD - 11.5) * 0.9; }
    function Y(l) { return (l - 3.6) * 0.6; }
    bind(box, function () {
      var lC = val(box, "c") / 10, g = GPUS[val(box, "gpu")], mfu = val(box, "mfu") / 100;
      show(box, "c", "10^" + lC.toFixed(1)); show(box, "mfu", mfu.toFixed(2));
      var b0 = -1.25, items = [                                            // 底部的坐标架贴着曲面的两条前缘：N 沿左前缘、D 沿右前缘，损失竖在前角
        { t: "arrow", a: [X(7), b0, Z(14)], b: [X(12) + 0.25, b0, Z(14)], cls: "aw-ax3", z: -99, hs: 6 }, { t: "text", p: [X(9.5), b0 - 0.32, Z(14) + 0.1], s: "参数量 N →", z: 99 },
        { t: "arrow", a: [X(12), b0, Z(9)], b: [X(12), b0, Z(14) + 0.25], cls: "aw-ax3", z: -99, hs: 6 }, { t: "text", p: [X(12) + 0.1, b0 - 0.32, Z(11.5)], s: "数据量 D →", z: 99 },
        { t: "text", p: [X(7) - 0.15, b0 - 0.3, Z(14) + 0.1], s: "10⁷", z: 99 }, { t: "text", p: [X(12) + 0.45, b0 + 0.05, Z(14) + 0.15], s: "10¹² · 10¹⁴", z: 99 }, { t: "text", p: [X(12) + 0.15, b0 - 0.3, Z(9) - 0.1], s: "10⁹", z: 99 }];
      items = items.concat(surface3(function (x, z) { return Y(loss(x / 0.9 + 9.5, z / 0.9 + 11.5)); }, X(7), X(12), Z(9), Z(14), 20, function (y) { return heat((y + 1.1) / 1.9); }));
      // 等算力线：log D = log C − log 6 − log N
      var prev = null, best = null, t;
      for (t = 0; t <= 100; t++) {
        var lN = 7 + 5 * t / 100, lD = lC - Math.log10(6) - lN;
        if (lD < 9 || lD > 14) { prev = null; continue; }
        var l = loss(lN, lD), p = [X(lN), Y(l) + 0.04, Z(lD)];
        if (prev) items.push({ t: "seg", a: prev, b: p, cls: "aw-arr3", stroke: "#f08c00", sw: 2.6, z: 30 });
        if (!best || l < best[2]) best = [lN, lD, l];
        prev = p;
      }
      // 解析最优：N* = G·(C/6)^a，a = β/(α+β)
      var a = be / (al + be), G = Math.pow(al * A / (be * B), 1 / (al + be)), Nopt = G * Math.pow(Math.pow(10, lC) / 6, a), Dopt = Math.pow(10, lC) / 6 / Nopt;
      var lNo = Math.log10(Nopt), lDo = Math.log10(Dopt), inGrid = lNo >= 7 && lNo <= 12 && lDo >= 9 && lDo <= 14;
      if (inGrid) items.push({ t: "pt", p: [X(lNo), Y(loss(lNo, lDo)) + 0.06, Z(lDo)], r: 5, fill: "#ff3b30", label: "最优（最低点）", z: 70 });
      MODELS3.forEach(function (mdl, j) { items.push({ t: "pt", p: [X(mdl[1]), Y(loss(mdl[1], mdl[2])) + 0.05, Z(mdl[2])], r: 3.5, fill: "#8e8e93", label: mdl[0], z: 65, ly: j % 2 ? 12 : -6, lx: 0, anchor: "middle" }); });
      v.set(items);
      var days = Math.pow(10, lC) / (g[2] * 1e12 * mfu) / 86400;
      function big(x) { return x >= 1e12 ? (x / 1e12).toFixed(1) + " T" : x >= 1e9 ? (x / 1e9).toFixed(1) + " B" : (x / 1e6).toFixed(0) + " M"; }
      out.innerHTML = "<p>预算 C = 10<sup>" + lC.toFixed(1) + "</sup> FLOP 时最优的分配：参数量 <b>" + big(Nopt) + "</b>，数据量 <b>" + big(Dopt) + "</b> token，D/N ≈ <b>" + (Dopt / Nopt).toFixed(0) +
        "</b>，预测损失 <b>" + loss(lNo, lDo).toFixed(3) + "</b>" + (inGrid ? "" : "（最优点在画面之外）") + "。用 " + val(box, "gpu") + " 按 MFU " + Math.round(mfu * 100) + "% 跑，需要约 <b>" +
        (days >= 1 ? Math.round(days).toLocaleString("zh-CN") + " 卡·天" : (days * 24).toFixed(1) + " 卡·小时") + "</b>。</p>" +
        '<p class="aw-note">高度和颜色都是损失（蓝低、橙高）。沿橙色的等算力线走：左边模型太小、数据太多，右边模型太大、每个参数见的数据太少，损失都更高，最低点就是 Chinchilla 最优。预算每涨 10 倍，N 和 D 各涨约 3 倍（指数 0.51 / 0.49）。灰点标出几个真实模型在 (N, D) 平面上的位置（高度取的是拟合曲面，不是它们的真实损失）：现代模型都在最优线的"数据多"一侧——为了推理便宜而故意过度训练，LLaMA-3-8B 的 D/N 接近 1900。</p>';
    });
  }

  // ---------------------------------------------------------------- 语言模型：每个位置的条件概率与交叉熵（真实模型的数据）
  // 数据：Qwen3-0.6B（fp32）对例句的输出，每个位置的前 6 个候选及概率、真实下一个 token 的概率和损失，与正文里的数字一致
  var NEXTWORD = {"tokens":["北京","是中国","的","首都","，","也是","全国","的政治","和","文化","中心","。"],"rows":[{"ctx":"北京","next":"是中国","p_next":0.0,"loss":17.86,"top":[["Question",0.047],[" Question",0.01],[" Answer",0.005],[" Name",0.004],["摘要",0.003],["글",0.003]]},{"ctx":"是中国","next":"的","p_next":0.1938,"loss":1.64,"top":[["的",0.194],["重要的",0.068],["古代",0.054],["最大的",0.036],["最早",0.03],["著名的",0.023]]},{"ctx":"的","next":"首都","p_next":0.1458,"loss":1.93,"top":[["首都",0.146],["____",0.058],["首",0.054],["直辖市",0.031],["第二",0.024],["第",0.023]]},{"ctx":"首都","next":"，","p_next":0.7463,"loss":0.29,"top":[["，",0.746],[",",0.131],["城市",0.036],["。",0.031],["，并",0.014],["和",0.006]]},{"ctx":"，","next":"也是","p_next":0.0951,"loss":2.35,"top":[["也是",0.095],["位于",0.09],["是",0.063],["拥有",0.058],["中国",0.045],["它",0.026]]},{"ctx":"也是","next":"全国","p_next":0.01,"loss":4.6,"top":[["世界",0.203],["中国",0.197],["世界上",0.125],["亚洲",0.092],["中国的",0.083],["我国",0.051]]},{"ctx":"全国","next":"的政治","p_next":0.0401,"loss":3.22,"top":[["最大的",0.251],["的",0.184],["重要的",0.149],["的政治",0.04],["的重要",0.034],["政治",0.031]]},{"ctx":"的政治","next":"和","p_next":0.0058,"loss":5.15,"top":[["、",0.898],["中心",0.078],["和",0.006],["经济",0.004],["文化",0.002],["，",0.002]]},{"ctx":"和","next":"文化","p_next":0.0422,"loss":3.16,"top":[["经济",0.914],["文化",0.042],["军事",0.036],["行政",0.002],["经济发展",0.002],[" economic",0.001]]},{"ctx":"文化","next":"中心","p_next":0.9953,"loss":0.0,"top":[["中心",0.995],["核心",0.001],["活动",0.0],["的核心",0.0],["生活",0.0],["、",0.0]]},{"ctx":"中心","next":"。","p_next":0.5901,"loss":0.53,"top":[["。",0.59],["，",0.358],["之一",0.014],["。\n",0.007],["。\n\n",0.005],[".",0.004]]}]};
  function nextword(box) {
    var rows = NEXTWORD.rows, toks = NEXTWORD.tokens;
    box.innerHTML = '<div class="aw-title">语言模型 = 条件概率函数：看过前面的 token 之后，下一个是什么</div><div class="aw-grid">' +
      row("已看到的位置 t", range2("t", 3, 1, rows.length), true) + '</div><div class="aw-toks"></div><svg class="aw-chart" viewBox="0 0 560 190"></svg><div class="aw-out"></div>';
    var tk = box.querySelector(".aw-toks"), svg = box.querySelector("svg"), out = box.querySelector(".aw-out");
    function esc(s) { return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/ /g, "␣").replace(/\n/g, "⏎"); }
    bind(box, function () {
      var t = val(box, "t"), r = rows[t - 1];
      show(box, "t", String(t));
      tk.innerHTML = toks.map(function (w, i) {
        return '<span class="aw-tok' + (i < t ? " aw-tok-seen" : i === t ? " aw-tok-next" : "") + '">' + esc(w) + "</span>";
      }).join("") + '<span class="aw-note">　（蓝：已看到；橙：真实的下一个 token）</span>';
      var S = "", inTop = false, i;
      var bars = r.top.slice();
      for (i = 0; i < bars.length; i++) if (bars[i][0] === r.next) inTop = true;
      if (!inTop) bars.push([r.next, r.p_next]);
      S += svgText(10, 16, "P(下一个 token | 前 " + t + " 个 token)，整个词表 151936 个候选里概率最高的几个：", "start");
      var pmax = 0;
      for (i = 0; i < bars.length; i++) pmax = Math.max(pmax, bars[i][1]);
      for (i = 0; i < bars.length; i++) {
        var y = 30 + i * 22, p = bars[i][1], w = Math.max(1.5, p / Math.max(0.25, pmax) * 300), truth = bars[i][0] === r.next;
        S += '<rect x="150" y="' + y + '" width="' + w.toFixed(1) + '" height="15" rx="3" class="' + (truth ? "aw-b" : "aw-f") + '"/>';
        S += svgText(144, y + 11.5, esc(bars[i][0]), "end");
        S += svgText(155 + w, y + 11.5, (p < 0.001 ? p.toExponential(1) : p.toFixed(3)) + (truth ? "  ← 真实的下一个" : ""), "start");
      }
      svg.setAttribute("viewBox", "0 0 560 " + (36 + bars.length * 22));
      svg.innerHTML = S;
      var sum = 0, j;
      for (j = 0; j < t; j++) sum += rows[j].loss;
      var allAvg = 0; for (j = 0; j < rows.length; j++) allAvg += rows[j].loss; allAvg /= rows.length;
      out.innerHTML = "<p>真实的下一个 token 是「" + esc(r.next) + "」，模型给它的概率 <b>" + (r.p_next < 0.001 ? r.p_next.toExponential(1) : r.p_next.toFixed(3)) +
        "</b>，这个位置的损失 −log P = <b>" + r.loss.toFixed(2) + "</b>" + (r.loss < 0.5 ? "（几乎没有悬念）" : r.loss > 8 ? "（完全没料到）" : r.loss > 4 ? "（很意外）" : "") +
        "。前 " + t + " 个位置的平均损失 " + (sum / t).toFixed(2) + "，困惑度 e<sup>" + (sum / t).toFixed(2) + "</sup> = <b>" + Math.exp(sum / t).toFixed(1) +
        "</b>；整句 " + rows.length + " 个位置平均 " + allAvg.toFixed(2) + "，困惑度 " + Math.exp(allAvg).toFixed(1) + "。</p>" +
        '<p class="aw-note">训练时这 ' + rows.length + ' 个位置的分布是一次前向同时算出来的（teacher forcing），损失是它们的平均；推理时只用最后一个位置的分布，从里面挑一个 token 接到后面再算下一个。' +
        '困惑度对个别意外的 token 很敏感：第 1 个位置的 17.86 一项就把整句的平均从 2.3 拉到了 3.7。</p>';
    });
  }

  // ---------------------------------------------------------------- 广播：两个形状怎么对齐
  function broadcast(box) {
    var PRESETS = {
      "[B, T, d] × [d]：RMSNorm 的权重乘到每个位置": ["2, 5, 896", "896"],
      "[B, T, d] / [B, T, 1]：每个位置除以自己的均方根": ["2, 5, 896", "2, 5, 1"],
      "[T, 1] 与 [1, T]：外积，造因果掩码": ["5, 1", "1, 5"],
      "[B, H, T, T] + [1, 1, T, T]：注意力分数加掩码": ["2, 8, 5, 5", "1, 1, 5, 5"],
      "[B, T, d] + [T, d]：位置编码加到每个 batch": ["2, 5, 896", "5, 896"],
      "[2, 3] 与 [3, 2]：对不上": ["2, 3", "3, 2"]
    };
    box.innerHTML = '<div class="aw-title">广播：从最后一维开始对齐，1 可以拉伸，其他必须相等</div><div class="aw-grid">' +
      row("常见情形", select("preset", Object.keys(PRESETS), Object.keys(PRESETS)[0]), true) +
      row("形状 A", '<input type="text" data-k="a" value="2, 5, 896" spellcheck="false">') + row("形状 B", '<input type="text" data-k="b" value="896" spellcheck="false">') +
      '</div><svg class="aw-chart" viewBox="0 0 560 120"></svg><div class="aw-out"></div>';
    var svg = box.querySelector("svg"), out = box.querySelector(".aw-out");
    function parse(s) { return s.split(/[,，\s\[\]]+/).filter(function (x) { return x !== ""; }).map(function (x) { return +x; }); }
    box.querySelector('[data-k="preset"]').addEventListener("change", function () {
      var pr = PRESETS[this.value]; input(box, "a").value = pr[0]; input(box, "b").value = pr[1]; draw();
    });
    function draw() {
      var A = parse(input(box, "a").value), B = parse(input(box, "b").value);
      if (!A.length || !B.length || A.concat(B).some(function (x) { return !(x >= 1) || x % 1; })) { out.innerHTML = "<p>形状要写成逗号分隔的正整数。</p>"; svg.innerHTML = ""; return; }
      var n = Math.max(A.length, B.length), rowsA = [], rowsB = [], R = [], bad = -1, i;
      for (i = 0; i < n; i++) {                                   // 从右往左对齐；短的在左边补 1
        var a = i < n - A.length ? null : A[i - (n - A.length)], b = i < n - B.length ? null : B[i - (n - B.length)];
        var ea = a == null ? 1 : a, eb = b == null ? 1 : b;
        rowsA.push(a); rowsB.push(b);
        if (ea === eb) R.push(ea); else if (ea === 1) R.push(eb); else if (eb === 1) R.push(ea); else { R.push(null); bad = i; }
      }
      var S = "", x0 = 90, cw = Math.min(90, 440 / n);
      function line(y, label, vals, other) {
        S += svgText(x0 - 10, y + 15, label, "end");
        for (var k = 0; k < n; k++) {
          var v = vals[k], x = x0 + k * cw, ov = other[k], stretched = v === 1 && ov !== null && ov !== 1 || v === null;
          S += '<rect x="' + x + '" y="' + y + '" width="' + (cw - 6) + '" height="22" rx="4" class="' + (k === bad ? "aw-b" : stretched ? "aw-pad" : "aw-f") + '"/>';
          S += svgText(x + (cw - 6) / 2, y + 15, v === null ? "(1)" : v, "middle");
        }
      }
      line(8, "A", rowsA, rowsB); line(40, "B", rowsB, rowsA);
      S += svgText(x0 - 10, 90, "结果", "end");
      for (i = 0; i < n; i++) {
        var x = x0 + i * cw;
        S += '<rect x="' + x + '" y="75" width="' + (cw - 6) + '" height="22" rx="4" class="' + (R[i] === null ? "aw-b" : "aw-on") + '"/>';
        S += svgText(x + (cw - 6) / 2, 90, R[i] === null ? "✗" : R[i], "middle");
      }
      S += svgText(x0 + n * cw + 4, 23, "← 右对齐", "start");
      svg.innerHTML = S;
      if (bad >= 0) out.innerHTML = "<p><b>不能广播</b>：第 " + (bad + 1) + " 维一个是 " + (rowsA[bad] == null ? 1 : rowsA[bad]) + "、一个是 " + (rowsB[bad] == null ? 1 : rowsB[bad]) + "，既不相等又没有 1。PyTorch 会报 <code>The size of tensor a must match the size of tensor b</code>。要么 <code>unsqueeze</code> 一个大小为 1 的维度，要么 <code>transpose</code>。</p>";
      else {
        var copies = 1, copiesB = 1;
        for (i = 0; i < n; i++) { if ((rowsA[i] == null ? 1 : rowsA[i]) === 1 && R[i] !== 1) copies *= R[i]; if ((rowsB[i] == null ? 1 : rowsB[i]) === 1 && R[i] !== 1) copiesB *= R[i]; }
        function cp(name, c) { return c === 1 ? name + " 不用复制" : name + " 被复制了 " + c + " 份"; }
        out.innerHTML = "<p>结果形状 <b>[" + R.join(", ") + "]</b>。虚线框是被\"拉伸\"的维度：" + cp("A", copies) + "、" + cp("B", copiesB) + "——只是逻辑上的复制，PyTorch 用步长为 0 的视图实现，不占额外显存。</p>" +
          '<p class="aw-note">规则只有两条：从最后一维往前对齐，缺的维度当作 1；每一维要么相等，要么其中一个是 1。<code>keepdim=True</code> 的作用就是保住那个 1，让 [B, T, 1] 能直接和 [B, T, d] 运算。</p>';
      }
    }
    bind(box, draw);
  }

  // ---------------------------------------------------------------- BPE：一步一步合并
  function bpe(box) {
    var CORPUS = {
      "low lower lowest newer newest wider widest": "low low low low low lower lower lowest newer newer newer newest wider widest",
      "hug pug pun bun hugs（Hugging Face 教程里的例子）": "hug hug hug hug hug hug hug hug hug hug pug pug pug pug pug pun pun pun pun pun pun pun pun pun pun pun pun bun bun bun bun hugs hugs hugs hugs hugs",
      "推理 推理引擎 推理服务 引擎 服务 调度 调度器": "推理 推理 推理 推理引擎 推理引擎 推理服务 推理服务 推理服务 引擎 引擎 服务 服务 服务 调度 调度 调度器 调度器"
    };
    box.innerHTML = '<div class="aw-title">BPE：统计最常见的相邻对，合并，再统计，再合并</div><div class="aw-grid">' +
      row("语料", select("corpus", Object.keys(CORPUS), Object.keys(CORPUS)[0]), true) + row("合并次数", range2("steps", 3, 0, 12)) +
      row("再编码新文本", '<input type="text" data-k="probe" value="lowest newest" spellcheck="false">', true) + '</div><div class="aw-out"></div>';
    var out = box.querySelector(".aw-out");
    function esc(s) { return s.replace(/&/g, "&amp;").replace(/</g, "&lt;"); }
    function chips(seq, merged) {
      return seq.map(function (t) { return '<span class="aw-tok' + (t.length > 1 ? " aw-tok-seen" : "") + '">' + esc(t) + "</span>"; }).join("");
    }
    function train(text, steps) {
      // 按词切开，词尾加 ▁ 标记（Sennrich 的字符级 BPE；真实实现是字节级，这里为了看得清按字符合并）
      var words = {}, seqs = {}, merges = [], log = [];
      text.split(/\s+/).forEach(function (w) { if (!w) return; words[w] = (words[w] || 0) + 1; seqs[w] = Array.from(w).concat(["▁"]); });
      for (var s = 0; s < steps; s++) {
        var counts = {}, best = null;
        Object.keys(words).forEach(function (w) {
          var q = seqs[w];
          for (var i = 0; i + 1 < q.length; i++) { var key = q[i] + "\u0001" + q[i + 1]; counts[key] = (counts[key] || 0) + words[w]; }
        });
        Object.keys(counts).forEach(function (k) { if (!best || counts[k] > counts[best]) best = k; });
        if (!best || counts[best] < 2) break;
        var pair = best.split("\u0001"), top = Object.keys(counts).sort(function (a, b) { return counts[b] - counts[a]; }).slice(0, 4);
        log.push({ pair: pair, n: counts[best], top: top.map(function (k) { return [k.split("\u0001"), counts[k]]; }) });
        merges.push(pair);
        Object.keys(words).forEach(function (w) { seqs[w] = applyMerge(seqs[w], pair); });
      }
      return { words: words, seqs: seqs, merges: merges, log: log };
    }
    function applyMerge(q, pair) {
      var o = [], i = 0;
      while (i < q.length) { if (i + 1 < q.length && q[i] === pair[0] && q[i + 1] === pair[1]) { o.push(q[i] + q[i + 1]); i += 2; } else { o.push(q[i]); i += 1; } }
      return o;
    }
    function encode(text, merges) {
      var ids = [];
      text.split(/\s+/).forEach(function (w) {
        if (!w) return;
        var q = Array.from(w).concat(["▁"]);
        merges.forEach(function (pair) { q = applyMerge(q, pair); });     // 按学到的先后顺序应用
        ids = ids.concat(q);
      });
      return ids;
    }
    bind(box, function () {
      var steps = val(box, "steps"), text = CORPUS[val(box, "corpus")], r = train(text, steps);
      show(box, "steps", String(steps));
      var vocab = {};
      Object.keys(r.seqs).forEach(function (w) { r.seqs[w].forEach(function (t) { vocab[t] = 1; }); });
      var html = "<p>语料里的词（频次）：" + Object.keys(r.words).map(function (w) { return esc(w) + " ×" + r.words[w]; }).join("，") + "</p>";
      html += "<p>合并 " + r.merges.length + " 次之后语料被切成：</p><p>" + Object.keys(r.seqs).map(function (w) { return chips(r.seqs[w]); }).join(" ") + "</p>";
      if (r.log.length) {
        var last = r.log[r.log.length - 1];
        html += "<p>第 " + r.log.length + " 次合并前出现最多的相邻对：" + last.top.map(function (e) { return "「" + esc(e[0][0]) + "」+「" + esc(e[0][1]) + "」×" + e[1]; }).join("，") +
          "，于是把「" + esc(last.pair[0]) + "」「" + esc(last.pair[1]) + "」合成新 token「<b>" + esc(last.pair.join("")) + "</b>」。</p>";
        html += "<p>合并规则表（编号越小越先学到、编码时也越先用）：" + r.merges.map(function (m, i) { return (i + 1) + ". " + esc(m[0]) + "+" + esc(m[1]); }).join("　") + "</p>";
      } else html += '<p class="aw-note">还没合并：每个字符（加上词尾标记 ▁）各是一个 token。</p>';
      html += "<p>词表大小：字符 + " + r.merges.length + " 条合并 = <b>" + Object.keys(vocab).length + "</b>；语料共 " + Object.keys(r.seqs).reduce(function (n, w) { return n + r.seqs[w].length * r.words[w]; }, 0) + " 个 token。</p>";
      var probe = input(box, "probe").value;
      if (probe.trim()) html += "<p>用这些规则编码「" + esc(probe) + "」：" + chips(encode(probe, r.merges)) + '　<span class="aw-note">（没见过的字符仍能表示——真实的字节级 BPE 从 256 个字节起步，所以永远没有未知词）</span></p>';
      out.innerHTML = html;
    });
  }

  // ---------------------------------------------------------------- 浮点数：拆成符号、指数、尾数
  function floatBits(box) {
    var FMT = { "FP32": [8, 23, 127, true], "FP16": [5, 10, 15, true], "BF16": [8, 7, 127, true], "FP8 E4M3": [4, 3, 7, false], "FP8 E5M2": [5, 2, 15, true] };   // 指数位、尾数位、偏置、有无 inf
    box.innerHTML = '<div class="aw-title">浮点数的表示：x = (−1)<sup>s</sup> × 1.m × 2<sup>e − bias</sup></div><div class="aw-grid">' +
      row("格式", select("fmt", Object.keys(FMT), "BF16")) + row("数值", '<input type="text" data-k="x" value="3.14159" spellcheck="false">') +
      '</div><svg class="aw-chart" viewBox="0 0 560 70"></svg><div class="aw-out"></div>';
    var svg = box.querySelector("svg"), out = box.querySelector(".aw-out");
    function encode(x, eb, mb, bias, hasInf) {                   // 返回 {s, e, m, value, kind}，就近舍入到偶数
      var maxE = (1 << eb) - 1, s = x < 0 || (x === 0 && 1 / x < 0) ? 1 : 0, a = Math.abs(x);
      var emax = hasInf ? maxE - 1 : maxE;                         // E4M3（fn）没有 inf，最大指数也用来表示有限数
      var maxVal = hasInf ? (2 - Math.pow(2, -mb)) * Math.pow(2, emax - bias) : (2 - Math.pow(2, -mb) * 2) * Math.pow(2, emax - bias);   // E4M3 的最大值 448 = 1.75 × 2^8
      if (!isFinite(a) || isNaN(x)) return { s: s, e: maxE, m: hasInf ? 0 : (1 << mb) - 1, value: NaN, kind: "NaN" };
      if (a === 0) return { s: s, e: 0, m: 0, value: 0, kind: "零" };
      var e = Math.floor(Math.log2(a)), E = e + bias;
      if (E <= 0) {                                                // 次正规数：指数固定为 1 − bias，没有隐含的 1
        var scale = Math.pow(2, mb - (1 - bias)), m0 = Math.round(a * scale);
        if (m0 >= (1 << mb)) return { s: s, e: 1, m: 0, value: Math.pow(2, 1 - bias) * (s ? -1 : 1), kind: "最小正规数" };
        return { s: s, e: 0, m: m0, value: m0 / scale * (s ? -1 : 1), kind: m0 ? "次正规数" : "下溢成 0" };
      }
      var frac = a / Math.pow(2, e) - 1, mq = frac * (1 << mb), m = Math.round(mq);
      if (Math.abs(mq - Math.floor(mq) - 0.5) < 1e-12) m = Math.floor(mq) % 2 === 0 ? Math.floor(mq) : Math.floor(mq) + 1;
      if (m === (1 << mb)) { m = 0; e += 1; E += 1; }
      if (E > emax) return { s: s, e: maxE, m: hasInf ? 0 : (1 << mb) - 1, value: hasInf ? Infinity * (s ? -1 : 1) : NaN, kind: hasInf ? "溢出成 inf" : "溢出成 NaN（E4M3 没有 inf）", maxVal: maxVal };
      return { s: s, e: E, m: m, value: (1 + m / (1 << mb)) * Math.pow(2, e) * (s ? -1 : 1), kind: "正规数", maxVal: maxVal };
    }
    function bits(v, n) { var s = v.toString(2); while (s.length < n) s = "0" + s; return s; }
    bind(box, function () {
      var f = FMT[val(box, "fmt")], eb = f[0], mb = f[1], bias = f[2], x = parseFloat(input(box, "x").value.replace(/[，]/g, ""));
      if (isNaN(x)) { out.innerHTML = "<p>请输入一个数。</p>"; svg.innerHTML = ""; return; }
      var r = encode(x, eb, mb, bias, f[3]), n = 1 + eb + mb, w = Math.min(16, 520 / n), x0 = 20, S = "";
      var fields = [[1, r.s, "aw-b", "符号"], [eb, r.e, "aw-f", "指数（" + eb + " 位）"], [mb, r.m, "aw-on", "尾数（" + mb + " 位）"]], pos = 0;
      fields.forEach(function (fd) {
        var str = bits(fd[1], fd[0]);
        for (var i = 0; i < fd[0]; i++) {
          var xx = x0 + (pos + i) * w;
          S += '<rect x="' + xx + '" y="18" width="' + (w - 1.5) + '" height="22" class="' + fd[2] + '"/>';
          if (w >= 9) S += svgText(xx + w / 2 - 0.75, 33, str[i], "middle");
        }
        S += svgText(x0 + (pos + fd[0] / 2) * w, 58, fd[3], "middle");
        pos += fd[0];
      });
      svg.innerHTML = S;
      var ulp = r.kind === "正规数" ? Math.pow(2, r.e - bias - mb) : Math.pow(2, 1 - bias - mb);
      var html = "<p>存成 <b>" + val(box, "fmt") + "</b>：" + r.kind + "，实际存下的值是 <b>" + (isFinite(r.value) ? r.value.toPrecision(mb >= 10 ? 9 : 6) : String(r.value)) + "</b>";
      if (isFinite(r.value) && r.value !== 0 && x !== 0) html += "，和输入差 " + Math.abs(r.value - x).toExponential(2) + "（相对误差 " + (Math.abs(r.value - x) / Math.abs(x) * 100).toFixed(3) + "%）";
      html += "。</p>";
      if (r.kind === "正规数") html += "<p>指数字段 " + r.e + " − 偏置 " + bias + " = 2<sup>" + (r.e - bias) + "</sup>，尾数 1 + " + r.m + "/" + (1 << mb) + " = " + (1 + r.m / (1 << mb)).toFixed(Math.min(8, mb + 1)) +
        "。这个量级上相邻两个可表示数的间隔（ulp）是 2<sup>" + (r.e - bias) + "</sup> × 2<sup>−" + mb + "</sup> = <b>" + ulp.toExponential(2) + "</b>：数越大间隔越大，相对精度不变（eps = 2<sup>−" + mb + "</sup> = " + Math.pow(2, -mb).toExponential(2) + "）。</p>";
      if (r.maxVal) html += '<p class="aw-note">这种格式最大能表示 ' + r.maxVal.toPrecision(4) + (f[3] ? "，再大就是 inf" : "，再大直接变成 NaN（E4M3 把本该给 inf 的编码也拿来表示有限数，换到了 448 这个上限）") + "；试试 70000（FP16 溢出）、0.0001（FP8 下溢）、1.001（BF16 存不下这么小的差别）。</p>";
      out.innerHTML = html;
    });
  }

  // ---------------------------------------------------------------- 注意力：拖动 query，看权重和加权平均怎么变
  function attention2d(box) {
    var KEYS = [["猫", -2.2, 1.6], ["狗", -1.5, 2.1], ["老虎", -2.7, 0.6], ["北京", 2.1, 1.5], ["上海", 2.6, 0.6], ["三", 1.1, -2.0], ["四", 1.9, -1.5], ["红色", -1.4, -1.9]];
    box.innerHTML = '<div class="aw-title">注意力 = 软查表：query 和每个 key 的点积是分数，softmax 成权重，再按权重混合 value（拖动橙色的 query）</div><div class="aw-grid">' +
      row("分数的放大倍数", range2("scale", 10, 1, 40), true) + '</div><svg class="aw-chart" viewBox="0 0 560 300"></svg><div class="aw-out"></div>';
    var svg = box.querySelector("svg"), out = box.querySelector(".aw-out"), q = [0.8, 0.9], drag = false;
    var cx = 185, cy = 150, u = 46;                                 // 左边画平面，右边画权重条
    function P(x, y) { return [cx + x * u, cy - y * u]; }
    function draw() {
      var s = val(box, "scale") / 10; show(box, "scale", s.toFixed(1));
      var scores = KEYS.map(function (k) { return (q[0] * k[1] + q[1] * k[2]) * s; });
      var m = Math.max.apply(null, scores), ex = scores.map(function (v) { return Math.exp(v - m); }), Z = ex.reduce(function (a, b) { return a + b; }, 0);
      var w = ex.map(function (v) { return v / Z; }), ox = 0, oy = 0, i;
      for (i = 0; i < KEYS.length; i++) { ox += w[i] * KEYS[i][1]; oy += w[i] * KEYS[i][2]; }
      var S = '<line x1="20" y1="' + cy + '" x2="350" y2="' + cy + '" class="aw-gl"/><line x1="' + cx + '" y1="10" x2="' + cx + '" y2="290" class="aw-gl"/>', qq = P(q[0], q[1]);
      for (i = 0; i < KEYS.length; i++) {
        var p = P(KEYS[i][1], KEYS[i][2]);
        S += '<line x1="' + qq[0].toFixed(1) + '" y1="' + qq[1].toFixed(1) + '" x2="' + p[0] + '" y2="' + p[1] + '" stroke="#007aff" stroke-opacity="' + (0.12 + 0.88 * w[i]).toFixed(2) + '" stroke-width="' + (0.6 + 6 * w[i]).toFixed(1) + '"/>';
      }
      for (i = 0; i < KEYS.length; i++) {
        var pk = P(KEYS[i][1], KEYS[i][2]);
        S += '<circle cx="' + pk[0] + '" cy="' + pk[1] + '" r="' + (4 + 10 * w[i]).toFixed(1) + '" class="aw-idot" fill-opacity="0.85"/>' + svgText(pk[0] + 9, pk[1] - 8, KEYS[i][0], "start");
      }
      var o = P(ox, oy);
      S += '<rect x="' + (o[0] - 6).toFixed(1) + '" y="' + (o[1] - 6).toFixed(1) + '" width="12" height="12" transform="rotate(45 ' + o[0].toFixed(1) + ' ' + o[1].toFixed(1) + ')" fill="#34c759"/>' +
        svgText(o[0], o[1] + 22, "输出 = 加权平均", "middle");
      S += '<circle cx="' + qq[0].toFixed(1) + '" cy="' + qq[1].toFixed(1) + '" r="8" class="aw-dot"/>' + svgText(qq[0] + 11, qq[1] + 4, "query", "start");
      S += svgText(385, 18, "softmax 权重", "start");
      for (i = 0; i < KEYS.length; i++) {
        var y = 30 + i * 31;
        S += svgText(412, y + 12, KEYS[i][0], "end") + '<rect x="420" y="' + y + '" width="' + (w[i] * 100).toFixed(1) + '" height="16" rx="3" class="aw-f"/>' + svgText(424 + w[i] * 100, y + 12, w[i].toFixed(2), "start");
      }
      svg.innerHTML = S;
      var H = 0;
      for (i = 0; i < w.length; i++) if (w[i] > 0) H -= w[i] * Math.log(w[i]);
      var best = scores.indexOf(Math.max.apply(null, scores));
      out.innerHTML = "<p>query 在 (" + q[0].toFixed(1) + ", " + q[1].toFixed(1) + ")：点积最大的是「" + KEYS[best][0] + "」，权重 " + w[best].toFixed(2) +
        "。输出是 8 个 value 按权重的混合——这里把 value 画在和 key 相同的位置，所以输出落在被选中的几个点之间（真实模型里 value 是另一组向量，key 负责\"被找到\"，value 负责\"给什么\"）。权重的熵 " + H.toFixed(2) + "（" + Math.log(KEYS.length).toFixed(2) + " = 平均分配，0 = 只看一个）。</p>" +
        '<p class="aw-note">放大倍数就是分数的尺度：太大时 softmax 变成 one-hot（硬查表，梯度几乎为零），太小时变成平均（什么都没选）。点积的方差随维度 d 线性增长，所以要除以 √d 把尺度拉回来。把 query 拖远一点、再拉大倍数，看权重怎么集中到一个点上。</p>';
    }
    function toPlane(e) {
      var r = svg.getBoundingClientRect(), x = (e.clientX - r.left) / r.width * 560, y = (e.clientY - r.top) / r.height * 300;
      return [Math.max(-3.4, Math.min(3.4, (x - cx) / u)), Math.max(-2.9, Math.min(2.9, (cy - y) / u))];
    }
    svg.style.touchAction = "none"; svg.style.cursor = "crosshair";
    svg.addEventListener("pointerdown", function (e) { drag = true; svg.setPointerCapture(e.pointerId); q = toPlane(e); draw(); e.preventDefault(); });
    svg.addEventListener("pointermove", function (e) { if (drag) { q = toPlane(e); draw(); } });
    svg.addEventListener("pointerup", function () { drag = false; });
    svg.addEventListener("pointercancel", function () { drag = false; });
    bind(box, draw);
  }

  // ---------------------------------------------------------------- LayerNorm 与 RMSNorm：同一个向量，两种归一化
  function normw(box) {
    var CASES = {
      "普通的 8 维向量": [1.2, -0.7, 0.4, 2.0, -1.5, 0.3, -0.2, 0.9],
      "整体偏移 +5（均值不为 0）": [6.2, 4.3, 5.4, 7.0, 3.5, 5.3, 4.8, 5.9],
      "整体放大 ×20（深层的残差流）": [24, -14, 8, 40, -30, 6, -4, 18],
      "有一个离群维度（真实模型里常见）": [1.2, -0.7, 0.4, 30, -1.5, 0.3, -0.2, 0.9]
    };
    box.innerHTML = '<div class="aw-title">同一个 token 向量，LayerNorm 和 RMSNorm 各做了什么</div><div class="aw-grid">' +
      row("输入 x", select("case", Object.keys(CASES), Object.keys(CASES)[0]), true) + row("γ（学到的缩放）", range2("gamma", 10, 2, 30)) +
      '</div><svg class="aw-chart" viewBox="0 0 560 190"></svg><div class="aw-out"></div>';
    var svg = box.querySelector("svg"), out = box.querySelector(".aw-out");
    function bars(x0, vals, scale, label, cls) {
      var S = svgText(x0 + 72, 14, label, "middle"), base = 112, i;
      S += '<text x="' + (x0 + 72) + '" y="28" class="aw-t" font-size="9" text-anchor="middle">' + vals.map(function (v) { return Math.abs(v) >= 10 ? v.toFixed(0) : v.toFixed(1); }).join("  ") + "</text>";
      S += '<line x1="' + x0 + '" y1="' + base + '" x2="' + (x0 + 150) + '" y2="' + base + '" class="aw-axis"/>';
      for (i = 0; i < vals.length; i++) {
        var h = Math.max(-66, Math.min(66, vals[i] * scale)), x = x0 + 4 + i * 18;
        S += '<rect x="' + x + '" y="' + (h >= 0 ? base - h : base).toFixed(1) + '" width="14" height="' + Math.abs(h).toFixed(1) + '" rx="2" class="' + cls + '"/>';
      }
      return S;
    }
    bind(box, function () {
      var x = CASES[val(box, "case")], g = val(box, "gamma") / 10, d = x.length, i;
      show(box, "gamma", g.toFixed(1));
      var mu = 0; for (i = 0; i < d; i++) mu += x[i]; mu /= d;
      var v = 0; for (i = 0; i < d; i++) v += (x[i] - mu) * (x[i] - mu); v /= d;
      var ms = 0; for (i = 0; i < d; i++) ms += x[i] * x[i]; ms /= d;
      var sd = Math.sqrt(v + 1e-6), rms = Math.sqrt(ms + 1e-6);
      var ln = x.map(function (t) { return (t - mu) / sd * g; }), rn = x.map(function (t) { return t / rms * g; });
      var mx = Math.max.apply(null, x.map(Math.abs)), S = bars(10, x, 66 / mx, "x", "aw-off") + bars(200, ln, 22, "LayerNorm(x)", "aw-f") + bars(390, rn, 22, "RMSNorm(x)", "aw-on");
      S += svgText(280, 160, "μ = " + mu.toFixed(2) + "，σ = " + sd.toFixed(2) + "，RMS = " + rms.toFixed(2) + "；归一化后：LayerNorm 均值 0、标准差 " + g.toFixed(1) + "；RMSNorm 均方根 " + g.toFixed(1) + "、均值 " + (mu / rms * g).toFixed(2), "middle");
      S += svgText(280, 180, "右边两组用同一比例画（±3 顶满）；左边按自己的最大值缩放", "middle");
      svg.innerHTML = S;
      var c = val(box, "case"), msg = c.indexOf("偏移") >= 0 ? "LayerNorm 把整体偏移减掉了，RMSNorm 没有——它只除以均方根，偏移被一起缩小但还在（均值 " + (mu / rms * g).toFixed(2) + "）。实践里这点差别几乎不影响效果，于是省掉减均值这一步。" :
        c.indexOf("放大") >= 0 ? "放大 20 倍后两种归一化的输出和原来完全一样：归一化对尺度不敏感，这就是 Pre-Norm 里残差流可以越长越大、每层读到的输入却始终是同样尺度的原因。" :
        c.indexOf("离群") >= 0 ? "一个 30 把 RMS 拉到 " + rms.toFixed(1) + "，其余 7 个维度被压到接近 0——离群维度主导了归一化。这就是量化里头疼的\"激活离群值\"：它们在归一化前后都存在，而且 γ 常常会把它们放得更大。" :
        "普通情况下两种结果几乎一样：每个维度除以一个\"整体尺度\"。γ 是可学习的逐维缩放，推理引擎常把它融合进后面的矩阵乘（或者融合进归一化 kernel 里）。";
      out.innerHTML = "<p>" + msg + "</p>" + '<p class="aw-note">统计量（μ、σ、RMS）要用 FP32 算：BF16 下对几千个数求平方和会丢精度。推理引擎的 RMSNorm kernel 一次读入整个向量、一次归约、再写出，和残差相加融合成一个 kernel（add + rmsnorm）。</p>';
    });
  }

  // ---------------------------------------------------------------- 一层里参数怎么分：注意力 vs FFN，以及整个模型
  function paramShare(box) {
    var PRESETS = {                 // d, d_ff, n_h, n_kv, d_h, 层数, 词表, 是否共享嵌入
      "Qwen3-0.6B": [1024, 3072, 16, 8, 128, 28, 151936, 1], "LLaMA-7B": [4096, 11008, 32, 32, 128, 32, 32000, 0],
      "LLaMA-3-8B": [4096, 14336, 32, 8, 128, 32, 128256, 0], "Qwen2.5-7B": [3584, 18944, 28, 4, 128, 28, 152064, 0], "LLaMA-3-70B": [8192, 28672, 64, 8, 128, 80, 128256, 0]
    };
    box.innerHTML = '<div class="aw-title">一层的参数怎么分：q / k / v / o 和 gate / up / down；整个模型再加上嵌入与输出层</div><div class="aw-grid">' +
      row("模型", select("preset", Object.keys(PRESETS), "Qwen3-0.6B")) + row("d", num("d", 1024, 64, 32768, 64)) + row("d_ff", num("dff", 3072, 64, 131072, 64)) +
      row("query 头数", num("nh", 16, 1, 256)) + row("KV 头数", num("nkv", 8, 1, 256)) + row("头维 d_h", num("dh", 128, 8, 512, 8)) +
      row("层数", num("L", 28, 1, 256)) + row("词表", num("V", 151936, 256, 1000000, 256)) + row("嵌入与输出层共享", select("tie", ["是", "否"], "是")) + row("上下文长度 T", num("T", 4096, 1, 1048576, 256)) +
      '</div><svg class="aw-chart" viewBox="0 0 560 120"></svg><div class="aw-out"></div>';
    var svg = box.querySelector("svg"), out = box.querySelector(".aw-out");
    box.querySelector('[data-k="preset"]').addEventListener("change", function () {
      var p = PRESETS[this.value], keys = ["d", "dff", "nh", "nkv", "dh", "L", "V"];
      keys.forEach(function (k, i) { input(box, k).value = p[i]; });
      input(box, "tie").value = p[7] ? "是" : "否"; draw();
    });
    function fmtP(n) { return n >= 1e9 ? (n / 1e9).toFixed(2) + " B" : (n / 1e6).toFixed(1) + " M"; }
    function stack(y, parts, total, label) {
      var S = svgText(8, y + 14, label, "start"), x = 120, i;
      for (i = 0; i < parts.length; i++) {
        var w = parts[i][1] / total * 430;
        S += '<rect x="' + x.toFixed(1) + '" y="' + y + '" width="' + Math.max(0, w - 1).toFixed(1) + '" height="22" class="' + parts[i][2] + '"/>';
        if (w > 34) S += svgText(x + w / 2, y + 15, parts[i][0] + " " + Math.round(parts[i][1] / total * 100) + "%", "middle");
        x += w;
      }
      return S;
    }
    function draw() {
      var d = val(box, "d"), dff = val(box, "dff"), nh = val(box, "nh"), nkv = val(box, "nkv"), dh = val(box, "dh"), L = val(box, "L"), V = val(box, "V"), tie = val(box, "tie") === "是", T = val(box, "T");
      var q = d * nh * dh, k = d * nkv * dh, v = k, o = nh * dh * d, attn = q + k + v + o, mlp = 3 * d * dff, layer = attn + mlp;
      var embed = V * d, head = tie ? 0 : V * d, total = L * layer + embed + head;
      svg.innerHTML = stack(8, [["q", q, "aw-f"], ["k", k, "aw-i2"], ["v", v, "aw-i2"], ["o", o, "aw-f"], ["gate", d * dff, "aw-b"], ["up", d * dff, "aw-b"], ["down", d * dff, "aw-b"]], layer, "一层") +
        stack(48, [["注意力", L * attn, "aw-f"], ["FFN", L * mlp, "aw-b"], ["嵌入", embed, "aw-j2"]].concat(head ? [["输出层", head, "aw-j2"]] : []), total, "整个模型") +
        svgText(8, 100, "每个参数对每个 token 贡献 2 FLOP，所以参数占比 ≈ 权重部分的计算占比", "start");
      var scoreFlops = L * 4 * T * nh * dh, weightFlops = 2 * (L * layer + (tie ? V * d : head));   // 注意力分数：QKᵀ 和 PV 各 2·T·d_h 每头每 token
      out.innerHTML = "<p>一层 <b>" + fmtP(layer) + "</b> 个参数：注意力 " + fmtP(attn) + "（" + Math.round(attn / layer * 100) + "%），FFN " + fmtP(mlp) + "（" + Math.round(mlp / layer * 100) + "%）" +
        (nkv < nh ? "；KV 头是 query 头的 1/" + (nh / nkv) + "，k、v 投影只有 q 的 1/" + (nh / nkv) + "（GQA）" : "") + "。整个模型 <b>" + fmtP(total) + "</b>：" + L + " 层共 " + fmtP(L * layer) +
        "，嵌入 " + fmtP(embed) + (tie ? "（输出层共享这份权重）" : "，输出层 " + fmtP(head)) + "——词表部分占 " + Math.round((embed + head) / total * 100) + "%" + ((embed + head) / total > 0.2 ? "，小模型里这一块非常大，所以小模型常共享嵌入和输出层" : "") + "。</p>" +
        "<p>上下文 " + T.toLocaleString("zh-CN") + " 时，一个 token 过权重要 " + (weightFlops / 1e9).toFixed(2) + " GFLOP，注意力分数（QKᵀ 和 PV）另外要 " + (scoreFlops / 1e9).toFixed(2) + " GFLOP，占 " + Math.round(scoreFlops / (scoreFlops + weightFlops) * 100) + "%" +
        (scoreFlops > weightFlops ? "——长上下文下注意力分数反过来成了大头，这是长上下文推理要专门优化注意力 kernel 的原因" : "") + "。</p>";
    }
    bind(box, draw);
  }

  // ---------------------------------------------------------------- MoE：token 怎么被路由，负载怎么不均衡
  function moeRoute(box) {
    box.innerHTML = '<div class="aw-title">MoE 路由：64 个 token，每个选 top-k 个专家；路由器越偏心，负载越不均衡</div><div class="aw-grid">' +
      row("专家数 E", select("E", ["8", "16", "32", "64"], "16")) + row("每个 token 选 k 个", select("k", ["1", "2", "4", "8"], "2")) +
      row("路由器的偏心程度", range2("skew", 30, 0, 100)) + row("容量因子", range2("cap", 125, 100, 250)) +
      '</div><svg class="aw-chart" viewBox="0 0 560 170"></svg><div class="aw-out"></div>';
    var svg = box.querySelector("svg"), out = box.querySelector(".aw-out"), N = 64;
    function rng(seed) { var s = seed >>> 0; return function () { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; }; }
    bind(box, function () {
      var E = +val(box, "E"), k = Math.min(+val(box, "k"), E), skew = val(box, "skew") / 100, cap = val(box, "cap") / 100, r = rng(7), i, e;
      show(box, "skew", skew.toFixed(2)); show(box, "cap", cap.toFixed(2));
      var load = [], Psum = [], fav = [];
      for (e = 0; e < E; e++) { load.push(0); Psum.push(0); fav.push(e < Math.max(1, Math.round(E / 8)) ? 1 : 0); }   // 偏心：路由器偏爱前 1/8 的专家
      for (i = 0; i < N; i++) {
        var logits = [], m = -1e9, Z = 0, probs = [];
        for (e = 0; e < E; e++) { var g = Math.sqrt(-2 * Math.log(r() + 1e-12)) * Math.cos(2 * Math.PI * r()); logits.push(g + skew * 4 * fav[e]); m = Math.max(m, logits[e]); }
        for (e = 0; e < E; e++) { probs.push(Math.exp(logits[e] - m)); Z += probs[e]; }
        for (e = 0; e < E; e++) { probs[e] /= Z; Psum[e] += probs[e] / N; }
        var order = logits.map(function (v, j) { return [v, j]; }).sort(function (a, b) { return b[0] - a[0]; });
        for (var t = 0; t < k; t++) load[order[t][1]] += 1;
      }
      var capacity = Math.ceil(cap * N * k / E), dropped = 0, aux = 0, maxLoad = 0;
      for (e = 0; e < E; e++) { dropped += Math.max(0, load[e] - capacity); aux += (load[e] / (N * k)) * Psum[e]; maxLoad = Math.max(maxLoad, load[e]); }
      aux *= E;
      var S = "", bw = 500 / E, hmax = Math.max(maxLoad, capacity) * 1.05, Y = function (n) { return 140 - n / hmax * 120; };
      for (e = 0; e < E; e++) {
        var x = 40 + e * bw, inCap = Math.min(load[e], capacity);
        S += '<rect x="' + x.toFixed(1) + '" y="' + Y(inCap).toFixed(1) + '" width="' + (bw - 2).toFixed(1) + '" height="' + (140 - Y(inCap)).toFixed(1) + '" class="aw-f"/>';
        if (load[e] > capacity) S += '<rect x="' + x.toFixed(1) + '" y="' + Y(load[e]).toFixed(1) + '" width="' + (bw - 2).toFixed(1) + '" height="' + (Y(inCap) - Y(load[e])).toFixed(1) + '" class="aw-b"/>';
        if (E <= 16) S += svgText(x + bw / 2 - 1, 154, "E" + e, "middle");
      }
      S += '<line x1="40" y1="' + Y(capacity).toFixed(1) + '" x2="540" y2="' + Y(capacity).toFixed(1) + '" class="aw-dash"/>';
      S += '<line x1="40" y1="' + Y(N * k / E).toFixed(1) + '" x2="540" y2="' + Y(N * k / E).toFixed(1) + '" class="aw-axis"/>';
      S += svgText(36, 20, "每个专家分到的 token 数", "start") + svgText(540, 20, "虚线 = 容量 " + capacity + "，实线 = 均匀 " + (N * k / E).toFixed(1), "end");
      svg.innerHTML = S;
      out.innerHTML = "<p>" + N + " 个 token × top-" + k + " = " + N * k + " 次分配，均匀时每个专家 " + (N * k / E).toFixed(1) + " 个；现在最忙的专家 <b>" + maxLoad + "</b> 个，辅助损失 E·Σ f<sub>i</sub>P<sub>i</sub> = <b>" + aux.toFixed(2) +
        "</b>（完全均匀时为 1）；容量 " + capacity + "，超出的 <b>" + dropped + "</b> 个 token 被丢弃（直接走残差）。</p>" +
        '<p class="aw-note">每个 token 只算 ' + k + ' 个专家：计算量是同样总参数的稠密 FFN 的 ' + k + '/' + E + '，这就是 MoE "参数多、算得少"的来源；但权重还是全都要放在显存里，推理时专家按 token 分组做 GEMM，负载不均衡就意味着有的卡（专家并行时）在等最忙的那个。偏心程度拉到最右，就是训练初期路由器"塌缩"的样子。</p>';
    });
  }

  var WIDGETS = { "kv-calc": kvCalc, roofline: roofline, mask: mask, pipeline: pipeline,
                  linmap: linmap, lowrank: lowrank, softmax: softmaxw, graddesc: graddesc,
                  coalesce: coalesce, bankconf: bankconf, scanviz: scanviz, occupancy: occupancy,
                  pagewalk: pagewalk, cachemap: cachemap, hashring: hashring,
                  pagedkv: pagedkv, radixcache: radixcache, contbatch: contbatch, spectree: spectree,
                  ringreduce: ringreduce, zeromem: zeromem, structlayout: structlayout,
                  embed3d: embed3d, "rope-helix": ropeHelix, swiglu3d: swiglu3d, scaling3d: scaling3d,
                  nextword: nextword, broadcast: broadcast, bpe: bpe, "float-bits": floatBits,
                  attention2d: attention2d, norm: normw, "param-share": paramShare, "moe-route": moeRoute };
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
