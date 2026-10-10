// 章节里的交互小工具。页面里放一个占位：<div class="aig-widget" data-widget="kv-calc"></div>
//   kv-calc   KV Cache 与显存：一张卡能同时服务多少个请求（大模型原理 · KV Cache）
//   roofline  矩阵乘的屋顶线：batch 变大，瓶颈从访存变成算力（数学基础 · 性能与服务中的数学）
//   mask      注意力掩码：因果、带历史、滑动窗口、序列打包、前缀双向，以及按块跳过（大模型原理 · 注意力机制）
//   pipeline  流水线的调度与气泡：GPipe 与 1F1B（分布式训练 · 流水线并行）
//   linmap    线性变换：2×2 矩阵把网格变成什么样（数学基础 · 线性代数）
//   lowrank   低秩近似：保留几个奇异值就够（数学基础 · 线性代数）
//   softmax   softmax 与采样：温度、top-k、top-p（数学基础 · 概率与采样）
//   graddesc  梯度下降：学习率、动量与条件数（数学基础 · 微积分与反向传播）
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
//   float-bits 浮点数拆成符号、指数、尾数，看舍入误差和 ulp（数学基础 · 浮点与数值计算）
//   attention2d 注意力 = 软查表：拖动 query 看权重和加权平均（大模型原理 · 注意力机制）
//   norm      LayerNorm 与 RMSNorm 对同一个向量各做了什么（大模型原理 · 归一化与残差流）
//   param-share 一层和整个模型的参数怎么分，注意力分数的 T² 项何时成为大头（大模型原理 · 前馈网络与 SwiGLU）
//   moe-route MoE 路由：top-k、负载不均衡、容量与丢弃（大模型原理 · 混合专家 MoE）
//   dpo-loss  DPO 损失曲线：β 与 margin（大模型原理 · 后训练）
//   lora-params LoRA 的秩与目标模块：可训练参数、训练显存（大模型原理 · 后训练）
//   estimator 从 config.json 到延迟下限：参数量、每 token 计算量、TPOT / TTFT（大模型原理 · 参数量、算力与显存估算）
//   quant     量化误差：离群值与按组量化（大模型原理 · 量化原理）
//   model-map 十个模型的地图：总参数、激活参数、KV Cache（大模型原理 · 主流模型架构巡礼）
//   diffusion3d 加噪路径的三维图：数据 → 噪声（图像与视频生成 · 扩散与流匹配）
//   cfg-guide 无分类器引导：外推怎么改变分布（图像与视频生成 · 扩散与流匹配）
//   ode-solver 欧拉 vs Heun：步数与阶数（图像与视频生成 · 扩散与流匹配）
//   sigma-schedule 流匹配的时间步与 shift（图像与视频生成 · 采样器与调度器）
//   diffusion-flops 一步的 FLOP 与一次生成的时间（图像与视频生成 · 算账）
//   offload-cost 三种 offload 各搬多少、拖慢多少（图像与视频生成 · 显存与 offload）
//   cache-skip TeaCache 式跳步：阈值决定跳哪些步（图像与视频生成 · 特征缓存）
//   video3d   视频 token 的三维注意力模式（图像与视频生成 · 视频模型的结构）
//   video-stack 视频推理的手段叠加（图像与视频生成 · 视频推理的瓶颈）
//   alphabeta α-β 模型：消息多大才能跑满带宽（推理系统 · 互联）
//   collective-cost ring / tree / one-shot / two-shot / NVLS 的 α-β 时间（推理系统 · NCCL）
//   ragged-waste 填充 batch 与拼接 batch 的浪费（推理系统 · batch 布局）
//   launch-overhead kernel 启动开销：eager、CUDA Graph、融合（推理系统 · CUDA Graph 与编译）
//   linear-memory 混合模型的显存账本：KV 与固定状态（推理系统 · 线性注意力）
//   spec-load 投机解码的收益随负载变化（推理系统 · 投机解码进阶）
//   pd-ratio  xPyD 配比（推理系统 · 分离式调度）
//   dp-straggler DP attention 的负载：最慢的 rank 决定速度（推理系统 · EP 部署）
//   kv-evict  KV 淘汰：滑动窗口与注意力汇聚（推理系统 · 长上下文）
//   vision-tokens 一张图变成多少个 token（推理系统 · 多模态）
//   open-closed 开环与闭环压测（推理系统 · 压测与容量规划）
//   online-softmax 一次遍历维护 (m, d)（CUDA · softmax 与归一化）
//   gemm3d    GEMM 的三级分块：block / warp / mma tile 的三维图（CUDA · Tensor Core）
//   cute-layout 形状 : 步长 → 坐标到偏移（CUDA · CuTe 布局）
//   stream-overlap 多条流让拷贝与计算重叠（CUDA · 流与事件）
//   grid-index blockIdx、threadIdx 与 grid-stride（CUDA · 第一个 kernel）
//   stride-view storage + sizes + strides（CUDA · 张量）
//   adamw-step AdamW 与 Adam + L2 的衰减差别（分布式训练 · 优化器）
//   newton-schulz Muon 的正交化：奇异值被推向 1（分布式训练 · 优化器）
//   lr-schedule warmup、余弦、WSD（分布式训练 · 训练稳定性）
//   softmax-entropy 注意力 logits 的尺度与熵坍缩（分布式训练 · 训练稳定性）
//   grpo-adv  GRPO 的组内优势与损失平均（分布式训练 · RL 算法）
//   kl-estimators KL 的三种单样本估计量（分布式训练 · RL 算法）
//   train-time 6ND / (卡数 × 峰值 × MFU)（分布式训练 · 总览）
//   ddp-overlap DDP 按桶 all-reduce 与反向重叠（分布式训练 · DDP）
//   tp-comm   张量并行的通信占比（分布式训练 · 张量并行与序列并行）
//   complexity 复杂度：增长与常数（计算机基础 · 算法总览）
//   window-step 滑动窗口逐步演示（计算机基础 · 数组与字符串）
//   edit-distance 编辑距离的 DP 表怎么填（计算机基础 · 动态规划）
//   ridge-gen 每一代 GPU 的屋脊点与 decode 所需 batch（计算机基础 · GPU 的演进）
//   little-law 在途数据 = 带宽 × 延迟（计算机基础 · GPU 内存）
//   mm1-latency 排队论：利用率与延迟（计算机基础 · 负载均衡）
//   vector-realloc vector 扩容时是移动还是拷贝（C++ · 移动语义）
//   sgl-timeline SGLang 按月的提交数、版本与大事件（SGLang 设计演进 · 首页）
// 后四个用文件中段的 view3d 小引擎：SVG 里的画家算法 + 拖动旋转，不依赖任何 3D 库。
// 英文版（en/ 下的页面，<html lang="en">）：EN 为真，界面文字用 zhen("中文", "English") 取；还没双语化的小工具照常显示中文。
// 字节数按 1024 进位（和正文里"每个 token 112 KB"的算法一致）。
(function () {
  var EN = /^en/.test(document.documentElement.lang || "");   // 英文版页面（en/）：小工具的界面文字用英文；全部十四本书用到的小工具都已双语化
  function zhen(zh, en) { return EN ? en : zh; }   // 不叫 L：好几个小工具里 L 是层数
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
  function select(id, opts, v) {                // opts 的每一项是值本身，或者 [值, 显示的文字]（英文版显示译文、值仍用中文）
    return '<select data-k="' + id + '">' + opts.map(function (o) {
      var value = Array.isArray(o) ? o[0] : o, label = Array.isArray(o) ? o[1] : o;
      return '<option value="' + value + '"' + (value === v ? " selected" : "") + ">" + label + "</option>";
    }).join("") + "</select>";
  }
  function opts(keys, en) { return keys.map(function (k) { return [k, zhen(k, en[k] || k)]; }); }   // 中文键 → 下拉选项（英文版显示 en 里的译文）
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
    box.innerHTML = '<div class="aw-title">' + zhen("KV Cache 与显存：一张卡能同时服务多少个请求", "KV cache and memory: how many requests one GPU can serve at once") + '</div><div class="aw-grid">' +
      row(zhen("模型", "Model"), select("model", opts(Object.keys(MODELS), { "DeepSeek-V3（MLA）": "DeepSeek-V3 (MLA)" }), "LLaMA-3-8B")) + row("GPU", select("gpu", Object.keys(GPUS), "H100 SXM")) +
      row(zhen("卡数（张量并行）", "GPUs (tensor parallel)"), select("tp", ["1", "2", "4", "8"], "1")) + row(zhen("权重精度", "Weight precision"), select("wdt", ["BF16", "FP8", "INT4"], "BF16")) +
      row(zhen("KV 精度", "KV precision"), select("kdt", ["BF16", "FP8"], "BF16")) + row(zhen("平均上下文（token）", "Average context (tokens)"), num("ctx", 4096, 1, 1048576, 256)) +
      row(zhen("每张卡预留（GB）", "Reserved per GPU (GB)"), num("reserve", 8, 0, 64)) + '</div><div class="aw-out"></div>';
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
      if (EN) {
        var h = "<p>KV per token per GPU: <b>" + fmtBytes(perTok) + "</b> (" + (mla ? m[0] + " layers × 576 dims × " + kb + " bytes; MLA caches only the latent vector" :
          "2 × " + m[0] + " layers × " + (headsPerCard % 1 ? headsPerCard.toFixed(1) : headsPerCard) + " KV heads × " + m[2] + " dims × " + kb + " bytes") +
          "); a request of " + fmtInt(ctx) + " tokens takes " + fmtBytes(perReq) + "</p>";
        if (!mla && tp > m[1]) h += '<p class="aw-note">There are more GPUs than KV heads (' + m[1] + "), so the extra GPUs can only hold copies of KV heads, and the KV per GPU stops shrinking.</p>";
        if (mla && tp > 1) h += '<p class="aw-note">MLA\'s latent vector is shared by all heads, so tensor parallelism cannot split it and every GPU stores a full copy. That is why MLA models are often deployed with data-parallel attention (DP attention): different GPUs serve different requests.</p>';
        h += "<p>Weights per GPU " + fmtBytes(weights) + ", space left for KV: <b>" + (free > 0 ? fmtBytes(free) : "the weights do not fit") + "</b></p>";
        if (free > 0) {
          var n2 = Math.floor(free / perReq), tokens2 = Math.floor(free / perTok);
          h += "<p>Fits <b>" + fmtInt(n2) + "</b> requests of this length at once (KV for " + fmtInt(tokens2) + " tokens in total)" +
            (n2 < 1 ? ": not even one request fits; add GPUs or shorten the context" : "") + "</p>";
          if (n2 >= 1) {
            var bytes2 = weights + perReq * n2, flops2 = 2 * m[4] * 1e9 * n2 / tp;
            var peak2 = (val(box, "wdt") === "FP8" && g[3] ? g[3] : g[2]) * 1e12;
            var tMem2 = bytes2 / (g[1] * 1e12) * 1e3, tCmp2 = flops2 / peak2 * 1e3, ms2 = Math.max(tMem2, tCmp2);
            h += '<p class="aw-note">At full load, each decode step reads ' + fmtBytes(bytes2) + " per GPU (the weights plus every request's KV), at least " + tMem2.toFixed(1) +
              " ms by bandwidth; it computes " + (flops2 / 1e12).toFixed(1) + " TFLOP, at least " + tCmp2.toFixed(1) + " ms by compute. So a step takes at least " + ms2.toFixed(1) + " ms (" +
              (tMem2 >= tCmp2 ? "memory bound" : "compute bound") + "), and the instance's throughput is at most about " + fmtInt(n2 / ms2 * 1e3) + " tokens/s. The larger the KV, the fewer requests fit, and the slower each step.</p>";
          }
        } else {
          h += '<p class="aw-note">The weights do not fit: add GPUs, or use a lower weight precision.</p>';
        }
        out.innerHTML = h;
        return;
      }
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
    box.innerHTML = '<div class="aw-title">' + zhen("矩阵乘的屋顶线：[m, k] × [k, n]，m 是一个 batch 里的 token 数", "Roofline of a matmul: [m, k] × [k, n], where m is the number of tokens in a batch") + '</div><div class="aw-grid">' +
      row("GPU", select("gpu", Object.keys(GPUS), "H100 SXM")) + row(zhen("精度", "Precision"), select("dt", ["BF16", "FP8"], "BF16")) +
      row(zhen("k（输入维）", "k (input dim)"), num("k", 4096, 64, 65536, 64)) + row(zhen("n（输出维）", "n (output dim)"), num("n", 4096, 64, 65536, 64)) +
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
      if (!peak) { out.innerHTML = "<p>" + zhen("这张卡没有 FP8 Tensor Core，换一张卡或者选 BF16。", "This GPU has no FP8 Tensor Cores; pick another GPU or BF16.") + "</p>"; svg.innerHTML = ""; return; }
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
        svgText(X(ridge) + 5, 190, zhen("屋脊点 ", "Ridge ") + ridge.toFixed(0)) + svgText(X(1e4) - 4, Y(peak / 1e12) - 7, zhen("峰值 ", "Peak ") + (peak / 1e12) + " TFLOPS", "end") +
        '<circle cx="' + X(ai) + '" cy="' + Y(got / 1e12) + '" r="6" class="aw-dot"/>' +
        svgText(300, 225, zhen("算术强度（FLOP/字节）", "Arithmetic intensity (FLOP/byte)"), "middle") + '<text x="12" y="104" class="aw-t" transform="rotate(-90 12 104)" text-anchor="middle">TFLOPS</text>';
      svg.innerHTML = s;
      var bound = ai < ridge;
      var work = flops < 1e9 ? (flops / 1e6).toFixed(1) + " MFLOP" : flops < 1e12 ? (flops / 1e9).toFixed(flops < 1e10 ? 2 : 0) + " GFLOP" : (flops / 1e12).toFixed(2) + " TFLOP";
      var dur = t * 1e6 < 1000 ? (t * 1e6).toFixed(1) + " μs" : (t * 1e3).toFixed(2) + " ms";
      var gtf = (got / 1e12).toFixed(got < 1e13 ? 1 : 0), pct = (100 * got / peak).toFixed(got / peak < 0.1 ? 1 : 0);
      if (EN) {
        out.innerHTML = "<p>Work " + work + ", reads and writes " + fmtBytes(bytes) + ", arithmetic intensity <b>" + ai.toFixed(1) +
          "</b> FLOP/byte (this GPU's ridge point is " + ridge.toFixed(0) + ")</p><p><b>" + (bound ? "Memory bound" : "Compute bound") + "</b>: at least " +
          dur + ", at most " + gtf + " TFLOPS (" + pct + "% of peak)</p>" +
          '<p class="aw-note">' + (bound ? "With m far below the ridge point, nearly all the time goes to reading the weights, and growing m from 1 to a few dozen barely changes it: that is why decode batches requests." :
            "Past the ridge point, a larger m only adds time in proportion: prefill and large-batch decode live on this side, where peak compute is what counts and FP8 is twice as fast.") + "</p>";
        return;
      }
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
  var MASKS_EN = {
    "因果": ["Causal", "", "The default mask for training and prefill: token i sees only itself and earlier tokens. "],
    "带历史的 prefill": ["Prefill with history", "Cached tokens", "With a KV cache (chunked prefill, multi-turn chat), new tokens see the whole history: the diagonal shifts right by the history length, i.e. tril(diagonal=S−T). Decode is the special case of a single new token, whose row is fully visible. "],
    "滑动窗口": ["Sliding window", "Window W", "Each token sees only the most recent W tokens (itself included). KV outside the window can be dropped, so these layers' KV cache has a fixed size of W. "],
    "序列打包": ["Sequence packing", "Documents", "When several documents are packed into one training sample, they must not see each other, or the model learns false cross-document correlations. FlashAttention's variable-length interface marks each document's boundaries with cu_seqlens and computes them separately. "],
    "前缀双向": ["Bidirectional prefix", "Prefix length", "Tokens inside the prefix see each other (e.g. the input part of a Prefix-LM, or image tokens in some multimodal models); after the prefix, it is causal again. "]
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
    box.innerHTML = '<div class="aw-title">' + zhen("注意力掩码：哪个 query（行）能看到哪个 key（列）", "Attention masks: which query (row) can see which key (column)") + '</div><div class="aw-grid">' +
      row(zhen("掩码", "Mask"), select("kind", Object.keys(MASKS).map(function (k) { return [k, zhen(k, MASKS_EN[k][0])]; }), "因果")) + row(zhen("块大小", "Block size"), select("blk", ["2", "4", "8"], "4")) +
      row(zhen("序列长度", "Sequence length"), range("T", 16, 8, 32), true) + row("", range("a", 4, 1, 16), true) +
      '</div><svg class="aw-chart aw-mask"></svg><div class="aw-out"></div>';
    var out = box.querySelector(".aw-out"), svg = box.querySelector("svg"), param = input(box, "a").parentNode;
    var HI = { "带历史的 prefill": function (T) { return T - 1; }, "滑动窗口": function (T) { return T; },
               "序列打包": function () { return 4; }, "前缀双向": function (T) { return T; } };
    bind(box, function () {
      var kind = val(box, "kind"), T = val(box, "T"), blk = +val(box, "blk"), ra = input(box, "a");
      ra.max = HI[kind] ? HI[kind](T) : 1;
      var a = val(box, "a");
      param.hidden = !MASKS[kind][0];
      param.firstChild.textContent = zhen(MASKS[kind][0], MASKS_EN[kind][1]);
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
        if (hw >= 24) S += svgText(x0 + hw / 2, 8, hw >= 100 ? zhen("KV Cache 里的历史", "history in the KV cache") : zhen("历史", "history"), "middle");
        if (nw >= 24) S += svgText(x0 + hw + nw / 2, 8, nw >= 56 ? zhen("本次输入", "this input") : zhen("新", "new"), "middle");
      }
      var vw = x0 + T * c + 4, vh = y0 + q * c + 4;
      svg.setAttribute("viewBox", "0 0 " + vw + " " + vh);
      svg.setAttribute("width", Math.round(vw * 1.3));
      svg.innerHTML = S;
      if (EN) {
        out.innerHTML = "<p>" + q + " queries × " + T + " keys, <b>" + on + "</b> pairs visible (" + (100 * on / (q * T)).toFixed(0) + "%). In " + blk + "×" + blk +
          " blocks: <b>" + full + "</b> computed whole, <b>" + part + "</b> need an elementwise mask, <b>" + skip + "</b> skipped entirely (dashed)</p>" +
          '<p class="aw-note">' + MASKS_EN[kind][2] + "FlashAttention and FlexAttention both work block by block: fully invisible blocks are skipped, so the sparser the mask, the more computation is saved.</p>";
        return;
      }
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
    box.innerHTML = '<div class="aw-title">' + zhen("流水线的调度与气泡（前向耗时 1，反向耗时 2）", "Pipeline schedules and bubbles (forward costs 1, backward costs 2)") + '</div><div class="aw-grid">' +
      row(zhen("stage 数 p", "Stages p"), range("p", 4, 2, 8), true) + row(zhen("micro-batch 数 m", "Micro-batches m"), range("m", 8, 1, 16), true) +
      row(zhen("调度", "Schedule"), select("kind", ["1F1B", "GPipe"], "1F1B")) + '</div><svg class="aw-chart"></svg><div class="aw-out"></div>';
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
      out.innerHTML = "<p>" + zhen("总时间 " + r.total + " 个单位，每个 stage 真正在算的是 " + 3 * m + " 个：气泡 <b>", "total time " + r.total + " units with each stage computing for " + 3 * m + ": bubbles <b>") + (100 * (1 - 3 * m / r.total)).toFixed(1) +
        zhen("%</b>，和公式 (p−1)/(m+p−1) = ", "%</b>, matching the formula (p−1)/(m+p−1) = ") + (100 * (p - 1) / (m + p - 1)).toFixed(1) + "%</p><p>" + zhen("stage 0 最多同时保存 <b>", "stage 0 holds at most <b>") + r.peak[0] + zhen("</b> 份激活", "</b> sets of activations") +
        (one ? zhen("：1F1B 不超过 stage 数，与 m 无关", ": 1F1B stays within the stage count, independent of m") : zhen("：GPipe 等于 micro-batch 数，m 越大越占显存", ": GPipe equals the micro-batch count, so a larger m takes more memory")) + "</p>" +
        '<p class="aw-note">' + zhen("两种调度的气泡一样大。想让气泡变小，只能增大 m（受全局 batch 限制），或者用交错调度、零气泡调度把空闲填上。", "Both schedules have the same bubble. Shrinking it requires a larger m (bounded by the global batch), or interleaved and zero-bubble schedules that fill the idle time.") + '</p>';
    });
  }

  // ---------------------------------------------------------------- 线性变换：矩阵把网格变成什么样
  function linmap(box) {
    var PRESETS = {}, SHEAR = zhen("剪切", "Shear");
    [[zhen("拉伸（对角阵）", "Stretch (diagonal)"), [1.6, 0, 0, 0.6]], [zhen("旋转 30°", "Rotate 30°"), [0.866, -0.5, 0.5, 0.866]],
     [SHEAR, [1, 1, 0, 1]], [zhen("投影到一条线（秩 1）", "Project onto a line (rank 1)"), [1, 1, 0.5, 0.5]],
     [zhen("翻折（行列式为负）", "Flip (negative determinant)"), [0, 1, 1, 0]], [zhen("单位阵", "Identity"), [1, 0, 0, 1]]].forEach(function (p) { PRESETS[p[0]] = p[1]; });
    box.innerHTML = '<div class="aw-title">' + zhen("线性变换：一个 2×2 矩阵把平面变成什么样", "Linear maps: what a 2×2 matrix does to the plane") + '</div><div class="aw-grid">' +
      row(zhen("常见变换", "Presets"), select("preset", Object.keys(PRESETS), SHEAR)) +
      row(zhen("动画进度 t", "Animation t"), range2("t", 100, 0, 100) + '<button type="button" data-k="play" class="aw-btn">' + zhen("播放", "Play") + '</button>') +
      row(zhen("a（i 的 x）", "a (x of i)"), num("a", 1, -3, 3, 0.1)) + row(zhen("b（j 的 x）", "b (x of j)"), num("b", 1, -3, 3, 0.1)) +
      row(zhen("c（i 的 y）", "c (y of i)"), num("c", 0, -3, 3, 0.1)) + row(zhen("d（j 的 y）", "d (y of j)"), num("d", 1, -3, 3, 0.1)) +
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
      var eig = EN ? (disc >= 0
        ? "Eigenvalues " + ((tr + Math.sqrt(disc)) / 2).toFixed(2) + " and " + ((tr - Math.sqrt(disc)) / 2).toFixed(2) +
          ": two lines keep their direction, and vectors on them are only stretched"
        : "Complex eigenvalues: no real vector keeps its direction, so the map contains a rotation") : disc >= 0
        ? "特征值 " + ((tr + Math.sqrt(disc)) / 2).toFixed(2) + " 和 " + ((tr - Math.sqrt(disc)) / 2).toFixed(2) +
          "：有两条方向不变的直线，向量只被拉伸"
        : "特征值是复数：没有方向不变的实向量，这个变换里有旋转成分";
      if (EN) {
        out.innerHTML = "<p>Determinant <b>" + det.toFixed(3) + "</b>: the unit square's area is scaled by " + Math.abs(det).toFixed(2) +
          (det < 0 ? ", and it is <b>flipped</b> (handedness reversed)" : det === 0 ? ", and the whole plane is squashed onto a line (<b>rank 1</b>, not invertible)" : "") + "</p>" +
          "<p>" + eig + "</p>" +
          '<p class="aw-note">Multiplying a vector by the matrix recombines it from the new positions of i and j: Wx = x₁·(column 1) + x₂·(column 2). ' +
          "That is exactly what a linear layer does, only in thousands of dimensions instead of two. A zero determinant means rank deficiency, the geometric meaning of LoRA's assumption that the update spans only a few directions.</p>";
        return;
      }
      out.innerHTML = "<p>行列式 <b>" + det.toFixed(3) + "</b>：单位正方形的面积变成了它的 " + Math.abs(det).toFixed(2) + " 倍" +
        (det < 0 ? "，并且被<b>翻折</b>了（左右手性反过来）" : det === 0 ? "，整个平面被压扁到一条线上（<b>秩 1</b>，不可逆）" : "") + "</p>" +
        "<p>" + eig + "</p>" +
        '<p class="aw-note">矩阵乘以一个向量，就是把它按 i、j 的新位置重新组合：Wx = x₁·(列 1) + x₂·(列 2)。' +
        '线性层做的就是这件事，只不过维度是几千而不是二。行列式为 0 对应秩亏，正是 LoRA 假设"增量只占几个方向"的几何含义。</p>';
    }
    function stop() { if (timer) { clearInterval(timer); timer = null; box.querySelector('[data-k="play"]').textContent = zhen("播放", "Play"); } }
    box.querySelector('[data-k="preset"]').addEventListener("change", function () {
      var m = PRESETS[val(box, "preset")];
      ["a", "b", "c", "d"].forEach(function (k, i) { input(box, k).value = m[i]; });
      draw();
    });
    box.querySelector('[data-k="play"]').addEventListener("click", function () {
      if (timer) return stop();
      this.textContent = zhen("暂停", "Pause");
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
    var K1 = zhen("低秩图案（两个方向）", "Low-rank pattern"), K2 = zhen("接近低秩 + 噪声", "Low rank + noise"), K3 = zhen("满秩噪声", "Full-rank noise");
    function build(kind) {                                   // 造一个 N×N 的"图案"矩阵
      var M = [];
      for (var i = 0; i < N; i++) {
        M[i] = [];
        for (var j = 0; j < N; j++) {
          var low = Math.sin(i / 3) * Math.cos(j / 4) + 0.6 * Math.sin(i / 7) * Math.cos(j / 5);
          if (kind === K1) M[i][j] = low;
          else if (kind === K2) M[i][j] = low + 0.22 * noise(i, j);
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
    box.innerHTML = '<div class="aw-title">' + zhen("低秩近似：保留几个奇异值，矩阵还剩多少信息", "Low-rank approximation: keep a few singular values, see how much of the matrix is left") + '</div><div class="aw-grid">' +
      row(zhen("矩阵", "Matrix"), select("kind", [K1, K2, K3], K2)) +
      row(zhen("保留的秩 r", "Rank kept r"), range2("r", 3, 1, 24) + '<button type="button" data-k="play" class="aw-btn">' + zhen("播放", "Play") + '</button>') + '</div>' +
      '<svg class="aw-chart aw-lr" viewBox="0 0 560 210"></svg><div class="aw-out"></div>';
    var svg = box.querySelector("svg"), out = box.querySelector(".aw-out"), cache = {}, timer = null;
    function stop() { if (timer) { clearInterval(timer); timer = null; box.querySelector('[data-k="play"]').textContent = zhen("播放", "Play"); } }
    box.querySelector('[data-k="play"]').addEventListener("click", function () {
      if (timer) return stop();
      this.textContent = zhen("暂停", "Pause");
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
      var S = svgText(x0 + N * cell / 2, 16, zhen("原矩阵", "Original"), "middle") + svgText(x1 + N * cell / 2, 16, zhen("秩 " + r + " 的近似", "Rank-" + r + " approximation"), "middle");
      for (i = 0; i < N; i++) for (j = 0; j < N; j++) {
        err += (A[i][j] - R[i][j]) * (A[i][j] - R[i][j]);
        tot += A[i][j] * A[i][j];
        S += '<rect x="' + (x0 + j * cell) + '" y="' + (26 + i * cell) + '" width="' + cell + '" height="' + cell +
             '" fill="' + heat(A[i][j]) + '"/>';
        S += '<rect x="' + (x1 + j * cell) + '" y="' + (26 + i * cell) + '" width="' + cell + '" height="' + cell +
             '" fill="' + heat(R[i][j]) + '"/>';
      }
      var bx = 400, bw = 130, bh = 120;                       // 奇异值柱状图
      S += svgText(bx + bw / 2, 16, zhen("奇异值", "Singular values"), "middle");
      var smax = dec.s[0] || 1;
      for (i = 0; i < N; i++) {
        var h = Math.max(1, dec.s[i] / smax * bh);
        S += '<rect x="' + (bx + i * (bw / N)) + '" y="' + (26 + bh - h) + '" width="' + (bw / N - 1) + '" height="' + h +
             '" class="' + (i < r ? "aw-on" : "aw-off") + '"/>';
      }
      S += '<line x1="' + (bx + r * (bw / N)) + '" y1="20" x2="' + (bx + r * (bw / N)) + '" y2="' + (26 + bh) + '" class="aw-dash"/>';
      S += svgText(x0 + N * cell / 2, 26 + N * cell + 16, zhen("24 × 24 = 576 个数", "24 × 24 = 576 numbers"), "middle");
      S += svgText(x1 + N * cell / 2, 26 + N * cell + 16, "2 × 24 × " + r + " = " + (2 * N * r) + zhen(" 个数", " numbers"), "middle");
      svg.innerHTML = S;
      var rel = Math.sqrt(err / tot);
      if (EN) {
        out.innerHTML = "<p>Relative error <b>" + (100 * rel).toFixed(1) + "%</b>, parameters <b>" +
          (100 * 2 * N * r / (N * N)).toFixed(0) + "%</b> (" + (2 * N * r) + " / " + (N * N) + ")</p>" +
          '<p class="aw-note">When the singular values drop fast, a few directions recover most of the information: LoRA bets that fine-tuning updates are like this, ' +
          "and MLA bets that KV activations are. Switch to full-rank noise and you will see r has to be close to full rank before it looks right.</p>";
        return;
      }
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
    var WORDS = EN ? ["the", "a", "of", "to", "and", "in", "is", "it", "he", "this", "on", "big"] : ["的", "是", "了", "在", "和", "有", "人", "我", "他", "这", "中", "大"];
    box.innerHTML = '<div class="aw-title">' + zhen("softmax 与采样：温度、top-k、top-p 各自在做什么", "softmax and sampling: what temperature, top-k and top-p each do") + '</div><div class="aw-grid">' +
      row(zhen("温度 T", "Temperature T"), range2("temp", 100, 10, 200)) + row(zhen("top-k（0 = 不限）", "top-k (0 = no limit)"), range2("k", 0, 0, 12)) +
      row("top-p", range2("p", 90, 10, 100) + '<button type="button" data-k="play" class="aw-btn">' + zhen("播放", "Play") + '</button>') + '</div>' +
      '<svg class="aw-chart aw-sm" viewBox="0 0 560 202"></svg><div class="aw-out"></div>';
    var svg = box.querySelector("svg"), out = box.querySelector(".aw-out"), timer = null;
    function stop() { if (timer) { clearInterval(timer); timer = null; box.querySelector('[data-k="play"]').textContent = zhen("播放", "Play"); } }
    box.querySelector('[data-k="play"]').addEventListener("click", function () {
      if (timer) return stop();
      this.textContent = zhen("暂停", "Pause");
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
      show(box, "temp", T.toFixed(2)); show(box, "k", k ? String(k) : zhen("不限", "none")); show(box, "p", p.toFixed(2));
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
      S += svgText(20, 190, zhen("蓝色 = 可能被采到，灰色 = 被 top-k / top-p 截掉", "Blue = can be sampled, gray = cut by top-k / top-p"), "start");
      svg.innerHTML = S;
      if (EN) {
        out.innerHTML = "<p>Entropy <b>" + entropy.toFixed(2) + " bits</b> (higher = more hesitant); <b>" + kept.length +
          "</b> candidates holding <b>" + (100 * ksum).toFixed(1) + "%</b> of the probability; top probability <b>" + (100 * prob[0]).toFixed(1) + "%</b></p>" +
          '<p class="aw-note">The temperature divides the logits: T &lt; 1 widens the gaps (more certain, more repetitive), T &gt; 1 flattens them (more diverse). ' +
          "T → 0 is greedy decoding. top-k keeps a fixed number of tokens; top-p keeps tokens by cumulative probability, so it keeps one or two when the distribution is sharp and more when it is flat.</p>";
        return;
      }
      out.innerHTML = "<p>熵 <b>" + entropy.toFixed(2) + " 比特</b>（越大越犹豫）；候选 <b>" + kept.length +
        "</b> 个，占总概率 <b>" + (100 * ksum).toFixed(1) + "%</b>；最大概率 <b>" + (100 * prob[0]).toFixed(1) + "%</b></p>" +
        '<p class="aw-note">温度除在 logits 上：T &lt; 1 放大差距（更确定、更容易重复），T &gt; 1 拉平（更发散）。' +
        'T → 0 就是贪心解码。top-k 固定留几个，top-p 按累计概率动态留——分布尖锐时只留一两个，平坦时留更多。</p>';
    });
  }

  // ---------------------------------------------------------------- 梯度下降
  function graddesc(box) {
    box.innerHTML = '<div class="aw-title">' + zhen("梯度下降：学习率与动量怎么影响轨迹", "Gradient descent: how learning rate and momentum shape the path") + '</div><div class="aw-grid">' +
      row(zhen("学习率", "Learning rate"), range2("lr", 20, 1, 100)) + row(zhen("动量", "Momentum"), range2("mom", 0, 0, 95)) +
      row(zhen("曲面的拉伸（条件数）", "Surface stretch (condition number)"), range2("cond", 8, 1, 20)) +
      row("", '<button type="button" data-k="play" class="aw-btn">' + zhen("播放", "Play") + '</button>', true) +
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
      S += svgText(st[0] + 8, st[1] - 8, zhen("起点", "Start"), "start");
      S += svgText(20, 18, zhen("椭圆 = 损失相同的点，正中心是最小值", "Ellipses = points of equal loss; the center is the minimum"), "start");
      svg.innerHTML = S;
      var last = path[n - 1];
      var loss = (last[0] * last[0] + cond * last[1] * last[1]) / 2;
      if (EN) {
        out.innerHTML = (diverged
          ? "<p><b>Diverged</b>: the learning rate exceeds 2 / max curvature (about " + (2 / cond).toFixed(2) + " here), so every step is amplified.</p>"
          : "<p>After " + (n - 1) + " steps, loss <b>" + loss.toExponential(1) + "</b>" +
            (loss < 1e-3 ? " (converged)" : loss < 1 ? " (still going down)" : " (barely moved)") + "</p>") +
          '<p class="aw-note">The flatter the ellipse (the larger the condition number), the more the path oscillates along the steep direction and the slower it moves along the flat one. ' +
          "That is why we normalize (make the surface rounder), use momentum (cancel the oscillation) and adapt step sizes per dimension like Adam. The largest curvature caps the learning rate: above 2 / L it always diverges.</p>";
        return;
      }
      out.innerHTML = diverged
        ? '<p><b>发散了</b>：学习率超过了 2 / 最大曲率（这里约 ' + (2 / cond).toFixed(2) + '），每一步都被放大。</p>'
        : "<p>" + (n - 1) + " 步之后，损失 <b>" + loss.toExponential(1) + "</b>" +
          (loss < 1e-3 ? "（收敛）" : loss < 1 ? "（还在往下走）" : "（几乎没动）") + "</p>";
      out.innerHTML += '<p class="aw-note">椭圆越扁（条件数越大），沿陡方向容易震荡、沿平方向走得慢——这就是为什么要做归一化' +
        '（把曲面拉圆）、用动量（把震荡抵消）、以及 Adam 那样按维度调步长。学习率的上界由最大曲率决定：超过 2 / L 必然发散。</p>';
    }
    function stop() { if (timer) { clearInterval(timer); timer = null; box.querySelector('[data-k="play"]').textContent = zhen("播放", "Play"); } }
    box.querySelector('[data-k="play"]').addEventListener("click", function () {
      if (timer) { stop(); shown = 999; run(); return; }
      this.textContent = zhen("暂停", "Pause");
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
    var MODE_EN = { "连续且对齐 a[k]": "contiguous and aligned, a[k]", "连续但偏移一个 float a[k+1]": "contiguous but off by one float, a[k+1]",
                    "跨步 2：a[2k]": "stride 2: a[2k]", "跨步 8：a[8k]": "stride 8: a[8k]",
                    "跨步 32：按列读二维数组": "stride 32: a 2-D array read by column", "所有线程读同一个地址 a[0]": "every thread reads a[0]" };
    box.innerHTML = '<div class="aw-title">' + zhen("合并访问：一个 warp 要把多少字节搬过总线", "Coalescing: how many bytes a warp moves across the bus") + '</div><div class="aw-grid">' +
      row(zhen("访问模式", "access pattern"), select("mode", opts(Object.keys(MODES), MODE_EN), "跨步 2：a[2k]"), true) +
      row(zhen("跨步（float 个数）", "stride (in floats)"), range2("stride", 2, 0, 32)) + row(zhen("起始偏移", "starting offset"), range2("off", 0, 0, 8)) + '</div>' +
      '<svg class="aw-chart aw-co" viewBox="0 0 560 168"></svg><div class="aw-out"></div>';
    var svg = box.querySelector("svg"), out = box.querySelector(".aw-out");
    box.querySelector('[data-k="mode"]').addEventListener("change", function () {
      var m = MODES[val(box, "mode")];
      input(box, "stride").value = m[0]; input(box, "off").value = m[1];
      draw();
    });
    function draw() {
      var stride = val(box, "stride"), off = val(box, "off");
      show(box, "stride", stride === 0 ? zhen("同一地址", "same address") : String(stride)); show(box, "off", String(off));
      var addr = [], used = {}, sec = {}, i;
      for (i = 0; i < 32; i++) { var a = (off + (stride ? i * stride : 0)) * 4; addr.push(a); used[a / 4] = 1; sec[Math.floor(a / 32)] = 1; }
      var secs = Object.keys(sec).map(Number).sort(function (p, q) { return p - q; });
      var cw = 6, sw = 8 * cw, pad = 6, perRow = 10;         // 只画真正被触及的扇区
      var list = secs.slice(0, 20), rows = Math.ceil(list.length / perRow);
      var S = svgText(20, 14, zhen("每个格子 = 4 字节的 float，每个框 = 一次 32 字节的扇区传输",
                                    "one cell = a 4-byte float, one box = one 32-byte sector transfer"), "start");
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
      if (secs.length > list.length) S += svgText(20 + perRow * (sw + pad) - 4, 32 + rows * 50 + 12,
        zhen("… 共 " + secs.length + " 个扇区", "… " + secs.length + " sectors in all"), "end");
      var ly = 32 + rows * 50 + 22;
      S += '<rect x="20" y="' + ly + '" width="10" height="10" class="aw-on"/>' + svgText(36, ly + 6, zhen("线程真正要的字节", "bytes the threads want"), "start");
      S += '<rect x="' + (EN ? 190 : 170) + '" y="' + ly + '" width="10" height="10" class="aw-b"/>' + svgText((EN ? 206 : 186), ly + 6, zhen("被一起搬上来、但没人用", "fetched along, used by nobody"), "start");
      svg.setAttribute("viewBox", "0 0 560 " + (ly + 22));
      svg.innerHTML = S;
      var moved = secs.length * 32, useful = stride === 0 ? 4 : 128;
      var pct = (100 * useful / moved).toFixed(1);
      out.innerHTML = (EN
        ? "<p><b>" + secs.length + "</b> sectors touched, so <b>" + moved + " bytes</b> crossed the bus and " + useful +
          " were actually used, a bandwidth utilization of <b>" + pct + "%</b>" +
          (stride === 0 ? " (the hardware broadcasts, so one transfer and nothing wasted)" : "") + "</p>" +
          '<p class="aw-note">Coalescing is entirely this picture: have neighbouring threads read neighbouring addresses and 32 threads fill exactly 4 sectors. ' +
          'The larger the stride, the fewer bytes of each sector get used, and measured bandwidth falls in the same proportion. Walking a 2-D array down a column with threadIdx.x makes the stride the row width.</p>'
        : "<p>触及 <b>" + secs.length + "</b> 个扇区 → 总线上搬了 <b>" + moved + " 字节</b>，真正用上 " + useful +
          " 字节，带宽利用率 <b>" + pct + "%</b>" +
          (stride === 0 ? "（硬件会广播，只有一次传输，不算浪费）" : "") + "</p>" +
          '<p class="aw-note">合并访问的全部内容就是这张图：让相邻线程读相邻地址，32 个线程正好铺满 4 个扇区。' +
          '跨步越大，每个扇区里被用上的字节越少，实测带宽就按同样的比例掉下去。二维数组里让 threadIdx.x 沿列走，就是跨步 = 行宽。</p>');
    }
    bind(box, draw);
  }

  // ---------------------------------------------------------------- 共享内存：bank 冲突
  function bankconf(box) {
    var MODES = { "按行访问 tile[0][tid]": 1, "按列访问 tile[tid][0]，行宽 32": 32,
                  "按列访问 tile[tid][0]，行宽 33（padding）": 33, "跨步 2": 2, "跨步 4": 4,
                  "所有线程读同一个字（广播）": 0 };
    var MODE_EN = { "按行访问 tile[0][tid]": "by row, tile[0][tid]", "按列访问 tile[tid][0]，行宽 32": "by column, tile[tid][0], row width 32",
                    "按列访问 tile[tid][0]，行宽 33（padding）": "by column, tile[tid][0], row width 33 (padded)",
                    "跨步 2": "stride 2", "跨步 4": "stride 4", "所有线程读同一个字（广播）": "every thread reads one word (broadcast)" };
    box.innerHTML = '<div class="aw-title">' + zhen("bank 冲突：32 个线程撞在几个 bank 上", "Bank conflicts: how many banks 32 threads land on") + '</div><div class="aw-grid">' +
      row(zhen("访问模式", "access pattern"), select("mode", opts(Object.keys(MODES), MODE_EN), "按列访问 tile[tid][0]，行宽 32"), true) +
      row(zhen("相邻线程的字间隔", "words between neighbouring threads"), range2("stride", 32, 0, 36)) + '</div>' +
      '<svg class="aw-chart aw-bk" viewBox="0 0 560 220"></svg><div class="aw-out"></div>';
    var svg = box.querySelector("svg"), out = box.querySelector(".aw-out");
    box.querySelector('[data-k="mode"]').addEventListener("change", function () {
      input(box, "stride").value = MODES[val(box, "mode")]; draw();
    });
    function draw() {
      var stride = val(box, "stride");
      show(box, "stride", stride === 0 ? zhen("同一个字", "the same word") : String(stride));
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
      var S = svgText(x0, 14, zhen("横轴是 32 个 bank（每个 4 字节宽），往上堆的每个格子 = 落在这个 bank 上的一个线程",
                                    "across: the 32 banks (4 bytes each); each cell stacked above = one thread landing on that bank"), "start");
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
      S += svgText(x0, base + 44, zhen("bank 编号", "bank number"), "start");
      svg.innerHTML = S;
      out.innerHTML = (EN
        ? "<p>The busiest bank holds <b>" + worst + "</b> distinct words" +
          (stride === 0 ? ": every thread reads one address, the hardware broadcasts, so this is <b>not a conflict</b>" :
           worst === 1 ? ": the 32 threads land on 32 different banks, so <b>one cycle does it</b>" :
           ": a <b>" + worst + "-way conflict</b>, so this access splits into " + worst + " and takes " + worst + "× as long") + "</p>" +
          '<p class="aw-note">A bank number is simply "word index % 32". So as long as the words between neighbouring threads are coprime with 32 (an odd gap), all 32 banks are covered. ' +
          'That is exactly what the extra column in <code>__shared__ float tile[32][33]</code> does: it turns a gap of 32 into 33.</p>'
        : "<p>最忙的 bank 上有 <b>" + worst + "</b> 个不同的字" +
          (stride === 0 ? "：所有线程读同一个地址，硬件广播，<b>不算冲突</b>" :
           worst === 1 ? "：32 个线程落在 32 个不同的 bank，<b>一个周期就能完成</b>" :
           "：<b>" + worst + " 路冲突</b>，这条访存指令要拆成 " + worst + " 次，耗时 " + worst + " 倍") + "</p>" +
          '<p class="aw-note">bank 编号就是"字编号 % 32"。所以只要相邻线程的字间隔和 32 互质（间隔为奇数），就一定铺满 32 个 bank——' +
          '这正是 <code>__shared__ float tile[32][33]</code> 那个多出来的一列在做的事：把间隔从 32 变成 33。</p>');
    }
    bind(box, draw);
  }

  // ---------------------------------------------------------------- 前缀和：两种并行算法
  function scanviz(box) {
    var A = [3, 1, 7, 0, 4, 1, 6, 3, 2, 5, 1, 2, 0, 4, 3, 1], N = 16;
    function hillis() {                                      // Hillis-Steele：步数少、加法多
      var st = [{ v: A.slice(), e: [], t: zhen("初始值", "initial values") }], cur = A.slice(), adds = 0;
      for (var d = 1; d < N; d *= 2) {
        var nx = cur.slice(), e = [];
        for (var i = N - 1; i >= d; i--) { nx[i] = cur[i] + cur[i - d]; e.push([i - d, i]); adds++; }
        cur = nx; st.push({ v: cur.slice(), e: e, t: zhen("步 " + st.length + "：", "step " + st.length + ": ") + "x[i] += x[i−" + d + "]" });
      }
      return { steps: st, adds: adds, note: zhen("包含扫描（inclusive），" + (Math.log(N) / Math.log(2)) + " 步、" + adds + " 次加法",
               "an inclusive scan: " + (Math.log(N) / Math.log(2)) + " steps and " + adds + " additions") };
    }
    function blelloch() {                                    // Blelloch：两趟、加法少
      var cur = A.slice(), st = [{ v: cur.slice(), e: [], t: zhen("初始值", "initial values") }], adds = 0, d, i, e;
      for (d = 1; d < N; d *= 2) {                           // 上扫：求部分和
        var nx = cur.slice(); e = [];
        for (i = d * 2 - 1; i < N; i += d * 2) { nx[i] = cur[i] + cur[i - d]; e.push([i - d, i]); adds++; }
        cur = nx; st.push({ v: cur.slice(), e: e, t: zhen("上扫 " + Math.round(Math.log(d * 2) / Math.log(2)) + "：间隔 " + d,
                            "up-sweep " + Math.round(Math.log(d * 2) / Math.log(2)) + ": gap " + d) });
      }
      cur = cur.slice(); cur[N - 1] = 0;
      st.push({ v: cur.slice(), e: [], t: zhen("把最后一个换成 0", "replace the last with 0") });
      for (d = N / 2; d >= 1; d /= 2) {                      // 下扫：把部分和散回去
        var n2 = cur.slice(); e = [];
        for (i = d * 2 - 1; i < N; i += d * 2) { n2[i - d] = cur[i]; n2[i] = cur[i] + cur[i - d]; e.push([i - d, i]); adds++; }
        cur = n2; st.push({ v: cur.slice(), e: e, t: zhen("下扫：间隔 " + d, "down-sweep: gap " + d) });
      }
      return { steps: st, adds: adds, note: zhen("排除扫描（exclusive），2×log₂N 步、" + adds + " 次加法，工作量是 O(N)",
               "an exclusive scan: 2×log₂N steps, " + adds + " additions, and O(N) work") };
    }
    var ALGO = { "Hillis-Steele（步数少，加法多）": hillis(), "Blelloch（两趟扫，加法只有 O(N)）": blelloch() };
    var ALGO_EN = { "Hillis-Steele（步数少，加法多）": "Hillis-Steele (fewer steps, more additions)",
                    "Blelloch（两趟扫，加法只有 O(N)）": "Blelloch (two sweeps, only O(N) additions)" };
    box.innerHTML = '<div class="aw-title">' + zhen("并行前缀和：两种经典算法一步一步看", "Parallel prefix sum: two classic algorithms, step by step") + '</div><div class="aw-grid">' +
      row(zhen("算法", "algorithm"), select("algo", opts(Object.keys(ALGO), ALGO_EN), "Hillis-Steele（步数少，加法多）"), true) +
      row(zhen("第几步", "step"), range2("step", 0, 0, 4) + '<button type="button" data-k="play" class="aw-btn">' + zhen("播放", "Play") + '</button>') + '</div>' +
      '<svg class="aw-chart aw-sc" viewBox="0 0 560 140"></svg><div class="aw-out"></div>';
    var svg = box.querySelector("svg"), out = box.querySelector(".aw-out"), timer = null;
    function stop() { if (timer) { clearInterval(timer); timer = null; box.querySelector('[data-k="play"]').textContent = zhen("播放", "Play"); } }
    box.querySelector('[data-k="play"]').addEventListener("click", function () {
      if (timer) return stop();
      this.textContent = zhen("暂停", "Pause");
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
      out.innerHTML = (EN
        ? "<p>Step <b>" + si + "</b> did <b>" + st.e.length + "</b> additions (of " + a.adds + " in all). " + a.note + "</p>" +
          '<p class="aw-note">Hillis-Steele keeps every thread busy at every step, with few steps but O(N log N) additions in total; ' +
          'Blelloch gathers the partial sums upward and then scatters them down for only O(N) additions, at the price of two sweeps and ever sparser active threads. ' +
          'On a GPU the former is used within a warp (a shuffle is enough) and the latter at block and device level.</p>'
        : "<p>第 <b>" + si + "</b> 步，这一步做了 <b>" + st.e.length + "</b> 次加法（全部 " + a.adds + " 次）。" + a.note + "</p>" +
          '<p class="aw-note">Hillis-Steele 每一步所有线程都在干活，步数少但总加法是 O(N log N)；' +
          'Blelloch 先把部分和往上收、再往下散，总加法只有 O(N)，代价是要跑两趟、且活跃线程越来越稀疏。' +
          'GPU 上 warp 内用前者（shuffle 就够），block 和设备级用后者。</p>');
    }
    bind(box, function () { stop(); draw(); });
  }

  // ---------------------------------------------------------------- 占用率
  function occupancy(box) {
    var ARCH = {                                             // 每 SM：最大线程、最大 block、寄存器数、共享内存 KB、每 SM 最大 warp
      "A100（sm_80）": [2048, 32, 65536, 164, 64], "H100（sm_90）": [2048, 32, 65536, 228, 64],
      "RTX 4090（sm_89）": [1536, 24, 65536, 100, 48], "T4（sm_75）": [1024, 16, 65536, 64, 32]
    };
    var ARCH_EN = { "A100（sm_80）": "A100 (sm_80)", "H100（sm_90）": "H100 (sm_90)",
                    "RTX 4090（sm_89）": "RTX 4090 (sm_89)", "T4（sm_75）": "T4 (sm_75)" };
    box.innerHTML = '<div class="aw-title">' + zhen("占用率：哪一项资源先卡住你", "Occupancy: which resource runs out first") + '</div><div class="aw-grid">' +
      row("GPU", select("arch", opts(Object.keys(ARCH), ARCH_EN), "A100（sm_80）")) +
      row(zhen("block 大小", "block size"), range2("bs", 8, 1, 32)) +
      row(zhen("每线程寄存器", "registers per thread"), range2("regs", 40, 16, 128)) +
      row(zhen("每 block 共享内存 KB", "shared memory per block, KB"), range2("smem", 16, 0, 164)) + '</div>' +
      '<svg class="aw-chart aw-oc" viewBox="0 0 560 150"></svg><div class="aw-out"></div>';
    var svg = box.querySelector("svg"), out = box.querySelector(".aw-out");
    bind(box, function () {
      var a = ARCH[val(box, "arch")], maxT = a[0], maxB = a[1], maxR = a[2], maxS = a[3], maxW = a[4];
      var bs = val(box, "bs") * 32, regs = val(box, "regs"), smem = Math.min(val(box, "smem"), maxS);
      show(box, "bs", bs + zhen(" 线程", " threads")); show(box, "regs", String(regs)); show(box, "smem", smem + " KB");
      var wpb = bs / 32;
      var regPerWarp = Math.ceil(regs * 32 / 256) * 256;      // 寄存器按 warp、每 256 个为一档分配
      var byThread = Math.floor(maxT / bs), byBlock = maxB;
      var byReg = Math.floor(maxR / (regPerWarp * wpb));
      var bySmem = smem > 0 ? Math.floor(maxS / smem) : 99;
      var limits = [[zhen("线程数上限", "thread limit"), byThread], [zhen("block 数上限", "block limit"), byBlock],
                    [zhen("寄存器", "registers"), byReg], [zhen("共享内存", "shared memory"), bySmem]];
      var blocks = Math.min(byThread, byBlock, byReg, bySmem);
      var warps = Math.min(blocks * wpb, maxW), occ = warps / maxW;
      var x0 = EN ? 128 : 118, W = EN ? 330 : 380, S = "", top = 16;
      var mx = Math.max(4, Math.min(12, Math.max.apply(null, limits.map(function (l) { return Math.min(l[1], 12); }))));
      limits.forEach(function (l, i) {
        var y = top + i * 26, n = Math.min(l[1], mx), bind_ = l[1] === blocks;
        S += svgText(x0 - 10, y + 9, l[0], "end");
        S += '<rect x="' + x0 + '" y="' + y + '" width="' + (n / mx * W) + '" height="18" rx="3" class="' + (bind_ ? "aw-b" : "aw-on") + '"/>';
        S += svgText(x0 + n / mx * W + 8, y + 9,
          (l[1] > 90 ? zhen("不限", "no limit") : zhen(l[1] + " 个 block", l[1] + " blocks")) + (bind_ ? zhen("  ← 瓶颈", "  ← the limit") : ""), "start");
      });
      var yb = top + 4 * 26 + 12;
      S += svgText(x0 - 10, yb + 10, zhen("实际占用率", "actual occupancy"), "end");
      S += '<rect x="' + x0 + '" y="' + yb + '" width="' + W + '" height="20" rx="3" class="aw-off"/>';
      S += '<rect x="' + x0 + '" y="' + yb + '" width="' + (occ * W) + '" height="20" rx="3" class="aw-on"/>';
      S += svgText(x0 + W + 8, yb + 10, (100 * occ).toFixed(0) + "%", "start");
      svg.innerHTML = S;
      var kregs = (regPerWarp * wpb / 1024).toFixed(1), pc = (100 * occ).toFixed(0);
      out.innerHTML = (EN
        ? "<p>Each SM can hold <b>" + blocks + "</b> blocks and <b>" + warps + "</b> warps at once, an occupancy of <b>" +
          pc + "%</b> (the ceiling is " + maxW + " warps). Each block wants " + kregs + "K registers and " + smem + " KB of shared memory.</p>" +
          '<p class="aw-note">Higher occupancy is not automatically better: what it buys is the ability to hide memory latency behind other warps. ' +
          'A memory-bound kernel wants high occupancy, while a compute-bound one often prefers spending registers on register tiling and runs faster at 25%. ' +
          'Read the Nsight Compute report first to see whether latency is exposed or the math units are starved, and only then touch these numbers.</p>'
        : "<p>每个 SM 能同时驻留 <b>" + blocks + "</b> 个 block、<b>" + warps + "</b> 个 warp，占用率 <b>" +
          pc + "%</b>（上限 " + maxW + " 个 warp）。每个 block 要 " + kregs + "K 个寄存器、" + smem + " KB 共享内存。</p>" +
          '<p class="aw-note">占用率不是越高越好：它买的是"用别的 warp 盖住访存延迟"的能力。' +
          '访存密集的 kernel 需要高占用率；而计算密集的 kernel 往往宁可多用寄存器做寄存器分块，占用率 25% 反而更快。' +
          '先看 Nsight Compute 报告里到底是延迟没盖住还是算力没喂饱，再决定要不要调这几个数。</p>');
    });
  }

  // ---------------------------------------------------------------- 多级页表的地址翻译
  function pagewalk(box) {
    var PAGES = { "4 KiB（四级页表）": 4096, "2 MiB（大页，三级）": 2 * 1024 * 1024, "1 GiB（大页，两级）": 1024 * 1024 * 1024 };
    var PAGES_EN = { "4 KiB（四级页表）": "4 KiB (four levels)", "2 MiB（大页，三级）": "2 MiB (huge page, three levels)",
                     "1 GiB（大页，两级）": "1 GiB (huge page, two levels)" };
    var LV = ["PML4（第 4 级）", "PDPT（第 3 级）", "PD（第 2 级）", "PT（第 1 级）"];
    box.innerHTML = '<div class="aw-title">' + zhen("地址翻译：一个虚拟地址是怎么变成物理地址的",
        "Address translation: how a virtual address becomes a physical one") + '</div><div class="aw-grid">' +
      row(zhen("虚拟地址", "virtual address"), '<input type="text" data-k="va" value="0x7F3A1C2D5E6F" spellcheck="false">' +
          '<button type="button" data-k="rand" class="aw-btn">' + zhen("换一个", "another") + '</button>', true) +
      row(zhen("页大小", "page size"), select("page", opts(Object.keys(PAGES), PAGES_EN), "4 KiB（四级页表）"), true) + '</div>' +
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
      for (i = 0; i < levels; i++) { segs.push([9, LV[i].split("（")[0] + zhen(" 下标", " index"), idx[i], "aw-on"]); bits += 9; }
      segs.push([offBits, zhen("页内偏移", "offset in page"), "0x" + off.toString(16), "aw-b"]);
      var total = bits + offBits, cx = x0;
      S += svgText(x0, 14, zhen("48 位虚拟地址 0x" + va.toString(16).toUpperCase() + " 的拆法",
        "how the 48-bit virtual address 0x" + va.toString(16).toUpperCase() + " is split"), "start");
      segs.forEach(function (sg) {
        var w = sg[0] / total * W;
        S += '<rect x="' + cx.toFixed(1) + '" y="24" width="' + (w - 2).toFixed(1) + '" height="26" rx="3" class="' + sg[3] + '"/>';
        S += svgText(cx + w / 2 - 1, 37, sg[0] + zhen(" 位", " bits"), "middle");
        S += svgText(cx + w / 2 - 1, 62, sg[2], "middle");
        cx += w;
      });
      // 查表链路
      var by = 100, bw = 96, gapx = (W - (levels + 1) * bw) / levels;
      S += svgText(x0, 86, zhen("页表遍历：从 CR3 出发，每一级用一个 9 位下标找下一级",
        "the page walk: start at CR3, and each level uses a 9-bit index to find the next"), "start");
      for (i = 0; i <= levels; i++) {
        var x = x0 + i * (bw + gapx), last = i === levels;
        S += '<rect x="' + x.toFixed(1) + '" y="' + by + '" width="' + bw + '" height="46" rx="5" class="' + (last ? "aw-b" : "aw-blk") + '"/>';
        S += svgText(x + bw / 2, by + 17, last ? zhen("物理页", "physical page") : LV[i].split("（")[0], "middle");
        S += svgText(x + bw / 2, by + 33, last ? zhen("+ 偏移", "+ offset") : "[" + idx[i] + "]", "middle");
        if (i < levels) {
          var xa = x + bw, xb = x + bw + gapx;
          S += '<line x1="' + xa + '" y1="' + (by + 23) + '" x2="' + (xb - 5) + '" y2="' + (by + 23) + '" class="aw-axis"/>';
          S += '<polygon points="' + xb + ',' + (by + 23) + ' ' + (xb - 7) + ',' + (by + 19) + ' ' + (xb - 7) + ',' + (by + 27) + '" class="aw-dot"/>';
        }
      }
      var cov = [512 * 512 * 512 * 4096, 512 * 512 * 4096, 512 * 4096, 4096];
      S += svgText(x0, 176, zhen("每一级表项管多大地址：", "how much address space one entry at each level covers:"), "start");
      for (i = 0; i < levels; i++) {
        var x2 = x0 + i * (bw + gapx), c = cov[4 - levels + i];
        S += svgText(x2 + bw / 2, 196, c >= 1e9 ? (c / 1024 / 1024 / 1024) + " GiB" : c >= 1e6 ? (c / 1024 / 1024) + " MiB" : (c / 1024) + " KiB", "middle");
      }
      S += svgText(x0, 222, zhen("一张页表 512 项 × 8 字节 = 4096 字节，正好一页",
        "one page table is 512 entries x 8 bytes = 4096 bytes, exactly one page"), "start");
      svg.innerHTML = S;
      out.innerHTML = EN
        ? "<p>On a TLB miss this access walks the table <b>" + levels + "</b> times before fetching the data, <b>" + (levels + 1) +
          "</b> memory accesses in all; on a TLB hit it takes 1.</p>" +
          '<p class="aw-note">A huge page simply cuts the bottom levels off: a 2 MiB page walks one level fewer, the offset grows from 12 bits to 21, ' +
          'and one TLB entry covers 2 MiB instead of 4 KiB, so the TLB hit rate rises with it. That is why pinning GPU memory and registering large ' +
          'buffers for RDMA both favour huge pages. PagedAttention uses the same idea: a table between contiguous logical addresses and scattered physical blocks.</p>'
        : "<p>TLB 未命中时，这一次访问要先做 <b>" + levels + "</b> 次查表、再取数据，一共 <b>" + (levels + 1) +
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
    box.innerHTML = '<div class="aw-title">' + zhen("组相联缓存：地址落进哪一组，什么时候开始互相踢",
        "A set-associative cache: which set an address lands in, and when they start evicting each other") + '</div><div class="aw-grid">' +
      row(zhen("缓存容量 KB", "cache size KB"), range2("cap", 32, 1, 64)) + row(zhen("相联度（路数）", "associativity (ways)"), range2("ways", 8, 1, 16)) +
      row(zhen("访问跨步（字节）", "access stride (bytes)"), range2("stride", 11, 6, 16)) + '</div>' +
      '<svg class="aw-chart aw-cm" viewBox="0 0 560 190"></svg><div class="aw-out"></div>';
    var svg = box.querySelector("svg"), out = box.querySelector(".aw-out");
    bind(box, function () {
      var LINE = 64, cap = val(box, "cap") * 1024, ways = val(box, "ways"), stride = Math.pow(2, val(box, "stride"));
      var nsets = Math.max(1, Math.round(cap / (ways * LINE)));
      show(box, "cap", (cap / 1024) + " KB"); show(box, "ways", ways + zhen(" 路", "-way"));
      show(box, "stride", stride >= 1024 ? (stride / 1024) + " KB" : stride + " B");
      var idxBits = Math.round(Math.log(nsets) / Math.log(2)), offBits = 6, tagBits = 48 - idxBits - offBits;
      var x0 = 22, W = 516, segs = [[tagBits, "tag", "aw-off"], [idxBits, zhen("组号", "set"), "aw-on"], [offBits, zhen("偏移", "offset"), "aw-b"]];
      var S = svgText(x0, 14, zhen("地址怎么拆：低 6 位选行内字节，中间 " + idxBits + " 位选组，剩下的是 tag",
        "how the address splits: the low 6 bits pick the byte in the line, the middle " + idxBits + " pick the set, the rest is the tag"), "start"), cx = x0;
      segs.forEach(function (sg) {
        var w = sg[0] / 48 * W;
        S += '<rect x="' + cx.toFixed(1) + '" y="22" width="' + (w - 2).toFixed(1) + '" height="30" rx="3" class="' + sg[2] + '"/>';
        S += svgText(cx + w / 2 - 1, 32, sg[1], "middle");
        S += svgText(cx + w / 2 - 1, 46, sg[0] + zhen(" 位", " bits"), "middle");
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
      S += svgText(x0, 74, zhen("按这个跨步连续访问 64 次，落进了哪些组（只画前 " + SHOWN + " 组）",
        "64 accesses at this stride, and the sets they land in (only the first " + SHOWN + " shown)"), "start");
      for (k = 0; k < SHOWN; k++) {
        var n = hit[k] || 0, h = n ? Math.min(ways, n) / ways * 56 : 0;
        S += '<rect x="' + (x0 + k * cw).toFixed(1) + '" y="88" width="' + Math.max(1, cw - 1).toFixed(1) + '" height="56" class="aw-off"/>';
        if (n) S += '<rect x="' + (x0 + k * cw).toFixed(1) + '" y="' + (144 - h).toFixed(1) + '" width="' + Math.max(1, cw - 1).toFixed(1) +
                    '" height="' + h.toFixed(1) + '" class="' + (n > ways ? "aw-b" : "aw-on") + '"/>';
      }
      S += svgText(x0, 162, zhen("组 0", "set 0"), "start") + svgText(x0 + W, 162, zhen("组 ", "set ") + (SHOWN - 1), "end");
      S += svgText(x0 + W / 2, 182, zhen("柱子高度 = 这一组里挤了几条不同的缓存行（满格 = " + ways + " 路）",
        "bar height = distinct cache lines crowded into this set (full = " + ways + " ways)"), "middle");
      svg.innerHTML = S;
      var span = nsets * LINE >= 1024 ? (nsets * LINE / 1024) + " KB" : nsets * LINE + " B";
      out.innerHTML = EN
        ? "<p>" + (cap / 1024) + " KB and " + ways + " ways gives <b>" + nsets + " sets</b> of " + ways +
          " lines each. Addresses a multiple of <b>" + span + "</b> apart always land in the same set. These 64 accesses used only <b>" +
          touched + "</b> sets, and the most crowded one holds <b>" + worst + "</b> distinct lines" +
          (worst > ways ? ", <b>past " + ways + " ways, so they start evicting each other (conflict misses)</b>" : ", which still fits") + ".</p>" +
          '<p class="aw-note">This is the mechanism behind walking a matrix by column being slow: when the row stride is a power of two, every element of ' +
          'a column lands in the same set, and a line that was read in is evicted before it is used a second time. Pad each row by one cache line ' +
          '(making the stride an odd multiple) and the addresses spread over all the sets, the same trick as padding a shared-memory array by one column to dodge bank conflicts.</p>'
        : "<p>" + (cap / 1024) + " KB、" + ways + " 路 → <b>" + nsets + " 组</b>，每组 " + ways +
          " 行。相距 <b>" + span +
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
    box.innerHTML = '<div class="aw-title">' + zhen("一致性哈希：加一台机器，要搬多少数据",
        "Consistent hashing: how much data moves when a machine is added") + '</div><div class="aw-grid">' +
      row(zhen("节点数", "nodes"), range2("n", 6, 2, 12)) + row(zhen("每节点虚拟节点数", "virtual nodes each"), range2("vn", 60, 1, 200)) +
      row(zhen("动作", "action"), select("act", opts(["不动", "加一台新机器", "挂掉一台机器"],
          {"不动": "no change", "加一台新机器": "add a machine", "挂掉一台机器": "lose a machine"}), "加一台新机器"), true) + '</div>' +
      '<svg class="aw-chart aw-hr" viewBox="0 0 560 230"></svg><div class="aw-out"></div>';
    var svg = box.querySelector("svg"), out = box.querySelector(".aw-out");
    var PAL = ["#007aff", "#f08c00", "#34c759", "#af52de", "#ff3b30", "#00b8d4", "#ffb300", "#8e8e93",
               "#5856d6", "#ff2d55", "#30b0c7", "#a2845e", "#66bb6a"];
    bind(box, function () {
      var n = val(box, "n"), vn = val(box, "vn"), act = val(box, "act");
      show(box, "n", zhen(n + " 台", String(n))); show(box, "vn", String(vn));
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
      S += svgText(cx, cy - 8, zhen(after.length + " 台机器", after.length + " machines"), "middle");
      S += svgText(cx, cy + 10, zhen(B.ring.length + " 个环上的点", B.ring.length + " points on the ring"), "middle");
      // 右边：每台机器分到多少 key
      var x0 = 250, W = 280, bh = 15, ideal = KEYS.length / after.length, mx = 0;
      after.forEach(function (m) { mx = Math.max(mx, cnt[m] || 0); });
      S += svgText(x0, 16, zhen("每台机器分到的 key 数（虚线 = 完全均匀）",
        "keys per machine (the dashed line = perfectly even)"), "start");
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
      var ACT_EN = { "不动": "no change", "加一台新机器": "adding a machine", "挂掉一台机器": "losing a machine" };
      var pct = (100 * moved / KEYS.length).toFixed(1), theory = (100 / Math.max(after.length, before.length)).toFixed(1);
      out.innerHTML = EN
        ? "<p>" + (act === "不动" ? "Nothing changed, so " : "After " + ACT_EN[act] + ", ") + "<b>" + pct +
          "%</b> of the keys changed owner" + (act === "不动" ? "" : " (the theoretical figure is about " + theory + "%)") +
          "; the heaviest machine holds <b>" + (hi2 / Math.max(1, lo)).toFixed(2) + "x</b> what the lightest does.</p>" +
          '<p class="aw-note">Modulo sharding moves almost all of the data when a machine is added; consistent hashing moves only 1/N. ' +
          'But with one point per machine the gaps on the ring are very uneven (try pulling the virtual-node count down to 1), ' +
          'so in practice each machine gets tens to hundreds of virtual nodes and the load flattens out. Routing by prefix hash to one machine in an inference system uses exactly this.</p>'
        : "<p>" + (act === "不动" ? "没有变动，" : act + "之后，") + "有 <b>" + pct +
          "%</b> 的 key 换了归属" + (act === "不动" ? "" : "（理论值约 " + theory + "%）") +
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
    box.innerHTML = '<div class="aw-title">' + zhen("分页 KV Cache：块大小怎么影响浪费", "The paged KV Cache: how block size affects waste") + '</div><div class="aw-grid">' +
      row(zhen("块大小（token）", "Block size (tokens)"), range2("bs", 2, 0, 5)) + row(zhen("请求数", "Requests"), range2("n", 10, 2, 12)) +
      row(zhen("场景", "Scenario"), select("mode", opts(["各自独立", "共享同一段长前缀（多轮对话）"], {"各自独立": "independent", "共享同一段长前缀（多轮对话）": "sharing one long prefix (multi-turn)"}), "各自独立"), true) + '</div>' +
      '<svg class="aw-chart aw-pk" viewBox="0 0 560 230"></svg><div class="aw-out"></div>';
    var svg = box.querySelector("svg"), out = box.querySelector(".aw-out");
    bind(box, function () {
      var bs = SIZES[val(box, "bs")], n = val(box, "n"), shared = val(box, "mode").indexOf("共享") === 0;
      show(box, "bs", String(bs)); show(box, "n", zhen(n + " 个", String(n)));
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
      var SHOW = Math.min(n, 12), S = svgText(x0, 12, zhen("每个小格 = 一个 " + bs + " token 的物理块；浅色 = 块里没用满的部分", "each cell = one physical block of " + bs + " tokens; the light part is the unused tail"), "start");
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
      S += '<rect x="' + x0 + '" y="' + ly + '" width="10" height="10" class="aw-on"/>' + svgText(x0 + 16, ly + 6, zhen("用满的块", "full blocks"), "start");
      S += '<rect x="' + (x0 + 110) + '" y="' + ly + '" width="10" height="10" class="aw-b"/>' + svgText(x0 + 126, ly + 6, zhen("最后一个块（平均浪费半块）", "the last block (half wasted on average)"), "start");
      if (shared) S += '<rect x="' + (x0 + 330) + '" y="' + ly + '" width="10" height="10" class="aw-f"/>' + svgText(x0 + 346, ly + 6, zhen("共享前缀，只分配一份", "a shared prefix, allocated once"), "start");
      svg.setAttribute("viewBox", "0 0 560 " + (ly + 26));
      svg.innerHTML = S;
      var alloc = blocks * bs, allocNoShare = blocksNoShare * bs, reserve = n * 4096;
      out.innerHTML = "<p>" + zhen("这些请求逻辑上一共 <b>", "these requests hold <b>") + fmtInt(useful) + zhen("</b> 个 token，分页之后占 <b>", "</b> tokens logically and take <b>") + fmtInt(allocNoShare) +
        zhen("</b> 个槽位，<b>", "</b> slots once paged, <b>") + (100 * useful / allocNoShare).toFixed(1) + zhen("%</b> 是有效的（差的那点就是每个请求最后一个块没用满）。", "%</b> of it useful (the difference is each request's unfilled last block). ") +
        zhen("同样这些请求按最大长度 4096 预留要 " + fmtInt(reserve) + " 个槽位，有效率只有 ", "Reserving by a maximum length of 4096 would take " + fmtInt(reserve) + " slots, only ") +
        (100 * useful / reserve).toFixed(1) + zhen("%。", "% useful.") + "</p>" +
        (shared ? "<p>" + zhen("共享前缀只存一份之后，实际只占 <b>", "with the shared prefix stored once, it takes only <b>") + fmtInt(alloc) + zhen("</b> 个槽位（", "</b> slots (") + blocks + zhen(" 个块），比每人存一份再省 <b>", " blocks), saving another <b>") + (100 * (1 - alloc / allocNoShare)).toFixed(0) + "%</b>.</p>" : "") +
        '<p class="aw-note">' + zhen('块越小越省（浪费只剩最后半个块），但块表更长、kernel 每次要处理的块更碎；块越大越省调度开销，却把内部碎片放大。实践中 16 是个常见折中。切到"共享长前缀"看另一半故事：多轮对话里系统提示 + 历史是所有请求共用的，分页让它们指向同一批物理块，这就是前缀缓存能省下大量显存和 prefill 的前提。', 'Smaller blocks waste less (only half a block at the end) but make block tables longer and the blocks a kernel handles more fragmented; larger blocks save scheduling overhead while magnifying internal fragmentation. In practice 16 is a common compromise. Switch to "sharing one long prefix" for the other half of the story: in multi-turn chat the system prompt and history are shared by every request, and paging lets them point at the same physical blocks, which is what makes prefix caching save so much memory and prefill.') + '</p>';
    });
  }

  // ---------------------------------------------------------------- 前缀缓存：基数树
  function radixcache(box) {
    var SYS = zhen("系统提示", "system"), EX = zhen("长示例", "examples"), Q = zhen("问题", "question"), A = zhen("回答", "answer"), REQ = zhen("请求", "request");
    var SCEN = {};
    SCEN[zhen("多轮对话（越聊前缀越长）", "multi-turn chat (the prefix grows)")] = [
        [[SYS, 40], [Q + " A", 25]],
        [[SYS, 40], [Q + " A", 25], [A + " A", 80], [Q + " B", 20]],
        [[SYS, 40], [Q + " A", 25], [A + " A", 80], [Q + " B", 20], [A + " B", 70], [Q + " C", 22]],
        [[SYS, 40], [Q + " D", 30]],
        [[SYS, 40], [Q + " D", 30], [A + " D", 60], [Q + " E", 18]]
      ];
    SCEN[zhen("few-shot：同一段长示例 + 不同问题", "few-shot: one long example + different questions")] = [
        [[EX, 320], [Q + " 1", 25]], [[EX, 320], [Q + " 2", 30]],
        [[EX, 320], [Q + " 3", 22]], [[EX, 320], [Q + " 4", 28]],
        [[EX, 320], [Q + " 5", 26]]
      ];
    SCEN[zhen("互不相关的请求", "unrelated requests")] = [
        [[REQ + " 1", 120]], [[REQ + " 2", 150]], [[REQ + " 3", 90]], [[REQ + " 4", 200]], [[REQ + " 5", 110]]
      ];
    box.innerHTML = '<div class="aw-title">' + zhen("前缀缓存：基数树是怎么长出来的", "Prefix caching: how the radix tree grows") + '</div><div class="aw-grid">' +
      row(zhen("请求序列", "Request sequence"), select("scen", Object.keys(SCEN), Object.keys(SCEN)[0]), true) +
      row(zhen("已经来了几个请求", "Requests so far"), range2("k", 3, 1, 5) + '<button type="button" data-k="play" class="aw-btn">' + zhen("播放", "Play") + '</button>') + '</div>' +
      '<svg class="aw-chart aw-rx" viewBox="0 0 560 240"></svg><div class="aw-out"></div>';
    var svg = box.querySelector("svg"), out = box.querySelector(".aw-out"), timer = null;
    function stop() { if (timer) { clearInterval(timer); timer = null; box.querySelector('[data-k="play"]').textContent = zhen("播放", "Play"); } }
    box.querySelector('[data-k="play"]').addEventListener("click", function () {
      if (timer) return stop();
      this.textContent = zhen("暂停", "Pause");
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
      S += '<circle cx="' + x0 + '" cy="' + root.y + '" r="8" class="aw-dot"/>' + svgText(x0, root.y + 22, zhen("根", "root"), "middle");
      S += '<rect x="330" y="' + (H - 22) + '" width="10" height="10" class="aw-f"/>' + svgText(346, H - 16, zhen("这次命中的", "hit this time"), "start");
      S += '<rect x="440" y="' + (H - 22) + '" width="10" height="10" class="aw-b"/>' + svgText(456, H - 16, zhen("这次新增的", "added this time"), "start");
      svg.setAttribute("viewBox", "0 0 560 " + H);
      svg.innerHTML = S;
      var last = reqs[k - 1], lastTot = 0, lastHit = 0, cur = root, stillHit = true;
      last.forEach(function (sg) {                            // 最后一个请求：前缀匹配到哪里为止
        lastTot += sg[1];
        var c = cur.ch[sg[0]];
        if (stillHit && c && c.added < k - 1) lastHit += sg[1]; else stillHit = false;
        cur = c || cur;
      });
      out.innerHTML = "<p>" + zhen("第 " + k + " 个请求一共 <b>", "request " + k + " has <b>") + lastTot + zhen("</b> 个 token，其中 <b>", "</b> tokens, of which <b>") + lastHit +
        zhen("</b> 个在缓存里命中（", "</b> hit the cache (") + (100 * lastHit / lastTot).toFixed(0) + zhen("%），只有剩下的 ", "%), leaving only ") + (lastTot - lastHit) +
        zhen(" 个需要真的做 prefill。前 " + k + " 个请求累计命中率 <b>", " to actually prefill. Over the first " + k + " requests the cumulative hit rate is <b>") + (100 * hitTok / totTok).toFixed(0) + "%</b>.</p>" +
        '<p class="aw-note">' + zhen('树的每条边是一段 token 和它们的 KV 槽位，从根到某个节点的路径就是一个缓存过的前缀。新请求沿树往下匹配，匹配不上的地方把边劈开、长出新枝。多轮对话和 few-shot 是最划算的两种形态：前者越聊共享前缀越长，后者几百个 token 的示例被所有请求共用。互不相关的请求则完全没得省。', "Each edge of the tree is a run of tokens and their KV slots, and the path from the root to a node is one cached prefix. A new request matches down the tree, splitting an edge and growing a branch where it diverges. Multi-turn chat and few-shot are the two best shapes: the former's shared prefix grows as the conversation goes on, and the latter's few hundred tokens of examples are shared by every request. Unrelated requests save nothing at all.") + '</p>';
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
    box.innerHTML = '<div class="aw-title">' + zhen("调度：静态批、连续批与分块 prefill 的差别", "Scheduling: static batching, continuous batching and chunked prefill") + '</div><div class="aw-grid">' +
      row(zhen("调度方式", "Scheduling"), select("mode", opts(MODES, { "静态批处理（攒满一批再跑）": "static batching (wait for a full batch)", "连续批处理": "continuous batching", "连续批处理 + 分块 prefill": "continuous batching + chunked prefill" }), "连续批处理 + 分块 prefill"), true) +
      row(zhen("每步 token 预算", "Token budget per step"), range2("budget", 2048, 256, 4096)) + '</div>' +
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
      var S = svgText(x0, 12, zhen("横轴 = 第几次前向，每一格是这一步里这个请求干的事", "across = forward pass number; each cell is what the request did in that step"), "start");
      st.forEach(function (r, i) {
        var y = 22 + i * rh;
        S += svgText(x0 - 8, y + 9, zhen("请求 ", "req ") + (i + 1), "end");
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
      S += '<rect x="' + (x0 + 90) + '" y="' + ly + '" width="10" height="10" class="aw-on"/>' + svgText(x0 + 106, ly + 6, zhen("decode（一步一个 token）", "decode (one token per step)"), "start");
      S += '<rect x="' + (x0 + 290) + '" y="' + ly + '" width="10" height="10" class="aw-off"/>' + svgText(x0 + 306, ly + 6, zhen("没在这一批里", "not in this batch"), "start");
      svg.setAttribute("viewBox", "0 0 560 " + (ly + 26));
      svg.innerHTML = S;
      var ttfts = st.map(function (r) { return r.ttft < 0 ? steps : r.ttft - r.arr; });
      var maxT = Math.max.apply(null, ttfts), avgT = ttfts.reduce(function (a, b) { return a + b; }, 0) / ttfts.length;
      var busy = trace.reduce(function (a, c) { return a + (c.length ? 1 : 0); }, 0);
      if (EN) {
        out.innerHTML = "<p>All " + st.length + " requests finish in <b>" + steps + "</b> steps; time to first token averages <b>" +
          avgT.toFixed(1) + "</b> steps, worst <b>" + maxT + "</b>.</p>" +
          '<p class="aw-note">Static batching waits for the whole batch to finish before swapping requests, so short requests wait alongside the longest one (see the long blank after request 2); ' +
          'continuous batching lets finished requests leave at once and new ones fill in at once. But as long as a prefill takes a step to itself, ' +
          'requests in decode get stuck behind a long prompt (TPOT jitter); chunked prefill cuts long prompts into pieces ' +
          'and mixes them into the same steps as decode, at the cost of a slightly slower prefill. The token budget is the knob between the two.</p>';
        return;
      }
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
    box.innerHTML = '<div class="aw-title">' + zhen("投机解码：草稿树的宽度、深度与加速比", "Speculative decoding: width, depth and speedup of the draft tree") + '</div><div class="aw-grid">' +
      row(zhen("每层候选数（宽度）", "Candidates per level (width)"), range2("k", 2, 1, 4)) + row(zhen("草稿深度", "Draft depth"), range2("d", 4, 1, 6)) +
      row(zhen("草稿 top-1 命中率", "Draft top-1 hit rate"), range2("a", 70, 30, 95)) + row(zhen("草稿一步 ÷ 目标一步", "Draft step ÷ target step"), range2("c", 15, 2, 40)) + '</div>' +
      '<svg class="aw-chart aw-sp" viewBox="0 0 560 210"></svg><div class="aw-out"></div>';
    var svg = box.querySelector("svg"), out = box.querySelector(".aw-out");
    bind(box, function () {
      var k = val(box, "k"), d = val(box, "d"), a = val(box, "a") / 100, c = val(box, "c") / 100;
      show(box, "k", k + zhen(" 个", "")); show(box, "d", d + zhen(" 层", "")); show(box, "a", (100 * a).toFixed(0) + "%");
      show(box, "c", (100 * c).toFixed(0) + "%");
      var p = 1 - Math.pow(1 - a, k);                         // 简化模型：k 个候选独立，命中率 1-(1-α)^k
      var exp = 1, acc = 0, i;
      for (i = 1; i <= d; i++) { acc += Math.pow(p, i); }
      exp = 1 + acc;                                          // 接受 a 个就吐出 a+1 个 token
      var nodes = 0;
      for (i = 1; i <= d; i++) nodes += Math.pow(k, i);
      var cost = 1 + c * d, speed = exp / cost;
      // 画树
      var x0 = 30, W = 330, S = svgText(x0, 12, zhen("草稿树：每层取 top-" + k + "，整棵树在一次前向里验证完", "draft tree: top-" + k + " per level, the whole tree verified in one forward pass"), "start");
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
        S += svgText(x0 + W + 14, levelY(i), zhen("第 " + i + " 层：走到这里的概率 " + (100 * pAcc).toFixed(0) + "%", "level " + i + ": reached with " + (100 * pAcc).toFixed(0) + "%"), "start");
        prev = cur;
      }
      svg.setAttribute("viewBox", "0 0 560 " + (levelY(d) + 30));
      svg.innerHTML = S;
      var chainP = a, chainExp = 1;
      for (i = 1; i <= d; i++) chainExp += Math.pow(chainP, i);
      if (EN) {
        out.innerHTML = "<p>The hit rate per level rises from top-1's " + (100 * a).toFixed(0) + "% to <b>" + (100 * p).toFixed(0) +
          "%</b> (any one of k candidates will do). One verification yields <b>" + exp.toFixed(2) + "</b> tokens on average" +
          " (a chain draft only " + chainExp.toFixed(2) + "), verifying <b>" + nodes + "</b> nodes; " +
          "the speedup after the draft cost is <b>" + speed.toFixed(2) + "×</b>" + (speed < 1 ? " (<b>actually slower</b>)" : "") + ".</p>" +
          '<p class="aw-note">The three knobs pull in opposite directions: more depth means longer expected runs, but later levels are reached less often (the probabilities multiply) while the draft cost grows linearly; ' +
          'more width raises the hit rate per level, at the cost of verifying k-to-the-power-depth tokens. With large batches the target model already saturates compute, ' +
          'so extra verified tokens are no longer free and speculative decoding stops paying off. So it pays most with low concurrency, long outputs and accurate drafts. ' +
          '(This uses a simplified model in which the k candidates are independent; real trees are pruned, and hit rates are not this ideal.)</p>';
        return;
      }
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
    box.innerHTML = '<div class="aw-title">' + zhen("环形 all-reduce：2(n−1) 步之后每张卡都拿到全量",
        "Ring all-reduce: after 2(n−1) steps every GPU holds the whole sum") + '</div><div class="aw-grid">' +
      row(zhen("卡数 n", "GPUs n"), range2("n", 4, 3, 6)) +
      row(zhen("第几步", "step"), range2("step", 0, 0, 6) + '<button type="button" data-k="play" class="aw-btn">' + zhen("播放", "Play") + '</button>') +
      row(zhen("每张卡的梯度大小", "gradient size per GPU"), range2("mb", 256, 16, 2048), true) + '</div>' +
      '<svg class="aw-chart aw-rr" viewBox="0 0 560 230"></svg><div class="aw-out"></div>';
    var svg = box.querySelector("svg"), out = box.querySelector(".aw-out"), timer = null;
    function stop() { if (timer) { clearInterval(timer); timer = null; box.querySelector('[data-k="play"]').textContent = zhen("播放", "Play"); } }
    box.querySelector('[data-k="play"]').addEventListener("click", function () {
      if (timer) return stop();
      this.textContent = zhen("暂停", "Pause");
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
      show(box, "n", zhen(n + " 张", String(n))); show(box, "step", t + " / " + (2 * (n - 1)));
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
      S += svgText(x0, 14, t === 0 ? zhen("开始：每张卡只有自己的那份梯度", "start: each GPU holds only its own gradient") :
        (t <= n - 1 ? zhen("reduce-scatter 第 " + t + " 步：每张卡把一块发给右边、收一块累加",
                           "reduce-scatter step " + t + ": each GPU sends one chunk right and adds the one it receives")
                    : zhen("all-gather 第 " + (t - n + 1) + " 步：把累加好的块沿环再传一圈",
                           "all-gather step " + (t - n + 1) + ": the finished chunks travel round the ring once more")), "start");
      for (c = 0; c < n; c++) S += svgText(x0 + c * cw + cw / 2, 34, zhen("块 " + c, "chunk " + c), "middle");
      for (r = 0; r < n; r++) {
        var y = 44 + r * rh;
        S += svgText(x0 - 10, y + 12, zhen("卡 " + r, "GPU " + r), "end");
        for (c = 0; c < n; c++) {
          var full = own[r][c].length === n;
          S += '<rect x="' + (x0 + c * cw + 1).toFixed(1) + '" y="' + y + '" width="' + (cw - 3).toFixed(1) + '" height="' + (rh - 6) +
               '" rx="3" class="' + (full ? "aw-on" : own[r][c].length > 1 ? "aw-b" : "aw-off") + '"/>';
          S += svgText(x0 + c * cw + cw / 2, y + 12, own[r][c].length === n ? zhen("全", "all") : zhen(own[r][c].length + " 份", String(own[r][c].length)), "middle");
        }
      }
      var ly = 44 + n * rh + 8;
      var lg = EN ? [0, 120, 260] : [0, 110, 230];
      S += '<rect x="' + (x0 + lg[0]) + '" y="' + ly + '" width="10" height="10" class="aw-off"/>' + svgText(x0 + lg[0] + 16, ly + 6, zhen("只有自己的", "its own only"), "start");
      S += '<rect x="' + (x0 + lg[1]) + '" y="' + ly + '" width="10" height="10" class="aw-b"/>' + svgText(x0 + lg[1] + 16, ly + 6, zhen("累加了一部分", "partly accumulated"), "start");
      S += '<rect x="' + (x0 + lg[2]) + '" y="' + ly + '" width="10" height="10" class="aw-on"/>' + svgText(x0 + lg[2] + 16, ly + 6, zhen("已经是全量的和", "the complete sum"), "start");
      svg.setAttribute("viewBox", "0 0 560 " + (ly + 26));
      svg.innerHTML = S;
      var perStep = mb / n, totalSent = 2 * (n - 1) * perStep;
      out.innerHTML = (EN
        ? "<p>Each GPU sends one chunk per step (" + perStep.toFixed(1) + " MB), over " + (2 * (n - 1)) + " steps, " +
          "so each sends <b>" + totalSent.toFixed(0) + " MB</b> in total ≈ 2 × " + mb + " MB ×(n−1)/n. " +
          "That is <b>almost independent of the GPU count</b>, which is why the ring scales.</p>" +
          '<p class="aw-note">The naive approach, every GPU sending its gradient to rank 0 and getting a broadcast back, makes rank 0\'s link the bottleneck ' +
          'and grows the traffic linearly with the GPU count. The ring spreads the bandwidth evenly over every link, at the price of latency growing linearly ' +
          '(2(n−1) hops), which is why NCCL switches to a tree for small messages: O(log n) latency and slightly less bandwidth.</p>'
        : "<p>每一步每张卡只发一块（" + perStep.toFixed(1) + " MB），一共 " + (2 * (n - 1)) + " 步，" +
          "每张卡总共发送 <b>" + totalSent.toFixed(0) + " MB</b> ≈ 2 × " + mb + " MB ×(n−1)/n。" +
          "<b>和卡数几乎无关</b>——这正是环形算法能扩展的原因。</p>" +
          '<p class="aw-note">朴素做法（所有卡把梯度发给 rank 0 再广播回来）会让 rank 0 的网卡成为瓶颈，' +
          '通信量随卡数线性增长。环形算法把带宽压力均分到每条链路上，代价是延迟随卡数线性增长（2(n−1) 次握手），' +
          '所以小消息上 NCCL 改用树形算法：延迟 O(log n)，带宽差一点。</p>');
    }
    bind(box, function () { stop(); draw(); });
  }

  // ---------------------------------------------------------------- ZeRO 各级的每卡显存
  function zeromem(box) {
    box.innerHTML = '<div class="aw-title">' + zhen("ZeRO：切掉哪些状态，每张卡还剩多少", "ZeRO: which states are partitioned, and what each card is left with") + '</div><div class="aw-grid">' +
      row(zhen("参数量（十亿）", "parameters (billions)"), range2("psi", 7, 1, 70)) + row(zhen("数据并行度 N", "data-parallel degree N"), range2("n", 8, 1, 64)) +
      row(zhen("激活 + 其他（GB/卡）", "activations + other (GB/card)"), range2("act", 12, 0, 40)) + row(zhen("单卡显存 GB", "memory per card, GB"), range2("cap", 80, 24, 192)) + '</div>' +
      '<svg class="aw-chart aw-zr" viewBox="0 0 560 210"></svg><div class="aw-out"></div>';
    var svg = box.querySelector("svg"), out = box.querySelector(".aw-out");
    bind(box, function () {
      var psi = val(box, "psi") * 1e9, N = val(box, "n"), act = val(box, "act"), cap = val(box, "cap");
      show(box, "psi", val(box, "psi") + " B"); show(box, "n", zhen(N + " 卡", N + " cards"));
      show(box, "act", act + " GB"); show(box, "cap", cap + " GB");
      var G = 1024 * 1024 * 1024;
      // bf16 参数 2Ψ + bf16 梯度 2Ψ + fp32 参数 4Ψ + Adam m/v 8Ψ = 16Ψ
      var LV = [
        [zhen("数据并行", "data parallel"), 2, 2, 12, 1, 1, 1], ["ZeRO-1", 2, 2, 12, 1, 1, N],
        ["ZeRO-2", 2, 2, 12, 1, N, N], ["ZeRO-3", 2, 2, 12, N, N, N]
      ];
      var bars = LV.map(function (l) {
        var p = l[1] * psi / l[4], g = l[2] * psi / l[5], o = l[3] * psi / l[6];
        return { name: l[0], parts: [[zhen("参数", "parameters"), p], [zhen("梯度", "gradients"), g], [zhen("优化器状态", "optimizer states"), o]], sum: (p + g + o) / G + act };
      });
      var mx = Math.max(cap, bars[0].sum), x0 = 92, W = 380, S = "";
      S += svgText(x0, 14, zhen("每张卡的显存占用（GB）", "memory used per card (GB)"), "start");
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
      S += svgText(xc, 186, zhen("单卡 " + cap + " GB", cap + " GB per card"), "middle");
      var lx = x0, names = EN ? [["parameters", "aw-on"], ["gradients", "aw-b"], ["optimizer states", "aw-f"], ["activations etc.", "aw-off"]]
        : [["参数", "aw-on"], ["梯度", "aw-b"], ["优化器状态", "aw-f"], ["激活等", "aw-off"]];
      names.forEach(function (nm) {
        S += '<rect x="' + lx + '" y="196" width="10" height="10" class="' + nm[1] + '"/>' + svgText(lx + 16, 202, nm[0], "start");
        lx += nm[0].length * (EN ? 6.2 : 13) + 34;
      });
      svg.innerHTML = S;
      var fit = bars.filter(function (b) { return b.sum <= cap; });
      out.innerHTML = EN
        ? "<p>" + (fit.length ? "The simplest workable choice is <b>" + fit[0].name + "</b> (" + fit[0].sum.toFixed(1) + " GB per card)."
            : "<b>None of the four stages fits</b>, so tensor or pipeline parallelism has to be added, or activation recomputation.") +
            " Data parallelism needs " + bars[0].sum.toFixed(1) + " GB and ZeRO-3 only " + bars[3].sum.toFixed(1) + " GB.</p>" +
          '<p class="aw-note">Under mixed-precision Adam the model states are 16Ψ bytes: 2Ψ of bf16 parameters, 2Ψ of bf16 gradients, 4Ψ of fp32 parameter copies and 8Ψ of first and second moments. ' +
          'ZeRO-1 and 2 add no communication (an all-reduce is already a reduce-scatter plus an all-gather); ' +
          'ZeRO-3 all-gathers even the parameters temporarily, about 1.5 times the communication, so use the lowest stage that works. ' +
          'Note that the activation segment does not shrink with N, and in long-sequence training it is usually the larger part.</p>'
        : "<p>" + (fit.length ? "最省事的可行方案是 <b>" + fit[0].name + "</b>（每卡 " + fit[0].sum.toFixed(1) + " GB）。"
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
    box.innerHTML = '<div class="aw-title">' + zhen("结构体布局：换个字段顺序，sizeof 就变了",
        "A struct's layout: reorder the fields and sizeof changes") + '</div>' +
      '<div class="aw-row aw-wide"><span>' + zhen("字段顺序", "field order") + '</span><span class="aw-chips"></span></div>' +
      '<div class="aw-row aw-wide"><span></span>' +
      '<button type="button" data-k="sort" class="aw-btn">' + zhen("按对齐从大到小排", "sort by alignment, largest first") + '</button>' +
      '<button type="button" data-k="reset" class="aw-btn">' + zhen("还原成随手写的顺序", "back to the original order") + '</button></div>' +
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
      var S = svgText(x0, 14, zhen("每一行 8 个字节（一个 64 位字）；灰色斜纹是编译器插进去的填充",
        "8 bytes per row (one 64-bit word); the grey hatching is padding the compiler inserted"), "start");
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
      S += '<rect x="' + x0 + '" y="' + ly + '" width="10" height="10" class="aw-pad"/>' +
           svgText(x0 + 16, ly + 6, zhen("填充字节（白占地方）", "padding bytes (space for nothing)"), "start");
      svg.setAttribute("viewBox", "0 0 560 " + (ly + 26));
      svg.innerHTML = S;
      var pct = (100 * (total - useful) / total).toFixed(0);
      out.innerHTML = EN
        ? "<p><code>sizeof</code> = <b>" + total + "</b>, <code>alignof</code> = <b>" + align +
          "</b>; the real data is only " + useful + " bytes and padding takes <b>" + (total - useful) + "</b> (" + pct + "%).</p>" +
          '<p class="aw-note">There are only two rules: every member\'s offset has to be a multiple of its own alignment, ' +
          'and the struct\'s size is rounded up to a multiple of the largest alignment. ' +
          'So the widely aligned fields go first and the small ones pack in behind, which leaves the least padding. ' +
          'A 32-byte struct down to 24 bytes fits one more per 64-byte cache line, which turns straight into bandwidth when traversing in bulk. ' +
          '(Click a field to move it one place forward; clicking the first one sends it to the end.)</p>'
        : "<p><code>sizeof</code> = <b>" + total + "</b>，<code>alignof</code> = <b>" + align +
          "</b>；真正的数据只有 " + useful + " 字节，填充占了 <b>" + (total - useful) + "</b> 字节（" + pct + "%）。</p>" +
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
    var GROUP_EN = { "国家": "countries", "城市": "cities", "数字": "numbers", "动物": "animals", "颜色": "colors", "英文": "English" };
    box.innerHTML = '<div class="aw-title">' + zhen("词向量空间：Qwen3-0.6B 的真实嵌入投影到三维（拖动旋转）", "Word vector space: Qwen3-0.6B's real embeddings projected to 3D (drag to rotate)") + '</div><div class="aw-grid">' +
      row(zhen("投影方式", "Projection"), select("proj", opts(["PCA 前三维（保留最多方差）", "随机挑三个方向"], { "PCA 前三维（保留最多方差）": "PCA top 3 (most variance)", "随机挑三个方向": "3 random directions" }), "PCA 前三维（保留最多方差）"), true) +
      row(zhen("画出关系", "Draw relation"), select("ana", opts(Object.keys(ANALOGY), { "不画": "none", "国家 → 城市（中国→北京、法国→巴黎、日本→东京）": "country → city (中国→北京, 法国→巴黎, 日本→东京)", "数字的顺序（一 → 十）": "number order (一 → 十)", "man → woman 与 king → queen": "man → woman and king → queen" }), "不画"), true) +
      row(zhen("标签", "Labels"), select("lab", opts(["每个词", "只标组名"], { "每个词": "every word", "只标组名": "group names only" }), "每个词")) +
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
        var c = cent[g]; items.push({ t: "pt", p: [c[0] / c[3], c[1] / c[3], c[2] / c[3]], r: 7, fill: COL[g], label: zhen(g, GROUP_EN[g]), lcls: "aw-ptl aw-ptb", z: 50 });
      });
      ANALOGY[val(box, "ana")].forEach(function (pr) {
        if (pos[pr[0]] && pos[pr[1]]) items.push({ t: "arrow", a: pos[pr[0]], b: pos[pr[1]], cls: "aw-arr3", stroke: "#f08c00", sw: 2, z: 60 });
      });
      v.set(items);
      var legend = '<p>' + Object.keys(COL).map(function (g) { return '<span class="aw-leg" style="background:' + COL[g] + '"></span>' + zhen(g, GROUP_EN[g]); }).join("　") + "</p>";
      if (EN) {
        out.innerHTML = legend + (pca ? "<p>PCA squeezes 1024 dimensions into 3 and keeps only 24% of the variance, yet words of the same kind already cluster: numbers in one pile, animals in another, colors in a third; countries and cities almost overlap, since the model treats them all as \"place names\". English words are scattered in the middle and do not sit in the same cluster as their Chinese counterparts (king / 国王, cat / 猫); cross-language correspondence only shows up with cosine similarity in the full 1024 dimensions.</p>" :
          "<p>Seen along three random directions, there is no structure at all: on each direction, words of the same kind and of different kinds are mixed. The structure is there, but spread across 1024 dimensions; PCA's job is exactly to find the few directions of largest variance.</p>") +
          '<p class="aw-note">In the full 1024 dimensions: words in the same group have an average cosine similarity of 0.28, words in different groups 0.08; for the vector arithmetic king − man + woman the nearest word is queen (cosine 0.58), for 北京 − 中国 + 法国 (Beijing − China + France) it is 巴黎 (Paris, 0.52), and for 东京 − 日本 + 法国 (Tokyo − Japan + France) also 巴黎 (0.57). In the 3D projection these arrows need not be parallel: most of their information is in the 76% of variance that was dropped.</p>';
        return;
      }
      out.innerHTML = legend +
        (pca ? '<p>PCA 把 1024 维压到 3 维，只保留了 24% 的方差，但同类的词已经聚在一起：数字一堆、动物一堆、颜色一堆；国家和城市几乎重叠——模型把它们都当"地名"。英文词分散在中间，和中文的对应词（king / 国王、cat / 猫）并不在同一个簇里，跨语言的对应要在完整的 1024 维里用余弦相似度才看得清。</p>' :
          '<p>随机挑三个方向看，什么结构都没有：每个方向上同类的词和异类的词混在一起。结构不是没有，而是分散在 1024 维里，PCA 的作用正是找到方差最大的几个方向。</p>') +
        '<p class="aw-note">在完整的 1024 维里：同组词的平均余弦相似度 0.28，异组 0.08；向量算术 king − man + woman 最近的词是 queen（余弦 0.58），北京 − 中国 + 法国 最近的是 巴黎（0.52），东京 − 日本 + 法国 也是 巴黎（0.57）。三维投影里这些箭头未必平行——丢掉的 76% 方差里有它们的大部分信息。</p>';
    });
  }

  // ---------------------------------------------------------------- RoPE：每一对维度随位置旋转，画成螺旋
  function ropeHelix(box) {
    var D = 128, L = 64;
    box.innerHTML = '<div class="aw-title">' + zhen("RoPE：第 i 对维度的 (cos mθ<sub>i</sub>, sin mθ<sub>i</sub>) 随位置 m 画成一条螺旋（拖动旋转）", "RoPE: (cos mθ<sub>i</sub>, sin mθ<sub>i</sub>) of dimension pair i drawn as a helix along position m (drag to rotate)") + '</div><div class="aw-grid">' +
      row(zhen("维度对 i（共 64 对）", "Dimension pair i (of 64)"), range2("i", 2, 0, 63)) + row("base", select("base", ["10000", "1000000"], "10000")) +
      row(zhen("q 的位置 m", "Position m of q"), range2("m", 20, 0, 63)) + row(zhen("k 的位置 n", "Position n of k"), range2("n", 14, 0, 63)) + '</div><svg class="aw-chart"></svg><div class="aw-out"></div>';
    var out = box.querySelector(".aw-out"), v = view3d(box.querySelector("svg"), { h: 320, scale: 72, ax: 0.3, ay: -1.0, dist: 14 });
    bind(box, function () {
      var i = val(box, "i"), base = +val(box, "base"), m = val(box, "m"), n = val(box, "n");
      show(box, "i", String(i)); show(box, "m", String(m)); show(box, "n", String(n));
      var th = Math.pow(base, -2 * i / D), r = 0.95;
      function pos(p) { return [p / (L - 1) * 5.2 - 2.6, r * Math.sin(p * th), r * Math.cos(p * th)]; }   // 位置沿 x 轴；(cos, sin) 画在垂直于它的平面里
      var items = [{ t: "arrow", a: [-2.8, 0, 0], b: [3.0, 0, 0], cls: "aw-ax3", z: -99, hs: 6 }, { t: "text", p: [3.25, -0.05, 0], s: zhen("位置 m", "position m"), anchor: "start", z: 99 },
                   { t: "text", p: [-2.6, -1.2, 0], s: "0", z: 99 }, { t: "text", p: [2.6, -1.2, 0], s: String(L - 1), z: 99 }];
      var steps = 480, prev = null, s;
      for (s = 0; s <= steps; s++) { var p = pos(s / steps * (L - 1)); if (prev) items.push({ t: "seg", a: prev, b: p, cls: "aw-hel", sw: 1.6 }); prev = p; }
      [[m, "q", "#007aff"], [n, "k", "#f08c00"]].forEach(function (c) {
        var x = pos(c[0])[0], pv = null, t;
        for (t = 0; t <= 48; t++) { var q = [x, r * Math.sin(t / 48 * 2 * Math.PI), r * Math.cos(t / 48 * 2 * Math.PI)]; if (pv) items.push({ t: "seg", a: pv, b: q, cls: "aw-dash" }); pv = q; }
        items.push({ t: "arrow", a: [x, 0, 0], b: pos(c[0]), cls: "aw-arr3", stroke: c[2], sw: 2.4, z: 60 });
        items.push({ t: "pt", p: pos(c[0]), r: 4, fill: c[2], label: c[1] + zhen("（", " (") + (c[1] === "q" ? "m" : "n") + "=" + c[0] + zhen("）", ")"), z: 61 });
      });
      v.set(items);
      var dl = (m - n) * th, wav = 2 * Math.PI / th;
      if (EN) {
        out.innerHTML = "<p>θ<sub>" + i + "</sub> = " + base + "<sup>−2·" + i + "/128</sup> = <b>" + th.toExponential(2) + "</b> radians per position, so a full turn takes <b>" +
          (wav >= 1e5 ? wav.toExponential(2) : Math.round(wav).toLocaleString("en-US")) + "</b> positions (the wavelength). q at m = " + m + " has turned " + (m * th).toFixed(3) +
          " radians and k at n = " + n + " has turned " + (n * th).toFixed(3) + ", so the angle between them = (m − n)·θ<sub>" + i + "</sub> = <b>" + dl.toFixed(3) + "</b>; this pair's contribution to q·k is |q||k|·cos(angle) = |q||k| × <b>" +
          Math.cos(dl).toFixed(3) + "</b>, depending only on m − n, not on the absolute positions.</p>" +
          '<p class="aw-note">Pairs with small i turn fast, like a second hand, telling apart nearby relative positions; pairs with large i barely turn, like an hour hand, distinguishing long distances. 128 dimensions = 64 hands turning at different speeds. A larger base slows every hand, so longer distances can be told apart, which is why long-context extensions change the base.</p>';
        return;
      }
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
    var FN_EN = {
      "SwiGLU：SiLU(a) · b": ["SwiGLU: SiLU(a) · b", "The gate is SiLU(a): near 0 (closed) when a is very negative, about a (open, and growing with a) when a is very positive, with a smooth transition in between; along b it is a straight line, so b only supplies \"content\" while a decides how far the gate opens. The negative region is not a hard 0: there are small negative values, so gradients get through."],
      "GEGLU：GELU(a) · b": ["GEGLU: GELU(a) · b", "GELU and SiLU have almost the same shape (GELU's negative region is shallower), so the surfaces are almost the same too; in practice the difference is small, and the choice mostly depends on kernel implementations."],
      "ReGLU：ReLU(a) · b": ["ReGLU: ReLU(a) · b", "The half with a < 0 is cut flat: the output is always 0 and so is the gradient, so on these inputs the neuron learns nothing (a \"dead neuron\"). The half with a > 0 is bilinear, just like SwiGLU."],
      "双线性：a · b（门不加非线性）": ["Bilinear: a · b (no nonlinearity in the gate)", "A saddle: there is no \"closed\" region, a and b are perfectly symmetric, and you cannot tell which is the gate and which the content. The gate needs a nonlinearity precisely to create two states, \"closed\" and \"open\"."],
      "不带门的 MLP：GELU(a)，与 b 无关": ["Ungated MLP: GELU(a), independent of b", "A classic two-layer MLP's activation looks at only one input: the surface is completely flat along b, missing one multiplication of expressive power. The GLU family does better at the same parameter count, and the difference is exactly this multiplication."]
    };
    box.innerHTML = '<div class="aw-title">' + zhen("门控激活的曲面：输出 = 门(a) × 内容 b（拖动旋转）", "Surface of a gated activation: output = gate(a) × content b (drag to rotate)") + '</div><div class="aw-grid">' +
      row(zhen("函数", "Function"), select("fn", Object.keys(FN).map(function (k) { return [k, zhen(k, FN_EN[k][0])]; }), "SwiGLU：SiLU(a) · b"), true) + '</div><svg class="aw-chart"></svg><div class="aw-out"></div>';
    var out = box.querySelector(".aw-out"), v = view3d(box.querySelector("svg"), { h: 340, scale: 50, ax: 0.5, ay: -0.6, dist: 10 });
    bind(box, function () {
      var f = FN[val(box, "fn")];
      var items = axes3(3.4, ["a = x·W_gate", "", "b = x·W_up"]);
      items.push({ t: "text", p: [0, 2.75, 0], s: zhen("输出", "output"), z: 99 });
      items = items.concat(surface3(function (a, b) { return Math.max(-2.4, Math.min(2.4, f[0](a, b) / 3.2)); }, -3, 3, -3, 3, 20, function (y) { return heat((y + 2.4) / 4.8); }));
      v.set(items);
      if (EN) {
        out.innerHTML = "<p>" + FN_EN[val(box, "fn")][1] + "</p>" + '<p class="aw-note">a and b both come from the same input x through different matrices (a = x·W<sub>gate</sub>, b = x·W<sub>up</sub>), so this surface describes the behavior of each intermediate neuron; d<sub>ff</sub> such neurons sit side by side and are then summed with weights by W<sub>down</sub>. Height is scaled by 1/3.2 and clipped at ±2.4, and color also shows height.</p>';
        return;
      }
      out.innerHTML = "<p>" + f[1] + "</p>" + '<p class="aw-note">a、b 都是同一个输入 x 经过不同矩阵得到的（a = x·W<sub>gate</sub>，b = x·W<sub>up</sub>），所以这个曲面描述的是中间层每一个神经元的行为；d<sub>ff</sub> 个这样的神经元并排，再被 W<sub>down</sub> 加权求和。高度按 1/3.2 缩放并截断在 ±2.4，颜色也表示高度。</p>';
    });
  }

  // ---------------------------------------------------------------- Scaling Law：损失曲面与等算力线
  function scaling3d(box) {
    // L(N, D) = E + A / N^α + B / D^β，系数取自对 Chinchilla 数据的复现拟合（Besiroglu 等，2024）；它给出的最优 D/N 约 20
    var E = 1.8172, A = 482.01, B = 2085.43, al = 0.3478, be = 0.3658;
    function loss(lN, lD) { return E + A / Math.pow(10, lN * al) + B / Math.pow(10, lD * be); }
    var MODELS3 = [["GPT-3", 11.24, 11.48], ["Chinchilla", 10.85, 12.15], ["LLaMA-3-8B", 9.9, 13.18], ["Qwen3-0.6B", 8.78, 13.56], ["DeepSeek-V3", 11.83, 13.17]];
    box.innerHTML = '<div class="aw-title">' + zhen("Scaling Law 的损失曲面：L(N, D) = E + A/N<sup>α</sup> + B/D<sup>β</sup>，橙线是算力预算 C = 6ND 下所有的 (N, D) 组合（拖动旋转）", "The scaling-law loss surface: L(N, D) = E + A/N<sup>α</sup> + B/D<sup>β</sup>; the orange line is every (N, D) pair at compute budget C = 6ND (drag to rotate)") + '</div><div class="aw-grid">' +
      row(zhen("算力预算 C", "Compute budget C"), range2("c", 210, 170, 260)) + row("GPU", select("gpu", Object.keys(GPUS), "H100 SXM")) + row("MFU", range2("mfu", 40, 10, 70)) +
      '</div><svg class="aw-chart"></svg><div class="aw-out"></div>';
    var out = box.querySelector(".aw-out"), v = view3d(box.querySelector("svg"), { h: 380, scale: 60, ax: 0.45, ay: -0.75, dist: 11 });
    function X(lN) { return (lN - 9.5) * 0.9; }
    function Z(lD) { return (lD - 11.5) * 0.9; }
    function Y(l) { return (l - 3.6) * 0.6; }
    bind(box, function () {
      var lC = val(box, "c") / 10, g = GPUS[val(box, "gpu")], mfu = val(box, "mfu") / 100;
      show(box, "c", "10^" + lC.toFixed(1)); show(box, "mfu", mfu.toFixed(2));
      var b0 = -1.25, items = [                                            // 底部的坐标架贴着曲面的两条前缘：N 沿左前缘、D 沿右前缘，损失竖在前角
        { t: "arrow", a: [X(7), b0, Z(14)], b: [X(12) + 0.25, b0, Z(14)], cls: "aw-ax3", z: -99, hs: 6 }, { t: "text", p: [X(9.5), b0 - 0.32, Z(14) + 0.1], s: zhen("参数量 N →", "parameters N →"), z: 99 },
        { t: "arrow", a: [X(12), b0, Z(9)], b: [X(12), b0, Z(14) + 0.25], cls: "aw-ax3", z: -99, hs: 6 }, { t: "text", p: [X(12) + 0.1, b0 - 0.32, Z(11.5)], s: zhen("数据量 D →", "data D →"), z: 99 },
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
      if (inGrid) items.push({ t: "pt", p: [X(lNo), Y(loss(lNo, lDo)) + 0.06, Z(lDo)], r: 5, fill: "#ff3b30", label: zhen("最优（最低点）", "optimum (lowest point)"), z: 70 });
      MODELS3.forEach(function (mdl, j) { items.push({ t: "pt", p: [X(mdl[1]), Y(loss(mdl[1], mdl[2])) + 0.05, Z(mdl[2])], r: 3.5, fill: "#8e8e93", label: mdl[0], z: 65, ly: j % 2 ? 12 : -6, lx: 0, anchor: "middle" }); });
      v.set(items);
      var days = Math.pow(10, lC) / (g[2] * 1e12 * mfu) / 86400;
      function big(x) { return x >= 1e12 ? (x / 1e12).toFixed(1) + " T" : x >= 1e9 ? (x / 1e9).toFixed(1) + " B" : (x / 1e6).toFixed(0) + " M"; }
      if (EN) {
        out.innerHTML = "<p>At a budget of C = 10<sup>" + lC.toFixed(1) + "</sup> FLOP, the optimal split is <b>" + big(Nopt) + "</b> parameters and <b>" + big(Dopt) + "</b> tokens of data, D/N ≈ <b>" + (Dopt / Nopt).toFixed(0) +
          "</b>, with a predicted loss of <b>" + loss(lNo, lDo).toFixed(3) + "</b>" + (inGrid ? "" : " (the optimum is outside the plot)") + ". On " + val(box, "gpu") + " at " + Math.round(mfu * 100) + "% MFU it takes about <b>" +
          (days >= 1 ? Math.round(days).toLocaleString("en-US") + " GPU-days" : (days * 24).toFixed(1) + " GPU-hours") + "</b>.</p>" +
          '<p class="aw-note">Both height and color show the loss (blue low, orange high). Walk along the orange iso-compute line: to the left the model is too small with too much data, to the right too large with too little data per parameter, and the loss is higher at both ends; the lowest point is the Chinchilla optimum. Each 10× increase in budget raises N and D about 3× each (exponents 0.51 / 0.49). Gray dots mark where a few real models sit in the (N, D) plane (their height is taken from the fitted surface, not their real loss): modern models all lie on the "more data" side of the optimum, deliberately overtrained to be cheap at inference; LLaMA-3-8B has D/N close to 1900.</p>';
        return;
      }
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
    box.innerHTML = '<div class="aw-title">' + zhen("语言模型 = 条件概率函数：看过前面的 token 之后，下一个是什么", "A language model = a conditional probability function: having seen the previous tokens, what comes next") + '</div><div class="aw-grid">' +
      row(zhen("已看到的位置 t", "Positions seen t"), range2("t", 3, 1, rows.length), true) + '</div><div class="aw-toks"></div><svg class="aw-chart" viewBox="0 0 560 190"></svg><div class="aw-out"></div>';
    var tk = box.querySelector(".aw-toks"), svg = box.querySelector("svg"), out = box.querySelector(".aw-out");
    function esc(s) { return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/ /g, "␣").replace(/\n/g, "⏎"); }
    bind(box, function () {
      var t = val(box, "t"), r = rows[t - 1];
      show(box, "t", String(t));
      tk.innerHTML = toks.map(function (w, i) {
        return '<span class="aw-tok' + (i < t ? " aw-tok-seen" : i === t ? " aw-tok-next" : "") + '">' + esc(w) + "</span>";
      }).join("") + '<span class="aw-note">' + zhen("　（蓝：已看到；橙：真实的下一个 token）", "  (blue: seen; orange: the true next token)") + '</span>';
      var S = "", inTop = false, i;
      var bars = r.top.slice();
      for (i = 0; i < bars.length; i++) if (bars[i][0] === r.next) inTop = true;
      if (!inTop) bars.push([r.next, r.p_next]);
      S += svgText(10, 16, zhen("P(下一个 token | 前 " + t + " 个 token)，整个词表 151936 个候选里概率最高的几个：", "P(next token | first " + t + " tokens): the most likely of all 151936 candidates:"), "start");
      var pmax = 0;
      for (i = 0; i < bars.length; i++) pmax = Math.max(pmax, bars[i][1]);
      for (i = 0; i < bars.length; i++) {
        var y = 30 + i * 22, p = bars[i][1], w = Math.max(1.5, p / Math.max(0.25, pmax) * 300), truth = bars[i][0] === r.next;
        S += '<rect x="150" y="' + y + '" width="' + w.toFixed(1) + '" height="15" rx="3" class="' + (truth ? "aw-b" : "aw-f") + '"/>';
        S += svgText(144, y + 11.5, esc(bars[i][0]), "end");
        S += svgText(155 + w, y + 11.5, (p < 0.001 ? p.toExponential(1) : p.toFixed(3)) + (truth ? zhen("  ← 真实的下一个", "  ← the true next") : ""), "start");
      }
      svg.setAttribute("viewBox", "0 0 560 " + (36 + bars.length * 22));
      svg.innerHTML = S;
      var sum = 0, j;
      for (j = 0; j < t; j++) sum += rows[j].loss;
      var allAvg = 0; for (j = 0; j < rows.length; j++) allAvg += rows[j].loss; allAvg /= rows.length;
      if (EN) {
        out.innerHTML = "<p>The true next token is “" + esc(r.next) + "”, to which the model gives probability <b>" + (r.p_next < 0.001 ? r.p_next.toExponential(1) : r.p_next.toFixed(3)) +
          "</b>; this position's loss −log P = <b>" + r.loss.toFixed(2) + "</b>" + (r.loss < 0.5 ? " (hardly in doubt)" : r.loss > 8 ? " (completely unexpected)" : r.loss > 4 ? " (very surprising)" : "") +
          ". The mean loss over the first " + t + " positions is " + (sum / t).toFixed(2) + ", perplexity e<sup>" + (sum / t).toFixed(2) + "</sup> = <b>" + Math.exp(sum / t).toFixed(1) +
          "</b>; over all " + rows.length + " positions of the sentence the mean is " + allAvg.toFixed(2) + ", perplexity " + Math.exp(allAvg).toFixed(1) + ".</p>" +
          '<p class="aw-note">In training, the distributions at these ' + rows.length + ' positions are computed at once in one forward pass (teacher forcing), and the loss is their mean; in inference only the last position\'s distribution is used, a token is picked from it and appended, and the next one is computed. ' +
          "Perplexity is sensitive to a few surprising tokens: the single 17.86 at position 1 pulls the whole sentence's mean from 2.3 up to 3.7.</p>";
        return;
      }
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
    var PRESETS_EN = {
      "[B, T, d] × [d]：RMSNorm 的权重乘到每个位置": "[B, T, d] × [d]: the RMSNorm weight applied at every position",
      "[B, T, d] / [B, T, 1]：每个位置除以自己的均方根": "[B, T, d] / [B, T, 1]: each position divided by its own RMS",
      "[T, 1] 与 [1, T]：外积，造因果掩码": "[T, 1] with [1, T]: an outer product to build a causal mask",
      "[B, H, T, T] + [1, 1, T, T]：注意力分数加掩码": "[B, H, T, T] + [1, 1, T, T]: adding a mask to attention scores",
      "[B, T, d] + [T, d]：位置编码加到每个 batch": "[B, T, d] + [T, d]: positional encoding added to every batch",
      "[2, 3] 与 [3, 2]：对不上": "[2, 3] with [3, 2]: does not match"
    };
    box.innerHTML = '<div class="aw-title">' + zhen("广播：从最后一维开始对齐，1 可以拉伸，其他必须相等", "Broadcasting: align from the last dimension; 1 can stretch, everything else must match") + '</div><div class="aw-grid">' +
      row(zhen("常见情形", "Common cases"), select("preset", opts(Object.keys(PRESETS), PRESETS_EN), Object.keys(PRESETS)[0]), true) +
      row(zhen("形状 A", "Shape A"), '<input type="text" data-k="a" value="2, 5, 896" spellcheck="false">') + row(zhen("形状 B", "Shape B"), '<input type="text" data-k="b" value="896" spellcheck="false">') +
      '</div><svg class="aw-chart" viewBox="0 0 560 120"></svg><div class="aw-out"></div>';
    var svg = box.querySelector("svg"), out = box.querySelector(".aw-out");
    function parse(s) { return s.split(/[,，\s\[\]]+/).filter(function (x) { return x !== ""; }).map(function (x) { return +x; }); }
    box.querySelector('[data-k="preset"]').addEventListener("change", function () {
      var pr = PRESETS[this.value]; input(box, "a").value = pr[0]; input(box, "b").value = pr[1]; draw();
    });
    function draw() {
      var A = parse(input(box, "a").value), B = parse(input(box, "b").value);
      if (!A.length || !B.length || A.concat(B).some(function (x) { return !(x >= 1) || x % 1; })) { out.innerHTML = "<p>" + zhen("形状要写成逗号分隔的正整数。", "Write shapes as comma-separated positive integers.") + "</p>"; svg.innerHTML = ""; return; }
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
      S += svgText(x0 - 10, 90, zhen("结果", "result"), "end");
      for (i = 0; i < n; i++) {
        var x = x0 + i * cw;
        S += '<rect x="' + x + '" y="75" width="' + (cw - 6) + '" height="22" rx="4" class="' + (R[i] === null ? "aw-b" : "aw-on") + '"/>';
        S += svgText(x + (cw - 6) / 2, 90, R[i] === null ? "✗" : R[i], "middle");
      }
      S += svgText(x0 + n * cw + 4, 23, zhen("← 右对齐", "← right-aligned"), "start");
      svg.innerHTML = S;
      if (EN) {
        if (bad >= 0) { out.innerHTML = "<p><b>Cannot broadcast</b>: dimension " + (bad + 1) + " is " + (rowsA[bad] == null ? 1 : rowsA[bad]) + " in one and " + (rowsB[bad] == null ? 1 : rowsB[bad]) + " in the other, neither equal nor 1. PyTorch raises <code>The size of tensor a must match the size of tensor b</code>. Either <code>unsqueeze</code> a dimension of size 1, or <code>transpose</code>.</p>"; return; }
        var cA = 1, cB = 1;
        for (i = 0; i < n; i++) { if ((rowsA[i] == null ? 1 : rowsA[i]) === 1 && R[i] !== 1) cA *= R[i]; if ((rowsB[i] == null ? 1 : rowsB[i]) === 1 && R[i] !== 1) cB *= R[i]; }
        var cpE = function (nm, c) { return c === 1 ? nm + " is not copied" : nm + " is copied " + c + " times"; };
        out.innerHTML = "<p>Result shape <b>[" + R.join(", ") + "]</b>. The dashed boxes are \"stretched\" dimensions: " + cpE("A", cA) + ", " + cpE("B", cB) + ", but only logically: PyTorch implements it as a view with stride 0, using no extra memory.</p>" +
          '<p class="aw-note">There are only two rules: align from the last dimension backward, treating missing dimensions as 1; and in each dimension the sizes must be equal or one of them must be 1. <code>keepdim=True</code> exists to keep that 1, so that [B, T, 1] can be combined directly with [B, T, d].</p>';
        return;
      }
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
    box.innerHTML = '<div class="aw-title">' + zhen("BPE：统计最常见的相邻对，合并，再统计，再合并", "BPE: count the most common adjacent pair, merge, count again, merge again") + '</div><div class="aw-grid">' +
      row(zhen("语料", "Corpus"), select("corpus", opts(Object.keys(CORPUS), { "hug pug pun bun hugs（Hugging Face 教程里的例子）": "hug pug pun bun hugs (the Hugging Face tutorial example)", "推理 推理引擎 推理服务 引擎 服务 调度 调度器": "推理 推理引擎 推理服务 引擎 服务 调度 调度器 (Chinese words)" }), Object.keys(CORPUS)[0]), true) +
      row(zhen("合并次数", "Merges"), range2("steps", 3, 0, 12)) +
      row(zhen("再编码新文本", "Encode new text"), '<input type="text" data-k="probe" value="lowest newest" spellcheck="false">', true) + '</div><div class="aw-out"></div>';
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
      if (EN) {
        var h = "<p>Words in the corpus (frequency): " + Object.keys(r.words).map(function (w) { return esc(w) + " ×" + r.words[w]; }).join(", ") + "</p>";
        h += "<p>After " + r.merges.length + " merges the corpus is split into:</p><p>" + Object.keys(r.seqs).map(function (w) { return chips(r.seqs[w]); }).join(" ") + "</p>";
        if (r.log.length) {
          var lastE = r.log[r.log.length - 1];
          h += "<p>Before merge " + r.log.length + ", the most frequent adjacent pairs were: " + lastE.top.map(function (e) { return "“" + esc(e[0][0]) + "” + “" + esc(e[0][1]) + "” ×" + e[1]; }).join(", ") +
            ", so “" + esc(lastE.pair[0]) + "” and “" + esc(lastE.pair[1]) + "” are merged into the new token “<b>" + esc(lastE.pair.join("")) + "</b>”.</p>";
          h += "<p>Merge rules (smaller numbers were learned earlier and are applied earlier when encoding): " + r.merges.map(function (m, i) { return (i + 1) + ". " + esc(m[0]) + "+" + esc(m[1]); }).join("  ") + "</p>";
        } else h += '<p class="aw-note">No merges yet: every character (plus the end-of-word marker ▁) is its own token.</p>';
        h += "<p>Vocabulary size: characters + " + r.merges.length + " merges = <b>" + Object.keys(vocab).length + "</b>; the corpus has " + Object.keys(r.seqs).reduce(function (n, w) { return n + r.seqs[w].length * r.words[w]; }, 0) + " tokens in total.</p>";
        var probeE = input(box, "probe").value;
        if (probeE.trim()) h += "<p>Encoding “" + esc(probeE) + "” with these rules: " + chips(encode(probeE, r.merges)) + '  <span class="aw-note">(unseen characters can still be represented: real byte-level BPE starts from the 256 bytes, so there are never unknown words)</span></p>';
        out.innerHTML = h;
        return;
      }
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
    box.innerHTML = '<div class="aw-title">' + zhen("浮点数的表示", "Floating-point representation") + zhen("：", ": ") + 'x = (−1)<sup>s</sup> × 1.m × 2<sup>e − bias</sup></div><div class="aw-grid">' +
      row(zhen("格式", "Format"), select("fmt", Object.keys(FMT), "BF16")) + row(zhen("数值", "Value"), '<input type="text" data-k="x" value="3.14159" spellcheck="false">') +
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
    var KIND = { "零": "zero", "最小正规数": "smallest normal number", "次正规数": "subnormal", "下溢成 0": "underflows to 0",
                 "溢出成 inf": "overflows to inf", "溢出成 NaN（E4M3 没有 inf）": "overflows to NaN (E4M3 has no inf)", "正规数": "normal number" };
    function bits(v, n) { var s = v.toString(2); while (s.length < n) s = "0" + s; return s; }
    bind(box, function () {
      var f = FMT[val(box, "fmt")], eb = f[0], mb = f[1], bias = f[2], x = parseFloat(input(box, "x").value.replace(/[，]/g, ""));
      if (isNaN(x)) { out.innerHTML = "<p>" + zhen("请输入一个数。", "Enter a number.") + "</p>"; svg.innerHTML = ""; return; }
      var r = encode(x, eb, mb, bias, f[3]), n = 1 + eb + mb, w = Math.min(16, 520 / n), x0 = 20, S = "";
      var fields = [[1, r.s, "aw-b", zhen("符号", "sign")], [eb, r.e, "aw-f", zhen("指数（" + eb + " 位）", "exponent (" + eb + " bits)")], [mb, r.m, "aw-on", zhen("尾数（" + mb + " 位）", "mantissa (" + mb + " bits)")]], pos = 0;
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
      if (EN) {
        var h = "<p>Stored as <b>" + val(box, "fmt") + "</b>: " + (KIND[r.kind] || r.kind) + "; the value actually stored is <b>" + (isFinite(r.value) ? r.value.toPrecision(mb >= 10 ? 9 : 6) : String(r.value)) + "</b>";
        if (isFinite(r.value) && r.value !== 0 && x !== 0) h += ", off from the input by " + Math.abs(r.value - x).toExponential(2) + " (relative error " + (Math.abs(r.value - x) / Math.abs(x) * 100).toFixed(3) + "%)";
        h += ".</p>";
        if (r.kind === "正规数") h += "<p>Exponent field " + r.e + " − bias " + bias + " = 2<sup>" + (r.e - bias) + "</sup>, mantissa 1 + " + r.m + "/" + (1 << mb) + " = " + (1 + r.m / (1 << mb)).toFixed(Math.min(8, mb + 1)) +
          ". At this magnitude the spacing between adjacent representable numbers (ulp) is 2<sup>" + (r.e - bias) + "</sup> × 2<sup>−" + mb + "</sup> = <b>" + ulp.toExponential(2) + "</b>: the larger the number, the larger the spacing, while the relative precision stays the same (eps = 2<sup>−" + mb + "</sup> = " + Math.pow(2, -mb).toExponential(2) + ").</p>";
        if (r.maxVal) h += '<p class="aw-note">The largest value this format can represent is ' + r.maxVal.toPrecision(4) + (f[3] ? "; anything larger becomes inf" : "; anything larger becomes NaN (E4M3 spends the encodings meant for inf on finite numbers, which buys its 448 maximum)") + ". Try 70000 (FP16 overflows), 0.0001 (FP8 underflows), 1.001 (BF16 cannot store such a small difference).</p>";
        out.innerHTML = h;
        return;
      }
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
    var KEYS = [[zhen("猫", "cat"), -2.2, 1.6], [zhen("狗", "dog"), -1.5, 2.1], [zhen("老虎", "tiger"), -2.7, 0.6], [zhen("北京", "Beijing"), 2.1, 1.5], [zhen("上海", "Shanghai"), 2.6, 0.6], [zhen("三", "three"), 1.1, -2.0], [zhen("四", "four"), 1.9, -1.5], [zhen("红色", "red"), -1.4, -1.9]];
    box.innerHTML = '<div class="aw-title">' + zhen("注意力 = 软查表：query 和每个 key 的点积是分数，softmax 成权重，再按权重混合 value（拖动橙色的 query）", "Attention = a soft lookup: the dot products of the query with each key are scores, softmax turns them into weights, and the values are mixed by weight (drag the orange query)") + '</div><div class="aw-grid">' +
      row(zhen("分数的放大倍数", "Score scale"), range2("scale", 10, 1, 40), true) + '</div><svg class="aw-chart" viewBox="0 0 560 300"></svg><div class="aw-out"></div>';
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
        svgText(o[0], o[1] + 22, zhen("输出 = 加权平均", "output = weighted average"), "middle");
      S += '<circle cx="' + qq[0].toFixed(1) + '" cy="' + qq[1].toFixed(1) + '" r="8" class="aw-dot"/>' + svgText(qq[0] + 11, qq[1] + 4, "query", "start");
      S += svgText(385, 18, zhen("softmax 权重", "softmax weights"), "start");
      for (i = 0; i < KEYS.length; i++) {
        var y = 30 + i * 31;
        S += svgText(412, y + 12, KEYS[i][0], "end") + '<rect x="420" y="' + y + '" width="' + (w[i] * 100).toFixed(1) + '" height="16" rx="3" class="aw-f"/>' + svgText(424 + w[i] * 100, y + 12, w[i].toFixed(2), "start");
      }
      svg.innerHTML = S;
      var H = 0;
      for (i = 0; i < w.length; i++) if (w[i] > 0) H -= w[i] * Math.log(w[i]);
      var best = scores.indexOf(Math.max.apply(null, scores));
      if (EN) {
        out.innerHTML = "<p>The query is at (" + q[0].toFixed(1) + ", " + q[1].toFixed(1) + "): the largest dot product is with “" + KEYS[best][0] + "”, weight " + w[best].toFixed(2) +
          ". The output is the weighted mix of the 8 values; here the values are drawn at the same places as the keys, so the output lands among the few selected points (in a real model the values are another set of vectors: keys are for \"being found\", values for \"what is given\"). The entropy of the weights is " + H.toFixed(2) + " (" + Math.log(KEYS.length).toFixed(2) + " = spread evenly, 0 = looking at one only).</p>" +
          '<p class="aw-note">The scale is the scale of the scores: too large and softmax becomes one-hot (a hard lookup with almost zero gradient), too small and it becomes an average (nothing selected). The variance of a dot product grows linearly with the dimension d, which is why scores are divided by √d to bring the scale back. Drag the query farther out and raise the scale to watch the weights concentrate on one point.</p>';
        return;
      }
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
    var CASES_EN = { "普通的 8 维向量": "an ordinary 8-dim vector", "整体偏移 +5（均值不为 0）": "shifted by +5 (nonzero mean)",
                     "整体放大 ×20（深层的残差流）": "scaled ×20 (a deep residual stream)", "有一个离群维度（真实模型里常见）": "one outlier dimension (common in real models)" };
    box.innerHTML = '<div class="aw-title">' + zhen("同一个 token 向量，LayerNorm 和 RMSNorm 各做了什么", "What LayerNorm and RMSNorm each do to the same token vector") + '</div><div class="aw-grid">' +
      row(zhen("输入 x", "Input x"), select("case", opts(Object.keys(CASES), CASES_EN), Object.keys(CASES)[0]), true) + row(zhen("γ（学到的缩放）", "γ (learned scale)"), range2("gamma", 10, 2, 30)) +
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
      S += svgText(280, 160, EN ? "μ = " + mu.toFixed(2) + ", σ = " + sd.toFixed(2) + ", RMS = " + rms.toFixed(2) + "; after: LayerNorm mean 0, std " + g.toFixed(1) + "; RMSNorm RMS " + g.toFixed(1) + ", mean " + (mu / rms * g).toFixed(2) :
        "μ = " + mu.toFixed(2) + "，σ = " + sd.toFixed(2) + "，RMS = " + rms.toFixed(2) + "；归一化后：LayerNorm 均值 0、标准差 " + g.toFixed(1) + "；RMSNorm 均方根 " + g.toFixed(1) + "、均值 " + (mu / rms * g).toFixed(2), "middle");
      S += svgText(280, 180, zhen("右边两组用同一比例画（±3 顶满）；左边按自己的最大值缩放", "the two right groups share one scale (±3 fills it); the left is scaled to its own maximum"), "middle");
      svg.innerHTML = S;
      var c = val(box, "case"), msg = c.indexOf("偏移") >= 0 ? "LayerNorm 把整体偏移减掉了，RMSNorm 没有——它只除以均方根，偏移被一起缩小但还在（均值 " + (mu / rms * g).toFixed(2) + "）。实践里这点差别几乎不影响效果，于是省掉减均值这一步。" :
        c.indexOf("放大") >= 0 ? "放大 20 倍后两种归一化的输出和原来完全一样：归一化对尺度不敏感，这就是 Pre-Norm 里残差流可以越长越大、每层读到的输入却始终是同样尺度的原因。" :
        c.indexOf("离群") >= 0 ? "一个 30 把 RMS 拉到 " + rms.toFixed(1) + "，其余 7 个维度被压到接近 0——离群维度主导了归一化。这就是量化里头疼的\"激活离群值\"：它们在归一化前后都存在，而且 γ 常常会把它们放得更大。" :
        "普通情况下两种结果几乎一样：每个维度除以一个\"整体尺度\"。γ 是可学习的逐维缩放，推理引擎常把它融合进后面的矩阵乘（或者融合进归一化 kernel 里）。";
      if (EN) {
        var msgE = c.indexOf("偏移") >= 0 ? "LayerNorm subtracts the global shift, RMSNorm does not: it only divides by the RMS, so the shift is scaled down along with everything else but remains (mean " + (mu / rms * g).toFixed(2) + "). In practice this difference barely affects quality, so the mean subtraction is dropped." :
          c.indexOf("放大") >= 0 ? "Scaled up 20 times, both normalizations give exactly the same output as before: normalization is insensitive to scale, which is why under Pre-Norm the residual stream can keep growing while every layer still reads inputs of the same scale." :
          c.indexOf("离群") >= 0 ? "A single 30 pulls the RMS up to " + rms.toFixed(1) + " and squeezes the other 7 dimensions close to 0: the outlier dominates the normalization. These are the \"activation outliers\" that make quantization hard: they exist before and after normalization, and γ often makes them even larger." :
          "In the ordinary case the two give almost the same result: each dimension is divided by an \"overall scale\". γ is a learnable per-dimension scale, which inference engines often fuse into the following matrix multiplication (or into the normalization kernel).";
        out.innerHTML = "<p>" + msgE + "</p>" + '<p class="aw-note">The statistics (μ, σ, RMS) must be computed in FP32: summing the squares of thousands of numbers in BF16 loses precision. An inference engine\'s RMSNorm kernel reads the whole vector once, reduces once and writes once, fused with the residual addition into one kernel (add + rmsnorm).</p>';
        return;
      }
      out.innerHTML = "<p>" + msg + "</p>" + '<p class="aw-note">统计量（μ、σ、RMS）要用 FP32 算：BF16 下对几千个数求平方和会丢精度。推理引擎的 RMSNorm kernel 一次读入整个向量、一次归约、再写出，和残差相加融合成一个 kernel（add + rmsnorm）。</p>';
    });
  }

  // ---------------------------------------------------------------- 一层里参数怎么分：注意力 vs FFN，以及整个模型
  function paramShare(box) {
    var PRESETS = {                 // d, d_ff, n_h, n_kv, d_h, 层数, 词表, 是否共享嵌入
      "Qwen3-0.6B": [1024, 3072, 16, 8, 128, 28, 151936, 1], "LLaMA-7B": [4096, 11008, 32, 32, 128, 32, 32000, 0],
      "LLaMA-3-8B": [4096, 14336, 32, 8, 128, 32, 128256, 0], "Qwen2.5-7B": [3584, 18944, 28, 4, 128, 28, 152064, 0], "LLaMA-3-70B": [8192, 28672, 64, 8, 128, 80, 128256, 0]
    };
    box.innerHTML = '<div class="aw-title">' + zhen("一层的参数怎么分：q / k / v / o 和 gate / up / down；整个模型再加上嵌入与输出层", "How a layer's parameters split: q / k / v / o and gate / up / down; the whole model adds the embedding and output layers") + '</div><div class="aw-grid">' +
      row(zhen("模型", "Model"), select("preset", Object.keys(PRESETS), "Qwen3-0.6B")) + row("d", num("d", 1024, 64, 32768, 64)) + row("d_ff", num("dff", 3072, 64, 131072, 64)) +
      row(zhen("query 头数", "Query heads"), num("nh", 16, 1, 256)) + row(zhen("KV 头数", "KV heads"), num("nkv", 8, 1, 256)) + row(zhen("头维 d_h", "Head dim d_h"), num("dh", 128, 8, 512, 8)) +
      row(zhen("层数", "Layers"), num("L", 28, 1, 256)) + row(zhen("词表", "Vocabulary"), num("V", 151936, 256, 1000000, 256)) + row(zhen("嵌入与输出层共享", "Tied embedding/output"), select("tie", opts(["是", "否"], { "是": "yes", "否": "no" }), "是")) + row(zhen("上下文长度 T", "Context length T"), num("T", 4096, 1, 1048576, 256)) +
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
      svg.innerHTML = stack(8, [["q", q, "aw-f"], ["k", k, "aw-i2"], ["v", v, "aw-i2"], ["o", o, "aw-f"], ["gate", d * dff, "aw-b"], ["up", d * dff, "aw-b"], ["down", d * dff, "aw-b"]], layer, zhen("一层", "one layer")) +
        stack(48, [[zhen("注意力", "attn"), L * attn, "aw-f"], ["FFN", L * mlp, "aw-b"], [zhen("嵌入", "embed"), embed, "aw-j2"]].concat(head ? [[zhen("输出层", "output"), head, "aw-j2"]] : []), total, zhen("整个模型", "whole model")) +
        svgText(8, 100, zhen("每个参数对每个 token 贡献 2 FLOP，所以参数占比 ≈ 权重部分的计算占比", "each parameter contributes 2 FLOP per token, so parameter shares ≈ compute shares of the weights"), "start");
      var scoreFlops = L * 4 * T * nh * dh, weightFlops = 2 * (L * layer + (tie ? V * d : head));   // 注意力分数：QKᵀ 和 PV 各 2·T·d_h 每头每 token
      if (EN) {
        out.innerHTML = "<p>One layer has <b>" + fmtP(layer) + "</b> parameters: attention " + fmtP(attn) + " (" + Math.round(attn / layer * 100) + "%), FFN " + fmtP(mlp) + " (" + Math.round(mlp / layer * 100) + "%)" +
          (nkv < nh ? "; KV heads are 1/" + (nh / nkv) + " of the query heads, so the k and v projections are only 1/" + (nh / nkv) + " of q (GQA)" : "") + ". The whole model has <b>" + fmtP(total) + "</b>: " + fmtP(L * layer) + " in " + L +
          " layers, embedding " + fmtP(embed) + (tie ? " (shared with the output layer)" : ", output layer " + fmtP(head)) + "; the vocabulary part is " + Math.round((embed + head) / total * 100) + "%" + ((embed + head) / total > 0.2 ? ", very large in a small model, which is why small models often tie the embedding and output layers" : "") + ".</p>" +
          "<p>At a context of " + T.toLocaleString("en-US") + ", one token takes " + (weightFlops / 1e9).toFixed(2) + " GFLOP through the weights, and the attention scores (QKᵀ and PV) take another " + (scoreFlops / 1e9).toFixed(2) + " GFLOP, " + Math.round(scoreFlops / (scoreFlops + weightFlops) * 100) + "%" +
          (scoreFlops > weightFlops ? ": at long contexts the attention scores become the largest part, which is why long-context inference optimizes attention kernels specifically" : "") + ".</p>";
        return;
      }
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
    box.innerHTML = '<div class="aw-title">' + zhen("MoE 路由：64 个 token，每个选 top-k 个专家；路由器越偏心，负载越不均衡", "MoE routing: 64 tokens each pick top-k experts; the more biased the router, the more unbalanced the load") + '</div><div class="aw-grid">' +
      row(zhen("专家数 E", "Experts E"), select("E", ["8", "16", "32", "64"], "16")) + row(zhen("每个 token 选 k 个", "k per token"), select("k", ["1", "2", "4", "8"], "2")) +
      row(zhen("路由器的偏心程度", "Router bias"), range2("skew", 30, 0, 100)) + row(zhen("容量因子", "Capacity factor"), range2("cap", 125, 100, 250)) +
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
      S += svgText(36, 20, zhen("每个专家分到的 token 数", "tokens per expert"), "start") + svgText(540, 20, zhen("虚线 = 容量 " + capacity + "，实线 = 均匀 " + (N * k / E).toFixed(1), "dashed = capacity " + capacity + ", solid = uniform " + (N * k / E).toFixed(1)), "end");
      svg.innerHTML = S;
      if (EN) {
        out.innerHTML = "<p>" + N + " tokens × top-" + k + " = " + N * k + " assignments, " + (N * k / E).toFixed(1) + " per expert if uniform; the busiest expert now has <b>" + maxLoad + "</b>, and the auxiliary loss E·Σ f<sub>i</sub>P<sub>i</sub> = <b>" + aux.toFixed(2) +
          "</b> (1 when perfectly uniform); with capacity " + capacity + ", <b>" + dropped + "</b> tokens over capacity are dropped (they take the residual path).</p>" +
          '<p class="aw-note">Each token computes only ' + k + ' experts: the compute is ' + k + '/' + E + ' of a dense FFN with the same total parameters, which is where MoE\'s "many parameters, little compute" comes from; but all the weights must still sit in memory, and at inference time experts run GEMMs on their groups of tokens, so imbalance means some GPUs (under expert parallelism) wait for the busiest one. Push the bias all the way right and you see what a router "collapsing" early in training looks like.</p>';
        return;
      }
      out.innerHTML = "<p>" + N + " 个 token × top-" + k + " = " + N * k + " 次分配，均匀时每个专家 " + (N * k / E).toFixed(1) + " 个；现在最忙的专家 <b>" + maxLoad + "</b> 个，辅助损失 E·Σ f<sub>i</sub>P<sub>i</sub> = <b>" + aux.toFixed(2) +
        "</b>（完全均匀时为 1）；容量 " + capacity + "，超出的 <b>" + dropped + "</b> 个 token 被丢弃（直接走残差）。</p>" +
        '<p class="aw-note">每个 token 只算 ' + k + ' 个专家：计算量是同样总参数的稠密 FFN 的 ' + k + '/' + E + '，这就是 MoE "参数多、算得少"的来源；但权重还是全都要放在显存里，推理时专家按 token 分组做 GEMM，负载不均衡就意味着有的卡（专家并行时）在等最忙的那个。偏心程度拉到最右，就是训练初期路由器"塌缩"的样子。</p>';
    });
  }

  // ---------------------------------------------------------------- DPO 损失：β 和 margin
  function dpoLoss(box) {
    box.innerHTML = '<div class="aw-title">' + zhen("DPO 损失 = −log σ(β · margin)，margin = 策略相对参考模型\"更偏好好回答\"的程度", "DPO loss = −log σ(β · margin), where margin is how much more the policy prefers the good answer than the reference does") + '</div><div class="aw-grid">' +
      row("β", range2("beta", 10, 1, 60)) + row(zhen("当前的 margin", "Current margin"), range2("m", 8, -40, 40), true) + '</div><svg class="aw-chart" viewBox="0 0 560 220"></svg><div class="aw-out"></div>';
    var svg = box.querySelector("svg"), out = box.querySelector(".aw-out");
    function sp(z) { return z > 30 ? z : Math.log(1 + Math.exp(z)); }                 // softplus
    bind(box, function () {
      var beta = val(box, "beta") / 100, m = val(box, "m");
      show(box, "beta", beta.toFixed(2)); show(box, "m", String(m));
      var X = function (v) { return 60 + (v + 40) / 80 * 470; }, Y = function (l) { return 180 - l / 4 * 150; }, S = "", i;
      for (i = -40; i <= 40; i += 10) S += '<line x1="' + X(i) + '" y1="30" x2="' + X(i) + '" y2="180" class="aw-gl"/>' + svgText(X(i), 196, i, "middle");
      for (i = 0; i <= 4; i++) S += '<line x1="60" y1="' + Y(i) + '" x2="530" y2="' + Y(i) + '" class="aw-gl"/>' + svgText(54, Y(i) + 4, i, "end");
      S += '<line x1="60" y1="180" x2="530" y2="180" class="aw-axis"/><line x1="60" y1="30" x2="60" y2="180" class="aw-axis"/>';
      S += svgText(295, 212, "margin = (log πθ(y_w) − log π_ref(y_w)) − (log πθ(y_l) − log π_ref(y_l))", "middle");
      var d1 = "", d2 = "";
      for (i = 0; i <= 160; i++) {
        var v = -40 + 80 * i / 160, l = Math.min(4, sp(-beta * v)), g = Math.min(4, beta / (1 + Math.exp(beta * v)) * 10);
        d1 += (i ? " L " : "M ") + X(v).toFixed(1) + " " + Y(l).toFixed(1);
        d2 += (i ? " L " : "M ") + X(v).toFixed(1) + " " + Y(g).toFixed(1);
      }
      S += '<path d="' + d1 + '" class="aw-i" fill="none"/><path d="' + d2 + '" class="aw-j" fill="none" stroke-dasharray="5 4"/>';
      S += svgText(70, 44, zhen("实线：损失 −log σ(β·margin)", "solid: loss −log σ(β·margin)"), "start") + svgText(70, 60, zhen("虚线：梯度大小 β·σ(−β·margin) × 10", "dashed: gradient size β·σ(−β·margin) × 10"), "start");
      var lm = sp(-beta * m), gm = beta / (1 + Math.exp(beta * m));
      S += '<circle cx="' + X(m).toFixed(1) + '" cy="' + Y(Math.min(4, lm)).toFixed(1) + '" r="5" class="aw-dot"/>';
      svg.innerHTML = S;
      if (EN) {
        out.innerHTML = "<p>β = " + beta.toFixed(2) + ", margin = " + m + ": loss <b>" + lm.toFixed(3) + "</b>, gradient size <b>" + gm.toFixed(3) + "</b> (= β × the probability of ranking good and bad wrong, σ(−β·margin) = " + (1 / (1 + Math.exp(beta * m))).toFixed(3) + ").</p>" +
          '<p class="aw-note">The margin is a difference of log-probabilities over whole answers, often a dozen or several dozen for answers of tens of tokens; when the margin is negative (the model ranks the bad answer first), the gradient approaches β and pushes hardest; when the margin is large, the gradient goes to 0 and stops pushing. It is a binary classifier with β·margin as its logit, the same function as the reward model\'s Bradley-Terry loss, only with the reward replaced by β·log(πθ/π_ref). A small β gives a gentle curve that needs a large margin to saturate, letting the policy move farther from the reference; a large β saturates at a slight deviation, acting as a stronger KL constraint.</p>';
        return;
      }
      out.innerHTML = "<p>β = " + beta.toFixed(2) + "，margin = " + m + "：损失 <b>" + lm.toFixed(3) + "</b>，梯度大小 <b>" + gm.toFixed(3) + "</b>（= β × 把好坏排错的概率 σ(−β·margin) = " + (1 / (1 + Math.exp(beta * m))).toFixed(3) + "）。</p>" +
        '<p class="aw-note">margin 是整段回答的对数概率之差，几十个 token 的回答里它常常有十几、几十；margin 为负（模型把差回答排在前面）时梯度接近 β，推得最用力；margin 很大时梯度趋于 0，不再推——它是一个以 β·margin 为 logit 的二分类器，和奖励模型的 Bradley-Terry 损失是同一个函数，只是把奖励换成了 β·log(πθ/π_ref)。β 小，曲线平缓，要很大的 margin 才饱和，允许策略离参考模型更远；β 大，稍微偏一点就饱和，相当于更强的 KL 约束。</p>';
    });
  }

  // ---------------------------------------------------------------- LoRA：秩、目标模块与可训练参数
  function loraParams(box) {
    var MODELS4 = {                 // L, d, n_h, n_kv, d_h, d_ff, V
      "Qwen3-0.6B": [28, 1024, 16, 8, 128, 3072, 151936], "LLaMA-3-8B": [32, 4096, 32, 8, 128, 14336, 128256], "Qwen2.5-7B": [28, 3584, 28, 4, 128, 18944, 152064], "LLaMA-3-70B": [80, 8192, 64, 8, 128, 28672, 128256]
    };
    var RANKS = [1, 2, 4, 8, 16, 32, 64, 128, 256];
    box.innerHTML = '<div class="aw-title">' + zhen("LoRA：秩 r 和目标模块决定可训练参数有多少、训练显存省多少", "LoRA: the rank r and the target modules decide how many parameters train and how much training memory is saved") + '</div><div class="aw-grid">' +
      row(zhen("模型", "Model"), select("model", Object.keys(MODELS4), "LLaMA-3-8B")) + row(zhen("秩 r", "Rank r"), range2("r", 3, 0, 8)) +
      row(zhen("目标模块", "Target modules"), select("mods", opts(["q、v", "q、k、v、o", "注意力 + FFN 全部"], { "q、v": "q, v", "q、k、v、o": "q, k, v, o", "注意力 + FFN 全部": "all attention + FFN" }), "q、k、v、o")) + '</div><svg class="aw-chart" viewBox="0 0 560 90"></svg><div class="aw-out"></div>';
    var svg = box.querySelector("svg"), out = box.querySelector(".aw-out");
    bind(box, function () {
      var c = MODELS4[val(box, "model")], L = c[0], d = c[1], nh = c[2], nkv = c[3], dh = c[4], dff = c[5], V = c[6], r = RANKS[val(box, "r")];
      show(box, "r", String(r));
      var MOD = { q: [d, nh * dh], k: [d, nkv * dh], v: [d, nkv * dh], o: [nh * dh, d], gate: [d, dff], up: [d, dff], down: [dff, d] };
      var sel = { "q、v": ["q", "v"], "q、k、v、o": ["q", "k", "v", "o"], "注意力 + FFN 全部": ["q", "k", "v", "o", "gate", "up", "down"] }[val(box, "mods")];
      var layer = 0, lora = 0, targeted = 0, k;
      for (k in MOD) layer += MOD[k][0] * MOD[k][1];
      sel.forEach(function (name) { lora += r * (MOD[name][0] + MOD[name][1]); targeted += MOD[name][0] * MOD[name][1]; });
      var N = V * d + L * layer, trainable = L * lora, memFull = N * 16, memLora = N * 2 + trainable * 16;
      function gb(b) { return (b / 1e9).toFixed(b >= 1e11 ? 0 : 1) + " GB"; }
      var S = svgText(8, 22, zhen("全参数微调", "full fine-tune"), "start") + '<rect x="110" y="8" width="430" height="20" rx="3" class="aw-b"/>' + svgText(325, 22, gb(memFull) + zhen("（权重 2 + 梯度 2 + Adam 8 + 主权重 4 字节/参数）", " (weights 2 + grads 2 + Adam 8 + master 4 bytes/param)"), "middle");
      S += svgText(8, 56, "LoRA", "start") + '<rect x="110" y="42" width="' + Math.max(2, memLora / memFull * 430).toFixed(1) + '" height="20" rx="3" class="aw-f"/>' +
        svgText(116 + Math.max(2, memLora / memFull * 430), 56, gb(memLora) + zhen("（冻结权重 2 字节 + LoRA 参数 16 字节）", " (frozen weights 2 bytes + LoRA params 16 bytes)"), "start");
      S += svgText(8, 82, zhen("都没算激活和 KV：序列长、batch 大时它们才是大头，LoRA 省不掉", "neither counts activations or KV: with long sequences and big batches they dominate, and LoRA cannot save them"), "start");
      svg.innerHTML = S;
      if (EN) {
        out.innerHTML = "<p>Trainable parameters <b>" + (trainable / 1e6).toFixed(1) + " M</b>, <b>" + (trainable / N * 100).toFixed(2) + "%</b> of all " + (N / 1e9).toFixed(2) + " B: each target matrix [d_out, d_in] is replaced by r × (d_in + d_out) parameters; " +
          "going from r = 8 to 16 doubles the parameters, still tiny compared with the full model. Unmerged, inference computes " + (2 * lora / (2 * targeted) * 100).toFixed(2) + "% more FLOP (two small matmuls); merged into W, zero.</p>" +
          '<p class="aw-note">Optimizer states and gradients are kept only for the LoRA parameters, so training memory drops from 16 bytes per parameter to about 2 plus a little; this is why a 7B–8B model can be fine-tuned on one GPU (QLoRA further quantizes the frozen weights to 4 bits). Targeting q and v is what the original paper did; today it is common to add LoRA to every linear layer, with ranks of 8 to 64.</p>';
        return;
      }
      out.innerHTML = "<p>可训练参数 <b>" + (trainable / 1e6).toFixed(1) + " M</b>，占全部 " + (N / 1e9).toFixed(2) + " B 的 <b>" + (trainable / N * 100).toFixed(2) + "%</b>：每个目标矩阵 [d_out, d_in] 换成 r × (d_in + d_out) 个参数，" +
        "r 从 8 翻到 16 参数翻倍，但相对全量仍然很小。不合并时推理多算 " + (2 * lora / (2 * targeted) * 100).toFixed(2) + "% 的 FLOP（两个小矩阵乘），合并进 W 之后为零。</p>" +
        '<p class="aw-note">优化器状态和梯度只为 LoRA 参数保存，所以训练显存从 16 字节/参数降到约 2 字节/参数再加一点；这就是单卡能微调 7B～8B 模型的原因（QLoRA 再把冻结权重量化到 4 位）。目标模块选 q、v 是原论文的做法，现在常见的是全部线性层都加，秩 8～64。</p>';
    });
  }

  // ---------------------------------------------------------------- 估算：参数量、每 token 计算量、显存、延迟下限
  function estimator(box) {
    var CFG = {                    // L, d, n_h, n_kv, d_h, d_ff, V, 共享嵌入
      "Qwen3-0.6B": [28, 1024, 16, 8, 128, 3072, 151936, 1], "LLaMA-3-8B": [32, 4096, 32, 8, 128, 14336, 128256, 0], "Qwen2.5-7B": [28, 3584, 28, 4, 128, 18944, 152064, 0],
      "Qwen3-32B": [64, 5120, 64, 8, 128, 25600, 151936, 0], "LLaMA-3-70B": [80, 8192, 64, 8, 128, 28672, 128256, 0]
    };
    box.innerHTML = '<div class="aw-title">' + zhen("从 config.json 到延迟下限：参数量 → 每 token 的计算量与读取量 → TPOT、TTFT", "From config.json to latency lower bounds: parameters → compute and bytes per token → TPOT, TTFT") + '</div><div class="aw-grid">' +
      row(zhen("模型", "Model"), select("model", Object.keys(CFG), "LLaMA-3-8B")) + row("GPU", select("gpu", Object.keys(GPUS), "H100 SXM")) + row(zhen("卡数（张量并行）", "GPUs (tensor parallel)"), select("tp", ["1", "2", "4", "8"], "1")) +
      row(zhen("权重精度", "Weight precision"), select("wdt", ["BF16", "FP8", "INT4"], "BF16")) + row(zhen("KV 精度", "KV precision"), select("kdt", ["BF16", "FP8"], "BF16")) + row("batch", num("batch", 1, 1, 4096)) +
      row(zhen("上下文 S", "Context S"), num("ctx", 4096, 1, 1048576, 256)) + row(zhen("prefill 长度 T", "Prefill length T"), num("T", 2000, 1, 1048576, 100)) + row("MFU", range2("mfu", 50, 10, 80)) +
      '</div><svg class="aw-chart" viewBox="0 0 560 70"></svg><div class="aw-out"></div>';
    var svg = box.querySelector("svg"), out = box.querySelector(".aw-out");
    bind(box, function () {
      var c = CFG[val(box, "model")], L = c[0], d = c[1], nh = c[2], nkv = c[3], dh = c[4], dff = c[5], V = c[6], tie = c[7], g = GPUS[val(box, "gpu")], tp = +val(box, "tp");
      var wb = { BF16: 2, FP8: 1, INT4: 0.5 }[val(box, "wdt")], kb = val(box, "kdt") === "FP8" ? 1 : 2, batch = val(box, "batch"), S = val(box, "ctx"), T = val(box, "T"), mfu = val(box, "mfu") / 100;
      show(box, "mfu", mfu.toFixed(2));
      var N = (tie ? 1 : 2) * V * d + L * (d * nh * dh + 2 * d * nkv * dh + nh * dh * d + 3 * d * dff + 2 * d) + d;
      var flopsTok = 2 * N + 4 * L * S * nh * dh, kvTok = 2 * L * nkv * dh * kb;
      var wBytes = N * wb, kvBytes = batch * S * kvTok, bw = g[1] * 1e12 * tp, peak = (val(box, "wdt") === "FP8" && g[3] ? g[3] : g[2]) * 1e12 * tp;
      var tW = wBytes / bw * 1e3, tK = kvBytes / bw * 1e3, tpot = tW + tK, ttft = T * (2 * N + 4 * L * (T / 2) * nh * dh) / (peak * mfu) * 1e3;
      var fit = (wBytes + kvBytes) / tp <= (g[0] - 4) * GB;
      var Sg = svgText(8, 20, zhen("decode 一步要读：", "one decode step reads:"), "start") + '<rect x="130" y="6" width="' + (tW / tpot * 400).toFixed(1) + '" height="20" rx="3" class="aw-f"/>' +
        '<rect x="' + (130 + tW / tpot * 400).toFixed(1) + '" y="6" width="' + (tK / tpot * 400).toFixed(1) + '" height="20" rx="3" class="aw-b"/>' +
        svgText(130 + tW / tpot * 200, 20, zhen("权重 ", "weights ") + fmtBytes(wBytes), "middle") + (tK / tpot > 0.12 ? svgText(130 + tW / tpot * 400 + tK / tpot * 200, 20, "KV " + fmtBytes(kvBytes), "middle") : "") +
        svgText(8, 56, EN ? "at " + (bw / 1e12).toFixed(1) + " TB/s total over " + tp + " GPU(s): at least " + tpot.toFixed(1) + " ms; weights " + Math.round(tW / tpot * 100) + "%, KV " + Math.round(tK / tpot * 100) + "%" :
          "按 " + tp + " 张卡合计 " + (bw / 1e12).toFixed(1) + " TB/s 的带宽，至少 " + tpot.toFixed(1) + " ms；权重占 " + Math.round(tW / tpot * 100) + "%、KV 占 " + Math.round(tK / tpot * 100) + "%", "start");
      svg.innerHTML = Sg;
      if (EN) {
        out.innerHTML = "<p>Parameters N = <b>" + (N / 1e9).toFixed(2) + " B</b> (" + (tie ? "tied" : "untied") + " embedding; attention " + ((L * (d * nh * dh + 2 * d * nkv * dh + nh * dh * d)) / 1e9).toFixed(2) + " B, FFN " + (L * 3 * d * dff / 1e9).toFixed(2) +
          " B, vocabulary " + ((tie ? 1 : 2) * V * d / 1e9).toFixed(2) + " B). Per token: 2N = " + (2 * N / 1e9).toFixed(1) + " GFLOP, plus " + (4 * L * S * nh * dh / 1e9).toFixed(1) + " GFLOP of attention scores at a context of " + S.toLocaleString("en-US") +
          "; KV " + fmtBytes(kvTok) + " per token.</p>" +
          "<p>Decode: TPOT ≥ <b>" + tpot.toFixed(1) + " ms</b>, ≤ " + Math.round(1000 / tpot) + " tok/s per request, total throughput ≤ <b>" + Math.round(batch * 1000 / tpot).toLocaleString("en-US") + " tok/s</b>" + (fit ? "" : " (<b>does not fit in memory</b>: weights plus KV exceed the roughly " + (g[0] - 4) + " GB available per GPU)") +
          ". Prefill of " + T.toLocaleString("en-US") + " tokens: TTFT ≈ <b>" + (ttft >= 1000 ? (ttft / 1000).toFixed(2) + " s" : ttft.toFixed(0) + " ms") + "</b> (at " + Math.round(mfu * 100) + "% MFU).</p>" +
          '<p class="aw-note">Pull the batch from 1 to 64: TPOT rises only a little (the extra reads are KV) while throughput grows dozens of times, which is batching; switch to INT4: the weights segment shrinks to a quarter; lengthen the context: the KV segment becomes the largest, which is when GQA / MLA and KV quantization matter. These are all lower bounds; reaching 70–85% of bandwidth in practice is very good.</p>';
        return;
      }
      out.innerHTML = "<p>参数量 N = <b>" + (N / 1e9).toFixed(2) + " B</b>（" + (tie ? "共享" : "不共享") + "嵌入；注意力 " + ((L * (d * nh * dh + 2 * d * nkv * dh + nh * dh * d)) / 1e9).toFixed(2) + " B，FFN " + (L * 3 * d * dff / 1e9).toFixed(2) +
        " B，词表 " + ((tie ? 1 : 2) * V * d / 1e9).toFixed(2) + " B）。每个 token：2N = " + (2 * N / 1e9).toFixed(1) + " GFLOP，上下文 " + S.toLocaleString("zh-CN") + " 时注意力分数再加 " + (4 * L * S * nh * dh / 1e9).toFixed(1) +
        " GFLOP；KV " + fmtBytes(kvTok) + "/token。</p>" +
        "<p>decode：TPOT ≥ <b>" + tpot.toFixed(1) + " ms</b>，单请求 ≤ " + Math.round(1000 / tpot) + " tok/s，总吞吐 ≤ <b>" + Math.round(batch * 1000 / tpot).toLocaleString("zh-CN") + " tok/s</b>" + (fit ? "" : "（<b>显存放不下</b>：权重加 KV 超过了每张卡约 " + (g[0] - 4) + " GB 的可用空间）") +
        "。prefill " + T.toLocaleString("zh-CN") + " 个 token：TTFT ≈ <b>" + (ttft >= 1000 ? (ttft / 1000).toFixed(2) + " s" : ttft.toFixed(0) + " ms") + "</b>（按 MFU " + Math.round(mfu * 100) + "%）。</p>" +
        '<p class="aw-note">把 batch 从 1 拉到 64：TPOT 只涨一点（多读的是 KV），吞吐涨几十倍——这就是批处理；换 INT4：权重那一段缩到四分之一；拉长上下文：KV 那一段变成大头，这时该看 GQA / MLA 和 KV 量化。都是下限，实际能到带宽的 70%～85% 就很好了。</p>';
    });
  }

  // ---------------------------------------------------------------- 量化：离群值与粒度
  function quantw(box) {
    box.innerHTML = '<div class="aw-title">' + zhen("量化误差：一个离群值怎么毁掉整行，按组量化怎么把它隔离", "Quantization error: how one outlier ruins a whole row, and how per-group quantization isolates it") + '</div><div class="aw-grid">' +
      row(zhen("位宽", "Bit width"), select("bits", ["INT8", "INT4", "INT3"], "INT4")) +
      row(zhen("粒度", "Granularity"), select("gran", opts(["按张量 / 整行（1024 个数共用一个缩放）", "按组 128", "按组 32"], { "按张量 / 整行（1024 个数共用一个缩放）": "per tensor / whole row (1024 numbers share one scale)", "按组 128": "per group of 128", "按组 32": "per group of 32" }), "按张量 / 整行（1024 个数共用一个缩放）"), true) +
      row(zhen("离群值 = 多少个 σ", "Outlier = how many σ"), range2("out", 20, 1, 60)) + '</div><svg class="aw-chart" viewBox="0 0 560 190"></svg><div class="aw-out"></div>';
    var svg = box.querySelector("svg"), out = box.querySelector(".aw-out"), N = 1024, xs = [], s = 12345, i;
    function rnd() { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; }
    for (i = 0; i < N; i++) xs.push(Math.sqrt(-2 * Math.log(rnd() + 1e-12)) * Math.cos(2 * Math.PI * rnd()));
    bind(box, function () {
      var bits = { INT8: 8, INT4: 4, INT3: 3 }[val(box, "bits")], G = { "按组 128": 128, "按组 32": 32 }[val(box, "gran")] || N, o = val(box, "out");
      show(box, "out", String(o));
      var x = xs.slice(); x[777] = o;                                                   // 第 777 个数是离群值
      var qmax = Math.pow(2, bits - 1) - 1, err = 0, norm = 0, zeros = 0, sOut = 0, sNorm = 0, g;
      for (g = 0; g < N / G; g++) {
        var mx = 0, j;
        for (j = g * G; j < (g + 1) * G; j++) mx = Math.max(mx, Math.abs(x[j]));
        var sc = mx / qmax;
        if (g === Math.floor(777 / G)) sOut = sc; else if (!sNorm) sNorm = sc;
        for (j = g * G; j < (g + 1) * G; j++) {
          var q = Math.max(-qmax - 1, Math.min(qmax, Math.round(x[j] / sc))), xh = q * sc;
          if (j !== 777) { err += (xh - x[j]) * (xh - x[j]); norm += x[j] * x[j]; if (q === 0) zeros++; }
        }
      }
      if (!sNorm) sNorm = sOut;
      var S = "", bins = 40, H = [], hmax = 0, k;
      for (k = 0; k < bins; k++) H.push(0);
      for (i = 0; i < N; i++) if (i !== 777) { k = Math.floor((x[i] + 4) / 8 * bins); if (k >= 0 && k < bins) { H[k]++; hmax = Math.max(hmax, H[k]); } }
      var X = function (v) { return 40 + (v + 4) / 8 * 480; };
      for (k = 0; k < bins; k++) S += '<rect x="' + X(-4 + 8 * k / bins).toFixed(1) + '" y="' + (150 - H[k] / hmax * 110).toFixed(1) + '" width="11" height="' + (H[k] / hmax * 110).toFixed(1) + '" class="aw-idle"/>';
      var lv;
      for (lv = -qmax - 1; lv <= qmax; lv++) { var vx = lv * sOut; if (Math.abs(vx) <= 4) S += '<line x1="' + X(vx).toFixed(1) + '" y1="30" x2="' + X(vx).toFixed(1) + '" y2="150" stroke="#ff3b30" stroke-width="1.4"/>'; }
      if (G < N) for (lv = -qmax - 1; lv <= qmax; lv++) { var vn = lv * sNorm; if (Math.abs(vn) <= 4) S += '<line x1="' + X(vn).toFixed(1) + '" y1="100" x2="' + X(vn).toFixed(1) + '" y2="150" stroke="#007aff" stroke-width="1" stroke-opacity="0.8"/>'; }
      S += '<line x1="40" y1="150" x2="520" y2="150" class="aw-axis"/>';
      for (i = -4; i <= 4; i += 2) S += svgText(X(i), 166, i + "σ", "middle");
      S += svgText(528, 92, zhen("离群值 →", "outlier →"), "start") + svgText(556, 108, o + "σ", "end");
      S += svgText(44, 20, zhen("灰：1023 个普通权重的分布；红线：含离群值那一组的量化格点" + (G < N ? "；蓝线：普通组的格点" : ""), "gray: the 1023 ordinary weights; red: grid of the group with the outlier" + (G < N ? "; blue: grid of ordinary groups" : "")), "start");
      svg.innerHTML = S;
      if (EN) {
        out.innerHTML = "<p>" + val(box, "bits") + " has " + (2 * qmax + 2) + " levels. The group containing the outlier has step s = " + o + "σ / " + qmax + " = <b>" + sOut.toFixed(3) + "σ</b>" + (G < N ? ", the other groups about " + sNorm.toFixed(3) + "σ" : "") +
          "; the relative error of the ordinary weights is <b>" + (Math.sqrt(err / norm) * 100).toFixed(1) + "%</b>, and <b>" + (zeros / (N - 1) * 100).toFixed(0) + "%</b> of them are rounded straight to 0.</p>" +
          '<p class="aw-note">The scale is set by the largest absolute value: one outlier stretches the row\'s grid so sparse that the ordinary weights all fall into the same cell. ' + (G < N ? "Per-group quantization confines the outlier's effect to its own " + G + " numbers" : "Per-group quantization can confine the outlier's effect to its own group") + ', at the cost of one extra scale per group (about 0.125 extra bits per parameter for groups of 128 at INT4). Activation outliers are worse: they sit in a few fixed channels, hence SmoothQuant (moving the activations\' scale into the weights) and per-channel / per-token scales.</p>';
        return;
      }
      out.innerHTML = "<p>" + val(box, "bits") + " 有 " + (2 * qmax + 2) + " 个格点。含离群值的那一组步长 s = " + o + "σ / " + qmax + " = <b>" + sOut.toFixed(3) + "σ</b>" + (G < N ? "，其余各组步长约 " + sNorm.toFixed(3) + "σ" : "") +
        "；普通权重的相对误差 <b>" + (Math.sqrt(err / norm) * 100).toFixed(1) + "%</b>，有 <b>" + (zeros / (N - 1) * 100).toFixed(0) + "%</b> 的普通权重被直接舍入成了 0。</p>" +
        '<p class="aw-note">缩放因子由最大绝对值决定：一个离群值把整行的格点撑得稀稀拉拉，普通权重全掉进同一个格子。' + (G < N ? "按组量化让离群值只影响它所在的 " + G + " 个数" : "按组量化可以让离群值只影响它所在的一组") + '，代价是每组多存一个缩放因子（组 128、INT4 时约多 0.125 位/参数）。激活的离群值更麻烦——它们固定出现在少数通道上，所以有了 SmoothQuant（把激活的尺度搬到权重上）和按通道 / 按 token 的缩放。</p>';
    });
  }

  // ---------------------------------------------------------------- 十个模型的地图：总参数、激活参数、KV Cache
  function modelMap(box) {
    var M = [   // 名字, 总参数 B, 激活 B, KV/token KB, 128K 上下文 KV GB, 类型, 注意力
      ["LLaMA-3-8B", 8.0, 8.0, 128, 17.2, "稠密", "GQA 8 组", "u"], ["Qwen2.5-7B", 7.6, 7.6, 56, 7.5, "稠密", "GQA 4 组", "d"], ["Qwen3-8B", 8.2, 8.2, 144, 19.3, "稠密", "GQA 8 组", "r"],
      ["Gemma-3-27B", 27.0, 27.0, 83, 11.2, "稠密", "滑动窗口 5:1 + GQA", "r"], ["Mixtral-8x7B", 46.7, 12.9, 128, 17.2, "MoE 8 选 2", "GQA 8 组", "r"],
      ["Qwen3-30B-A3B", 30.5, 3.4, 96, 12.9, "MoE 128 选 8", "GQA 4 组", "d"], ["Qwen3-235B-A22B", 235.1, 22.2, 188, 25.2, "MoE 128 选 8", "GQA 4 组", "r"],
      ["gpt-oss-120b", 116.8, 5.7, 36, 4.8, "MoE 128 选 4", "滑动窗口 1:1 + GQA", "r"], ["DeepSeek-V3", 671.0, 37.6, 69, 9.2, "MoE 256 选 8 + 1 共享", "MLA（576 维潜向量）", "l"],
      ["Qwen3-Next-80B-A3B", 79.7, 3.9, 24, 3.2, "MoE 512 选 10 + 1 共享", "3/4 的层是线性注意力", "u"]
    ];   // 最后一项是标签放在圆的哪一侧
    var AXES = { "激活参数（B）": [2, 2, 60, "log"], "KV/token（KB）": [3, 15, 220, "log"], "128K 上下文的 KV（GB）": [4, 2, 30, "log"] };
    var TXT = { "稠密": "dense", "GQA 8 组": "GQA, 8 groups", "GQA 4 组": "GQA, 4 groups", "滑动窗口 5:1 + GQA": "sliding window 5:1 + GQA", "MoE 8 选 2": "MoE, 2 of 8",
                "MoE 128 选 8": "MoE, 8 of 128", "MoE 128 选 4": "MoE, 4 of 128", "滑动窗口 1:1 + GQA": "sliding window 1:1 + GQA", "MoE 256 选 8 + 1 共享": "MoE, 8 of 256 + 1 shared",
                "MLA（576 维潜向量）": "MLA (576-dim latent)", "MoE 512 选 10 + 1 共享": "MoE, 10 of 512 + 1 shared", "3/4 的层是线性注意力": "3/4 of layers linear attention" };
    box.innerHTML = '<div class="aw-title">' + zhen("十个模型的地图：横轴总参数，纵轴可选；圆的大小 = 每个 token 的 KV", "A map of ten models: total parameters across, your choice up; circle size = KV per token") + '</div><div class="aw-grid">' +
      row(zhen("纵轴", "Vertical axis"), select("y", opts(Object.keys(AXES), { "激活参数（B）": "active parameters (B)", "KV/token（KB）": "KV/token (KB)", "128K 上下文的 KV（GB）": "KV at 128K context (GB)" }), "激活参数（B）"), true) + '</div><svg class="aw-chart" viewBox="0 0 560 300"></svg><div class="aw-out"></div>';
    var svg = box.querySelector("svg"), out = box.querySelector(".aw-out");
    bind(box, function () {
      var ax = AXES[val(box, "y")], yi = ax[0], ylo = ax[1], yhi = ax[2];
      var X = function (v) { return 60 + (Math.log10(v) - Math.log10(5)) / (Math.log10(1000) - Math.log10(5)) * 470; };
      var Y = function (v) { return 250 - (Math.log10(v) - Math.log10(ylo)) / (Math.log10(yhi) - Math.log10(ylo)) * 220; };
      var S = "", i, t;
      [5, 10, 20, 50, 100, 200, 500, 1000].forEach(function (v) { S += '<line x1="' + X(v).toFixed(1) + '" y1="30" x2="' + X(v).toFixed(1) + '" y2="250" class="aw-gl"/>' + svgText(X(v), 266, v, "middle"); });
      var yt = yi === 2 ? [2, 5, 10, 20, 50] : yi === 3 ? [20, 50, 100, 200] : [2, 5, 10, 20];
      yt.forEach(function (v) { S += '<line x1="60" y1="' + Y(v).toFixed(1) + '" x2="530" y2="' + Y(v).toFixed(1) + '" class="aw-gl"/>' + svgText(54, Y(v) + 4, v, "end"); });
      S += '<line x1="60" y1="250" x2="530" y2="250" class="aw-axis"/><line x1="60" y1="30" x2="60" y2="250" class="aw-axis"/>' + svgText(295, 284, zhen("总参数（B，对数）", "total parameters (B, log)"), "middle");
      if (yi === 2) {
        S += '<line x1="' + X(5).toFixed(1) + '" y1="' + Y(5).toFixed(1) + '" x2="' + X(60).toFixed(1) + '" y2="' + Y(60).toFixed(1) + '" class="aw-dash"/>' + svgText(X(60) + 4, Y(60) + 4, zhen("稠密：激活 = 总参数", "dense: active = total"), "start");
      }
      for (i = 0; i < M.length; i++) {
        var m = M[i], r = 4 + Math.sqrt(m[3]) * 1.1, cx = X(m[1]), cy = Y(m[yi]), moe = m[5].indexOf("MoE") === 0;
        S += '<circle cx="' + cx.toFixed(1) + '" cy="' + cy.toFixed(1) + '" r="' + r.toFixed(1) + '" class="' + (moe ? "aw-b" : "aw-f") + '"/>';
        var lp = m[7];
        S += lp === "l" ? svgText(cx - r - 3, cy + 4, m[0], "end") : lp === "u" ? svgText(cx, cy - r - 4, m[0], "middle") : lp === "d" ? svgText(cx, cy + r + 12, m[0], "middle") : svgText(cx + r + 3, cy + 4, m[0], "start");
      }
      S += svgText(70, 44, zhen("蓝 = 稠密，橙 = MoE", "blue = dense, orange = MoE"), "start");
      svg.innerHTML = S;
      var rows = M.map(function (m) { return "<tr><td>" + m[0] + "</td><td>" + zhen(m[5], TXT[m[5]]) + "</td><td>" + zhen(m[6], TXT[m[6]]) + "</td><td>" + m[1] + " B / " + m[2] + " B</td><td>" + m[3] + " KB</td></tr>"; }).join("");
      if (EN) {
        out.innerHTML = (yi === 2 ? "<p>The farther from the diagonal (dashed), the clearer MoE's \"many parameters, little compute\": DeepSeek-V3 activates only 37.6B of its 671B, Qwen3-Next only 3.9B of 80B. At inference time memory goes by total parameters and compute by active parameters, which is why MoE uses multi-GPU expert parallelism.</p>" :
          yi === 3 ? "<p>At the same 8B class, Qwen2.5-7B's KV is less than half of LLaMA-3-8B's (4 KV head groups instead of 8); a model as large as DeepSeek-V3 stores only 69 KB per token (MLA's latent vector); Qwen3-Next replaces 3/4 of its layers with linear attention and keeps only 24 KB.</p>" :
          "<p>The KV of one request at a 128K context: from 3 GB to 25 GB. It decides how many requests one GPU can serve at once with long contexts; the ways to shrink KV (GQA, MLA, sliding windows, linear attention) are the history of this column going down.</p>") +
          '<div class="aw-scroll"><table class="aw-table"><thead><tr><th>Model</th><th>FFN</th><th>Attention</th><th>Total / active</th><th>KV/token</th></tr></thead><tbody>' + rows + "</tbody></table></div>";
        return;
      }
      out.innerHTML = (yi === 2 ? "<p>越偏离对角线（虚线），MoE 的\"参数多、算得少\"越明显：DeepSeek-V3 671B 总参数只激活 37.6B，Qwen3-Next 80B 只激活 3.9B。推理时显存按总参数算、算力按激活参数算，这就是 MoE 要用多卡专家并行的原因。</p>" :
        yi === 3 ? "<p>同样是 8B 级别，Qwen2.5-7B 的 KV 是 LLaMA-3-8B 的一半不到（4 组 KV 头而不是 8 组）；DeepSeek-V3 这么大的模型每个 token 只存 69 KB（MLA 的潜向量）；Qwen3-Next 把 3/4 的层换成线性注意力，KV 只剩 24 KB。</p>" :
        "<p>128K 上下文下一个请求的 KV：从 3 GB 到 25 GB。它决定了长上下文时一张卡能同时服务几个请求——缩小 KV 的手段（GQA、MLA、滑动窗口、线性注意力）就是这一列往下走的历史。</p>") +
        '<div class="aw-scroll"><table class="aw-table"><thead><tr><th>模型</th><th>FFN</th><th>注意力</th><th>总 / 激活</th><th>KV/token</th></tr></thead><tbody>' + rows + "</tbody></table></div>";
    });
  }

  // ================================================================ 图像与视频生成推理手册
  // ---------------------------------------------------------------- 加噪路径：数据 → 噪声的三维图
  function diffusion3d(box) {
    var P0 = [], EPS = [], n = 28, seed = 99, i;
    function rnd() { seed = (seed * 1664525 + 1013904223) >>> 0; return seed / 4294967296; }
    function gauss() { return Math.sqrt(-2 * Math.log(rnd() + 1e-12)) * Math.cos(2 * Math.PI * rnd()); }
    for (i = 0; i < n; i++) { var a = i / n * 2 * Math.PI, r = 1 + 0.07 * gauss(); P0.push([r * Math.cos(a), r * Math.sin(a)]); EPS.push([Math.max(-1.6, Math.min(1.6, gauss() * 0.7)), Math.max(-1.6, Math.min(1.6, gauss() * 0.7))]); }
    box.innerHTML = '<div class="aw-title">' + zhen("加噪：每个数据点沿着一条路走向噪声；推理就是沿这些路倒着走（拖动旋转）",
      "Adding noise: every data point walks a path towards noise, and inference walks those paths backwards (drag to rotate)") + '</div><div class="aw-grid">' +
      row(zhen("加噪方式", "how noise is added"), select("sched", opts(["DDPM：x_t = √ᾱ·x₀ + √(1−ᾱ)·ε，余弦 ᾱ_t", "流匹配：x_t = (1−t)·x₀ + t·ε"],
        { "DDPM：x_t = √ᾱ·x₀ + √(1−ᾱ)·ε，余弦 ᾱ_t": "DDPM: x_t = √ᾱ·x₀ + √(1−ᾱ)·ε, cosine ᾱ_t", "流匹配：x_t = (1−t)·x₀ + t·ε": "flow matching: x_t = (1−t)·x₀ + t·ε" }),
        "DDPM：x_t = √ᾱ·x₀ + √(1−ᾱ)·ε，余弦 ᾱ_t"), true) +
      row(zhen("时刻 t", "time t"), range2("t", 40, 0, 100)) + '</div><svg class="aw-chart"></svg><div class="aw-out"></div>';
    var out = box.querySelector(".aw-out"), v = view3d(box.querySelector("svg"), { h: 360, scale: 62, ax: 0.32, ay: -1.05, dist: 18 });
    function abar(t) { var c = Math.cos(Math.PI * t / 2); return c * c; }
    bind(box, function () {
      var ddpm = val(box, "sched").indexOf("DDPM") === 0, t = val(box, "t") / 100, k, j;
      show(box, "t", t.toFixed(2));
      function coef(tt) { if (ddpm) { var ab = abar(tt); return [Math.sqrt(ab), Math.sqrt(1 - ab)]; } return [1 - tt, tt]; }
      function pos(i, tt) { var c = coef(tt), p0 = P0[i], e = EPS[i]; return [tt * 4.8 - 2.4, c[0] * p0[1] + c[1] * e[1], c[0] * p0[0] + c[1] * e[0]]; }
      var items = [{ t: "arrow", a: [-2.6, -2.0, 0], b: [2.7, -2.0, 0], cls: "aw-ax3", z: -99, hs: 6 },
        { t: "text", p: [-2.3, -2.3, 0], s: zhen("t = 0：数据", "t = 0: data"), z: 99 }, { t: "text", p: [2.2, -2.3, 0], s: zhen("t = 1：噪声", "t = 1: noise"), z: 99 }];
      for (i = 0; i < n; i++) {
        var prev = null, hi = i % 9 === 0;
        for (k = 0; k <= 30; k++) { var p = pos(i, k / 30); if (prev) items.push({ t: "seg", a: prev, b: p, cls: hi ? "aw-arr3" : "aw-path3", stroke: hi ? "#f08c00" : undefined, sw: hi ? 2 : 1 }); prev = p; }
      }
      var x = t * 4.8 - 2.4, fr = [[x, -1.7, -1.7], [x, 1.7, -1.7], [x, 1.7, 1.7], [x, -1.7, 1.7]];   // 当前时刻的截面框
      for (j = 0; j < 4; j++) items.push({ t: "seg", a: fr[j], b: fr[(j + 1) % 4], cls: "aw-dash", z: 5 });
      for (i = 0; i < n; i++) items.push({ t: "pt", p: pos(i, t), r: 3.5, fill: i % 9 === 0 ? "#f08c00" : "#007aff", z: 10 });
      v.set(items);
      var c = coef(t), snr = c[1] > 0 ? (c[0] * c[0]) / (c[1] * c[1]) : Infinity;
      var coefTxt = "x_t = <b>" + c[0].toFixed(3) + "</b>·x₀ + <b>" + c[1].toFixed(3) + "</b>·ε", snrTxt = isFinite(snr) ? snr.toFixed(2) : "∞";
      out.innerHTML = (EN ? "<p>t = " + t.toFixed(2) + ": " + coefTxt + ", a signal-to-noise ratio of " + snrTxt +
        ". The frame holds the 28 data points as they are at this moment: while t is small the ring is still visible, and near t = 1 only a cloud of Gaussian noise is left.</p>" +
        '<p class="aw-note">Each point\'s ε is fixed, so every path is deterministic; the two ways of adding noise differ only in how the signal-to-noise ratio moves with t (a cosine ᾱ starts slowly and hurries in the middle, flow matching is a straight line). What the model learns is which way the path goes at any (x_t, t) (predicting the noise or the velocity), and inference starts from a random point at t = 1 and walks the path backwards to t = 0. The straighter the path (flow matching), the more accurately a large Euler step follows it, which is one source of few-step sampling.</p>'
        : "<p>t = " + t.toFixed(2) + "：x_t = <b>" + c[0].toFixed(3) + "</b>·x₀ + <b>" + c[1].toFixed(3) + "</b>·ε，信噪比 " + snrTxt +
        "。截面框里是 28 个数据点此刻的样子：t 小时还能看出圆环，t 接近 1 时只剩一团高斯噪声。</p>" +
        '<p class="aw-note">每个点的 ε 固定，所以每条路都是确定的；两种加噪方式只是"信噪比随 t 怎么变"的曲线不同（余弦 ᾱ 开头慢、中间快，流匹配是直线）。模型学的是在任意 (x_t, t) 处"路往哪边走"（预测噪声或速度），推理就是从 t = 1 的随机点出发沿路倒着走回 t = 0。路越直（流匹配），用大步长的欧拉法走得越准，这也是少步数的来源之一。</p>');
    });
  }

  // ---------------------------------------------------------------- 无分类器引导：外推怎么改变分布
  function cfgGuide(box) {
    box.innerHTML = '<div class="aw-title">' + zhen("无分类器引导：ṽ = v_∅ + w·(v_c − v_∅)，等价于从 p_c(x)^w · p_∅(x)^(1−w) 里采样",
      "Classifier-free guidance: ṽ = v_∅ + w·(v_c − v_∅), equivalent to sampling from p_c(x)^w · p_∅(x)^(1−w)") + '</div><div class="aw-grid">' +
      row("guidance scale w", range2("w", 20, 0, 120), true) + '</div><svg class="aw-chart" viewBox="0 0 560 230"></svg><div class="aw-out"></div>';
    var svg = box.querySelector("svg"), out = box.querySelector(".aw-out");
    function lnN(x, mu, sd) { return -0.5 * ((x - mu) / sd) * ((x - mu) / sd) - Math.log(sd); }
    bind(box, function () {
      var w = val(box, "w") / 10; show(box, "w", w.toFixed(1));
      var xs = [], pu = [], pc = [], pg = [], i, N = 160, Zu = 0, Zc = 0, Zg = 0, mx = 0;
      for (i = 0; i <= N; i++) {
        var x = -4 + 9 * i / N, lu = lnN(x, 0, 1.5), lc = lnN(x, 1.3, 0.75), lg = w * lc + (1 - w) * lu;
        xs.push(x); pu.push(Math.exp(lu)); pc.push(Math.exp(lc)); pg.push(Math.exp(lg - 2));
      }
      for (i = 0; i <= N; i++) { Zu += pu[i]; Zc += pc[i]; Zg += pg[i]; }
      var mean = 0, m2 = 0;
      for (i = 0; i <= N; i++) { pu[i] /= Zu; pc[i] /= Zc; pg[i] /= Zg; mean += xs[i] * pg[i]; mx = Math.max(mx, pu[i], pc[i], pg[i]); }
      for (i = 0; i <= N; i++) m2 += (xs[i] - mean) * (xs[i] - mean) * pg[i];
      var X = function (x) { return 40 + (x + 4) / 9 * 340; }, Y = function (p) { return 190 - p / mx * 150; };
      function path(arr) { var d = ""; for (var k = 0; k <= N; k++) d += (k ? " L " : "M ") + X(xs[k]).toFixed(1) + " " + Y(arr[k]).toFixed(1); return d; }
      var S = '<line x1="40" y1="190" x2="380" y2="190" class="aw-axis"/>' + '<path d="' + path(pu) + '" fill="none" stroke="currentColor" stroke-opacity="0.45" stroke-width="1.6"/>' +
        '<path d="' + path(pc) + '" class="aw-i" fill="none"/>' + '<path d="' + path(pg) + '" class="aw-j" fill="none"/>' +
        svgText(44, 20, zhen("灰：无条件 p_∅   蓝：有条件 p_c   橙：引导后（w = " + w.toFixed(1) + "）",
          "grey: unconditional p_∅   blue: conditional p_c   orange: guided (w = " + w.toFixed(1) + ")"), "start") +
        svgText(210, 210, zhen("x（一维示意）", "x (a one-dimensional sketch)"), "middle");
      // 右侧：向量外推
      var ox = 450, oy = 150, vu = [-30, -40], vc = [50, -60], vg = [vu[0] + w * (vc[0] - vu[0]), vu[1] + w * (vc[1] - vu[1])], L = Math.sqrt(vg[0] * vg[0] + vg[1] * vg[1]), sc = L > 90 ? 90 / L : 1;
      function arrow(dx, dy, color, label) {
        var ex = ox + dx * sc, ey = oy + dy * sc, ang = Math.atan2(dy, dx);
        return '<line x1="' + ox + '" y1="' + oy + '" x2="' + ex.toFixed(1) + '" y2="' + ey.toFixed(1) + '" stroke="' + color + '" stroke-width="2.2"/><polygon points="' + ex.toFixed(1) + "," + ey.toFixed(1) + " " +
          (ex - 7 * Math.cos(ang - 0.45)).toFixed(1) + "," + (ey - 7 * Math.sin(ang - 0.45)).toFixed(1) + " " + (ex - 7 * Math.cos(ang + 0.45)).toFixed(1) + "," + (ey - 7 * Math.sin(ang + 0.45)).toFixed(1) + '" fill="' + color + '"/>' +
          svgText(ex + 6 * Math.cos(ang), ey + 6 * Math.sin(ang) + 4, label, dx >= 0 ? "start" : "end");
      }
      S += svgText(ox, 20, zhen("一步里的预测向量", "the predicted vectors in one step"), "middle") + arrow(vu[0], vu[1], "#8e8e93", "v_∅") + arrow(vc[0], vc[1], "#007aff", "v_c") + arrow(vg[0], vg[1], "#f08c00", "ṽ") +
        '<line x1="' + (ox + vu[0]) + '" y1="' + (oy + vu[1]) + '" x2="' + (ox + vc[0]) + '" y2="' + (oy + vc[1]) + '" class="aw-dash"/>' +
        (sc < 1 ? svgText(ox, 215, zhen("（箭头已缩小 " + (1 / sc).toFixed(1) + " 倍）", "(the arrows are scaled down " + (1 / sc).toFixed(1) + "x)"), "middle") : "");
      svg.innerHTML = S;
      out.innerHTML = (EN ? "<p>w = " + w.toFixed(1) + ": the guided distribution's mean is <b>" + mean.toFixed(2) + "</b> and its standard deviation <b>" + Math.sqrt(m2).toFixed(2) + "</b> (unconditional 0 / 1.5, conditional 1.3 / 0.75). " +
        (w === 0 ? "w = 0 is unconditional generation, where the prompt does nothing at all." : w < 1 ? "w < 1 sits between the two, more spread out than the conditional." : w === 1 ? "w = 1 is the p(x|c) the model learned, with no extrapolation." : w <= 4 ? "w > 1 pushes the distribution narrower than the conditional and further towards the condition — which is where \"following the prompt better\" comes from." : "w is too large: the distribution narrows to a point and is pushed outside the training distribution — the oversaturation and broken detail in the image.") + "</p>" +
        '<p class="aw-note">On the right is the same thing within one step: ṽ extrapolates w times along v_∅ → v_c. The cost is two forward passes per step for v_∅ and v_c (computed together as a batch of 2, a considerable number of FLOPs); guidance distillation turns w into an input scalar of the model and computes once per step.</p>'
        : "<p>w = " + w.toFixed(1) + "：引导后分布的均值 <b>" + mean.toFixed(2) + "</b>、标准差 <b>" + Math.sqrt(m2).toFixed(2) + "</b>（无条件 0 / 1.5，有条件 1.3 / 0.75）。" +
        (w === 0 ? "w = 0 就是无条件生成，提示词完全不起作用。" : w < 1 ? "w < 1 在两者之间，比有条件的更\"散\"。" : w === 1 ? "w = 1 就是模型学到的 p(x|c)，没有外推。" : w <= 4 ? "w > 1 把分布推得比有条件的更窄、更偏向条件方向——这就是\"更听话\"的来源。" : "w 太大：分布窄得只剩一个点、而且被推到训练分布之外——对应图里的过饱和、细节崩坏。") + "</p>" +
        '<p class="aw-note">右边是同一件事在一步里的样子：ṽ 沿着 v_∅ → v_c 的方向外推 w 倍。代价是每一步要算 v_∅ 和 v_c 两次前向（拼成 batch 2 一起算，FLOP 不少）；引导蒸馏把 w 变成模型的一个输入标量，一步只算一次。</p>');
    });
  }

  // ---------------------------------------------------------------- ODE 求解器：步数与阶数
  function odeSolver(box) {
    var STEPS = [2, 4, 8, 16, 32, 64];
    box.innerHTML = '<div class="aw-title">' + zhen("同一个玩具 ODE（dx/dσ = −2σx，从 σ = 1 走到 0，精确解 x(0) = e），步数和求解器各贡献多少精度",
      "One toy ODE (dx/dσ = −2σx, from σ = 1 to 0, with the exact solution x(0) = e): how much accuracy the step count and the solver each contribute") + '</div><div class="aw-grid">' +
      row(zhen("网络调用次数 NFE", "network calls (NFE)"), range2("nfe", 2, 0, 5)) + '</div><svg class="aw-chart" viewBox="0 0 560 230"></svg><div class="aw-out"></div>';
    var svg = box.querySelector("svg"), out = box.querySelector(".aw-out");
    function v(x, s) { return -2 * s * x; }
    function run(order, steps) {
      var x = 1, s = 1, d = 1 / steps, pts = [[1, 1]], k;
      for (k = 0; k < steps; k++) {
        if (order === 1) x = x - d * v(x, s);
        else { var v1 = v(x, s), xp = x - d * v1, v2 = v(xp, s - d); x = x - d * (v1 + v2) / 2; }
        s -= d; pts.push([s, x]);
      }
      return pts;
    }
    bind(box, function () {
      var nfe = STEPS[val(box, "nfe")]; show(box, "nfe", String(nfe));
      var X = function (s) { return 50 + (1 - s) * 470; }, Y = function (x) { return 200 - (x - 0.8) / 2.2 * 170; }, S = "", k;
      S += '<line x1="50" y1="200" x2="520" y2="200" class="aw-axis"/><line x1="50" y1="20" x2="50" y2="200" class="aw-axis"/>' +
        svgText(50, 216, zhen("σ = 1（噪声端）", "σ = 1 (the noise end)"), "start") + svgText(520, 216, zhen("σ = 0（数据端）", "σ = 0 (the data end)"), "end") + svgText(44, 24, "x", "end");
      var d = "";
      for (k = 0; k <= 100; k++) { var s = 1 - k / 100, xe = Math.exp(1 - s * s); d += (k ? " L " : "M ") + X(s).toFixed(1) + " " + Y(xe).toFixed(1); }
      S += '<path d="' + d + '" fill="none" stroke="currentColor" stroke-opacity="0.5" stroke-width="1.6" stroke-dasharray="5 4"/>' +
        svgText(60, 36, zhen("虚线：精确解 x(σ) = e^(1−σ²)", "dashed: the exact solution x(σ) = e^(1−σ²)"), "start");
      var pe = run(1, nfe), ph = run(2, Math.max(1, nfe / 2));
      [[pe, "aw-i", "#007aff"], [ph, "aw-j", "#f08c00"]].forEach(function (c) {
        var dd = c[0].map(function (p, j) { return (j ? " L " : "M ") + X(p[0]).toFixed(1) + " " + Y(p[1]).toFixed(1); }).join("");
        S += '<path d="' + dd + '" class="' + c[1] + '" fill="none"/>' + c[0].map(function (p) { return '<circle cx="' + X(p[0]).toFixed(1) + '" cy="' + Y(p[1]).toFixed(1) + '" r="3.5" fill="' + c[2] + '"/>'; }).join("");
      });
      S += svgText(60, 54, zhen("蓝：欧拉法 " + nfe + " 步（每步 1 次调用）   橙：Heun " + Math.max(1, nfe / 2) + " 步（每步 2 次调用）",
        "blue: Euler, " + nfe + " steps (1 call each)   orange: Heun, " + Math.max(1, nfe / 2) + " steps (2 calls each)"), "start");
      svg.innerHTML = S;
      var ee = Math.abs(pe[pe.length - 1][1] - Math.E), eh = Math.abs(ph[ph.length - 1][1] - Math.E);
      out.innerHTML = (EN ? "<p>The same " + nfe + " network calls: Euler's endpoint error is <b>" + ee.toFixed(4) + "</b> and Heun's <b>" + eh.toFixed(4) + "</b>" +
        (nfe <= 2 ? " — with very few calls a higher-order method is not necessarily better" : eh < ee ? ", the higher-order method " + (ee / eh).toFixed(1) + " times more accurate" : "") +
        ". Doubling the NFE halves Euler's error (first order) and quarters Heun's (second order).</p>" +
        '<p class="aw-note">The cost counts only the NFE (the network calls), not the "steps": one Heun step is two forward passes. A real model\'s velocity field is not a straight line, and DPM-Solver++ and UniPC exploit the diffusion ODE\'s semi-linear structure to come in under 20 steps; a flow-matching model\'s trajectory is nearly straight, so plain Euler needs only 20 to 50; below that it takes distillation.</p>'
        : "<p>同样 " + nfe + " 次网络调用：欧拉法终点误差 <b>" + ee.toFixed(4) + "</b>，Heun <b>" + eh.toFixed(4) + "</b>" + (nfe <= 2 ? "——调用次数极少时高阶方法不一定更好" : eh < ee ? "，高阶方法准 " + (ee / eh).toFixed(1) + " 倍" : "") +
        "。NFE 翻倍，欧拉的误差减半（一阶），Heun 降到四分之一（二阶）。</p>" +
        '<p class="aw-note">成本只看 NFE（网络调用次数），不看"步数"：一步 Heun = 两次前向。真实模型的速度场不是直线，DPM-Solver++ / UniPC 利用扩散 ODE 的半线性结构做到 20 步以内；流匹配模型的轨迹接近直线，朴素欧拉 20～50 步就够；再往下要靠蒸馏。</p>');
    });
  }

  // ---------------------------------------------------------------- 时间步与 shift
  function sigmaSchedule(box) {
    var SHIFT = { "1（不偏移）": 1, "3（SD3 1024² 常用）": 3, "6": 6, "按 FLUX 规则：512²（1.88）": 1.88, "按 FLUX 规则：1024²（3.16）": 3.16, "按 FLUX 规则：2048²（25.3）": 25.28 };
    box.innerHTML = '<div class="aw-title">' + zhen("流匹配的时间步：均匀的 t 经过 t′ = s·t / (1 + (s − 1)·t) 之后，步被挪向噪声端",
      "Flow matching's timesteps: a uniform t put through t′ = s·t / (1 + (s − 1)·t) moves the steps towards the noise end") + '</div><div class="aw-grid">' +
      row(zhen("步数", "steps"), range2("n", 8, 4, 50)) + row("shift s", select("s", opts(Object.keys(SHIFT),
        { "1（不偏移）": "1 (no shift)", "3（SD3 1024² 常用）": "3 (usual for SD3 at 1024²)", "按 FLUX 规则：512²（1.88）": "FLUX's rule: 512² (1.88)",
          "按 FLUX 规则：1024²（3.16）": "FLUX's rule: 1024² (3.16)", "按 FLUX 规则：2048²（25.3）": "FLUX's rule: 2048² (25.3)" }),
        "3（SD3 1024² 常用）"), true) + '</div><svg class="aw-chart" viewBox="0 0 560 150"></svg><div class="aw-out"></div>';
    var svg = box.querySelector("svg"), out = box.querySelector(".aw-out");
    bind(box, function () {
      var n = val(box, "n"), s = SHIFT[val(box, "s")], sig = [], i;
      show(box, "n", String(n));
      for (i = 0; i < n; i++) { var t = 1 - i / n; sig.push(s * t / (1 + (s - 1) * t)); }
      var X = function (v) { return 40 + v * 480; }, S = '<line x1="40" y1="60" x2="520" y2="60" class="aw-axis"/>', high = 0;
      for (i = 0; i <= 10; i++) S += '<line x1="' + X(i / 10) + '" y1="56" x2="' + X(i / 10) + '" y2="64" class="aw-axis"/>' + svgText(X(i / 10), 80, (i / 10).toFixed(1), "middle");
      S += svgText(40, 100, zhen("σ = 0：干净", "σ = 0: clean"), "start") + svgText(520, 100, zhen("σ = 1：纯噪声", "σ = 1: pure noise"), "end");
      for (i = 0; i < n; i++) { if (sig[i] > 0.5) high++; S += '<line x1="' + X(sig[i]).toFixed(1) + '" y1="30" x2="' + X(sig[i]).toFixed(1) + '" y2="56" stroke="' + (sig[i] > 0.5 ? "#f08c00" : "#007aff") + '" stroke-width="2"/>'; }
      S += svgText(40, 22, zhen("每一根线是网络被调用时的 σ；橙：σ > 0.5 的步", "each line is the σ at which the network is called; orange: the steps with σ > 0.5"), "start");
      // 均匀 t 的对照
      for (i = 0; i < n; i++) S += '<line x1="' + X(1 - i / n).toFixed(1) + '" y1="118" x2="' + X(1 - i / n).toFixed(1) + '" y2="132" stroke="currentColor" stroke-opacity="0.35" stroke-width="1.5"/>';
      S += svgText(40, 146, zhen("对照：shift = 1 时的均匀分布", "for comparison: the uniform spacing at shift = 1"), "start");
      svg.innerHTML = S;
      out.innerHTML = (EN ? "<p><b>" + high + "</b> of the " + n + " steps fall in the high-noise range σ > 0.5 (at shift = 1 it is " + Math.ceil(n / 2) + "). The first two steps differ by only " + (sig[0] - sig[1]).toFixed(3) + " in σ, the last two by " + (sig[n - 2] - sig[n - 1]).toFixed(3) + ".</p>" +
        '<p class="aw-note">The higher the resolution, the more the signal\'s energy concentrates in the latent space and the higher the signal-to-noise ratio at the same σ, so σ = 0.5 is already fairly clean and what has to be walked carefully is the stretch near σ = 1 — which is where shift moves the steps. FLUX computes the shift from the token count: 2048² wants 25, and still using 1024²\'s 3.16 is the common reason a 2K image comes out blurry.</p>'
        : "<p>" + n + " 步里 <b>" + high + "</b> 步落在 σ > 0.5 的高噪声段（shift = 1 时是 " + Math.ceil(n / 2) + " 步）。最前面两步之间 σ 只差 " + (sig[0] - sig[1]).toFixed(3) + "，最后两步差 " + (sig[n - 2] - sig[n - 1]).toFixed(3) + "。</p>" +
        '<p class="aw-note">分辨率越高，潜空间里的信号能量越集中，同样的 σ 对应的信噪比越高，于是 σ = 0.5 已经"相当干净"，真正要仔细走的是 σ 接近 1 那段——shift 就是把步往那边挪。FLUX 按 token 数算 shift：2048² 要 25，还用 1024² 的 3.16 就是"出 2K 图发糊"的常见原因。</p>');
    });
  }

  // ---------------------------------------------------------------- 扩散模型的算账
  function diffusionFlops(box) {
    var PRE = {   // 层数, 隐藏维, 图像 token, 文本 token, CFG 倍数, 典型步数（与算账一章的表一致）
      "SD3-medium 1024²": [24, 1536, 4096, 333, 2, 28], "FLUX.1-dev 1024²": [57, 3072, 4096, 512, 1, 28], "FLUX.1-dev 2048²": [57, 3072, 16384, 512, 1, 28],
      "CogVideoX-5B 480p 49 帧": [42, 3072, 17550, 226, 2, 50], "Wan 2.1-14B 720p 81 帧": [40, 5120, 75600, 512, 2, 50], "HunyuanVideo 720p 129 帧": [60, 3072, 118800, 256, 1, 50]
    };
    var GP = { "H100 SXM": 989, "RTX 4090": 165, "RTX 5070 Ti（估）": 170, "A100": 312 };
    box.innerHTML = '<div class="aw-title">' + zhen("一步要多少 FLOP、一次生成要多久：24·N·d² 的线性层 + 4·N²·d 的注意力",
      "The FLOPs in one step and the time for one generation: 24·N·d² of linear layers plus 4·N²·d of attention") + '</div><div class="aw-grid">' +
      row(zhen("模型", "model"), select("pre", opts(Object.keys(PRE),
        { "CogVideoX-5B 480p 49 帧": "CogVideoX-5B 480p, 49 frames", "Wan 2.1-14B 720p 81 帧": "Wan 2.1-14B 720p, 81 frames", "HunyuanVideo 720p 129 帧": "HunyuanVideo 720p, 129 frames" }),
        "FLUX.1-dev 1024²"), true) + row(zhen("层数 L", "layers L"), num("L", 57, 1, 200)) + row(zhen("隐藏维 d", "hidden size d"), num("d", 3072, 64, 16384, 64)) +
      row(zhen("图像 / 视频 token", "image / video tokens"), num("ni", 4096, 1, 2000000)) + row(zhen("文本 token", "text tokens"), num("nt", 512, 0, 4096)) +
      row(zhen("步数", "steps"), num("steps", 28, 1, 1000)) + row(zhen("CFG", "guidance"), select("cfg", opts(["关（1 次前向/步）", "开（2 次前向/步）"],
        { "关（1 次前向/步）": "off (1 pass per step)", "开（2 次前向/步）": "on (2 passes per step)" }), "关（1 次前向/步）")) +
      row("GPU", select("gpu", opts(Object.keys(GP), { "RTX 5070 Ti（估）": "RTX 5070 Ti (estimated)" }), "H100 SXM")) +
      row(zhen("MFU", "utilization"), range2("mfu", 45, 10, 70)) + '</div><svg class="aw-chart" viewBox="0 0 560 60"></svg><div class="aw-out"></div>';
    var svg = box.querySelector("svg"), out = box.querySelector(".aw-out");
    box.querySelector('[data-k="pre"]').addEventListener("change", function () {
      var p = PRE[this.value]; input(box, "L").value = p[0]; input(box, "d").value = p[1]; input(box, "ni").value = p[2]; input(box, "nt").value = p[3];
      input(box, "cfg").value = p[4] === 2 ? "开（2 次前向/步）" : "关（1 次前向/步）"; input(box, "steps").value = p[5]; draw();
    });
    function draw() {
      var L = val(box, "L"), d = val(box, "d"), N = val(box, "ni") + val(box, "nt"), steps = val(box, "steps"), cfg = val(box, "cfg").indexOf("开") === 0 ? 2 : 1, peak = GP[val(box, "gpu")], mfu = val(box, "mfu") / 100;
      show(box, "mfu", mfu.toFixed(2));
      var lin = L * 24 * N * d * d / 1e12, att = L * 4 * N * N * d / 1e12, step = lin + att, nfe = steps * cfg, secs = step * nfe / (mfu * peak);
      svg.innerHTML = svgText(8, 20, zhen("一步的 FLOP", "the FLOPs in one step"), "start") + '<rect x="100" y="6" width="' + (lin / step * 440).toFixed(1) + '" height="22" rx="3" class="aw-f"/><rect x="' + (100 + lin / step * 440).toFixed(1) + '" y="6" width="' + (att / step * 440).toFixed(1) + '" height="22" rx="3" class="aw-b"/>' +
        svgText(100 + lin / step * 220, 21, zhen("线性层 ", "linear ") + Math.round(lin / step * 100) + "%", "middle") + (att / step > 0.1 ? svgText(100 + lin / step * 440 + att / step * 220, 21, zhen("注意力 ", "attention ") + Math.round(att / step * 100) + "%", "middle") : "") +
        svgText(8, 50, zhen("临界点 N = 6d = " + (6 * d).toLocaleString("zh-CN") + "，当前 N = " + N.toLocaleString("zh-CN") + (N > 6 * d ? "：注意力主导" : "：线性层主导"),
          "the crossover N = 6d = " + (6 * d).toLocaleString("en-US") + ", here N = " + N.toLocaleString("en-US") + (N > 6 * d ? ": attention dominates" : ": the linear layers dominate")), "start");
      out.innerHTML = (EN ? "<p>One forward pass is <b>" + step.toFixed(1) + " TFLOP</b> (" + lin.toFixed(1) + " linear plus " + att.toFixed(1) + " attention); " + steps + " steps x " + cfg + " = " + nfe + " passes, " + (step * nfe / 1e3).toFixed(2) + " PFLOP in all; on an " + val(box, "gpu") +
        " (" + peak + " TFLOPS at " + Math.round(mfu * 100) + "% utilization) about <b>" + (secs < 600 ? secs.toFixed(1) + " s" : (secs / 60).toFixed(1) + " min") + "</b> (excluding the text encoding and the VAE).</p>" +
        '<p class="aw-note">Quadruple the token count (double the resolution): the linear layers x4 and attention x16; a video model has tens of times an image\'s tokens, which is why everything revolves around attention. Measuring slower than this means the gap is in the utilization (no compilation, a degraded attention backend), the memory (offload) or the VAE.</p>'
        : "<p>一次前向 <b>" + step.toFixed(1) + " TFLOP</b>（线性层 " + lin.toFixed(1) + " + 注意力 " + att.toFixed(1) + "）；" + steps + " 步 × " + cfg + " = " + nfe + " 次前向，共 " + (step * nfe / 1e3).toFixed(2) + " PFLOP；在 " + val(box, "gpu") +
        "（" + peak + " TFLOPS，MFU " + Math.round(mfu * 100) + "%）上约 <b>" + (secs < 600 ? secs.toFixed(1) + " 秒" : (secs / 60).toFixed(1) + " 分钟") + "</b>（不含文本编码和 VAE）。</p>" +
        '<p class="aw-note">把 token 数翻 4 倍（分辨率翻倍）：线性层 ×4，注意力 ×16；视频模型 token 是图像的几十倍，所以一切围绕注意力。实测慢于这个数，差距在 MFU（没编译、注意力后端退化）、显存（offload）或 VAE。</p>');
    }
    bind(box, draw);
  }

  // ---------------------------------------------------------------- 卸载的代价
  function offloadCost(box) {
    var MD = {   // 去噪权重 GB, 文本编码器 GB, NFE, 各卡一步秒数（算账一章的表反推）
      "FLUX.1-dev bf16（28 步）": [23.8, 9.8, 28, { "H100 SXM": 0.17, "RTX 4090": 1.0, "A100": 0.53 }], "FLUX.1-dev fp8（28 步）": [11.9, 4.9, 28, { "H100 SXM": 0.17, "RTX 4090": 1.0, "A100": 0.53 }],
      "SDXL bf16（30 步 × CFG）": [5.2, 1.64, 60, { "H100 SXM": 0.048, "RTX 4090": 0.29, "A100": 0.15 }], "Wan 2.1-14B bf16（50 步 × CFG）": [28, 11.4, 100, { "H100 SXM": 14.9, "RTX 4090": 89.7, "A100": 47.5 }]
    };
    var BUS = { "PCIe 4.0 x16（约 25 GB/s）": 25, "PCIe 5.0 x16（约 50 GB/s）": 50 };
    box.innerHTML = '<div class="aw-title">' + zhen("卸载到 CPU 的代价：每次生成在 PCIe 上搬多少字节、多花多少秒",
      "What offloading to the CPU costs: the bytes moved over PCIe per generation and the seconds it adds") + '</div><div class="aw-grid">' +
      row(zhen("模型", "model"), select("m", opts(Object.keys(MD),
        { "FLUX.1-dev bf16（28 步）": "FLUX.1-dev bf16 (28 steps)", "FLUX.1-dev fp8（28 步）": "FLUX.1-dev fp8 (28 steps)",
          "SDXL bf16（30 步 × CFG）": "SDXL bf16 (30 steps x guidance)", "Wan 2.1-14B bf16（50 步 × CFG）": "Wan 2.1-14B bf16 (50 steps x guidance)" }),
        "FLUX.1-dev bf16（28 步）"), true) + row("GPU", select("gpu", ["H100 SXM", "RTX 4090", "A100"], "RTX 4090")) +
      row(zhen("总线", "bus"), select("bus", opts(Object.keys(BUS),
        { "PCIe 4.0 x16（约 25 GB/s）": "PCIe 4.0 x16 (about 25 GB/s)", "PCIe 5.0 x16（约 50 GB/s）": "PCIe 5.0 x16 (about 50 GB/s)" }), "PCIe 4.0 x16（约 25 GB/s）")) +
      '</div><svg class="aw-chart" viewBox="0 0 560 120"></svg><div class="aw-out"></div>';
    var svg = box.querySelector("svg"), out = box.querySelector(".aw-out");
    bind(box, function () {
      var m = MD[val(box, "m")], wdn = m[0], wte = m[1], nfe = m[2], tstep = m[3][val(box, "gpu")], bw = BUS[val(box, "bus")], compute = nfe * tstep;
      var rows = [[zhen("不卸载（放得下）", "no offload (it fits)"), 0, 0], [zhen("模型卸载（按组件）", "model offload (per component)"), 2 * (wdn + wte), 2 * (wdn + wte) / bw],
                  [zhen("逐层卸载（无预取）", "per-layer offload (no prefetch)"), wte + wdn * nfe, (wte + wdn * nfe) / bw],
                  [zhen("逐层卸载 + 预取", "per-layer offload + prefetch"), wte + wdn * nfe, Math.max(0, (wte + wdn * nfe) / bw - compute)]];
      var mx = compute + rows[2][2], S = "", i;
      for (i = 0; i < rows.length; i++) {
        var y = 6 + i * 28, wc = compute / mx * 330, we = rows[i][2] / mx * 330;
        S += svgText(150, y + 14, rows[i][0], "end") + '<rect x="158" y="' + y + '" width="' + wc.toFixed(1) + '" height="18" rx="3" class="aw-f"/><rect x="' + (158 + wc).toFixed(1) + '" y="' + y + '" width="' + we.toFixed(1) + '" height="18" rx="3" class="aw-b"/>' +
          svgText(162 + wc + we, y + 13, (compute + rows[i][2]).toFixed(1) + " s", "start");
      }
      svg.innerHTML = S;
      out.innerHTML = (EN ? "<p>The computation alone is " + compute.toFixed(1) + " s (" + nfe + " passes x " + tstep + " s). Model offload moves <b>" + rows[1][1].toFixed(0) + " GB</b> per generation (each component in once and out once), adding " + rows[1][2].toFixed(1) + " s; per-layer offload moves <b>" + rows[2][1].toFixed(0) + " GB</b> (the denoising network crosses PCIe at every step), adding " + rows[2][2].toFixed(1) + " s" +
        (rows[3][2] === 0 ? "; prefetching hides the transfer entirely behind the computation (on this card computing a layer is slower than moving one)" : "; prefetching hides only part of it and " + rows[3][2].toFixed(1) + " s remains (computing a layer is faster than moving one, so it cannot be hidden)") + ".</p>" +
        '<p class="aw-note">The memory bandwidth (an H100\'s 3.35 TB/s) and PCIe (25 GB/s) differ 130 times, so offloading is always a last resort: the right use is to move out what has been used and will not be again (the text encoder) and what is cold (a rarely used LoRA or ControlNet). Prefetching needs pinned memory, or the bandwidth more than halves.</p>'
        : "<p>纯计算 " + compute.toFixed(1) + " s（" + nfe + " 次前向 × " + tstep + " s）。模型卸载每次生成搬 <b>" + rows[1][1].toFixed(0) + " GB</b>（每个组件进出各一次），多 " + rows[1][2].toFixed(1) + " s；逐层卸载要搬 <b>" + rows[2][1].toFixed(0) + " GB</b>（去噪网络每一步都走一遍 PCIe），多 " + rows[2][2].toFixed(1) + " s" +
        (rows[3][2] === 0 ? "；预取能把搬运全部藏在计算后面（这张卡算一层比搬一层慢）" : "；预取只能藏掉一部分，仍多 " + rows[3][2].toFixed(1) + " s（算一层比搬一层快，藏不住）") + "。</p>" +
        '<p class="aw-note">显存带宽（H100 3.35 TB/s）和 PCIe（25 GB/s）差 130 倍，所以卸载永远是不得已：正确用法是搬走用过就不再用的东西（文本编码器）和冷的东西（不常用的 LoRA / ControlNet）。预取要锁页内存，否则带宽掉一半以上。</p>');
    });
  }

  // ---------------------------------------------------------------- 跳步：阈值怎么决定跳哪些步
  function cacheSkip(box) {
    var DELTA = [0.422, 0.499, 0.511, 0.489, 0.454, 0.416, 0.378, 0.342, 0.308, 0.274, 0.242, 0.211, 0.181, 0.151, 0.123, 0.095, 0.068, 0.041];   // 正文里测出的"第一块输出的相对变化"，步 1～18
    box.innerHTML = '<div class="aw-title">' + zhen("TeaCache 式跳步：探针的相对变化累加超过阈值才真算，否则复用上一步的残差",
      "TeaCache-style skipping: the probe's relative change accumulates, and only a crossing of the threshold computes for real, otherwise the previous step's residual is reused") + '</div><div class="aw-grid">' +
      row(zhen("阈值", "threshold"), range2("thr", 60, 0, 150)) + row(zhen("首尾各强制计算", "forced at each end"), select("guard", opts(["1 步", "2 步", "0 步"],
        { "1 步": "1 step", "2 步": "2 steps", "0 步": "0 steps" }), "1 步")) + '</div><svg class="aw-chart" viewBox="0 0 560 170"></svg><div class="aw-out"></div>';
    var svg = box.querySelector("svg"), out = box.querySelector(".aw-out");
    bind(box, function () {
      var thr = val(box, "thr") / 100, guard = parseInt(val(box, "guard"), 10), acc = 0, comp = [true], skipped = 0, skippedChange = 0, i;
      show(box, "thr", thr.toFixed(2));
      for (i = 0; i < DELTA.length; i++) {
        var step = i + 1, forced = step < guard || step > DELTA.length - guard;
        acc += DELTA[i];
        if (forced || acc >= thr) { comp.push(true); acc = 0; } else { comp.push(false); skipped++; skippedChange += DELTA[i]; }
      }
      var S = svgText(40, 16, zhen("柱高 = 这一步探针相对上一步的变化；蓝 = 真算，灰 = 复用",
        "bar height = this step's probe change from the previous; blue = computed, grey = reused"), "start"), X = function (i) { return 40 + i * 27; };
      for (i = 0; i <= DELTA.length; i++) {
        var h = i ? DELTA[i - 1] / 0.55 * 100 : 100;
        S += '<rect x="' + X(i) + '" y="' + (130 - h).toFixed(1) + '" width="22" height="' + h.toFixed(1) + '" rx="2" class="' + (comp[i] ? "aw-f" : "aw-off") + '"/>' + svgText(X(i) + 11, 146, i, "middle");
      }
      S += '<line x1="40" y1="' + (130 - thr / 0.55 * 100).toFixed(1) + '" x2="' + (X(DELTA.length) + 22) + '" y2="' + (130 - thr / 0.55 * 100).toFixed(1) + '" class="aw-dash"/>' + svgText(44, (130 - thr / 0.55 * 100) - 4, zhen("阈值 ", "threshold ") + thr.toFixed(2), "start") + svgText(300, 164, zhen("步", "step"), "middle");
      svg.innerHTML = S;
      var nfe = DELTA.length + 1 - skipped;
      out.innerHTML = (EN ? "<p>Of the " + (DELTA.length + 1) + " steps, <b>" + nfe + "</b> are computed and " + skipped + " reused, bringing the network calls down to " + Math.round(nfe / (DELTA.length + 1) * 100) + "% (a speedup of about <b>" + ((DELTA.length + 1) / nfe).toFixed(2) + "x</b>); the skipped steps' accumulated change is " + skippedChange.toFixed(2) + " (the larger it is, the more likely the picture blurs).</p>" +
        '<p class="aw-note">The first and last few steps, where the change is large, are kept by the threshold naturally while the middle is skipped in stretches — the dividend of "two neighbouring steps look too much alike". A threshold of 0.1 to 0.2 is hard to see and above 0.3 starts to blur; a real implementation also fits a polynomial to the accumulated value, and only steps within one request can be reused, so requests in a batch that are out of step cannot skip.</p>'
        : "<p>" + (DELTA.length + 1) + " 步里真算 <b>" + nfe + "</b> 步、复用 " + skipped + " 步，网络调用减少到 " + Math.round(nfe / (DELTA.length + 1) * 100) + "%（加速约 <b>" + ((DELTA.length + 1) / nfe).toFixed(2) + "×</b>）；被跳过的步累计变化量 " + skippedChange.toFixed(2) + "（越大画面越可能糊）。</p>" +
        '<p class="aw-note">变化大的开头几步和收尾几步会被阈值自然保住，中段被成片跳过——这就是"相邻两步长得太像"的红利。阈值 0.1～0.2 肉眼难辨、0.3 以上开始糊；真实实现还会给累加值套一个拟合的多项式，并且只有同一请求的多步才能复用，batch 里的请求不同步时跳不了。</p>');
    });
  }

  // ---------------------------------------------------------------- 视频 token 的三维注意力模式
  function video3d(box) {
    var T = 4, H = 5, W = 7, Q = [1, 2, 3];
    box.innerHTML = '<div class="aw-title">' + zhen("视频 token 是一个 T×H×W 的立方阵：一个 query（橙）在不同注意力模式下看哪些 token（拖动旋转）",
      "Video tokens are a T x H x W cube: which tokens one query (orange) sees under each attention pattern (drag to rotate)") + '</div><div class="aw-grid">' +
      row(zhen("注意力模式", "attention pattern"), select("mode", opts(["3D 全注意力", "分解：空间（同一帧内）", "分解：时间（同一位置跨帧）", "时空窗口（t±1，h±1，w±1）", "空间 + 时间 两步合起来看"],
        { "3D 全注意力": "full 3D attention", "分解：空间（同一帧内）": "factorised: spatial (within one frame)", "分解：时间（同一位置跨帧）": "factorised: temporal (one position across frames)",
          "时空窗口（t±1，h±1，w±1）": "a spatiotemporal window (t±1, h±1, w±1)", "空间 + 时间 两步合起来看": "spatial + temporal, the two steps together" }), "3D 全注意力"), true) +
      '</div><svg class="aw-chart"></svg><div class="aw-out"></div>';
    var out = box.querySelector(".aw-out"), v = view3d(box.querySelector("svg"), { h: 330, scale: 48, ax: 0.5, ay: -0.6, dist: 12 });
    bind(box, function () {
      var mode = val(box, "mode"), items = [], t, h, w, n = 0, total = T * H * W;
      function on(t, h, w) {
        if (t === Q[0] && h === Q[1] && w === Q[2]) return false;
        if (mode.indexOf("3D") === 0) return true;
        if (mode.indexOf("空间（") >= 0) return t === Q[0];
        if (mode.indexOf("时间（") >= 0) return h === Q[1] && w === Q[2];
        if (mode.indexOf("窗口") >= 0) return Math.abs(t - Q[0]) <= 1 && Math.abs(h - Q[1]) <= 1 && Math.abs(w - Q[2]) <= 1;
        return t === Q[0] || (h === Q[1] && w === Q[2]);
      }
      for (t = 0; t < T; t++) {
        var x0 = (t - (T - 1) / 2) * 2.4, c = [[x0, -1.3, -1.6], [x0, 1.3, -1.6], [x0, 1.3, 1.6], [x0, -1.3, 1.6]], k;
        for (k = 0; k < 4; k++) items.push({ t: "seg", a: c[k], b: c[(k + 1) % 4], cls: "aw-dash", z: -50 });
        items.push({ t: "text", p: [x0, -1.75, 0], s: zhen("帧 ", "frame ") + (t + 1), z: 99 });
        for (h = 0; h < H; h++) for (w = 0; w < W; w++) {
          var isQ = t === Q[0] && h === Q[1] && w === Q[2], a = on(t, h, w);
          if (a) n++;
          items.push({ t: "pt", p: [x0 + (w - (W - 1) / 2) * 0.1, (h - (H - 1) / 2) * 0.55, (w - (W - 1) / 2) * 0.46], r: isQ ? 6 : 4, fill: isQ ? "#f08c00" : a ? "#007aff" : "rgba(128,128,128,0.25)", cls: isQ || a ? "aw-p3" : "aw-p3 aw-p3-dim" });
        }
      }
      v.set(items);
      var full = total - 1, note = EN ? (mode.indexOf("3D") === 0 ? "Every token sees all " + full + ": the computation is ∝ (T·H·W)², which in a real model is a hundred thousand tokens squared — this is where attention's seventy or eighty percent of a step comes from." :
        mode.indexOf("空间（") >= 0 ? "Only the same frame: ∝ T·(H·W)², T times cheaper than full attention, but it cannot see any other frame — motion has to come from the temporal attention." :
        mode.indexOf("时间（") >= 0 ? "Only the same position in other frames: ∝ H·W·T², extremely cheap, but it cannot see an object moving from the left to the right." :
        mode.indexOf("窗口") >= 0 ? "Only the neighbouring spatiotemporal blocks: the computation is proportional to the window and independent of the sequence length — the idea behind block-sparse attention, with distant information passed along by stacking layers." :
        "The two steps of factorised attention together still cover only a cross: \"another position at another time\" is never seen, which is why the newer models all moved to full 3D attention.")
        : (mode.indexOf("3D") === 0 ? "每个 token 看全部 " + full + " 个：计算量 ∝ (T·H·W)²，真实模型里是十万 token 的平方——注意力占一步的七八成就是这么来的。" :
        mode.indexOf("空间（") >= 0 ? "只看同一帧：∝ T·(H·W)²，比全注意力便宜 T 倍，但完全看不到别的帧——运动要靠时间注意力补。" :
        mode.indexOf("时间（") >= 0 ? "只看同一位置的其他帧：∝ H·W·T²，极便宜，但一个物体从左边移到右边时它看不见。" :
        mode.indexOf("窗口") >= 0 ? "只看邻近的时空块：计算量和窗口大小成正比，和序列长度无关——块稀疏注意力的思路，远处的信息靠多层叠加传过去。" :
        "分解注意力的两步合起来也只覆盖了一个十字形：\"另一时刻的另一位置\"永远看不到，这就是新模型都换成 3D 全注意力的原因。");
      out.innerHTML = (EN ? "<p>This query sees <b>" + n + "</b> of " + full + " tokens (" + Math.round(n / full * 100) + "%). " + note + "</p>" +
        '<p class="aw-note">There are only 4x5x7 = 140 tokens here; Wan 720p at 81 frames is 21x90x160 = 302,400 latent pixels and 75,600 tokens after patching, and HunyuanVideo at 129 frames is near 120 thousand. What sparse attention does is compute, within full attention, only the ten or twenty percent of blocks in this cube that really matter.</p>'
        : "<p>这个 query 看到 <b>" + n + "</b> / " + full + " 个 token（" + Math.round(n / full * 100) + "%）。" + note + "</p>" +
        '<p class="aw-note">这里只有 4×5×7 = 140 个 token；Wan 720p 81 帧是 21×90×160 = 302,400 个潜像素、打包后 75,600 个 token，HunyuanVideo 129 帧近 12 万。稀疏注意力做的事，就是在全注意力里只算这个立方体里真正相关的那一两成块。</p>');
    });
  }

  // ---------------------------------------------------------------- 视频推理的手段叠加
  function videoStack(box) {
    var PEAK = 989, MFU = 0.47, BASE = { "线性层": 2 * 50 * 1903.0, "注意力": 2 * 50 * 4682.0, "VAE": 25.0 * PEAK * MFU };
    var GPUS4 = { "1 卡": [1, 1], "2 卡：CFG 并行": [1.9, 1], "4 卡：CFG 2 × Ulysses 2": [3.6, 2], "8 卡：CFG 2 × Ulysses 4": [6.8, 4] };
    var ONOFF = { "关": "off", "开": "on" };
    box.innerHTML = '<div class="aw-title">' + zhen("Wan 2.1-14B 720p 81 帧 50 步：把手段一项项叠上去，一次生成从二十多分钟到一分钟量级",
      "Wan 2.1-14B, 720p, 81 frames, 50 steps: stacking the techniques one by one takes a generation from over twenty minutes to about a minute") + '</div><div class="aw-grid">' +
      row(zhen("SageAttention（注意力 2.5×）", "SageAttention (attention 2.5x)"), select("sage", opts(["关", "开"], ONOFF), "关")) +
      row(zhen("FP8 线性层 + 编译（1.4×）", "FP8 linear layers + compilation (1.4x)"), select("fp8", opts(["关", "开"], ONOFF), "关")) +
      row(zhen("块稀疏注意力（4×）", "block-sparse attention (4x)"), select("sparse", opts(["关", "开"], ONOFF), "关")) +
      row(zhen("TeaCache（1.7×）", "TeaCache (1.7x)"), select("tea", opts(["关", "开"], ONOFF), "关")) +
      row(zhen("多卡", "cards"), select("gpus", opts(Object.keys(GPUS4),
        { "1 卡": "1 card", "2 卡：CFG 并行": "2 cards: guidance parallel", "4 卡：CFG 2 × Ulysses 2": "4 cards: guidance 2 x Ulysses 2", "8 卡：CFG 2 × Ulysses 4": "8 cards: guidance 2 x Ulysses 4" }),
        "1 卡"), true) + '</div><svg class="aw-chart" viewBox="0 0 560 70"></svg><div class="aw-out"></div>';
    var svg = box.querySelector("svg"), out = box.querySelector(".aw-out");
    bind(box, function () {
      var sp = { "线性层": 1, "注意力": 1, "VAE": 1 }, g = GPUS4[val(box, "gpus")];
      if (val(box, "sage") === "开") sp["注意力"] *= 2.5;
      if (val(box, "fp8") === "开") sp["线性层"] *= 1.4;
      if (val(box, "sparse") === "开") sp["注意力"] *= 4.0;
      if (val(box, "tea") === "开") { sp["线性层"] *= 1.7; sp["注意力"] *= 1.7; }
      sp["线性层"] *= g[0]; sp["注意力"] *= g[0]; sp["VAE"] *= g[1];
      var t = {}, total = 0, base = 0, k;
      for (k in BASE) { t[k] = BASE[k] / (PEAK * MFU) / sp[k]; total += t[k]; base += BASE[k] / (PEAK * MFU); }
      var S = "", x = 100, cls = { "线性层": "aw-f", "注意力": "aw-b", "VAE": "aw-j2" };
      var PARTS = { "线性层": "linear", "注意力": "attention", "VAE": "VAE" };
      for (k in t) { var w = t[k] / base * 440; S += '<rect x="' + x.toFixed(1) + '" y="8" width="' + Math.max(0, w - 1).toFixed(1) + '" height="22" rx="2" class="' + cls[k] + '"/>'; if (w > 40) S += svgText(x + w / 2, 23, zhen(k, PARTS[k]) + " " + Math.round(t[k]) + " s", "middle"); x += w; }
      S += svgText(8, 23, zhen("一次生成", "one generation"), "start") + svgText(8, 56, zhen("条的长度按基线 " + Math.round(base) + " s 为满格；蓝线性层、橙注意力、紫 VAE",
        "the bar is full at the baseline's " + Math.round(base) + " s; blue linear, orange attention, purple VAE"), "start");
      svg.innerHTML = S;
      out.innerHTML = (EN ? "<p>This combination: <b>" + (total >= 120 ? (total / 60).toFixed(1) + " min" : Math.round(total) + " s") + "</b> (" + Math.round(t["线性层"]) + " s linear, " + Math.round(t["注意力"]) + " s attention, " + Math.round(t["VAE"]) + " s VAE), 1/" + (base / total).toFixed(1) + " of the baseline.</p>" +
        '<p class="aw-note">Stacked ideally it reaches about a minute, while published 8-card figures are 1.5 to 3 minutes — the two or three times between them is pipeline gaps, communication waits, the repeated computation of VAE tiling and a less than ideal utilization, which is the systems engineering. Every item has its preconditions: sparsity needs calibration and a dedicated kernel, TeaCache needs a swept threshold and synchronised decisions across cards, several cards need NVLink; start with the nearly free SageAttention and FP8.</p>'
        : "<p>当前组合：<b>" + (total >= 120 ? (total / 60).toFixed(1) + " 分钟" : Math.round(total) + " 秒") + "</b>（线性层 " + Math.round(t["线性层"]) + " s，注意力 " + Math.round(t["注意力"]) + " s，VAE " + Math.round(t["VAE"]) + " s），是基线的 1/" + (base / total).toFixed(1) + "。</p>" +
        '<p class="aw-note">理想叠加能到一分钟量级，公开的 8 卡实测在 1.5～3 分钟——差的两三倍是 pipeline 空隙、通信等待、VAE 分块重复计算和不理想的 MFU，也就是系统工程的活。每一项都有前提：稀疏要校准和专用 kernel，TeaCache 要扫阈值且多卡要同步决策，多卡要 NVLink；先上几乎免费的 SageAttention 和 FP8。</p>');
    });
  }

  // ================================================================ 推理系统手册（通信、分布式、引擎、前沿）
  // ---------------------------------------------------------------- α-β 模型：消息多大才能跑满带宽
  function alphaBeta(box) {
    var LINKS = {}; [[zhen("NVLink（450 GB/s，α 1 µs）", "NVLink (450 GB/s, α 1 µs)"), [450e9, 1e-6]], [zhen("PCIe 5 x16（55 GB/s，α 2 µs）", "PCIe 5 x16 (55 GB/s, α 2 µs)"), [55e9, 2e-6]], [zhen("IB 400G（50 GB/s，α 3 µs）", "IB 400G (50 GB/s, α 3 µs)"), [50e9, 3e-6]]].forEach(function (p) { LINKS[p[0]] = p[1]; });
    box.innerHTML = '<div class="aw-title">' + zhen("α-β 模型：T(n) = α + n/β，消息多大才能跑满带宽", "The α-β model: T(n) = α + n/β, and how large a message must be to saturate bandwidth") + '</div><div class="aw-grid">' +
      row(zhen("消息大小", "Message size"), range2("n", 20, 10, 30)) + '</div><svg class="aw-chart" viewBox="0 0 560 230"></svg><div class="aw-out"></div>';
    var svg = box.querySelector("svg"), out = box.querySelector(".aw-out");
    function fmt(n) { return n >= 1 << 20 ? (n / (1 << 20)).toFixed(n >= 16 << 20 ? 0 : 1) + " MiB" : (n / 1024).toFixed(0) + " KiB"; }
    bind(box, function () {
      var n = Math.pow(2, val(box, "n")); show(box, "n", fmt(n));
      var X = function (v) { return 60 + (Math.log2(v) - 10) / 20 * 470; }, Y = function (f) { return 190 - f * 160; }, S = "", i, k;
      for (i = 10; i <= 30; i += 4) S += '<line x1="' + X(Math.pow(2, i)) + '" y1="20" x2="' + X(Math.pow(2, i)) + '" y2="190" class="aw-gl"/>' + svgText(X(Math.pow(2, i)), 206, fmt(Math.pow(2, i)), "middle");
      for (i = 0; i <= 4; i++) S += '<line x1="60" y1="' + Y(i / 4) + '" x2="530" y2="' + Y(i / 4) + '" class="aw-gl"/>' + svgText(54, Y(i / 4) + 4, (i * 25) + "%", "end");
      S += '<line x1="60" y1="190" x2="530" y2="190" class="aw-axis"/>' + svgText(300, 224, zhen("消息大小（对数）；纵轴 = 实际达到的带宽 / 链路带宽", "message size (log); y = achieved bandwidth / link bandwidth"), "middle");
      var cols = ["#007aff", "#f08c00", "#af52de"], names = Object.keys(LINKS), html = "";
      for (k = 0; k < names.length; k++) {
        var b = LINKS[names[k]][0], a = LINKS[names[k]][1], d = "";
        for (i = 0; i <= 100; i++) { var v = Math.pow(2, 10 + 20 * i / 100), eff = (v / (a + v / b)) / b; d += (i ? " L " : "M ") + X(v).toFixed(1) + " " + Y(eff).toFixed(1); }
        S += '<path d="' + d + '" fill="none" stroke="' + cols[k] + '" stroke-width="2.2"/>';
        var e = (n / (a + n / b)) / b;
        S += '<circle cx="' + X(n).toFixed(1) + '" cy="' + Y(e).toFixed(1) + '" r="4.5" fill="' + cols[k] + '"/>';
        S += svgText(66, 34 + k * 16, names[k].split(EN ? " (" : "（")[0] + zhen("：半带宽点 α·β = ", ": half-bandwidth point α·β = ") + fmt(a * b), "start");
        html += "<p><span class=\"aw-leg\" style=\"background:" + cols[k] + "\"></span>" + names[k] + zhen("：", ": ") + fmt(n) + zhen(" 的消息用 ", " takes ") + ((a + n / b) * 1e6).toFixed(1) + zhen(" µs，实际带宽 ", " µs, achieving ") + (n / (a + n / b) / 1e9).toFixed(1) + zhen(" GB/s（", " GB/s (") + Math.round(e * 100) + zhen("%），固定开销占 ", "%), with fixed cost at ") + Math.round(a / (a + n / b) * 100) + "%</p>";
      }
      svg.innerHTML = S;
      out.innerHTML = html + '<p class="aw-note">' + zhen("decode 时 TP 的一次 all-reduce 只有几百 KiB（batch 32、hidden 8192、bf16 = 512 KiB），固定开销占一半——所以推理引擎用定制 all-reduce（一步直接读对方显存）、CUDA Graph 把几十次通信录在一起、把通信和 RMSNorm 融合；prefill 的消息上百 MiB，带宽才是瓶颈。", "In decode, one TP all-reduce is only a few hundred KiB (batch 32, hidden 8192, bf16 = 512 KiB), half of it fixed cost, hence custom all-reduce (reading peers' memory in one step), CUDA Graphs capturing dozens of communications together, and fusing communication with RMSNorm; prefill's messages are hundreds of MiB, where bandwidth is the bottleneck.") + '</p>';
    });
  }

  // ---------------------------------------------------------------- 集合通信：ring / tree / one-shot / two-shot / NVLS
  function collectiveCost(box) {
    var LINKS = {}; [[zhen("NVLink（450 GB/s，α 1.5 µs）", "NVLink (450 GB/s, α 1.5 µs)"), [450e9, 1.5e-6, true]], [zhen("IB 400G（50 GB/s，α 3 µs）", "IB 400G (50 GB/s, α 3 µs)"), [50e9, 3e-6, false]], [zhen("PCIe 5（55 GB/s，α 2 µs）", "PCIe 5 (55 GB/s, α 2 µs)"), [55e9, 2e-6, false]]].forEach(function (p) { LINKS[p[0]] = p[1]; });
    box.innerHTML = '<div class="aw-title">' + zhen("一次 all-reduce 有几种走法：按 α-β 模型比较它们的时间", "The ways to run one all-reduce: comparing their times with the α-β model") + '</div><div class="aw-grid">' +
      row(zhen("卡数 n", "GPUs n"), select("n", ["2", "4", "8", "16", "32", "64"], "8")) + row(zhen("链路", "Link"), select("link", Object.keys(LINKS), Object.keys(LINKS)[0]), true) + row(zhen("每卡数据 S", "Data per GPU S"), range2("s", 18, 12, 28)) +
      '</div><svg class="aw-chart" viewBox="0 0 560 190"></svg><div class="aw-out"></div>';
    var svg = box.querySelector("svg"), out = box.querySelector(".aw-out");
    function fmt(n) { return n >= 1 << 20 ? (n / (1 << 20)).toFixed(0) + " MiB" : (n / 1024).toFixed(0) + " KiB"; }
    bind(box, function () {
      var N = +val(box, "n"), L = LINKS[val(box, "link")], B = L[0], A = L[1], S = Math.pow(2, val(box, "s")); show(box, "s", fmt(S));
      var algos = [["ring", 2 * (N - 1) * A + 2 * (N - 1) / N * S / B, zhen("2(n−1) 步，每卡 2(n−1)/n·S", "2(n−1) steps, 2(n−1)/n·S per GPU")], ["tree", 2 * Math.log2(N) * A + 2 * S / B, zhen("2·log₂n 步，每卡 2S", "2·log₂n steps, 2S per GPU")],
                   ["one-shot", 2 * A + (N - 1) * S / B, zhen("1 次同步，读其他卡的全部 (n−1)S", "1 sync, reads (n−1)S from peers")], ["two-shot", 4 * A + 2 * (N - 1) / N * S / B, zhen("2 次同步，RS + AG，每卡 2(n−1)/n·S", "2 syncs, RS + AG, 2(n−1)/n·S per GPU")]];
      if (L[2]) algos.push(["NVLS", 2 * A + S / B, zhen("交换机做加法，每卡发一份收一份", "switch-side add; 1 send, 1 receive")]);
      var mx = 0, i; for (i = 0; i < algos.length; i++) mx = Math.max(mx, algos[i][1]);
      var best = algos.slice().sort(function (a, b) { return a[1] - b[1]; })[0], Sg = "";
      for (i = 0; i < algos.length; i++) {
        var y = 8 + i * 34, w = algos[i][1] / mx * 220;
        Sg += svgText(70, y + 14, algos[i][0], "end") + '<rect x="78" y="' + y + '" width="' + w.toFixed(1) + '" height="20" rx="3" class="' + (algos[i] === best ? "aw-b" : "aw-f") + '"/>' +
          svgText(84 + w, y + 14, (algos[i][1] * 1e6).toFixed(1) + " µs", "start") + '<text x="365" y="' + (y + 14) + '" class="aw-t" font-size="9.5">' + algos[i][2] + "</text>";
      }
      svg.setAttribute("viewBox", "0 0 560 " + (10 + algos.length * 34));
      svg.innerHTML = Sg;
      out.innerHTML = "<p>" + N + zhen(" 卡、每卡 ", " GPUs, ") + fmt(S) + zhen("：最快的是 <b>", " each: the fastest is <b>") + best[0] + "</b>" + zhen("（", " (") + (best[1] * 1e6).toFixed(1) + zhen(" µs）。ring 的 busbw = 2(n−1)/n × S / t = ", " µs). Ring's busbw = 2(n−1)/n × S / t = ") + (2 * (N - 1) / N * S / algos[0][1] / 1e9).toFixed(0) + " GB/s.</p>" +
        '<p class="aw-note">' + zhen("小消息看步数（α 项）：one-shot / two-shot 只同步一两次，所以 decode 的 all-reduce 都走定制 kernel；大消息看每卡要搬多少（β 项）：ring 和 two-shot 都是带宽最优的 2(n−1)/n，one-shot 要读 (n−1)S 直接出局。tree 的步数是 log n，卡多（跨机）时胜出；NVLS 把加法放进交换机，两头都省。", "Small messages are about step count (the α term): one-shot / two-shot sync only once or twice, which is why decode's all-reduce uses custom kernels; large messages are about bytes per GPU (the β term): ring and two-shot are both bandwidth-optimal at 2(n−1)/n, while one-shot reading (n−1)S is out. Tree takes log n steps and wins with many GPUs (across machines); NVLS puts the addition in the switch and saves on both.") + '</p>';
    });
  }

  // ---------------------------------------------------------------- 不填充的 batch：浪费了多少
  function raggedWaste(box) {
    box.innerHTML = '<div class="aw-title">' + zhen("填充成 [B, max_len] 与拼成一条 [N] 的差别", "Padding to [B, max_len] versus packing into one [N]") + '</div><div class="aw-grid">' +
      row(zhen("各请求本步的 token 数", "Tokens per request this step"), '<input type="text" data-k="lens" value="2000, 512, 1, 1, 1, 1, 1, 1" spellcheck="false">', true) + '</div><svg class="aw-chart" viewBox="0 0 560 200"></svg><div class="aw-out"></div>';
    var svg = box.querySelector("svg"), out = box.querySelector(".aw-out");
    bind(box, function () {
      var lens = input(box, "lens").value.split(/[,，\s]+/).filter(function (x) { return x !== ""; }).map(function (x) { return Math.max(0, Math.floor(+x) || 0); }).filter(function (x) { return x > 0; }).slice(0, 16);
      if (!lens.length) { out.innerHTML = "<p>" + zhen("输入若干个正整数。", "Enter a few positive integers.") + "</p>"; svg.innerHTML = ""; return; }
      var B = lens.length, mx = Math.max.apply(null, lens), total = lens.reduce(function (a, b) { return a + b; }, 0), padded = B * mx, S = "", i;
      var rh = Math.min(14, 120 / B), X = function (n) { return 110 + n / mx * 300; };
      S += svgText(8, 16, zhen("填充布局 [", "padded layout [") + B + ", " + mx + "]", "start");
      for (i = 0; i < B; i++) {
        var y = 24 + i * rh;
        S += '<rect x="110" y="' + y + '" width="300" height="' + (rh - 2) + '" class="aw-off"/><rect x="110" y="' + y + '" width="' + (X(lens[i]) - 110).toFixed(1) + '" height="' + (rh - 2) + '" class="aw-on"/>';
        if (rh >= 10) S += svgText(104, y + rh - 4, zhen("请求 ", "request ") + (i + 1) + zhen("：", ": ") + lens[i], "end");
      }
      var y2 = 24 + B * rh + 24, start = 0;
      S += svgText(8, y2 - 8, zhen("拼成一条 [N = ", "packed into one [N = ") + total + zhen("]：query_start_loc = [0, ", "]: query_start_loc = [0, ") + lens.map(function (v) { start += v; return start; }).join(", ") + "]", "start");
      var acc = 0;
      for (i = 0; i < B; i++) { var w = lens[i] / total * 440; S += '<rect x="' + (110 + acc).toFixed(1) + '" y="' + y2 + '" width="' + Math.max(1, w - 1).toFixed(1) + '" height="14" class="' + (i % 2 ? "aw-f" : "aw-on") + '"/>'; acc += w; }
      svg.setAttribute("viewBox", "0 0 560 " + (y2 + 30));
      svg.innerHTML = S;
      out.innerHTML = "<p>" + zhen("填充要算 <b>", "padding computes <b>") + fmtInt(padded) + zhen("</b> 个 token，实际只有 ", "</b> tokens where only ") + fmtInt(total) + zhen(" 个，浪费 <b>", " are real, wasting <b>") + Math.round((1 - total / padded) * 100) + zhen("%</b>；拼成一条之后所有逐 token 的层（嵌入、QKV、MLP、归一化）直接按 [N, d] 算，注意力按 query_start_loc 分段，prefill、decode、分块 prefill 没有任何区别。", "%</b>; packed into one, every per-token layer (embedding, QKV, MLP, norms) computes over [N, d] directly, attention segments by query_start_loc, and prefill, decode and chunked prefill are no different.") + "</p>" +
        '<p class="aw-note">' + zhen("试试全是 decode（8 个 1）：没有浪费但 N 很小，这时瓶颈是读权重，要靠攒 batch；再试一个 8000 加七个 1：没有不填充的布局，这一步就要算 64000 个 token，decode 请求全被拖着。", "Try all decode (eight 1s): nothing is wasted but N is tiny, so the bottleneck is reading weights and only a larger batch helps; then try one 8000 plus seven 1s: without the packed layout this step would compute 64000 tokens, dragging every decode request along.") + '</p>';
    });
  }

  // ---------------------------------------------------------------- kernel 启动开销：eager、CUDA Graph、编译
  function launchOverhead(box) {
    box.innerHTML = '<div class="aw-title">' + zhen("decode 一步有上千个小 kernel：启动开销、CUDA Graph 与算子融合各省多少", "One decode step launches thousands of small kernels: what launch overhead, CUDA Graphs and fusion each save") + '</div><div class="aw-grid">' +
      row(zhen("一步的 kernel 数", "Kernels per step"), num("k", 1617, 10, 20000, 10)) + row(zhen("每个 kernel 的启动开销（µs）", "Launch overhead per kernel (µs)"), num("gap", 5, 0, 50, 0.5)) + row(zhen("每个 kernel 平均执行（µs）", "Average execution per kernel (µs)"), num("run", 4, 0.5, 500, 0.5)) +
      row(zhen("融合后 kernel 数变为", "Kernels after fusion"), select("fuse", ["1/2", "1/3", "1/4"], "1/3")) + '</div><svg class="aw-chart" viewBox="0 0 560 150"></svg><div class="aw-out"></div>';
    var svg = box.querySelector("svg"), out = box.querySelector(".aw-out");
    bind(box, function () {
      var k = val(box, "k"), gap = val(box, "gap"), run = val(box, "run"), f = { "1/2": 2, "1/3": 3, "1/4": 4 }[val(box, "fuse")];
      var eager = k * (gap + run), graph = k * run + 20, comp = Math.ceil(k / f) * (run * 1.15 + gap), both = Math.ceil(k / f) * run * 1.15 + 20;   // 融合后每个 kernel 稍长；图回放一次约 20 µs
      var rows = [[zhen("eager：逐个发射", "eager: one by one"), eager, "aw-off"], [zhen("CUDA Graph：一次回放", "CUDA Graph: one replay"), graph, "aw-f"], [zhen("torch.compile：融合", "torch.compile: fused"), comp, "aw-b"], [zhen("两者都用", "both"), both, "aw-on"]], mx = eager, S = "", i;
      for (i = 0; i < 4; i++) {
        var y = 6 + i * 30, w = rows[i][1] / mx * 300;
        S += svgText(150, y + 14, rows[i][0], "end") + '<rect x="158" y="' + y + '" width="' + w.toFixed(1) + '" height="20" rx="3" class="' + rows[i][2] + '"/>' + svgText(164 + w, y + 14, (rows[i][1] / 1000).toFixed(2) + " ms", "start");
      }
      S += svgText(8, 140, zhen("示意：eager 时每个 kernel 之间都有 CPU 发射的空隙；图回放时 GPU 背靠背执行；融合减少 kernel 个数", "illustrative: in eager mode every kernel has a CPU launch gap; a graph replay runs back to back; fusion reduces the kernel count"), "start");
      svg.innerHTML = S;
      out.innerHTML = "<p>" + zhen("eager 一步 <b>", "one eager step takes <b>") + (eager / 1000).toFixed(2) + zhen(" ms</b>，其中启动开销 ", " ms</b>, of which launch overhead is ") + Math.round(k * gap / eager * 100) + zhen("%（GPU 有一半时间在等 CPU 发射）；CUDA Graph 把它压到 ", "% (the GPU spends half its time waiting on CPU launches); CUDA Graphs squeeze it to ") + (graph / 1000).toFixed(2) + zhen(" ms（", " ms (") + (eager / graph).toFixed(2) + zhen("×），融合再把 kernel 数减到 1/", "×), and fusion cuts the kernel count to 1/") + f + zhen("；两者都用 <b>", "; using both gives <b>") + (both / 1000).toFixed(2) + " ms</b> (" + (eager / both).toFixed(2) + "×).</p>" +
        '<p class="aw-note">' + zhen("把\"每个 kernel 平均执行\"调大（大 batch、长上下文），启动开销的占比就缩小——CUDA Graph 是小 batch decode 的救命稻草，对 prefill 几乎没用。代价：图要求固定形状（按 batch 分桶录制）、固定地址（权重热更新要注意）、不能有 CPU 侧的分支。", "Raise \"average execution per kernel\" (large batches, long contexts) and launch overhead's share shrinks: CUDA Graphs are the lifeline of small-batch decode and nearly useless for prefill. The costs: graphs need fixed shapes (captured in batch buckets), fixed addresses (mind hot weight updates), and no CPU-side branches.") + '</p>';
    });
  }

  // ---------------------------------------------------------------- 线性注意力的显存账本
  function linearMemory(box) {
    box.innerHTML = '<div class="aw-title">' + zhen("混合模型的显存账本：全注意力的 KV 随上下文增长，线性注意力的状态固定", "A hybrid model's memory ledger: full attention's KV grows with context while linear attention's state is fixed") + '</div><div class="aw-grid">' +
      row(zhen("层数", "Layers"), num("L", 48, 4, 200)) + row(zhen("全注意力层占", "Share of full-attention layers"), select("ratio", opts(["全部", "1/2", "1/4", "1/8"], {"全部": "all"}), "1/4")) + row(zhen("全注意力每 token KV（KiB）", "KV per token in full attention (KiB)"), num("kv", 2, 0.5, 64, 0.5)) +
      row(zhen("线性层每请求状态（MiB）", "State per request in linear layers (MiB)"), num("st", 2, 0.25, 64, 0.25)) + '</div><svg class="aw-chart" viewBox="0 0 560 220"></svg><div class="aw-out"></div>';
    var svg = box.querySelector("svg"), out = box.querySelector(".aw-out");
    bind(box, function () {
      var L = val(box, "L"), r = { "全部": 1, "1/2": 0.5, "1/4": 0.25, "1/8": 0.125 }[val(box, "ratio")], kv = val(box, "kv") * 1024, st = val(box, "st") * 1048576, F = Math.round(L * r), LIN = L - F;
      var dense = function (n) { return L * kv * n; }, hybrid = function (n) { return F * kv * n + LIN * st; };
      var X = function (n) { return 60 + (Math.log2(n) - 8) / 12 * 470; }, ymax = dense(1 << 20), Y = function (b) { return 190 - Math.log10(Math.max(b, 1 << 20) / (1 << 20)) / Math.log10(ymax / (1 << 20)) * 165; }, S = "", i;
      [256, 1024, 4096, 16384, 65536, 262144, 1048576].forEach(function (n) { S += '<line x1="' + X(n) + '" y1="20" x2="' + X(n) + '" y2="190" class="aw-gl"/>' + svgText(X(n), 206, n >= 1024 ? (n / 1024) + "K" : n, "middle"); });
      [1, 10, 100, 1000, 10000].forEach(function (m) { if (m * (1 << 20) <= ymax) S += '<line x1="60" y1="' + Y(m * (1 << 20)).toFixed(1) + '" x2="530" y2="' + Y(m * (1 << 20)).toFixed(1) + '" class="aw-gl"/>' + svgText(54, Y(m * (1 << 20)) + 4, m >= 1024 ? (m / 1024).toFixed(0) + " GiB" : m + " MiB", "end"); });
      var d1 = "", d2 = "";
      for (i = 0; i <= 100; i++) { var n = Math.pow(2, 8 + 12 * i / 100); d1 += (i ? " L " : "M ") + X(n).toFixed(1) + " " + Y(dense(n)).toFixed(1); d2 += (i ? " L " : "M ") + X(n).toFixed(1) + " " + Y(hybrid(n)).toFixed(1); }
      S += '<path d="' + d1 + '" class="aw-i" fill="none"/><path d="' + d2 + '" class="aw-j" fill="none"/>' + svgText(66, 34, zhen("蓝：全部用全注意力   橙：混合（KV + 固定状态）", "blue: all full attention\u3000orange: hybrid (KV + fixed state)"), "start") + svgText(300, 218, zhen("上下文长度（token，对数）；纵轴 = 每个请求占的显存（对数）", "context length (tokens, log); y = memory per request (log)"), "middle");
      svg.innerHTML = S;
      var cross = LIN * st / ((L - F) * kv);
      out.innerHTML = "<p>" + zhen(L + " 层里 " + F + " 层全注意力、" + LIN + " 层线性注意力", F + " of " + L + " layers are full attention and " + LIN + " are linear") + zhen("：固定状态共 <b>", ": the fixed state totals <b>") + (LIN * st / 1048576).toFixed(0) + zhen(" MiB</b> / 请求。上下文短于 <b>", " MiB</b> per request. Below <b>") + fmtInt(Math.round(cross)) + zhen("</b> 个 token 时混合模型反而占得更多；32K 上下文时混合是全注意力的 ", "</b> tokens of context the hybrid takes more; at 32K the hybrid is ") + Math.round(hybrid(32768) / dense(32768) * 100) + zhen("%，256K 时 ", "% of full attention, and at 256K ") + Math.round(hybrid(262144) / dense(262144) * 100) + "%.</p>" +
        '<p class="aw-note">' + zhen("状态按请求而不是按 token 分配，所以分页 KV 那套\"按块\"的管理要给它单独开一类池；前缀缓存也要重做：每 B 个 token 存一个状态检查点，B 小时检查点比它覆盖的 KV 大十几倍。", "State is allocated per request rather than per token, so paged KV's \"by block\" management needs a separate pool for it; prefix caching must be redone too: with a state checkpoint every B tokens, a small B makes the checkpoint over ten times larger than the KV it covers.") + '</p>';
    });
  }

  // ---------------------------------------------------------------- 投机解码：负载越高，K 越要小
  function specLoad(box) {
    box.innerHTML = '<div class="aw-title">' + zhen('投机解码的收益随负载变化：batch 大了之后验证不再"免费"', 'Speculative decoding\'s gain shifts with load: verification stops being "free" at larger batches') + '</div><div class="aw-grid">' +
      row("batch", range2("b", 5, 0, 9)) + row(zhen("接受率 α", "Acceptance rate α"), range2("alpha", 80, 40, 95)) + row(zhen("草稿一次前向（ms）", "One draft forward pass (ms)"), num("td", 0.25, 0.02, 5, 0.01)) + row(zhen("出草稿方式", "Drafting"), select("par", opts(["逐个出（K 次草稿前向）", "一次出整块（1 次）"], {"逐个出（K 次草稿前向）": "one at a time (K draft passes)", "一次出整块（1 次）": "a whole block at once (1 pass)"}), "逐个出（K 次草稿前向）")) +
      '</div><svg class="aw-chart" viewBox="0 0 560 200"></svg><div class="aw-out"></div>';
    var svg = box.querySelector("svg"), out = box.querySelector(".aw-out");
    var W_BYTES = 16e9, BW = 3.35e12 * 0.8, FLOP = 2 * 8e9, FLOPS = 989e12 * 0.5;
    function stepTime(tokens) { return Math.max(W_BYTES / BW, tokens * FLOP / FLOPS); }
    bind(box, function () {
      var batch = Math.pow(2, val(box, "b")), alpha = val(box, "alpha") / 100, td = val(box, "td") / 1000, par = val(box, "par").indexOf("一次") === 0;
      show(box, "b", String(batch)); show(box, "alpha", alpha.toFixed(2));
      function thr(k) { var acc = 0, i; for (i = 1; i <= k; i++) acc += Math.pow(alpha, i); var draft = k === 0 ? 0 : (par ? td : k * td); return batch * (1 + acc) / (draft + stepTime(batch * (1 + k))); }
      var base = thr(0), ks = [], best = 0, i, mx = 0;
      for (i = 0; i <= 8; i++) { ks.push(thr(i) / base); if (ks[i] > ks[best]) best = i; mx = Math.max(mx, ks[i]); }
      var S = '<line x1="60" y1="170" x2="530" y2="170" class="aw-axis"/><line x1="60" y1="20" x2="60" y2="170" class="aw-axis"/>', Y = function (v) { return 170 - v / Math.max(1.2, mx) * 145; };
      S += '<line x1="60" y1="' + Y(1).toFixed(1) + '" x2="530" y2="' + Y(1).toFixed(1) + '" class="aw-dash"/>' + svgText(528, Y(1) - 4, zhen("不投机 = 1×", "no speculation = 1×"), "end");
      for (i = 0; i <= 8; i++) { var x = 60 + i / 8 * 460; S += '<rect x="' + (x - 14) + '" y="' + Y(ks[i]).toFixed(1) + '" width="28" height="' + (170 - Y(ks[i])).toFixed(1) + '" rx="3" class="' + (i === best ? "aw-b" : "aw-f") + '"/>' + svgText(x, 186, "K=" + i, "middle") + svgText(x, Y(ks[i]) - 5, ks[i].toFixed(2) + "×", "middle"); }
      svg.innerHTML = S;
      out.innerHTML = "<p>batch " + batch + zhen("：不投机 ", ": without speculation ") + fmtInt(Math.round(base)) + zhen(" tok/s；最佳 <b>K = ", " tok/s; the best <b>K = ") + best + zhen("</b>，加速 <b>", "</b> gives <b>") + ks[best].toFixed(2) + zhen("×</b>。一步前向在 batch×(1+K) = ", "×</b>. At batch×(1+K) = ") + (batch * (1 + best)) + zhen(" 个 token 时", " tokens, one forward pass is ") + (batch * (1 + best) * FLOP / FLOPS > W_BYTES / BW ? zhen("已经算力受限", "already compute-bound") : zhen("仍是访存受限（多算几个 token 几乎免费）", "still memory-bound (a few more tokens are nearly free)")) + ".</p>" +
        '<p class="aw-note">' + zhen("8B 模型、H100 的模型：读一遍权重 ", "An 8B model on an H100: reading the weights once takes ") + (W_BYTES / BW * 1e3).toFixed(1) + zhen(" ms 是每步的下限，token 数超过 ", " ms, the floor per step, and beyond ") + fmtInt(Math.round(W_BYTES / BW * FLOPS / FLOP)) + zhen(" 之后变成算力受限，验证的草稿 token 就要真金白银地花时间。所以最佳 K 随负载下降，高负载时干脆关掉——按负载决定验证多少，是下一节的出发点。", " tokens it becomes compute-bound, so verifying draft tokens costs real time. The best K therefore falls with load, and at high load you turn it off; deciding how much to verify by load is where the next section starts.") + '</p>';
    });
  }

  // ---------------------------------------------------------------- xPyD 配比
  function pdRatio(box) {
    box.innerHTML = '<div class="aw-title">' + zhen("PD 分离的配比：负载形态一变，瓶颈就从一侧跳到另一侧", "The xPyD ratio: when the load's shape changes, the bottleneck jumps from one side to the other") + '</div><div class="aw-grid">' +
      row(zhen("每秒请求数", "Requests per second"), num("rate", 100, 1, 10000)) + row(zhen("平均提示词长度", "Average prompt length"), num("isl", 2000, 1, 200000, 100)) + row(zhen("平均输出长度", "Average output length"), num("osl", 500, 1, 50000, 50)) + row(zhen("前缀缓存命中率", "Prefix cache hit rate"), range2("hit", 60, 0, 95)) +
      row(zhen("prefill 实例每秒 token", "Tokens/s per prefill instance"), num("pcap", 20000, 100, 1000000, 1000)) + row(zhen("decode 实例每秒 token", "Tokens/s per decode instance"), num("dcap", 3000, 10, 100000, 100)) + row(zhen("实例总数", "Total instances"), num("total", 24, 2, 1000)) + row(zhen("固定配比里的 prefill 实例数", "Prefill instances in the fixed ratio"), range2("p", 12, 1, 100)) +
      '</div><svg class="aw-chart" viewBox="0 0 560 80"></svg><div class="aw-out"></div>';
    var svg = box.querySelector("svg"), out = box.querySelector(".aw-out");
    bind(box, function () {
      var rate = val(box, "rate"), isl = val(box, "isl"), osl = val(box, "osl"), hit = val(box, "hit") / 100, pc = val(box, "pcap"), dc = val(box, "dcap"), total = val(box, "total");
      var P = Math.min(val(box, "p"), total - 1); show(box, "hit", hit.toFixed(2)); show(box, "p", String(P));
      var needP = Math.ceil(rate * isl * (1 - hit) / pc), needD = Math.ceil(rate * osl / dc), loadP = rate * isl * (1 - hit) / (P * pc), loadD = rate * osl / ((total - P) * dc);
      function bar(y, label, load) { var w = Math.min(1.6, load) / 1.6 * 300; return svgText(120, y + 14, label, "end") + '<rect x="128" y="' + y + '" width="300" height="20" class="aw-off"/><rect x="128" y="' + y + '" width="' + w.toFixed(1) + '" height="20" class="' + (load > 1 ? "aw-b" : "aw-f") + '"/><line x1="' + (128 + 300 / 1.6) + '" y1="' + (y - 2) + '" x2="' + (128 + 300 / 1.6) + '" y2="' + (y + 22) + '" class="aw-dash"/>' + svgText(434, y + 14, Math.round(load * 100) + "%" + (load > 1 ? zhen("（过载）", " (overloaded)") : ""), "start"); }
      svg.innerHTML = bar(8, zhen(P + " 个 prefill 实例", P + " prefill instances"), loadP) + bar(44, zhen((total - P) + " 个 decode 实例", (total - P) + " decode instances"), loadD);
      out.innerHTML = "<p>" + zhen("这个负载需要 <b>", "this load needs <b>") + needP + "P" + needD + "D</b>" + zhen("（", " (") + (needP + needD <= total ? zhen("够用", "enough") : zhen("共需 " + (needP + needD) + " 个，不够", (needP + needD) + " in total, not enough")) + zhen("）；固定 ", "); with a fixed ") + P + "P" + (total - P) + zhen("D 下 prefill 负载 ", "D, prefill load is ") + Math.round(loadP * 100) + zhen("%、decode 负载 ", "% and decode load ") + Math.round(loadD * 100) + "%" + (loadP > 1 || loadD > 1 ? zhen("，有一侧过载", ", so one side is overloaded") : "") + ".</p>" +
        '<p class="aw-note">' + zhen("对话（输出长、缓存命中高）要 decode，长文档总结（提示词长、命中低）要 prefill，代码智能体两头都要但命中率高——没有一个固定配比能同时满足它们，所以要么按时段改配比（实例在两个角色之间切换），要么不完全分离：提示词短时混合调度，很长时才走 PD。", "Chat (long outputs, high cache hits) wants decode, long-document summarization (long prompts, low hits) wants prefill, and code agents want both with high hits; no fixed ratio satisfies them all, so either change the ratio by time of day (instances switching roles) or don't fully disaggregate: mix when prompts are short and use PD only when they are very long.") + '</p>';
    });
  }

  // ---------------------------------------------------------------- DP attention：最慢的 rank 决定速度
  function dpStraggler(box) {
    box.innerHTML = '<div class="aw-title">' + zhen("DP attention 的负载：每个 rank 的注意力时间由它手里的 KV 总量决定，MoE 的 dispatch 要等最慢的那个", "DP attention's load: each rank's attention time follows the KV it holds, and MoE dispatch waits for the slowest") + '</div><div class="aw-grid">' +
      row(zhen("DP rank 数", "DP ranks"), select("ranks", ["8", "16", "32", "64"], "32")) + row(zhen("平均每 rank 请求数", "Requests per rank"), num("per", 64, 1, 512)) + row(zhen("上下文长度的离散度 σ（对数正态）", "Spread of context length σ (log-normal)"), range2("sigma", 10, 0, 20)) +
      row(zhen("分配方式", "Assignment"), select("policy", opts(["轮流分配（round-robin）", "按 KV 总量分配（最少者优先）"], {"轮流分配（round-robin）": "round-robin", "按 KV 总量分配（最少者优先）": "by total KV (least first)"}), "轮流分配（round-robin）"), true) + '</div><svg class="aw-chart" viewBox="0 0 560 170"></svg><div class="aw-out"></div>';
    var svg = box.querySelector("svg"), out = box.querySelector(".aw-out");
    function rng(seed) { var s = seed >>> 0; return function () { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; }; }
    bind(box, function () {
      var R = +val(box, "ranks"), per = val(box, "per"), sigma = val(box, "sigma") / 10, byKV = val(box, "policy").indexOf("按") === 0, r = rng(11), ctx = [], i, k;
      show(box, "sigma", sigma.toFixed(1));
      for (i = 0; i < R * per; i++) { var g = Math.sqrt(-2 * Math.log(r() + 1e-12)) * Math.cos(2 * Math.PI * r()); ctx.push(Math.min(128000, Math.round(3000 * Math.exp(sigma * g)))); }
      var kv = [], cnt = [];
      for (k = 0; k < R; k++) { kv.push(0); cnt.push(0); }
      for (i = 0; i < ctx.length; i++) {
        var tgt = i % R;
        if (byKV) { tgt = 0; for (k = 1; k < R; k++) if (kv[k] < kv[tgt] && cnt[k] < 2 * per) tgt = k; }
        kv[tgt] += ctx[i]; cnt[tgt] += 1;
      }
      var t = kv.map(function (x) { return (187e6 + x * 1152) / 3.35e12 * 1e6; }), tmax = Math.max.apply(null, t), tmean = t.reduce(function (a, b) { return a + b; }, 0) / R, S = "", bw = 500 / R;
      for (k = 0; k < R; k++) S += '<rect x="' + (40 + k * bw).toFixed(1) + '" y="' + (140 - t[k] / tmax * 120).toFixed(1) + '" width="' + (bw - 1.5).toFixed(1) + '" height="' + (t[k] / tmax * 120).toFixed(1) + '" class="' + (t[k] === tmax ? "aw-b" : "aw-f") + '"/>';
      S += '<line x1="40" y1="' + (140 - tmean / tmax * 120).toFixed(1) + '" x2="540" y2="' + (140 - tmean / tmax * 120).toFixed(1) + '" class="aw-dash"/>' + svgText(40, 20, zhen("每个 rank 一层注意力的时间（µs）；虚线 = 平均；橙 = 最慢的 rank", "one layer of attention per rank (µs); dashed = average; orange = the slowest rank"), "start") + svgText(300, 160, "rank", "middle");
      svg.innerHTML = S;
      out.innerHTML = "<p>" + zhen("最慢的 rank <b>", "the slowest rank takes <b>") + tmax.toFixed(0) + zhen(" µs</b>，平均 ", " µs</b> against an average of ") + tmean.toFixed(0) + zhen(" µs：其他 rank 平均空等 <b>", " µs: the others idle <b>") + (tmax - tmean).toFixed(0) + zhen(" µs</b>（", " µs</b> on average (") + Math.round((1 - tmean / tmax) * 100) + zhen("% 的时间），每一层都要等一次。", "% of the time), once per layer.") + "</p>" +
        '<p class="aw-note">' + zhen("上下文长度是长尾的（对数正态），轮流分配时总有一个 rank 攒到几个超长请求；按 KV 总量分配能把差距压掉大半。真实系统还要考虑在途请求的增长、前缀缓存的亲和性，以及\"请求数上限\"这个硬约束。", "Context lengths are long-tailed (log-normal), so round-robin always leaves one rank with a few very long requests; assigning by total KV removes most of the gap. Real systems must also account for in-flight growth, prefix cache affinity, and the hard cap on requests.") + '</p>';
    });
  }

  // ---------------------------------------------------------------- KV 淘汰：滑动窗口与注意力汇聚
  function kvEvict(box) {
    box.innerHTML = '<div class="aw-title">' + zhen('KV 淘汰：只留最近的 N 个，还是再留住开头几个"汇聚"token', 'KV eviction: keep only the last N, or also hold the first few "sink" tokens') + '</div><div class="aw-grid">' +
      row(zhen("已生成的 token 数", "Tokens generated"), range2("len", 1200, 64, 4096)) + row(zhen("KV 预算", "KV budget"), select("budget", ["128", "256", "512", "1024"], "256")) + row(zhen("保留的开头 token（汇聚）", "Leading tokens kept (sinks)"), select("sinks", ["0", "1", "4", "8"], "4")) +
      '</div><svg class="aw-chart" viewBox="0 0 560 110"></svg><div class="aw-out"></div>';
    var svg = box.querySelector("svg"), out = box.querySelector(".aw-out");
    bind(box, function () {
      var L = val(box, "len"), B = +val(box, "budget"), sinks = +val(box, "sinks"); show(box, "len", String(L));
      var win = Math.max(0, B - sinks), X = function (p) { return 40 + p / L * 480; }, S = '<rect x="40" y="30" width="480" height="24" class="aw-off"/>';
      if (L > B) {
        S += '<rect x="40" y="30" width="' + (X(sinks) - 40).toFixed(1) + '" height="24" class="aw-b"/><rect x="' + X(L - win).toFixed(1) + '" y="30" width="' + (X(L) - X(L - win)).toFixed(1) + '" height="24" class="aw-f"/>';
        S += svgText((X(sinks) + X(L - win)) / 2, 46, zhen("被淘汰的 " + fmtInt(L - B) + " 个", fmtInt(L - B) + " evicted"), "middle");
      } else S += '<rect x="40" y="30" width="480" height="24" class="aw-f"/>' + svgText(280, 46, zhen("还没超过预算，全部保留", "still within budget, all kept"), "middle");
      S += svgText(40, 20, zhen("位置 0", "position 0"), "start") + svgText(520, 20, zhen("位置 " + fmtInt(L) + "（最新）", "position " + fmtInt(L) + " (newest)"), "end") + svgText(40, 80, zhen("橙：开头的汇聚 token　蓝：最近的 " + win + " 个　灰：已淘汰", "orange: the leading sink tokens\u3000blue: the last " + win + "\u3000grey: evicted"), "start");
      S += svgText(40, 100, zhen("每个 token 的位置编码用它在缓存里的相对位置（不是原文位置），窗口才能无限滑下去", "each token is positioned by where it sits in the cache (not in the text), which lets the window slide forever"), "start");
      svg.innerHTML = S;
      out.innerHTML = "<p>" + zhen("缓存里 " + Math.min(L, B) + " 个 token = ", Math.min(L, B) + " tokens in the cache = ") + (L > B ? zhen(sinks + " 个汇聚 + " + win + " 个最近的", sinks + " sinks + the last " + win) : zhen("全部", "all of them")) + zhen("；显存固定为预算的大小，不再随长度增长。", "; memory is fixed at the budget and no longer grows with length.") + "</p>" +
        '<p class="aw-note">' + (sinks === 0 ? zhen("一个汇聚 token 都不留，窗口一滑出开头几个 token，困惑度就会爆掉：模型把大量注意力分数\"存\"在第一个 token 上，它一旦不在，softmax 的分母塌了。", "Keep no sink at all and perplexity explodes as soon as the window slides past the first few tokens: the model \"parks\" a lot of attention on token 0, and without it the softmax denominator collapses.") : zhen("留住开头的几个 token（它们不携带内容，只是注意力的\"垃圾桶\"），窗口怎么滑困惑度都稳定——这就是 StreamingLLM。代价是模型真的看不到窗口外的内容，长文档问答做不了，它解决的是\"无限长对话不崩\"而不是\"记住一切\"。", "Hold the first few tokens (they carry no content, just serving as attention's \"trash bin\") and perplexity stays stable however the window slides; this is StreamingLLM. The cost is that the model truly cannot see outside the window, so long-document Q&A is out: it solves \"an endless conversation that doesn't collapse\", not \"remembering everything\".")) + '</p>';
    });
  }

  // ---------------------------------------------------------------- 一张图变成多少个 token
  function visionTokens(box) {
    box.innerHTML = '<div class="aw-title">' + zhen("一张图变成多少个 token：16×16 的 patch，再把相邻 2×2 个合并", "How many tokens an image becomes: 16×16 patches, then merging each 2×2 group") + '</div><div class="aw-grid">' +
      row(zhen("宽", "Width"), num("w", 896, 32, 8192, 32)) + row(zhen("高", "Height"), num("h", 896, 32, 8192, 32)) + row(zhen("patch 大小", "Patch size"), select("patch", ["14", "16"], "16")) + row(zhen("合并", "Merge"), select("merge", opts(["不合并", "2×2"], {"不合并": "none"}), "2×2")) +
      row(zhen("视觉编码器层数", "Vision encoder layers"), num("L", 12, 1, 64)) + row(zhen("隐藏维", "Hidden size"), num("d", 1024, 64, 8192, 64)) + '</div><svg class="aw-chart" viewBox="0 0 560 150"></svg><div class="aw-out"></div>';
    var svg = box.querySelector("svg"), out = box.querySelector(".aw-out");
    bind(box, function () {
      var w = val(box, "w"), h = val(box, "h"), p = +val(box, "patch"), m = val(box, "merge") === "2×2" ? 2 : 1, L = val(box, "L"), d = val(box, "d");
      var gw = Math.max(m, Math.round(w / (p * m)) * m), gh = Math.max(m, Math.round(h / (p * m)) * m), patches = gw * gh, tokens = patches / (m * m);
      var flops = L * (24 * patches * d * d + 4 * patches * patches * d), S = "", sc = Math.min(140 / gh, 220 / gw, 12), x0 = 40, y0 = 10, i, j;
      for (i = 0; i < gh; i++) for (j = 0; j < gw; j++) S += '<rect x="' + (x0 + j * sc).toFixed(1) + '" y="' + (y0 + i * sc).toFixed(1) + '" width="' + (sc - 0.6).toFixed(1) + '" height="' + (sc - 0.6).toFixed(1) + '" class="' + (((Math.floor(i / m) + Math.floor(j / m)) % 2) ? "aw-f" : "aw-on") + '"/>';
      S += svgText(x0 + gw * sc + 12, 24, gw + " × " + gh + zhen(" 个 patch", " patches"), "start") + svgText(x0 + gw * sc + 12, 44, "→ " + (gw / m) + " × " + (gh / m) + " = " + tokens + zhen(" 个 token", " tokens"), "start") + svgText(x0 + gw * sc + 12, 64, zhen("（棋盘格 = 合并后的 token）", "(checkerboard = merged tokens)"), "start");
      svg.setAttribute("viewBox", "0 0 560 " + Math.max(80, y0 + gh * sc + 10));
      svg.innerHTML = S;
      out.innerHTML = "<p>" + zhen(w + "×" + h + " 的图：按 " + p + " 像素切成 ", "a " + w + "×" + h + " image: cut into " + p + "-pixel patches gives ") + gw + "×" + gh + " = " + patches + zhen(" 个 patch，", " patches, ") + (m > 1 ? zhen("合并 2×2 后 ", "and after merging 2×2, ") : "") + "<b>" + tokens + zhen(" 个图像 token</b>（相当于 ", " image tokens</b> (about as much as ") + Math.round(tokens / 1.5) + "–" + Math.round(tokens / 0.7) + zhen(" 个汉字的文本）。视觉编码器要算约 <b>", " words of text). The vision encoder computes about <b>") + (flops / 1e12).toFixed(2) + zhen(" TFLOP</b>（", " TFLOP</b> (") + L + zhen(" 层、隐藏维 ", " layers, hidden size ") + d + zhen("，注意力在合并前的 " + patches + " 个 patch 上做）。", ", with attention over the " + patches + " patches before merging).") + "</p>" +
        '<p class="aw-note">' + zhen("分辨率翻倍，token 翻 4 倍，语言模型那边的 prefill 和 KV 都跟着翻 4 倍，编码器的注意力翻 16 倍——所以要限制 max_pixels，或者让模型自己选分辨率。图像 token 的编码结果可以缓存（同一张图在多轮对话里反复出现），这就是 encoder cache。", "Double the resolution and tokens quadruple, so the language model's prefill and KV quadruple too while the encoder's attention grows 16×; hence capping max_pixels, or letting the model pick the resolution. The encoded image tokens can be cached (the same image recurs across turns), which is the encoder cache.") + '</p>';
    });
  }

  // ---------------------------------------------------------------- 开环与闭环压测
  function openClosed(box) {
    box.innerHTML = '<div class="aw-title">' + zhen("压测的两种负载模式：固定并发（闭环）与固定到达速率（开环）", "Two load modes for load testing: fixed concurrency (closed loop) and a fixed arrival rate (open loop)") + '</div><div class="aw-grid">' +
      row(zhen("并发槽位（引擎同时跑的请求数）", "Concurrency slots (requests the engine runs at once)"), num("cap", 32, 1, 1024)) + row(zhen("每个请求的平均服务时间（s）", "Average service time per request (s)"), num("svc", 2, 0.1, 60, 0.1)) + row(zhen("闭环：虚拟用户数", "Closed loop: virtual users"), range2("users", 48, 1, 256)) + row(zhen("开环：到达速率（req/s）", "Open loop: arrival rate (req/s)"), range2("rate", 120, 1, 400)) +
      '</div><svg class="aw-chart" viewBox="0 0 560 60"></svg><div class="aw-out"></div>';
    var svg = box.querySelector("svg"), out = box.querySelector(".aw-out");
    function rng(seed) { var s = seed >>> 0; return function () { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; }; }
    function pct(a, q) { var b = a.slice().sort(function (x, y) { return x - y; }); return b[Math.min(b.length - 1, Math.floor(q * b.length))]; }
    bind(box, function () {
      var cap = val(box, "cap"), svc = val(box, "svc"), users = val(box, "users"), rate = val(box, "rate") / 10, T = 600;
      show(box, "users", String(users)); show(box, "rate", rate.toFixed(1));
      function service(r) { return svc * Math.exp(0.4 * (Math.sqrt(-2 * Math.log(r() + 1e-12)) * Math.cos(2 * Math.PI * r())) - 0.08); }
      // 闭环：users 个用户，各自收到结果就立刻发下一个；cap 个槽位，先到先服务
      function closed() {
        var r = rng(3), free = [], next = [], lat = [], i;
        for (i = 0; i < cap; i++) free.push(0);
        for (i = 0; i < users; i++) next.push(0);
        while (true) {
          var u = 0; for (i = 1; i < users; i++) if (next[i] < next[u]) u = i;
          var t = next[u]; if (t > T) break;
          var s = 0; for (i = 1; i < cap; i++) if (free[i] < free[s]) s = i;
          free[s] = Math.max(t, free[s]) + service(r); lat.push(free[s] - t); next[u] = free[s];
        }
        return lat;
      }
      function open() {
        var r = rng(5), free = [], lat = [], t = 0, i;
        for (i = 0; i < cap; i++) free.push(0);
        while (true) {
          t += -Math.log(r() + 1e-12) / rate; if (t > T) break;
          var s = 0; for (i = 1; i < cap; i++) if (free[i] < free[s]) s = i;
          free[s] = Math.max(t, free[s]) + service(r); lat.push(free[s] - t);
        }
        return lat;
      }
      var lc = closed(), lo = open(), capRate = cap / svc;
      svg.innerHTML = svgText(8, 20, zhen("容量 ≈ 槽位 / 服务时间 = ", "capacity ≈ slots / service time = ") + capRate.toFixed(1) + " req/s", "start") +
        '<rect x="280" y="8" width="260" height="16" class="aw-off"/><rect x="280" y="8" width="' + Math.min(260, rate / capRate * 130).toFixed(1) + '" height="16" class="' + (rate > capRate ? "aw-b" : "aw-f") + '"/><line x1="410" y1="4" x2="410" y2="28" class="aw-dash"/>' +
        svgText(410, 44, zhen("开环到达速率 / 容量 = ", "open-loop arrival rate / capacity = ") + Math.round(rate / capRate * 100) + "%", "middle");
      out.innerHTML = "<p>" + zhen("闭环 " + users + " 个用户：吞吐 <b>", "closed loop with " + users + " users: throughput <b>") + (lc.length / T).toFixed(1) + zhen(" req/s</b>，延迟 P50 ", " req/s</b>, latency P50 ") + pct(lc, 0.5).toFixed(1) + zhen(" s、P99 <b>", " s, P99 <b>") + pct(lc, 0.99).toFixed(1) + zhen(" s</b>（在途永远不超过 " + users + "，所以 P99 有界）。", " s</b> (in-flight never exceeds " + users + ", so P99 is bounded).") + "</p>" +
        "<p>" + zhen("开环 " + rate.toFixed(1) + " req/s：吞吐 <b>", "open loop at " + rate.toFixed(1) + " req/s: throughput <b>") + (lo.length / T).toFixed(1) + zhen(" req/s</b>，延迟 P50 ", " req/s</b>, latency P50 ") + pct(lo, 0.5).toFixed(1) + zhen(" s、P99 <b>", " s, P99 <b>") + pct(lo, 0.99).toFixed(1) + " s</b>" + (rate > capRate ? zhen("——超过容量，队列无限增长，P99 随压测时长发散", ": beyond capacity, the queue grows without bound and P99 diverges with the test duration") : "") + ".</p>" +
        '<p class="aw-note">' + zhen("闭环回答\"N 个用户同时用时体验如何\"，系统越慢用户发得越慢，永远测不出过载；开环回答\"每秒来 λ 个请求能不能满足 SLO\"，线上流量就是这样的。容量规划要用开环：找到 SLO 刚好满足的最大 λ，再留两三成余量。", "The closed loop answers \"what is the experience with N simultaneous users\", where a slower system makes users send more slowly, so overload never shows; the open loop answers \"can the SLO be met at λ requests per second\", which is what production traffic looks like. Capacity planning needs the open loop: find the largest λ that just meets the SLO, then leave 20%–30% headroom.") + '</p>';
    });
  }

  // ================================================================ CUDA 进阶手册
  // ---------------------------------------------------------------- online softmax：一次遍历，边走边改最大值
  function onlineSoftmax(box) {
    box.innerHTML = '<div class="aw-title">' + zhen("Online softmax：一次遍历维护 (m, d)，最大值变大时把旧的和按 e^(m−m′) 缩小",
        "Online softmax: one pass keeping (m, d), rescaling the old sum by e^(m−m′) whenever the maximum grows") + '</div><div class="aw-grid">' +
      row(zhen("一行的分数 x", "one row of scores x"), '<input type="text" data-k="xs" value="2.0, 1.0, 5.0, 3.0, 4.5, 5.2, 0.5, 2.5" spellcheck="false">', true) +
      row(zhen("走到第几个", "up to element"), range2("i", 3, 1, 8)) +
      '</div><svg class="aw-chart" viewBox="0 0 560 200"></svg><div class="aw-out"></div>';
    var svg = box.querySelector("svg"), out = box.querySelector(".aw-out");
    bind(box, function () {
      var xs = input(box, "xs").value.split(/[,，\s]+/).filter(function (t) { return t !== ""; }).map(Number).filter(function (v) { return isFinite(v); }).slice(0, 16);
      if (xs.length < 2) { out.innerHTML = "<p>" + zhen("至少给两个数。", "Give at least two numbers.") + "</p>"; svg.innerHTML = ""; return; }
      input(box, "i").max = xs.length; var k = Math.min(val(box, "i"), xs.length); show(box, "i", String(k));
      var m = -Infinity, d = 0, rows = [], j;
      for (j = 0; j < k; j++) { var m2 = Math.max(m, xs[j]), scale = m === -Infinity ? 0 : Math.exp(m - m2), d2 = d * scale + Math.exp(xs[j] - m2); rows.push([xs[j], m2, scale, d2]); m = m2; d = d2; }
      var mx = Math.max.apply(null, xs), S = "", bw = Math.min(40, 480 / xs.length), lo = Math.min.apply(null, xs), span = Math.max(1e-6, mx - lo);
      for (j = 0; j < xs.length; j++) {
        var h = 20 + (xs[j] - lo) / span * 80, x = 40 + j * bw;
        S += '<rect x="' + x + '" y="' + (120 - h).toFixed(1) + '" width="' + (bw - 4) + '" height="' + h.toFixed(1) + '" rx="2" class="' + (j < k ? (j === k - 1 ? "aw-b" : "aw-f") : "aw-off") + '"/>' + svgText(x + bw / 2 - 2, 134, xs[j], "middle");
      }
      S += '<line x1="40" y1="' + (120 - (20 + (m - lo) / span * 80)).toFixed(1) + '" x2="' + (40 + xs.length * bw) + '" y2="' + (120 - (20 + (m - lo) / span * 80)).toFixed(1) + '" class="aw-dash"/>' + svgText(44 + xs.length * bw, (120 - (20 + (m - lo) / span * 80)) + 4, "m = " + m, "start");
      S += svgText(40, 16, zhen("蓝：已经走过的；橙：当前；灰：还没看；虚线：目前的最大值 m",
                                "blue: already seen; orange: current; grey: not yet; dashed: the maximum m so far"), "start");
      S += svgText(40, 160, zhen("每一步：m′ = max(m, x)，d′ = d·e^(m−m′) + e^(x−m′)",
                                 "each step: m′ = max(m, x), d′ = d·e^(m−m′) + e^(x−m′)"), "start") +
           svgText(40, 182, zhen("走完一遍，softmax(x_i) = e^(x_i − m) / d；朴素做法要三遍（求 max、求和、输出）",
                                 "after one pass, softmax(x_i) = e^(x_i − m) / d; the naive way takes three (max, sum, output)"), "start");
      svg.innerHTML = S;
      var last = rows[rows.length - 1], html = "<table class=\"aw-table\"><thead><tr><th>" + zhen("第 i 个", "i") + "</th><th>x</th><th>m′</th><th>" + zhen("缩放 e^(m−m′)", "scale e^(m−m′)") + "</th><th>d′</th></tr></thead><tbody>";
      for (j = 0; j < rows.length; j++) html += "<tr><td>" + (j + 1) + "</td><td>" + rows[j][0] + "</td><td>" + rows[j][1] + "</td><td>" + (j === 0 ? "—" : rows[j][2].toFixed(4)) + "</td><td>" + rows[j][3].toFixed(4) + "</td></tr>";
      html += "</tbody></table>";
      out.innerHTML = '<div class="aw-scroll">' + html + "</div>" + (EN
        ? "<p>After element " + k + ": m = <b>" + last[1] + "</b>, d = <b>" + last[3].toFixed(4) + "</b>" + (k === xs.length ? ". That is the final softmax denominator, identical to the two-pass version (an identity, not an approximation)." : ".") + "</p>" +
          '<p class="aw-note">Two (m, d) pairs also merge: m = max(m₁, m₂), d = d₁e^(m₁−m) + d₂e^(m₂−m). So it reduces in parallel just like a sum: each thread handles its own stretch, then warp shuffles and shared memory combine them level by level. This one property is what lets FlashAttention compute softmax in tiles.</p>'
        : "<p>走到第 " + k + " 个：m = <b>" + last[1] + "</b>，d = <b>" + last[3].toFixed(4) + "</b>" + (k === xs.length ? "；最终 softmax 的分母就是它，和两遍法完全相同（数学上恒等，不是近似）。" : "。") + "</p>" +
          '<p class="aw-note">(m, d) 还能两两合并：m = max(m₁, m₂)，d = d₁e^(m₁−m) + d₂e^(m₂−m)——所以它像求和一样可以并行归约：每个线程先算自己的那段，再 warp shuffle、共享内存逐级合并。FlashAttention 里 softmax 之所以能分块算，靠的就是这一条。</p>');
    });
  }

  // ---------------------------------------------------------------- GEMM 的三级分块：block tile、warp tile、mma tile（3D）
  function gemm3d(box) {
    box.innerHTML = '<div class="aw-title">' + zhen("GEMM 的分块：C 的一个 block tile 沿 K 方向一段一段地累加，block 里再切成 warp tile 和 mma tile（拖动旋转）",
        "Tiling a GEMM: one block tile of C accumulates along K piece by piece, and the block splits further into warp tiles and mma tiles (drag to rotate)") + '</div><div class="aw-grid">' +
      row("BM × BN", select("bmn", ["64 × 64", "128 × 128", "128 × 256", "256 × 128"], "128 × 128")) + row("BK", select("bk", ["32", "64", "128"], "32")) +
      row("warp tile", select("wt", ["32 × 32", "64 × 32", "64 × 64"], "64 × 64")) + row(zhen("当前 K 步", "K step"), range2("ks", 2, 0, 7)) + '</div><svg class="aw-chart"></svg><div class="aw-out"></div>';
    var out = box.querySelector(".aw-out"), v = view3d(box.querySelector("svg"), { h: 340, scale: 44, ax: 0.42, ay: -0.7, dist: 12 });
    var M = 512, N = 512, K = 256, u = 4 / 512;                                   // 画布单位：512 个元素 = 4
    bind(box, function () {
      var bmn = val(box, "bmn").split(" × ").map(Number), BM = bmn[0], BN = bmn[1], BK = +val(box, "bk"), wt = val(box, "wt").split(" × ").map(Number), ks = val(box, "ks");
      var nk = K / BK; ks = Math.min(ks, nk - 1); show(box, "ks", ks + " / " + nk);
      var items = [], i, j;
      function P(m, n, k) { return [(n - N / 2) * u, (M / 2 - m) * u, (k - K / 2) * u * 1.5]; }   // x = N，y = M（向下），z = K
      // 大矩阵 C（M×N）的轮廓和 K 的深度
      var c = [[0, 0, 0], [0, N, 0], [M, N, 0], [M, 0, 0]], e;
      for (i = 0; i < 4; i++) { items.push({ t: "seg", a: P.apply(null, c[i]), b: P.apply(null, c[(i + 1) % 4]), cls: "aw-ax3", z: -50 }); items.push({ t: "seg", a: P(c[i][0], c[i][1], K), b: P(c[(i + 1) % 4][0], c[(i + 1) % 4][1], K), cls: "aw-dash", z: -50 }); items.push({ t: "seg", a: P.apply(null, c[i]), b: P(c[i][0], c[i][1], K), cls: "aw-dash", z: -50 }); }
      items.push({ t: "text", p: P(M + 30, N / 2, 0), s: "N = " + N, z: 99 }); items.push({ t: "text", p: P(M / 2, -40, 0), s: "M = " + M, z: 99 }); items.push({ t: "text", p: P(M + 30, N + 40, K / 2), s: "K = " + K, z: 99 });
      // 一个 block tile：C 里的位置 (BM, BN)，当前 K 步的一段 A、B
      var m0 = BM, n0 = BN, k0 = ks * BK;
      function boxPolys(m1, m2, n1, n2, k1, k2, fill) {
        var q = [[m1, n1, k1], [m1, n2, k1], [m2, n2, k1], [m2, n1, k1], [m1, n1, k2], [m1, n2, k2], [m2, n2, k2], [m2, n1, k2]].map(function (p) { return P(p[0], p[1], p[2]); });
        [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [2, 3, 7, 6], [1, 2, 6, 5], [0, 3, 7, 4]].forEach(function (f) { items.push({ t: "poly", pts: [q[f[0]], q[f[1]], q[f[2]], q[f[3]]], fill: fill, z: 1 }); });
      }
      boxPolys(m0, m0 + BM, n0, n0 + BN, k0, k0 + BK, "#f08c00");                   // 当前 K 步正在算的 block tile 体积
      boxPolys(m0, m0 + BM, n0, n0 + BN, -8, 0, "#007aff");                          // C 里的 block tile（累加结果）
      // warp tile 的分割线（画在 C 面上）
      for (i = wt[0]; i < BM; i += wt[0]) items.push({ t: "seg", a: P(m0 + i, n0, -9), b: P(m0 + i, n0 + BN, -9), cls: "aw-ax3", z: 5 });
      for (j = wt[1]; j < BN; j += wt[1]) items.push({ t: "seg", a: P(m0, n0 + j, -9), b: P(m0 + BM, n0 + j, -9), cls: "aw-ax3", z: 5 });
      items.push({ t: "text", p: P(m0 - 20, n0 + BN / 2, 0), s: "block tile " + BM + "×" + BN, z: 99 });
      v.set(items);
      var warps = (BM / wt[0]) * (BN / wt[1]), mmaPerWarp = (wt[0] / 16) * (wt[1] / 8) * (BK / 16), smem = (BM + BN) * BK * 2, ai = BM * BN / (BM + BN);
      var smemKB = (smem / 1024).toFixed(0), mflop = (2 * BM * BN * BK / 1e6).toFixed(2);
      out.innerHTML = (EN
        ? "<p>A block tile of " + BM + "×" + BN + " taking " + BK + " along K at a time: the block moves " + BM + "×" + BK + " of A and " + BK + "×" + BN + " of B into shared memory (<b>" + smemKB + " KB</b> per stage, times the number of stages when pipelined) and computes 2·" + BM + "·" + BN + "·" + BK + " = " + mflop + " MFLOP, an arithmetic intensity of <b>" + ai.toFixed(0) + " FLOP/byte</b> (= BM·BN/(BM+BN), independent of BK).</p>" +
          "<p>The block holds " + warps + " warps, each owning a " + wt[0] + "×" + wt[1] + " warp tile; per K step each warp issues " + mmaPerWarp + " m16n8k16 mma instructions, with the accumulator living in registers (" + (wt[0] * wt[1] / 32) + " fp32 per thread).</p>" +
          '<p class="aw-note">Bigger tiles mean higher arithmetic intensity and a better-fed Tensor Core, but more shared memory and registers and fewer resident blocks (see occupancy); BK sets how much each pipeline stage moves and how deep the pipeline is. All CUTLASS / CuTe really does is arrange these three levels of tiling and the transfers (cp.async / TMA).</p>'
        : "<p>block tile " + BM + "×" + BN + "，K 方向每次取 " + BK + "：一个 block 要把 A 的 " + BM + "×" + BK + " 和 B 的 " + BK + "×" + BN + " 搬进共享内存（每级 <b>" + smemKB + " KB</b>，多级流水再乘级数），算 2·" + BM + "·" + BN + "·" + BK + " = " + mflop + " MFLOP，算术强度 <b>" + ai.toFixed(0) + " FLOP/字节</b>（= BM·BN/(BM+BN)，与 BK 无关）。</p>" +
          "<p>block 里 " + warps + " 个 warp，每个管 " + wt[0] + "×" + wt[1] + " 的 warp tile；每个 K 步每个 warp 发 " + mmaPerWarp + " 条 m16n8k16 的 mma 指令，累加器常驻寄存器（" + (wt[0] * wt[1] / 32) + " 个 fp32/线程）。</p>" +
          '<p class="aw-note">tile 越大，算术强度越高、越容易打满 Tensor Core，但共享内存和寄存器用得越多、能同时驻留的 block 越少（见占用率）；BK 决定每级流水搬多少、流水多深。CUTLASS / CuTe 的全部工作就是把这三级分块和搬运（cp.async / TMA）排好。</p>');
    });
  }

  // ---------------------------------------------------------------- CuTe 的布局：形状与步长怎么把坐标变成偏移
  function cuteLayout(box) {
    var PRESETS = { "列优先 (4,8):(1,4)": ["(4,8)", "(1,4)"], "行优先 (4,8):(8,1)": ["(4,8)", "(8,1)"], "带 padding (4,8):(1,5)": ["(4,8)", "(1,5)"], "嵌套 ((2,2),(2,4)):((1,8),(2,16))": ["((2,2),(2,4))", "((1,8),(2,16))"], "零步长的广播 (4,8):(1,0)": ["(4,8)", "(1,0)"] };
    var PRESET_EN = { "列优先 (4,8):(1,4)": "column-major (4,8):(1,4)", "行优先 (4,8):(8,1)": "row-major (4,8):(8,1)",
                      "带 padding (4,8):(1,5)": "padded (4,8):(1,5)", "嵌套 ((2,2),(2,4)):((1,8),(2,16))": "nested ((2,2),(2,4)):((1,8),(2,16))",
                      "零步长的广播 (4,8):(1,0)": "broadcast by zero stride (4,8):(1,0)" };
    box.innerHTML = '<div class="aw-title">' + zhen("布局 = 形状 : 步长，坐标 → 偏移；换一个步长就是换一种视图，数据一个字节都不动",
        "A layout is shape : stride, mapping a coordinate to an offset; a different stride is a different view, with not one byte of data moved") + '</div><div class="aw-grid">' +
      row(zhen("例子", "example"), select("preset", opts(Object.keys(PRESETS), PRESET_EN), Object.keys(PRESETS)[0]), true) +
      row(zhen("形状", "shape"), '<input type="text" data-k="shape" value="(4,8)" spellcheck="false">') +
      row(zhen("步长", "stride"), '<input type="text" data-k="stride" value="(1,4)" spellcheck="false">') +
      '</div><svg class="aw-chart" viewBox="0 0 560 200"></svg><div class="aw-out"></div>';
    var svg = box.querySelector("svg"), out = box.querySelector(".aw-out");
    function parse(s) { s = s.replace(/\s+/g, ""); if (!/^[\d(),]+$/.test(s)) throw new Error("x"); return JSON.parse(s.replace(/\(/g, "[").replace(/\)/g, "]").replace(/,\]/g, "]")); }
    function flat(a) { return Array.isArray(a) ? [].concat.apply([], a.map(flat)) : [a]; }
    function size(sh) { return flat(sh).reduce(function (p, q) { return p * q; }, 1); }
    function crd2idx(crd, sh, st) {                           // 整数坐标按列优先拆到各维（第 0 维变化最快）
      if (Array.isArray(sh)) { var idx = 0, i; for (i = 0; i < sh.length; i++) { var n = size(sh[i]); var c = i === sh.length - 1 ? crd : crd % n; idx += crd2idx(c, sh[i], st[i]); crd = Math.floor(crd / n); } return idx; }
      return crd * st;
    }
    box.querySelector('[data-k="preset"]').addEventListener("change", function () { var p = PRESETS[this.value]; input(box, "shape").value = p[0]; input(box, "stride").value = p[1]; draw(); });
    function draw() {
      var sh, st;
      try { sh = parse(input(box, "shape").value); st = parse(input(box, "stride").value); if (flat(sh).length !== flat(st).length) throw new Error("x"); } catch (e) { out.innerHTML = "<p>" + zhen("形状和步长要写成同样结构的元组，比如 (4,8) 和 (1,4)。", "Write the shape and the stride as tuples of the same structure, such as (4,8) and (1,4).") + "</p>"; svg.innerHTML = ""; return; }
      var rows = Array.isArray(sh) ? size(sh[0]) : size(sh), cols = Array.isArray(sh) && sh.length > 1 ? size(sh.slice(1)) : 1, n = rows * cols, cw = Math.min(44, 500 / cols), ch = Math.min(34, 150 / rows), S = "", r, c, maxIdx = 0, used = {}, dup = 0;
      for (r = 0; r < rows; r++) for (c = 0; c < cols; c++) {
        var idx = crd2idx(r + c * rows, sh, st); maxIdx = Math.max(maxIdx, idx); if (used[idx]) dup++; used[idx] = 1;
        S += '<rect x="' + (40 + c * cw) + '" y="' + (26 + r * ch) + '" width="' + (cw - 2) + '" height="' + (ch - 2) + '" rx="3" fill="' + heat(idx / Math.max(1, n - 1)) + '" fill-opacity="0.35"/>' + svgText(40 + c * cw + cw / 2 - 1, 26 + r * ch + ch / 2 + 4, idx, "middle");
      }
      S += svgText(40, 16, zhen("格子 = 坐标 (行, 列)，格子里的数 = 这个坐标落在内存的第几个元素（颜色随偏移变化）",
                                "a cell is a coordinate (row, column); the number in it is which element of memory it lands on (colour follows the offset)"), "start");
      svg.setAttribute("viewBox", "0 0 560 " + (40 + rows * ch));
      svg.innerHTML = S;
      out.innerHTML = (EN
        ? "<p>Size " + n + ", spanning memory 0 to " + maxIdx + (dup ? ", with " + dup + " coordinates landing on the same address (a 0 in the stride: a broadcast, or an overlap)" : maxIdx + 1 === n ? ", compact and one to one" : ", with " + (maxIdx + 1 - n) + " holes in between (padding: to stagger banks, or to align)") + ".</p>" +
          '<p class="aw-note">Over the same memory, (4,8):(1,4) is column-major, (4,8):(8,1) is row-major and (4,8):(1,5) has an extra column of padding. Tiling, transposing and splitting across threads are all just a new layout; a nested shape ((2,2),(2,4)) gives "position within a tile" and "which tile" a dimension each, which is what local_tile / local_partition rest on. How a Tensor Core fragment is spread across registers is a layout too.</p>'
        : "<p>大小 " + n + "，覆盖的内存范围 0～" + maxIdx + (dup ? "，有 " + dup + " 个坐标落到同一个地址（步长里有 0：广播，或者有重叠）" : maxIdx + 1 === n ? "，紧凑、一一对应" : "，中间有 " + (maxIdx + 1 - n) + " 个空洞（padding：错开 bank，或者对齐）") + "。</p>" +
          '<p class="aw-note">同一块内存，(4,8):(1,4) 是列优先、(4,8):(8,1) 是行优先、(4,8):(1,5) 多了一列 padding——分块、转置、按线程划分都只是算一个新布局；嵌套形状 ((2,2),(2,4)) 让"块里的位置"和"块的编号"各占一维，local_tile / local_partition 就靠它。Tensor Core 的 fragment 在寄存器里的分布也是一个布局。</p>');
    }
    bind(box, draw);
  }

  // ---------------------------------------------------------------- 流：拷贝与计算重叠
  function streamOverlap(box) {
    box.innerHTML = '<div class="aw-title">' + zhen("把数据切成几段，用多条流让 H2D 拷贝、kernel、D2H 拷贝互相重叠",
        "Cut the data into chunks and let several streams overlap the H2D copy, the kernel and the D2H copy") + '</div><div class="aw-grid">' +
      row(zhen("分成几段", "chunks"), range2("n", 4, 1, 8)) + row(zhen("流的数量", "streams"), select("s", ["1", "2", "3", "4"], "2")) +
      row(zhen("每段 H2D（ms）", "H2D per chunk (ms)"), num("h2d", 2, 0.1, 50, 0.1)) + row(zhen("每段 kernel（ms）", "kernel per chunk (ms)"), num("k", 3, 0.1, 50, 0.1)) +
      row(zhen("每段 D2H（ms）", "D2H per chunk (ms)"), num("d2h", 2, 0.1, 50, 0.1)) +
      '</div><svg class="aw-chart" viewBox="0 0 560 150"></svg><div class="aw-out"></div>';
    var svg = box.querySelector("svg"), out = box.querySelector(".aw-out");
    bind(box, function () {
      var n = val(box, "n"), s = +val(box, "s"), th = val(box, "h2d"), tk = val(box, "k"), td = val(box, "d2h"); show(box, "n", String(n));
      // 三个引擎（H2D 拷贝引擎、计算、D2H 拷贝引擎）各自串行；同一条流里的操作按顺序；拷贝引擎按流轮转
      var engH = 0, engK = 0, engD = 0, streamT = [], ops = [], i;
      for (i = 0; i < s; i++) streamT.push(0);
      for (i = 0; i < n; i++) {
        var st = i % s, t0 = Math.max(engH, streamT[st]); engH = t0 + th; var t1 = Math.max(engK, engH); engK = t1 + tk; var t2 = Math.max(engD, engK); engD = t2 + td; streamT[st] = engD;
        ops.push([t0, th, 0, i], [t1, tk, 1, i], [t2, td, 2, i]);
      }
      var total = engD, serial = n * (th + tk + td), X = function (t) { return 90 + t / total * 450; }, S = "",
          names = EN ? ["H2D engine", "compute", "D2H engine"] : ["H2D 引擎", "计算", "D2H 引擎"], cls = ["aw-b", "aw-on", "aw-j2"];
      for (i = 0; i < 3; i++) S += svgText(84, 30 + i * 36, names[i], "end") + '<line x1="90" y1="' + (36 + i * 36) + '" x2="540" y2="' + (36 + i * 36) + '" class="aw-gl"/>';
      ops.forEach(function (o) { S += '<rect x="' + X(o[0]).toFixed(1) + '" y="' + (18 + o[2] * 36) + '" width="' + Math.max(1, (o[1] / total * 450) - 1).toFixed(1) + '" height="24" rx="3" class="' + cls[o[2]] + '"/>' + svgText(X(o[0] + o[1] / 2), 34 + o[2] * 36, zhen("段 " + (o[3] + 1), "chunk " + (o[3] + 1)), "middle"); });
      S += svgText(90, 136, "0 ms", "start") + svgText(540, 136, total.toFixed(1) + " ms", "end");
      svg.innerHTML = S;
      var lower = Math.max(n * th, n * tk, n * td).toFixed(1), saved = Math.round((1 - total / serial) * 100);
      out.innerHTML = (EN
        ? "<p>Total <b>" + total.toFixed(1) + " ms</b>; fully serial (one stream, no chunking) would be " + serial.toFixed(1) + " ms, so overlapping saves " + saved + "%. The theoretical floor is the busiest of the three engines: max(" + (n * th).toFixed(1) + ", " + (n * tk).toFixed(1) + ", " + (n * td).toFixed(1) + ") = " + lower + " ms, plus whatever the start and the end cannot fill.</p>" +
          '<p class="aw-note">This needs pinned memory for the copies (cudaMemcpyAsync degrades to synchronous on pageable memory), copies and kernels on different streams, and a GPU with separate copy engines (an H100 has several and can move both directions at once). More chunks mean smaller gaps at the ends, but chunks that are too small bring back the kernel-launch and per-copy overheads.</p>'
        : "<p>总时间 <b>" + total.toFixed(1) + " ms</b>；完全串行（一条流、不分段）要 " + serial.toFixed(1) + " ms，重叠省下 " + saved + "%。理论下限是三种引擎里最忙的那个：max(" + (n * th).toFixed(1) + ", " + (n * tk).toFixed(1) + ", " + (n * td).toFixed(1) + ") = " + lower + " ms，再加首尾填不满的部分。</p>" +
          '<p class="aw-note">前提：拷贝用锁页内存（cudaMemcpyAsync 对可分页内存会退化成同步）、拷贝和 kernel 在不同的流、GPU 有独立的拷贝引擎（H100 有多个，两个方向可以同时）。段数越多首尾的空隙越小，但每段太小时 kernel 启动开销和拷贝的固定开销又上来了。</p>');
    });
  }

  // ---------------------------------------------------------------- 线程索引：blockIdx、threadIdx 与 grid-stride
  function gridIndex(box) {
    box.innerHTML = '<div class="aw-title">' + zhen("哪个线程算哪个元素：i = blockIdx.x × blockDim.x + threadIdx.x",
        "Which thread computes which element: i = blockIdx.x × blockDim.x + threadIdx.x") + '</div><div class="aw-grid">' +
      row(zhen("元素个数 n", "elements n"), range2("n", 22, 1, 64)) + row("blockDim.x", select("bd", ["4", "8", "16"], "8")) +
      row(zhen("启动的 block 数", "blocks launched"), select("gd", opts(["按 n 算（ceil）", "固定 2 个 + grid-stride loop"],
        { "按 n 算（ceil）": "from n (ceil)", "固定 2 个 + grid-stride loop": "a fixed 2 + a grid-stride loop" }), "按 n 算（ceil）"), true) +
      '</div><svg class="aw-chart" viewBox="0 0 560 170"></svg><div class="aw-out"></div>';
    var svg = box.querySelector("svg"), out = box.querySelector(".aw-out");
    bind(box, function () {
      var n = val(box, "n"), bd = +val(box, "bd"), fixed = val(box, "gd").indexOf("固定") === 0, gd = fixed ? 2 : Math.ceil(n / bd), threads = gd * bd, S = "", i, cw = Math.min(16, 500 / Math.max(n, threads));
      show(box, "n", String(n));
      S += svgText(8, 16, zhen("线程（按 block 分色）：", "threads (coloured by block):"), "start");
      for (i = 0; i < threads; i++) S += '<rect x="' + (40 + i * cw) + '" y="24" width="' + (cw - 1.5) + '" height="18" rx="2" class="' + ((Math.floor(i / bd) % 2) ? "aw-f" : "aw-on") + '"/>';
      for (i = 0; i < gd; i++) S += svgText(40 + (i * bd + bd / 2) * cw, 56, "block " + i, "middle");
      S += svgText(8, 84, zhen("元素 a[i]（标着由哪个线程处理）：", "elements a[i] (labelled with the thread handling them):"), "start");
      for (i = 0; i < n; i++) {
        var t = i % threads, over = i >= threads;
        S += '<rect x="' + (40 + i * cw) + '" y="92" width="' + (cw - 1.5) + '" height="18" rx="2" class="' + (over ? "aw-b" : (Math.floor(t / bd) % 2) ? "aw-f" : "aw-on") + '"/>';
        if (cw >= 12) S += '<text x="' + (40 + i * cw + cw / 2) + '" y="105" class="aw-t" font-size="8" text-anchor="middle">' + t + "</text>";
      }
      if (!fixed && threads > n) S += svgText(40 + n * cw + 4, 105, zhen("← 多出 " + (threads - n) + " 个线程要 if (i < n) 挡住",
        "← the extra " + (threads - n) + " threads need if (i < n)"), "start");
      if (fixed) S += svgText(40, 140, zhen("橙色元素由前面的线程在第二、三轮处理：for (i = tid; i < n; i += gridDim.x × blockDim.x)",
        "the orange elements are handled by earlier threads on a second or third round: for (i = tid; i < n; i += gridDim.x × blockDim.x)"), "start");
      else S += svgText(40, 140, zhen("gridDim.x = ceil(n / blockDim.x) = " + gd + "，最后一个 block 可能不满",
        "gridDim.x = ceil(n / blockDim.x) = " + gd + ", and the last block may be partly empty"), "start");
      svg.innerHTML = S;
      out.innerHTML = (EN
        ? "<p>" + gd + " blocks × " + bd + " threads = " + threads + " threads for " + n + " elements" + (fixed ? ": fewer threads than elements, so each loops with a stride over " + Math.ceil(n / threads) + " of them. A grid-stride loop keeps the kernel correct for any n, and the block count can simply follow the SM count." : ": one element per thread" + (threads > n ? ", with the last " + (threads - n) + " threads out of range and filtered by if (i < n)" : "") + ".") + "</p>" +
          '<p class="aw-note">blockDim is usually 128-256 (a multiple of 32, since a warp is 32 threads); memory requests only coalesce when a warp\'s threads touch neighbouring elements. Real kernels often handle 4 elements per thread (float4) on top of the grid stride.</p>'
        : "<p>" + gd + " 个 block × " + bd + " 个线程 = " + threads + " 个线程，" + n + " 个元素" + (fixed ? "：线程比元素少，每个线程跨步循环处理 " + Math.ceil(n / threads) + " 个——grid-stride loop 让 kernel 对任何 n 都正确，block 数按 SM 数量定就行。" : "：每个线程正好处理一个元素" + (threads > n ? "，末尾 " + (threads - n) + " 个线程越界，必须用 if (i < n) 过滤" : "") + "。") + "</p>" +
          '<p class="aw-note">blockDim 一般取 128～256（32 的倍数，一个 warp 32 个线程）；同一个 warp 的线程访问相邻元素时内存请求才能合并。真实的 kernel 常常一个线程处理 4 个元素（float4），再乘上 grid-stride。</p>');
    });
  }

  // ---------------------------------------------------------------- 张量 = storage + sizes + strides
  function strideView(box) {
    var OP_EN = { "原始（连续）": "original (contiguous)", ".t()：转置": ".t(): transpose", "[:, ::2]：隔一列取": "[:, ::2]: every other column",
                  "[1:]：去掉第一行": "[1:]: drop the first row", ".unsqueeze(1).expand(-1, 2, -1)：广播": ".unsqueeze(1).expand(-1, 2, -1): broadcast" };
    box.innerHTML = '<div class="aw-title">' + zhen("张量只是 storage 加上 sizes 和 strides：转置、切片、广播都不动数据，只改元数据",
        "A tensor is just a storage plus sizes and strides: transposing, slicing and broadcasting move no data, only metadata") + '</div><div class="aw-grid">' +
      row(zhen("原始形状", "original shape"), select("shape", ["[3, 4]", "[4, 6]"], "[3, 4]")) +
      row(zhen("操作", "operation"), select("op", opts(["原始（连续）", ".t()：转置", "[:, ::2]：隔一列取", "[1:]：去掉第一行", ".unsqueeze(1).expand(-1, 2, -1)：广播"], OP_EN), ".t()：转置"), true) +
      '</div><svg class="aw-chart" viewBox="0 0 560 170"></svg><div class="aw-out"></div>';
    var svg = box.querySelector("svg"), out = box.querySelector(".aw-out");
    bind(box, function () {
      var sh = JSON.parse(val(box, "shape")), R = sh[0], C = sh[1], op = val(box, "op"), sizes, strides, offset = 0, label;
      if (op.indexOf("原始") === 0) { sizes = [R, C]; strides = [C, 1]; label = zhen("行优先：strides = [C, 1]", "row-major: strides = [C, 1]"); }
      else if (op.indexOf(".t()") === 0) { sizes = [C, R]; strides = [1, C]; label = zhen("转置只是交换 sizes 和 strides", "a transpose just swaps sizes and strides"); }
      else if (op.indexOf("[:, ::2]") === 0) { sizes = [R, Math.ceil(C / 2)]; strides = [C, 2]; label = zhen("切片：列的 stride 变成 2", "a slice: the column stride becomes 2"); }
      else if (op.indexOf("[1:]") === 0) { sizes = [R - 1, C]; strides = [C, 1]; offset = C; label = zhen("切片：storage_offset = C，strides 不变", "a slice: storage_offset = C, strides unchanged"); }
      else { sizes = [R, 2, C]; strides = [C, 0, 1]; label = zhen("expand：新维度的 stride = 0，同一个元素被读两次", "expand: the new dimension has stride 0, so each element is read twice"); }
      var S = svgText(8, 16, zhen("storage（" + R * C + " 个元素）：", "storage (" + R * C + " elements):"), "start"), i, cw = Math.min(22, 500 / (R * C)), reads = {}, contiguous = true, expect = 1, k;
      for (k = sizes.length - 1; k >= 0; k--) { if (sizes[k] !== 1 && strides[k] !== expect) contiguous = false; expect *= sizes[k]; }
      function visit(dims, idx) { if (dims.length === sizes.length) { reads[idx] = (reads[idx] || 0) + 1; return; } for (var q = 0; q < sizes[dims.length]; q++) visit(dims.concat([q]), idx + q * strides[dims.length]); }
      visit([], offset);
      for (i = 0; i < R * C; i++) S += '<rect x="' + (40 + i * cw) + '" y="24" width="' + (cw - 1.5) + '" height="20" rx="2" class="' + (reads[i] ? (reads[i] > 1 ? "aw-b" : "aw-on") : "aw-off") + '"/>' + (cw >= 14 ? '<text x="' + (40 + i * cw + cw / 2) + '" y="38" class="aw-t" font-size="8" text-anchor="middle">' + i + "</text>" : "");
      S += svgText(40, 66, zhen("蓝：这个视图会读到的元素　橙：被读多次　灰：读不到",
        "blue: elements this view reads\u3000orange: read more than once\u3000grey: never reached"), "start");
      // 按视图的行列画一遍，格子里写 storage 下标
      var vr = sizes[0], vc = sizes.length === 3 ? sizes[1] * sizes[2] : sizes[1], cw2 = Math.min(26, 400 / vc), r, c;
      S += svgText(8, 92, zhen("视图 sizes " + JSON.stringify(sizes) + "：", "view sizes " + JSON.stringify(sizes) + ":"), "start");
      for (r = 0; r < vr; r++) for (c = 0; c < vc; c++) {
        var idx = sizes.length === 3 ? offset + r * strides[0] + Math.floor(c / sizes[2]) * strides[1] + (c % sizes[2]) * strides[2] : offset + r * strides[0] + c * strides[1];
        S += '<rect x="' + (180 + c * cw2) + '" y="' + (80 + r * 20) + '" width="' + (cw2 - 1.5) + '" height="18" rx="2" class="aw-f"/>' + '<text x="' + (180 + c * cw2 + cw2 / 2) + '" y="' + (93 + r * 20) + '" class="aw-t" font-size="9" text-anchor="middle">' + idx + "</text>";
      }
      svg.setAttribute("viewBox", "0 0 560 " + Math.max(170, 90 + vr * 20 + 10));
      svg.innerHTML = S;
      out.innerHTML = (EN
        ? "<p>sizes = <b>" + JSON.stringify(sizes) + "</b>, strides = <b>" + JSON.stringify(strides) + "</b>, storage_offset = " + offset + ". " + label + "; " + (contiguous ? "this view is contiguous, so a kernel can read it as a one-dimensional array." : "this view is <b>not contiguous</b>: addresses jump as you walk a row, so many kernels copy with .contiguous() first, or take the generic TensorIterator slow path.") + "</p>" +
          '<p class="aw-note">The address is storage_offset + Σ idx_k × stride_k. expand "copies" one element into many through a stride of 0 (which is how GQA widens KV heads to the query head count, at no memory cost); but it cannot be written in place, nor viewed directly as another shape.</p>'
        : "<p>sizes = <b>" + JSON.stringify(sizes) + "</b>，strides = <b>" + JSON.stringify(strides) + "</b>，storage_offset = " + offset + "。" + label + "；" + (contiguous ? "这个视图是连续的，kernel 可以当一维数组读。" : "这个视图<b>不连续</b>：按行走时地址会跳，很多 kernel 会先 .contiguous() 拷一份，或者走通用的 TensorIterator 慢路径。") + "</p>" +
          '<p class="aw-note">地址 = storage_offset + Σ idx_k × stride_k。expand 用 stride 0 把一个元素"复制"成很多个（GQA 里把 KV 头扩到 query 头数就是这样，不占显存）；但它不能原地写，也不能直接 view 成别的形状。</p>');
    });
  }

  // ================================================================ 分布式训练手册
  // ---------------------------------------------------------------- AdamW：解耦的权重衰减 vs L2 正则
  function adamwStep(box) {
    box.innerHTML = '<div class="aw-title">' + zhen("同样的 weight_decay，Adam + L2 正则和 AdamW 对\"梯度很小\"和\"梯度很大\"的参数各衰减了多少",
        "At the same weight_decay, how much Adam with L2 and AdamW each decay a parameter with a tiny gradient and one with a huge gradient") + '</div><div class="aw-grid">' +
      row(zhen("步数", "steps"), range2("steps", 1000, 100, 3000)) + row(zhen("学习率", "learning rate"), select("lr", ["1e-3", "1e-2", "3e-2"], "1e-2")) + row("weight_decay", select("wd", ["0.01", "0.1", "0.3"], "0.1")) +
      '</div><svg class="aw-chart" viewBox="0 0 560 210"></svg><div class="aw-out"></div>';
    var svg = box.querySelector("svg"), out = box.querySelector(".aw-out");
    bind(box, function () {
      var steps = val(box, "steps"), lr = +val(box, "lr"), wd = +val(box, "wd"), b1 = 0.9, b2 = 0.999, eps = 1e-8; show(box, "steps", String(steps));
      function run(decoupled, scale) {
        var w = 1, m = 0, v = 0, t, hist = [1];
        for (t = 1; t <= steps; t++) {
          var g = scale * (t % 2 ? 1 : -1);                     // 正负交替、平均为 0 的梯度
          if (!decoupled) g += wd * w;                           // L2 正则：衰减项混进梯度，一起被归一化
          m = b1 * m + (1 - b1) * g; v = b2 * v + (1 - b2) * g * g;
          var mh = m / (1 - Math.pow(b1, t)), vh = v / (1 - Math.pow(b2, t));
          w -= lr * mh / (Math.sqrt(vh) + eps);
          if (decoupled) w -= lr * wd * w;                       // AdamW：直接在权重上衰减
          hist.push(w);
        }
        return hist;
      }
      var runs = [[zhen("Adam + L2，小梯度 0.01", "Adam + L2, small gradient 0.01"), run(false, 0.01), "#007aff", "3 3"], [zhen("Adam + L2，大梯度 10", "Adam + L2, large gradient 10"), run(false, 10), "#007aff", ""], [zhen("AdamW，小梯度 0.01", "AdamW, small gradient 0.01"), run(true, 0.01), "#f08c00", "3 3"], [zhen("AdamW，大梯度 10", "AdamW, large gradient 10"), run(true, 10), "#f08c00", ""]];
      var X = function (t) { return 50 + t / steps * 480; }, Y = function (w) { return 180 - Math.max(0, Math.min(1.05, w)) / 1.05 * 160; }, S = "", i, k;
      for (i = 0; i <= 4; i++) S += '<line x1="50" y1="' + Y(i / 4) + '" x2="530" y2="' + Y(i / 4) + '" class="aw-gl"/>' + svgText(44, Y(i / 4) + 4, (i / 4).toFixed(2), "end");
      S += '<line x1="50" y1="180" x2="530" y2="180" class="aw-axis"/>' + svgText(290, 198, zhen("步", "steps"), "middle");
      runs.forEach(function (r) { var d = ""; for (k = 0; k <= steps; k += Math.max(1, Math.floor(steps / 200))) d += (d ? " L " : "M ") + X(k).toFixed(1) + " " + Y(r[1][k]).toFixed(1); S += '<path d="' + d + '" fill="none" stroke="' + r[2] + '" stroke-width="2"' + (r[3] ? ' stroke-dasharray="' + r[3] + '"' : "") + "/>"; });
      S += svgText(56, 20, zhen("权重 w 从 1 开始；蓝：Adam + L2，橙：AdamW；虚线：梯度量级 0.01，实线：梯度量级 10",
        "the weight w starts at 1; blue: Adam + L2, orange: AdamW; dashed: gradient 0.01, solid: gradient 10"), "start");
      svg.innerHTML = S;
      var theory = Math.pow(1 - lr * wd, steps);
      out.innerHTML = EN
        ? "<p>After " + steps + " steps: Adam + L2 decayed the small-gradient parameter to <b>" + runs[0][1][steps].toFixed(3) + "</b> and the large-gradient one to <b>" + runs[1][1][steps].toFixed(3) + "</b>; AdamW took both to <b>" + runs[3][1][steps].toFixed(3) + "</b> (the theoretical (1 − lr·wd)^" + steps + " = " + theory.toFixed(3) + ").</p>" +
          '<p class="aw-note">L2 adds λw to the gradient and then divides it all by √v: a parameter with a large gradient has a large √v and the decay is divided away, while one with a small gradient is decayed hard. The decay\'s strength depends on the gradient\'s magnitude rather than on the λ you set. AdamW takes the decay outside the normalisation so that every parameter shrinks by the same proportion, which is what decoupled means.</p>'
        : "<p>" + steps + " 步后：Adam + L2 把小梯度的参数衰减到 <b>" + runs[0][1][steps].toFixed(3) + "</b>、大梯度的参数到 <b>" + runs[1][1][steps].toFixed(3) + "</b>；AdamW 两者都是 <b>" + runs[3][1][steps].toFixed(3) + "</b>（理论值 (1 − lr·wd)^" + steps + " = " + theory.toFixed(3) + "）。</p>" +
          '<p class="aw-note">L2 正则把 λw 加进梯度之后再被 √v 归一化：梯度大的参数 √v 大，衰减被除没了；梯度小的参数反而被狠狠衰减——衰减强度取决于梯度量级，不是你设的 λ。AdamW 把衰减拿到归一化外面，每个参数都按同样的比例缩，这就是"解耦"。</p>';
    });
  }

  // ---------------------------------------------------------------- Muon：Newton-Schulz 把奇异值推向 1
  function newtonSchulz(box) {
    box.innerHTML = '<div class="aw-title">' + zhen("Newton-Schulz 正交化：对奇异值反复套一个五次多项式，不管开头大小，几步之后都落在 1 附近",
        "Newton-Schulz orthogonalisation: a quintic polynomial applied to the singular values again and again lands them near 1 within a few steps, whatever they started at") + '</div><div class="aw-grid">' +
      row(zhen("迭代次数", "iterations"), range2("k", 2, 0, 6)) + row(zhen("动量矩阵的奇异值分布", "the momentum matrix's singular values"), select("dist", opts(["几个方向主导（1, 0.3, 0.1, 0.03, 0.01）", "比较均匀（1, 0.8, 0.6, 0.5, 0.4）", "极端悬殊（1, 0.1, 0.01, 0.001, 0.0001）"],
        { "几个方向主导（1, 0.3, 0.1, 0.03, 0.01）": "a few directions dominate (1, 0.3, 0.1, 0.03, 0.01)", "比较均匀（1, 0.8, 0.6, 0.5, 0.4）": "fairly even (1, 0.8, 0.6, 0.5, 0.4)",
          "极端悬殊（1, 0.1, 0.01, 0.001, 0.0001）": "wildly uneven (1, 0.1, 0.01, 0.001, 0.0001)" }), "几个方向主导（1, 0.3, 0.1, 0.03, 0.01）"), true) +
      '</div><svg class="aw-chart" viewBox="0 0 560 200"></svg><div class="aw-out"></div>';
    var svg = box.querySelector("svg"), out = box.querySelector(".aw-out"), a = 3.4445, b = -4.7750, c = 2.0315;
    bind(box, function () {
      var k = val(box, "k"), sv = val(box, "dist").match(/[\d.]+/g).map(Number), norm = Math.sqrt(sv.reduce(function (p, q) { return p + q * q; }, 0)), i, j;
      show(box, "k", String(k));
      var x = sv.map(function (s) { return s / norm; }), hist = [x.slice()];
      for (i = 0; i < k; i++) { x = x.map(function (s) { return a * s + b * Math.pow(s, 3) + c * Math.pow(s, 5); }); hist.push(x.slice()); }
      var S = svgText(40, 16, zhen("每根柱子是一个奇异值；左：输入（按 Frobenius 范数缩放到 ≤ 1），右：迭代 " + k + " 次之后",
        "each bar is one singular value; left: the input (scaled to at most 1 by the Frobenius norm), right: after " + k + " iterations"), "start"), n = sv.length;
      for (j = 0; j < 2; j++) {
        var vals = j ? hist[k] : hist[0], x0 = 60 + j * 270;
        S += '<line x1="' + x0 + '" y1="160" x2="' + (x0 + n * 40) + '" y2="160" class="aw-axis"/><line x1="' + x0 + '" y1="' + (160 - 110) + '" x2="' + (x0 + n * 40) + '" y2="' + (160 - 110) + '" class="aw-dash"/>' + svgText(x0 + n * 40 + 4, 54, "1", "start");
        for (i = 0; i < n; i++) { var h = Math.min(1.25, vals[i]) * 110; S += '<rect x="' + (x0 + i * 40) + '" y="' + (160 - h).toFixed(1) + '" width="30" height="' + h.toFixed(1) + '" rx="2" class="' + (j ? "aw-on" : "aw-off") + '"/>' + svgText(x0 + i * 40 + 15, 176, vals[i] < 0.01 ? vals[i].toExponential(0) : vals[i].toFixed(2), "middle"); }
      }
      S += svgText(160, 196, zhen("输入 G / ‖G‖", "the input G / ‖G‖"), "middle") + svgText(430, 196, zhen("NS 迭代 " + k + " 次", k + " Newton-Schulz steps"), "middle");
      svg.innerHTML = S;
      var last = hist[k], spread = Math.max.apply(null, last) / Math.min.apply(null, last);
      out.innerHTML = EN
        ? "<p>After " + k + " iterations the singular values lie between " + Math.min.apply(null, last).toFixed(3) + " and " + Math.max.apply(null, last).toFixed(3) + " (largest / smallest = " + (spread > 1e4 ? spread.toExponential(1) : spread.toFixed(1)) + "), against " + (sv[0] / sv[sv.length - 1]).toFixed(0) + " times at the input. " + (k >= 5 ? "After five steps the directions are kept and the scale is flattened: the update is about UVᵀ." : "A few more iterations and they all stick near 1.") + "</p>" +
          '<p class="aw-note">The polynomial p(s) = 3.4445·s − 4.7750·s³ + 2.0315·s⁵ acts on the singular values using only matrix multiplies (X Xᵀ and its square), with no SVD; the coefficients deliberately converge to a band around 1 rather than exactly 1, which buys fast amplification in the first few steps. A Transformer\'s gradients are dominated by a few directions, and orthogonalisation gives the rare but useful directions the same size of update, which is where Muon\'s saving over AdamW comes from.</p>'
        : "<p>迭代 " + k + " 次后奇异值在 " + Math.min.apply(null, last).toFixed(3) + "～" + Math.max.apply(null, last).toFixed(3) + " 之间（最大 / 最小 = " + (spread > 1e4 ? spread.toExponential(1) : spread.toFixed(1)) + "）；输入时是 " + (sv[0] / sv[sv.length - 1]).toFixed(0) + " 倍。" + (k >= 5 ? "五次之后方向保留、尺度抹平：更新矩阵 ≈ UVᵀ。" : "再多迭代几次就会全部贴到 1 附近。") + "</p>" +
          '<p class="aw-note">多项式 p(s) = 3.4445·s − 4.7750·s³ + 2.0315·s⁵ 作用在奇异值上，只用矩阵乘（X Xᵀ 和它的平方），没有 SVD；系数故意不收敛到正好 1 而是 1 附近的一个区间，换来前几步的快速放大。Transformer 的梯度被少数方向主导，正交化让稀有但有用的方向也得到同样大的更新，这是 Muon 比 AdamW 省步数的来源。</p>';
    });
  }

  // ---------------------------------------------------------------- 学习率调度：warmup、余弦、WSD
  function lrSchedule(box) {
    box.innerHTML = '<div class="aw-title">' + zhen("学习率调度：warmup 之后怎么降", "The learning-rate schedule: how it comes down after the warm-up") + '</div><div class="aw-grid">' +
      row(zhen("总步数", "total steps"), num("T", 10000, 100, 1000000, 100)) + row(zhen("warmup 步数", "warm-up steps"), num("W", 500, 0, 100000, 50)) + row(zhen("调度", "schedule"), select("kind", opts(["余弦衰减到 10%", "WSD：恒定，最后 15% 快速衰减", "恒定", "线性衰减到 0"],
        { "余弦衰减到 10%": "cosine, down to 10%", "WSD：恒定，最后 15% 快速衰减": "warmup-stable-decay: constant, then a fast decay over the last 15%",
          "恒定": "constant", "线性衰减到 0": "linear, down to 0" }), "WSD：恒定，最后 15% 快速衰减"), true) +
      '</div><svg class="aw-chart" viewBox="0 0 560 190"></svg><div class="aw-out"></div>';
    var svg = box.querySelector("svg"), out = box.querySelector(".aw-out");
    bind(box, function () {
      var T = val(box, "T"), W = Math.min(val(box, "W"), T), kind = val(box, "kind");
      function lr(t) {
        if (t < W) return t / Math.max(1, W);
        var p = (t - W) / Math.max(1, T - W);
        if (kind.indexOf("余弦") === 0) return 0.1 + 0.9 * 0.5 * (1 + Math.cos(Math.PI * p));
        if (kind.indexOf("WSD") === 0) return p < 0.85 ? 1 : 1 - (p - 0.85) / 0.15 * 0.95;
        if (kind.indexOf("线性") === 0) return 1 - p;
        return 1;
      }
      var X = function (t) { return 50 + t / T * 480; }, Y = function (v) { return 160 - v * 130; }, d = "", i;
      for (i = 0; i <= 200; i++) d += (i ? " L " : "M ") + X(T * i / 200).toFixed(1) + " " + Y(lr(T * i / 200)).toFixed(1);
      var S = '<line x1="50" y1="160" x2="530" y2="160" class="aw-axis"/><line x1="50" y1="20" x2="50" y2="160" class="aw-axis"/>' + svgText(44, 34, zhen("峰值", "peak"), "end") + svgText(44, 164, "0", "end") + svgText(290, 182, zhen("步", "steps"), "middle");
      if (W > 0) S += '<rect x="50" y="20" width="' + (X(W) - 50).toFixed(1) + '" height="140" class="aw-idle"/>' + svgText(X(W) + 4, 150, zhen("← warmup", "← warm-up"), "start");
      S += '<path d="' + d + '" class="aw-i" fill="none"/>';
      svg.innerHTML = S;
      var avg = 0; for (i = 0; i < 200; i++) avg += lr(T * (i + 0.5) / 200) / 200;
      out.innerHTML = EN
        ? "<p>The average learning rate is " + Math.round(avg * 100) + "% of the peak. " + (kind.indexOf("WSD") === 0 ? "Warmup-stable-decay's constant phase can be continued at any point, or branched at any point into a decayed version for evaluation, which is what continued pretraining and data-proportion experiments rest on; that final fast decay contributes most of the loss reduction." :
            kind.indexOf("余弦") === 0 ? "A cosine has to fix the total step count in advance: training longer means replanning the whole curve, which is why warmup-stable-decay replaced it." : kind.indexOf("恒定") === 0 ? "Without a decay the final loss is clearly higher: the decay phase amounts to averaging the parameters into a flatter region." : "A linear decay to 0 is common in small experiments and works about as well as a cosine.") + "</p>" +
          '<p class="aw-note">Why warm up: Adam\'s early second-moment estimate is inaccurate, so the first step amounts to η·sign(g) with every parameter taking a full step, exactly when a randomly initialised network is fragile. A few thousand warm-up steps plus global gradient clipping (a norm of 1.0) is standard, and how often the clipping fires is itself a health indicator.</p>'
        : "<p>平均学习率是峰值的 " + Math.round(avg * 100) + "%。" + (kind.indexOf("WSD") === 0 ? "WSD 的恒定阶段可以随时接着训练、或从任意一点分叉出一个衰减版本做评估——持续预训练和数据配比实验都靠这个；最后那段快速衰减贡献了大部分的 loss 下降。" :
            kind.indexOf("余弦") === 0 ? "余弦必须事先定好总步数：想多训一段就得重新规划整条曲线，这是 WSD 取代它的原因。" : kind.indexOf("恒定") === 0 ? "不衰减的话最后的 loss 明显偏高：衰减阶段相当于把参数平均到一个更平的区域。" : "线性衰减到 0 在小规模实验里常用，效果和余弦相近。") + "</p>" +
          '<p class="aw-note">warmup 的原因：Adam 早期的二阶矩估计不准，第一步相当于 η·sign(g)，每个参数都走满一步；随机初始化的网络这时很脆。几千步的 warmup 加上全局梯度裁剪（范数 1.0）是标配；裁剪发生的频率本身就是一个健康指标。</p>';
    });
  }

  // ---------------------------------------------------------------- 注意力 logits 变大 → 熵坍缩
  function softmaxEntropy(box) {
    var L0 = [0.8, -0.3, 1.5, 0.2, -1.1, 0.9, 2.2, -0.5, 0.1, 1.0, -0.8, 0.4, 1.8, -0.2, 0.6, -1.4];
    box.innerHTML = '<div class="aw-title">' + zhen("注意力 logits 的尺度与熵：W_q、W_k 的范数一起变大，logits 按乘积增长，softmax 越来越尖",
        "Attention logits' scale and entropy: as W_q's and W_k's norms grow together the logits grow with their product and the softmax sharpens") + '</div><div class="aw-grid">' +
      row(zhen("W_q、W_k 的范数放大倍数", "how much W_q's and W_k's norms are scaled"), range2("s", 10, 2, 60)) + row("QK-Norm", select("qk", opts(["关", "开（q、k 各做一次 RMSNorm）"],
        { "关": "off", "开（q、k 各做一次 RMSNorm）": "on (an RMSNorm on each of q and k)" }), "关"), true) + '</div><svg class="aw-chart" viewBox="0 0 560 190"></svg><div class="aw-out"></div>';
    var svg = box.querySelector("svg"), out = box.querySelector(".aw-out");
    bind(box, function () {
      var s = val(box, "s") / 10, qk = val(box, "qk").indexOf("开") === 0, scale = qk ? 1 : s * s, i; show(box, "s", s.toFixed(1) + "×");
      var z = L0.map(function (v) { return v * scale; }), m = Math.max.apply(null, z), ex = z.map(function (v) { return Math.exp(v - m); }), Z = ex.reduce(function (p, q) { return p + q; }, 0), p = ex.map(function (v) { return v / Z; });
      var H = 0; for (i = 0; i < p.length; i++) if (p[i] > 0) H -= p[i] * Math.log(p[i]);
      var S = svgText(40, 16, zhen("一个 query 对 16 个 key 的注意力权重；logits 的最大值 " + m.toFixed(1),
        "one query's attention weights over 16 keys; the largest logit is " + m.toFixed(1)), "start");
      for (i = 0; i < 16; i++) S += '<rect x="' + (40 + i * 30) + '" y="' + (160 - p[i] * 130).toFixed(1) + '" width="24" height="' + (p[i] * 130).toFixed(1) + '" rx="2" class="' + (p[i] === Math.max.apply(null, p) ? "aw-b" : "aw-f") + '"/>' + svgText(52 + i * 30, 176, i, "middle");
      S += '<line x1="40" y1="160" x2="520" y2="160" class="aw-axis"/>';
      svg.innerHTML = S;
      out.innerHTML = EN
        ? "<p>The entropy is <b>" + H.toFixed(2) + "</b> (uniform is " + Math.log(16).toFixed(2) + " and looking at one key is 0) and the largest weight is <b>" + Math.max.apply(null, p).toFixed(2) + "</b>. " + (qk ? "QK-Norm fixes q's and k's norms, so the logits no longer grow with W_q's and W_k's: pull the scale all the way up and nothing moves." : scale > 20 ? "The logits are now so large that the softmax is nearly an argmax: the gradient vanishes and this head can no longer learn, which is entropy collapse." : "Reasonably healthy.") + "</p>" +
          '<p class="aw-note">logits = q·k/√d, and q\'s and k\'s norms both grow linearly with the weights\' norms, so the logits grow quadratically; tens of thousands of steps later a few heads collapse to one-hot and the loss spikes. QK-Norm (Qwen3, Gemma and others), soft-capping the attention logits (Gemma 2\'s tanh) and weight decay all address this one thing.</p>'
        : "<p>熵 <b>" + H.toFixed(2) + "</b>（均匀分布是 " + Math.log(16).toFixed(2) + "，只看一个是 0），最大权重 <b>" + Math.max.apply(null, p).toFixed(2) + "</b>。" + (qk ? "QK-Norm 把 q、k 的范数固定住，logits 不再随 W_q、W_k 的范数增长——放大倍数拉到头也没变化。" : scale > 20 ? "logits 已经大到 softmax 几乎变成 argmax：梯度消失，这个头再也学不动，这就是熵坍缩。" : "还算健康。") + "</p>" +
          '<p class="aw-note">logits = q·k/√d，q、k 的范数都随权重范数线性涨，所以 logits 按平方涨；几万步之后少数头会塌成 one-hot，loss 出现突刺。QK-Norm（Qwen3、Gemma 等）、注意力 logits 软截断（Gemma 2 的 tanh）、权重衰减都是在管这一件事。</p>';
    });
  }

  // ---------------------------------------------------------------- GRPO：组内优势与损失平均
  function grpoAdv(box) {
    box.innerHTML = '<div class="aw-title">' + zhen("GRPO 的组内优势，以及\"按回答平均\"和\"按 token 平均\"各给每个 token 多大的权重",
        "GRPO's group-relative advantage, and the weight each token gets under averaging by answer and by token") + '</div><div class="aw-grid">' +
      row(zhen("一组回答的奖励", "the group's rewards"), '<input type="text" data-k="r" value="1, 0, 0, 1" spellcheck="false">') + row(zhen("各回答的长度", "the answers' lengths"), '<input type="text" data-k="len" value="200, 50, 2000, 400" spellcheck="false">') +
      row(zhen("优势归一化", "advantage normalisation"), select("norm", opts(["减均值", "减均值再除以标准差"], { "减均值": "subtract the mean", "减均值再除以标准差": "subtract the mean, then divide by the standard deviation" }), "减均值再除以标准差")) +
      row(zhen("损失平均方式", "how the loss is averaged"), select("agg", opts(["序列平均（GRPO）", "token 平均（DAPO）"], { "序列平均（GRPO）": "by sequence (GRPO)", "token 平均（DAPO）": "by token (DAPO)" }), "序列平均（GRPO）")) + '</div><div class="aw-out"></div>';
    var out = box.querySelector(".aw-out");
    bind(box, function () {
      var R = input(box, "r").value.split(/[,，\s]+/).filter(Boolean).map(Number), L = input(box, "len").value.split(/[,，\s]+/).filter(Boolean).map(Number), G = Math.min(R.length, L.length), i;
      if (G < 2 || R.some(isNaN) || L.some(function (v) { return !(v > 0); })) { out.innerHTML = "<p>" + zhen("奖励和长度各给至少两个数。", "Give at least two numbers for the rewards and for the lengths.") + "</p>"; return; }
      R = R.slice(0, G); L = L.slice(0, G);
      var mean = R.reduce(function (p, q) { return p + q; }, 0) / G, sd = Math.sqrt(R.reduce(function (p, q) { return p + (q - mean) * (q - mean); }, 0) / Math.max(1, G - 1)), useSd = val(box, "norm").indexOf("再") >= 0, seq = val(box, "agg").indexOf("序列") === 0;
      var adv = R.map(function (r) { return useSd ? (r - mean) / (sd + 1e-6) : r - mean; }), total = L.reduce(function (p, q) { return p + q; }, 0);
      var head = EN ? ["answer", "reward", "length", "advantage", "weight per token", "the answer's total weight"] : ["回答", "奖励", "长度", "优势", "每个 token 的权重", "整条回答的总权重"];
      var html = "<table class=\"aw-table\"><thead><tr><th>" + head.join("</th><th>") + "</th></tr></thead><tbody>";
      for (i = 0; i < G; i++) { var w = seq ? adv[i] / L[i] / G : adv[i] / total; html += "<tr><td>" + (i + 1) + "</td><td>" + R[i] + "</td><td>" + L[i] + "</td><td>" + adv[i].toFixed(2) + "</td><td>" + w.toExponential(2) + "</td><td>" + (w * L[i]).toFixed(3) + "</td></tr>"; }
      html += "</tbody></table>";
      var allSame = adv.every(function (a) { return Math.abs(a) < 1e-9; });
      out.innerHTML = EN
        ? '<div class="aw-scroll">' + html + "</div><p>" + (allSame ? "<b>An all-right or all-wrong group has advantages of 0 throughout and contributes nothing to the gradient.</b> DAPO's dynamic sampling filters such groups out and resamples." :
            seq ? "By sequence: each answer's total weight depends only on its advantage and not on its length, so a short answer's tokens get far more weight each and a long answer's are diluted (the length bias)." : "By token: every token has the same weight and a long answer's share of the loss grows with its length. DAPO uses this to stop long answers being diluted, at the cost of letting them dominate the gradient.") +
            (useSd ? " Dividing by the standard deviation amplifies the advantage on hard questions (where the reward's variance is small) and shrinks it on easy ones; Dr. GRPO calls this a difficulty bias and removes it." : "") + "</p>" +
          '<p class="aw-note">No critic: the baseline is the mean reward of G answers to the same question, so a larger G gives a more accurate baseline but one question\'s rollouts have to be done together. The inference engine samples and returns by group, which is one of the things reinforcement-learning training asks of it.</p>'
        : '<div class="aw-scroll">' + html + "</div><p>" + (allSame ? "<b>全对或全错的组优势全是 0，这一组对梯度没有任何贡献</b>——DAPO 的动态采样就是把这种组过滤掉再补采。" :
            seq ? "序列平均：每条回答的总权重只取决于优势，和长度无关——短回答里每个 token 分到的权重大得多，长回答的 token 被稀释（长度偏差）。" : "token 平均：每个 token 权重相同，长回答在损失里占的份额按长度变大——DAPO 用它避免长回答被稀释，但也让长回答主导梯度。") +
            (useSd ? " 除以标准差把难题（奖励方差小）的优势放大、简单题缩小，Dr. GRPO 认为这是一种题目难度偏差，去掉了它。" : "") + "</p>" +
          '<p class="aw-note">没有 critic：基线就是同一个问题 G 个回答的平均奖励，所以 G 越大基线越准，但一个问题的 rollout 要一起做；推理引擎按组采样、按组回传，这也是 RL 训练对推理端的要求之一。</p>';
    });
  }

  // ---------------------------------------------------------------- KL 的三种单样本估计量
  function klEstimators(box) {
    box.innerHTML = '<div class="aw-title">' + zhen("KL 惩罚的三种单样本估计量：令 r = p_ref(x) / q(x)，x 从当前策略 q 采样",
        "The KL penalty's three single-sample estimators, with r = p_ref(x) / q(x) and x sampled from the current policy q") + '</div><div class="aw-grid">' +
      row(zhen("当前这个 token 的 r", "this token's r"), range2("r", 100, 20, 300), true) + '</div><svg class="aw-chart" viewBox="0 0 560 210"></svg><div class="aw-out"></div>';
    var svg = box.querySelector("svg"), out = box.querySelector(".aw-out");
    bind(box, function () {
      var r = val(box, "r") / 100; show(box, "r", r.toFixed(2));
      var fns = [["k1 = −log r", function (x) { return -Math.log(x); }, "#8e8e93"], ["k2 = (log r)² / 2", function (x) { return Math.log(x) * Math.log(x) / 2; }, "#007aff"], ["k3 = (r − 1) − log r", function (x) { return x - 1 - Math.log(x); }, "#f08c00"]];
      var X = function (x) { return 50 + (x - 0.2) / 2.8 * 480; }, Y = function (v) { return 170 - (v + 1.2) / 3.2 * 150; }, S = "", i, k;
      for (i = 0; i <= 3; i++) S += '<line x1="' + X(i || 0.2) + '" y1="20" x2="' + X(i || 0.2) + '" y2="170" class="aw-gl"/>' + svgText(X(i || 0.2), 186, i || 0.2, "middle");
      S += '<line x1="50" y1="' + Y(0) + '" x2="530" y2="' + Y(0) + '" class="aw-axis"/>' + svgText(44, Y(0) + 4, "0", "end") + svgText(290, 204, "r = p_ref / q", "middle");
      fns.forEach(function (f, j) {
        var d = ""; for (k = 0; k <= 140; k++) { var x = 0.2 + 2.8 * k / 140, v = Math.max(-1.2, Math.min(2, f[1](x))); d += (k ? " L " : "M ") + X(x).toFixed(1) + " " + Y(v).toFixed(1); }
        S += '<path d="' + d + '" fill="none" stroke="' + f[2] + '" stroke-width="2"/>' + '<circle cx="' + X(r).toFixed(1) + '" cy="' + Y(Math.max(-1.2, Math.min(2, f[1](r)))).toFixed(1) + '" r="4" fill="' + f[2] + '"/>' + svgText(56, 34 + j * 16, f[0], "start");
      });
      svg.innerHTML = S;
      out.innerHTML = EN
        ? "<p>At r = " + r.toFixed(2) + ": k1 = " + fns[0][1](r).toFixed(3) + ", k2 = " + fns[1][1](r).toFixed(3) + ", k3 = " + fns[2][1](r).toFixed(3) + ". " + (r < 1 ? "The current policy prefers this token more than the reference model does (q > p_ref)." : r > 1 ? "The current policy likes this token less than the reference model does." : "The two models agree and all three estimates are 0.") + "</p>" +
          '<p class="aw-note">Taking the expectation over x ~ q: k1 is unbiased but negative half the time and high-variance; k2 is always non-negative and low-variance but biased; k3 = k1 + (r − 1), and the added term has an expectation of 0, so it is unbiased, non-negative and low-variance, which is what GRPO uses. All three are only reliable while q is not far from p_ref; the KL coefficient, and whether to have a KL at all (DAPO dropped it), trade off not going too far against not being held back.</p>'
        : "<p>r = " + r.toFixed(2) + "：k1 = " + fns[0][1](r).toFixed(3) + "，k2 = " + fns[1][1](r).toFixed(3) + "，k3 = " + fns[2][1](r).toFixed(3) + "。" + (r < 1 ? "当前策略比参考模型更偏爱这个 token（q > p_ref）。" : r > 1 ? "当前策略比参考模型更不爱这个 token。" : "两个模型一致，三个估计都是 0。") + "</p>" +
          '<p class="aw-note">对 x ~ q 取期望：k1 无偏但一半时候是负的、方差大；k2 永远非负、方差小但有偏；k3 = k1 + (r − 1)，加的那一项期望为 0，所以无偏、又非负、方差也小——GRPO 用的就是 k3。它们只有在 q 离 p_ref 不远时才可靠；KL 系数、要不要 KL（DAPO 去掉了）都是在"别跑太远"和"别被拖住"之间取舍。</p>';
    });
  }

  // ---------------------------------------------------------------- 训练要多久：6ND / (卡数 × 峰值 × MFU)
  function trainTime(box) {
    box.innerHTML = '<div class="aw-title">' + zhen("训练时间 = 6ND / (卡数 × 单卡峰值 × MFU)", "Training time = 6ND / (cards x the peak per card x the model FLOPs utilization)") + '</div><div class="aw-grid">' +
      row(zhen("参数量 N（B）", "parameters N (B)"), num("N", 8, 0.01, 2000, 0.1)) + row(zhen("token 数 D（B）", "tokens D (B)"), num("D", 15000, 1, 100000000, 100)) + row(zhen("卡数", "cards"), num("G", 1024, 1, 1000000)) + row("GPU", select("gpu", Object.keys(GPUS), "H100 SXM")) + row("MFU", range2("mfu", 40, 10, 70)) +
      '</div><div class="aw-out"></div>';
    var out = box.querySelector(".aw-out");
    bind(box, function () {
      var N = val(box, "N") * 1e9, D = val(box, "D") * 1e9, G = val(box, "G"), g = GPUS[val(box, "gpu")], mfu = val(box, "mfu") / 100; show(box, "mfu", mfu.toFixed(2));
      var flops = 6 * N * D, secs = flops / (G * g[2] * 1e12 * mfu), days = secs / 86400, gpuDays = days * G;
      var LOC = EN ? "en-US" : "zh-CN";
      out.innerHTML = EN
        ? "<p>The total compute 6ND = <b>" + flops.toExponential(2) + " FLOP</b>; on " + G + " " + val(box, "gpu") + " at " + Math.round(mfu * 100) + "% utilization that is <b>" + (days >= 2 ? days.toFixed(1) + " days" : (secs / 3600).toFixed(1) + " hours") + "</b>, " + Math.round(gpuDays).toLocaleString(LOC) + " card-days in all; training on one token costs 6N = " + (6 * N / 1e9).toFixed(1) + " GFLOP, three times inference's 2N.</p>" +
          '<p class="aw-note">The utilization sits between 30% and 50%: what is lost goes to unhidden communication, the pipeline bubble, recomputation\'s forward passes (which do not count toward it), small kernels and CPU overhead. Doubling the cards does not necessarily halve the time, because the communication\'s share rises and the utilization falls. That is why the parallelism has to be combined with care.</p>'
        : "<p>总计算量 6ND = <b>" + flops.toExponential(2) + " FLOP</b>；" + G + " 张 " + val(box, "gpu") + "、MFU " + Math.round(mfu * 100) + "%：<b>" + (days >= 2 ? days.toFixed(1) + " 天" : (secs / 3600).toFixed(1) + " 小时") + "</b>，共 " + Math.round(gpuDays).toLocaleString(LOC) + " 卡·天；每个 token 的训练成本 6N = " + (6 * N / 1e9).toFixed(1) + " GFLOP，是推理（2N）的 3 倍。</p>" +
          '<p class="aw-note">MFU 30%～50% 之间：丢掉的部分是没藏住的通信、流水线气泡、重计算的前向（不算进 MFU）、小 kernel 和 CPU 开销。把卡数翻倍时间未必减半——通信占比上升，MFU 会掉；这就是并行策略要精心组合的原因。</p>';
    });
  }

  // ---------------------------------------------------------------- DDP：按桶 all-reduce，和反向重叠
  function ddpOverlap(box) {
    box.innerHTML = '<div class="aw-title">' + zhen("DDP 把梯度按桶发：一个桶的梯度到齐就立刻 all-reduce，网卡在传后面层的梯度时 GPU 在算前面层",
        "DDP sends the gradients by bucket: a bucket all-reduces as soon as it is full, so the card moves the later layers' gradients while the GPU computes the earlier ones") + '</div><div class="aw-grid">' +
      row(zhen("层数", "layers"), range2("L", 12, 2, 32)) + row(zhen("每层反向耗时（ms）", "backward per layer (ms)"), num("tb", 4, 0.5, 100, 0.5)) + row(zhen("每层梯度（MB）", "gradients per layer (MB)"), num("mb", 25, 1, 1000)) +
      row(zhen("桶大小（MB）", "bucket size (MB)"), select("bucket", opts(["25", "50", "100", "不分桶（等反向结束）"], { "不分桶（等反向结束）": "no buckets (wait for the backward pass)" }), "25")) + row(zhen("有效带宽（GB/s）", "effective bandwidth (GB/s)"), num("bw", 20, 1, 500)) +
      '</div><svg class="aw-chart" viewBox="0 0 560 110"></svg><div class="aw-out"></div>';
    var svg = box.querySelector("svg"), out = box.querySelector(".aw-out");
    bind(box, function () {
      var L = val(box, "L"), tb = val(box, "tb"), mb = val(box, "mb"), bsel = val(box, "bucket"), bw = val(box, "bw"), noOverlap = bsel.indexOf("不") === 0, B = noOverlap ? Infinity : +bsel; show(box, "L", String(L));
      var comm = [], t = 0, acc = 0, accStart = 0, i, link = 0;                 // 反向从最后一层到第一层；桶满了就发（通信串行排队）
      for (i = 0; i < L; i++) { t += tb; acc += mb; if (acc >= B || i === L - 1) { var start = Math.max(t, link), dur = acc * 2 / bw; comm.push([start, dur]); link = start + dur; acc = 0; } }
      var backward = L * tb, total = noOverlap ? backward + L * mb * 2 / bw : Math.max(backward, link), X = function (x) { return 90 + x / total * 450; }, S = "";
      S += svgText(84, 30, zhen("反向计算", "backward"), "end") + '<rect x="90" y="16" width="' + (backward / total * 450).toFixed(1) + '" height="22" rx="3" class="aw-on"/>';
      S += svgText(84, 66, "all-reduce", "end");
      comm.forEach(function (c, j) { S += '<rect x="' + X(c[0]).toFixed(1) + '" y="52" width="' + Math.max(1, c[1] / total * 450 - 1).toFixed(1) + '" height="22" rx="3" class="' + (j === comm.length - 1 ? "aw-b" : "aw-f") + '"/>'; });
      if (noOverlap) S += '<rect x="' + X(backward).toFixed(1) + '" y="52" width="' + (L * mb * 2 / bw / total * 450).toFixed(1) + '" height="22" rx="3" class="aw-b"/>';
      S += svgText(90, 100, "0", "start") + svgText(540, 100, total.toFixed(1) + " ms", "end");
      svg.innerHTML = S;
      var exposed = total - backward;
      out.innerHTML = EN
        ? "<p>The backward pass is " + backward.toFixed(0) + " ms and the communication " + (L * mb * 2 / bw).toFixed(1) + " ms in all (each bucket's bytes x 2(n−1)/n, about 2 times); " + (noOverlap ? "without overlap all of it is exposed, so the step is <b>" + total.toFixed(1) + " ms</b>." : "bucketed and overlapped, only <b>" + exposed.toFixed(1) + " ms</b> is exposed (the last bucket, the gradients near the first layer), so the step is <b>" + total.toFixed(1) + " ms</b>.") + "</p>" +
          '<p class="aw-note">Too small a bucket means many messages, each paying α; too large a one means the first bucket waits a long time to go out and there is less overlap. PyTorch defaults to 25 MB. When the bandwidth is short (InfiniBand between machines) the communication is longer than the backward pass and no bucketing can hide it, at which point either synchronise less often (gradient accumulation) or switch to ZeRO or a larger batch.</p>'
        : "<p>反向 " + backward.toFixed(0) + " ms，通信共 " + (L * mb * 2 / bw).toFixed(1) + " ms（每桶字节数 × 2(n−1)/n ≈ 2 倍）；" + (noOverlap ? "不重叠时全部暴露，一步 <b>" + total.toFixed(1) + " ms</b>。" : "分桶重叠后只暴露 <b>" + exposed.toFixed(1) + " ms</b>（最后一个桶，也就是第一层附近的梯度），一步 <b>" + total.toFixed(1) + " ms</b>。") + "</p>" +
          '<p class="aw-note">桶太小：消息多、每次都付 α；桶太大：第一个桶要等很久才发出去、重叠变少。PyTorch 默认 25 MB。带宽不够（跨机 IB）时通信比反向还长，怎么分桶都藏不住——这时要么减少同步频率（梯度累积），要么换 ZeRO / 更大的 batch。</p>';
    });
  }

  // ---------------------------------------------------------------- 张量并行的通信量
  function tpComm(box) {
    box.innerHTML = '<div class="aw-title">' + zhen("张量并行每层两次 all-reduce，都在关键路径上：算一算它和计算的比例",
        "Tensor parallelism does two all-reduces per layer, both on the critical path: work out their ratio to the computation") + '</div><div class="aw-grid">' +
      row(zhen("TP 度数", "TP degree"), select("tp", ["2", "4", "8", "16"], "8")) + row(zhen("隐藏维 h", "hidden dimension h"), num("h", 8192, 512, 32768, 256)) + row(zhen("序列 × micro-batch（token）", "sequence x micro-batch (tokens)"), num("sb", 8192, 1, 1000000, 256)) +
      row(zhen("链路", "link"), select("link", opts(["NVLink（450 GB/s，α 1.5 µs）", "IB 400G（50 GB/s，α 3 µs）"],
        { "NVLink（450 GB/s，α 1.5 µs）": "NVLink (450 GB/s, α 1.5 µs)", "IB 400G（50 GB/s，α 3 µs）": "IB 400G (50 GB/s, α 3 µs)" }), "NVLink（450 GB/s，α 1.5 µs）"), true) + row("GPU", select("gpu", Object.keys(GPUS), "H100 SXM")) + row("MFU", range2("mfu", 50, 10, 80)) +
      '</div><div class="aw-out"></div>';
    var out = box.querySelector(".aw-out");
    bind(box, function () {
      var tp = +val(box, "tp"), h = val(box, "h"), sb = val(box, "sb"), nv = val(box, "link").indexOf("NVLink") === 0, B = nv ? 450e9 : 50e9, A = nv ? 1.5e-6 : 3e-6, g = GPUS[val(box, "gpu")], mfu = val(box, "mfu") / 100; show(box, "mfu", mfu.toFixed(2));
      var bytes = sb * h * 2, tComm = 2 * (tp - 1) * A + 2 * (tp - 1) / tp * bytes / B, perLayer = 4 * tComm;        // 前向 2 次 + 反向 2 次
      var flops = 3 * 24 * sb * h * h, tCompute = flops / tp / (g[2] * 1e12 * mfu);                                  // 一层前向+反向约 3 × 24·s·b·h²，切成 tp 份
      out.innerHTML = EN
        ? "<p>Each all-reduce moves " + fmtBytes(bytes) + " (s·b·h bf16 values), which a ring over " + tp + " cards takes <b>" + (tComm * 1e3).toFixed(2) + " ms</b> to do; 4 of them per layer forward and backward is " + (perLayer * 1e3).toFixed(2) + " ms, against about " + (tCompute * 1e3).toFixed(2) + " ms of computation for that layer split " + tp + " ways. Communication is <b>" + Math.round(perLayer / (perLayer + tCompute) * 100) + "%</b>" +
            (perLayer / (perLayer + tCompute) > 0.3 ? ", which is no longer acceptable: either lower the degree or move to NVLink" : "") + ".</p>" +
          '<p class="aw-note">The volume does not change with the tensor-parallel degree (it is always s·b·h) while the computation is divided by it, so a higher degree means a larger communication share, which is why tensor parallelism stays within one machine covered by NVLink and at a degree of at most 8. Sequence parallelism also partitions the activations at the normalisations and dropouts and replaces the two all-reduces with a reduce-scatter and an all-gather, leaving the volume unchanged while saving activation memory.</p>'
        : "<p>每次 all-reduce " + fmtBytes(bytes) + "（s·b·h 个 bf16），ring 在 " + tp + " 卡上要 <b>" + (tComm * 1e3).toFixed(2) + " ms</b>；一层前向 + 反向 4 次共 " + (perLayer * 1e3).toFixed(2) + " ms，而这一层切成 " + tp + " 份之后的计算约 " + (tCompute * 1e3).toFixed(2) + " ms——通信占 <b>" + Math.round(perLayer / (perLayer + tCompute) * 100) + "%</b>" +
            (perLayer / (perLayer + tCompute) > 0.3 ? "，已经没法接受：要么度数降下来、要么换 NVLink" : "") + "。</p>" +
          '<p class="aw-note">通信量不随 TP 度数变（每次都是 s·b·h），计算却被切成 1/TP：度数越大通信占比越高，所以 TP 基本只在 NVLink 覆盖的一台机器内做、不超过 8。序列并行把归一化、dropout 处的激活也切开，把两次 all-reduce 换成 reduce-scatter + all-gather，通信量不变但省了激活显存。</p>';
    });
  }

  // ================================================================ 计算机基础手册
  // ---------------------------------------------------------------- 复杂度：增长与常数
  function complexity(box) {
    box.innerHTML = '<div class="aw-title">' + zhen("复杂度看的是增长的形状：n 翻 10 倍，各种算法的开销各涨多少",
        "Complexity is about the shape of the growth: with n ten times larger, how much more does each algorithm cost") + '</div><div class="aw-grid">' +
      row("n", range2("n", 20, 4, 60)) +
      row(zhen("O(n log n) 的常数 / O(n²) 的常数", "the constant on O(n log n) vs on O(n^2)"), range2("c", 10, 1, 100)) +
      '</div><svg class="aw-chart" viewBox="0 0 560 230"></svg><div class="aw-out"></div>';
    var svg = box.querySelector("svg"), out = box.querySelector(".aw-out");
    bind(box, function () {
      var LOC = EN ? "en-US" : "zh-CN";
      var n = Math.round(Math.pow(10, val(box, "n") / 10)), c = val(box, "c"); show(box, "n", n.toLocaleString(LOC)); show(box, "c", c + "×");
      var fns = [["log n", function (x) { return Math.log2(x); }, "#8e8e93"], ["n", function (x) { return x; }, "#34c759"], [zhen("n log n（常数 " + c + "）", "n log n (constant " + c + ")"), function (x) { return c * x * Math.log2(x); }, "#007aff"], ["n²", function (x) { return x * x; }, "#f08c00"], ["2ⁿ", function (x) { return Math.pow(2, Math.min(x, 60)); }, "#ff3b30"]];
      var X = function (x) { return 50 + (Math.log10(x) - 0.6) / 5.4 * 470; }, Y = function (v) { return 190 - Math.min(14, Math.log10(Math.max(1, v))) / 14 * 165; }, S = "", i, k;
      for (i = 1; i <= 6; i++) S += '<line x1="' + X(Math.pow(10, i)) + '" y1="20" x2="' + X(Math.pow(10, i)) + '" y2="190" class="aw-gl"/>' + svgText(X(Math.pow(10, i)), 206, "10^" + i, "middle");
      for (i = 0; i <= 14; i += 2) S += '<line x1="50" y1="' + Y(Math.pow(10, i)) + '" x2="520" y2="' + Y(Math.pow(10, i)) + '" class="aw-gl"/>' + svgText(44, Y(Math.pow(10, i)) + 4, "10^" + i, "end");
      S += svgText(285, 224, zhen("n（对数）；纵轴 = 操作次数（对数）", "n (log scale); the y axis is the operation count (log scale)"), "middle");
      fns.forEach(function (f, j) { var d = ""; for (k = 0; k <= 100; k++) { var x = Math.pow(10, 0.6 + 5.4 * k / 100); d += (k ? " L " : "M ") + X(x).toFixed(1) + " " + Y(f[1](x)).toFixed(1); } S += '<path d="' + d + '" fill="none" stroke="' + f[2] + '" stroke-width="2"/>' + '<circle cx="' + X(n).toFixed(1) + '" cy="' + Y(f[1](n)).toFixed(1) + '" r="4" fill="' + f[2] + '"/>'; });
      S += svgText(60, 14, zhen("灰 log n　绿 n　蓝 n log n（常数 " + c + "）　橙 n²　红 2ⁿ",
        "grey log n · green n · blue n log n (constant " + c + ") · orange n² · red 2ⁿ"), "start");
      svg.innerHTML = S;
      function fmt(v) { return v >= 1e15 ? v.toExponential(1) : Math.round(v).toLocaleString(LOC); }
      var grow = (10 * Math.log2(n * 10) / Math.log2(n)).toFixed(1);
      out.innerHTML = EN
        ? "<p>n = " + n.toLocaleString(LOC) + ": log n ≈ " + fmt(fns[0][1](n)) + ", n = " + fmt(n) + ", " + c + "·n log n ≈ " + fmt(fns[2][1](n)) + ", n² = " + fmt(n * n) + ", 2ⁿ " + (n > 60 ? "is astronomical already" : "= " + fmt(fns[4][1](n))) + ". " +
          (c * n * Math.log2(n) > n * n ? "With this constant, n log n is now slower than n²: the constant really matters at small n (which is why a sorting library switches to insertion sort for small arrays)." : "However large n gets, a constant of " + c + " cannot save n²: ten times larger than " + fmt(n) + ", n log n grows " + grow + "x and n² grows 100x.") + "</p>" +
          '<p class="aw-note">In an interview, "the size of n" is the hint: 10⁵ to 10⁶ needs O(n log n) or O(n), 10³ to 10⁴ allows O(n²), and only around 20 allows an exponential search. The same arithmetic in an inference system: attention is O(n²), so ten times the context is a hundred times the compute, which is where every long-context optimization starts.</p>'
        : "<p>n = " + n.toLocaleString(LOC) + "：log n ≈ " + fmt(fns[0][1](n)) + "，n = " + fmt(n) + "，" + c + "·n log n ≈ " + fmt(fns[2][1](n)) + "，n² = " + fmt(n * n) + "，2ⁿ " + (n > 60 ? "早已天文数字" : "= " + fmt(fns[4][1](n))) + "。" +
          (c * n * Math.log2(n) > n * n ? "现在常数大的 n log n 比 n² 还慢——常数在小 n 下真的重要（排序库对小数组切换成插入排序就是这个原因）。" : "n 再大，常数 " + c + " 也救不了 n²：从 " + fmt(n) + " 翻 10 倍，n log n 涨 " + grow + " 倍，n² 涨 100 倍。") + "</p>" +
          '<p class="aw-note">面试里"n 的规模"就是提示：10⁵～10⁶ 要 O(n log n) 或 O(n)，10³～10⁴ 可以 O(n²)，20 左右才能指数搜索。推理系统里同样的账：注意力是 O(n²)，上下文翻 10 倍算力翻 100 倍，这就是长上下文一切优化的出发点。</p>';
    });
  }

  // ---------------------------------------------------------------- 滑动窗口：一步一步看指针怎么走
  function windowStep(box) {
    var A = [2, 3, 1, 2, 4, 3, 1, 5, 2, 1, 1, 3];
    box.innerHTML = '<div class="aw-title">' + zhen("滑动窗口：右指针负责扩张、左指针负责收缩，每个元素最多进出一次",
        "The sliding window: the right pointer expands, the left one shrinks, and each element goes in and out at most once") + '</div><div class="aw-grid">' +
      row(zhen("题目", "problem"), select("kind", opts(["和 ≥ 7 的最短子数组", "和 ≤ 7 的最长子数组"],
          { "和 ≥ 7 的最短子数组": "the shortest subarray with a sum >= 7", "和 ≤ 7 的最长子数组": "the longest subarray with a sum <= 7" }),
        "和 ≥ 7 的最短子数组"), true) + row(zhen("走到第几步", "step"), range2("step", 6, 0, 40)) + '</div><svg class="aw-chart" viewBox="0 0 560 120"></svg><div class="aw-out"></div>';
    var svg = box.querySelector("svg"), out = box.querySelector(".aw-out");
    bind(box, function () {
      var shortest = val(box, "kind").indexOf("≥") >= 0, K = 7, states = [], l = 0, sum = 0, best = shortest ? Infinity : 0, bestLR = null, r;
      for (r = 0; r < A.length; r++) {
        sum += A[r]; states.push({ l: l, r: r, sum: sum, note: zhen("右指针吃进 a[" + r + "] = " + A[r], "the right pointer takes a[" + r + "] = " + A[r]) });
        if (shortest) { while (sum >= K) { if (r - l + 1 < best) { best = r - l + 1; bestLR = [l, r]; } states[states.length - 1].note += zhen("；和 " + sum + " ≥ " + K + "，记录长度 " + (r - l + 1) + "，左指针收缩", "; the sum " + sum + " >= " + K + ", so record the length " + (r - l + 1) + " and shrink from the left"); sum -= A[l]; l++; states.push({ l: l, r: r, sum: sum, note: zhen("左指针移到 " + l + "，和 " + sum, "the left pointer moves to " + l + ", sum " + sum) }); } }
        else { while (sum > K) { sum -= A[l]; l++; states.push({ l: l, r: r, sum: sum, note: zhen("和超过 " + K + "，左指针移到 " + l + "，和 " + sum, "the sum is over " + K + ", so the left pointer moves to " + l + ", sum " + sum) }); } if (r - l + 1 > best) { best = r - l + 1; bestLR = [l, r]; } }
      }
      input(box, "step").max = states.length - 1; var st = Math.min(val(box, "step"), states.length - 1), s = states[st]; show(box, "step", st + " / " + (states.length - 1));
      var S = "", i;
      for (i = 0; i < A.length; i++) { var inWin = i >= s.l && i <= s.r; S += '<rect x="' + (40 + i * 40) + '" y="30" width="34" height="30" rx="4" class="' + (inWin ? "aw-on" : "aw-off") + '"/>' + svgText(57 + i * 40, 50, A[i], "middle") + svgText(57 + i * 40, 76, i, "middle"); }
      S += svgText(57 + s.l * 40, 20, "L", "middle") + svgText(57 + s.r * 40, 20, s.l === s.r ? "L R" : "R", "middle") +
        svgText(40, 104, zhen("窗口 [" + s.l + ", " + s.r + "]，和 = " + s.sum + "；", "window [" + s.l + ", " + s.r + "], sum = " + s.sum + "; ") + s.note, "start");
      svg.innerHTML = S;
      out.innerHTML = EN
        ? "<p>" + (st === states.length - 1 ? "Done: the answer is " + (bestLR ? "the window [" + bestLR[0] + ", " + bestLR[1] + "], length " + best : "that there is none") + ". " : "Keep advancing the step. ") +
          "The right pointer advances only " + A.length + " times and the left one at most " + A.length + ", so it is O(n) in all, despite the two nested loops.</p>" +
          '<p class="aw-note">The three forms share one framework: a fixed-length window (one in on the right, one out on the left), the longest satisfying a condition (shrink while it fails), and the shortest satisfying one (shrink and record once it holds). The premise is that the condition is monotone as the window grows. With negative numbers the sum is no longer monotone, so a sliding window will not do and you need prefix sums with a hash table.</p>'
        : "<p>" + (st === states.length - 1 ? "走完了：答案是" + (bestLR ? "窗口 [" + bestLR[0] + ", " + bestLR[1] + "]，长度 " + best : "不存在") + "。" : "继续拨步数。") + "右指针只前进 " + A.length + " 次，左指针也最多前进 " + A.length + " 次，总共 O(n)，虽然代码里有两层循环。</p>" +
          '<p class="aw-note">三种形态共用一套框架：定长窗口（右进一个左出一个）、最长满足条件（不满足时收缩）、最短满足条件（满足时收缩并记录）。前提是"条件随窗口扩张单调"——有负数时和不再单调，就不能用滑动窗口，要换前缀和 + 哈希。</p>';
    });
  }

  // ---------------------------------------------------------------- 动态规划：编辑距离的表怎么填
  function editDistance(box) {
    box.innerHTML = '<div class="aw-title">' + zhen("动态规划的表：编辑距离 dp[i][j] 只依赖左、上、左上三个格子，按行填一遍就是答案",
        "The dynamic-programming table: edit distance dp[i][j] depends only on the cells left, above and diagonally above, so one pass by row gives the answer") + '</div><div class="aw-grid">' +
      row(zhen("字符串 A", "string A"), '<input type="text" data-k="a" value="kitten" spellcheck="false">') + row(zhen("字符串 B", "string B"), '<input type="text" data-k="b" value="sitting" spellcheck="false">') + row(zhen("填到第几个格子", "cells filled"), range2("k", 20, 0, 200), true) +
      '</div><svg class="aw-chart" viewBox="0 0 560 240"></svg><div class="aw-out"></div>';
    var svg = box.querySelector("svg"), out = box.querySelector(".aw-out");
    bind(box, function () {
      var a = input(box, "a").value.slice(0, 10), b = input(box, "b").value.slice(0, 12), n = a.length, m = b.length, dp = [], i, j;
      for (i = 0; i <= n; i++) { dp.push([]); for (j = 0; j <= m; j++) dp[i].push(i === 0 ? j : j === 0 ? i : 0); }
      for (i = 1; i <= n; i++) for (j = 1; j <= m; j++) dp[i][j] = a[i - 1] === b[j - 1] ? dp[i - 1][j - 1] : 1 + Math.min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]);
      var total = n * m; input(box, "k").max = total; var k = Math.min(val(box, "k"), total); show(box, "k", k + " / " + total);
      var cw = Math.min(36, 480 / (m + 2)), ch = Math.min(22, 200 / (n + 2)), S = "", filled = 0, cur = null;
      for (j = 0; j <= m; j++) S += svgText(60 + (j + 1) * cw + cw / 2, 14, j ? b[j - 1] : "ε", "middle");
      for (i = 0; i <= n; i++) {
        S += svgText(60 + cw / 2, 20 + (i + 1) * ch + ch / 2 + 4, i ? a[i - 1] : "ε", "middle");
        for (j = 0; j <= m; j++) {
          var base = i === 0 || j === 0, idx = base ? -1 : (i - 1) * m + j, show_ = base || idx <= k, isCur = idx === k && !base;
          if (isCur) cur = [i, j];
          S += '<rect x="' + (60 + (j + 1) * cw) + '" y="' + (20 + (i + 1) * ch) + '" width="' + (cw - 2) + '" height="' + (ch - 2) + '" rx="3" class="' + (isCur ? "aw-b" : show_ ? (base ? "aw-f" : "aw-on") : "aw-off") + '"/>' + (show_ ? svgText(60 + (j + 1) * cw + cw / 2 - 1, 20 + (i + 1) * ch + ch / 2 + 3, dp[i][j], "middle") : "");
        }
      }
      svg.setAttribute("viewBox", "0 0 560 " + (30 + (n + 2) * ch));
      svg.innerHTML = S;
      var msg = EN
        ? (cur ? "filling dp[" + cur[0] + "][" + cur[1] + "] (" + a[cur[0] - 1] + " vs " + b[cur[1] - 1] + "): " + (a[cur[0] - 1] === b[cur[1] - 1] ? "the characters match, so it is simply the diagonal dp[" + (cur[0] - 1) + "][" + (cur[1] - 1) + "] = " + dp[cur[0] - 1][cur[1] - 1] : "they differ, so 1 + min(above " + dp[cur[0] - 1][cur[1]] + ", left " + dp[cur[0]][cur[1] - 1] + ", diagonal " + dp[cur[0] - 1][cur[1] - 1] + ") = " + dp[cur[0]][cur[1]]) : "all filled: the edit distance is dp[" + n + "][" + m + "] = <b>" + dp[n][m] + "</b>.")
        : (cur ? "正在填 dp[" + cur[0] + "][" + cur[1] + "]（" + a[cur[0] - 1] + " vs " + b[cur[1] - 1] + "）：" + (a[cur[0] - 1] === b[cur[1] - 1] ? "字符相同，直接等于左上 dp[" + (cur[0] - 1) + "][" + (cur[1] - 1) + "] = " + dp[cur[0] - 1][cur[1] - 1] : "不同，1 + min(上 " + dp[cur[0] - 1][cur[1]] + "，左 " + dp[cur[0]][cur[1] - 1] + "，左上 " + dp[cur[0] - 1][cur[1] - 1] + ") = " + dp[cur[0]][cur[1]]) : "全部填完：编辑距离 = dp[" + n + "][" + m + "] = <b>" + dp[n][m] + "</b>。");
      out.innerHTML = "<p>" + msg + "</p>" + (EN
        ? '<p class="aw-note">Brute-force recursion computes the same subproblem an exponential number of times; memoisation stores it; iteration fills the table in the order that puts the cells it depends on first, O(1) per cell and O(n·m) in all. When only the previous row is needed, the table compresses to one row, which is the standard space optimisation for DP.</p>'
        : '<p class="aw-note">暴力递归会把同一个子问题算指数次；记忆化把它存起来；递推则按"依赖的格子先填"的顺序把表填满，每个格子 O(1)，总共 O(n·m)。只依赖上一行的话还能把表压成一行，这是 DP 空间优化的套路。</p>');
    });
  }

  // ---------------------------------------------------------------- 算力涨得比带宽快：屋脊点与 batch
  function ridgeGen(box) {
    var G = [["V100", 2017, 125, 900], ["A100", 2020, 312, 2039], ["H100", 2022, 989, 3350], ["B200", 2024, 2250, 8000]];   // bf16 TFLOPS、带宽 GB/s
    box.innerHTML = '<div class="aw-title">' + zhen("每一代算力涨得都比带宽快：屋脊点越来越高，decode 要攒的 batch 也越来越大",
        "Every generation raises compute faster than bandwidth: the ridge point keeps rising and decode needs an ever larger batch") + '</div><div class="aw-grid">' +
      row(zhen("权重精度", "weight precision"), select("wb", opts(["bf16（2 字节）", "fp8（1 字节）", "int4（0.5 字节）"],
          { "bf16（2 字节）": "bf16 (2 bytes)", "fp8（1 字节）": "fp8 (1 byte)", "int4（0.5 字节）": "int4 (0.5 bytes)" }),
        "bf16（2 字节）"), true) + '</div><svg class="aw-chart" viewBox="0 0 560 200"></svg><div class="aw-out"></div>';
    var svg = box.querySelector("svg"), out = box.querySelector(".aw-out");
    bind(box, function () {
      var wb = { "bf16（2 字节）": 2, "fp8（1 字节）": 1, "int4（0.5 字节）": 0.5 }[val(box, "wb")], S = "", i, mx = 0, rows = [];
      for (i = 0; i < G.length; i++) { var ridge = G[i][2] * 1e12 / (G[i][3] * 1e9), batch = ridge * wb / 2; rows.push([G[i][0], ridge, batch]); mx = Math.max(mx, batch); }
      for (i = 0; i < G.length; i++) {
        var x = 60 + i * 120, h = rows[i][2] / mx * 130;
        S += '<rect x="' + x + '" y="' + (160 - h).toFixed(1) + '" width="70" height="' + h.toFixed(1) + '" rx="3" class="aw-f"/>' + svgText(x + 35, 176, zhen(G[i][0] + "（" + G[i][1] + "）", G[i][0] + " (" + G[i][1] + ")"), "middle") + svgText(x + 35, 154 - h, Math.round(rows[i][2]) + zhen(" token", " tokens"), "middle");
      }
      S += svgText(40, 16, zhen("decode 时让算力和带宽同时跑满所需的 batch（token 数）= 屋脊点 × 每参数字节数 / 2",
        "the batch (in tokens) that fills compute and bandwidth at once during decode = the ridge point x bytes per parameter / 2"), "start") + '<line x1="40" y1="160" x2="540" y2="160" class="aw-axis"/>';
      svg.innerHTML = S;
      out.innerHTML = EN
        ? "<p>The ridge point (how many operations per byte read keep the compute fed): " + rows.map(function (r) { return r[0] + " " + Math.round(r[1]); }).join(", ") + " FLOP/byte. From V100 to B200 the compute grew " + (G[3][2] / G[0][2]).toFixed(0) + "x and the bandwidth only " + (G[3][3] / G[0][3]).toFixed(1) + "x.</p>" +
          '<p class="aw-note">Each decoded token reads the weights once (' + wb + ' bytes per parameter) and does 2 operations: below the token count the ridge point implies, the compute idles while the bandwidth is full. So a new card is only kept busy by a larger batch, by speculative decoding (verifying several tokens at once) and by lower weight precision (more operations per byte); low precision is viable in the first place because compute keeps getting cheaper and bandwidth dearer.</p>'
        : "<p>屋脊点（每读 1 字节要做多少次运算才喂饱算力）：" + rows.map(function (r) { return r[0] + " " + Math.round(r[1]); }).join("，") + " FLOP/字节；V100 到 B200 算力涨 " + (G[3][2] / G[0][2]).toFixed(0) + " 倍，带宽只涨 " + (G[3][3] / G[0][3]).toFixed(1) + " 倍。</p>" +
          '<p class="aw-note">decode 每个 token 要读一遍权重（每参数 ' + wb + ' 字节）、做 2 次运算：batch 小于屋脊点对应的 token 数时算力闲着、带宽满着。所以新卡要靠更大的 batch、投机解码（一次验证多个 token）、更低的权重精度（每字节换来更多运算）才用得满；低精度之所以"可行"也因为算力越来越便宜、带宽越来越贵。</p>';
    });
  }

  // ---------------------------------------------------------------- Little 定律：要跑满带宽，在途要有多少数据
  function littleLaw(box) {
    box.innerHTML = '<div class="aw-title">' + zhen("Little 定律：在途的数据量 = 带宽 × 延迟，要跑满显存带宽就得有足够多的访存请求同时在路上",
        "Little's law: the data in flight = bandwidth x latency, so filling the memory bandwidth needs enough accesses on the way at once") + '</div><div class="aw-grid">' +
      row(zhen("带宽（GB/s）", "bandwidth (GB/s)"), num("bw", 3350, 10, 20000, 50)) + row(zhen("访存延迟（ns）", "access latency (ns)"), num("lat", 600, 50, 5000, 50)) + row(zhen("SM 数", "SM count"), num("sms", 132, 1, 1000)) + row(zhen("每个请求的大小（字节）", "bytes per access"), select("req", ["4", "16", "128"], "16")) +
      '</div><div class="aw-out"></div>';
    var out = box.querySelector(".aw-out");
    bind(box, function () {
      var bw = val(box, "bw") * 1e9, lat = val(box, "lat") * 1e-9, sms = val(box, "sms"), req = +val(box, "req"), inflight = bw * lat, perSm = inflight / sms, reqs = inflight / req;
      var LOC = EN ? "en-US" : "zh-CN";
      out.innerHTML = EN
        ? "<p>Data in flight = " + val(box, "bw") + " GB/s x " + val(box, "lat") + " ns = <b>" + (inflight / 1e6).toFixed(2) + " MB</b>, which is " + (perSm / 1024).toFixed(1) + " KB spread over each of " + sms + " SMs. At " + req + " bytes per access, the whole card needs <b>" + Math.round(reqs).toLocaleString(LOC) + "</b> accesses in flight at once, " + Math.round(reqs / sms).toLocaleString(LOC) + " per SM. With at most 2048 threads per SM, " + (reqs / sms / 2048 > 1 ? "each thread has to carry " + (reqs / sms / 2048).toFixed(1) + " outstanding accesses, through loop unrolling, vectorised loads and asynchronous copies" : "there are enough threads, but each one still has to have something else to do while it waits") + ".</p>" +
          '<p class="aw-note">This is why a GPU needs a vast number of threads, float4 and 128-bit loads, and cp.async or TMA: not because the compute is short, but because it cannot afford to wait. The higher the bandwidth and the longer the latency, the more has to be in flight. B200\'s 8 TB/s is more than twice H100\'s, so the queue of waiting accesses has to be more than twice as long.</p>'
        : "<p>在途数据 = " + val(box, "bw") + " GB/s × " + val(box, "lat") + " ns = <b>" + (inflight / 1e6).toFixed(2) + " MB</b>；摊到 " + sms + " 个 SM 每个 " + (perSm / 1024).toFixed(1) + " KB；如果每个请求 " + req + " 字节，整卡要同时有 <b>" + Math.round(reqs).toLocaleString(LOC) + "</b> 个请求在飞，每个 SM " + Math.round(reqs / sms).toLocaleString(LOC) + " 个——每 SM 最多 2048 个线程，" + (reqs / sms / 2048 > 1 ? "每个线程得同时挂着 " + (reqs / sms / 2048).toFixed(1) + " 个未返回的请求（靠展开循环、向量化加载、异步拷贝）" : "线程够用，但每个线程仍要在等待时有别的事做") + "。</p>" +
          '<p class="aw-note">这就是 GPU 需要海量线程、需要 float4 / 128 位加载、需要 cp.async / TMA 的原因：不是算力不够，是等不起。带宽越高、延迟越长，在途要求越大——B200 的 8 TB/s 比 H100 多一倍多，等待的队伍也要长一倍多。</p>';
    });
  }

  // ---------------------------------------------------------------- 排队论：利用率与延迟
  function mm1Latency(box) {
    box.innerHTML = '<div class="aw-title">' + zhen("排队论：利用率接近 1 时排队时间非线性地涨，服务器多一点就平缓得多（M/M/c）",
        "Queueing theory: as utilisation approaches 1 the queueing time grows non-linearly, and a few more servers flatten it a lot (M/M/c)") + '</div><div class="aw-grid">' +
      row(zhen("利用率 ρ", "utilisation ρ"), range2("rho", 80, 10, 98)) + row(zhen("平均服务时间（ms）", "mean service time (ms)"), num("svc", 10, 1, 10000)) + row(zhen("服务器数 c", "servers c"), select("c", ["1", "4", "16"], "4")) + '</div><svg class="aw-chart" viewBox="0 0 560 200"></svg><div class="aw-out"></div>';
    var svg = box.querySelector("svg"), out = box.querySelector(".aw-out");
    function fact(n) { var r = 1, i; for (i = 2; i <= n; i++) r *= i; return r; }
    function erlangC(c, rho) { var a = c * rho, top = Math.pow(a, c) / fact(c) / (1 - rho), bottom = top, k; for (k = 0; k < c; k++) bottom += Math.pow(a, k) / fact(k); return top / bottom; }
    function wait(c, rho, s) { return erlangC(c, rho) * s / (c * (1 - rho)); }
    bind(box, function () {
      var rho = val(box, "rho") / 100, s = val(box, "svc"), c = +val(box, "c"); show(box, "rho", rho.toFixed(2));
      var X = function (r) { return 50 + (r - 0.1) / 0.88 * 470; }, ymax = Math.max(1, wait(1, 0.98, 1) * 0.05), Y = function (w) { return 170 - Math.min(ymax, w) / ymax * 150; }, S = "", k, i;
      [[1, "#8e8e93"], [4, "#007aff"], [16, "#f08c00"]].forEach(function (cc, j) { var d = ""; for (k = 0; k <= 100; k++) { var r = 0.1 + 0.88 * k / 100; d += (k ? " L " : "M ") + X(r).toFixed(1) + " " + Y(wait(cc[0], r, 1)).toFixed(1); } S += '<path d="' + d + '" fill="none" stroke="' + cc[1] + '" stroke-width="' + (cc[0] === c ? 2.8 : 1.4) + '"/>' + svgText(56, 34 + j * 15, zhen(cc[0] + " 台", cc[0] + (cc[0] === 1 ? " server" : " servers")), "start"); });
      for (i = 2; i <= 9; i++) S += '<line x1="' + X(i / 10) + '" y1="20" x2="' + X(i / 10) + '" y2="170" class="aw-gl"/>' + svgText(X(i / 10), 186, i * 10 + "%", "middle");
      S += '<line x1="50" y1="170" x2="520" y2="170" class="aw-axis"/><line x1="' + X(rho).toFixed(1) + '" y1="20" x2="' + X(rho).toFixed(1) + '" y2="170" class="aw-dash"/>' + svgText(285, 198, zhen("利用率；纵轴 = 平均排队时间 / 服务时间", "utilisation; the y axis is the mean queueing time / the service time"), "middle");
      svg.innerHTML = S;
      var w = wait(c, rho, s), p99 = w > 0 ? -Math.log(0.01 / erlangC(c, rho)) * s / (c * (1 - rho)) : 0;
      out.innerHTML = EN
        ? "<p>At " + Math.round(rho * 100) + "% utilisation on " + c + " server" + (c === 1 ? "" : "s") + " with a " + s + " ms service time: an arrival queues with probability " + Math.round(erlangC(c, rho) * 100) + "%, the mean queue is <b>" + w.toFixed(1) + " ms</b>, and the queueing time's p99 is about <b>" + Math.max(0, p99).toFixed(0) + " ms</b> (an exponential tail). One server at the same utilisation queues " + wait(1, rho, s).toFixed(1) + " ms on average.</p>" +
          '<p class="aw-note">This is why average utilisation of 80% still gives an ugly p99: the queueing time goes as 1/(1-ρ) and arrivals come in random clusters. Several servers sharing one queue (M/M/c) is far better than splitting the traffic c ways with a queue each, which is why an inference gateway wants a global queue rather than static sharding. Keeping twenty or thirty percent in reserve is not waste, it is the premium paid for randomness.</p>'
        : "<p>利用率 " + Math.round(rho * 100) + "%、" + c + " 台、服务 " + s + " ms：到达要排队的概率 " + Math.round(erlangC(c, rho) * 100) + "%，平均排队 <b>" + w.toFixed(1) + " ms</b>，排队时间的 P99 约 <b>" + Math.max(0, p99).toFixed(0) + " ms</b>（指数尾）。单台在同样利用率下平均要排 " + wait(1, rho, s).toFixed(1) + " ms。</p>" +
          '<p class="aw-note">这就是"平均利用率 80%，P99 却很难看"：排队时间 ∝ 1/(1−ρ)，到达又是随机成簇的。多台共享一个队列（M/M/c）比把流量切成 c 份各排各的好得多——所以推理网关要做全局队列而不是静态分片；留两三成余量不是浪费，是给随机性付的保险。</p>';
    });
  }

  // ================================================================ C++ 进阶手册
  // ---------------------------------------------------------------- vector 扩容：拷贝还是移动，noexcept 决定
  function vectorRealloc(box) {
    box.innerHTML = '<div class="aw-title">' + zhen("push_back 到满了就要扩容：元素是被拷贝还是被移动，取决于移动构造函数是不是 noexcept",
        "A push_back into a full vector reallocates: whether the elements are copied or moved depends on the move constructor being noexcept") + '</div><div class="aw-grid">' +
      row(zhen("push_back 次数", "push_backs"), range2("n", 20, 1, 100)) +
      row(zhen("扩容倍数", "growth factor"), select("g", opts(["2（libstdc++）", "1.5（MSVC）"],
        { "2（libstdc++）": "2 (libstdc++)", "1.5（MSVC）": "1.5 (MSVC)" }), "2（libstdc++）")) +
      row(zhen("元素的移动构造", "the element's move constructor"), select("ne",
        opts(["noexcept：扩容时移动", "可能抛异常：扩容时拷贝", "没有移动构造：只能拷贝"],
          { "noexcept：扩容时移动": "noexcept: moved on reallocation",
            "可能抛异常：扩容时拷贝": "may throw: copied on reallocation",
            "没有移动构造：只能拷贝": "no move constructor: copied" }), "noexcept：扩容时移动"), true) +
      '</div><svg class="aw-chart" viewBox="0 0 560 150"></svg><div class="aw-out"></div>';
    var svg = box.querySelector("svg"), out = box.querySelector(".aw-out");
    bind(box, function () {
      var n = val(box, "n"), g = val(box, "g").indexOf("2") === 0 ? 2 : 1.5, mode = val(box, "ne"), cap = 0, size = 0, moves = 0, copies = 0, reallocs = [], i; show(box, "n", String(n));
      for (i = 0; i < n; i++) { if (size === cap) { var nc = cap === 0 ? 1 : Math.ceil(cap * g); if (size) { if (mode.indexOf("noexcept") === 0) moves += size; else copies += size; } reallocs.push([i, nc]); cap = nc; } size++; }
      var S = svgText(40, 16, zhen("每次扩容时已有的元素都要搬到新内存（蓝 = 搬运量）；容量按 " + g + " 倍增长",
        "Each reallocation moves the existing elements (blue = how many); capacity grows " + g + "x"), "start"),
        X = function (i) { return 40 + i / n * 480; }, mx = cap;
      reallocs.forEach(function (r) { S += '<rect x="' + X(r[0]).toFixed(1) + '" y="' + (120 - r[1] / mx * 90).toFixed(1) + '" width="' + Math.max(2, 480 / n - 1).toFixed(1) + '" height="' + (r[1] / mx * 90).toFixed(1) + '" class="aw-f"/>'; });
      S += '<line x1="40" y1="120" x2="520" y2="120" class="aw-axis"/>' +
           svgText(40, 140, zhen("第 1 次 push", "push 1"), "start") +
           svgText(520, 140, zhen("第 " + n + " 次 push（最终容量 " + cap + "）", "push " + n + " (final capacity " + cap + ")"), "end");
      svg.innerHTML = S;
      var total = moves + copies;
      out.innerHTML = EN
        ? "<p>" + n + " push_backs trigger " + reallocs.length + " reallocations moving <b>" + total + "</b> elements in all (about " +
          (total / n).toFixed(1) + " times the element count, so amortized each push is O(1)): " +
          (mode.indexOf("noexcept") === 0 ? "all of them <b>moves</b>, each only a few pointers."
           : mode.indexOf("可能") === 0 ? "all of them <b>copies</b>: the vector dares not move because of the strong exception guarantee, since a throw halfway through could not be rolled back, so an unmarked move constructor means it copies."
           : "all of them <b>copies</b>: with no move constructor, every element is deep-copied.") + "</p>" +
          '<p class="aw-note">A resource class (holding a buffer or a handle) must mark its move constructor noexcept; a =default move usually is noexcept already. reserve saves all of this moving; emplace_back saves something else (constructing in place, one temporary fewer).</p>'
        : "<p>" + n + " 次 push_back 触发 " + reallocs.length + " 次扩容，一共搬了 <b>" + total + "</b> 个元素（约 " + (total / n).toFixed(1) + " 倍于元素数——摊还下来每次 push 是 O(1)）：" + (mode.indexOf("noexcept") === 0 ? "全部是<b>移动</b>，每个只是搬几个指针。" : mode.indexOf("可能") === 0 ? "全部是<b>拷贝</b>：vector 为了强异常保证不敢移动——移动到一半抛异常就没法回滚了，所以只要移动构造没标 noexcept，它就老老实实拷贝。" : "全部是<b>拷贝</b>：没有移动构造，每个元素深拷贝一遍。") + "</p>" +
          '<p class="aw-note">资源类（持有 buffer、句柄）的移动构造务必标 noexcept；=default 生成的移动构造通常自动是 noexcept。reserve 能把这些搬运全省掉；emplace_back 省的是另一件事（原地构造，少一次临时对象）。</p>';
    });
  }


  // ---------------------------------------------------------------- SGLang 的提交时间线（基准 29f6d408c0，2026-10-02）
  var SGL_MONTHS = [["2023-10",1],["2023-11",0],["2023-12",0],["2024-01",88],["2024-02",49],["2024-03",35],["2024-04",23],["2024-05",53],["2024-06",42],["2024-07",223],["2024-08",243],["2024-09",149],["2024-10",207],["2024-11",271],["2024-12",236],["2025-01",305],["2025-02",259],["2025-03",468],["2025-04",460],["2025-05",397],["2025-06",384],["2025-07",433],["2025-08",664],["2025-09",630],["2025-10",803],["2025-11",899],["2025-12",1094],["2026-01",954],["2026-02",761],["2026-03",1007],["2026-04",1026],["2026-05",1201],["2026-06",1314],["2026-07",1238],["2026-08",1620],["2026-09",1642],["2026-10",68]];
  var SGL_TAGS = { "2024-01": "v0.1.3", "2024-07": "v0.2.0", "2024-09": "v0.3.0", "2024-12": "v0.4.0", "2025-04": "v0.4.6", "2025-08": "v0.5.0rc0", "2026-09": "v0.5.21" };
  var SGL_EV_EN = {             // 英文版页面上的事件文字（键是中文原文）
    "建仓：.gitignore、LICENSE、一行 README": "the repository created: .gitignore, LICENSE, a one-line README",
    "论文 v1 上 arXiv（2312.07104）": "the paper's v1 on arXiv (2312.07104)",
    "release initial code：145 个文件、1.78 万行；#7 修基数树匹配；LMSYS 博客": "release initial code: 145 files, 17.8 thousand lines; #7 fixes the radix matching; the LMSYS blog",
    "RadixAttention 第一版、从 Outlines 改编的 FSM、前端语言": "RadixAttention's first version, an FSM adapted from Outlines, the frontend language",
    "jump-forward（#144）、import outlines（#168）": "jump-forward (#144), import outlines (#168)",
    "静态数据并行 #480：controller 与 tp_worker": "static data parallelism #480: the controller and tp_worker",
    "去掉 rpyc #646、目录重构 #807、CUDA Graph 默认 #612、v0.2 博客（3.1×）": "rpyc removed #646, the directory restructuring #807, CUDA graphs by default #612, the v0.2 blog (3.1x)",
    "mem_cache/、model_executor/、sampling/ 拆分；MLA Triton kernel": "mem_cache/, model_executor/ and sampling/ split out; the MLA Triton kernel",
    "v0.3：torch.compile、注意力后端抽象 #1381/#1547、scheduler.py 独立 #1538": "v0.3: torch.compile, the attention backend abstraction #1381/#1547, scheduler.py on its own #1538",
    "重叠调度 #1738、xgrammar #1752、rust/ 路由器": "overlapped scheduling #1738, xgrammar #1752, the rust/ router",
    "DP attention #1970、sgl-kernel 起步": "DP attention #1970, sgl-kernel begins",
    "v0.4 博客：零开销调度；EAGLE 四部曲开始 #2150": "the v0.4 blog: zero-overhead scheduling; EAGLE's four parts begin #2150",
    "entrypoints/、去 vLLM 依赖系列、HiCache 控制器": "entrypoints/, the remove-vLLM series, the HiCache controller",
    "HiCache #2693、llguidance #3298": "HiCache #2693, llguidance #3298",
    "PD 分离 #4655、FA3 后端 #4709、页大小 > 1 #4356、删 jump-forward #4032": "PD disaggregation #4655, the FA3 backend #4709, page size > 1 #4356, jump-forward deleted #4032",
    "Mooncake、NIXL 传输后端；sgl-router 独立目录": "the Mooncake and NIXL transfer backends; sgl-router its own directory",
    "TBO #4068、EPLB、96 卡 H100 博客": "TBO #4068, EPLB, the 96-card H100 blog",
    "OpenAI server 重构 #7167、eplb/": "the OpenAI server restructured #7167, eplb/",
    "multimodal/、weight_sync/、HiCache 存储后端": "multimodal/, weight_sync/, HiCache's storage backends",
    "v0.5.0rc0；简化前端 #9029": "v0.5.0rc0; the frontend simplified #9029",
    "gRPC 入口": "the gRPC entry point",
    "分配逻辑拆出调度器 #11313、piecewise CUDA graph #11490": "the allocation logic split from the scheduler #11313, piecewise CUDA graphs #11490",
    "SGLang Diffusion（multimodal_gen/）#12484": "SGLang Diffusion (multimodal_gen/) #12484",
    "RadixTree 重构系列开始": "the RadixTree restructuring series begins",
    "SLRU 淘汰 #18843、SWA 基数树": "SLRU eviction #18843, the SWA radix tree",
    "v0.5.21；基准提交 29f6d408c0（10-02）": "v0.5.21; the baseline commit 29f6d408c0 (10-02)"
  };
  var SGL_EVENTS = [            // [月份, 事件, 章节路径（相对手册根目录；空串 = 还没写到）]
    ["2023-10", "建仓：.gitignore、LICENSE、一行 README", "origins/paper/"], ["2023-12", "论文 v1 上 arXiv（2312.07104）", "origins/paper/"],
    ["2024-01", "release initial code：145 个文件、1.78 万行；#7 修基数树匹配；LMSYS 博客", "origins/first-commit/"],
    ["2024-01", "RadixAttention 第一版、从 Outlines 改编的 FSM、前端语言", "origins/radix-v1/"], ["2024-02", "jump-forward（#144）、import outlines（#168）", "origins/fsm-jump/"],
    ["2024-05", "静态数据并行 #480：controller 与 tp_worker", ""], ["2024-07", "去掉 rpyc #646、目录重构 #807、CUDA Graph 默认 #612、v0.2 博客（3.1×）", ""],
    ["2024-08", "mem_cache/、model_executor/、sampling/ 拆分；MLA Triton kernel", ""], ["2024-09", "v0.3：torch.compile、注意力后端抽象 #1381/#1547、scheduler.py 独立 #1538", ""],
    ["2024-10", "重叠调度 #1738、xgrammar #1752、rust/ 路由器", ""], ["2024-11", "DP attention #1970、sgl-kernel 起步", ""], ["2024-12", "v0.4 博客：零开销调度；EAGLE 四部曲开始 #2150", ""],
    ["2025-01", "entrypoints/、去 vLLM 依赖系列、HiCache 控制器", ""], ["2025-02", "HiCache #2693、llguidance #3298", ""], ["2025-03", "PD 分离 #4655、FA3 后端 #4709、页大小 > 1 #4356、删 jump-forward #4032", ""],
    ["2025-04", "Mooncake、NIXL 传输后端；sgl-router 独立目录", ""], ["2025-05", "TBO #4068、EPLB、96 卡 H100 博客", ""], ["2025-06", "OpenAI server 重构 #7167、eplb/", ""],
    ["2025-07", "multimodal/、weight_sync/、HiCache 存储后端", ""], ["2025-08", "v0.5.0rc0；简化前端 #9029", ""], ["2025-09", "gRPC 入口", ""],
    ["2025-10", "分配逻辑拆出调度器 #11313、piecewise CUDA graph #11490", ""], ["2025-11", "SGLang Diffusion（multimodal_gen/）#12484", ""],
    ["2026-01", "RadixTree 重构系列开始", ""], ["2026-03", "SLRU 淘汰 #18843、SWA 基数树", ""], ["2026-09", "v0.5.21；基准提交 29f6d408c0（10-02）", ""]
  ];
  function sglTimeline(box) {
    box.innerHTML = '<div class="aw-title">' + zhen("SGLang 的提交时间线：按月的提交数、版本与大事件（点一根柱子看那个月发生了什么）",
      "SGLang's commit timeline: the commits per month, the releases and the events (click a bar for that month)") + '</div>' +
      '<div class="aw-scroll"><svg class="aw-chart" viewBox="0 0 720 230" style="min-width:600px"></svg></div><div class="aw-out"></div>';
    var svg = box.querySelector("svg"), out = box.querySelector(".aw-out"), n = SGL_MONTHS.length, i;
    var base = (function () { var p = location.pathname, k = p.indexOf("/sglang/"); return k >= 0 ? p.slice(0, k + 8) : "./"; })();
    var mx = 0; for (i = 0; i < n; i++) mx = Math.max(mx, SGL_MONTHS[i][1]);
    var X = function (k) { return 46 + k * (660 / n); }, W = 660 / n - 2, Y = function (v) { return 180 - v / mx * 150; };
    function draw(sel) {
      var S = '<line x1="44" y1="180" x2="710" y2="180" class="aw-axis"/><line x1="44" y1="20" x2="44" y2="180" class="aw-axis"/>';
      [0, 500, 1000, 1500].forEach(function (v) { if (v <= mx) S += '<line x1="44" y1="' + Y(v).toFixed(1) + '" x2="710" y2="' + Y(v).toFixed(1) + '" class="aw-gl"/>' + svgText(40, Y(v) + 4, String(v), "end"); });
      for (i = 0; i < n; i++) {
        var m = SGL_MONTHS[i], x = X(i), y = Y(m[1]), has = SGL_EVENTS.some(function (e) { return e[0] === m[0]; });
        S += '<rect data-i="' + i + '" x="' + x.toFixed(1) + '" y="' + y.toFixed(1) + '" width="' + W.toFixed(1) + '" height="' + (180 - y).toFixed(1) + '" rx="2" class="' + (i === sel ? "aw-b" : (has ? "aw-f" : "aw-off")) + '" style="cursor:pointer"/>';
        if (m[0].slice(5) === "01") S += '<line x1="' + (x - 1).toFixed(1) + '" y1="180" x2="' + (x - 1).toFixed(1) + '" y2="186" class="aw-axis"/>' + svgText(x + W / 2, 198, m[0].slice(0, 4), "middle");
        if (SGL_TAGS[m[0]]) S += '<line x1="' + (x + W / 2).toFixed(1) + '" y1="' + (y - 4).toFixed(1) + '" x2="' + (x + W / 2).toFixed(1) + '" y2="' + (y - 16).toFixed(1) + '" class="aw-dash"/>' + svgText(x + W / 2, y - 20, SGL_TAGS[m[0]], "middle");
      }
      S += svgText(48, 222, zhen("蓝：这个月有书里讲到的事件；灰：没有；橙：当前选中。柱高 = 当月并入 main 的提交数（2024 年约 50 → 2026 年 1600）",
        "blue: a month with an event in the book; grey: none; orange: selected. Bar height = that month's commits merged into main"), "start");
      svg.innerHTML = S;
    }
    function describe(i) {
      var m = SGL_MONTHS[i], ev = SGL_EVENTS.filter(function (e) { return e[0] === m[0]; });
      var html = EN ? "<p><b>" + m[0] + "</b>: " + m[1] + " commits" + (SGL_TAGS[m[0]] ? ", " + SGL_TAGS[m[0]] + " released" : "") + "</p>"
        : "<p><b>" + m[0] + "</b>：" + m[1] + " 个提交" + (SGL_TAGS[m[0]] ? "，发布 " + SGL_TAGS[m[0]] : "") + "</p>";
      if (ev.length) html += "<ul>" + ev.map(function (e) {
        var t = zhen(e[1], SGL_EV_EN[e[1]] || e[1]);
        return "<li>" + (e[2] ? '<a href="' + base + e[2] + '">' + t + "</a>" : t + '<span class="aw-note">' + zhen("（后面的章节）", " (a later chapter)") + "</span>") + "</li>";
      }).join("") + "</ul>";
      else html += '<p class="aw-note">' + zhen("这个月没有书里单独讲的事件——日常的模型支持、修 bug 和 CI。",
        "no event of its own in the book this month — the everyday model support, bug fixes and CI.") + "</p>";
      out.innerHTML = html;
    }
    var sel = n - 2;
    draw(sel); describe(sel);
    svg.addEventListener("click", function (e) { var r = e.target.closest && e.target.closest("rect[data-i]"); if (!r) return; sel = +r.getAttribute("data-i"); draw(sel); describe(sel); });
  }

  var WIDGETS = { "sgl-timeline": sglTimeline, "kv-calc": kvCalc, roofline: roofline, mask: mask, pipeline: pipeline,
                  linmap: linmap, lowrank: lowrank, softmax: softmaxw, graddesc: graddesc,
                  coalesce: coalesce, bankconf: bankconf, scanviz: scanviz, occupancy: occupancy,
                  pagewalk: pagewalk, cachemap: cachemap, hashring: hashring,
                  pagedkv: pagedkv, radixcache: radixcache, contbatch: contbatch, spectree: spectree,
                  ringreduce: ringreduce, zeromem: zeromem, structlayout: structlayout,
                  embed3d: embed3d, "rope-helix": ropeHelix, swiglu3d: swiglu3d, scaling3d: scaling3d,
                  nextword: nextword, broadcast: broadcast, bpe: bpe, "float-bits": floatBits,
                  attention2d: attention2d, norm: normw, "param-share": paramShare, "moe-route": moeRoute,
                  "dpo-loss": dpoLoss, "lora-params": loraParams, estimator: estimator, quant: quantw, "model-map": modelMap,
                  diffusion3d: diffusion3d, "cfg-guide": cfgGuide, "ode-solver": odeSolver, "sigma-schedule": sigmaSchedule, "diffusion-flops": diffusionFlops,
                  "offload-cost": offloadCost, "cache-skip": cacheSkip, video3d: video3d, "video-stack": videoStack,
                  alphabeta: alphaBeta, "collective-cost": collectiveCost, "ragged-waste": raggedWaste, "launch-overhead": launchOverhead,
                  "linear-memory": linearMemory, "spec-load": specLoad, "pd-ratio": pdRatio,
                  "dp-straggler": dpStraggler, "kv-evict": kvEvict, "vision-tokens": visionTokens, "open-closed": openClosed,
                  "online-softmax": onlineSoftmax, gemm3d: gemm3d, "cute-layout": cuteLayout, "stream-overlap": streamOverlap, "grid-index": gridIndex, "stride-view": strideView,
                  "adamw-step": adamwStep, "newton-schulz": newtonSchulz, "lr-schedule": lrSchedule, "softmax-entropy": softmaxEntropy, "grpo-adv": grpoAdv,
                  "kl-estimators": klEstimators, "train-time": trainTime, "ddp-overlap": ddpOverlap, "tp-comm": tpComm,
                  complexity: complexity, "window-step": windowStep, "edit-distance": editDistance, "ridge-gen": ridgeGen, "little-law": littleLaw, "mm1-latency": mm1Latency,
                  "vector-realloc": vectorRealloc };
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
