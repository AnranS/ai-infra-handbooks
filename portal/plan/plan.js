// 冲刺计划页：数据 + 渲染。打卡和开始日期保存在本地浏览器（localStorage），练习题的完成状态读自练习题页。
(function () {
  "use strict";

  var BOOK = { python: "Python", llm: "大模型", cuda: "CUDA", serving: "推理系统", minisgl: "mini-sglang" };
  function L(book, path, title) { return [book, path, title]; }

  // ---------------------------------------------------------------- 目标能力
  var ABILITIES = [
    { c: "c-blue", t: "编码与算法", why: "所有面试的第一关，也是最容易被低估的一关。", pass: [
      "LeetCode 中等题 25 分钟内一次写对，困难题能给出可行思路和复杂度",
      "Hot 100 全部完成 + 约 100 道专项，做错的题二刷",
      "20 个推理系统高频组件能现场手写（注意力、采样、Radix Cache、线程池……）",
      "Python 和 C++ 都能写：推理系统的代码两种语言各占一半"],
      links: [L("serving", "career/coding", "手撕代码题"), ["practice", "", "练习题库"]] },
    { c: "c-indigo", t: "C++ 与系统编程", why: "开源推理基础库（通信、算子、存储）大量是 C++，读不懂就没法讨论细节。", pass: [
      "现代 C++ 写得地道：RAII、移动语义、智能指针、模板与 constexpr",
      "并发：线程、atomic、内存序、无锁队列、线程池",
      "能读懂并修改 DeepEP、FlashInfer、Mooncake 这一级别的 C++ / CUDA 工程代码",
      "会用 pybind11 / PyTorch C++ 扩展把实现暴露给 Python，会用 sanitizer 和 perf 定位问题"],
      links: [L("cuda", "basics/first-kernel", "第一个 CUDA 程序"), ["todo", "", "C++ 进阶手册"]] },
    { c: "c-orange", t: "GPU 与 CUDA", why: "推理优化最终落在 kernel 和带宽上，面试会追问到硬件数字。", pass: [
      "讲清 SM、warp、内存层次、Tensor Core，记住 H100 / H800 的关键数字",
      "手写归约、转置、分块 GEMM、online softmax、FlashAttention 前向",
      "用 Nsight Compute 找到瓶颈并用数字说明离屋顶线还差多少",
      "Triton 能写融合算子；理解 Hopper 的 TMA、WGMMA、FP8 在做什么"],
      links: [L("cuda", "kernels/gemm", "GEMM 优化之路"), L("cuda", "advanced/attention", "FlashAttention"), L("cuda", "tools/profiling", "Nsight")] },
    { c: "c-purple", t: "大模型原理与数值", why: "推理系统的每一处优化都要回到模型结构和数值精度上找依据。", pass: [
      "手推注意力、FFN、RoPE、RMSNorm 的前向与反向",
      "MLA 的低秩压缩与矩阵吸收、MoE 路由与负载均衡、MTP 的原理",
      "FP8 / INT4 量化的误差来源与缩放粒度",
      "对任意模型估算参数量、FLOPs、KV Cache 大小和 decode 的带宽上限"],
      links: [L("llm", "transformer/attention-variants", "MQA / GQA / MLA"), L("llm", "transformer/moe", "MoE"), L("llm", "inference/estimation", "估算")] },
    { c: "c-green", t: "推理引擎", why: "岗位的主战场：要能从请求到 token 讲清每一步、每一处取舍。", pass: [
      "画出 SGLang / vLLM 从请求到返回 token 的全流程，解释每处设计取舍",
      "手写分页 KV、连续批处理、前缀缓存、分块 prefill、重叠调度、CUDA Graph",
      "自己写的引擎在 GPU 上跑通，并和 SGLang 做性能对比与消融",
      "至少给一个主流推理框架提交过被合入的 PR"],
      links: [L("serving", "engine/overview", "从零写推理引擎"), L("minisgl", "overview/architecture", "手写 mini-sglang"), L("serving", "source/sglang", "SGLang 源码")] },
    { c: "c-teal", t: "分布式推理与大规模部署", why: "大模型推理是多机多卡的大系统，系统设计题几乎都落在这里；训练侧的并行也常被追问。", pass: [
      "TP / EP / PP / CP 各自的通信模式与通信量",
      "DeepEP 的高吞吐与低延迟两种模式；RDMA、NVLink 与 NVSHMEM 的角色",
      "PD 分离与 KV 传输、以 KV Cache 为中心的多级缓存架构",
      "训练侧：DP / ZeRO / FSDP、Megatron TP + SP、PP 调度，能算清显存账本和通信量",
      "设计一个数百 B 参数 MoE 模型的在线服务，给出机器数、吞吐和延迟估算"],
      links: [L("serving", "distributed/expert-parallel", "专家并行"), L("serving", "distributed/pd-disagg", "PD 分离"), ["todo", "", "通信与存储"], ["todo", "", "分布式训练手册"]] },
  ];

  // ---------------------------------------------------------------- 三类岗位
  var DIRS = [
    ["日常工作",
      "调度、KV 管理、并行策略；新模型适配与特性开发",
      "注意力、GEMM、MoE、量化 kernel 与通信算子；Profiling 与端到端调优",
      "大规模部署、PD 分离、KV 缓存池、路由与弹性伸缩；稳定性与成本"],
    ["考察重点",
      "引擎全流程与每处设计取舍；源码细节；手写调度器与缓存",
      "CUDA / Triton 现场手写；硬件关键数字；屋顶线分析与数值精度",
      "系统设计与容量估算；通信与存储；故障处理与扩缩容"],
    ["必读代码",
      "SGLang、vLLM、mini-sglang、FlashInfer 的调用侧",
      "FlashAttention、FlashMLA、DeepGEMM、CUTLASS / CuTe、Triton 教程",
      "Mooncake、NVIDIA Dynamo、LMCache、llm-d、SGLang Model Gateway"],
    ["计划里对应", "第 5～8 周 + 作品 A", "第 2～3、10、13 周 + 作品 C", "第 9、11～12、14 周 + 作品 B"],
  ];

  // ---------------------------------------------------------------- 差异化
  var EDGE = [
    { c: "c-blue", t: "主线：推理系统基本盘", why: "所有候选人都要过；达不到这一线，放大器无从谈起。", pass: [
      "调度、KV 管理、分布式、性能分析四件事讲得清、写得出",
      "C++ / CUDA 达到能读懂、能修改开源基础库的程度",
      "mini-sglang 全链路复刻 + 性能报告，作为最基础的作品"] },
    { c: "c-orange", t: "放大器一：Rust 与系统工程", why: "推理平台的数据面越来越多用 Rust：SGLang 的 Router / Model Gateway、Hugging Face tokenizers 都是 Rust 实现。", pass: [
      "作品 B：用 Rust 写一个 PD 感知、缓存感知的推理网关",
      "把网关里的经验变成给 SGLang Rust 网关的 PR",
      "用数字说话：尾延迟、吞吐、缓存命中率相对基线的提升"] },
    { c: "c-purple", t: "放大器二：大型系统架构", why: "系统设计面试最能拉开差距：PD 分离、KV 缓存池、多租户、全局调度本质上都是分布式系统问题。", pass: [
      "10 道推理系统设计题，每道都有容量估算、故障域和扩缩容方案",
      "用分布式系统的通用语言讲推理系统：调度、缓存、传输、隔离、可观测性",
      "对比两种大规模推理架构：以 KV Cache 为中心的分离式架构，和大规模专家并行 + PD 分离，作为面试谈资"] },
    { c: "c-teal", t: "放大器三：运行时与工具化", why: "推理引擎本身就是一个运行时：事件循环、任务调度、内存池、异步执行（CUDA Stream / Graph）、零拷贝。", pass: [
      "从运行时的视角讲重叠调度、CUDA Graph、KV 内存管理，讲出自己的理解",
      "作品 D（可选）：推理性能可视化 / 调度回放工具，把 profiler 和引擎日志变成可交互的时间线",
      "面试时现场演示工具定位一个真实的性能问题，辨识度很高"] },
  ];

  // ---------------------------------------------------------------- 阶段与逐周
  var PHASES = [
    { c: "c-blue", name: "阶段一 · 基础夯实", weeks: [1, 5] },
    { c: "c-green", name: "阶段二 · 框架深入", weeks: [6, 9] },
    { c: "c-orange", name: "阶段三 · 前沿专题与作品", weeks: [10, 13] },
    { c: "c-pink", name: "阶段四 · 面试冲刺", weeks: [14, 17] },
  ];

  var WEEKS = [
    { t: "诊断 + 数学与 Transformer 复盘 + C++ 起步", g: "摸清自己的底；把大模型的数学和 Transformer 前向复习到能手写。",
      learn: [L("llm", "synthesis/quiz", "自测题库（先做诊断）"), L("llm", "math/linear-algebra", "线性代数"), L("llm", "math/probability", "概率与采样"),
        L("llm", "math/information-theory", "信息论"), L("llm", "math/calculus", "微积分与反向传播"), L("llm", "math/floating-point", "浮点与数值计算"),
        L("llm", "transformer/attention", "注意力机制"), L("llm", "transformer/position", "RoPE"), L("llm", "transformer/norm-residual", "归一化与残差"),
        L("llm", "transformer/ffn", "SwiGLU")],
      todo: ["C++ 手册：值语义、RAII、移动语义、智能指针"],
      practice: ["llm-stable-softmax", "llm-cross-entropy", "llm-micrograd", "llm-linear-ce-backward", "llm-bf16", "llm-causal-mha", "llm-rope", "llm-rmsnorm", "llm-swiglu"],
      algo: "数组、哈希、双指针、滑动窗口：25 题",
      out: ["开发环境：本地 NVIDIA GPU（WSL2）+ 按需租用的 Hopper 云主机账号；Mac 上装好练习题环境（practice/env）",
        "开一个技术笔记仓库（后面整理成博客），第一篇：手推 softmax + 交叉熵的反向传播"],
      check: ["用自测题库做一次诊断，把不会的章节标出来，调整后面几周的顺序", "20 分钟手写带因果掩码的多头注意力（numpy）",
        "推导 softmax + 交叉熵的梯度，解释数值稳定为什么要减最大值", "讲清 fp16、bf16、fp8 的位宽分配和各自的风险"] },
    { t: "GPU 架构与 CUDA 基础 + C++ 内存", g: "建立 GPU 的心智模型：线程层级、内存层次、占用率、warp 级编程。",
      learn: [L("cuda", "basics/gpu-architecture", "GPU 架构"), L("cuda", "basics/first-kernel", "第一个 CUDA 程序"), L("cuda", "basics/memory", "内存层次"),
        L("cuda", "basics/execution", "执行模型"), L("cuda", "basics/sync-warp", "同步与 warp 编程")],
      todo: ["C++ 手册：对象布局、对齐、分配器与内存池"],
      practice: ["cu-thread-index", "cu-vector-add", "cu-stencil-smem", "cu-grid-stride", "cu-occupancy", "cu-warp-reduce", "cu-histogram"],
      algo: "链表、栈、队列、二分：25 题",
      out: ["向量加法的 CUDA C++ 版本在自己的 GPU 上通过判题，记录带宽占峰值的比例"],
      check: ["讲清 SM、warp、线程块、共享内存、L2、HBM 的关系，说出 H100 的关键数字", "手算一个 kernel 的占用率并指出瓶颈资源",
        "白板写出 warp shuffle 归约和共享内存私有化的直方图", "解释合并访存和 bank conflict，并用练习题里模拟器的统计验证"] },
    { t: "经典算子与 Triton + C++ 并发", g: "把优化 kernel 的套路练熟，并学会用 profiler 说话。",
      learn: [L("cuda", "kernels/reduction", "归约"), L("cuda", "kernels/transpose", "转置"), L("cuda", "kernels/gemm", "GEMM 优化之路"),
        L("cuda", "kernels/softmax-norm", "Softmax 与归一化"), L("cuda", "kernels/scan", "前缀和"), L("cuda", "tools/triton", "Triton"),
        L("cuda", "tools/profiling", "Nsight")],
      todo: ["C++ 手册：线程、atomic、内存序、线程池", "PyTorch 内部机制：张量与 stride、autograd、dispatcher 与自定义算子、显存分配器"],
      practice: ["cu-reduction", "cu-transpose-smem", "cu-gemm-tiled", "cu-online-softmax", "cu-block-scan", "cu-triton-softmax", "cu-triton-matmul"],
      algo: "二叉树、DFS、BFS：25 题",
      out: ["四个算子在真卡上的性能报告（归约、转置、GEMM、softmax 的 CUDA C++ 版本），附 Nsight Compute 分析"],
      check: ["45 分钟写出共享内存分块 GEMM 和 online softmax（CUDA C++）", "用 ncu 读懂 memory / compute throughput，说出离屋顶线差在哪",
        "用 Triton 写出融合 softmax 和分块 matmul，说出和 CUDA 写法的分工差异", "用 C++ 写出带任务队列、可优雅退出的线程池"] },
    { t: "现代模型结构与推理原理", g: "把 GQA、MLA、MoE、量化和估算方法吃透，为后面的系统设计打数字基础。",
      learn: [L("llm", "transformer/build-llm", "从零组装大模型"), L("llm", "transformer/attention-variants", "MQA / GQA / MLA"), L("llm", "transformer/moe", "MoE"),
        L("llm", "inference/decoding", "解码与采样"), L("llm", "inference/kv-cache", "KV Cache"), L("llm", "inference/estimation", "参数量与显存估算"),
        L("llm", "inference/quantization", "量化原理"), L("llm", "inference/serving", "推理服务概念"), L("llm", "math/performance-math", "性能数学")],
      practice: ["llm-mini-qwen", "llm-gqa", "llm-moe-router", "llm-kv-cache-decode", "llm-param-count", "llm-int4-quant", "llm-roofline", "llm-sampling", "llm-beam-search", "llm-serving-metrics"],
      algo: "堆、贪心、区间：20 题",
      out: ["一页纸估算：一个 600B 级 MLA + MoE 模型（如 DeepSeek-V3）decode 一步的权重读取量、KV 读取量、FLOPs 与带宽下限"],
      check: ["手推 MLA：KV 为什么能压到 512 + 64 维，矩阵吸收合并了哪些矩阵，decode 时省了什么", "解释 MoE 的 top-k 路由、负载均衡损失，以及无辅助损失的均衡方法",
        "对任意 config.json 算出参数量、每 token FLOPs、KV 字节数", "用屋顶线讲清 prefill 与 decode 的瓶颈差异，以及攒批为什么有效"] },
    { t: "推理引擎原理：从零写 nano engine", g: "亲手实现分页 KV、连续批处理、前缀缓存、采样与流式输出。",
      learn: [L("serving", "engine/overview", "一个请求的一生"), L("serving", "engine/paged-kv", "分页 KV Cache"), L("serving", "engine/batch-layout", "变长批处理"),
        L("serving", "engine/scheduler", "调度器"), L("serving", "engine/prefix-cache", "前缀缓存"), L("serving", "engine/sampler-api", "采样与接口"),
        L("serving", "engine/graphs-compile", "CUDA Graphs 与 torch.compile")],
      practice: ["sv-block-pool-cow", "sv-batch-layout", "sv-scheduler", "sv-prefix-cache-pool", "sv-stop-strings", "sv-graph-buckets"],
      algo: "回溯、动态规划入门：20 题（累计约 115）",
      out: ["跟着手册写完 nano engine，在 GPU 上跑通并测出吞吐 / 延迟曲线"],
      check: ["讲清连续批处理、分块 prefill、抢占（重算 vs 换出）的取舍", "讲清哈希块与基数树两种前缀缓存的差异",
        "解释 CUDA Graph 为什么只用于 decode，以及补齐的代价", "里程碑 M1 全部达标"] },
    { t: "手写 mini-sglang（上）：算得对、排得好", g: "对照一个真实、完整的代码库，从零实现模型执行与调度器。",
      learn: [L("minisgl", "overview/architecture", "导读"), L("minisgl", "compute/core", "核心数据结构"), L("minisgl", "compute/kvcache", "KV 池与 page table"),
        L("minisgl", "compute/attention", "注意力后端"), L("minisgl", "compute/engine", "Engine 与采样"), L("minisgl", "schedule/scheduler", "调度器"),
        L("minisgl", "schedule/cache-manager", "CacheManager"), L("minisgl", "schedule/radix-cache", "Radix Cache"), L("minisgl", "schedule/chunked-prefill", "分块 prefill"),
        L("minisgl", "schedule/overlap", "重叠调度")],
      practice: ["ms-req-lengths", "ms-base-op", "ms-weight-stream", "ms-paged-alloc", "ms-attn-metadata", "ms-batched-sampler", "ms-prefill-adder", "ms-cache-req", "ms-radix-cache", "ms-chunk-size", "ms-overlap-loop"],
      algo: "动态规划：20 题",
      out: ["自己的 mini-sglang 仓库：前 12 章的代码与测试全部通过（先自己写，写完再对照手册）"],
      check: ["不看资料画出进程结构和一个请求经过的所有消息", "解释准入控制为什么按最坏情况预留、代价是什么",
        "讲清重叠调度的四个问题各自的触发条件和修法", "40 分钟内写出 Radix Cache 的匹配、插入、加锁、淘汰"] },
    { t: "手写 mini-sglang（下）：服务化与更快", g: "把引擎变成服务，接上 GPU 注意力后端、CUDA Graph、张量并行和 MoE，做性能报告。",
      learn: [L("minisgl", "serve/message", "消息与 ZMQ"), L("minisgl", "serve/tokenizer", "增量反分词"), L("minisgl", "serve/api-server", "API Server"),
        L("minisgl", "perf/tensor-parallel", "张量并行"), L("minisgl", "perf/gpu-attention", "FlashInfer / FlashAttention"), L("minisgl", "perf/cuda-graph", "CUDA Graph"),
        L("minisgl", "perf/kernels", "自定义 kernel"), L("minisgl", "perf/moe", "fused MoE"), L("minisgl", "perf/benchmark", "基准测试")],
      todo: ["AI 编译器：计算图优化、torch.compile（Dynamo + Inductor）、TVM / MLIR 的思路"],
      practice: ["ms-message-serde", "ms-incremental-detok", "ms-sse-stream", "ms-shard-tensor", "ms-flashinfer-meta", "ms-graph-replay", "ms-store-kv-kernel", "ms-moe-align"],
      algo: "图、并查集、拓扑排序：15 题",
      out: ["作品 A：mini-sglang 在 GPU 上跑通 Qwen3，性能报告对比 SGLang 的吞吐、TTFT、TPOT，并做消融（重叠调度、CUDA Graph、Radix Cache 各贡献多少）"],
      check: ["说清增量反分词为什么要保留上下文窗口", "说清 FlashInfer 的 plan / run 分离在解决什么问题",
        "说清 CUDA Graph 漏拷一个输入会怎样、怎样在 CPU 上发现", "性能报告里的每个数字都能解释来源"] },
    { t: "SGLang / vLLM 源码 + 第一个 PR", g: "从“写过一个”走到“读懂工业级实现”，并开始出现在开源社区里。",
      learn: [L("serving", "source/sglang", "SGLang 源码导读"), L("serving", "source/vllm", "vLLM V1 源码导读"), L("minisgl", "wrap/next-steps", "与 SGLang 的差距")],
      practice: [],
      algo: "字符串、前缀树、单调栈：15 题",
      out: ["源码笔记：SGLang 从 HTTP 请求到返回 token 的调用链（附图），标出和 mini-sglang 的差异",
        "第一个 PR：从文档、测试或可复现的小 bug 开始（SGLang / vLLM / Mooncake 任选）"],
      check: ["5 分钟内在 SGLang 源码里找到调度器、KV 分配、CUDA Graph、模型注册的位置", "讲清 vLLM V1 与 SGLang 在调度和前缀缓存上的主要差异", "PR 已提交"] },
    { t: "分布式推理与训练、通信与存储", g: "掌握多卡多机的并行方式、通信原语和 KV 传输；训练侧的并行也要能讲清。",
      learn: [L("serving", "distributed/tensor-parallel", "张量并行"), L("serving", "distributed/expert-parallel", "专家并行与 DP Attention"), L("serving", "distributed/pp-cp", "流水线与上下文并行"),
        L("serving", "distributed/pd-disagg", "PD 分离与 KV 传输"), L("serving", "distributed/kv-offload", "KV 分层缓存"), L("cuda", "tools/multi-gpu", "多 GPU 与 NCCL"),
        L("cuda", "tools/streams", "流与 CUDA Graphs")],
      todo: ["通信与存储：RDMA 与 GPUDirect、NVSHMEM 与 DeepEP、KV 传输引擎", "分布式训练：DP 与 ZeRO / FSDP、Megatron TP + SP、PP 调度、混合精度与重计算"],
      practice: ["sv-tp-mlp", "sv-ep-dispatch", "sv-ring-attention", "sv-kv-transfer-plan", "sv-kv-offload", "cu-ring-allreduce", "cu-stream-schedule", "cu-trace-analysis"],
      algo: "错题重做 + 每周 2 场限时模拟",
      out: ["估算文档：8 卡节点上 TP=8 与 EP=8 部署同一个 MoE 模型，每步的通信量与耗时对比"],
      check: ["讲清 all-reduce、all-gather、reduce-scatter、all-to-all 的通信量与适用场景", "讲清 DeepEP 高吞吐与低延迟两种模式为什么这样设计",
        "讲清 PD 分离的收益、代价和 KV 传输方案", "算出 70B 模型用 ZeRO-3 与 TP + PP 训练时每卡的显存账本", "里程碑 M2 全部达标"] },
    { t: "前沿专题一：大规模 MoE 推理", g: "把 MLA、FP8、大规模专家并行和 MTP 这一套开源推理栈读到能讨论细节、能提改进的程度。",
      learn: [L("llm", "transformer/attention-variants", "MLA 回顾"), L("cuda", "advanced/attention", "FlashAttention 与推理算子"), L("cuda", "advanced/async-hopper", "Hopper 异步拷贝与 TMA"),
        L("cuda", "advanced/tensor-core", "Tensor Core 与 mma"), L("cuda", "advanced/quantization", "量化与 GEMV"), L("serving", "topics/speculative", "投机解码进阶")],
      todo: ["专题：MLA 与 FlashMLA、FP8 与 DeepGEMM、DeepEP 与大规模 EP、MTP 与稀疏注意力、开源推理系统复盘"],
      practice: ["cu-flash-attn", "cu-paged-decode", "cu-mma-layout", "cu-async-pipeline", "cu-gemv-int4", "sv-spec-verify", "sv-tree-verify"],
      algo: "每周 2 场限时模拟",
      out: ["《大规模 MoE 推理系统复盘》：以公开的 DeepSeek-V3 推理系统为例，PD 分离、EP 规模、双 micro-batch 重叠、负载均衡，每个设计都配估算数字",
        "租一台 Hopper 机器跑通 FlashMLA、DeepGEMM 的 benchmark，读懂它们的主循环"],
      check: ["讲清 FlashMLA 解决什么瓶颈、和普通分页 decode 的区别", "讲清 DeepGEMM 的细粒度 FP8 缩放与 JIT 的动机",
        "讲清 MTP 如何用于投机解码、DSA 的索引器在选什么", "回答：把这套推理系统搬到另一种硬件上，哪些设计要改"] },
    { t: "前沿专题二：分离式架构、长上下文与 RL 推理", g: "吃透以 KV Cache 为中心的分离式架构，以及长上下文与 RL 场景的推理问题。",
      learn: [L("serving", "distributed/pd-disagg", "PD 分离回顾"), L("serving", "distributed/kv-offload", "KV 分层缓存"), L("serving", "topics/long-context", "长上下文与 KV 淘汰"),
        L("serving", "topics/rl-rollout", "RL 训练中的推理"), L("serving", "perf/benchmark", "压测与 SLO")],
      todo: ["专题：KV 中心的分离式架构（Mooncake、Dynamo、LMCache）、稀疏与线性注意力、RL rollout 与权重同步"],
      practice: ["sv-cache-aware-router", "sv-kv-eviction", "sv-rollout-sharing", "sv-memory-plan", "sv-capacity-plan", "sv-step-breakdown", "sv-json-fsm"],
      algo: "每周 2 场限时模拟",
      out: ["《分离式推理架构分析》：调度器、KV 池、传输引擎，对比 Mooncake、NVIDIA Dynamo、LMCache 三种方案",
        "在 SGLang 或 vLLM 里用 Mooncake 做一次 PD 分离实验（单机多卡即可），记录 TTFT / TPOT 的变化"],
      check: ["讲清以 KV Cache 为中心的调度在优化什么目标、有哪些约束", "讲清稀疏注意力（NSA、MoBA）与线性注意力（Gated DeltaNet 一类）分别怎样降低长上下文的代价",
        "讲清 RL rollout 和在线服务的差异，以及权重更新要解决什么"] },
    { t: "作品冲刺一：Rust 推理网关", g: "做出最能体现系统工程能力的作品，并把经验反馈到开源项目。",
      learn: [L("serving", "engine/prefix-cache", "缓存感知的调度与路由"), L("serving", "distributed/pd-disagg", "PD 分离"), L("serving", "career/projects", "作品集建议")],
      todo: ["作品指南：PD 感知的 Rust 网关"],
      practice: ["sv-cache-aware-router", "sv-capacity-plan"],
      algo: "每天 2 题保持手感",
      out: ["作品 B：OpenAI 兼容与流式、缓存感知路由、PD 配对、指标与压测报告", "第 2、3 个 PR（优先 SGLang 的 Rust 网关或 KV 传输相关项目）"],
      check: ["压测报告有基线对比（轮询、最少连接）", "README 能让陌生人 10 分钟跑起来", "能讲清每个设计取舍和一个失败的尝试"] },
    { t: "作品冲刺二：GPU 算子 + 技术博客", g: "证明能下到 GPU 底层，并把前 12 周的积累写成公开文章。",
      learn: [L("cuda", "advanced/attention", "推理算子"), L("cuda", "career/projects", "CUDA 作品集"), L("cuda", "tools/profiling", "Nsight")],
      practice: [],
      algo: "每天 2 题",
      out: ["作品 C：MLA decode 算子（Triton 或 CUDA）或分块缩放的 FP8 GEMM，含对拍测试与 benchmark",
        "发布 3 篇技术博客：mini-sglang 性能报告、大规模 MoE 推理复盘、作品 B 或 C", "（可选）作品 D：推理性能可视化工具"],
      check: ["作品 C 的性能占峰值比例有数字，并用 ncu 解释剩余差距", "里程碑 M3 全部达标"] },
    { t: "系统设计 + 简历 + 练手面试", g: "把知识组织成面试能用的形状。",
      learn: [L("serving", "career/system-design", "系统设计题"), L("serving", "career/interview", "推理岗面试题库"), L("serving", "career/projects", "作品集与简历")],
      todo: ["系统设计题库扩充到 10 题", "生产部署与运维：Kubernetes 上的推理服务、模型加载加速、弹性伸缩、可观测性"],
      practice: [],
      algo: "每周 3 场限时模拟",
      out: ["10 道系统设计题的答案（架构图 + 估算）", "简历定稿（一页），每个项目三段式：问题 → 方案 → 数字", "投递 2～3 家同类岗位练手"],
      check: ["每道系统设计题 45 分钟内讲完且有容量估算", "请人做 2 次模拟面试并复盘"] },
    { t: "基础追问冲刺 + 第一批正式面试", g: "把知识点磨成“一句话答案 + 能接住追问”。",
      learn: [L("serving", "career/interview", "推理岗面试题库"), L("llm", "synthesis/quiz", "大模型自测题库"), L("cuda", "career/interview", "CUDA 面试题库"),
        L("serving", "career/hardware", "硬件与生态速查")],
      practice: [],
      algo: "Hot 100 二刷中做错的题",
      out: ["面试题卡：每个知识点一句话答案 + 一个追问，约 150 张", "第一批正式投递（优先内推）"],
      check: ["模拟面试 ≥ 2 次", "每场真实面试后 24 小时内写复盘，补上没答好的点"] },
    { t: "第二批正式面试", g: "最想去的岗位放在状态最好的时候面。",
      learn: [L("serving", "distributed/expert-parallel", "专家并行回顾"), L("llm", "transformer/moe", "MoE 回顾"), L("cuda", "advanced/attention", "推理算子回顾")],
      practice: ["llm-causal-mha", "llm-sampling", "ms-radix-cache", "py-blocking-queue", "cu-gemm-tiled"],
      algo: "每天 2 题 + 手撕组件轮换",
      out: ["第二批投递（优先内推）", "针对岗位描述，把相关论文各准备 10 个可能的追问和答案"],
      check: ["限时手撕一遍：注意力、top-p 采样、Radix Cache、线程池、分块 GEMM", "每个项目准备 3 个难点、3 个数字、1 个失败的尝试"] },
    { t: "缓冲、复盘与决策", g: "应对加面、补面，做出选择。",
      learn: [L("serving", "career/projects", "学习计划与作品集")],
      practice: [],
      algo: "按面试反馈补弱项",
      out: ["加面 / 补面准备", "比较 offer：团队方向（推理系统、训练基础设施、存储与网络）、成长空间"],
      check: ["整理 17 周的笔记与作品，形成可长期维护的个人主页"] },
  ];

  var MILESTONES = [
    { c: "c-blue", when: "第 5 周末", t: "M1 基础合格", items: ["Python、大模型、CUDA 三本的练习题通过率 ≥ 80%", "30 分钟白板写出带 KV Cache 的 GQA 注意力 + RoPE",
      "45 分钟写出 warp 归约与分块 GEMM（CUDA C++）", "算法题累计约 115 道，中等题 25 分钟通过率 ≥ 70%", "能用数字讲清 prefill 与 decode 的瓶颈"] },
    { c: "c-green", when: "第 9 周末", t: "M2 引擎合格", items: ["自己的 mini-sglang 在 GPU 上跑通，性能报告完成", "推理系统与 mini-sglang 的练习题通过率 ≥ 80%",
      "画出 SGLang 全流程并逐一解释设计取舍", "第 1 个 PR 已提交"] },
    { c: "c-orange", when: "第 13 周末", t: "M3 作品合格", items: ["作品 B、C 完成，3 篇技术博客发布", "PR ≥ 3，至少 1 个被合入",
      "两份前沿专题复盘能脱稿各讲 15 分钟", "达到这里就可以开始投递，不必等到第 15 周"] },
    { c: "c-pink", when: "第 16 周", t: "M4 面试就绪", items: ["算法题约 200 道，Hot 100 二刷完成", "10 道系统设计题、模拟面试 ≥ 6 次",
      "题卡约 150 张、精读清单全部读完", "每个项目能经受 30 分钟的连续追问"] },
  ];

  var PROJECTS = [
    { c: "c-green", t: "作品 A · mini-sglang 全链路复刻", why: "必做（第 6～7 周）。证明理解推理引擎的每一个环节。", pass: [
      "按手册从零实现（先自己写，写完再对照），GPU 上跑通 Qwen3，输出与 Hugging Face 逐 token 一致",
      "与 SGLang 对比吞吐、TTFT、TPOT；消融：重叠调度、CUDA Graph、Radix Cache、分块 prefill 各贡献多少",
      "交付：代码仓库 + 性能报告 + 一篇博客"], links: [L("minisgl", "overview/architecture", "手写 mini-sglang"), L("minisgl", "perf/benchmark", "基准测试与消融")] },
    { c: "c-orange", t: "作品 B · PD 感知的 Rust 推理网关", why: "差异化（第 12 周）。体现系统工程与 Rust 能力，直接对接真实推理框架。", pass: [
      "OpenAI 兼容接口与 SSE 流式；缓存感知路由（近似前缀树）+ 负载均衡",
      "prefill / decode 实例配对与 KV 传输协调（对接 SGLang 的 PD 分离）；健康检查、熔断、重试；Prometheus 指标",
      "压测：对比轮询与最少连接的 TTFT、吞吐、缓存命中率；故障注入测试",
      "加分：思路或代码贡献到 SGLang 的 Rust 网关"], links: [L("serving", "engine/prefix-cache", "缓存感知路由"), L("serving", "distributed/pd-disagg", "PD 分离"), ["practice", "sv-cache-aware-router", "练习：缓存感知路由"]] },
    { c: "c-purple", t: "作品 C · GPU 算子", why: "深度（第 13 周）。证明能下到 GPU 底层，而不只会调用库。", pass: [
      "二选一：MLA decode 算子（参考 FlashMLA 的思路，先 Triton 后 CUDA），或分块缩放的 FP8 GEMM（参考 DeepGEMM）",
      "与参考实现对拍；在 Hopper 上给出带宽或 TFLOPS 占峰值的比例",
      "用 Nsight Compute 解释剩余差距和下一步优化方向"], links: [L("cuda", "advanced/attention", "推理算子"), L("cuda", "advanced/async-hopper", "Hopper 异步拷贝"), L("cuda", "career/projects", "CUDA 作品集")] },
    { c: "c-teal", t: "作品 D · 推理性能可视化工具（可选）", why: "加分项。把工程工具化能力变成面试时可以现场演示的东西。", pass: [
      "把 torch profiler trace 与引擎日志变成可交互时间线：每步 batch 组成、KV 占用、CPU / GPU 空闲、请求 TTFT 分解",
      "直接加载 SGLang / vLLM 的日志与 trace，给出自动诊断（例如 CPU 开销占比过高、CUDA Graph 未命中）",
      "交付：在线 demo + 用它定位一个真实性能问题的案例文章"], links: [L("serving", "perf/profiling", "Profiling 推理引擎"), ["practice", "cu-trace-analysis", "练习：时间线分析"]] },
    { c: "c-pink", t: "开源贡献（贯穿第 8～16 周）", why: "面试官能直接读到你的代码，比任何描述都有说服力。", pass: [
      "目标 3～5 个 PR，至少 1～2 个非文档的实质修改",
      "路径：文档与测试 → 复现并修 bug → 小功能；复现步骤、测试和基准数据写进 PR 描述",
      "关注 SGLang（含 Rust 网关）、vLLM、FlashInfer、Mooncake 等项目的 issue"], links: [L("serving", "source/sglang", "SGLang 源码导读"), L("serving", "source/vllm", "vLLM 源码导读")] },
  ];

  var ALGO = ["第 1 周：数组、哈希、双指针、滑动窗口（25）", "第 2 周：链表、栈、队列、二分（25）", "第 3 周：二叉树、DFS / BFS（25）",
    "第 4 周：堆、贪心、区间（20）", "第 5 周：回溯、动态规划入门（20）", "第 6 周：动态规划（20）", "第 7 周：图、并查集、拓扑排序（15）",
    "第 8 周：字符串、前缀树、单调栈（15）", "第 9～13 周：每周 2 场限时模拟 + 错题重做（约 35）", "第 14～16 周：Hot 100 二刷错题，面试前每天 2 题保持手感"];
  var HANDWRITE = [
    ["数值稳定的 softmax / logsumexp", "llm-stable-softmax"], ["带因果掩码的多头 / GQA 注意力 + KV Cache", "llm-kv-cache-decode"], ["RoPE", "llm-rope"],
    ["RMSNorm / LayerNorm", "llm-rmsnorm"], ["temperature / top-k / top-p 采样", "llm-sampling"], ["束搜索", "llm-beam-search"], ["BPE 训练与编码", "llm-bpe"],
    ["LRU 缓存", "py-lru-cache"], ["Radix Cache", "ms-radix-cache"], ["分页 KV 块分配器（引用计数、写时复制）", "sv-block-pool-cow"],
    ["连续批处理调度器", "sv-scheduler"], ["MoE 路由与分发", "llm-moe-router"], ["投机解码验证", "sv-spec-verify"], ["阻塞队列 / 线程池", "py-blocking-queue"],
    ["无锁单生产者单消费者队列（C++）", ""], ["CUDA：归约、转置、分块 GEMM", "cu-gemm-tiled"], ["online softmax（CUDA / Triton）", "cu-online-softmax"],
    ["FlashAttention 前向（分块）", "cu-flash-attn"], ["ring all-reduce", "cu-ring-allreduce"], ["动态批处理器（asyncio）", "py-dynamic-batcher"]];
  var DESIGN = ["600B 级 MoE 模型的在线推理服务：PD 分离 + 大规模 EP，给出机器数、吞吐、延迟",
    "以 KV Cache 为中心的多级缓存池：GPU / CPU / SSD 分层、跨机传输、淘汰与一致性",
    "全局调度与缓存感知的推理网关：路由策略、负载均衡、故障转移、灰度发布",
    "多租户 LoRA 服务：适配器加载与换出、批处理、隔离与计费",
    "RL rollout 系统：训练与推理共用集群、权重同步、长尾请求、弹性",
    "百万 token 长上下文服务：上下文并行、稀疏注意力、KV 卸载、首 token 延迟",
    "投机解码服务化：草稿模型部署、接受率监控、与批处理的相互影响",
    "结构化输出与工具调用服务：约束解码的开销、流式解析、超时与重试",
    "推理可观测性与自动扩缩容：指标体系、SLO、容量预测、冷启动",
    "跨地域多集群推理平台：流量调度、模型分发、容灾与成本优化"];
  var PAPERS = [
    "★ Orca：迭代级调度（OSDI'22）", "★ vLLM：PagedAttention（SOSP'23）", "★ SGLang：RadixAttention（NeurIPS'24）",
    "★ Sarathi-Serve：分块 prefill（OSDI'24）", "★ DistServe / Splitwise：PD 分离（OSDI'24 / ISCA'24）", "★ FlashAttention 1 / 2 / 3",
    "★ DeepSeek-V2 / V3 技术报告（MLA、MoE、MTP、FP8）", "★ Mooncake：以 KV Cache 为中心的分离式架构（FAST'25）", "★ Megatron-LM 与 ZeRO：训练侧的并行与显存",
    "FlashInfer（MLSys'25）", "DeepSeek 开源推理系统概览与 FlashMLA、DeepEP、DeepGEMM", "Insights into DeepSeek-V3（ISCA'25，硬件与模型协同）",
    "NVIDIA Dynamo、LMCache、llm-d 的设计文档", "稀疏与线性注意力：NSA、MoBA、Gated DeltaNet", "Medusa 与 EAGLE 1 / 2 / 3：投机解码",
    "SmoothQuant、AWQ、GPTQ：量化", "GShard / Switch Transformer：MoE 与专家并行", "DeepSeek-R1 与 GRPO：推理模型与 RL",
    "HybridFlow（verl）：RL 训练框架中的推理"];
  var TIMELINE = ["第 8 周起：在开源社区和博客里留下可检索的痕迹；整理内推渠道（同事、社区、技术群）",
    "第 13～14 周：投 2～3 家同类岗位练手，熟悉面试节奏", "第 15 周：第一批正式投递（优先内推），用结果调整最后两周的重点",
    "第 16 周：第二批正式投递，最想去的岗位放在状态最好的时候", "每场面试后 24 小时内复盘：问了什么、答得如何、要补什么",
    "招聘窗口不确定时，以“M3 达标即可投递”为准"];
  var RESUME = ["一页；第一屏放目标方向（推理系统 / AI Infra）、作品和开源贡献，再放过往经历",
    "每个项目三段式：问题（为什么难）→ 方案（关键设计）→ 数字（吞吐、延迟、成本、规模）",
    "过往经历用推理系统的语言重述：调度、内存与缓存、异步执行、网络传输、可观测性、稳定性",
    "准备 3 个故事：一次性能优化、一次复杂系统设计、一次线上问题排查，每个都有数字",
    "GitHub、博客、作品 demo、PR 链接直接写进简历"];
  var RHYTHM = [["c-blue", "7 h", "算法题：工作日每天 1 小时（约 2 题）"], ["c-green", "14 h", "主线学习：章节 + 练习题 + 验收清单"],
    ["c-orange", "6 h", "作品 / 开源：周末集中推进"], ["c-purple", "3 h", "输出与复盘：周笔记、博客、题卡；周日对照验收清单打勾"]];
  var BUILD = [
    { c: "c-indigo", t: "P0 · C++ 进阶手册（面向 AI Infra）", why: "约 14 章 + 本地判题的 C++ 练习。对应第 1～3 周。", pass: [
      "现代 C++ 核心：值语义、RAII、移动、智能指针、模板与 constexpr", "内存：对象布局、对齐、分配器与内存池",
      "并发：线程、atomic、内存序、无锁队列、线程池与协程", "工程：CMake、测试、sanitizer、perf；pybind11 与 PyTorch C++ 扩展；阅读开源基础库的方法"] },
    { c: "c-green", t: "P0 · 分布式训练手册", why: "约 11 章 + 练习题。现有手册只覆盖推理侧的并行，训练侧是明显缺口。对应第 9 周，也支撑 RL 推理。", pass: [
      "总论与显存账本：参数、梯度、优化器状态、激活", "集合通信原语与 NCCL；DP / DDP；ZeRO 1 / 2 / 3 与 FSDP",
      "Megatron TP + SP、PP 调度（1F1B、交错、零气泡）、CP 与长序列训练、MoE 与 EP", "混合精度与 FP8 训练、重计算与卸载；3D 并行策略选择；Megatron-LM、DeepSpeed、torchtitan、verl 实战"] },
    { c: "c-teal", t: "P0 · 通信与存储（推理系统手册新篇章）", why: "对应第 9 周。", pass: [
      "RDMA 与 GPUDirect RDMA、NVLink / NVSwitch 与 NCCL 的算法", "NVSHMEM 与 DeepEP 的实现", "KV 传输引擎与 KV 缓存存储（Mooncake Transfer Engine、3FS）"] },
    { c: "c-blue", t: "P1 · 前沿专题：大规模 MoE 推理", why: "5 章。对应第 10 周。", pass: [
      "MLA 与 FlashMLA", "FP8 细粒度量化与 DeepGEMM", "DeepEP 与大规模专家并行、EPLB", "MTP 与稀疏注意力（NSA、DSA）", "开源推理系统复盘与估算"] },
    { c: "c-purple", t: "P1 · 前沿专题：分离式架构、长上下文与 RL 推理", why: "4 章。对应第 11 周。", pass: [
      "KV 中心的分离式架构：Mooncake、NVIDIA Dynamo、LMCache 对比", "稀疏与线性注意力的推理实现", "超大 MoE 的低比特推理：INT4 QAT、FP4", "RL rollout、权重同步与训练推理一体化"] },
    { c: "c-orange", t: "P1 · PyTorch 内部机制与 AI 编译器", why: "各 3～4 章。对应第 3、7 周。", pass: [
      "张量、stride 与视图；autograd 引擎；dispatcher 与自定义算子；CUDA 显存分配器", "torch.distributed 与 torchrun",
      "计算图优化与算子融合；torch.compile（Dynamo + Inductor）；TVM / MLIR / XLA 的思路；TileLang 与 CuTe DSL"] },
    { c: "c-red", t: "P1 · 作品指南", why: "对应第 8、12、13 周。", pass: [
      "作品 B、C、D 的设计文档模板、里程碑和验收指标", "开源贡献入门：各项目的代码导览、如何挑 issue、PR 的写法"] },
    { c: "c-pink", t: "P2 · 面试题库扩充", why: "对应第 14～16 周。", pass: [
      "系统设计 10 题的完整参考答案（架构图 + 估算）", "手撕 20 题全部配上练习题（补上 C++ 的无锁队列、线程池）",
      "模拟面试套卷：按真实面试结构组卷（基础追问 + CUDA / PyTorch 编程 + 算法），附答案与章节链接", "按主题整理的追问清单与算法题单页"] },
    { c: "c-indigo", t: "P2 · 推理系统手册补充", why: "对应第 14 周。", pass: [
      "生产部署与运维：Kubernetes 上的推理服务、模型加载加速、弹性伸缩、灰度与可观测性", "推理框架选型：vLLM、SGLang、TensorRT-LLM、LMDeploy 的取舍",
      "多 LoRA 服务", "端侧推理：llama.cpp 与 GGUF、MLX、ExecuTorch"] },
  ];

  // ---------------------------------------------------------------- 工具
  function $(s) { return document.querySelector(s); }
  function esc(s) { return String(s).replace(/[&<>"]/g, function (c) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]; }); }
  function href(l) {
    if (l[0] === "practice") return "../practice/" + (l[1] ? "#/p/" + l[1] : "");
    return "../" + l[0] + "/" + (l[1] ? l[1] + "/" : "");
  }
  function linkHTML(l) {
    if (l[0] === "todo") return '<span>' + esc(l[2]) + '<span class="tag todo">建设中</span></span>';
    var book = BOOK[l[0]] ? BOOK[l[0]] + " · " : "";
    return '<a href="' + href(l) + '">' + esc(book + l[2]) + "</a>";
  }
  var state = { start: "", checks: {}, open: {} };
  try { var saved = JSON.parse(localStorage.getItem("aig-plan") || "null"); if (saved) state = Object.assign(state, saved); } catch (e) { /* ignore */ }
  function save() { try { localStorage.setItem("aig-plan", JSON.stringify(state)); } catch (e) { /* ignore */ } }
  var practiceStatus = {};
  try { practiceStatus = JSON.parse(localStorage.getItem("aig-practice:status") || "{}") || {}; } catch (e) { /* ignore */ }

  function cards(el, list) {
    el.innerHTML = list.map(function (a) {
      return '<div class="card ' + a.c + '"><h3><i></i>' + esc(a.t) + "</h3>" + (a.why ? '<p class="why">' + esc(a.why) + "</p>" : "") +
        "<ul>" + a.pass.map(function (p) { return "<li>" + esc(p) + "</li>"; }).join("") + "</ul>" +
        (a.links ? '<div class="links">' + a.links.map(linkHTML).join(" · ") + "</div>" : "") + "</div>";
    }).join("");
  }

  // ---------------------------------------------------------------- 渲染
  var totalChecks = WEEKS.reduce(function (n, w) { return n + w.check.length; }, 0);
  var practiceIds = {};
  WEEKS.forEach(function (w) { w.practice.forEach(function (p) { practiceIds[p] = 1; }); });
  $("#stats").innerHTML = [["17", "周"], ["4", "个阶段"], ["3 + 1", "个作品"], ["约 200", "道算法题"], [Object.keys(practiceIds).length, "道配套练习题"], ["10", "道系统设计"]]
    .map(function (s) { return '<div class="stat"><b>' + s[0] + "</b><span>" + s[1] + "</span></div>"; }).join("");
  cards($("#abilities"), ABILITIES);
  cards($("#edgeCards"), EDGE);
  cards($("#projCards"), PROJECTS);
  cards($("#buildCards"), BUILD);
  $("#msCards").innerHTML = MILESTONES.map(function (m) {
    return '<div class="card ' + m.c + '"><div class="when">' + esc(m.when) + "</div><h3>" + esc(m.t) + "</h3><ul>" +
      m.items.map(function (i) { return "<li>" + esc(i) + "</li>"; }).join("") + "</ul></div>";
  }).join("");
  $("#cmp").innerHTML = "<tr><th></th><th>推理框架</th><th>推理优化</th><th>推理平台</th></tr>" + DIRS.map(function (r) {
    return "<tr><td>" + esc(r[0]) + "</td><td>" + esc(r[1]) + "</td><td>" + esc(r[2]) + "</td><td>" + esc(r[3]) + "</td></tr>";
  }).join("");
  function ol(el, list) { el.innerHTML = list.map(function (x) { return "<li>" + esc(x) + "</li>"; }).join(""); }
  ol($("#algo"), ALGO);
  ol($("#design"), DESIGN);
  ol($("#papers"), PAPERS);
  ol($("#timeline"), TIMELINE);
  ol($("#resume"), RESUME);
  $("#handwrite").innerHTML = HANDWRITE.map(function (h) {
    return "<li>" + (h[1] ? '<a href="../practice/#/p/' + h[1] + '" style="color:var(--blue)">' + esc(h[0]) + "</a>" : esc(h[0]) + '<span class="tag todo">建设中</span>') + "</li>";
  }).join("");
  $("#rhythmCards").innerHTML = RHYTHM.map(function (r) {
    return '<div class="card ' + r[0] + '"><b>' + r[1] + "</b><span>" + esc(r[2]) + "</span></div>";
  }).join("");

  // 阶段条
  var ph = "";
  PHASES.forEach(function (p) {
    var nm = p.name.split(" · ");
    ph += '<div class="ph ' + p.c + '" style="grid-column:' + p.weeks[0] + " / " + (p.weeks[1] + 1) + '"><b><span class="pn">' + esc(nm[0]) + "<i> · </i></span>" + esc(nm[1]) +
      "</b><small>第 " + p.weeks[0] + "～" + p.weeks[1] + " 周</small></div>";
  });
  for (var i = 1; i <= 17; i++) ph += '<div class="wk" data-w="' + i + '">' + i + "</div>";
  $("#phases").innerHTML = ph;

  function phaseOf(n) { return PHASES.filter(function (p) { return n >= p.weeks[0] && n <= p.weeks[1]; })[0]; }
  var titles = {};
  function renderWeeks(names) {
    $("#weekList").innerHTML = WEEKS.map(function (w, idx) {
      var n = idx + 1, p = phaseOf(n);
      var practice = w.practice.length ? '<h4>练习题</h4><div class="chips">' + w.practice.map(function (id) {
        var meta = names[id];
        var label = meta ? meta.number + ". " + meta.title : id;
        return '<a class="chip' + (practiceStatus[id] === "solved" ? " solved" : "") + '" href="../practice/#/p/' + id + '">' + esc(label) + "</a>";
      }).join("") + "</div>" : "";
      var todo = (w.todo || []).map(function (t) { return "<li>" + esc(t) + '<span class="tag todo">建设中</span></li>'; }).join("");
      return '<div class="week ' + p.c + '" id="w' + n + '" data-w="' + n + '"><div class="wh"><div class="no">第 ' + n + ' 周<small class="date"></small></div>' +
        '<div class="tt"><b>' + esc(w.t) + "</b><span>" + esc(w.g) + '</span></div><div class="pc"></div><div class="arr">›</div></div>' +
        '<div class="wb"><h4>主线学习</h4><ul>' + w.learn.map(function (l) { return "<li>" + linkHTML(l) + "</li>"; }).join("") + todo + "</ul>" +
        practice + "<h4>算法题</h4><ul><li>" + esc(w.algo) + "</li></ul>" +
        "<h4>产出</h4><ul>" + w.out.map(function (o) { return "<li>" + esc(o) + "</li>"; }).join("") + "</ul>" +
        '<h4>验收清单</h4><ul class="checks">' + w.check.map(function (c, j) {
          var id = "w" + n + "-" + j;
          return '<li><input type="checkbox" id="' + id + '"' + (state.checks[id] ? " checked" : "") + '><label for="' + id + '">' + esc(c) + "</label></li>";
        }).join("") + "</ul></div></div>";
    }).join("");
    document.querySelectorAll(".week .wh").forEach(function (h) {
      h.addEventListener("click", function () {
        var wk = h.parentElement;
        wk.classList.toggle("open");
        state.open[wk.dataset.w] = wk.classList.contains("open");
        save();
      });
    });
    document.querySelectorAll(".checks input").forEach(function (cb) {
      cb.addEventListener("change", function () { state.checks[cb.id] = cb.checked; save(); refresh(); });
    });
    refresh();
  }

  function currentWeek() {
    if (!state.start) return 0;
    var d0 = new Date(state.start + "T00:00:00");
    var days = Math.floor((Date.now() - d0.getTime()) / 86400000);
    return days < 0 ? 0 : Math.floor(days / 7) + 1;
  }
  function refresh() {
    var done = 0;
    WEEKS.forEach(function (w, idx) {
      var n = idx + 1, k = 0;
      w.check.forEach(function (_, j) { if (state.checks["w" + n + "-" + j]) k++; });
      done += k;
      var el = document.getElementById("w" + n);
      if (!el) return;
      el.querySelector(".pc").textContent = k + " / " + w.check.length;
      var cell = document.querySelector('.wk[data-w="' + n + '"]');
      cell.classList.toggle("done", k === w.check.length);
      var cw = currentWeek();
      el.classList.toggle("now", cw === n);
      cell.classList.toggle("now", cw === n);
      if (state.open[n] === undefined && cw === n) el.classList.add("open");
      else if (state.open[n]) el.classList.add("open");
      var date = el.querySelector(".date");
      if (state.start) {
        var d = new Date(state.start + "T00:00:00");
        d.setDate(d.getDate() + (n - 1) * 7);
        date.textContent = (d.getMonth() + 1) + "/" + d.getDate();
      } else date.textContent = "";
    });
    $("#bar").style.width = (done / totalChecks * 100).toFixed(1) + "%";
    $("#progText").textContent = done + " / " + totalChecks;
    var cw = currentWeek();
    $("#nowText").textContent = !state.start ? "（设置后会标出当前周）" : cw === 0 ? "（还没开始）" : cw > 17 ? "（计划已结束）" : "· 现在是第 " + cw + " 周";
  }

  $("#start").value = state.start || "";
  $("#start").addEventListener("change", function (e) { state.start = e.target.value; state.open = {}; save(); renderWeeks(titles); });
  $("#reset").addEventListener("click", function () {
    if (confirm("清空所有验收清单的打卡记录？")) { state.checks = {}; save(); renderWeeks(titles); }
  });
  $("#expand").addEventListener("click", function () {
    var all = document.querySelectorAll(".week");
    var open = Array.prototype.some.call(all, function (w) { return !w.classList.contains("open"); });
    all.forEach(function (w) { w.classList.toggle("open", open); state.open[w.dataset.w] = open; });
    $("#expand").textContent = open ? "全部收起" : "全部展开";
    save();
  });
  $("#phases").addEventListener("click", function (e) {
    var c = e.target.closest(".wk");
    if (!c) return;
    var w = document.getElementById("w" + c.dataset.w);
    w.classList.add("open");
    state.open[c.dataset.w] = true;
    save();
    w.scrollIntoView({ behavior: "smooth", block: "start" });
  });

  renderWeeks({});
  fetch("../practice/data/index.json").then(function (r) { return r.json(); }).then(function (idx) {
    idx.problems.forEach(function (p) { titles[p.slug] = { number: p.number, title: p.title }; });
    renderWeeks(titles);
  }).catch(function () { /* 离线时只显示 id */ });

  document.getElementById("theme").addEventListener("click", function () {
    var root = document.documentElement;
    var dark = root.dataset.theme ? root.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
    root.dataset.theme = dark ? "light" : "dark";
    try { localStorage.setItem("aig-theme", root.dataset.theme); } catch (e) { /* ignore */ }
  });
})();
