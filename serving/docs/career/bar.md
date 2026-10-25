# 面试达标线：推理岗要达到什么水平

<p class="lead">“学完了”和“能去面试了”之间隔着一条线：原理要能讲清、代码要能限时写对、数字要能当场估出来、项目要经得起追问。这一章把这条线拆成能自己检验的条目，每条写明要做到什么程度、在哪里学、怎么验证。条目按推理相关岗位的招聘信息里反复出现的要求整理，分成所有推理岗都要过的<b>基本盘</b>、推理框架 / 推理优化 / 推理平台三类岗位各自的<b>加深项</b>，以及做大规模推理的团队常问的<b>前沿专题</b>。</p>

!!! tip "怎么用"
    - 判断标准是“能不能当场做到”，不是“学过没有”。每条标了目标等级：**L1 讲清**（不看资料讲出原理、取舍和数字）、**L2 写对**（独立写出能通过测试的代码）、**L3 做快**（限时完成，或性能达到给定门槛）、**L4 交付**（有可复现的作品、报告或合入的 PR）。四级各自怎么验证见[冲刺计划](root://plan/)。
    - “讲清”用[模拟面试的评分表](mock-interview.md#评分表)来量：正确性、深度、数字、沟通四项都 ≥ 3 分才算过。
    - 先过基本盘，再按目标岗位补加深项。投递前把整章自测一遍，没过的条目就是最后要补的；招聘信息写到而这里没有的，以招聘信息为准。

## 基本盘：所有推理岗都要过

### 编码

| 要求 | 达标线 | 在哪练 |
| --- | --- | --- |
| 算法题 | 常见题型都练过：数组与哈希、双指针与滑动窗口、链表、栈与单调栈、二分、树与图、堆与贪心、回溯与动态规划、前缀树与并查集。中等题 25 分钟内写对的比例 ≥ 70%，困难题能说出思路和复杂度；累计约 200 道，热题 100 至少二刷。**L3** | [算法面试怎么准备](cs://algo/overview/)和算法各章 |
| 手撕模型组件 | 30 分钟内写出带 KV Cache 的 GQA 注意力和 RoPE；写对 temperature、top-k、top-p 采样和数值稳定的 softmax。**L3** | [手撕代码题](coding.md)，练习题 [GQA 注意力](root://practice/#/p/llm-gqa)、[采样](root://practice/#/p/llm-sampling) |
| 手撕系统组件 | 40 分钟内写出 Radix Cache 的匹配、插入、加锁、淘汰；写对连续批处理的调度器（token 预算、分块 prefill、抢占）；写对线程池或有界阻塞队列。**L3** | 练习题 [Radix Cache](root://practice/#/p/ms-radix-cache)、[调度器](root://practice/#/p/sv-scheduler)、[线程池](root://practice/#/p/cpp-thread-pool) |
| 手写 GPU kernel | 45 分钟内用 CUDA C++ 写出 warp 归约和共享内存分块 GEMM，并说清每一步优化省下了什么。推理平台岗做到 L1 即可。**L3** | [归约](cuda://kernels/reduction/)、[GEMM 优化之路](cuda://kernels/gemm/)，练习题 [warp 归约](root://practice/#/p/cu-warp-reduce)、[分块 GEMM](root://practice/#/p/cu-gemm-tiled) |

### 估算

推理岗的面试几乎每一轮都会让你“估一下”。估算是拉开差距最快的一项：同样讲清了原理，能当场给出量级的人明显更可信。

| 要求 | 达标线 | 在哪练 |
| --- | --- | --- |
| 屋顶线 | 用算术强度和屋脊点讲清 decode 为什么受访存限制、prefill 为什么受计算限制、攒批为什么有效。**L1** | [一个 token 的完整旅程](llm://synthesis/token-journey/)、[面试题库的估算题](interview.md#一估算与基础) |
| 模型账本 | 拿到任意 config.json，算出参数量、每 token 的 FLOPs、每 token 的 KV 字节数（MHA、GQA、MLA 都会算），误差 ≤ 10%。**L3** | [参数量、算力与显存估算](llm://inference/estimation/)、[估算类练习题](root://practice/#/?q=估算) |
| 部署账本 | 估出一张卡能撑多少并发、batch=1 时 TPOT 的下限、一个服务要多少张卡；估出 TP 与 EP 每步的通信量。**L3** | [压测、SLO 与容量规划](../perf/benchmark.md)、[专家并行与 DP Attention](../distributed/expert-parallel.md) |
| 训练显存 | 算出一个 70B 模型在 ZeRO-3 和 TP + PP 两种方案下每张卡的显存账本。**L3** | [分布式训练总论](train://basics/overview/)、[ZeRO 与 FSDP](train://data/zero-fsdp/) |

### 推理引擎

- 讲清一个请求从进入服务到返回 token 经过的每一步，以及每一步的瓶颈。**L1** → [一个请求的一生](../engine/overview.md)
- 讲清连续批处理、分块 prefill、抢占（重算还是换出）的取舍，分页 KV 解决了什么、代价是什么，前缀缓存的哈希块与基数树两种做法。**L1** → [调度器](../engine/scheduler.md)、[分页 KV Cache](../engine/paged-kv.md)、[前缀缓存](../engine/prefix-cache.md)
- 讲清 CUDA Graph 为什么只用于 decode、补齐的代价，重叠调度在省什么，增量反分词为什么要保留上下文窗口。**L1** → [CUDA Graphs 与 torch.compile](../engine/graphs-compile.md)、[采样器与流式输出](../engine/sampler-api.md)
- 自己写过一个能跑的迷你引擎，贪心解码的输出与 Hugging Face 逐 token 一致。**L2** → [从零写一个推理引擎](../engine/overview.md)、[手写 mini-sglang](minisgl://)
- 5 分钟内在 SGLang 源码里找到调度器、KV 分配、CUDA Graph、模型注册的位置；讲清 vLLM V1 与 SGLang 在调度和前缀缓存上的主要差异。**L3** → [SGLang 源码导读](../source/sglang.md)、[vLLM V1 源码导读](../source/vllm.md)

### 分布式与通信

- 讲清 TP、PP、EP、DP、CP 各自切什么、通信什么；all-reduce、all-gather、reduce-scatter、all-to-all 的通信量和适用场景。**L1** → [张量并行](../distributed/tensor-parallel.md)、[流水线并行与上下文并行](../distributed/pp-cp.md)、[集合通信](../comm/nccl.md)
- 讲清 PD 分离的收益、代价和 KV 传输方案，DeepEP 高吞吐与低延迟两种模式为什么这样设计；说出 NVLink、PCIe、RDMA 网络的带宽量级。**L1** → [PD 分离与 KV 传输](../distributed/pd-disagg.md)、[NVSHMEM 与 DeepEP](../comm/nvshmem-deepep.md)、[GPU 互联与网络](../comm/interconnect.md)
- 从零实现 DDP（梯度分桶 + 通信与计算重叠），多进程的结果与单进程对齐。**L2** → [数据并行与 DDP](train://data/ddp/)

### 计算机基础

做系统方向的岗位，面试里常有一轮专门问基础；推理框架岗也常从基础追问到推理场景（比如从虚拟内存问到分页 KV）。

| 方向 | 达标线（都是 **L1**） | 在哪学 |
| --- | --- | --- |
| 操作系统 | 进程、线程与调度；虚拟内存、页表与大页；一次写入从系统调用到落盘（page cache、fsync、O_DIRECT）；epoll 与 io_uring；共享内存与 CUDA IPC；namespace 与 cgroup；锁页内存、DMA 与 NUMA；会用 perf 一类工具定位瓶颈 | [进程、线程与调度](cs://os/process-thread/)、[虚拟内存](cs://os/virtual-memory/)、[一次写入如何落盘](cs://os/io-stack/)、[epoll 与 io_uring](cs://os/io-models/)、[进程间通信](cs://os/ipc/)、[容器](cs://os/containers/)、[锁页内存与 NUMA](cs://os/pinned-numa/)、[性能分析工具](cs://os/perf-tools/) |
| 网络 | TCP 的握手、拥塞控制与队头阻塞；HTTP 流式输出；负载均衡与排队的基本结论；RDMA 的单边操作与内存注册 | [TCP](cs://net/tcp/)、[HTTP 与流式输出](cs://net/http-stream/)、[负载均衡与排队](cs://net/load-balance/)、[RDMA 编程模型](../comm/rdma.md) |
| 分布式基础 | 一致性哈希与分片；复制与共识（Raft 的选主与日志复制） | [一致性哈希与分片](cs://dist/hash-shard/)、[复制与共识](cs://dist/replication/) |
| 体系结构 | CPU 的缓存与一致性；GPU 的 SM、warp、Tensor Core 与内存层次，说出主流卡的关键数字 | [CPU 体系结构速成](cs://arch/cpu/)、[GPU 的 SM 与 Tensor Core](cs://arch/gpu-sm/)、[GPU 内存系统](cs://arch/gpu-memory/) |
| 编译原理 | 从源码到语法树、中间表示与优化 pass、循环变换、代码生成；AI 编译器（torch.compile、Triton、MLIR 一类）各在哪一层 | [编译原理速成](cuda://framework/compiler-basics/)、[AI 编译器全景](cuda://framework/compilers/) |

### 系统设计

- [系统设计题](system-design.md)里的 10 道题，每道 45 分钟内讲完：澄清需求 → 容量估算 → 架构 → 关键路径 → 故障、扩缩容与可观测性，每一步都带数字。**L3** → [参考答案（一）](design-answers-1.md)、[参考答案（二）](design-answers-2.md)

### 项目、开源与简历

- 至少一个能跑、有对比数字的推理系统作品。比如自己写的 mini-sglang：同一张卡、同一负载下吞吐达到 SGLang 的 60% 以上，性能报告里每个数字都能说出来源。**L4** → [作品集、简历与学习计划](projects.md)
- 每个项目准备 3 个难点、3 个数字、1 个失败的尝试，经得起 30 分钟的连续追问。**L3** → [作品指南](portfolio-guide.md)
- 至少 1 个 PR 被推理框架或相关开源项目合入；文档、测试、可复现的小 bug 都是好的起点。**L4** → [开源贡献入门](portfolio-guide.md#开源贡献入门)
- 简历一页，每个项目按“问题 → 方案 → 数字”三段写。**L4** → [作品集、简历与学习计划](projects.md)

### 模拟面试

- 至少 6 场完整的模拟面试（编码、基础追问、系统设计、项目深挖四段），最近 3 场四项评分都 ≥ 3 分。**L3** → [评分表与复盘](mock-interview.md)、[模拟面试套卷](mock-exams.md)
- 每个知识点写成“一句话答案 + 一个追问”的题卡，累计约 150 张；[学习卡](root://cards/)复习到都答得上来。**L1**

## 加深项：按岗位

三类岗位的基本盘相同，面试深挖的方向不同。三类岗位各自的日常工作见[岗位画像](projects.md#岗位画像)。

### 推理框架开发

- 引擎细节能对照源码讲到实现：调度器、KV 管理与前缀缓存、CUDA Graph 与重叠调度、投机解码（EAGLE、MTP）、结构化输出。**L1** → [投机解码进阶](../topics/speculative.md)、[结构化输出与工具调用](../topics/structured-output.md)
- 并行在框架里怎么落地：TP 的权重切分与通信、EP 与 DP 注意力、PD 分离的 KV 传输。**L2** → [张量并行：从零实现](../distributed/tensor-parallel.md)、[专家并行与 DP Attention](../distributed/expert-parallel.md)
- 新模型接入与精度对齐、权重热更新、多 LoRA 服务。**L1** → [新模型接入与精度对齐](../ops/new-model.md)、[权重热更新](../ops/weight-update.md)、[多 LoRA 服务](../ops/multi-lora.md)
- PyTorch 的 CUDA 运行时（异步执行、stream、缓存分配器）、dispatcher 与自定义算子；Python 之外熟练 C++ 或 Rust 中的一门。**L2** → [PyTorch 的 CUDA 运行时](cuda://framework/cuda-runtime/)、[dispatcher 与自定义算子](cuda://framework/dispatcher/)
- 达标作品：自己的迷你引擎在 GPU 上跑通，并与 SGLang 或 vLLM 在同一负载下对比；最好再有一个非文档类的 PR 合入推理框架。**L4**

### 推理优化 / 算子

- 真卡性能门槛：归约达到实测带宽的 80%，SGEMM 达到 cuBLAS 的 70%（练习题的银档）；能用 Nsight Compute 讲清离屋顶线还差在哪。**L3** → [性能分析](cuda://tools/profiling/)
- 用 Triton 写出融合 softmax 和分块 matmul，说出它和 CUDA 写法的分工；了解 TileLang 这类更低一层的 tile 语言。**L2** → [Triton](cuda://tools/triton/)、[TileLang](cuda://tools/tilelang/)
- Hopper 以后的新特性：TMA 与异步拷贝、WGMMA、CuTe 的布局代数；读懂 FlashMLA、DeepGEMM 的主循环，并能在真卡上复现公开 benchmark 的量级。**L3** → [异步拷贝与 Hopper 新特性](cuda://advanced/async-hopper/)、[CuTe 的布局代数](cuda://advanced/cute-layout/)、[MLA 推理与 FlashMLA](../moe/mla.md)、[FP8 分组 GEMM](../moe/fp8-gemm.md)
- 量化：FP8 细粒度缩放、INT4 与 FP4（MXFP4、NVFP4）、GPTQ、AWQ、SmoothQuant 与量化感知训练各自的取舍，能估算收益。**L1** → [量化与 GEMV](cuda://advanced/quantization/)、[量化部署实战](../perf/quantization-deploy.md)、[超大 MoE 的低比特推理](../frontier/low-bit.md)
- kernel 之间的空隙：PDL、megakernel、CUDA Graph。**L1** → [PDL 与 megakernel](cuda://tools/pdl-megakernel/)
- 达标作品：一个 MLA decode 注意力或分块缩放的 FP8 GEMM 算子，带对拍测试和 benchmark，性能占峰值的比例有数字。**L4** → [作品指南：GPU 算子](portfolio-guide.md#作品-cgpu-算子)

### 推理平台 / 部署

- Kubernetes：GPU 怎么被调度和隔离、gang 调度、Operator、探针、滚动发布与扩缩容。**L1** → [GPU 在 Kubernetes 里怎么被管起来](../k8s/gpu.md)、[调度器](../k8s/scheduling.md)、[部署形态](../k8s/deploy-scale.md)、[Operator、配额与排障](../k8s/operator-ops.md)
- 路由与流量：缓存感知路由、PD 配对、负载均衡、排队与限流。**L2** → [分离式架构的全局调度](../frontier/disagg-sched.md)、[负载均衡与排队](cs://net/load-balance/)
- KV 分层与存储：卸载到 CPU 内存和 SSD、分布式 KV 存储与传输引擎；io_uring、NVMe、RDMA 在这条链路里各管哪一段。**L1** → [KV 分层缓存与卸载](../distributed/kv-offload.md)、[KV 传输引擎与分布式 KV 存储](../comm/kv-storage.md)
- 压测、SLO 与容量规划；指标、日志与追踪。**L3** → [压测、SLO 与容量规划](../perf/benchmark.md)、[生产部署与运维](../ops/deploy.md)
- Go 或 Rust 至少熟练一门。**L2** → [vLLM 的 Rust 前端](../source/rust-frontend.md)
- 达标作品：一个 OpenAI 兼容、支持流式输出的推理网关，带缓存感知路由、指标和压测报告，并和轮询、最少连接做基线对比。**L4** → [作品指南：推理网关](portfolio-guide.md#作品-bpd-感知的推理网关rust)

## 前沿专题：做大规模推理的团队常问

招聘信息里写到“大规模分布式推理”“MoE”“长上下文”“RL 系统”的岗位，在基本盘之外通常会深挖下面这些。达标线是 **L1**：每个专题能脱稿讲 15 分钟，讲清动机、做法、代价，并带上数字。

- 大规模 MoE 推理：EP 的规模、DP 注意力、双 micro-batch 重叠、分层 EPLB、容错与弹性扩缩；能拿一个公开的大规模 MoE 推理系统做完整复盘。→ [复盘：一个公开的大规模 MoE 推理系统](../moe/case-study.md)、[大规模 EP 部署](../moe/ep-deploy.md)、[容错与弹性扩缩](../moe/ep-elastic.md)
- 注意力与解码的新做法：MLA 与 FlashMLA、MTP 投机解码、稀疏注意力（NSA、DSA）、整块草稿的投机解码。→ [MLA 推理](../moe/mla.md)、[MTP 与稀疏注意力](../moe/mtp-sparse.md)、[投机解码的新做法](../frontier/spec-next.md)
- 分离式架构与长上下文：KV 感知路由、xPyD、PD 复用，长上下文的 KV 淘汰，线性注意力与混合架构。→ [分离式架构的全局调度](../frontier/disagg-sched.md)、[PD 复用](../frontier/pd-multiplex.md)、[长上下文](../topics/long-context.md)、[线性注意力与混合架构](../frontier/linear-attn.md)
- RL 推理系统：异步 rollout、策略陈旧度、权重更新。→ [RL 推理系统](../frontier/rl-async.md)、[RL 训练中的推理](../topics/rl-rollout.md)
- 新模型与多硬件：新一代开源模型的结构变化，platform 抽象与非 NVIDIA 加速卡的适配思路。→ [新一代开源模型](../frontier/new-models.md)、[多硬件支持](../ops/platforms.md)

## 投递前自测

下面每一条都当场做一遍。做不到的，就是投递前最后要补的。

- [ ] 随机抽 3 道中等算法题，每道 25 分钟内写对
- [ ] 30 分钟写出带 KV Cache 的 GQA 注意力和 RoPE，40 分钟写出 Radix Cache
- [ ] 给一个没见过的模型的 config.json，10 分钟内算出参数量、每 token 的 KV 字节数和 batch=1 时 TPOT 的下限
- [ ] 不看资料画出一个请求在推理引擎里经过的所有步骤，讲清每一步的瓶颈
- [ ] 讲清 TP、EP、PD 分离各自的通信量和代价
- [ ] 讲清一次文件写入从系统调用到落盘、一次流式响应从 GPU 到浏览器各经过了什么
- [ ] 任选一道系统设计题，45 分钟讲完并给出容量估算
- [ ] 每个项目都能讲出 3 个难点、3 个数字、1 个失败的尝试
- [ ] 至少 1 个 PR 已合入，简历一页定稿
- [ ] 最近 3 场模拟面试，四项评分都 ≥ 3 分
