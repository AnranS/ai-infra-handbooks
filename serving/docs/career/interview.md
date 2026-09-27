# 推理岗面试题库

<p class="lead">这里汇总了推理优化、推理框架开发、AI Infra 岗位面试中最常见的问题，按主题分组，每题给出答题要点和展开阅读的章节。模型原理类的题目见大模型手册的<a href="llm://synthesis/quiz/">自测题库</a>，CUDA 与算子类的题目见 CUDA 手册的<a href="cuda://career/interview/">面试题库</a>，这里侧重推理系统本身。</p>

!!! tip "怎么用"
    - 先只看问题，用 2～3 分钟口头回答，再展开要点对照；
    - 答题的结构通常是：**是什么 → 为什么（解决了什么瓶颈）→ 怎么做 → 代价与取舍 → 数字**。最后一项最能拉开差距：能估算的地方就给出数量级；
    - 标了 ★ 的是高频题，务必熟练。

## 一、估算与基础

**1. ★ decode 为什么是访存受限的？prefill 呢？**

??? success "要点"
    decode 时每个请求每步只处理 1 个 token，每读一遍权重（2 字节/参数）只做约 2 次浮点运算，算术强度约等于 batch 大小，远低于 H100 约 295 FLOP/字节的屋脊点；prefill 一次处理成百上千个 token，强度高，计算受限。所以 decode 优化读字节（批处理、量化、KV 压缩），prefill 优化计算（FP8、分块、高效注意力）。见[一个 token 的完整旅程](llm://synthesis/token-journey/#decode-的时间花在哪里)。

**2. ★ 估算 LLaMA-3-8B 在 H100 上 batch=1 的 TPOT 下限。**

??? success "要点"
    权重 8B × 2 字节 = 16 GB，H100 带宽 3.35 TB/s，约 4.8 ms/token；加上 KV（短上下文可忽略）和 kernel 开销，实际约 6～7 ms。INT4 权重约 1/3.5 的时间。见[参数量、算力与显存估算](llm://inference/estimation/)。

**3. ★ 一个 token 的 KV Cache 多大？LLaMA-3-70B 呢？**

??? success "要点"
    2（K 和 V）× 层数 × KV 头数 × head_dim × 字节数。LLaMA-3-70B：2 × 80 × 8 × 128 × 2 = 320 KB/token，128K 上下文约 43 GB。MLA 的 DeepSeek-V3 约 69 KB/token。见[主流模型架构巡礼](llm://synthesis/models/)。

**4. 一张 80 GB 的卡部署 Qwen2.5-7B，平均上下文 3000，能支持多少并发？**

??? success "要点"
    权重约 15 GB，预留约 10 GB 给激活与运行时，剩约 55 GB；每 token KV 56 KB，每请求约 168 MB，约 340 个并发（分页，块大小 16）。按最大长度预留则少一个数量级。见[分页 KV Cache 的练习](../engine/paged-kv.md#练习)。

**5. TTFT、TPOT、ITL、goodput 分别是什么？**

??? success "要点"
    TTFT：首 token 延迟（排队 + prefill）；TPOT：首 token 之后平均每个 token 的时间；ITL：相邻 token 的间隔（看卡顿）；goodput：每秒满足 SLO 的请求数。延迟看 P99。见[压测](../perf/benchmark.md#指标)。

**6. 为什么同一个请求在温度为 0 时，两次结果可能不同？**

??? success "要点"
    浮点加法不满足结合律，batch 组成、分块方式、kernel 选择改变累加顺序，logits 有微小差异，遇到接近平局的 token 时 argmax 翻转。需要严格一致时使用 batch 不变的 kernel。见[哪些优化会改变输出](llm://synthesis/token-journey/#哪些优化会改变输出)。

## 二、推理引擎

**7. ★ 从用户发出请求到收到第一个 token，推理引擎内部经过了哪些步骤？**

??? success "要点"
    HTTP 解析 → 对话模板与分词 → 发给引擎核心（跨进程）→ 前缀缓存查找、进入等待队列 → 调度器分配 KV 块和 token 预算 → 构造批次元数据 → 前向（可能分多步 prefill）→ 采样 → 反分词、停止条件检查 → 流式返回。TTFT = 排队 + prefill + CPU 开销。见[一个请求的一生](../engine/overview.md)。

**8. ★ 连续批处理是怎么实现的？**

??? success "要点"
    迭代级调度：每一步重新组批，结束的请求立即退出，新请求立即加入；统一的 token 预算让 prefill 分块与 decode 混在同一步；分页 KV 按需分配，不够时抢占。见[调度器](../engine/scheduler.md)。

**9. ★ PagedAttention 解决了什么问题？代价是什么？**

??? success "要点"
    KV Cache 按固定大小的块分配，消除预留浪费和外部碎片（浪费从 60%～80% 降到 4% 以内），支持块级共享（前缀缓存、并行采样）。代价：注意力 kernel 要按块表间接寻址，块内有少量内部碎片，块表和 slot mapping 等元数据的维护开销。见[分页 KV Cache](../engine/paged-kv.md)。

**10. vLLM V1 调度器"没有 prefill 和 decode 阶段"是什么意思？**

??? success "要点"
    每个请求只有 `num_computed_tokens` 与 `num_tokens`，调度器在 token 预算内让前者追上后者。prefill、分块 prefill、decode、抢占后的重算、前缀缓存命中、投机解码的草稿都是这一规则的特例。见[调度器](../engine/scheduler.md#统一的-token-预算)。

**11. `max_num_batched_tokens` 调大调小各有什么影响？**

??? success "要点"
    大：长提示词一步算完，TTFT 短，GPU 利用率高，但同批 decode 会遇到卡顿（ITL 尖峰）；小：decode 平稳，TTFT 变长。本书实测：预算 8192 时 ITL 最大 252 ms，512 时 16 ms。见[压测](../perf/benchmark.md#容量规划)。

**12. ★ 显存不足时 vLLM 和 SGLang 分别怎么处理？**

??? success "要点"
    vLLM：能分配就接收，不够时抢占运行队列末尾（或优先级最低）的请求，释放块并放回等待队列前面，之后重算（V1 去掉了 swap）；SGLang：按 `new_token_ratio` 预估未来需求保守接收，decode 时不够就 `retract_decode` 撤回部分请求并调高比例。见[调度器](../engine/scheduler.md#抢占vllm-与-sglang-的两种思路)。

**13. 为什么 vLLM 把 EngineCore 放在独立进程？**

??? success "要点"
    分词、反分词、HTTP、JSON 都是 CPU 密集的 Python 工作，与引擎主循环抢 GIL 会让 GPU 空等；拆成进程后真正并行。代价是跨进程通信，所以用 ZMQ + msgpack，并且调度结果只传增量。见[采样器与 API](../engine/sampler-api.md#为什么要多进程)。

**14. ★ CUDA Graph 为什么能加速 decode？有什么限制？**

??? success "要点"
    decode 一步上千次 kernel 启动，每次几微秒，发射开销常比 GPU 计算还长；CUDA Graph 录制整步、一次提交。限制：形状与地址固定（按 batch 大小分桶录制、填充、静态缓冲区），图内不能有 CPU 同步和依赖数据的分支，注意力元数据要在固定缓冲区里（因此有 piecewise 与 full 两种模式）。见 [CUDA Graphs 与 torch.compile](../engine/graphs-compile.md)。

**15. 异步调度（或 SGLang 的 overlap scheduler）解决什么问题？**

??? success "要点"
    调度与输入准备是 CPU 工作，与 GPU 计算串行时 GPU 会空等；异步调度在 GPU 执行第 N 步时就准备第 N+1 步，采样结果用占位 token 或 GPU 上的"未来 token"衔接。代价是结果处理晚一步，连续 prefill 或结构化输出需要同步时会关闭重叠。见 [SGLang 源码导读](../source/sglang.md#主线二scheduler-的事件循环)。

**16. 为什么推理引擎只对最后一个位置计算 logits？**

??? success "要点"
    生成只需要最后一个位置的分布，输出层（词表很大）只算需要采样的位置，prefill 中可省下可观的计算和 `[token 数, 词表]` 的显存。vLLM 用 `logits_indices = query_start_loc[1:] - 1`。见[变长批处理](../engine/batch-layout.md)。

**17. 采样为什么不用 `torch.multinomial`？**

??? success "要点"
    它会引入 CPU-GPU 同步。vLLM 用指数竞赛：`argmax(p / E)`，E 服从指数分布，等价于按 p 采样，而且每个请求可以用自己的生成器保证种子可复现。见[批量采样](../engine/sampler-api.md#批量采样)。

**18. 流式输出时如何处理停止字符串？**

??? success "要点"
    增量反分词得到新文本后检查是否包含停止字符串；为避免已经发出的内容里包含停止字符串的前半截，末尾扣住 `max(len(stop)) - 1` 个字符，确认后再发。还要处理被拆开的多字节字符。见[停止条件与增量反分词](../engine/sampler-api.md#停止条件与增量反分词)。

## 三、KV Cache 与前缀缓存

**19. ★ vLLM 与 SGLang 的前缀缓存有什么区别？**

??? success "要点"
    vLLM：链式哈希块（哈希包含父块哈希），空闲队列兼作 LRU，块被复用时才失效；SGLang：基数树，token 粒度，叶子 LRU，引用锁保护正在使用的节点，配合缓存感知调度（LPM）。命中率相近，差在工程取舍。见[前缀缓存](../engine/prefix-cache.md)。

**20. 为什么提示词完全命中缓存时还要重算最后一个 token？**

??? success "要点"
    需要一次前向得到下一个 token 的 logits；缓存里只有 KV，没有 logits。vLLM 最多命中 `num_tokens - 1` 个 token。

**21. 多租户共享前缀缓存有什么风险？**

??? success "要点"
    时间侧信道：通过 TTFT 判断某段内容是否在缓存里。用租户专属的 `cache_salt` 参与首块哈希，租户之间互不命中。多模态请求还要把图片哈希纳入块哈希，否则不同图片会错误共享。

**22. KV Cache 放不下怎么办？**

??? success "要点"
    减少 KV（GQA/MLA、KV 量化、滑动窗口、淘汰）→ 更好地管理（分页、前缀缓存、抢占）→ 扩展容量（CPU/SSD/分布式分层缓存，读回比重算快几十倍）→ 分摊（TP、上下文并行）。见 [KV 分层缓存](../distributed/kv-offload.md)。

**23. KV Cache FP8 量化需要注意什么？**

??? success "要点"
    K 比 V 敏感（K 有离群值），按张量缩放时小数值落入非规格化区，误差大；需要校准的缩放因子或按头的缩放；做长上下文评测。本书实测：只量化 V 几乎无损，只量化 K 困惑度 +12%，按头缩放 +2%。见[量化部署](../perf/quantization-deploy.md#kv-cache-量化k-比-v-敏感)。

**24. StreamingLLM 为什么要保留开头的几个 token？**

??? success "要点"
    注意力汇聚：模型把大量注意力放在最开头的 token 上，淘汰它们会让 softmax 分母骤减、分布扭曲。本书实测：只留最近 256 个 token 困惑度从 24 升到 190，多留 4 个开头 token 回到 29.6。见[长上下文](../topics/long-context.md)。

## 四、并行与分布式

**25. ★ 手推张量并行：MLP 和注意力怎么切？通信几次？**

??? success "要点"
    MLP：gate/up 按列切，SiLU×up 本地完成，down 按行切，一次 all-reduce；注意力：按头切 QKV，KV Cache 随头切，o_proj 按行切，一次 all-reduce。每层两次，通信量 = token 数 × hidden。见[张量并行](../distributed/tensor-parallel.md)。

**26. TP 大于 KV 头数时怎么办？**

??? success "要点"
    KV 头复制，多张卡存同一个 KV 头，KV 显存成倍浪费；另外 query 头数必须能被 TP 整除（Qwen2.5-7B 的 28 个头不能做 TP=8）。

**27. ★ decode 时 TP 的通信瓶颈在哪？怎么优化？**

??? success "要点"
    消息小（batch × hidden），由启动延迟主导（每层两次，80 层就是 160 次）。优化：custom all-reduce（NVLink P2P、one-shot/two-shot）、对称内存、all-reduce 与 RMSNorm 融合、减小 TP。prefill 时由带宽主导，可以计算通信重叠。

**28. ★ 为什么 DeepSeek 类模型用 DP Attention + EP，而不是 TP？**

??? success "要点"
    MLA 的潜在 KV 无法按头切分，TP 时每张卡都存完整 KV，浪费严重；DP Attention 各卡处理不同请求、各存各的 KV；MoE 部分用 EP，把全局 batch 汇集到专家上，提高专家计算效率。两者之间用 all-to-all。见[专家并行](../distributed/expert-parallel.md)。

**29. EP 的两次 all-to-all 分别传什么？通信量多大？**

??? success "要点"
    dispatch 把 token 隐藏状态发给专家所在的卡（常用 FP8），combine 把结果发回（BF16）；量级 = token 数 × top-k × hidden。跨机时每层可达数百微秒，需要限制跨节点路由、DeepEP、两批重叠。

**30. EPLB 做了什么？**

??? success "要点"
    统计专家负载，复制热门专家（冗余专家），再重新放置使各卡负载均衡。一层的时间由最慢的卡决定，本书模拟中最忙的卡从 2.26 倍平均负载降到 1.00 倍。

**31. DeepEP 的普通模式和低延迟模式分别用于什么？**

??? success "要点"
    普通模式：prefill，大批量高吞吐，同时利用 NVLink 与 RDMA；低延迟模式：decode，纯 RDMA、小消息低延迟、可被 CUDA Graph 录制。

**32. PP 能降低延迟吗？推理中怎么用 PP？**

??? success "要点"
    不能，单个 token 仍要依次经过所有 stage，还多了传输；PP 提高吞吐、扩展模型规模，通信只在 stage 边界点对点，适合跨机。需要多个批次同时在流水线中（vLLM 的批次队列）。见[流水线并行](../distributed/pp-cp.md)。

**33. ring attention 怎么合并分块的结果？为什么要之字形切分？**

??? success "要点"
    记录每行的 log-sum-exp，按 $O = \sum_i O_i e^{\text{LSE}_i - \text{LSE}}$ 合并；因果注意力下按连续区间切分，后面的卡计算量大得多，之字形切分让各卡负载相同（本书：136～904 变成全部 520）。

**34. ★ PD 分离的动机、流程和代价？**

??? success "要点"
    动机：prefill 与 decode 互相干扰、最优配置不同、TTFT/TPOT 解耦；流程：decode 预分配块 → prefill 计算并逐层推送 KV（RDMA）→ decode 接管；代价：KV 传输（每 token KV × 长度，可与计算重叠）、布局转换、xPyD 配比随负载调整、系统复杂度。见 [PD 分离](../distributed/pd-disagg.md)。

## 五、量化

**35. ★ W4A16 和 W8A8 分别加速了什么？怎么选？**

??? success "要点"
    W4A16 减少读权重的字节，加速访存受限的 decode，不加速计算，饱和容量几乎不变；W8A8（FP8/INT8）同时加速计算，容量翻倍。低并发延迟敏感选 W4A16，高吞吐选 FP8。本书模拟：W4A16 低负载 TPOT 从 5.2 降到 1.9 ms，容量 22.5 → 24.6 req/s；FP8 容量 46.5 req/s。见[量化部署](../perf/quantization-deploy.md#对容量的影响)。

**36. FP8 为什么用 E4M3？缩放粒度怎么选？**

??? success "要点"
    E4M3 精度更高，范围配合缩放够用；E5M2 多用于训练的梯度。粒度：按张量 < 按通道/按 token < 分块（128×128 权重、1×128 激活），本书实测困惑度损失 4.5% / 1.2% / 0.3%。

**37. MXFP4 与 NVFP4 的区别？**

??? success "要点"
    都是 E2M1 数值；MXFP4 每 32 个数一个 2 的幂次缩放（E8M0），NVFP4 每 16 个数一个 FP8 缩放加张量级缩放，精度更好（本书朴素量化：MXFP4 +27%，NVFP4 +7%）。Blackwell 原生支持。

**38. 为什么激活比权重难量化？SmoothQuant 做了什么？**

??? success "要点"
    激活有固定通道上的巨大离群值，且每次不同只能在线计算缩放；SmoothQuant 用数学等价变换把离群值的难度从激活转移到权重。见大模型手册的[量化原理](llm://inference/quantization/#激活量化与离群值)。

**39. 哪些层通常不量化？**

??? success "要点"
    嵌入层（查表，量化不加速）、lm_head（直接影响 logits 排序）、MoE 路由器（小误差可能改变专家选择），以及部分敏感的首尾层。

**40. 量化上线的评测流程？**

??? success "要点"
    困惑度快速检查 → 通用基准（lm-evaluation-harness：MMLU、GSM8K 等）→ 业务评测集；单独评测长上下文、代码、数学；与基线对比，准备回滚方案。

## 六、投机解码

**41. ★ 投机解码为什么能加速？为什么不改变输出？**

??? success "要点"
    decode 访存受限，验证 k 个 token 几乎和生成 1 个一样快；贪心时逐个比较 argmax，输出不变；采样时用拒绝采样（以 min(1, p/q) 接受，拒绝时从 norm(max(0, p−q)) 采样），分布严格不变。见大模型手册的[投机解码](llm://inference/serving/#投机解码)。

**42. 树形草稿怎么验证？**

??? success "要点"
    所有节点排成一个序列，用树注意力掩码让每个节点只看前缀、祖先和自己，位置取"前缀长度 + 深度"，一次前向；从根沿目标模型的预测走出最长匹配路径，路径上的 KV 直接复用。见[投机解码进阶](../topics/speculative.md)。

**43. EAGLE 和 MTP 是什么？**

??? success "要点"
    EAGLE：约一层 Transformer 的小草稿网络，输入目标模型的隐藏状态与下一个 token 的嵌入，自回归预测；EAGLE-3 融合多层特征。MTP：模型训练时自带的多 token 预测模块（DeepSeek-V3 等），推理时作为草稿。

**44. ★ 投机解码在什么情况下会变慢？**

??? success "要点"
    batch 大时 decode 接近计算受限，验证 k+1 个 token 的代价成倍增加；接受率低时白算。本书估算：batch 1 加速 2.8 倍，batch 128 为 1.3 倍，batch 256 为 0.9 倍。

**45. 投机解码对引擎有哪些改动？**

??? success "要点"
    调度器按草稿 token 分配预算与槽位，被拒绝的不推进 num_computed；验证批次每请求多个 query；拒绝采样在 GPU 上批量完成；草稿网络有自己的 KV 与 CUDA Graph；与结构化输出、前缀缓存、PD 分离的交互。

## 七、性能分析与调优

**46. ★ TPOT 比预期慢一倍，怎么排查？**

??? success "要点"
    与屋顶线下限对比 → 看服务端指标（batch、KV 使用率、抢占、排队）→ nsys 看时间线：kernel 之间有空隙是 CPU 开销（CUDA Graph、异步调度、去同步），GPU 一直忙则看 kernel 本身（ncu、带宽利用率、kernel 选择）→ 检查通信（TP 的 all-reduce）。见 [Profiling](../perf/profiling.md)。

**47. 为什么 `.item()` 会降低吞吐？**

??? success "要点"
    触发 CPU-GPU 同步：CPU 等 GPU 做完，GPU 再等 CPU 准备下一步，两者交替空闲。

**48. ★ 怎么评估一个服务能扛多少 QPS？**

??? success "要点"
    定 SLO → 构造接近真实的负载 → 开环（固定到达率）压测，画延迟–吞吐曲线 → 找满足 SLO 的最大速率（goodput 最大点）→ 按目标流量加冗余算卡数。注意吞吐最大时 goodput 往往已崩溃，闭环压测会掩盖过载。见[压测](../perf/benchmark.md)。

**49. 如何降低 TTFT？**

??? success "要点"
    排队：扩容、调度策略、限流；prefill：前缀缓存、FP8、分块、PD 分离（prefill 专用实例）、上下文并行；CPU 开销：分词进程化、多 API server。先确认是哪一段长。

**50. 如何降低 TPOT？**

??? success "要点"
    减小每步的读取量（权重量化、KV 量化、MLA/GQA）、减少干扰（分块 prefill 预算、PD 分离）、减少 CPU 开销（CUDA Graph、异步调度）、投机解码（低并发时）、TP（单请求延迟）。

## 八、专题

**51. 结构化输出怎么实现？性能瓶颈在哪？**

??? success "要点"
    文法 → 自动机，每步计算允许 token 的位掩码，屏蔽后采样。瓶颈是大词表下的掩码计算：xgrammar 预计算上下文无关 token 并缓存，引擎把掩码计算与 GPU 前向重叠；jump-forward 跳过确定的文本（注意分词一致性）。见[结构化输出](../topics/structured-output.md)。

**52. 工具调用在引擎里是怎么实现的？**

??? success "要点"
    对话模板渲染 tools → 模型按训练格式输出调用（如 `<tool_call>`）→ 工具解析器提取为 OpenAI 的 `tool_calls`（流式增量解析）→ 必要时用参数 Schema 做约束解码。推理模型还需要推理解析器分离思考内容。

**53. 多模态推理有什么特殊之处？**

??? success "要点"
    图像 token 数随分辨率线性增长（Qwen2.5-VL：像素 / 784）；视觉编码只在 prefill、可能占 prefill 一半，用 encoder cache 复用；M-RoPE 三维位置；前缀缓存必须把图片哈希纳入键；EPD 分离。见[多模态推理](../topics/multimodal.md)。

**54. ★ RL 训练对推理引擎有哪些特殊要求？**

??? success "要点"
    长尾 rollout（partial rollout、异步 RL、过量采样）；训练与推理概率不一致（BF16 下差异可达百分之几十，需要返回采样 logprobs、重要性采样修正、batch 不变 kernel）；显存切换（sleep/wake）；高效权重同步（IPC、NCCL、重新切分）。见 [RL 训练中的推理](../topics/rl-rollout.md)。

**55. 1M 上下文怎么支持？**

??? success "要点"
    模型结构（GQA/MLA、局部全局混合、线性注意力、原生稀疏注意力如 DSA）+ 系统（分块 prefill、上下文并行、KV 卸载、稀疏注意力 kernel）+ 有损近似（KV 淘汰）。

## 九、项目与开放题

**56. 介绍一个你做过的性能优化。**

??? success "要点"
    用"现象 → 假设 → 工具 → 证据 → 改动 → 效果"的结构，给出优化前后的数字和理论下限的对比。没有真实项目时，可以用本书的迷你引擎：例如"profiling 发现逐请求的注意力在 batch 64 时占 82%，改成批量 kernel 后……"。见[作品集](projects.md)。

**57. 如果让你从零设计一个推理服务平台，你会怎么做？**

??? success "要点"
    见[系统设计题](system-design.md)：需求与 SLO → 容量估算 → 单实例配置（引擎、并行、量化）→ 多实例（路由、缓存感知、PD 分离）→ 弹性与可靠性 → 监控与成本。

**58. vLLM 和 SGLang 你更熟悉哪个？它最近有什么重要变化？**

??? success "要点"
    要能讲出自己读过的具体模块（调度器、KV 管理、某个注意力后端），以及近期的方向：两者都在做 CPU/GPU 重叠、大规模 EP 与 PD 分离、混合架构（线性注意力、稀疏注意力）的缓存管理、RL 支持；vLLM 有 Model Runner V2、Rust 前端，SGLang 有统一的基数树缓存与分阶段的 CUDA Graph 配置。以源码为准，说明你关注的是最新代码。

**59. 你给开源推理框架贡献过代码吗？**

??? success "要点"
    有则讲清楚问题、方案、评审中的讨论和最终效果；没有也可以讲读源码时发现的问题、复现过的 issue。注意各项目对 AI 辅助贡献有明确规定（例如 vLLM 要求 PR 说明 AI 使用情况、不接受无实质内容的 PR），不要为了"刷"贡献提交琐碎改动。

**60. 未来一两年推理系统最重要的方向是什么？**

??? success "要点"
    开放题，言之有物即可：大规模 MoE 的 EP 与 PD 分离、以 KV 为中心的架构（分布式 KV 存储、跨实例复用）、长上下文（稀疏注意力、线性注意力的混合模型）、低精度（FP4 计算）、Agent 负载（多轮、工具调用、前缀复用）、RL 推理一体化、推理成本优化。选一两个深入讲，比罗列更好。

## 小结

- [x] 估算题要能口算：权重/KV 字节与带宽、KV per token、并发数。
- [x] 引擎题围绕请求生命周期、调度、KV 管理、CUDA Graph、多进程与重叠。
- [x] 分布式题围绕通信模式：TP 的 all-reduce、EP 的 all-to-all、PP 的点对点、CP 的 LSE 合并、PD 的 KV 传输。
- [x] 量化、投机解码都要说清"加速了什么、什么时候失效"，并给出数字。
