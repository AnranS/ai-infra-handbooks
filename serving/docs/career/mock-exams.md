# 模拟面试套卷：四套按真实结构组好的题

<p class="lead"><a href="../mock-interview/">模拟面试</a>一章给了面试的结构、评分表和复盘模板；这一章按同样的结构组好四套题：前三套分别偏向推理框架、推理优化、推理平台三类岗位，第四套偏向大规模 MoE 与 RL 推理。每套 60 分钟：编码 20 分钟、基础追问 15 分钟、系统设计 15 分钟、项目深挖 10 分钟。追问题都给了要点和对应章节，先自己口述一遍、录音，再展开核对。</p>

!!! tip "怎么用"
    请同事或学习搭子按下面的题目提问，严格计时；追问要一层层往下问，直到你答不上来为止——那个位置就是需要回去补的地方。每套题各做一次，间隔一周；每次结束后用评分表打分、写复盘。也可以把本页的题目和评分表交给大模型，让它扮演面试官（提示词见模拟面试一章）。

## 第一套：推理框架方向

**编码（20 分钟）**：手写一个连续批处理调度器的核心循环——维护等待队列和运行队列，每一步在 token 预算内先安排 decode、再用剩余预算做分块 prefill，KV 不足时抢占最后到达的请求。练习题：[连续批处理调度器](root://practice/#/p/sv-scheduler)。备选：[LRU 缓存](root://practice/#/p/py-lru-cache)。

**基础追问（15 分钟）**

**1. 为什么需要分页的 KV Cache？** → 块大小怎么选？→ 两个请求共享前缀时，块表和引用计数怎样变化？

??? success "要点"
    连续分配要按最大长度预留，内部碎片和外部碎片让实际利用率很低；分页后按需分配固定大小的块，用块表把逻辑位置映射到物理块，碎片只剩最后一块的一部分。块越大元数据越少、kernel 访存越连续，但内部碎片越多、前缀共享的粒度越粗；传输时块太小会被网卡的消息速率卡住（[RDMA](../comm/rdma.md#请求数与消息速率)）。共享前缀：两个请求的块表指向同一批物理块，引用计数加一；某个请求要写入共享块时写时复制。见[分页 KV Cache](../engine/paged-kv.md)、[前缀缓存](../engine/prefix-cache.md)。

**2. 分块 prefill 解决了什么问题？** → token 预算设多大？→ 它和 PD 分离是什么关系？

??? success "要点"
    长提示词的 prefill 会独占一步、让正在 decode 的请求等待，TPOT 出现尖刺；分块 prefill 把它切成若干块，每一步和 decode 混在一起，限制每一步的总 token 数。预算越大，prefill 越快（TTFT 低）但每一步越慢（ITL 高），要按 SLO 压测选择，常见在 2K～8K。PD 分离是更彻底的解法：prefill 和 decode 在不同实例上，互不干扰，代价是 KV 传输和配比问题；干扰不严重时分块 prefill 就够了。见[调度器](../engine/scheduler.md)、[全局调度](../frontier/disagg-sched.md#xpyd-配比跟着负载变)。

**3. decode 为什么要用 CUDA Graph？** → batch 大小每步都在变，怎么处理？→ 哪些东西不能录进图里？

??? success "要点"
    decode 每一步有几百个小 kernel，CPU 逐个提交的开销可能和 GPU 计算时间相当；CUDA Graph 把整步录制下来一次提交。batch 大小按档位（1、2、4、8…）分别录制，运行时补齐到最近的档位；图里的地址是固定的，输入要拷进预先分配的缓冲区。不能录的：依赖 CPU 同步的操作（比如需要把数量拷回 CPU 的 all-to-all）、动态形状、会改变内存分配的操作——这也是 DeepEP 低延迟模式用固定槽位的原因。见 [CUDA Graphs 与 torch.compile](../engine/graphs-compile.md)。

**系统设计（15 分钟）**：[全局调度与缓存感知的推理网关](design-answers-1.md#3-全局调度与缓存感知的推理网关)。

**项目深挖（10 分钟）**：讲你的推理引擎（手写 mini-sglang 或作品 A）：一个请求从 HTTP 到最后一个 token 经过了哪些模块？你做过的最有效的一次性能优化是什么，数字是多少？有没有失败的尝试？

## 第二套：推理优化方向

**编码（20 分钟）**：写一个 online softmax 的 CUDA 或 Triton kernel（一行一个线程块，一遍扫描同时维护最大值和和），并说明它为什么是 FlashAttention 的基础。练习题：[online softmax](root://practice/#/p/cu-online-softmax)。备选：[分块 GEMM](root://practice/#/p/cu-gemm-tiled)、[FlashAttention 前向](root://practice/#/p/cu-flash-attn)。

**基础追问（15 分钟）**

**1. decode 为什么是访存瓶颈？** → batch 增大到多少会变成计算瓶颈？→ MLA 的 decode 注意力为什么不一样？

??? success "要点"
    decode 每个请求每步只有一个 token，线性层是矩阵 × 向量，每读一个权重只做 2 次运算；batch 为 $b$ 时算术强度约为 $b$（bf16 权重），H100 的屋脊点约 295，所以 batch 要到几百才接近计算瓶颈——但 KV 的读取随 batch 线性增长，注意力部分一直是访存瓶颈。MLA 吸收之后 128 个头共享一份 576 维的潜向量，每字节约 240 次运算，接近计算瓶颈，所以要专门的 kernel（FlashMLA）。见 [MLA 推理](../moe/mla.md#decode-注意力变成了计算问题)。

**2. DeepSeek-V3 的 FP8 为什么用分块缩放？** → 缩放因子在 GEMM 的哪一步乘进去？→ MoE 的分组 GEMM 有什么特别？

??? success "要点"
    E4M3 只有 3 位尾数，极端离群值会让逐张量缩放把正常值压进非规格化数；激活 1×128、权重 128×128 分块，离群值只影响自己那一组。GEMM 沿 K 每 128 个元素做一次 FP8 矩阵乘，部分和提升到 fp32 时乘上两个缩放，顺带解决了 Tensor Core 累加精度只有约 14 位的问题。MoE：每个专家分到的 token 数不同，连续布局（补齐到块大小，用于 prefill）或带掩码的固定布局（配合 DeepEP 低延迟模式、可录 CUDA Graph，用于 decode）；decode 时每个专家要分到约 300 个 token 才能摆脱权重读取的限制。见 [FP8 与分组 GEMM](../moe/fp8-gemm.md)。

**3. vLLM 为什么要自己实现 all-reduce？** → one-shot 和 two-shot 的区别？→ 新硬件上怎么变？

??? success "要点"
    decode 的 TP all-reduce 只有几百 KB，NCCL 的 ring 要 $2(n-1)$ 步，时间几乎全是固定开销；定制实现用 CUDA IPC 直接读对端显存。one-shot 每卡读其他卡的完整数据、一次同步，适合小消息；two-shot 先做 reduce-scatter 再 all-gather，两次同步、通信量更小，适合中等消息；大消息交给 NCCL。H100 的 NVSwitch 支持 NVLS（交换机内归约），通信量减半、不占 SM。见[集合通信](../comm/nccl.md)。

**系统设计（15 分钟）**：[百万 token 长上下文服务](design-answers-2.md#6-百万-token-长上下文服务)。

**项目深挖（10 分钟）**：讲你写过的最快的 kernel（作品 C）：达到了峰值的多少？用 Nsight Compute 看到剩下的差距在哪里？如果再给你一周，下一步优化什么？

## 第三套：推理平台方向

**编码（20 分钟）**：用 asyncio 写一个动态批处理器：请求到达后等待最多 $t$ 毫秒或凑够 $b$ 个再一起处理，处理完把结果分别返回给各自的调用者，并正确处理超时和异常。练习题：[动态批处理器](root://practice/#/p/py-dynamic-batcher)。备选：[按预计 TTFT 路由与提前拒绝](root://practice/#/p/sv-ttft-router)。

**基础追问（15 分钟）**

**1. 什么时候该做 PD 分离？** → KV 传输要多久？→ prefill 和 decode 实例的比例怎么定？

??? success "要点"
    长提示词的 prefill 严重干扰 decode 的 TPOT、或者两个阶段想用不同的并行方式和硬件时才值得；小规模、提示词短时分块 prefill 就够了。传输量 = 提示词长度 × 每 token 的 KV（70B GQA FP8 约 160 KB/token，4K 提示词约 0.65 GB，400G 网卡十几毫秒），可以逐层推送和计算重叠。配比按"prefill 需要的 token 算力 / decode 需要的输出吞吐"算，负载形态一变瓶颈就会在两侧跳动，要规划器按分钟调整。见 [PD 分离](../distributed/pd-disagg.md)、[全局调度](../frontier/disagg-sched.md)。

**2. 缓存感知路由会带来什么问题？** → 怎样同时考虑缓存和负载？→ 路由器怎么知道实例缓存了什么？

??? success "要点"
    只看缓存会把同一类请求压在同一个实例上形成热点；用"预计 TTFT = 排队时间 + 未命中部分的 prefill 时间"统一两者，过载时按预测提前拒绝（模拟中高负载下达标率从 86% 升到 98%）。索引靠实例发布的 KV 事件（存入、淘汰）维护一棵全局前缀树，是近似的，路由错了只多算不算错。见[全局调度](../frontier/disagg-sched.md#路由缓存与排队的权衡)。

**3. 自动扩缩容看什么指标？** → 冷启动要多久、怎么加速？→ 缩容时要注意什么？

??? success "要点"
    看排队长度、KV 使用率和 TTFT 的趋势，不看 GPU 利用率（decode 天然不高）；按日内曲线提前扩容，突发靠热备（冷启动期间流量还在涨，按 Little 定律和涨速算热备数）。冷启动 = 拉权重（对象存储上百秒、本地 NVMe 二十秒、从已运行实例 RDMA 拉取不到一秒）+ 初始化 + CUDA Graph 录制与编译（可以缓存）。缩容要等请求跑完、避免抖动，长请求多时缩容要更慢。见[系统设计参考答案（二）](design-answers-2.md#9-推理可观测性与自动扩缩容)。

**系统设计（15 分钟）**：[以 KV Cache 为中心的多级缓存池](design-answers-1.md#2-以-kv-cache-为中心的多级缓存池)。

**项目深挖（10 分钟）**：讲你的推理网关（作品 B）：压测中缓存感知路由相对轮询改善了多少？一个实例故障时发生了什么，恢复用了多久？设计文档里最难的一个取舍是什么？

## 第四套：大规模 MoE 与 RL 推理方向

**编码（20 分钟）**：实现专家并行的分发与合并：每个 rank 按目标专家把 token 排好、算出发往每个 rank 的数量，all-to-all 之后各专家计算，再按门控权重合并回原来的顺序。练习题：[专家并行的 all-to-all 分发与合并](root://practice/#/p/sv-ep-dispatch)。备选：[EP 通信的账本](root://practice/#/p/sv-deepep-layout)、[MLA 的吸收路径](root://practice/#/p/sv-mla-absorb)。

**基础追问（15 分钟）**

**1. MLA 的 decode 为什么走"吸收"，prefill 为什么走"展开"？** → MLA 的 KV 用 FP8 存之后，decode 注意力的瓶颈变成了什么？→ MLA 模型用张量并行部署有什么问题？

??? success "要点"
    吸收把 $W_{UK}$ 并进 query、$W_{UV}$ 并进输出投影，所有头共享 576 维的潜向量"键"，每对 (query, key) 贵约 3.4 倍，但不用把缓存展开；decode 新 token 少、缓存多，展开的代价远大于点积变贵，prefill 反之。FP8 的 KV 每个缓存 token 只读 576 字节，算术强度约 484 FLOP/字节，超过 H800 BF16 的屋脊点，decode 注意力从访存受限变成计算受限，kernel 的重点转向 Tensor Core 利用率（FlashMLA）。张量并行切不开潜向量——它是所有头共用的，每张卡都要存完整的一份，KV 容量不随卡数增加，所以 MLA 模型通常用 DP 注意力 + 专家并行。见 [MLA 推理](../moe/mla.md)、[专家并行与 DP Attention](../distributed/expert-parallel.md)。

**2. 大规模 EP 的 decode，一步的时间花在哪里？** → 双 batch 重叠什么时候有用？→ 专家的负载不均怎么处理？

??? success "要点"
    每层三块：注意力（读权重和 KV）、专家 GEMM（读专家权重）、两次 all-to-all（dispatch 用 FP8、combine 用 BF16），decode 时通信常常和计算一样长甚至更长。双 batch 重叠把 batch 拆成两半交替计算和通信，每卡 batch 64～128、计算与通信相当时快 1.45～1.65 倍；batch 很小时拆开后要多读一遍权重，几乎没用。负载：EPLB 按统计的专家负载复制热门专家、重新摆放（prefill 分层、decode 全局）；DP 注意力按 KV 总量分配请求，最慢的 rank 决定整步的速度。见[大规模 EP 部署](../moe/ep-deploy.md)。

**3. RL 训练对推理引擎有哪些特殊要求？** → 全异步 RL 的陈旧度有什么偏差？→ MoE 模型的训练端为什么会和推理端"走不同的专家"？

??? success "要点"
    四件事：长尾（partial rollout、异步 RL、过量采样）、概率不一致（重要性修正、batch 不变 kernel）、显存切换（sleep / wake）、权重同步（共置走 IPC，分离部署按切分映射点对点发送、实例间流水接力，几秒内完成且与实例数无关）。全异步的陈旧度集中在长回答上，设上限时被丢的也是长回答，模型可能偏向短回答，需要重要性加权、按长度组批、逐 token 记录 logprob。MoE：两边数值的微小差异会让 top-k 路由换专家，变成完全不同的计算路径；路由重放记录推理时每层选中的专家，训练端直接使用。见 [RL 训练中的推理](../topics/rl-rollout.md)、[RL 推理系统](../frontier/rl-async.md)。

**系统设计（15 分钟）**：[600B 级 MoE 模型的在线推理服务](design-answers-1.md#1-600b-级-moe-模型的在线推理服务)。备选：[RL rollout 系统](design-answers-1.md#5-rl-rollout-系统)。

**项目深挖（10 分钟）**：讲你接入混合架构模型的大作业（[大作业四](minisgl://wrap/assignment-hybrid/)）或者手写引擎里的 MoE 部分：精度怎么对齐的，第一个不一致出现在哪一层、哪个位置？每个请求的状态多大，短请求时为什么反而更占显存？前缀缓存和投机解码要怎么改？

## 评分与复盘

用[模拟面试](mock-interview.md#评分表)一章的评分表，四项（正确性、深度与数字、表达结构、取舍与权衡）各 1～5 分。几套题做完后对比：哪一类问题总在同一层追问上卡住，就回到对应章节重读，并把卡住的问题写进自己的题卡。

## 小结

- [x] 四套题分别偏向推理框架（调度、分页 KV、CUDA Graph）、推理优化（屋顶线、FP8、通信 kernel）、推理平台（PD 分离、路由、扩缩容）、大规模 MoE 与 RL 推理（MLA、EP、异步 RL）。
- [x] 每套按真实结构计时：编码 20 分钟、追问 15 分钟、系统设计 15 分钟、项目 10 分钟；追问一层层往下问，找到答不上来的那一层。
- [x] 复盘时按评分表打分，把卡住的问题写进题卡，回到对应章节补齐。
