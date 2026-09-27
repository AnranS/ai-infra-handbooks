# 自测题库

<p class="lead">学完一遍后，用这些题检验自己是否真正"融会贯通"。题目按章节组织，最后一部分是跨章节的综合题，也是推理方向面试中最常见的题型。先自己回答（最好说出来或写下来），再展开参考答案。答不上来的题，回到对应章节重读。</p>

!!! tip "怎么用这个题库"
    - 第一遍：每题限时 2 分钟，只标记"能答 / 模糊 / 不会"；
    - 第二遍：只看"模糊"和"不会"的题，回到链接的章节重读，然后**不看答案**重新回答；
    - 面试前：挑综合题，练习用 3～5 分钟完整地讲清楚，包括数字估算。

## 基础

**1. 语言模型在推理时的输入和输出是什么？为什么生成只能一个 token 一个 token 地进行？**

??? success "参考答案"
    输入是 token 序列，输出是**每个位置**上下一个 token 的概率分布（logits 经 softmax）。生成时第 t+1 个 token 要以第 t 个 token 为条件，而第 t 个 token 要等上一步的采样结果，所以只能串行。训练时所有位置的目标都已知，可以并行计算（teacher forcing）。见[语言模型](../basics/language-model.md#训练可以并行推理只能串行)。

**2. 交叉熵损失 2.166 对应的困惑度是多少？困惑度的直观含义是什么？**

??? success "参考答案"
    $e^{2.166} \approx 8.72$。直观含义：模型平均而言相当于在约 8.7 个候选中均匀地猜。均匀分布在整个词表上的困惑度等于词表大小（Qwen 为 151936）。见[训练目标：交叉熵](../basics/language-model.md#训练目标交叉熵)。

**3. softmax 为什么要减去最大值？在注意力中，这个技巧如何演变成 online softmax？**

??? success "参考答案"
    $e^x$ 在 x 较大时溢出，减去最大值后所有指数都 ≤ 1，结果不变。online softmax 在分块处理时维护"当前最大值"和"当前分母"，遇到更大的值就把之前的累加结果乘以 $e^{m_{old} - m_{new}}$ 校正。这是 FlashAttention 不物化整个 scores 矩阵的关键。见[softmax 与数值稳定](../basics/math-torch.md#softmax-与数值稳定)。

**4. BF16 和 FP16 的区别？为什么大模型训练和推理多用 BF16？**

??? success "参考答案"
    两者都是 16 位。BF16 有 8 位指数（与 FP32 相同）、7 位尾数；FP16 有 5 位指数、10 位尾数。BF16 的表示范围与 FP32 相同，不容易溢出，不需要 loss scaling；代价是精度较低。大模型的激活中有很大的离群值，范围比精度更重要。见[数值格式](../basics/math-torch.md#数值格式)。

**5. BPE 分词是怎么训练的？字节级 BPE 有什么好处？**

??? success "参考答案"
    从单个字节（或字符）开始，反复统计相邻 token 对的频率，把最频繁的一对合并成新 token，直到达到目标词表大小。编码时按学到的合并顺序应用合并规则。字节级 BPE 的基础词表是 256 个字节，任何文本（任何语言、表情、乱码）都能编码，不会出现未知 token。见 [BPE](../basics/tokenization.md#bpe从字节开始不断合并)。

**6. 为什么流式输出时不能每生成一个 token 就单独 decode 它？**

??? success "参考答案"
    一个汉字在 UTF-8 中通常占 3 个字节，字节级 BPE 可能把它拆到两个 token 中，单独 decode 一个 token 会得到不完整的字节（显示为 �）。正确做法是增量反分词：记录已输出的文本，每次 decode 一个窗口，只输出新增且完整的部分。见[流式输出与增量反分词](../basics/tokenization.md#流式输出与增量反分词)。

**7. 用错了对话模板会怎样？**

??? success "参考答案"
    模型在后训练时见到的都是特定格式的对话（特殊 token、角色标记），格式不对时，模型会把输入当作普通文本续写，表现明显下降：比如续写用户的话、不停止、答非所问。推理引擎要使用模型自带的 chat template，并正确设置停止 token。见[特殊 token 与对话模板](../basics/tokenization.md#特殊-token-与对话模板)。

## Transformer

**8. 写出缩放点积注意力的公式。为什么要除以 $\sqrt{d_k}$？**

??? success "参考答案"
    $\text{softmax}(QK^\top / \sqrt{d_k}) V$。如果 q、k 的各分量独立、方差为 1，点积的方差为 $d_k$，标准差为 $\sqrt{d_k}$。不缩放的话，分数的量级随维度增长，softmax 趋于 one-hot，梯度很小。见[缩放点积注意力](../transformer/attention.md#缩放点积注意力)。

**9. 有 KV Cache 时，新 token 的因果掩码怎么写？**

??? success "参考答案"
    query 有 T 个（新 token），key 有 S 个（历史 + 新 token）。第 i 个新 token 的绝对位置是 S − T + i，它能看到位置 ≤ S − T + i 的 key，即 `ones(T, S).tril(diagonal=S - T)`。decode 时 T = 1，不需要掩码。见[从零组装一个大模型](../transformer/build-llm.md#完整代码)中的实现。

**10. RoPE 的核心思想是什么？为什么说它编码的是相对位置？**

??? success "参考答案"
    把 q、k 的每两维看作一个二维向量，按"位置 × 频率"的角度旋转。位置 m 的 q 与位置 n 的 k 做点积时，旋转矩阵合并为 $R_{n-m}$，结果只依赖相对距离 n − m。它只作用于 q、k，不作用于 v；在推理中，缓存的 K 是已经旋转过的。见[RoPE](../transformer/position.md#rope用旋转编码位置)。

**11. 注意力汇聚（attention sink）是什么？它对推理有什么影响？**

??? success "参考答案"
    很多头会把大量注意力放在第一个 token 上（本手册在 Qwen 上测到中间层 35%～52%），第一个 token 相当于一个"什么都不看"时的垃圾桶。影响：StreamingLLM 等 KV 淘汰方法必须保留最初几个 token；这些 token 伴随巨大的激活值，给激活量化带来困难；gpt-oss 干脆为每个头加了一个可学习的汇聚项。见[注意力汇聚](../transformer/attention.md#真实模型里的注意力注意力汇聚)。

**12. Pre-Norm 和 Post-Norm 的区别？为什么现在都用 Pre-Norm？**

??? success "参考答案"
    Post-Norm 在残差相加之后归一化，Pre-Norm 在子层之前归一化、残差通路保持恒等。Pre-Norm 的梯度可以沿残差通路直接回传，深层模型训练更稳定，不需要精细的预热。代价是残差流的数值随深度增长，最后需要一个 final norm。见[Pre-Norm 与 Post-Norm](../transformer/norm-residual.md#pre-norm-与-post-norm)。

**13. SwiGLU 相比普通 MLP 多了什么？参数量怎么保持相当？**

??? success "参考答案"
    多了一个门控投影：$\text{down}(\text{SiLU}(\text{gate}(x)) \odot \text{up}(x))$，三个矩阵而不是两个。为了保持参数量相当，中间维度从 4d 缩到约 $\frac{8}{3}d$（LLaMA 取整到 256 的倍数）。推理中 gate 和 up 合并成一次 GEMM，SiLU 与乘法融合成一个 kernel。见[门控：SwiGLU](../transformer/ffn.md#门控swiglu)。

**14. 一个稠密模型中，参数主要在哪里？**

??? success "参考答案"
    每层约 $4d^2$（注意力，GQA 时更少）+ $3 d \cdot d_{ff}$（FFN，通常约为注意力的 2～3 倍），加上词表 × d 的嵌入和输出层。大模型中 FFN 占约三分之二；小模型中词表的占比很大（Qwen2.5-0.5B 的嵌入占 27.6%）。见[各部分的参数量](../transformer/build-llm.md#各部分的参数量)。

**15. MQA、GQA、MLA 分别怎样减少 KV Cache？MLA 的代价是什么？**

??? success "参考答案"
    MQA：所有 query 头共享一组 K、V；GQA：每组 query 头共享一组 K、V（折中）；MLA：把 K、V 压缩成一个低维潜在向量来缓存，用时再通过矩阵"解压"，推理时可以把解压矩阵吸收进 query 和输出投影中。MLA 的代价：RoPE 与低秩压缩不兼容，需要单独的解耦 RoPE 维度；注意力 kernel 更复杂；潜在 KV 无法按头切分，不适合张量并行。见[注意力变体](../transformer/attention-variants.md)。

**16. MoE 的路由过程是什么？总参数和激活参数的区别对推理意味着什么？**

??? success "参考答案"
    路由器为每个 token 计算各专家的分数，选 top-k 个，用（归一化后的）分数加权它们的输出。显存要装下全部专家（总参数），但每个 token 只计算 k 个专家（激活参数）。所以 MoE 的显存需求大、单 token 计算量小；decode 时 batch 中的 token 被分散到各专家，需要更大的 batch 才能摊薄权重读取。见[MoE](../transformer/moe.md)。

## 训练与对齐

**17. 训练一个 7B 模型、2T token，大约需要多少算力？1000 张 H100、MFU 40% 需要多久？**

??? success "参考答案"
    $6ND = 6 \times 7 \times 10^9 \times 2 \times 10^{12} = 8.4 \times 10^{22}$ FLOPs。1000 × 989 TFLOPS × 0.4 ≈ $4 \times 10^{17}$ FLOP/s，约 $2.1 \times 10^5$ 秒，约 59 小时。见[训练需要多少算力](../training/pretraining.md#训练需要多少算力)。

**18. Chinchilla 定律说了什么？为什么今天的小模型都"过度训练"？**

??? success "参考答案"
    给定训练算力，最优的 token 数约为参数量的 20 倍。但 Chinchilla 只优化训练成本；模型部署后的推理成本与参数量成正比、与训练 token 数无关，所以值得用远超 20 倍的数据训练较小的模型（LLaMA 3 8B 用了 15T token，约 1900 倍）。见[Scaling Law](../training/pretraining.md#scaling-law)。

**19. SFT 的损失和预训练有什么不同？**

??? success "参考答案"
    损失函数相同（下一个 token 的交叉熵），但只在回答部分计算，系统提示和用户输入的位置被掩掉（标签设为 −100）。数据是对话格式，使用对话模板。见[SFT](../training/post-training.md#sft监督微调)。

**20. DPO 相比 RLHF（PPO）省掉了什么？GRPO 又省掉了什么？**

??? success "参考答案"
    DPO 省掉了奖励模型和强化学习的采样循环，直接用偏好对（好回答、差回答）和一个冻结的参考模型构造分类式的损失。GRPO 仍然是在线强化学习，但省掉了价值网络（critic）：对同一问题采样一组回答，用组内奖励的均值和标准差做归一化，作为优势。推理模型（R1 类）主要用 GRPO + 可验证奖励训练。见[后训练](../training/post-training.md)。

**21. LoRA 的原理是什么？多 LoRA 服务是怎么做到的？**

??? success "参考答案"
    冻结原权重 W，只训练低秩增量 $BA$（r 远小于 d），输出为 $Wx + BAx$。推理时既可以把 BA 合并进 W，也可以保持分离：多 LoRA 服务让所有请求共享基座模型的一次 GEMM，再为每个请求按其 adapter 计算低秩部分（Punica、S-LoRA 的分段 GEMM kernel），一个服务就能同时服务成百上千个微调版本。见[LoRA](../training/post-training.md#lora低秩微调)。

## 推理原理

**22. 温度、top-k、top-p、min-p 分别怎么作用于 logits？执行顺序重要吗？**

??? success "参考答案"
    温度把 logits 除以 T；top-k 只保留最大的 k 个；top-p 保留累计概率达到 p 的最小集合；min-p 保留概率不低于"最大概率 × p"的 token。顺序会影响结果（比如先温度再 top-p，温度会改变累计概率），各框架的顺序不完全一样，这也是"同样的参数、不同框架结果不同"的原因之一。见[解码与采样](../inference/decoding.md)。

**23. 为什么 HF generate 的"贪心"结果可能和你自己写的 argmax 不一样？**

??? success "参考答案"
    模型自带的 `generation_config.json` 可能设置了默认的重复惩罚、温度、top-p、top-k（Qwen2.5 就设置了 repetition_penalty 1.1 等），这些默认值会被自动应用。要做严格对比，需要显式关掉它们。见[模型自带的默认参数](../inference/decoding.md#模型自带的默认参数)。

**24. 为什么 K、V 可以缓存，Q 不需要？**

??? success "参考答案"
    因果掩码保证了历史 token 的 K、V 只依赖于它们自己及之前的 token，新 token 的到来不会改变它们；而 Q 只在当前 token 计算注意力时使用一次，之后再也用不到。见[为什么 K、V 可以缓存](../inference/kv-cache.md#为什么-kv-可以缓存)。

**25. prefill 和 decode 分别是计算受限还是访存受限？为什么？**

??? success "参考答案"
    prefill 一次处理成百上千个 token，每读一次权重做很多次运算，算术强度高，计算受限；decode 每个请求每步只处理 1 个 token，每读 2 字节的权重只做约 2 次运算，算术强度约为 batch 大小，远低于 GPU 的屋脊点（H100 约 295），访存受限。见[prefill 与 decode](../inference/kv-cache.md#prefill-与-decode)。

**26. LLaMA-3-70B（BF16）在 4 张 H100 上做张量并行，batch = 1 时 decode 每个 token 的时间下限是多少？每个 token 的 KV Cache 多大？**

??? success "参考答案"
    权重 70.55B × 2 字节 ≈ 141 GB，4 张卡的总带宽 4 × 3.35 TB/s = 13.4 TB/s，下限约 10.5 ms/token（再加上通信和其他开销）。KV Cache：2 × 80 层 × 8 个 KV 头 × 128 × 2 字节 = 320 KB/token。见[参数量、算力与显存估算](../inference/estimation.md)。

**27. 为什么 INT4 weight-only 量化能让 decode 快约 3～4 倍，却几乎不能加速 prefill？**

??? success "参考答案"
    decode 访存受限，时间约等于读取权重的时间，权重字节数变为约 1/4，时间也接近 1/4。prefill 计算受限，weight-only 量化在计算前要把权重反量化回 BF16，计算量不变，反而多了反量化开销。要加速 prefill，需要 W8A8（FP8/INT8）这类激活也量化的方案，用低精度 Tensor Core 计算。见[量化原理](../inference/quantization.md)。

**28. 为什么按组量化（group size 128）比按通道量化的精度好得多？为什么激活比权重难量化？**

??? success "参考答案"
    同一个缩放因子覆盖的数越少，受离群值的影响越小（本手册在 Qwen 上测到 W4 按通道困惑度 50.35，按组 26.98）。激活难量化，是因为少数通道存在巨大的离群值（本手册测到约 1650），而且激活每次都不同，只能在线计算缩放因子。SmoothQuant 把激活的离群值"挪"到权重上，AWQ 根据激活的大小保护重要的权重通道。见[量化原理](../inference/quantization.md#激活量化与离群值)。

## 推理服务

**29. TTFT、TPOT、吞吐量分别由什么决定？它们怎样相互制约？**

??? success "参考答案"
    TTFT = 排队时间 + prefill 时间；TPOT 由每步 decode 的时间决定，随 batch 和上下文长度增加；吞吐量随 batch 增大而提高。增大 batch 提高吞吐但拖慢 TPOT；新请求的 prefill 插入会让正在 decode 的请求卡顿。目标是在满足 SLO 的前提下最大化 goodput。见[指标](../inference/serving.md#指标)。

**30. 连续批处理解决了什么问题？它对注意力 kernel 提出了什么要求？**

??? success "参考答案"
    静态批处理中，短请求结束后要等同批的长请求，GPU 空转。连续批处理逐步调度，结束即离开、新请求立即加入。要求：同一个 batch 中的请求上下文长度不同、KV 存放位置不同，甚至处于不同阶段，所以注意力 kernel 要支持变长序列和分页 KV（块表），输入布局是把所有 token 拼成一维而不是填充成矩形。见[连续批处理](../inference/serving.md#连续批处理)。

**31. PagedAttention 借鉴了操作系统的什么思想？它带来了哪些好处？**

??? success "参考答案"
    虚拟内存与分页：KV Cache 被切成固定大小的块，每个请求用块表把逻辑块映射到物理块。好处：几乎没有碎片和预留浪费，并发数显著提高；相同前缀可以共享物理块（前缀缓存、并行采样、束搜索）；抢占时可以按块换出。见[KV Cache 的显存管理](../inference/kv-cache.md#kv-cache-的显存管理)。

**32. 投机解码为什么能加速？为什么它不改变输出分布？什么情况下收益小？**

??? success "参考答案"
    decode 访存受限，验证 k 个 token 和生成 1 个 token 读的权重一样多，猜中就相当于一次前向前进多个 token。贪心时逐个比较 argmax，输出与普通贪心完全相同；采样时用拒绝采样（以 $\min(1, p/q)$ 接受，拒绝则从 $\max(0, p-q)$ 归一化后采样），输出分布严格等于大模型的分布。收益小的情况：草稿质量差（接受率低）、batch 大（已接近计算受限，验证不再"免费"）。见[投机解码](../inference/serving.md#投机解码)。

**33. 为什么要做 PD 分离？它的代价是什么？**

??? success "参考答案"
    prefill 计算密集、decode 访存密集，混在一起互相干扰（TTFT 与 TPOT 互相拖累），也无法分别选择最优的并行策略和 batch 大小。分离后可以分别优化和扩缩容。代价：KV Cache 需要跨机传输（需要 RDMA 等高速网络）、系统更复杂、资源配比需要随负载调整。见[PD 分离](../inference/serving.md#pd-分离)。

## 综合题

**34. 从用户按下回车到看到第一个字，中间发生了什么？请尽量完整地讲一遍。**

??? success "参考答案"
    HTTP 请求到达 API 服务 → 应用对话模板、分词 → 进入调度器的等待队列 → 调度器查找前缀缓存，为未命中的部分分配 KV 块，把它（可能分块）加入某一步的 batch → 模型执行：嵌入 → 每层（RMSNorm → QKV 投影 → RoPE → 写入分页 KV → FlashAttention → O 投影 → 残差 → RMSNorm → SwiGLU MLP → 残差），张量并行时每层有 all-reduce → final norm → 只对最后一个位置做输出层 → 采样（温度、top-p 等）→ 增量反分词 → 流式返回第一个 token。TTFT = 排队 + prefill（+ 网络和分词）。见[一个 token 的完整旅程](token-journey.md)。

**35. 服务的 batch 从 1 增大到 64，decode 每步的时间怎么变？瓶颈从哪里转移到了哪里？**

??? success "参考答案"
    权重类算子的强度约等于 batch，从 1 增大到 64 后时间几乎不变（仍略低于屋脊点）；注意力的强度等于 GQA 分组数，与 batch 无关，读的 KV 随 batch 线性增长。以 LLaMA-3-8B、4K 上下文、H100 为例，下限从 4.6 ms 增加到 14.8 ms，注意力从 3% 增长到 70%。瓶颈从"读权重"转移到"读 KV Cache"。所以大 batch 下 KV 量化、MLA、分页 decode kernel 的价值更大。见 [decode 的时间花在哪里](token-journey.md#decode-的时间花在哪里)。

**36. 为什么同一个请求，温度设为 0，在线上服务里两次结果可能不同？**

??? success "参考答案"
    浮点加法不满足结合律。请求所在的 batch 大小、与哪些请求拼在一起、分块 prefill 的切分方式、选择的 kernel 实现都可能改变累加顺序，得到略有不同的 logits。当两个候选 token 的概率非常接近时，argmax 就会翻转，之后的文本完全不同。本手册在 CPU 上就测到单独计算与 batch 中计算的 logits 相差约 3e-5。要严格复现，需要 batch 不变的 kernel。见[哪些优化会改变输出](token-journey.md#哪些优化会改变输出)。

**37. 给你一个新模型的 config.json，你会怎么评估部署它需要多少资源？**

??? success "参考答案"
    1. 数参数：总参数（决定权重显存）与激活参数（决定计算量）；
    2. 算每个 token 的 KV Cache（注意 GQA、MLA、滑动窗口、线性注意力层）；
    3. 显存：权重 + KV Cache（目标并发 × 平均上下文 × 每 token KV）+ 激活和运行时开销（预留 10%～20%）；
    4. 延迟：decode 下限 ≈（权重字节 + batch 的 KV 字节）/ 带宽；prefill ≈ 2 × 激活参数 × token 数 / 有效算力；
    5. 并行：头数、KV 头数是否能被 TP 整除；MoE 是否需要 EP；
    6. 特殊结构：是否需要特定的 kernel（MLA、软截断、注意力汇聚、线性注意力）以及推理引擎是否支持。

    见[主流模型架构巡礼](models.md)与[参数量、算力与显存估算](../inference/estimation.md)。

**38. 列举你知道的"用计算换访存"或"用访存换计算"的推理优化。**

??? success "参考答案"
    - 用计算换访存：KV Cache 量化（读更少的字节，多做反量化）、weight-only 量化（同理）、FlashAttention（不写出 scores，反向时重算）、MLA（缓存潜在向量，用时再做矩阵乘法）、投机解码（用额外的验证计算换更少的权重读取次数）；
    - 用访存换计算：KV Cache 本身（存下 K、V，避免重复计算整个前缀）、前缀缓存（同理，跨请求复用）。

    本质上都是在屋顶线的两个维度之间做交换：看瓶颈在哪一侧，就把代价转移到另一侧。

**39. 如果让你为一个 RAG 应用（输入很长、输出很短、大量请求共享同一批文档）优化推理，你会从哪些方面入手？**

??? success "参考答案"
    这类负载 prefill 占主导，TTFT 是关键指标：

    - 前缀缓存：共享的系统提示和文档放在提示词的最前面，让前缀尽量相同，最大化命中率（RadixAttention / APC）；
    - 分块 prefill，避免长 prefill 阻塞其他请求的 decode；
    - prefill 计算受限，用 FP8（W8A8）比 weight-only INT4 更有效；
    - 如果并发大，考虑 PD 分离，给 prefill 分配更多资源；
    - 长上下文的注意力开销大，确保使用 FlashAttention 等高效 kernel；
    - 输出短，投机解码收益有限。

**40. 把整本手册压缩成三句话。**

??? success "参考答案"
    （参考写法，你可以有自己的版本）

    1. 大模型是一个按"下一个 token 的概率"训练出来的 Transformer：嵌入、若干层（注意力 + FFN，带残差和归一化）、输出层，推理时只能一个 token 一个 token 地生成。
    2. 推理分两个阶段：prefill 并行处理提示词、计算受限；decode 每步一个 token、要读全部权重和 KV Cache、访存受限。
    3. 几乎所有推理优化都在做三件事：让读的字节更少（量化、GQA/MLA、稀疏）、让读一次被用得更多（批处理、投机解码、前缀共享）、让硬件别闲着（连续批处理、分块 prefill、PD 分离、算子融合、CUDA Graphs）。
