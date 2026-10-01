# FP8 细粒度量化与分组 GEMM：DeepGEMM

<p class="lead">DeepSeek-V3 用 FP8 训练，推理也直接用 FP8 权重和 FP8 激活（W8A8）：权重按 128×128 的块、激活按每个 token 每 128 个通道各取一个缩放因子。这一章讲这种细粒度量化为什么必要、GEMM 怎样在累加时乘上缩放因子，再看 MoE 特有的问题——几百个专家各自只分到几十到几千个 token，分组 GEMM 怎样组织数据，以及 decode 的专家 GEMM 为什么需要很大的全局 batch 才能摆脱访存瓶颈。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. FP8 的 E4M3 格式有多少位尾数？逐张量缩放在什么情况下会出问题？
    2. 细粒度量化的 GEMM，缩放因子在哪一步乘进去？为什么每 128 个元素要把部分和提升到 fp32？
    3. MoE 的分组 GEMM 有"连续"和"带掩码"两种布局，分别配合 DeepEP 的哪种模式？
    4. decode 时一个专家每步要分到多少个 token，专家 GEMM 才不再受权重读取限制？
    5. DeepGEMM 为什么在运行时 JIT 编译 kernel？

??? success "自测参考答案（先自己答，再展开对照）"
    1. 3 位尾数（E4M3：4 位指数、3 位尾数，最大 448）。有极端离群值时，逐张量的缩放因子被离群值拉大，其他数值被压进非规格化数甚至 0，误差迅速放大。
    2. 沿 K 维每 128 个元素做一次 FP8 矩阵乘（Tensor Core），得到的部分和在 CUDA Core 的 fp32 寄存器里乘上激活和权重的两个缩放，再累加。H800 的 FP8 Tensor Core 内部累加精度有限（约 14 位），K 很大时一口气累加完会损失精度，每 128 个元素提升一次到 fp32，顺便也就把缩放乘进去了。
    3. 连续布局（各专家的 token 首尾相接、补齐到块大小）配合 DeepEP 的高吞吐模式，用于 prefill；带掩码的布局（每个专家一块固定大小的缓冲区 + 实际 token 数）配合低延迟模式的固定槽位，形状固定、可以录进 CUDA Graph，用于 decode。
    4. 专家 GEMM 的算术强度约等于每个专家的 token 数 × 2（FP8 权重每字节）；H800 FP8 的屋脊点约 591 FLOP/字节，所以每个专家每步至少要约 295 个 token。decode 要靠 DP Attention + 大规模 EP 汇集全局的 batch，才能让每个专家分到几百个 token。
    5. 不同的专家数、token 数、形状需要不同的 kernel 配置（块大小、流水级数），运行时按实际的形状 JIT 编译，把形状作为编译期常量，能得到针对性更强的代码，又不用预先编译所有组合。

## 缩放粒度

![图：FP8 的缩放粒度——激活按 1×128、权重按 128×128，GEMM 沿 K 每 128 累加一次](../assets/figures/fp8-scaling.svg){.aig-svg}

FP8 的 E4M3 只有 3 位尾数，最大值 448。量化时要先除以一个缩放因子把数值放进这个范围。粒度越粗，一个大的离群值就会把整组的缩放因子拉大，其他数值被压到很小、甚至落到非规格化数或者 0。分布式训练手册的[混合精度与 FP8](train://practice/mixed-precision/#fp8-训练)一章从训练的角度测过这一点；这里换成推理 GEMM 的形状（输入维 7168），并且把 kernel 的真实算法也写出来——沿 K 维每 128 个元素做一次 FP8 矩阵乘，在 fp32 里乘上激活和权重两个缩放因子后累加：

```python
import torch

torch.manual_seed(0)
M, K, N, G = 64, 7168, 256, 128                   # token 数、输入维（DeepSeek-V3 的 hidden）、输出维、量化分组大小
FP8, FP8_MAX = torch.float8_e4m3fn, 448.0
w = torch.randn(K, N) / K**0.5


def quant_act(a, fine):
    """返回 (FP8 值, 缩放)。fine=True：每个 token 每 128 个通道一个缩放；否则整个张量一个"""
    g = a.view(M, K // G, G)
    s = g.abs().amax(-1, keepdim=True) / FP8_MAX if fine else a.abs().max() / FP8_MAX * torch.ones(M, K // G, 1)
    return (g / s).to(FP8), s


def quant_w(fine):
    b = w.view(K // G, G, N // G, G)              # 权重：每个 128×128 的块一个缩放
    s = b.abs().amax((1, 3), keepdim=True) / FP8_MAX if fine else w.abs().max() / FP8_MAX * torch.ones(K // G, 1, N // G, 1)
    return (b / s).to(FP8), s


def gemm(aq, sa, wq, sw):
    """DeepGEMM 的算法：沿 K 每 128 个元素做一次 FP8 矩阵乘，在 fp32 里乘上两个缩放后累加"""
    out = torch.zeros(M, N)
    for kb in range(K // G):
        part = aq[:, kb].float() @ wq[kb].float().reshape(G, N)
        out += part * sa[:, kb] * sw[kb, 0, :, 0].repeat_interleave(G)
    return out


a = torch.randn(M, K)
aq, sa = quant_act(a, True)
wq, sw = quant_w(True)
deq = (aq.float() * sa).view(M, K) @ (wq.float() * sw).view(K, N)
print("分块累加与先反量化再相乘一致：", torch.allclose(gemm(aq, sa, wq, sw), deq, rtol=1e-4, atol=1e-4))

outliers = torch.randperm(K)[:8]                  # 8 个离群通道
normal = torch.ones(K, dtype=torch.bool)
normal[outliers] = False
print("离群值倍数   逐张量缩放   分块缩放   （正常通道那部分输出的相对误差）")
for mag in (1e1, 1e3, 3e4, 1e5):
    a = torch.randn(M, K)
    a[:, outliers] *= mag
    ref = a[:, normal].double() @ w[normal].double()
    errs = []
    for fine in (False, True):
        aq, sa = quant_act(a, fine)
        wq, sw = quant_w(fine)
        a_dq = (aq.float() * sa).view(M, K)       # 缩放由包含离群值的整行 / 整个张量决定
        out = a_dq[:, normal] @ (wq.float() * sw).view(K, N)[normal]
        errs.append(((out.double() - ref).norm() / ref.norm()).item())
    print(f"{mag:>8.0e}   {errs[0]:>9.4f}   {errs[1]:>8.4f}")
```

```text title="输出"
分块累加与先反量化再相乘一致： True
离群值倍数   逐张量缩放   分块缩放   （正常通道那部分输出的相对误差）
   1e+01      0.0373     0.0370
   1e+03      0.0374     0.0367
   3e+04      0.1379     0.0388
   1e+05      0.4375     0.0592
```

两点观察：

- 约 3.7% 的误差是 E4M3 自身 3 位尾数的底线，任何缩放方式都绕不过去；FP8 的指数范围很宽（最小的规格化数约 $2^{-6}$，非规格化数到 $2^{-9}$），离群值只有十倍、千倍时，逐张量缩放也能应付；
- 离群值到了 $3 \times 10^4$ 以上，逐张量缩放把正常通道压进非规格化数甚至 0，误差迅速放大；分块缩放只让离群值所在的那一组（128 个通道）受影响，其他组的缩放因子不变。训练中的激活和梯度确实会出现这种量级的离群值，模型用分块缩放训练出来，推理也必须按同样的粒度量化，否则精度会掉。

**缩放因子在哪里乘。** 输出的每个元素 $y_{mn} = \sum_k a_{mk} w_{kn}$，按 K 分成 128 一块后，第 $b$ 块内的激活共享缩放 $s^a_{m,b}$、权重共享 $s^w_{b,n'}$（$n'$ 是 $n$ 所在的 128 列块），所以

$$
y_{mn} = \sum_b s^a_{m,b}\, s^w_{b,n'} \sum_{k \in b} \hat a_{mk}\, \hat w_{kn}
$$

内层的求和在 Tensor Core 上用 FP8 做，外层乘缩放、累加在 CUDA Core 的 fp32 寄存器里做。这恰好也解决了另一个问题：H800 的 FP8 Tensor Core 内部累加器的精度有限（DeepSeek-V3 论文测得大约只有 14 位），K 很大时直接在 Tensor Core 里累加完整个 7168 维会损失精度；每 128 个元素就把部分和"提升"到 fp32，精度损失可以忽略。缩放因子的乘法因此几乎是免费的——它本来就要在这一步把部分和搬出来。

## MoE 的分组 GEMM

MoE 层里，一张卡上的几个专家各自要和分到自己的 token 做一次 GEMM。token 数每一步都在变，而且专家之间差别很大。常见的两种组织方式：

- **连续布局**（contiguous）：把分到各个专家的 token 按专家排好、首尾相接，每个专家的那一段补齐到 kernel 的块大小（BLOCK_M，比如 64 或 128）的整数倍，再配一个"每一行属于哪个专家"的索引。适合 prefill：token 多，补齐浪费的比例小。DeepEP 高吞吐模式的输出正好可以按这种方式排列；
- **带掩码的布局**（masked）：每个专家一块固定大小的缓冲区 `[专家数, 最大 token 数, hidden]`，再给一个"每个专家实际有几个 token"的数组，kernel 在 GPU 上读这个数组、跳过空的块。形状固定、不需要 CPU 知道真实的 token 数，可以录进 CUDA Graph。它正好对应 DeepEP 低延迟模式的固定槽位输出（见 [NVSHMEM 与 DeepEP](../comm/nvshmem-deepep.md#低延迟模式固定槽位没有-cpu-同步)），用于 decode。

专家 GEMM 的效率取决于每个专家分到多少 token。模拟 256 个专家、每卡 4 个（EP=64），比较 decode 和 prefill：

```python
import numpy as np

rng = np.random.default_rng(0)
EXPERTS, LOCAL, TOPK, HIDDEN, INTER = 256, 4, 8, 7168, 2048   # 每张卡 4 个专家（EP=64）
w_bytes = 3 * HIDDEN * INTER                                   # 一个专家 gate/up/down 的 FP8 权重字节数
ridge = 1979e12 / 3.35e12                                      # H800 FP8 的屋脊点（FLOP/字节）
print(f"一个专家的权重 {w_bytes / 2**20:.0f} MiB；每个 token 过一个专家 {2 * w_bytes / 1e6:.0f} MFLOPs")
print(f"H800 FP8 的屋脊点约 {ridge:.0f} FLOP/字节：每个专家每步至少要 {ridge / 2:.0f} 个 token，专家 GEMM 才不再受权重读取限制")

popularity = rng.dirichlet(np.full(EXPERTS, 2.0))              # 专家的冷热程度
for stage, global_tokens in (("decode，全局每步 4K token", 4096), ("decode，全局每步 32K token", 32768), ("prefill，全局 256K token", 262144)):
    counts = rng.multinomial(global_tokens * TOPK, popularity)[:LOCAL]   # 本卡 4 个专家各分到多少个 token
    for block_m in (64, 128):
        padded = int(sum(-(-c // block_m) * block_m for c in counts))    # 连续布局：每个专家的 token 数补齐到 BLOCK_M 的倍数
        print(f"{stage}，BLOCK_M={block_m}：每专家 {counts.min()}～{counts.max()} 个 token，"
              f"补齐后有效行 {counts.sum() / padded:.0%}，算术强度 {2 * counts.mean():.0f} FLOP/字节")
```

```text title="输出"
一个专家的权重 42 MiB；每个 token 过一个专家 88 MFLOPs
H800 FP8 的屋脊点约 591 FLOP/字节：每个专家每步至少要 295 个 token，专家 GEMM 才不再受权重读取限制
decode，全局每步 4K token，BLOCK_M=64：每专家 69～248 个 token，补齐后有效行 88%，算术强度 308 FLOP/字节
decode，全局每步 4K token，BLOCK_M=128：每专家 69～248 个 token，补齐后有效行 80%，算术强度 308 FLOP/字节
decode，全局每步 32K token，BLOCK_M=64：每专家 552～2052 个 token，补齐后有效行 97%，算术强度 2440 FLOP/字节
decode，全局每步 32K token，BLOCK_M=128：每专家 552～2052 个 token，补齐后有效行 93%，算术强度 2440 FLOP/字节
prefill，全局 256K token，BLOCK_M=64：每专家 4467～16722 个 token，补齐后有效行 100%，算术强度 20004 FLOP/字节
prefill，全局 256K token，BLOCK_M=128：每专家 4467～16722 个 token，补齐后有效行 100%，算术强度 20004 FLOP/字节
```

这张表解释了大规模 MoE 推理的一个核心设计：

- 专家 GEMM 的算术强度是"每个专家分到的 token 数 × 2"。每步全局只有 4K 个 token 时，平均每个专家 128 个 token，算术强度远低于 FP8 的屋脊点（约 590），**专家 GEMM 的时间几乎就是读一遍专家权重的时间**，token 再少也省不了多少；
- 所以 decode 要把**全局** batch 做大：DP Attention 让每张卡处理不同的请求，MoE 层把所有卡的 token 汇集到各个专家——几十、上百张卡的 batch 加在一起，每个专家才能分到几百个 token。这就是 DeepSeek 在 decode 阶段用上百张卡组成一个 EP 组的原因之一：不只是为了放下权重，更是为了让每个专家"吃饱"；
- 块越大，补齐浪费越多；decode 时 token 少，要用更小的 BLOCK_M 或者带掩码的布局。prefill 时每个专家有成千上万个 token，补齐几乎没有浪费，是纯粹的计算问题。

## DeepGEMM

DeepGEMM 是 DeepSeek 开源的 FP8 GEMM 库，支持上面的细粒度缩放，以及普通 GEMM、连续布局和带掩码布局的分组 GEMM。几个设计值得注意：

- **运行时 JIT 编译**：安装时不编译任何 kernel，第一次遇到某个形状（N、K、分组数等）时才生成并编译对应的 kernel，把形状作为编译期常量，编译器可以完全展开循环、选择最合适的分块；结果缓存在磁盘上。推理框架启动时通常会用常见形状预热；
- **Hopper 的全套特性**：TMA 异步搬运、warp 专门化（一部分 warp 只负责搬数据、一部分只负责计算）、WGMMA；这些在 CUDA 手册的 [Hopper 异步编程](cuda://advanced/async-hopper/)一章里讲过；
- **代码量小**：核心 kernel 只有几百行，便于学习和修改，官方发布时报告在 H800 上达到 1350 TFLOPS 以上的 FP8 算力；
- **新硬件**：Blackwell 在硬件上原生支持"每 32 个元素一个缩放因子"的块缩放格式（MXFP8），缩放因子是 2 的幂（UE8M0）；DeepGEMM 和后续的 DeepSeek 模型也跟进了这种格式——缩放因子取 2 的幂，乘法变成指数加法。

在推理框架里：SGLang 和 vLLM 的 DeepSeek 模型都可以用 DeepGEMM 做 FP8 的线性层和 MoE 分组 GEMM（也可以用 CUTLASS、Triton 或 FlashInfer 的实现）；配合 DeepEP 时，prefill 走"高吞吐 dispatch → 连续布局分组 GEMM"，decode 走"低延迟 dispatch → 带掩码的分组 GEMM"，形状固定、整层可以录进 CUDA Graph。

!!! interview "面试怎么答"
    讲 FP8 推理时，先讲格式和粒度：E4M3 三位尾数，逐张量缩放在极端离群值下失效，所以用激活 1×128、权重 128×128 的分块缩放；再讲 GEMM：每 128 个元素做一次 FP8 矩阵乘，部分和提升到 fp32 并乘上两个缩放，顺带解决了 Tensor Core 累加精度不足的问题。讲 MoE 时一定要点出"专家 GEMM 的算术强度 = 每专家 token 数 × 2"，decode 要靠 DP Attention + 大 EP 汇集全局 batch 才能让专家吃饱；分组 GEMM 的连续布局配 prefill、带掩码布局配 decode 和 CUDA Graph。

!!! info "相关章节"
    - [量化部署实战](../perf/quantization-deploy.md)（本书：部署层面的选择）
    - [量化与 GEMV](cuda://advanced/quantization/)、[Tensor Core](cuda://advanced/tensor-core/)（CUDA：kernel 层面）

## 练习

**1. 专家要多少 token 才吃饱？** 如果专家权重用 BF16 存放（其他条件相同），每个专家每步要多少 token 才能摆脱访存瓶颈？用 FP4 呢？

??? success "参考答案"
    BF16：每个参数 2 字节，每 token 的 FLOPs 不变，算术强度变成"每专家 token 数"（而不是 ×2）；屋脊点用 BF16 的峰值算约 $989 / 3.35 \approx 295$ FLOP/字节，所以要约 295 个 token——和 FP8 差不多，因为 FP8 的峰值和字节数同时翻了一倍。FP4（比如 Blackwell 的 NVFP4，峰值再翻倍、字节再减半）：算术强度是 4 × token 数，屋脊点也翻倍，仍然要几百个 token。结论：低精度省的是显存和带宽，让同样的延迟下能放下更多的专家和 KV，但"每个专家要分到几百个 token"这件事不会因为精度变低而改变。

**2. 连续布局的补齐。** prefill 时一张卡上 32 个专家（EP 较小），平均每个专家 100 个 token，BLOCK_M = 128。补齐浪费大约多少？有什么办法减少？

??? success "参考答案"
    每个专家的 token 数补齐到 128 的倍数，平均 100 个 token 的专家大多补到 128，浪费约 $1 - 100/128 \approx 22\%$（专家之间不均匀时更多或更少）。减少的办法：用更小的 BLOCK_M（64）；让 kernel 支持不补齐的变长分组（按专家的真实长度切块，最后一块用掩码）；或者增大 prefill 的 batch，让每个专家的 token 更多。

## 小结

- [x] E4M3 只有 3 位尾数，约 3.7% 的误差是底线；极端离群值会让逐张量缩放失效，所以用激活 1×128、权重 128×128 的分块缩放。
- [x] 分块缩放的 GEMM：每 128 个 K 元素做一次 FP8 矩阵乘，部分和提升到 fp32 并乘上两个缩放；同时解决了 Tensor Core 累加精度有限的问题。
- [x] 专家 GEMM 的算术强度 = 每专家 token 数 × 2（FP8）；decode 要靠 DP Attention + 大规模 EP 汇集全局 batch，让每个专家分到几百个 token。
- [x] 分组 GEMM 的连续布局（补齐到 BLOCK_M，配合高吞吐 dispatch）用于 prefill，带掩码的布局（固定形状、可录 CUDA Graph，配合低延迟 dispatch）用于 decode。
- [x] DeepGEMM 运行时按形状 JIT 编译，用 TMA、warp 专门化和 WGMMA；Blackwell 上跟进硬件原生的块缩放格式。
