# 算子加速：注意力、编译与融合

<p class="lead">不改模型、不少走一步，单靠把每一步的算子做对，SDXL 和 FLUX 就能快 1.5～3 倍——这是所有加速手段里最"白拿"的一层。这一章讲三件事：注意力 kernel（FlashAttention、SageAttention 为什么对扩散模型特别有效），torch.compile 和算子融合（把 AdaLN、归一化、激活函数这些访存小算子收拾掉），以及 CUDA Graph（扩散模型固定形状的天然优势）。每一项都先在 CPU 上把"它在算什么、省了什么"算清楚，再给真卡上的典型收益。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. 朴素注意力和 FlashAttention 的显存、访存量各是多少？对十万 token 的视频模型意味着什么？
    2. SageAttention 把什么量化成了 INT8 / FP8？为什么扩散模型能承受这种精度？
    3. 一个 DiT 块里哪些算子是访存受限的？它们占多少时间？torch.compile 怎么处理它们？
    4. CUDA Graph 省的是什么？为什么扩散模型比 LLM 更适合用它？
    5. 开了编译之后第一张图特别慢、换个分辨率又慢一次，是为什么？怎么办？

??? success "自测参考答案（先自己答，再展开对照）"
    1. 朴素注意力要把 $N \times N$ 的分数矩阵写进显存再读回来做 softmax，显存 $O(N^2)$、访存 $O(N^2)$；FlashAttention 分块在片上算完 softmax 和加权和，显存 $O(N)$、访存降到 $O(N^2 d / M)$（$M$ 是片上缓存大小）。十万 token 的 $N^2$ 矩阵单头就是 20 GB——没有 FlashAttention 视频模型根本跑不起来。
    2. 把 $Q$、$K$ 量化成 INT8（按块平滑之后），$P$、$V$ 用 FP8 或 FP16 累加，矩阵乘走 INT8 / FP8 Tensor Core，比 FlashAttention-2 快 2～3 倍。扩散模型每一步的输出是"方向"而不是精确值，几十步的去噪对单步误差有平均效应，所以注意力里的量化误差肉眼看不出来。
    3. LayerNorm / RMSNorm、AdaLN 的缩放和偏移、GELU / SiLU、残差相加、RoPE——它们 FLOP 很少但要把整个激活读写一遍，不编译时能占到一步时间的 30%～50%。torch.compile（Inductor）把相邻的逐元素算子融成一个 kernel，一次读写搞定，还能消掉 Python 和 kernel 启动开销。
    4. 省的是 CPU 侧的 kernel 启动开销：一步几百上千个 kernel，每个几微秒的启动时间加起来可观，小模型、低分辨率时能占一步的一半。扩散模型每一步的形状完全相同（没有 KV 在长），一张 graph 录一次、重放几十次，比 LLM 要为不同 batch / 序列长度准备多张 graph 容易得多。
    5. 编译是按形状的：第一次遇到某个形状要跑编译（几十秒到几分钟），换分辨率、换 batch 就是新形状。办法：固定服务支持的几种分辨率、启动时预热编译、开启 Inductor 的持久缓存、用 `dynamic=False` 避免动态形状的反复重编译。

## 注意力：从 $O(N^2)$ 显存到分块

![图：FlashAttention——分块、online softmax，不落地 N×N 的注意力矩阵](../assets/figures/flash-attention.svg){.aig-svg}

朴素注意力把 $S = QK^\top$ 整个算出来，softmax 之后再乘 $V$。$N$ 大的时候那张 $N \times N$ 的矩阵本身就是灾难。先把账算出来：

```python
def attn_bytes(n, d_head, heads, dtype=2):
    """一层注意力的显存与访存（字节）：朴素实现要落地 N×N 的分数矩阵"""
    scores = n * n * heads * dtype                       # QKᵀ 的结果，每个头一份
    qkv = 3 * n * heads * d_head * dtype
    naive = scores * 2 + qkv + n * heads * d_head * dtype   # 写分数、读回做 softmax、再写 P
    flash = qkv + n * heads * d_head * dtype             # 只读 QKV、写输出；分数在片上
    return scores, naive, flash

GB = 1024 ** 3
print(f"{'模型':<26} {'token':>8} {'N×N 分数矩阵':>12} {'朴素访存':>10} {'Flash 访存':>10}")
for name, n, dh, h in [("FLUX 1024²", 4608, 128, 24), ("FLUX 2048²", 16896, 128, 24),
                       ("Wan 2.1 720p", 76112, 128, 40), ("HunyuanVideo 720p", 119056, 128, 24)]:
    s, nv, fl = attn_bytes(n, dh, h)
    print(f"{name:<26} {n:>8,} {s / GB:>10.1f} GB {nv / GB:>8.1f} GB {fl / GB:>8.2f} GB")
```

```text title="输出"
模型                            token     N×N 分数矩阵       朴素访存   Flash 访存
FLUX 1024²                    4,608        0.9 GB      2.0 GB     0.11 GB
FLUX 2048²                   16,896       12.8 GB     25.9 GB     0.39 GB
Wan 2.1 720p                 76,112      431.6 GB    866.1 GB     2.90 GB
HunyuanVideo 720p           119,056      633.6 GB   1270.0 GB     2.72 GB
```

HunyuanVideo 一层的分数矩阵 630 GB——朴素实现在任何卡上都不可能。FlashAttention 把 $Q$ 的每个块和 $K$、$V$ 的每个块在片上算完 softmax（online softmax，见 [FlashAttention](cuda://advanced/attention/)），分数矩阵从不落地，显存从 $O(N^2)$ 变 $O(N)$，访存降两个数量级。**对扩散模型来说它不是优化，是前提**。

在 PyTorch 里用 `F.scaled_dot_product_attention` 就会自动选 FlashAttention / memory-efficient 后端；用一个小例子确认三种写法结果一致、并看中间张量的大小：

```python
import torch
import torch.nn.functional as F

torch.manual_seed(0)
B, H, N, D = 1, 4, 256, 32
q, k, v = (torch.randn(B, H, N, D) for _ in range(3))

def naive(q, k, v):
    s = (q @ k.transpose(-1, -2)) / D ** 0.5            # [B, H, N, N] 落地
    return s.softmax(-1) @ v, s.numel() * 4

def blocked(q, k, v, block=64):                           # 分块 + online softmax：分数矩阵只有 block×N 大
    out = torch.zeros_like(q)
    peak = 0
    for i in range(0, N, block):
        qi = q[:, :, i:i + block]
        m = torch.full((B, H, qi.shape[2], 1), float("-inf"))
        l = torch.zeros(B, H, qi.shape[2], 1)
        acc = torch.zeros(B, H, qi.shape[2], D)
        for j in range(0, N, block):
            s = (qi @ k[:, :, j:j + block].transpose(-1, -2)) / D ** 0.5
            m_new = torch.maximum(m, s.amax(-1, keepdim=True))
            p = (s - m_new).exp()
            l = l * (m - m_new).exp() + p.sum(-1, keepdim=True)
            acc = acc * (m - m_new).exp() + p @ v[:, :, j:j + block]
            m = m_new
            peak = max(peak, s.numel() * 4)
        out[:, :, i:i + block] = acc / l
    return out, peak

o1, bytes1 = naive(q, k, v)
o2, bytes2 = blocked(q, k, v)
o3 = F.scaled_dot_product_attention(q, k, v)
print(f"三种写法一致：{torch.allclose(o1, o2, atol=1e-5)} {torch.allclose(o1, o3, atol=1e-5)}")
print(f"朴素实现落地的分数矩阵 {bytes1 / 1024:.0f} KB，分块实现任一时刻只有 {bytes2 / 1024:.0f} KB（{bytes1 / bytes2:.0f} 倍）")
```

```text title="输出"
三种写法一致：True True
朴素实现落地的分数矩阵 1024 KB，分块实现任一时刻只有 64 KB（16 倍）
```

### SageAttention：把注意力也量化

注意力里的两个矩阵乘 $QK^\top$ 和 $PV$ 在 FlashAttention 里仍是 FP16 / BF16。SageAttention 的做法：$Q$、$K$ 按块减去均值（平滑掉 $K$ 的离群通道）后量化成 INT8，$QK^\top$ 走 INT8 Tensor Core；$P$、$V$ 用 FP8（SageAttention2）或 FP16 累加。效果是比 FlashAttention-2 快 2～3 倍、显存不变、生成结果肉眼无差。

它为什么对扩散模型可行而 LLM 要谨慎：扩散模型每一步预测的是"往哪走"，几十步的迭代对单步误差有平均效应；LLM 的 decode 每一步的输出直接决定下一个 token，误差会被采样放大。视频模型的注意力占了八成时间，SageAttention 几乎是它们默认的后端。

## 访存受限的小算子：编译与融合

一个 DiT 块除了矩阵乘和注意力，还有一串"不怎么算、但要把激活读写一遍"的算子。把它们的 FLOP 和访存列出来：

```python
def block_profile(n, d, dtype=2):
    """一个 DiT 块里各算子的 FLOP 与访存字节数（粗估：逐元素算子读一次写一次）"""
    act = n * d * dtype
    ops = [
        ("AdaLN 调制 (×2)",      2 * 3 * n * d,            2 * 2 * act),
        ("RMSNorm (×2)",         2 * 4 * n * d,            2 * 2 * act),
        ("QKV 投影",             2 * n * d * 3 * d,        act + 3 * act + 3 * d * d * dtype),
        ("RoPE",                 6 * n * d,                2 * 2 * act),
        ("注意力 (Flash)",        4 * n * n * d,            4 * act),
        ("输出投影",             2 * n * d * d,            2 * act + d * d * dtype),
        ("残差相加 + 门控 (×2)",  2 * 2 * n * d,            2 * 3 * act),
        ("MLP 升维",             2 * n * d * 4 * d,        act + 4 * act + 4 * d * d * dtype),
        ("GELU",                 8 * n * 4 * d,            2 * 4 * act),
        ("MLP 降维",             2 * n * 4 * d * d,        4 * act + act + 4 * d * d * dtype),
    ]
    return ops

PEAK, BW = 989e12, 3.35e12                                 # H100：bf16 算力、显存带宽
n, d = 4608, 3072                                          # FLUX 1024²
print(f"{'算子':<22} {'GFLOP':>8} {'访存 MB':>8} {'强度':>6} {'受限于':>6} {'时间 μs':>8}")
tot = 0
for name, flops, byts in block_profile(n, d):
    t = max(flops / PEAK, byts / BW) * 1e6                 # 屋顶线：取两者中慢的
    tot += t
    print(f"{name:<22} {flops / 1e9:>8.1f} {byts / 2 ** 20:>8.0f} {flops / byts:>6.0f} {'算力' if flops / PEAK > byts / BW else '带宽':>6} {t:>8.0f}")
small = sum(max(f / PEAK, b / BW) for nm, f, b in block_profile(n, d) if nm.split()[0] in ("AdaLN", "RMSNorm", "RoPE", "残差相加", "GELU")) * 1e6
print(f"合计 {tot:.0f} μs，其中逐元素小算子 {small:.0f} μs（{small / tot:.0%}）——它们 FLOP 不到 1%，时间却占这么多")
```

```text title="输出"
算子                        GFLOP    访存 MB     强度    受限于    时间 μs
AdaLN 调制 (×2)               0.1      108      1     带宽       34
RMSNorm (×2)                0.1      108      1     带宽       34
QKV 投影                    260.9      162   1536     算力      264
RoPE                        0.1      108      1     带宽       34
注意力 (Flash)               260.9      108   2304     算力      264
输出投影                       87.0       72   1152     算力       88
残差相加 + 门控 (×2)              0.1      162      0     带宽       51
MLP 升维                    347.9      207   1603     算力      352
GELU                        0.5      216      2     带宽       68
MLP 降维                    347.9      207   1603     算力      352
合计 1539 μs，其中逐元素小算子 220 μs（14%）——它们 FLOP 不到 1%，时间却占这么多
```

这是屋顶线模型下的理想值（没算 kernel 启动、没算 L2），但结论稳：**小算子 FLOP 不到 1%，时间却占一成多到两成**；实际没编译的实现里，因为每个小算子各是一个 kernel、各自启动、各自读写，占比会更高。torch.compile（Inductor）做的事：

- **融合**：相邻的逐元素算子（调制 → 归一化 → 激活）合成一个 kernel，激活只读写一次；
- **消掉 Python 开销**：一步几百个算子的 Python 调度变成一段生成的代码；
- **选更好的矩阵乘配置**（`mode="max-autotune"` 会对每个形状试多种 kernel）。

典型收益：FLUX、SD3 这类 DiT 1.3～1.8 倍，SDXL 的 UNet 1.2～1.5 倍。代价是**按形状编译**：第一次见到某个形状要几十秒到几分钟，换分辨率、换 batch 又来一次。服务里的做法：固定支持的分辨率列表、启动时全部预热、开 `TORCHINDUCTOR_CACHE_DIR` 持久缓存、`dynamic=False`。

## CUDA Graph：固定形状的红利

上千个小 kernel 的启动开销有多大、CUDA Graph 和融合各省多少，用推理系统手册里的同一个时间模型拨一拨：

<div class="aig-widget" data-widget="launch-overhead"></div>

一步去噪要启动几百到上千个 kernel，每个几微秒的 CPU 侧开销，小模型、小分辨率时能占一步的一半——GPU 在等 CPU 发指令。CUDA Graph 把一步的全部 kernel 录下来，之后一次提交整张图（见 [CUDA Graphs 与 torch.compile](serving://engine/graphs-compile/)）。

扩散模型是 CUDA Graph 的理想用户：**每一步的形状完全相同**。LLM 要为不同的 batch 大小和序列长度各录一张图、还要处理 KV 在增长，扩散模型一张图就够（CFG 的两路也是固定 batch）。算一下它值多少：

```python
def step_time(kernels, launch_us, gpu_us, graph):
    cpu = 0 if graph else kernels * launch_us                # 用 graph：启动开销几乎为零
    return max(cpu, gpu_us) if not graph else gpu_us + 10     # 不用 graph：CPU 发不过来时 GPU 空等

print(f"{'情形':<30} {'kernel 数':>8} {'GPU 计算':>9} {'无 graph':>9} {'有 graph':>9}  收益")
for name, kernels, gpu in [("SD 1.5 512²（H100）", 600, 8000), ("SDXL 1024²（H100）", 900, 45000),
                           ("FLUX 1024²（H100）", 1200, 170000), ("SD 1.5 512² · batch 4 · 小卡", 600, 30000)]:
    a, b = step_time(kernels, 30, gpu, False), step_time(kernels, 30, gpu, True)
    print(f"{name:<30} {kernels:>8} {gpu / 1e3:>7.1f} ms {a / 1e3:>7.1f} ms {b / 1e3:>7.1f} ms  {a / b:>4.2f}×")
print("（PyTorch eager 里每个算子的 CPU 侧开销（Python + 分发 + 启动）按 30 μs 估；GPU 快、模型小时 CPU 发不过来，graph 的收益最大）")
```

```text title="输出"
情形                             kernel 数    GPU 计算   无 graph   有 graph  收益
SD 1.5 512²（H100）                   600     8.0 ms    18.0 ms     8.0 ms  2.25×
SDXL 1024²（H100）                    900    45.0 ms    45.0 ms    45.0 ms  1.00×
FLUX 1024²（H100）                   1200   170.0 ms   170.0 ms   170.0 ms  1.00×
SD 1.5 512² · batch 4 · 小卡          600    30.0 ms    30.0 ms    30.0 ms  1.00×
（PyTorch eager 里每个算子的 CPU 侧开销（Python + 分发 + 启动）按 30 μs 估；GPU 快、模型小时 CPU 发不过来，graph 的收益最大）
```

FLUX 这种大模型一步 170 ms，CPU 侧那三十几毫秒完全被 GPU 的计算盖住；SD 1.5 这种小模型在快卡上 GPU 只要 8 ms，CPU 发指令却要 18 ms，GPU 一半时间在等——这时 graph 直接把一步砍掉一半以上。torch.compile 本身也会把每个算子的 CPU 开销压低，所以两者叠加后 graph 的额外收益会小一些。所以 CUDA Graph 对"小模型 + 快卡 + 交互式"的场景（实时绘画、少步蒸馏模型）收益最大，对大模型锦上添花。torch.compile 的 `mode="reduce-overhead"` 会自动用上它。

## 把它们叠起来

三种手段作用在不同的地方，可以叠加：

| 手段 | 作用对象 | 典型收益 | 代价 |
| --- | --- | --- | --- |
| FlashAttention（`sdpa`） | 注意力的显存与访存 | 从"跑不起来"到能跑；长序列上 2～4× | 无 |
| SageAttention | 注意力的矩阵乘精度 | 注意力部分 2～3×，视频模型整体 1.5～2× | 需要对应的 kernel 包；极少数模型要调参 |
| torch.compile | 逐元素小算子、Python 开销、矩阵乘配置 | DiT 1.3～1.8× | 按形状编译，首次慢，要预热和缓存 |
| CUDA Graph（reduce-overhead） | kernel 启动开销 | 小模型 + 快卡 1.5～2×，大模型几个百分点 | 形状必须固定，显存多占一份 |
| channels_last（UNet） | 卷积的内存布局 | SDXL 1.1～1.3× | 无 |
| fused QKV / fused AdaLN | 线性层的 kernel 数 | 几个百分点 | 要改模型代码 |

顺序建议：**sdpa → compile → SageAttention（视频必上）→ reduce-overhead（小模型）**。前两项几乎零成本，后面按模型和场景选。

!!! interview "怎么讲清楚"
    讲"不改模型怎么把扩散推理加速"，按层次答：注意力层——FlashAttention 让 $N^2$ 的分数矩阵不落地，是视频模型的前提；SageAttention 把 $QK^\top$ 量化到 INT8，扩散对单步误差有平均效应所以可行，视频模型 1.5～2×。算子层——AdaLN、归一化、激活函数 FLOP 不到 1% 但时间占一两成，torch.compile 融合它们，DiT 1.3～1.8×。启动层——CUDA Graph 消掉 kernel 启动开销，扩散模型每步形状固定所以一张图通吃，小模型快卡上收益最大。最后提代价：编译按形状，服务要固定分辨率、预热、持久缓存。

## 练习

1. 把 `blocked` 的 `block` 从 64 改成 16 和 256，分数矩阵的峰值各是多少？真实的 FlashAttention 为什么不把块设得越小越好？

??? success "参考答案"
    峰值和 block × N 成正比：16 时是 64 的 1/4，256 时是 4 倍。但块越小，$K$、$V$ 被重复从显存读进片上的次数越多（访存量 $\propto N^2 d / \text{block}$），而且小块填不满 Tensor Core。真实实现把块大小定在片上 SRAM 刚好放下 $Q$、$K$、$V$ 块和中间结果的量级（64～128）。

2. 用 `block_profile` 算 Wan 2.1 720p（$n = 76112$、$d = 5120$）一个块的时间分布。小算子占比和 FLUX 1024² 相比怎么变？为什么？

??? success "参考答案"
    注意力的 $4N^2 d$ 项随 $N^2$ 增长，变成绝对大头（八成以上），小算子占比降到个位数。token 数越大，注意力越主导，编译融合的相对收益越小、注意力 kernel 的收益越大——这就是视频模型"SageAttention 必上、compile 锦上添花"的原因。

3. 服务要支持 512²、768²、1024² 三种分辨率和 batch 1～4，开了 `torch.compile(dynamic=False)`。启动时要预热多少种形状？如果用户还能传任意长宽比呢？

??? success "参考答案"
    3 种分辨率 × 4 种 batch × （CFG 的两路拼 batch 时 batch 翻倍，但那是固定的）= 12 种形状，每种编译几十秒，启动预热要十来分钟，所以要开持久缓存让第二次启动直接复用。任意长宽比意味着形状不可枚举：要么把分辨率对齐到有限的几档（多数服务的做法），要么对主干用 `dynamic=True` 接受一部分性能损失，要么对非常见形状退回不编译的路径。

## 小结

- [x] FlashAttention 让 $N \times N$ 的分数矩阵不落地：显存 $O(N)$、访存降两个数量级，对视频模型是前提而不是优化。
- [x] SageAttention 把 $QK^\top$ 量化到 INT8 / FP8，快 2～3×；扩散的多步迭代对单步误差有平均效应，所以承受得住。
- [x] AdaLN、归一化、激活函数这些小算子 FLOP 不到 1% 却占一两成时间，torch.compile 融合它们，DiT 1.3～1.8×；代价是按形状编译，服务要固定分辨率、预热、持久缓存。
- [x] CUDA Graph 消掉 kernel 启动开销，扩散每步形状固定所以一张图通吃，小模型 + 快卡时收益最大。
