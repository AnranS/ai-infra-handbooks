# 去噪网络：从 UNet 到 DiT

<p class="lead">去噪网络吃掉了生成的几乎全部计算量，它长什么样决定了能用什么优化。SD 1.5 和 SDXL 用的是卷积 UNet：分辨率金字塔、跳连、只在低分辨率层做注意力；PixArt、SD3、FLUX 和所有主流视频模型换成了 DiT：把潜变量切成 patch 变成一串 token，然后就是一个标准的 Transformer。这一章用最小配置把两种结构都搭出来，看清它们的 token 数怎么算、计算量分布在哪、条件信息从哪里注入——这些直接决定了后面注意力加速、序列并行和特征缓存怎么做。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. UNet 的注意力为什么只在低分辨率层做？它的计算量分布和 DiT 有什么不同？
    2. DiT 的 token 数怎么算？FLUX 生成 1024×1024 一张图，去噪网络处理多少个 token？
    3. 时间步和文本条件分别是怎么注入 UNet 和 DiT 的？AdaLN-Zero 是什么？
    4. MMDiT（SD3 / FLUX）的"双流"和 DiT 的交叉注意力有什么区别？对推理有什么影响？
    5. 为什么说 DiT 比 UNet 更适合推理系统的那套优化（序列并行、缓存、编译）？

??? success "自测参考答案（先自己答，再展开对照）"
    1. 注意力是 token 数的平方，UNet 最高分辨率的特征图 token 太多（SDXL 128×128 = 16384），所以只在下采样 2～4 倍的层做自注意力和交叉注意力，高分辨率层只有卷积。UNet 的计算量分散在卷积和注意力里、各层形状不同；DiT 每一层形状相同、计算量几乎全在注意力和 MLP 的矩阵乘里。
    2. token 数 = (潜变量高 / patch) × (潜变量宽 / patch)。FLUX：1024/8 = 128，128/2 = 64，64 × 64 = 4096 个图像 token，再拼上 512 个 T5 文本 token 一起做联合注意力。
    3. UNet：时间步经正弦编码 + MLP 后加到每个残差块的特征上，文本通过交叉注意力注入。DiT：时间步（和池化的文本向量）经 MLP 生成每层 LayerNorm 的缩放、偏移和门控系数（AdaLN），门控初始化为零（AdaLN-Zero）让每个块一开始是恒等映射；文本 token 通过交叉注意力（PixArt）或联合注意力（SD3 / FLUX）注入。
    4. 交叉注意力里文本只当 key / value；MMDiT 把文本 token 和图像 token 拼成一个序列做自注意力，两种 token 各有自己的权重（双流），文本表示也随层更新。推理上序列更长（图像 + 文本），但结构更统一，一套注意力 kernel 通吃。
    5. 每层形状相同、全是矩阵乘和注意力——和 LLM 的 Transformer 块一模一样，所以 FlashAttention、序列并行、torch.compile、按层缓存都能直接搬过来；UNet 的卷积、不同分辨率的跳连、各层不同的形状让这些事都要特判。

## UNet：分辨率金字塔上的卷积

SD 1.5 / SDXL 的 UNet 由三部分组成：下采样路径（分辨率每级减半、通道数翻倍）、中间块、上采样路径（和下采样对称，通过跳连把同分辨率的特征拼回来）。注意力只放在低分辨率的层：

```python
import torch
from diffusers import UNet2DConditionModel

torch.manual_seed(0)
# 三级金字塔：16 → 8 → 4，只在后两级做注意力（和 SD 1.5 "最高分辨率层不做注意力"的做法一致）
unet = UNet2DConditionModel(sample_size=16, in_channels=4, out_channels=4, block_out_channels=(32, 64, 64), layers_per_block=1,
                            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D"),
                            up_block_types=("CrossAttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
                            cross_attention_dim=32, attention_head_dim=8, norm_num_groups=8)

# 用 hook 记下每个块的输出形状，看分辨率怎么变
shapes = []
def rec(name):
    def hook(m, i, o):
        out = o[0] if isinstance(o, tuple) else o
        shapes.append((name, tuple(out.shape[-2:]), out.shape[1]))
    return hook
for i, blk in enumerate(unet.down_blocks):
    blk.register_forward_hook(rec(f"down{i} {type(blk).__name__}"))
unet.mid_block.register_forward_hook(rec("mid"))
for i, blk in enumerate(unet.up_blocks):
    blk.register_forward_hook(rec(f"up{i} {type(blk).__name__}"))

with torch.no_grad():
    unet(torch.randn(1, 4, 16, 16), torch.tensor([500]), encoder_hidden_states=torch.randn(1, 8, 32))
for name, hw, c in shapes:
    attn = "注意力" if "CrossAttn" in name or name == "mid" else "只有卷积"
    print(f"{name:<28} 特征图 {hw[0]:>2}×{hw[1]:<2} 通道 {c:>3}  token 数 {hw[0] * hw[1]:>4}  {attn}")
```

```text title="输出"
down0 DownBlock2D            特征图  8×8  通道  32  token 数   64  只有卷积
down1 CrossAttnDownBlock2D   特征图  4×4  通道  64  token 数   16  注意力
down2 CrossAttnDownBlock2D   特征图  4×4  通道  64  token 数   16  注意力
mid                          特征图  4×4  通道  64  token 数   16  注意力
up0 CrossAttnUpBlock2D       特征图  8×8  通道  64  token 数   64  注意力
up1 UpBlock2D                特征图 16×16 通道  64  token 数  256  只有卷积
up2 UpBlock2D                特征图 16×16 通道  32  token 数  256  只有卷积
```

把它放大到 SDXL 的真实尺寸：潜变量 128×128，三级金字塔 128 → 64 → 32。最高分辨率层如果做注意力，token 数是 16384，注意力矩阵 16384² ≈ 2.7 亿个元素；所以 SDXL 只在 64×64（4096 token）和 32×32（1024 token）做，而且把大部分 Transformer 块堆在 32×32 这一级。**UNet 的计算量分散在不同形状的卷积和注意力里**，这是它难优化的根源——每一层都要单独写 kernel、单独决定怎么切。

## DiT：把图切成 token

![图：UNet 在分辨率金字塔上做卷积；DiT 把潜变量切成 token，整个网络是一叠 Transformer 块](../assets/figures/unet-vs-dit.svg){.aig-svg}

DiT 把潜变量按 $p \times p$ 的 patch 切开（通常 $p = 2$），每个 patch 拉平成一个向量，经线性层变成 token——然后就是标准 Transformer：每层形状相同，全是矩阵乘和注意力。

```python
from diffusers import DiTTransformer2DModel

torch.manual_seed(0)
dit = DiTTransformer2DModel(num_attention_heads=2, attention_head_dim=8, in_channels=4, out_channels=8,
                            num_layers=2, sample_size=16, patch_size=2, num_embeds_ada_norm=1000)
x = torch.randn(1, 4, 16, 16)
with torch.no_grad():
    out = dit(x, timestep=torch.tensor([500]), class_labels=torch.tensor([3])).sample
tokens = (16 // 2) ** 2
print(f"潜变量 16×16，patch 2 → {tokens} 个 token，每个 token {dit.config.num_attention_heads * dit.config.attention_head_dim} 维")
print(f"输出 {tuple(out.shape)}：out_channels=8 是 4 个均值 + 4 个方差（DiT 原版同时预测方差，推理时只用均值）")
print(f"每层参数 {sum(p.numel() for p in dit.transformer_blocks[0].parameters()) / 1e3:.1f}K，{len(dit.transformer_blocks)} 层形状完全相同")
```

```text title="输出"
潜变量 16×16，patch 2 → 64 个 token，每个 token 16 维
输出 (1, 8, 16, 16)：out_channels=8 是 4 个均值 + 4 个方差（DiT 原版同时预测方差，推理时只用均值）
每层参数 25.2K，2 层形状完全相同
```

token 数的公式是所有性能估算的起点：

$$
N = \frac{H}{f \cdot p} \times \frac{W}{f \cdot p} \quad (\times\ \frac{T}{f_t \cdot p_t}\ \text{视频})
$$

$f$ 是 VAE 的空间压缩比（8 或 16），$p$ 是 patch 大小（1 或 2），视频再乘上时间维（$f_t$ 通常是 4）。把主流模型代进去：

```python
MODELS = [
    # 名字,              分辨率 (T, H, W),  VAE 压缩 (ft, f),  patch (pt, p),  文本 token
    ("PixArt-Σ 1024²",   (1, 1024, 1024),   (1, 8),  (1, 2),  300),
    ("SD3-medium 1024²", (1, 1024, 1024),   (1, 8),  (1, 2),  333),
    ("FLUX.1 1024²",     (1, 1024, 1024),   (1, 8),  (1, 2),  512),
    ("FLUX.1 2048²",     (1, 2048, 2048),   (1, 8),  (1, 2),  512),
    ("CogVideoX-5B 480p 49 帧", (49, 480, 720), (4, 8), (1, 2), 226),
    ("Wan 2.1 720p 81 帧",  (81, 720, 1280),  (4, 8),  (1, 2),  512),
    ("HunyuanVideo 720p 129 帧", (129, 720, 1280), (4, 8), (1, 2), 256),
]
print(f"{'模型':<26} {'潜变量 T×H×W':>16} {'图像/视频 token':>14} {'文本 token':>9} {'注意力序列':>9}")
for name, (T, H, W), (ft, f), (pt, p), txt in MODELS:
    lt = 1 + (T - 1) // ft if T > 1 else 1               # 视频 VAE 一般是"首帧 + 每 4 帧压 1 帧"
    lh, lw = H // f, W // f
    n = (lt // pt) * (lh // p) * (lw // p)
    print(f"{name:<26} {f'{lt}×{lh}×{lw}':>16} {n:>14,} {txt:>9} {n + txt:>9,}")
```

```text title="输出"
模型                                潜变量 T×H×W    图像/视频 token  文本 token     注意力序列
PixArt-Σ 1024²                    1×128×128          4,096       300     4,396
SD3-medium 1024²                  1×128×128          4,096       333     4,429
FLUX.1 1024²                      1×128×128          4,096       512     4,608
FLUX.1 2048²                      1×256×256         16,384       512    16,896
CogVideoX-5B 480p 49 帧             13×60×90         17,550       226    17,776
Wan 2.1 720p 81 帧                 21×90×160         75,600       512    76,112
HunyuanVideo 720p 129 帧           33×90×160        118,800       256   119,056
```

图像模型的序列在几千，视频模型直接到了十万量级——注意力的计算量和 $N^2$ 成正比，十万 token 的一层注意力就是 $10^{10}$ 量级的分数。这就是视频推理的瓶颈那一章的全部起因。

## 条件怎么注入：AdaLN 与两种注意力

时间步和文本是两种不同性质的条件：时间步是一个标量，对所有 token 一样；文本是一串向量，要和每个图像 token 互动。

**时间步 → AdaLN。** DiT 把时间步编码（加上池化的文本向量）送进一个 MLP，输出每层 LayerNorm 的缩放 $\gamma$、偏移 $\beta$ 和残差门控 $\alpha$：$x \leftarrow x + \alpha \cdot \text{Block}(\gamma \cdot \text{LN}(x) + \beta)$。AdaLN-Zero 把 $\alpha$ 初始化为 0，每个块一开始是恒等映射，深网络训练才稳。推理上它意味着：**同一步里所有 token 共享一组调制系数**，这组系数只依赖时间步——特征缓存（TeaCache）正是靠比较相邻两步的调制后输入来判断"这一步能不能跳过"的。

**文本 → 注意力。** 有两种做法：

| | 交叉注意力（PixArt、UNet） | 联合注意力 / MMDiT（SD3、FLUX、Wan） |
| --- | --- | --- |
| 文本的角色 | 只当 key / value，不更新 | 和图像 token 拼成一个序列做自注意力，文本表示逐层更新 |
| 权重 | 图像一套 | 图像和文本各一套 QKV / MLP（双流），FLUX 后半段改成共享（单流） |
| 序列长度 | 图像 token 数 | 图像 + 文本 token 数 |
| 对推理的影响 | 文本 K/V 可以提前算好缓存 | 序列更长，但只有一种注意力 kernel；文本部分不能缓存 |

用 FLUX 的结构看双流 / 单流的区别——它的前 19 层是双流（图像、文本各一套参数），后 38 层是单流（拼在一起用同一套参数）：

```python
from diffusers import FluxTransformer2DModel

torch.manual_seed(0)
flux = FluxTransformer2DModel(patch_size=1, in_channels=16, num_layers=2, num_single_layers=4, attention_head_dim=8,
                              num_attention_heads=2, joint_attention_dim=16, pooled_projection_dim=16, axes_dims_rope=(2, 2, 4),
                              guidance_embeds=True)                           # dev 版：引导强度是一个输入
n_img, n_txt, d = 64, 8, 16
img = torch.randn(1, n_img, 16)                                   # 已经打包成 token 的图像潜变量
txt = torch.randn(1, n_txt, 16)                                   # T5 的文本 token
img_ids = torch.zeros(n_img, 3); img_ids[:, 1] = torch.arange(n_img) // 8; img_ids[:, 2] = torch.arange(n_img) % 8   # 2D 位置
txt_ids = torch.zeros(n_txt, 3)
with torch.no_grad():
    out = flux(hidden_states=img, encoder_hidden_states=txt, pooled_projections=torch.randn(1, 16),
               timestep=torch.tensor([0.5]), img_ids=img_ids, txt_ids=txt_ids, guidance=torch.tensor([3.5])).sample
double = sum(p.numel() for p in flux.transformer_blocks[0].parameters())
single = sum(p.numel() for p in flux.single_transformer_blocks[0].parameters())
print(f"输出 {tuple(out.shape)}：只还给图像 token，文本 token 在最后被丢掉")
print(f"双流块参数 {double / 1e3:.1f}K（图像、文本各一套），单流块参数 {single / 1e3:.1f}K（共享一套）")
print(f"注意力序列长度 {n_img + n_txt}（图像 {n_img} + 文本 {n_txt}），guidance 是一个输入标量——引导已经蒸馏进模型")
```

```text title="输出"
输出 (1, 64, 16)：只还给图像 token，文本 token 在最后被丢掉
双流块参数 9.7K（图像、文本各一套），单流块参数 4.0K（共享一套）
注意力序列长度 72（图像 64 + 文本 8），guidance 是一个输入标量——引导已经蒸馏进模型
```

FLUX 的位置信息用的是 **RoPE 的 2D / 3D 版本**：每个 token 的 id 是 (t, h, w) 三元组，旋转分别作用在三段维度上。和 LLM 的一维 RoPE 一样，它不改变向量长度、只改变夹角（见[线性代数](math://linear-algebra/)），所以分辨率外推、不同长宽比都能自然处理——也是序列并行时各卡只需知道自己那段 token 的 id 的原因。

## 两种结构的计算量分布

同样是一次前向，UNet 和 DiT 的 FLOP 花在哪里差别很大。用 FLOP 计数器按算子类型分开统计：

```python
from torch.utils.flop_counter import FlopCounterMode

def breakdown(fn):
    with FlopCounterMode(display=False) as fc:
        fn()
    tot = fc.get_total_flops()
    by = {}
    for mod, ops in fc.get_flop_counts().items():
        if mod != "Global":
            continue
        for op, n in ops.items():
            by[str(op).split(".")[-1]] = by.get(str(op).split(".")[-1], 0) + n
    return tot, {k: v / tot for k, v in sorted(by.items(), key=lambda kv: -kv[1])[:3]}

tot_u, by_u = breakdown(lambda: unet(torch.randn(1, 4, 16, 16), torch.tensor([500]), encoder_hidden_states=torch.randn(1, 8, 32)))
tot_d, by_d = breakdown(lambda: dit(torch.randn(1, 4, 16, 16), timestep=torch.tensor([500]), class_labels=torch.tensor([3])))
print("UNet：", ", ".join(f"{k} {v:.0%}" for k, v in by_u.items()))
print("DiT： ", ", ".join(f"{k} {v:.0%}" for k, v in by_d.items()))
```

```text title="输出"
UNet： convolution 86%, addmm 10%, mm 3%
DiT：  addmm 96%, convolution 4%
```

UNet 的 FLOP 主要在卷积里，DiT 几乎全是矩阵乘（线性层 + 注意力的 bmm）。这决定了优化手段的不同：卷积靠 cuDNN 和 channels_last，矩阵乘靠 Tensor Core 和 FlashAttention；量化对矩阵乘友好、对卷积里的 GroupNorm 不友好；序列并行天然适合 token 序列、对卷积的空间维度要特判（见[多卡并行](../perf/parallel.md)）。

## 主流去噪网络一览

| 模型 | 结构 | 参数量 | 条件注入 | 位置编码 | 备注 |
| --- | --- | --- | --- | --- | --- |
| SD 1.5 | UNet | 0.86B | 时间步加法、文本交叉注意力 | 卷积隐含 | 注意力在 64×64 以下 |
| SDXL | UNet | 2.6B | 同上 + 池化文本、尺寸条件 | 卷积隐含 | 32×32 级堆了 10 个 Transformer 块 |
| PixArt-α / Σ | DiT | 0.6B | AdaLN（时间步）、交叉注意力（T5） | 2D 正弦 | 最早证明 DiT 能做文生图 |
| SD3 / 3.5 | MMDiT | 2B / 8B | AdaLN、联合注意力 | 2D 正弦 | 文本、图像双流 |
| FLUX.1 | MMDiT | 12B | AdaLN、联合注意力（19 双流 + 38 单流） | 2D RoPE | dev 版引导已蒸馏 |
| CogVideoX | DiT | 2B / 5B | AdaLN、联合注意力 | 3D RoPE | 3D 全注意力 |
| HunyuanVideo | MMDiT | 13B | AdaLN、联合注意力（双流 + 单流） | 3D RoPE | 3D 全注意力 |
| Wan 2.1 / 2.2 | DiT | 1.3B / 14B（2.2 为 MoE） | AdaLN、交叉注意力（umT5） | 3D RoPE | 2.2 按噪声高低分两个专家 |

趋势很清楚：**新模型全是 DiT 一族**，视频模型无一例外。对推理工程师的好消息是——它们的主体和 LLM 的 Transformer 块是同一个东西，LLM 推理那一整套（FlashAttention、序列并行、编译、量化）大部分能直接复用；坏消息是没有 KV Cache，每一步都要把十万 token 的注意力完整算一遍。

!!! interview "面试怎么答"
    被问"DiT 相比 UNet 对推理有什么好处"，分三层：结构上每层形状相同、全是矩阵乘和注意力，所以 FlashAttention / 序列并行 / torch.compile 直接可用，不像 UNet 要为不同分辨率的卷积和跳连特判；条件注入上 AdaLN 让同一步的调制系数只依赖时间步，特征缓存可以靠它判断相邻步的相似度；代价是 token 数进了注意力的平方项，视频模型十万级的序列让注意力成为绝对瓶颈——没有 KV Cache 可以省，只能靠 kernel、并行和稀疏。

## 练习

1. 把本章的 `dit` 改成 `patch_size=1`，token 数变成多少？FLOP 变成几倍？注意力部分变成几倍？

??? success "参考答案"
    16×16 = 256 个 token，是原来的 4 倍。线性层的 FLOP 和 token 数成正比，变 4 倍；注意力的 QK 和 PV 两个 bmm 和 token 数平方成正比，变 16 倍。patch 大小是"分辨率"和"计算量"之间最直接的旋钮，这也是 FLUX 在 VAE 的 f8 之上再用 2×2 打包的原因。

2. 对 UNet 的例子，统计三个分辨率级各自的 FLOP 占比（提示：给每个 block 单独套一个 `FlopCounterMode`）。最高分辨率层只有卷积，它的占比大还是小？

??? success "参考答案"
    最高分辨率层通道少但特征图大，卷积的 FLOP 和 $H \times W \times C_{in} \times C_{out}$ 成正比，占比并不小；SDXL 里最高分辨率级没有注意力，却仍有可观的卷积计算，而且它的激活是最大的。这也是 UNet 难以均匀切分做并行的原因之一。

3. FLUX 的双流块里文本和图像各有一套 QKV。如果把文本 token 的 K/V 在第一步算好之后缓存起来、后面 27 步复用，省下的计算是多少？这样做对吗？

??? success "参考答案"
    文本 token 只占序列的约 11%（512 / 4608），它的 QKV 投影只是线性层里的一小部分，省不了多少；而且 MMDiT 里文本表示每一层、每一步都在被更新（它参与自注意力、又依赖图像 token），缓存第一步的 K/V 会改变结果——这和交叉注意力里文本只当 K/V 的情况不同。真正能缓存的是交叉注意力结构（PixArt、Wan）里文本的 K/V。

## 小结

- [x] UNet 是分辨率金字塔 + 跳连，注意力只在低分辨率层；计算量分散在不同形状的卷积和注意力里，优化要逐层特判。
- [x] DiT 把潜变量切成 token，每层形状相同、全是矩阵乘和注意力；token 数 = 潜变量尺寸 ÷ patch，视频乘上时间维后轻松到十万。
- [x] 时间步通过 AdaLN 注入（同一步所有 token 共享调制系数），文本通过交叉注意力或联合注意力注入；FLUX 用双流 + 单流，位置用 2D / 3D RoPE。
- [x] 新模型全是 DiT 一族，LLM 推理的那套手段大部分能复用，但没有 KV Cache——注意力的平方项是视频推理的核心瓶颈。
