# 多 LoRA、ControlNet 与 IP-Adapter 的服务

<p class="lead">线上的文生图服务很少只跑一个裸模型：用户选风格（LoRA）、上传线稿（ControlNet）、给参考图（IP-Adapter）。它们对推理的影响完全不同——LoRA 是几十 MB 的低秩增量，可以融进权重也可以旁路算；ControlNet 是半个去噪网络，每一步都要多跑一遍；IP-Adapter 只是多几十个 token。这一章把三种插件的计算和显存成本算出来，比较"融合"和"旁路"两种 LoRA 服务方式的切换代价，再给出多插件服务的放置策略。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. LoRA 加在扩散模型的哪些层上？融合进权重和旁路计算各有什么代价？
    2. 一个 batch 里的请求用不同的 LoRA，怎么一起算？
    3. ControlNet 为什么每一步都要多跑一个网络？它比 T2I-Adapter 贵在哪？
    4. IP-Adapter 的成本在哪？它和参考图拼进序列有什么区别？
    5. 服务里几十个 LoRA、几个 ControlNet 怎么放？

??? success "自测参考答案（先自己答，再展开对照）"
    1. 通常加在注意力的 Q、K、V、O 投影上，有时也加在 MLP 上；SD 的 UNet 里有时还加在卷积上。融合：把 $BA$ 加进 $W$，推理零开销，但换 LoRA 要先减掉旧的再加新的——理论上只是把受影响的权重读写一遍（几毫秒），实现里常常要几百毫秒（逐层的 Python 操作，量化权重还得重新量化、编译图可能失效）；旁路：每层多算 $x A^\top B^\top$，开销随秩线性增长（秩 64 约多 4% 受影响层的计算），切换是瞬时的，而且一个 batch 里可以混用。
    2. 用旁路方式：每个请求带自己的 LoRA 索引，低秩矩阵乘按请求分组做（和 LLM 服务里的 SGMV / punica 同一套，见[多 LoRA 服务](serving://ops/multi-lora/)）；融合方式做不到，只能串行。
    3. ControlNet 是去噪网络编码器的一个副本，接收控制图，输出每一层的残差加到主干上；它的输入含当前的带噪潜变量，所以每一步都要重算。T2I-Adapter 只看控制图不看潜变量，整个生成过程只算一次。
    4. 把参考图经 CLIP / SigLIP 编码成几十个 token，通过额外的一组交叉注意力注入；成本是每层多一个小的交叉注意力（token 数几十），几乎可以忽略。把参考图的 VAE 潜变量拼进序列（Kontext / OmniGen 的做法）要付注意力的平方，贵得多。
    5. LoRA 几十 MB，全部常驻显存，旁路计算、按请求分组；ControlNet 几 GB，热的常驻、冷的放锁页内存按需搬（几百毫秒）或按类型分实例；IP-Adapter 的编码器常驻。

## LoRA：融合还是旁路

LoRA 给权重加一个低秩增量 $W' = W + BA$（$A$ 是 $r \times d$，$B$ 是 $d \times r$，$r$ 通常 8～128）。推理时有两种算法：

- **融合**（merge）：提前算 $W + BA$，之后和没有 LoRA 一样；
- **旁路**（unmerged）：每层多算一次 $x A^\top B^\top$，加到 $xW^\top$ 上。

先算旁路的开销和融合的切换代价：

```python
def lora_costs(d_model, layers, adapted_per_layer, rank, n_tokens, hbm_gbps=3350, dtype=2):
    """旁路：每个被加 LoRA 的线性层多 2×N×(d×r + r×d) FLOP；融合 / 解融合：读写一遍受影响的权重"""
    base_flops = 2 * n_tokens * d_model * d_model * adapted_per_layer * layers           # 只算被 LoRA 覆盖的那些线性层
    lora_flops = 2 * n_tokens * (d_model * rank + rank * d_model) * adapted_per_layer * layers
    weight_bytes = d_model * d_model * dtype * adapted_per_layer * layers
    merge_ms = 2 * weight_bytes / (hbm_gbps * 1e9) * 1e3                              # 读 + 写
    lora_mb = (d_model * rank * 2) * dtype * adapted_per_layer * layers / 2 ** 20
    return lora_flops / base_flops, merge_ms, lora_mb

print("FLUX：57 层，每层 4 个注意力投影加 LoRA，1024² 共 4608 个 token")
print(f"{'秩 r':>5} {'旁路的额外计算（相对受影响的层）':>22} {'LoRA 大小':>9} {'融合或解融合一次':>12}")
for rank in (8, 16, 64, 128):
    extra, merge_ms, mb = lora_costs(3072, 57, 4, rank, 4608)
    print(f"{rank:>5} {extra:>22.1%} {mb:>7.0f} MB {merge_ms:>10.0f} ms")
```

```text title="输出"
FLUX：57 层，每层 4 个注意力投影加 LoRA，1024² 共 4608 个 token
  秩 r       旁路的额外计算（相对受影响的层）   LoRA 大小     融合或解融合一次
    8                   0.5%      21 MB          3 ms
   16                   1.0%      43 MB          3 ms
   64                   4.2%     171 MB          3 ms
  128                   8.3%     342 MB          3 ms
```

两边的代价都不大，但性质不同：

| | 融合 | 旁路 |
| --- | --- | --- |
| 每步开销 | 0 | 秩 64 时受影响的层多 4%，整体 1%～3% |
| 切换 LoRA | 解融合旧的 + 融合新的：理论几毫秒，实现里常几百毫秒（逐层操作、重新量化、重编译） | 换一个指针 |
| batch 里混用不同 LoRA | 不行（权重只有一份） | 可以：按请求分组做低秩矩阵乘 |
| 多个 LoRA 叠加 | 加权求和后融合一次 | 多算几个旁路 |
| 和量化 / 编译的兼容 | 融合会破坏量化权重（要重新量化）；编译图不变 | 量化权重不动；旁路要进编译图 |

**服务里几乎都用旁路**：切换瞬时、能混 batch、和 FP8 权重兼容。融合只在"一个实例固定跑一个风格"的离线场景里用。多 LoRA 混 batch 的实现和 LLM 服务完全一样——按请求分组的分段矩阵乘（SGMV），见[多 LoRA 服务](serving://ops/multi-lora/)，那里有可运行的实现。

一个扩散特有的细节：**LoRA 的权重可以随时间步变**。有些方法（LCM-LoRA 的变体、分步风格 LoRA）在不同噪声水平上用不同的缩放，旁路方式天然支持（每步换个标量），融合方式做不到。

## ControlNet：每步多跑半个网络

ControlNet 把去噪网络的编码器（UNet 的下采样路径，或 DiT 的前几层）复制一份，输入控制图（线稿、深度、姿态）和**当前的带噪潜变量**，输出每一层的残差加回主干。因为输入里有当前潜变量，**每一步都要重算**：

```python
def controlnet_cost(main_params_b, control_params_b, n_control, steps, cfg):
    """每步多算一个 ControlNet；T2I-Adapter 只算一次"""
    per_step = main_params_b + n_control * control_params_b
    return per_step / main_params_b

for name, main, ctrl in [("SD 1.5", 0.86, 0.36), ("SDXL", 2.6, 1.25), ("FLUX（Union ControlNet）", 11.9, 3.3)]:
    for n in (1, 2, 3):
        print(f"{name:<22} {n} 个 ControlNet：每步计算 {controlnet_cost(main, ctrl, n, 30, 2):>4.2f}×，显存多 {n * ctrl * 2:>4.1f} GB（bf16）")
```

```text title="输出"
SD 1.5                 1 个 ControlNet：每步计算 1.42×，显存多  0.7 GB（bf16）
SD 1.5                 2 个 ControlNet：每步计算 1.84×，显存多  1.4 GB（bf16）
SD 1.5                 3 个 ControlNet：每步计算 2.26×，显存多  2.2 GB（bf16）
SDXL                   1 个 ControlNet：每步计算 1.48×，显存多  2.5 GB（bf16）
SDXL                   2 个 ControlNet：每步计算 1.96×，显存多  5.0 GB（bf16）
SDXL                   3 个 ControlNet：每步计算 2.44×，显存多  7.5 GB（bf16）
FLUX（Union ControlNet） 1 个 ControlNet：每步计算 1.28×，显存多  6.6 GB（bf16）
FLUX（Union ControlNet） 2 个 ControlNet：每步计算 1.55×，显存多 13.2 GB（bf16）
FLUX（Union ControlNet） 3 个 ControlNet：每步计算 1.83×，显存多 19.8 GB（bf16）
```

SDXL 挂一个 ControlNet 每步贵一半，挂三个翻倍——而且这些计算在 CFG 的两路上都要做。几种降低成本的做法：

| 做法 | 省在哪 | 代价 |
| --- | --- | --- |
| T2I-Adapter | 不看潜变量，整个过程只算一次 | 控制力弱于 ControlNet |
| ControlNet 只在前 k 步生效 | 构图在前几步定下，后面不需要控制 | 细节控制变弱 |
| 多个控制合并成一个 Union 模型 | 一次前向处理多种控制 | 需要专门训练 |
| 把 ControlNet 融进主干（ControlNet-XS、LoRA 式控制） | 参数少一个量级 | 效果略降 |
| 和特征缓存一起跳步 | ControlNet 的输出也缓存 | 同缓存的误差 |

"只在前 60% 的步生效"是服务里最常用的一条：几乎不损质量，省四成的 ControlNet 计算。

## IP-Adapter：便宜的参考图

IP-Adapter 把参考图经 CLIP / SigLIP 图像编码器变成几十个 token，用一组**额外的交叉注意力**注入（和文本的交叉注意力并行）。成本：图像编码器跑一次（几十毫秒），每层多一个 token 数几十的交叉注意力——几乎可以忽略。

对比"把参考图拼进序列"的做法（FLUX Kontext、OmniGen、Wan 的参考图模式）：参考图的 VAE 潜变量作为 token 和噪声潜变量一起做联合注意力，序列长度翻倍，注意力成本四倍：

```python
def ref_image_cost(n_img, n_ref_tokens, d, layers, mode):
    attn_base = layers * 4 * n_img ** 2 * d
    if mode == "IP-Adapter（交叉注意力）":
        extra = layers * 2 * 2 * n_img * n_ref_tokens * d                   # 每层一个小交叉注意力
    else:                                                                  # 拼进序列：联合注意力
        n = n_img + n_ref_tokens
        extra = layers * 4 * n ** 2 * d - attn_base
    return extra / attn_base

for mode, n_ref in [("IP-Adapter（交叉注意力）", 64), ("参考图拼进序列（联合注意力）", 4096)]:
    print(f"{mode:<22} 注意力部分多 {ref_image_cost(4608, n_ref, 3072, 57, mode):>5.0%}")
```

```text title="输出"
IP-Adapter（交叉注意力）      注意力部分多    1%
参考图拼进序列（联合注意力）         注意力部分多  257%
```

便宜的办法控制力弱（只能传"风格 / 主体大概长什么样"），贵的办法能做精确编辑。产品上常常两种都提供——调度时它们是两种形状的请求（见[生成服务的调度](scheduling.md)）。

## 放置：热的常驻，冷的搬

把三种插件按大小和使用频率放到合适的地方：

```python
GB = 1024 ** 3
inventory = [
    # 名字,              单个大小 GB, 数量, 每次请求用到的概率
    ("LoRA（秩 64）",     0.10,  200, 0.6),
    ("IP-Adapter 编码器",  0.8,    1, 0.3),
    ("ControlNet（SDXL）", 2.5,    6, 0.25),
]
budget = 80 - 24 - 9.5 - 6                                                 # 80 GB 卡：FLUX + T5 + 激活与 VAE 峰值
print(f"插件的显存预算约 {budget:.0f} GB")
total_all = sum(size * n for _, size, n, _ in inventory)
print(f"全部常驻要 {total_all:.0f} GB：{'放得下' if total_all <= budget else '放不下'}")
pcie = 25.0                                                                # GB/s
for name, size, n, p in inventory:
    plan = "全部常驻" if size < 1.0 and size * n <= 0.5 * budget else "按 LRU 常驻几个，其余放锁页内存按需搬"
    print(f"  {name:<18} {n:>3} 个 × {size:>4.1f} GB = {size * n:>5.1f} GB → {plan}（按需搬一次 {size / pcie * 1e3:>4.0f} ms）")
```

```text title="输出"
插件的显存预算约 40 GB
全部常驻要 36 GB：放得下
  LoRA（秩 64）         200 个 ×  0.1 GB =  20.0 GB → 全部常驻（按需搬一次    4 ms）
  IP-Adapter 编码器       1 个 ×  0.8 GB =   0.8 GB → 全部常驻（按需搬一次   32 ms）
  ControlNet（SDXL）     6 个 ×  2.5 GB =  15.0 GB → 按 LRU 常驻几个，其余放锁页内存按需搬（按需搬一次  100 ms）
```

规则很朴素：**LoRA 全部常驻**（200 个也才 20 GB，秩小的更少），按请求分组旁路计算；**ControlNet 按 LRU 常驻几个**，其余放锁页内存，按需搬一个 100 毫秒——比一次生成的几秒短得多；或者干脆**按 ControlNet 类型分实例**，路由层按请求带的控制类型分发，每个实例只常驻自己那一个。

和 LLM 的多 LoRA 服务比，扩散这边多了两个变量：插件有"每步都算"（ControlNet）和"只算一次"（T2I-Adapter、IP-Adapter 编码器）之分，影响的是每步成本而不只是显存；以及 LoRA 的作用可以随时间步变。调度器估算一个请求的耗时，要把这些算进去（见[生成服务的调度](scheduling.md)里的预计耗时函数）。

!!! interview "面试怎么答"
    被问"文生图服务怎么支持几十个 LoRA 和 ControlNet"，先分清三类插件的成本性质：LoRA 是几十 MB 的低秩增量，旁路算每步多百分之几、切换瞬时、能在一个 batch 里混（和 LLM 的 SGMV 同一套），融合零开销但切换要遍历权重、不能混 batch，所以服务用旁路；ControlNet 是半个去噪网络、每步都要重算、SDXL 挂一个每步贵一半，常用"只在前 60% 步生效"和 Union 模型来省；IP-Adapter 只是几十个 token 的交叉注意力、几乎免费，而把参考图拼进序列要付注意力的平方。放置上 LoRA 全常驻、ControlNet LRU 常驻加锁页内存按需搬或按类型分实例。

## 练习

1. 用 `lora_costs` 算 SDXL（UNet 的注意力层约 70 个、$d = 1280$ 到 2048 不等，按 1600 估）秩 128 的旁路开销和融合代价。和 FLUX 比，为什么融合在 SDXL 上更划算一些？

??? success "参考答案"
    SDXL 受影响的权重更小（几百 MB），融合一次几十毫秒，而旁路在秩 128 时对小 $d$ 的层占比更高（$2r/d$，$d$ 小则比例大）。所以 SDXL 这种小模型固定风格时融合更合算；FLUX 的 $d = 3072$ 让旁路的相对开销小、融合的绝对代价大，旁路更合算。

2. 一个 batch 里 4 个请求用 4 个不同的 LoRA，旁路方式怎么组织计算？如果其中两个请求用的是同一个 LoRA 呢？

??? success "参考答案"
    把 batch 的激活按 LoRA 索引分组，每组做自己的 $x A^\top B^\top$（分段矩阵乘，一个 kernel 处理所有组），结果按原顺序加回主干输出；同一个 LoRA 的两个请求自然在一组，矩阵乘更大更高效。这正是 punica / SGMV 的做法。

3. ControlNet 只在前 60% 的步生效，对特征缓存（TeaCache）的跳步决策有什么影响？

??? success "参考答案"
    ControlNet 生效和失效的那一步，主干的输入残差会有一个跳变，探针距离突然变大，缓存会（正确地）强制真算一步；之后 ControlNet 不再参与，相邻步更相似，缓存能跳得更多。实现上要把 ControlNet 的输出也纳入缓存的残差，否则跳过的步会丢掉控制信号。

## 小结

- [x] LoRA：旁路算每步多百分之几、切换瞬时、能混 batch、和 FP8 权重兼容，服务里几乎都用它；融合零开销但切换要遍历权重，只适合固定风格的离线场景。
- [x] ControlNet 每步重算半个网络（SDXL 挂一个每步贵一半），T2I-Adapter 只算一次；"只在前 60% 步生效"和 Union 模型是常用的省法。
- [x] IP-Adapter 几十个 token 的交叉注意力几乎免费；参考图拼进序列要付注意力的平方，是两种形状的请求。
- [x] 放置：LoRA 全常驻、按请求分组旁路计算；ControlNet LRU 常驻 + 锁页内存按需搬（百毫秒）或按类型分实例；预计耗时要把"每步都算"的插件算进去。
