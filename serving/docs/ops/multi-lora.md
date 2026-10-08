# 多 LoRA 服务：分段矩阵乘与适配器调度

<p class="lead">一个基座模型、上千个 LoRA 微调版本，是 SaaS 平台和企业内部最常见的部署形态。为每个版本单独起一个实例太浪费；把它们放进同一个实例、同一个 batch 里一起算，才是正确的做法——但这要求一个 batch 里的不同请求乘上不同的低秩矩阵。这一章实现两种批量计算的方法（BGMV 与 SGMV），再看适配器在显存、内存和存储之间怎样调度，以及它和前缀缓存、路由的关系。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. LoRA 的前向是什么？合并权重和不合并各有什么利弊？
    2. 一个 batch 里的请求用不同的适配器，怎样一次算完？BGMV 和 SGMV 有什么区别？
    3. 一个适配器有多大？能在显存里常驻多少个？
    4. 多 LoRA 服务的前缀缓存要注意什么？

??? success "自测参考答案（先自己答，再展开对照）"
    1. $y = xW + s \cdot (xA)B$。合并（把 $sAB$ 加进 $W$）后推理和原模型一样快，但一个实例只能服务一个适配器；不合并时基座部分所有请求一起算，LoRA 部分按请求分别算，一个实例可以服务很多适配器，代价是额外的 LoRA 计算和读取。
    2. 基座部分整批一次矩阵乘；LoRA 部分按请求各自的适配器算。BGMV：每个 token 按自己的适配器编号取出 A、B，逐 token 做小的矩阵 × 向量，适合 decode；SGMV：把用同一个适配器的 token 排到一起，每一段做一次矩阵乘，适合 prefill。
    3. 8B 的基座、秩 16、全部 7 个投影时约 80 MB；留 8 GB 的显存能常驻约 100 个，CPU 内存可以缓存上千个。
    4. 同样的前缀在不同的适配器下，K、V 是不同的（LoRA 作用在 q、k、v 投影上），所以块的哈希键里必须混入适配器的编号，否则会命中别的适配器算出的 KV。

## 不合并权重，批量计算

LoRA 的前向是 $y = xW + s \cdot (xA)B$，其中 $A \in \mathbb{R}^{d \times r}$、$B \in \mathbb{R}^{r \times d'}$，秩 $r$ 通常只有 8～64，$s$ 是缩放系数。单个适配器时可以把 $sAB$ 合并进 $W$，推理和原模型一样快；但多个适配器共用一个实例时不能合并——每个请求要的 $W$ 不同。做法是基座部分所有请求一起算一次 $xW$，LoRA 部分按请求各自的适配器单独算。两种批量的写法：

- **BGMV**（batched gather matrix-vector）：每个 token 按自己的适配器编号"取出"对应的 $A$、$B$，逐 token 做小的矩阵 × 向量。decode 时每个请求只有一个 token，正合适；
- **SGMV**（segmented gather matrix-vector）：把使用同一个适配器的 token 排在一起，每一段做一次小矩阵乘。prefill 时一个请求有很多 token，一段就是一次正经的矩阵乘，效率高得多。

![图：多 LoRA 服务——基座矩阵乘一起算，LoRA 小矩阵按适配器分组计算再加回去](../assets/figures/multi-lora-batch.svg){.aig-svg}

```python
import torch

torch.manual_seed(0)
D_IN, D_OUT, R, N_ADAPTERS = 256, 256, 16, 8
W = torch.randn(D_IN, D_OUT) / 16                                   # 共享的基座权重
A = torch.randn(N_ADAPTERS, D_IN, R) / 16                           # 每个适配器一对低秩矩阵
B = torch.randn(N_ADAPTERS, R, D_OUT) / 4
SCALE = 2.0                                                         # alpha / r

# 一个混合批次：6 个请求，各自用不同的适配器，token 数不同（decode 的请求 1 个 token，prefill 的请求很多个）
lengths, adapters = [1, 1, 37, 1, 12, 1], [3, 5, 3, 0, 7, 5]
x = torch.randn(sum(lengths), D_IN)
tok_adapter = torch.tensor([a for a, n in zip(adapters, lengths) for _ in range(n)])


def per_request():
    out, s = [], 0
    for a, n in zip(adapters, lengths):                             # 朴素：逐个请求调用一次矩阵乘
        xi = x[s:s + n]
        out.append(xi @ W + SCALE * (xi @ A[a]) @ B[a])
        s += n
    return torch.cat(out)


def bgmv():
    """BGMV（batched gather matrix-vector）：每个 token 按自己的适配器编号取 A、B，适合 decode"""
    shrink = torch.einsum("ti,tir->tr", x, A[tok_adapter])          # 先降到秩 r
    return x @ W + SCALE * torch.einsum("tr,tro->to", shrink, B[tok_adapter])


def sgmv():
    """SGMV（segmented gather matrix-vector）：把使用同一个适配器的 token 排在一起，每段做一次小矩阵乘，适合 prefill"""
    order = torch.argsort(tok_adapter, stable=True)
    xs, out = x[order], torch.empty(len(x), D_OUT)
    ids, counts = torch.unique_consecutive(tok_adapter[order], return_counts=True)
    s = 0
    for a, n in zip(ids.tolist(), counts.tolist()):
        out[s:s + n] = SCALE * (xs[s:s + n] @ A[a]) @ B[a]
        s += n
    lora = torch.empty_like(out)
    lora[order] = out
    return x @ W + lora


ref = per_request()
print("BGMV 与逐请求一致：", torch.allclose(bgmv(), ref, atol=1e-5), "；SGMV 与逐请求一致：", torch.allclose(sgmv(), ref, atol=1e-5))
print(f"基座矩阵乘 1 次（{len(x)} 行）；逐请求要调用 {len(lengths)} 次；SGMV 按适配器分成 {len(set(adapters))} 段")
```

```text title="输出"
BGMV 与逐请求一致： True ；SGMV 与逐请求一致： True
基座矩阵乘 1 次（53 行）；逐请求要调用 6 次；SGMV 按适配器分成 4 段
```

真实的 kernel（Punica 提出的 BGMV / SGMV，以及后来的各种变体）把"取出适配器 + 小矩阵乘"融合进一个 kernel，先做降维（shrink，$x \to xA$，维度降到 $r$），再做升维（expand，$\to (xA)B$）并直接加到基座的输出上。计算量很小——秩 16 的 LoRA 只占基座计算的 0.5% 左右（[系统设计参考答案](../career/design-answers-1.md#4-多租户-lora-服务)算过）——但 decode 时它是访存受限的：一个 batch 里用到的不同适配器越多，要读的 LoRA 权重就越多。

## 适配器的调度

| 层级 | 放什么 | 容量（8B 基座、秩 16、全部 7 个投影） |
| --- | --- | --- |
| GPU 显存 | 当前 batch 用到的和最热门的适配器 | 每个约 80 MB，留 8 GB 能常驻约 100 个 |
| CPU 内存 | 全部活跃的适配器 | 上千个，换入一次约几毫秒（PCIe） |
| 本地盘 / 对象存储 | 全部适配器 | 首次加载要下载，几百毫秒到秒级 |

- **显存管理**：S-LoRA 把适配器的权重和 KV Cache 放在统一的分页内存里，按需换入换出，避免两者各自预留造成浪费；
- **每个 batch 的适配器上限**：一个 batch 里同时出现的适配器数量有上限（显存里的槽位），超过的请求要等下一批。这也是一个调度约束：调度器挑选请求时要考虑"这个请求的适配器是否已经在显存里"；
- **路由亲和**：让同一个适配器的请求集中到少数几个实例，每个实例的 batch 里不同的适配器更少，换入换出更少；热门适配器复制到多个实例；
- **前缀缓存要区分适配器**：LoRA 作用在 Q、K、V 投影上时，同样的前缀在不同适配器下的 KV 不同，块的哈希键必须混入适配器的编号（[KV 传输与存储](../comm/kv-storage.md)）。

在推理框架里：vLLM 用 `--enable-lora` 打开，`--max-loras` 是一个 batch 里最多的适配器数，`--max-lora-rank` 是支持的最大秩，`--max-cpu-loras` 是 CPU 上缓存的数量，批量计算在 `vllm/lora/punica_wrapper/` 和 `vllm/lora/ops/` 下；SGLang 用 `--enable-lora`、`--lora-paths`、`--max-loras-per-batch`、`--max-lora-rank`，`--lora-backend` 选择 kernel（默认 `csgmv`），实现在 `srt/lora/`（含显存池 `mem_pool.py` 与淘汰策略 `eviction_policy.py`）。

!!! interview "怎么讲清楚"
    多 LoRA 的核心是"基座一起算、LoRA 按请求分别算"：decode 用 BGMV（逐 token 取适配器），prefill 用 SGMV（按适配器分段做矩阵乘），真实 kernel 把取权重和 shrink / expand 融合在一起。再讲调度：适配器约几十 MB，显存常驻热门的、CPU 缓存全部，每个 batch 的适配器数有上限；路由按适配器亲和；前缀缓存的键要混入适配器编号。能算出"LoRA 计算只占 0.5%，但 batch 里的适配器越多，decode 要读的权重越多"，说明你理解瓶颈在哪里。

## 练习

**1. 什么时候应该把 LoRA 合并进基座权重？**

??? success "参考答案"
    某个适配器的流量大到足以独占一个或多个实例时：合并后推理和原模型一样快，没有 LoRA 的额外开销，也不占 batch 里的适配器槽位。代价是这些实例只能服务这一个版本，失去了混合批处理的灵活性。常见的做法是"头部适配器合并后独立部署，长尾适配器在共享实例上多 LoRA 服务"。

**2. 秩 64 和秩 16 的适配器混在一个 batch 里，会有什么问题？**

??? success "参考思路"
    kernel 通常按"最大秩"分配和计算，秩 16 的适配器也要按 64 补齐，浪费计算和显存；每个适配器在显存里的占用也按最大秩预留。做法：按秩分组调度（同一个 batch 尽量只含相近秩的适配器），或者使用支持变长秩的 kernel；`--max-lora-rank` 不要设得比实际需要大。

## 小结

- [x] 多 LoRA 不合并权重：基座部分一起算，LoRA 部分按请求分别算；BGMV 适合 decode，SGMV 适合 prefill，两者与逐请求计算结果一致。
- [x] LoRA 的计算量很小，瓶颈在 decode 时读取不同适配器的权重；一个 batch 的适配器数有上限。
- [x] 适配器分层存放（显存、内存、存储），按适配器做路由亲和，前缀缓存的键要混入适配器编号。
