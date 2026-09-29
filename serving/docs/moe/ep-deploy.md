# 大规模 EP 部署：分层 EPLB、DP 注意力的负载与双 batch 重叠

<p class="lead"><a href="../../distributed/expert-parallel/">专家并行与 DP Attention</a>一章在 4 个进程上实现了 EP + DP Attention，并用冗余专家和重新放置把最忙的卡压回平均水平。规模上到几十、上百张卡之后，会出现三个新问题：负载均衡要和路由约束、节点拓扑配合；DP 注意力的各个 rank 负载不同，而 MoE 的 all-to-all 每层都把它们同步一次；通信和计算怎样重叠。这一章用三个模拟逐个回答，最后看 prefill 和 decode 为什么要用规模相差很大的 EP。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. DeepSeek-V3 的"分组限制路由"是什么？它和 EPLB 的分层策略有什么关系？
    2. 分层 EPLB 和全局 EPLB 各适合什么场景？
    3. DP Attention 下，为什么一个 rank 的请求特别长会拖慢所有 rank？怎样缓解？
    4. 双 batch 重叠在什么情况下收益最大？batch 很小时为什么几乎没用？
    5. 为什么 decode 的 EP 规模通常比 prefill 大得多？

??? success "自测参考答案（先自己答，再展开对照）"
    1. 每个 token 先按节点（组）打分，只在得分最高的 4 个节点里选专家（最多发往 4 个节点）。分层 EPLB 先把专家组分到节点、再在节点内复制和放置专家，保证复制之后每个 token 仍然最多去 4 个节点，保住了路由的这个上限。
    2. 分层 EPLB：节点少的 prefill（EP 规模小），保住节点限制、减少跨节点流量；全局 EPLB：EP 规模很大的 decode，不受节点边界约束，负载可以均衡得更平。
    3. MoE 层的 all-to-all 每一层都要所有 rank 一起参与，相当于每层同步一次：一个 rank 的请求特别长（注意力读的 KV 多），它的注意力算得慢，所有 rank 都要等它。缓解：按 KV 总量而不是请求数分配请求、让空闲的 rank 陪跑空批次、PD 分离让 decode 的负载更平稳。
    4. 计算和通信的时间差不多长时收益最大（本章估算每卡 batch 64～128 时快 1.45～1.65 倍）。batch 很小时计算几乎全在读权重，拆成两个 micro-batch 后每个都要重新读一遍权重，读权重的时间翻倍，抵消了重叠的收益。
    5. decode 每个 token 只经过每个专家一次、计算很少，要把全局的 batch 汇集起来，让每个专家分到足够的 token；同时把显存留给 KV（每卡只放少量专家）。prefill 的 token 本来就多，小规模的 EP 就能让专家吃饱，节点少、跨机流量也少。

## EPLB 与路由约束：分层还是全局

DeepSeek-V3 的 256 个专家分成 8 组，路由分两步：先按每组最高的 2 个分数之和选出 4 个组，再在这 4 个组里选 8 个专家（**分组限制路由**）。如果每个组的专家都放在同一个节点上，一个 token 最多只会发往 4 个节点，跨节点流量有了上限。

EPLB 的开源实现（DeepSeek 的 `eplb.py`，SGLang 原样收录在 `srt/eplb/eplb_algorithms/deepseek.py`）有两种策略：

- **分层**：① 把专家组按负载均衡地打包到节点上（每个节点分到相同数量的组）；② 在每个节点内部，把最热的专家复制几份（冗余专家）；③ 把节点内的物理专家均衡地打包到各张卡上。专家组始终不跨节点；
- **全局**：不看节点，在全部专家上复制、在全部卡上打包。

用这三步的算法（与开源实现相同的贪心规则）模拟两种规模：

```python
import numpy as np

rng = np.random.default_rng(0)
E, GROUPS, TOPK_GROUP, TOPK = 256, 8, 4, 8                 # DeepSeek-V3：256 个专家分 8 组，每个 token 先选 4 组、再在组内选 8 个专家
GSIZE = E // GROUPS

bias = rng.normal(0, 1.0, E)                               # 专家的冷热差异
scores = rng.random((20000, E)) + 0.15 * bias
grp = np.sort(scores.reshape(-1, GROUPS, GSIZE), -1)[..., -2:].sum(-1)      # 组得分：组内最高的 2 个分数之和
keep = np.argsort(-grp, 1)[:, :TOPK_GROUP]
mask = np.full((len(scores), GROUPS), -np.inf)
np.put_along_axis(mask, keep, 0.0, 1)
routed = np.argsort(-(scores + np.repeat(mask, GSIZE, 1)), 1)[:, :TOPK]      # 每个 token 选中的 8 个专家
load = np.bincount(routed.ravel(), minlength=E).astype(float)


def pack(weights, n_packs):
    """balanced_packing：从重到轻，每个物品放进"还有空位且当前最轻"的包，每包物品数相同"""
    per = len(weights) // n_packs
    packs, sums = [[] for _ in range(n_packs)], [0.0] * n_packs
    for i in np.argsort(-np.asarray(weights), kind="stable"):
        p = min((j for j in range(n_packs) if len(packs[j]) < per), key=sums.__getitem__)
        packs[p].append(i)
        sums[p] += weights[i]
    return packs


def replicate(ids, n_phys):
    """replicate_experts：反复给"单个副本负载最大"的专家再加一个副本"""
    cnt = {e: 1 for e in ids}
    for _ in range(n_phys - len(ids)):
        e = max(ids, key=lambda x: load[x] / cnt[x])
        cnt[e] += 1
    return [e for e in ids for _ in range(cnt[e])], cnt


def place(groups_to_nodes, GPUS, PHYS):
    """返回每张卡上的物理专家列表；groups_to_nodes=None 时不分层（整个集群当成一个节点）"""
    node_ids = ([sum((list(range(g * GSIZE, (g + 1) * GSIZE)) for g in gs), []) for gs in groups_to_nodes]
                if groups_to_nodes else [list(range(E))])
    gpus = []
    for ids in node_ids:
        phys, cnt = replicate(ids, PHYS * len(ids) // E)
        w = [load[e] / cnt[e] for e in phys]
        gpus += [[phys[i] for i in p] for p in pack(w, GPUS * len(ids) // E)]
    return gpus


def report(name, gpus, NODES):
    cnt = {}
    for g in gpus:
        for e in g:
            cnt[e] = cnt.get(e, 0) + 1
    gpu_load = [sum(load[e] / cnt[e] for e in g) for g in gpus]
    where = {}                                              # 每个专家的副本在哪些节点上
    for i, g in enumerate(gpus):
        for e in g:
            where.setdefault(e, set()).add(i // (len(gpus) // NODES))
    touched = []
    for row in routed[:2000]:                               # 每个 token 要发往几个节点（副本优先选已经要去的节点）
        nodes = set()
        for e in row:
            if not (where[e] & nodes):
                nodes.add(min(where[e]))
        touched.append(len(nodes))
    print(f"  {name}：最忙的卡是平均的 {max(gpu_load) / np.mean(gpu_load):.2f} 倍，每个 token 平均发往 {np.mean(touched):.2f} 个节点")


groups_load = load.reshape(GROUPS, GSIZE).sum(1)
for NODES, GPUS, PHYS in ((4, 32, 288), (8, 64, 320)):     # 冗余 32 / 64 个槽位：每卡 9 / 5 个物理专家
    print(f"{NODES} 个节点、{GPUS} 张卡、{PHYS - E} 个冗余专家：")
    per = E // GPUS
    report("不做 EPLB（按编号放）", [list(range(i * per, (i + 1) * per)) for i in range(GPUS)], NODES)
    report("全局 EPLB（不看节点）", place(None, GPUS, PHYS), NODES)
    report("分层 EPLB（先把专家组打包到节点）", place(pack(groups_load, NODES), GPUS, PHYS), NODES)
```

```text title="输出"
4 个节点、32 张卡、32 个冗余专家：
  不做 EPLB（按编号放）：最忙的卡是平均的 2.87 倍，每个 token 平均发往 3.18 个节点
  全局 EPLB（不看节点）：最忙的卡是平均的 1.03 倍，每个 token 平均发往 3.63 个节点
  分层 EPLB（先把专家组打包到节点）：最忙的卡是平均的 1.09 倍，每个 token 平均发往 3.27 个节点
8 个节点、64 张卡、64 个冗余专家：
  不做 EPLB（按编号放）：最忙的卡是平均的 3.97 倍，每个 token 平均发往 3.96 个节点
  全局 EPLB（不看节点）：最忙的卡是平均的 1.17 倍，每个 token 平均发往 5.21 个节点
  分层 EPLB（先把专家组打包到节点）：最忙的卡是平均的 1.79 倍，每个 token 平均发往 3.96 个节点
```

- **4 个节点**（每个节点 2 个组）：分层 EPLB 把负载压到 1.09 倍，同时保住了"每个 token 最多 4 个节点"的上限；全局 EPLB 负载更平，但把同一组的专家拆散到各个节点，跨节点流量变多。这正是 DeepSeek-V3 prefill 的规模（4 个节点、32 张卡、每卡 8 个专家 + 1 个冗余）；
- **8 个节点**（每个节点只有 1 个组）：分层策略只能在组的粒度上均衡——哪个组热，那个节点就忙，最忙的卡是平均的 1.79 倍；全局策略负载好得多，代价是每个 token 多发 1.25 个节点。

开源实现的建议与此一致：节点数能整除组数、EP 规模较小时（prefill）用分层策略；EP 规模很大时（decode）用全局策略。

在 SGLang 里，`--ep-num-redundant-experts` 设置冗余专家数，`--init-expert-location` 指定初始放置（可以是离线统计好的专家分布），`--enable-eplb` 打开运行时的重新均衡：`srt/eplb/expert_distribution.py` 统计各专家的负载，`eplb_manager.py` 定期调用上面的算法，`expert_location_updater.py` 在卡之间搬运专家权重。

## DP Attention 的负载：最慢的 rank 决定速度

DP Attention 下，每个 rank 处理自己的一批请求。注意力的时间取决于这个 rank 上所有请求的 KV 总量；而 MoE 层的 dispatch 要等所有 rank 都算完注意力才能交换 token——**每一层都有一次隐式的全局同步**，最慢的 rank 决定所有 rank 的速度。上下文长度是长尾分布，一个 rank 分到几个超长请求，其他 rank 就要空等：

```python
import heapq

import numpy as np

rng = np.random.default_rng(1)
RANKS, PER_RANK, MAX_RUNNING = 32, 64, 128                # 32 个 DP rank，平均每个 rank 64 个请求，最多 128 个
ctx = np.minimum(rng.lognormal(np.log(3000), 1.0, RANKS * PER_RANK), 128000).astype(int)   # 各请求的上下文长度


def attn_time(kv_tokens):                                 # 一层注意力（µs）：读注意力权重 + 读这个 rank 所有请求的潜向量 KV
    return (187e6 + kv_tokens * 1152) / 3.35e12 * 1e6


def report(name, kv):
    t = [attn_time(x) for x in kv]
    print(f"{name}：注意力最慢的 rank {max(t):.0f} µs，平均 {np.mean(t):.0f} µs，其他 rank 平均空等 {max(t) - np.mean(t):.0f} µs")


report("轮流分配请求", [ctx[r::RANKS].sum() for r in range(RANKS)])
heap = [(0, r, 0) for r in range(RANKS)]                  # (KV token 总数, rank, 请求数)
for c in ctx:                                             # 按 KV 总量分配：交给当前 KV 最少、且请求数没到上限的 rank
    skipped = []
    while True:
        kv, r, n = heapq.heappop(heap)
        if n < MAX_RUNNING:
            break
        skipped.append((kv, r, n))
    heapq.heappush(heap, (kv + int(c), r, n + 1))
    for s in skipped:
        heapq.heappush(heap, s)
report("按 KV 总量分配", [kv for kv, _, _ in heap])
```

```text title="输出"
轮流分配请求：注意力最慢的 rank 243 µs，平均 165 µs，其他 rank 平均空等 78 µs
按 KV 总量分配：注意力最慢的 rank 183 µs，平均 165 µs，其他 rank 平均空等 18 µs
```

轮流分配时，每层有近三分之一的注意力时间在空等；按 KV 总量分配（分到长请求的 rank 少接几个请求）把空等压到原来的四分之一。SGLang 的 `--load-balance-method` 提供了 `round_robin`、`total_requests`、`total_tokens` 等策略。真实系统里请求不断地到达和结束，分配只能近似均衡，所以还有其他配套的做法：

- **空闲的 rank 也要参与每一层的 all-to-all**：某个 rank 暂时没有请求，也必须跑一个"空批次"陪其他 rank 走完每一层，否则其他 rank 的 dispatch 会一直等它；
- **CUDA Graph 的 batch 要对齐**：各 rank 的 batch 大小不同，录制的 CUDA Graph 按档位补齐，补齐的部分也是浪费；
- **prefill 和 decode 分开**（PD 分离）：否则一个 rank 在做长 prompt 的 prefill 时，其他 rank 的 decode 全部被拖住。

## 双 batch 重叠

decode 的一层里，dispatch 和 combine 的通信可能比计算还长。**双 batch 重叠**（two-batch overlap，TBO）把每个 rank 的 batch 拆成两个 micro-batch：一个在算注意力和专家时，另一个的 token 在网络上传输，两者交替。但拆分不是免费的：decode 的注意力和专家 GEMM 都受**权重读取**限制，拆成两半后每个 micro-batch 都要把权重重新读一遍。按 DeepSeek-V3 在 H800 上 decode 的一层估算：

```python
HBM, BF16, FP8, NIC = 3.35e12, 989e12, 1979e12, 50e9     # H800：显存带宽、BF16 / FP8 算力、每卡网卡带宽
ATTN_W = 187e6                                          # 一层 MLA 的权重（FP8 字节）：q/kv 的降维与升维、输出投影
EXPERT_W, EXPERT_FLOP = 44e6, 88e6                       # 一个专家的 FP8 权重字节数；一个 token 过一个专家的 FLOPs
LOCAL_EXPERTS, CTX = 3, 4096                             # 每卡 2 个路由专家 + 1 个共享专家；平均上下文长度
TOK_IN, TOK_OUT = 7168 + 224, 7168 * 2                   # dispatch 发 FP8（带缩放），combine 回 BF16


def layer(b):
    """一张卡、一层、b 个 token 的各段时间（秒）：计算取访存与算力中较慢的一个"""
    attn = max((ATTN_W + b * CTX * 1152) / HBM, b * CTX * 278528 / BF16)         # 读注意力权重 + 读潜向量 KV
    moe = max(LOCAL_EXPERTS * EXPERT_W / HBM, b * 9 * EXPERT_FLOP / FP8)          # 读专家权重；b×8 个路由 + b 个共享
    return attn, moe, b * 8 * TOK_IN / NIC, b * 8 * TOK_OUT / NIC                 # 最后两项：dispatch、combine


print(" 每卡 batch   不重叠（µs/层）   双 batch 重叠   加速   计算 : 通信")
for b in (32, 64, 128, 256):
    attn, moe, disp, comb = layer(b)
    serial = attn + moe + disp + comb
    ha, hm, hd, hc = layer(b // 2)                      # 拆成两个 micro-batch：各自重新读一遍权重
    compute, comm = 2 * (ha + hm), 2 * (hd + hc)
    tbo = max(compute, comm)                            # 理想情况：一个 micro-batch 计算时，另一个在通信
    print(f"{b:>9}   {serial * 1e6:>14.0f}   {tbo * 1e6:>12.0f}   {serial / tbo:>4.2f}   {compute / comm:>5.2f}")
```

```text title="输出"
 每卡 batch   不重叠（µs/层）   双 batch 重叠   加速   计算 : 通信
       32              252            236   1.07    2.12
       64              408            281   1.45    1.26
      128              732            445   1.65    0.83
      256             1409            890   1.58    0.65
```

- batch 很小时（32），计算几乎全是读权重，拆成两半后读权重的时间翻倍，抵消了重叠的收益；
- batch 在 64～128 之间、计算和通信差不多长时，收益最大（接近理想的 1.5～1.7 倍）；
- batch 更大时，通信成了主要部分，重叠只能藏住计算，加速比开始回落——这时要靠减少通信量（FP8 dispatch、节点受限路由）或者更快的网络。

模型里没有算上的：kernel 启动和同步的开销（CUDA Graph 可以消除大部分）、两个 micro-batch 同时运行时抢 SM 和显存带宽（DeepEP 低延迟模式用 hook 让通信不占 SM，正是为了这一点）。SGLang 用 `--enable-two-batch-overlap` 打开；DeepSeek 公开的推理系统在 decode 阶段用了更细的流水线，把注意力拆成两段，形成五级流水，进一步平衡每一级的时长。prefill 阶段同样用双 micro-batch：一个的注意力和专家计算，与另一个的 dispatch / combine 重叠。

## prefill 和 decode 用不同规模的 EP

DeepSeek-V3 论文中的部署：

| | prefill | decode |
| --- | --- | --- |
| 最小部署单元 | 4 个节点、32 张卡 | 40 个节点、320 张卡 |
| 注意力 | TP4 + 序列并行，DP8 | TP4 + 序列并行，DP80 |
| MoE | EP32，32 个冗余专家（每卡 8 + 1 个专家） | EP320，每卡 1 个专家，64 张卡放冗余专家和共享专家 |
| 通信 | 高吞吐 all-to-all（节点内 NVLink 转发） | 点对点直发（IBGDA），低延迟 |

规模差这么多，原因都能在前面几章找到：

- **decode 要让专家吃饱**：专家 GEMM 的算术强度等于每个专家分到的 token 数 × 2（FP8），每个专家每步需要几百个 token（[FP8 与分组 GEMM](fp8-gemm.md)）。只有几百张卡的 DP 注意力把 batch 汇集起来，才能做到；
- **decode 要省显存放 KV**：每卡只放 1 个专家，专家权重只占几十 GB 中的一小部分，剩下的显存全部给 KV Cache，每卡能跑更多的请求；
- **prefill 本来就算力受限**：一次 prefill 就有成千上万个 token，小规模的 EP 已经足够让专家吃饱；规模小、节点少，还能用分层 EPLB 限制跨节点流量；
- **PD 分离让两边独立扩展**：两侧的实例数、EP 规模、并行方式、通信模式各自选择。

几百张卡的 EP 组里，单卡故障和扩缩容会成为日常，怎样让一张卡坏了不拖垮整组，见[大规模 EP 的容错、弹性扩缩与排障](ep-elastic.md)。

!!! interview "面试怎么答"
    系统设计题问"部署一个 DeepSeek 规模的 MoE 模型"，这一章的内容可以组织成一个完整的回答：注意力 DP、MoE 大规模 EP，PD 分离后 prefill 小 EP（几十卡、分层 EPLB、高吞吐 all-to-all、双 micro-batch 重叠）、decode 大 EP（上百卡、全局 EPLB、低延迟 all-to-all、每卡少量专家、显存留给 KV）；DP 注意力按 KV 总量分配请求、空闲 rank 陪跑；双 batch 重叠在计算和通信相当时收益最大。每一点都能给出一个数字，是这类题的高分答法。

## 练习

**1. 冗余专家的显存。** DeepSeek-V3 有 58 个 MoE 层，每个专家的 FP8 权重约 44 MB。prefill 部署的 32 个冗余专家一共占多少显存？分摊到 32 张卡上每卡多少？

??? success "参考答案"
    每个冗余"专家"在每个 MoE 层各有一份：$32 \times 58 \times 44\,\text{MB} \approx 81.7$ GB，分到 32 张卡上每卡约 2.55 GB。相对 80 GB 的显存不算多，换来的是把最忙的卡从平均负载的近 3 倍压到 1.1 倍以内——对一层的延迟来说几乎是减半。

**2. 为什么 decode 用全局 EPLB？** decode 的 EP 很大（几百张卡、几十个节点），而 256 个专家只分 8 组。用分层策略会怎样？

??? success "参考答案"
    分层策略要求节点数整除组数，并且以组为单位分配到节点。几十个节点只有 8 个组，要么大部分节点分不到组，要么只能退化成全局策略。即使节点数恰好是 8，本章的模拟也显示：以组为粒度只能做到组级别的均衡，最忙的卡是平均的 1.79 倍。decode 每卡只放 1～2 个专家，负载均衡的粒度要细到单个专家，所以用全局策略，接受更多的跨节点流量（decode 的 batch 小，通信量本来就不大，而且低延迟模式本来就是点对点直发）。

**3. 双 batch 重叠的下限。** 根据本章的模型，写出双 batch 重叠有收益的条件（近似）。

??? success "参考答案"
    设整批的计算时间为 $C(b)$、通信时间为 $M(b)$，拆成两半后的计算为 $2C(b/2)$、通信约为 $M(b)$。重叠后的时间约为 $\max(2C(b/2), M(b))$，收益条件是它小于 $C(b) + M(b)$。当计算完全受权重读取限制时 $C(b/2) \approx C(b)$，条件变成 $\max(2C, M) < C + M$，即 $M > C$——通信必须比计算长，重叠才有意义。batch 较大、计算按 token 数线性增长时 $2C(b/2) \approx C(b)$，重叠后接近 $\max(C, M)$，收益最大。

## 小结

- [x] 分组限制路由让每个 token 最多发往 4 个节点；分层 EPLB（组 → 节点 → 节点内复制 → 卡）保住这个上限，适合节点少的 prefill；全局 EPLB 负载更平，适合 EP 很大的 decode。
- [x] DP Attention 每层都被 all-to-all 同步一次，最慢的 rank 决定速度；按 KV 总量分配请求、空闲 rank 陪跑空批次、PD 分离都是为此。
- [x] 双 batch 重叠要付出重读权重的代价，在计算与通信相当时收益最大（约 1.5 倍），batch 太小时几乎没用。
- [x] prefill 小 EP、decode 大 EP：decode 要汇集全局 batch 让专家吃饱、把显存留给 KV；PD 分离让两侧独立选择规模和通信模式。
