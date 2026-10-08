# MoE 与专家并行

<p class="lead">MoE 模型的总参数很大（几百 B 到上万亿），但每个 token 只经过少数几个专家。把专家分散到多张卡上就是专家并行（EP）：每张卡只放一部分专家，token 通过 all-to-all 被送到它选中的专家所在的卡上计算，再送回来。这一章在 4 个进程上实现一个带反向的专家并行 MoE 层，与单进程逐元素对齐；再看 MoE 训练特有的问题——负载均衡，以及 DeepSeek-V3 的无辅助损失均衡。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. 专家并行的一次前向有哪两次 all-to-all？各传什么？
    2. 为什么 all-to-all 之前要先交换一次"每个 rank 发多少个 token"？
    3. 负载不均会造成什么问题？辅助损失是怎么定义的？
    4. DeepSeek-V3 的"无辅助损失"均衡是怎么做的？
    5. EP 通常和哪些并行组合？注意力部分怎么办？

??? success "自测参考答案（先自己答，再展开对照）"
    1. dispatch：把每个 token 的隐藏向量发给它选中的专家所在的卡；combine：把专家算完的结果送回 token 原来所在的卡，再按门控权重加权求和。
    2. 每张卡发给别人的 token 数由路由决定、每次都不一样，接收方事先不知道要收多少，没法分配接收缓冲区、也没法确定 all-to-all 的切分。
    3. 少数专家过载，所在的卡成为整层的瓶颈（容量有限时还要丢 token），其余专家得不到训练。辅助损失 $L_{aux} = E \sum_e f_e P_e$：$f_e$ 是分给专家 e 的 token 比例，$P_e$ 是它的平均路由概率，完全均衡时为 1。
    4. 给每个专家一个偏置，只在选择 top-k 时加到分数上，不参与门控权重；每一步按负载调整：负载高于平均就把偏置调低一点，低于平均就调高。不需要额外的损失项，也就不干扰主目标的梯度。
    5. 注意力部分不用 EP：用数据并行（DP Attention）或张量并行；MoE 层用 EP，EP 组通常就是 DP 组的一部分。跨节点时限制每个 token 路由的节点数，并把 all-to-all 与计算重叠。

## 一次前向的流程

每张卡上有自己的一批 token 和 $E/P$ 个专家：

1. **路由**：每个 token 算出对所有专家的分数，选 top-k 个专家和对应的门控权重；
2. **排序**：把 (token, 专家) 对按目标专家排好，这样发给同一张卡的 token 在内存里是连续的；
3. **交换数量**：每张卡要发给别的卡多少个 token 各不相同，接收方事先不知道要收多少，所以先用一次小的 all-to-all 交换数量；
4. **dispatch**：all-to-all 把 token 送到专家所在的卡；
5. **专家计算**：每张卡上的每个专家处理分给它的 token（这就是"分组 GEMM"）；
6. **combine**：反方向的 all-to-all 把结果送回原来的卡，按门控权重加权求和。

![图：专家并行的一次前向——路由、排序、交换数量、dispatch、分组 GEMM、combine](../assets/figures/moe-flow.svg){.aig-svg}

```python title="moe_ep.py" torchrun="4"
import torch
import torch.distributed as dist
from torch.distributed.nn.functional import all_to_all_single   # 带 autograd 的版本：反向自动做反方向的 all-to-all

dist.init_process_group("gloo")
rank, P = dist.get_rank(), dist.get_world_size()
E, K, H, F, T = 8, 2, 16, 32, 12              # 专家数、top-k、隐藏维度、专家中间维度、每个 rank 的 token 数
LOCAL = E // P                                 # 每个 rank 放 E/P 个专家


def make_params():
    torch.manual_seed(0)
    return torch.randn(E, H) * 0.5, torch.randn(E, F, H) / H ** 0.5, torch.randn(E, H, F) / F ** 0.5


def expert(x, w1, w2):
    return torch.nn.functional.silu(x @ w1.T) @ w2.T


router, W1, W2 = make_params()
torch.manual_seed(1)
X_all = torch.randn(P * T, H)                  # 所有 rank 的 token，用来算单进程参照
X = X_all[rank * T:(rank + 1) * T]

# ---- 单进程参照：所有专家都在本地
ref_w1, ref_w2 = W1.clone().requires_grad_(), W2.clone().requires_grad_()
probs = torch.softmax(X_all @ router.T, dim=-1)
gate, idx = probs.topk(K, dim=-1)
ref = torch.zeros_like(X_all)
for e in range(E):
    tok, slot = (idx == e).nonzero(as_tuple=True)
    ref.index_add_(0, tok, gate[tok, slot, None] * expert(X_all[tok], ref_w1[e], ref_w2[e]))
ref.square().sum().backward()

# ---- 专家并行：rank r 持有专家 [r*LOCAL, (r+1)*LOCAL)
w1 = W1[rank * LOCAL:(rank + 1) * LOCAL].clone().requires_grad_()
w2 = W2[rank * LOCAL:(rank + 1) * LOCAL].clone().requires_grad_()
probs = torch.softmax(X @ router.T, dim=-1)
gate, idx = probs.topk(K, dim=-1)                          # [T, K]
flat_e = idx.flatten()                                     # 每个 (token, k) 要去的专家
order = flat_e.argsort(stable=True)                        # 按专家（也就是按目标 rank）排序
send_x = X.repeat_interleave(K, dim=0)[order]              # 每个 token 复制 K 份，按目标排好
send_counts = torch.bincount(flat_e // LOCAL, minlength=P) # 发给每个 rank 多少个
recv_counts = torch.empty_like(send_counts)
dist.all_to_all_single(recv_counts, send_counts)           # 先交换数量，才知道要收多少
recv_x = all_to_all_single(torch.empty(int(recv_counts.sum()), H), send_x,
                           recv_counts.tolist(), send_counts.tolist())          # dispatch
send_e = flat_e[order] % LOCAL                             # 本地专家编号也一起发过去
recv_e = torch.empty(int(recv_counts.sum()), dtype=torch.long)
dist.all_to_all_single(recv_e, send_e, recv_counts.tolist(), send_counts.tolist())

recv_y = torch.zeros_like(recv_x)
for e in range(LOCAL):                                     # 本地专家各算各的 token
    m = recv_e == e
    recv_y = recv_y.index_put((m.nonzero(as_tuple=True)[0],), expert(recv_x[m], w1[e], w2[e]))
back = all_to_all_single(torch.empty_like(send_x), recv_y, send_counts.tolist(), recv_counts.tolist())   # combine
y_sorted = torch.empty_like(back).index_copy(0, order, back)                    # 还原成 (token, k) 的原始顺序
out = (y_sorted.view(T, K, H) * gate[..., None]).sum(dim=1)
out.square().sum().backward()

ok_out = torch.allclose(out, ref.detach()[rank * T:(rank + 1) * T], atol=1e-5)
ok_grad = torch.allclose(w1.grad, ref_w1.grad[rank * LOCAL:(rank + 1) * LOCAL], atol=1e-4) and \
    torch.allclose(w2.grad, ref_w2.grad[rank * LOCAL:(rank + 1) * LOCAL], atol=1e-4)
flags = torch.tensor([int(ok_out), int(ok_grad)])
dist.all_reduce(flags, op=dist.ReduceOp.MIN)
loads = [None] * P
dist.all_gather_object(loads, int(recv_counts.sum()))
if rank == 0:
    print(f"{P} 个 rank、{E} 个专家、top-{K}：每个 rank 收到的 token 数 {loads}")
    print("输出与单进程一致：", bool(flags[0]), "；本地专家的权重梯度一致：", bool(flags[1]))
dist.destroy_process_group()
```

```text title="输出"
4 个 rank、8 个专家、top-2：每个 rank 收到的 token 数 [19, 28, 27, 22]
输出与单进程一致： True ；本地专家的权重梯度一致： True
```

- `torch.distributed.nn.functional.all_to_all_single` 是带 autograd 的版本：反向时自动做一次反方向的 all-to-all，把输出的梯度送回专家所在的卡，所以专家的权重梯度不需要任何额外的通信——每个专家只在一张卡上；
- 最后一行显示了**负载不均**：每张卡收到的 token 数不同（19 到 28），最忙的那张卡决定了整层的时间；
- 通信量：每个 token 的隐藏向量发出去 $k$ 次、收回来 $k$ 次，每层前向约 $2k \cdot sbh$ 个元素，和 EP 的度数几乎无关；DeepEP 这类库会把 dispatch 用 FP8 传、combine 用 BF16 传，并把跨节点的部分和节点内的 NVLink 转发分开处理。

## 负载均衡

路由器是学出来的，很容易"偏爱"少数几个专家：它们被训练得更多、变得更好、于是更被偏爱。后果有两个：少数专家过载，所在的卡成为瓶颈（甚至要丢弃超过容量的 token）；多数专家几乎不被训练，浪费参数。

**辅助损失**（Switch Transformer、GShard）：在训练损失里加一项

$$L_{\text{aux}} = E \sum_{e=1}^{E} f_e \, P_e$$

$f_e$ 是分给专家 $e$ 的 token 比例，$P_e$ 是所有 token 对专家 $e$ 的平均路由概率。完全均衡时它等于 1，越不均衡越大。它的问题是和主要的训练目标"抢梯度"：系数太小不起作用，太大会损害模型质量。

**无辅助损失的均衡**（DeepSeek-V3）：给每个专家一个偏置 $b_e$，只在**选择** top-k 时加到分数上，不参与门控权重的计算；每一步根据负载调整偏置——负载高于平均就调低一点，低于平均就调高一点。不需要额外的损失项，也就不干扰主目标的梯度：

```python title="balance_bias.py"
import torch

torch.manual_seed(0)
E, K, T = 16, 2, 4096
skew = torch.linspace(1.5, -1.5, E)                      # 路由器天生偏爱前面几个专家
bias = torch.zeros(E)                                    # 只用于"选哪几个专家"，不参与门控权重
u = 0.02                                                 # 偏置的更新步长


def route(scores):
    _, idx = (scores + bias).topk(K, dim=-1)             # 选择时加偏置
    gate = torch.softmax(scores, -1).gather(-1, idx)     # 门控权重仍然用原始分数
    return idx, gate


for step in range(301):
    scores = torch.randn(T, E) + skew
    idx, gate = route(scores)
    load = torch.bincount(idx.flatten(), minlength=E).float()
    if step % 100 == 0:
        f = load / (T * K)                               # 每个专家分到的 token 比例
        p = torch.softmax(scores, -1).mean(0)            # 每个专家的平均路由概率
        aux = E * (f * p).sum().item()                   # Switch Transformer 的负载均衡损失（均衡时为 1）
        print(f"step {step:>3}：最忙 / 最闲专家的负载比 {load.max() / load.min():6.2f}，辅助损失 {aux:.3f}")
    bias += u * torch.sign(load.mean() - load)           # 负载高于平均就调低偏置，反之调高
```

```text title="输出"
step   0：最忙 / 最闲专家的负载比 635.33，辅助损失 1.928
step 100：最忙 / 最闲专家的负载比   1.24，辅助损失 1.001
step 200：最忙 / 最闲专家的负载比   1.22，辅助损失 1.016
step 300：最忙 / 最闲专家的负载比   1.19，辅助损失 1.008
```

一开始最忙的专家分到的 token 是最闲的 600 多倍，偏置调整一百步之后就接近均衡了。DeepSeek-V3 在这之外还保留了一个很小的**序列级**辅助损失，防止单条序列内部极端不均；推理时，EPLB（专家并行负载均衡器）再根据实际负载复制热门专家、重新摆放专家的位置。

## 与其他并行的组合

- **注意力部分不用 EP**：注意力是稠密的，通常用数据并行（每张卡处理不同的请求 / 样本）或张量并行；MoE 层再用 EP。训练时常见的组合是"注意力 TP + MoE 层 EP"或者"注意力 DP + MoE 层 EP"（推理时的 DP Attention 就是后者，见推理系统手册的[专家并行与 DP Attention](serving://distributed/expert-parallel/)）；
- **EP 与 DP 共用卡**：EP 组通常就是 DP 组的一部分——同一批卡在注意力部分做数据并行，在 MoE 部分做专家并行；
- **专家的 TP**：专家本身很大时（早期的 MoE），可以再对专家做张量并行；现代的细粒度 MoE（DeepSeek-V3 每个专家的中间维度只有 2048）通常不需要。
- **跨节点**：EP 的 all-to-all 跨节点时很贵，DeepSeek-V3 用"每个 token 最多路由到 4 个节点"的限制，加上 DualPipe 把 all-to-all 和计算重叠起来。

!!! interview "怎么讲清楚"
    讲专家并行：路由 → 按目标排序 → 先交换"每个 rank 发多少"（接收方要据此分配缓冲区）→ dispatch（all-to-all）→ 专家计算 → combine（反向的 all-to-all）→ 按门控权重加权求和；通信量约 $2k \cdot sbh$，与 EP 度数几乎无关。负载不均时最忙的卡决定整层的时间：辅助损失 $E \sum_e f_e P_e$ 会和主目标抢梯度，DeepSeek-V3 改用只影响"选哪几个专家"的偏置动态均衡。组合方式：注意力用 DP 或 TP，MoE 层用 EP；跨节点时限制每个 token 去的节点数，并把 all-to-all 和计算重叠。

## 练习

1. 本章的实现里，一个 token 的两个专家恰好在同一张卡上时，它的隐藏向量被发了两次。怎样只发一次？能省多少通信？

??? success "参考答案"
    按"目标 rank"而不是"目标专家"去重：每个 token 对每个目标 rank 只发一份隐藏向量，同时附带"它在这个 rank 上要去哪几个专家"的信息，接收方在本地复制给各个专家；combine 时在发送方先把同一个 rank 上几个专家的结果加权相加，再发回。
    省下的比例取决于 top-k 专家落在同一张卡上的概率；专家数多、EP 度数小时更明显。DeepEP 在节点层面做同样的去重：一个 token 发往一个节点只发一次，到节点内再用 NVLink 转发。

2. 为什么无辅助损失均衡里，偏置只用于"选择"，而门控权重仍然用原始分数？

??? success "参考答案"
    偏置的作用是调整"去哪个专家"的决策来均衡负载，不应该改变"专家的输出按多大权重混合"。如果门控权重也加上偏置，偏置就会直接影响模型的输出和梯度，相当于把一个非学习得来的量混进了前向计算，干扰训练；只用于选择时，它对损失没有直接贡献，模型的输出仍然只由学到的路由分数决定。

## 小结

- [x] 专家并行：路由 → 按目标排序 → 交换数量 → dispatch（all-to-all）→ 专家计算 → combine（反向 all-to-all）→ 加权求和。
- [x] 带 autograd 的 all-to-all 让反向自动完成；专家的权重梯度不需要额外通信。
- [x] 通信量约 $2k \cdot sbh$、与 EP 度数几乎无关；负载不均时最忙的卡决定整层时间。
- [x] 辅助损失用 $E \sum f_e P_e$ 惩罚不均衡，但会干扰主目标；DeepSeek-V3 用只影响选择的偏置动态均衡负载。
- [x] 注意力用 DP 或 TP，MoE 层用 EP；跨节点时限制路由的节点数并把 all-to-all 与计算重叠。
