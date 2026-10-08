# Mixture of experts and expert parallelism

<p class="lead">A mixture-of-experts model has an enormous total parameter count (hundreds of billions to trillions), but each token passes through only a few experts. Spreading the experts across cards is expert parallelism: each card holds some of the experts, and tokens are sent by an all-to-all to the card holding the expert they chose, computed there and sent back. This chapter implements an expert-parallel mixture-of-experts layer with a backward pass across 4 processes, lined up elementwise against a single process; then it looks at the problem particular to training a mixture of experts, load balancing, and at DeepSeek-V3's auxiliary-loss-free balancing.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. Which two all-to-alls does one expert-parallel forward pass have? What does each carry?
    2. Why does the all-to-all have to be preceded by an exchange of how many tokens each rank is sending?
    3. What does an uneven load cause? How is the auxiliary loss defined?
    4. How does DeepSeek-V3's auxiliary-loss-free balancing work?
    5. Which kinds of parallelism is expert parallelism usually combined with? What happens to the attention part?

??? success "Answers for the self-test (answer first, then open this)"
    1. Dispatch: send each token's hidden vector to the card holding the expert it chose. Combine: send the experts' results back to the card the token came from, where they are weighted by the gate and summed.
    2. How many tokens each card sends to each other card is decided by the routing and differs every time; the receiver does not know in advance how many to expect, so it cannot allocate a receive buffer or determine the all-to-all's split.
    3. A few experts are overloaded and the cards holding them become the whole layer's bottleneck (and tokens have to be dropped when there is a capacity limit), while the rest get no training. The auxiliary loss is $L_{aux} = E \sum_e f_e P_e$, where $f_e$ is the fraction of tokens sent to expert e and $P_e$ its average routing probability, which equals 1 under perfect balance.
    4. Give each expert a bias that is added to the score only when choosing the top-k, taking no part in the gate weights; adjust it every step by the load, lowering it a little when the load is above average and raising it when below. No extra loss term is needed, so nothing interferes with the main objective's gradients.
    5. The attention part does not use expert parallelism: it uses data parallelism (DP attention) or tensor parallelism, and the mixture-of-experts layer uses expert parallelism, with the expert-parallel group usually part of the data-parallel group. Across nodes, cap how many nodes each token routes to and overlap the all-to-all with the computation.

## The flow of one forward pass {#一次前向的流程}

Each card has its own batch of tokens and $E/P$ experts:

1. **Routing**: each token scores all of the experts and picks the top-k and their gate weights.
2. **Sorting**: sort the (token, expert) pairs by destination expert, so that the tokens going to one card are contiguous in memory.
3. **Exchanging the counts**: how many tokens each card sends to each other card differs, and the receiver does not know in advance, so a small all-to-all exchanges the counts first.
4. **Dispatch**: an all-to-all sends the tokens to the cards holding their experts.
5. **The expert computation**: each expert on each card handles the tokens assigned to it (this is the grouped GEMM).
6. **Combine**: an all-to-all in the other direction sends the results back to the original cards, where they are weighted by the gate and summed.

![Figure: one expert-parallel forward pass, with routing, sorting, exchanging the counts, dispatch, the grouped GEMM and combine](../assets/figures/moe-flow.svg){.aig-svg}

```python title="moe_ep.py" torchrun="4"
import torch
import torch.distributed as dist
from torch.distributed.nn.functional import all_to_all_single   # the autograd-aware version: the backward pass does an all-to-all in the other direction automatically

dist.init_process_group("gloo")
rank, P = dist.get_rank(), dist.get_world_size()
E, K, H, F, T = 8, 2, 16, 32, 12              # the expert count, top-k, the hidden dimension, the experts' intermediate dimension, the tokens per rank
LOCAL = E // P                                 # each rank holds E/P experts


def make_params():
    torch.manual_seed(0)
    return torch.randn(E, H) * 0.5, torch.randn(E, F, H) / H ** 0.5, torch.randn(E, H, F) / F ** 0.5


def expert(x, w1, w2):
    return torch.nn.functional.silu(x @ w1.T) @ w2.T


router, W1, W2 = make_params()
torch.manual_seed(1)
X_all = torch.randn(P * T, H)                  # every rank's tokens, used for the single-process reference
X = X_all[rank * T:(rank + 1) * T]

# ---- the single-process reference: every expert is local
ref_w1, ref_w2 = W1.clone().requires_grad_(), W2.clone().requires_grad_()
probs = torch.softmax(X_all @ router.T, dim=-1)
gate, idx = probs.topk(K, dim=-1)
ref = torch.zeros_like(X_all)
for e in range(E):
    tok, slot = (idx == e).nonzero(as_tuple=True)
    ref.index_add_(0, tok, gate[tok, slot, None] * expert(X_all[tok], ref_w1[e], ref_w2[e]))
ref.square().sum().backward()

# ---- expert parallelism: rank r holds experts [r*LOCAL, (r+1)*LOCAL)
w1 = W1[rank * LOCAL:(rank + 1) * LOCAL].clone().requires_grad_()
w2 = W2[rank * LOCAL:(rank + 1) * LOCAL].clone().requires_grad_()
probs = torch.softmax(X @ router.T, dim=-1)
gate, idx = probs.topk(K, dim=-1)                          # [T, K]
flat_e = idx.flatten()                                     # the expert each (token, k) pair goes to
order = flat_e.argsort(stable=True)                        # sorted by expert, which is to say by destination rank
send_x = X.repeat_interleave(K, dim=0)[order]              # each token replicated K times and sorted by destination
send_counts = torch.bincount(flat_e // LOCAL, minlength=P) # how many go to each rank
recv_counts = torch.empty_like(send_counts)
dist.all_to_all_single(recv_counts, send_counts)           # exchange the counts first, so you know how many to receive
recv_x = all_to_all_single(torch.empty(int(recv_counts.sum()), H), send_x,
                           recv_counts.tolist(), send_counts.tolist())          # dispatch
send_e = flat_e[order] % LOCAL                             # the local expert index is sent along too
recv_e = torch.empty(int(recv_counts.sum()), dtype=torch.long)
dist.all_to_all_single(recv_e, send_e, recv_counts.tolist(), send_counts.tolist())

recv_y = torch.zeros_like(recv_x)
for e in range(LOCAL):                                     # each local expert computes its own tokens
    m = recv_e == e
    recv_y = recv_y.index_put((m.nonzero(as_tuple=True)[0],), expert(recv_x[m], w1[e], w2[e]))
back = all_to_all_single(torch.empty_like(send_x), recv_y, send_counts.tolist(), recv_counts.tolist())   # combine
y_sorted = torch.empty_like(back).index_copy(0, order, back)                    # restore the original (token, k) order
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

```text title="output"
4 个 rank、8 个专家、top-2：每个 rank 收到的 token 数 [19, 28, 27, 22]
输出与单进程一致： True ；本地专家的权重梯度一致： True
```

- `torch.distributed.nn.functional.all_to_all_single` is the autograd-aware version: the backward pass automatically does an all-to-all in the other direction to return the output's gradients to the cards holding the experts, so the experts' weight gradients need no extra communication, since each expert lives on exactly one card.
- The last line shows **the uneven load**: the cards receive different numbers of tokens (19 to 28), and the busiest sets the whole layer's time.
- The volume: each token's hidden vector goes out $k$ times and comes back $k$ times, about $2k \cdot sbh$ elements per layer forward, essentially independent of the expert-parallel degree. Libraries like DeepEP send the dispatch in FP8 and the combine in BF16, and handle the cross-node part separately from NVLink forwarding within a node.

## Load balancing {#负载均衡}

The router is learned, and it easily comes to favour a few experts: they are trained more, get better, and are favoured more. Two consequences: a few experts are overloaded and the cards holding them become the bottleneck (and tokens beyond the capacity may be dropped); most experts are barely trained, wasting parameters.

**The auxiliary loss** (Switch Transformer, GShard) adds a term to the training loss:

$$L_{\text{aux}} = E \sum_{e=1}^{E} f_e \, P_e$$

where $f_e$ is the fraction of tokens sent to expert $e$ and $P_e$ is the average routing probability of all tokens for expert $e$. It equals 1 under perfect balance and grows as the imbalance does. Its problem is that it competes with the main training objective for gradients: too small a coefficient does nothing and too large a one damages the model's quality.

**Auxiliary-loss-free balancing** (DeepSeek-V3): give each expert a bias $b_e$ that is added to the score only when **choosing** the top-k, taking no part in computing the gate weights; adjust the bias every step by the load, lowering it a little when the load is above average and raising it when below. No extra loss term is needed, so nothing interferes with the main objective's gradients:

```python title="balance_bias.py"
import torch

torch.manual_seed(0)
E, K, T = 16, 2, 4096
skew = torch.linspace(1.5, -1.5, E)                      # the router inherently favours the first few experts
bias = torch.zeros(E)                                    # used only to choose which experts, taking no part in the gate weights
u = 0.02                                                 # the bias's update step


def route(scores):
    _, idx = (scores + bias).topk(K, dim=-1)             # the bias is added when choosing
    gate = torch.softmax(scores, -1).gather(-1, idx)     # the gate weights still use the raw scores
    return idx, gate


for step in range(301):
    scores = torch.randn(T, E) + skew
    idx, gate = route(scores)
    load = torch.bincount(idx.flatten(), minlength=E).float()
    if step % 100 == 0:
        f = load / (T * K)                               # the fraction of tokens each expert receives
        p = torch.softmax(scores, -1).mean(0)            # each expert's average routing probability
        aux = E * (f * p).sum().item()                   # Switch Transformer's load-balancing loss (1 when balanced)
        print(f"step {step:>3}：最忙 / 最闲专家的负载比 {load.max() / load.min():6.2f}，辅助损失 {aux:.3f}")
    bias += u * torch.sign(load.mean() - load)           # lower the bias when the load is above average and raise it when below
```

```text title="output"
step   0：最忙 / 最闲专家的负载比 635.33，辅助损失 1.928
step 100：最忙 / 最闲专家的负载比   1.24，辅助损失 1.001
step 200：最忙 / 最闲专家的负载比   1.22，辅助损失 1.016
step 300：最忙 / 最闲专家的负载比   1.19，辅助损失 1.008
```

At the start the busiest expert gets over 600 times the tokens of the idlest, and a hundred steps of bias adjustment bring it close to balance. Beyond this, DeepSeek-V3 keeps a very small **sequence-level** auxiliary loss to prevent extreme imbalance within one sequence; at inference time the expert-parallel load balancer replicates popular experts and rearranges where the experts sit according to the actual load.

## Combining with the other kinds of parallelism {#与其他并行的组合}

- **The attention part does not use expert parallelism**: attention is dense and usually uses data parallelism (each card handling different requests or samples) or tensor parallelism, with expert parallelism only in the mixture-of-experts layers. The common combinations in training are tensor-parallel attention with expert-parallel mixture-of-experts layers, or data-parallel attention with expert-parallel layers (DP attention at inference time is the latter, see [Expert parallelism and DP attention](serving://distributed/expert-parallel/) in the inference-systems handbook).
- **Expert and data parallelism share cards**: the expert-parallel group is usually part of the data-parallel group, so the same cards do data parallelism for attention and expert parallelism for the mixture-of-experts layers.
- **Tensor parallelism over the experts**: when the experts themselves are large (as in early mixtures of experts), they can be tensor-parallel too; modern fine-grained mixtures (DeepSeek-V3's experts have an intermediate dimension of only 2048) usually do not need it.
- **Across nodes**: expert parallelism's all-to-all is expensive across nodes, and DeepSeek-V3 caps each token at 4 nodes and uses DualPipe to overlap the all-to-all with the computation.

!!! interview "How to explain it"
    On expert parallelism: route, sort by destination, exchange how many each rank is sending (the receiver needs it to allocate buffers), dispatch (an all-to-all), compute the experts, combine (an all-to-all the other way), then weight by the gate and sum; the volume is about $2k \cdot sbh$ and is essentially independent of the expert-parallel degree. When the load is uneven the busiest card sets the whole layer's time: the auxiliary loss $E \sum_e f_e P_e$ competes with the main objective for gradients, and DeepSeek-V3 instead balances dynamically with a bias that affects only which experts are chosen. The combinations: data or tensor parallelism for attention and expert parallelism for the mixture-of-experts layers; across nodes, cap how many nodes each token goes to and overlap the all-to-all with the computation.

## Exercises {#练习}

1. In this chapter's implementation, when a token's two experts happen to be on the same card, its hidden vector is sent twice. How would you send it once? How much communication does that save?

??? success "Answer"
    Deduplicate by destination rank rather than by destination expert: send one copy of each token's hidden vector per destination rank, carrying alongside it which experts on that rank it is going to, and have the receiver replicate it locally to each expert; on the combine, weight and sum the results of the several experts on one rank at the sender before sending back.
    How much that saves depends on the probability of the top-k experts landing on one card, and it is more noticeable with many experts and a small expert-parallel degree. DeepEP deduplicates the same way at the node level: a token goes to a node once and is forwarded over NVLink within it.

2. In auxiliary-loss-free balancing, why is the bias used only for choosing while the gate weights still use the raw scores?

??? success "Answer"
    The bias exists to adjust which expert a token goes to in order to balance the load, and it should not change the weight with which the experts' outputs are mixed. If the gate weights carried the bias too, it would directly affect the model's output and gradients, mixing an unlearned quantity into the forward computation and interfering with training; used only for choosing, it contributes nothing to the loss directly and the model's output still depends only on the learned routing scores.

## Summary {#小结}

- [x] Expert parallelism: route, sort by destination, exchange the counts, dispatch (an all-to-all), compute the experts, combine (an all-to-all the other way), weight and sum.
- [x] An autograd-aware all-to-all makes the backward pass automatic; the experts' weight gradients need no extra communication.
- [x] The volume is about $2k \cdot sbh$ and is essentially independent of the expert-parallel degree; when the load is uneven the busiest card sets the whole layer's time.
- [x] The auxiliary loss penalises imbalance with $E \sum f_e P_e$ but interferes with the main objective; DeepSeek-V3 balances the load dynamically with a bias that affects only the choice.
- [x] Attention uses data or tensor parallelism and the mixture-of-experts layers use expert parallelism; across nodes, cap how many nodes a token routes to and overlap the all-to-all with the computation.
