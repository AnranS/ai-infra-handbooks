# Mixture of experts (MoE)

<p class="lead">A mixture of experts replaces one large feed-forward network with many small "experts", and each token activates only a few of them. The model's total parameters can then be very large (a large knowledge capacity) while the compute per token stays small. DeepSeek-V3, the MoE versions of Qwen3, Mixtral and gpt-oss are all MoE models. MoE brings entirely new problems to inference: memory must hold every expert while computation uses only a small part of them, and the experts need load balancing and cross-GPU communication.</p>

!!! question "Self-test: if you can answer these, skip this chapter"
    1. What parts make up an MoE layer? What happens when a token passes through one?
    2. What are "total parameters" and "active parameters"? What are they for DeepSeek-V3?
    3. Why is load balancing needed? What are the ways to do it?
    4. When implementing MoE, why group the tokens by expert instead of computing token by token?
    5. Why is an MoE model's decode harder to optimize than a dense model with the same number of active parameters?

??? success "Answers (try first, then expand to compare)"
    1. A router (a linear layer that scores each expert) + top-k selection + several small FFN experts + a weighted combination by the gating weights (some models also have shared experts). A token gets its scores, picks k experts, is sent through each of them, and their results are summed with weights.
    2. Total parameters are all the experts combined and decide memory; active parameters are the ones a token actually passes through and decide compute. DeepSeek-V3 has 671B / 37B.
    3. Routing tends to favor a few experts more and more: they get overloaded and become bottlenecks (and may drop tokens), while the other experts get no training. Remedies: an auxiliary loss, a capacity limit, and DeepSeek-V3's auxiliary-loss-free bias (added only when selecting experts and adjusted by load).
    4. Computing token by token is a pile of tiny matrix multiplications that cannot fill a GPU; after grouping by expert (sort, count, segment), each expert does one large matrix multiplication over all the tokens assigned to it, and the results are scattered back in the original order. This is the skeleton of the fused MoE kernel.
    5. In decode the tokens of a batch are spread over many experts, so each expert gets only a few tokens yet must read its full weights, and far more bytes are read than for a dense model with the same active parameters; with experts spread over several GPUs there is also all-to-all communication and load imbalance to handle.

<!-- comic ../assets/comics/moe.webp is in Chinese; put it back once the English version exists -->

## Structure {#结构}

An MoE layer replaces the FFN in a Transformer layer:

1. **Router (gate)**: a linear layer `d → E` that scores the E experts for each token;
2. **Selection**: each token picks the k experts with the highest scores (top-k, commonly k = 2 to 8);
3. **Computation**: each selected expert (a small SwiGLU FFN) processes the token;
4. **Weighted combination**: the outputs of the k experts are summed with the routing weights.

$$
y = \sum_{i \in \text{TopK}(g(x))} w_i(x)\, \text{Expert}_i(x)
$$

DeepSeekMoE made two improvements on this, adopted by many later models:

- **Fine-grained experts**: cut the experts smaller and use more of them (DeepSeek-V3 has 256 routed experts and each token picks 8), giving richer combinations;
- **Shared experts**: add 1 expert that every token passes through, holding general knowledge, so the routed experts can specialize in their own domains.

![Figure: an MoE layer; the router scores each token and sends it only to the k highest-scoring experts, then sums their outputs with weights](../assets/figures/moe-structure.svg){.aig-svg}

## Implementation: token by token vs. grouped by expert {#实现逐-token-与按专家分组}

The most intuitive implementation loops over each token's selected experts. That is mathematically correct but extremely inefficient on a GPU: each expert processes one token at a time, so everything is a matrix-vector product. Real implementations **group by expert**: first find which tokens each expert handles, have each expert do one matrix multiplication over all its tokens, then scatter the results back to their positions with the weights. The two give the same result:

```python title="moe.py"
"""moe.py —— 一个 top-k 路由的 MoE 层（带共享专家），以及两种等价的前向实现。"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    def __init__(self, d, d_ff):
        super().__init__()
        self.gate_proj = nn.Linear(d, d_ff, bias=False)
        self.up_proj = nn.Linear(d, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MoE(nn.Module):
    def __init__(self, d, d_ff, n_experts, top_k, n_shared=1):
        super().__init__()
        self.top_k = top_k
        self.router = nn.Linear(d, n_experts, bias=False)
        self.experts = nn.ModuleList(Expert(d, d_ff) for _ in range(n_experts))
        self.shared = nn.ModuleList(Expert(d, d_ff) for _ in range(n_shared))

    def route(self, x):                                   # x: [N, d], N tokens
        probs = self.router(x).softmax(dim=-1)            # [N, E]
        weights, idx = probs.topk(self.top_k, dim=-1)     # [N, k]
        weights = weights / weights.sum(-1, keepdim=True) # renormalize the k selected weights
        return weights, idx, probs

    def forward_naive(self, x):
        """逐个 token、逐个专家地计算：正确但低效。"""
        weights, idx, _ = self.route(x)
        out = torch.zeros_like(x)
        for t in range(x.shape[0]):
            for j in range(self.top_k):
                e = idx[t, j].item()
                out[t] += weights[t, j] * self.experts[e](x[t])
        return out + sum(s(x) for s in self.shared)

    def forward_grouped(self, x):
        """按专家分组：每个专家对分给它的全部 token 做一次矩阵乘法。"""
        weights, idx, _ = self.route(x)
        flat_expert = idx.flatten()                                   # [N*k]: which expert each (token, slot) goes to
        flat_token = torch.arange(x.shape[0]).repeat_interleave(self.top_k)
        order = flat_expert.argsort(stable=True)                      # sort by expert, so each expert's tokens are contiguous
        counts = torch.bincount(flat_expert, minlength=len(self.experts))
        out = torch.zeros_like(x)
        start = 0
        for e, n in enumerate(counts.tolist()):
            if n == 0:
                continue
            sel = order[start:start + n]
            tok = flat_token[sel]
            y = self.experts[e](x[tok])                               # process n tokens at once
            out.index_add_(0, tok, y * weights.flatten()[sel, None])  # add back to the original positions, weighted
            start += n
        return out + sum(s(x) for s in self.shared)
```

```python
import torch
from moe import MoE

torch.manual_seed(0)
moe = MoE(d=64, d_ff=128, n_experts=16, top_k=4)
x = torch.randn(50, 64)
with torch.no_grad():
    a, b = moe.forward_naive(x), moe.forward_grouped(x)
assert torch.allclose(a, b, atol=1e-5)

_, idx, _ = moe.route(x)
counts = torch.bincount(idx.flatten(), minlength=16)
print("每个专家分到的 token 数:", counts.tolist())
```

The "sort → count per expert → compute by segment → scatter back with weights" in `forward_grouped` is exactly the skeleton of the **fused MoE** kernel in inference engines: the count is a histogram, each expert's start position is a prefix sum, and one grouped GEMM computes all the experts at once; see [prefix sums and MoE dispatch](cuda://kernels/scan/#在大模型里的应用moe-的-token-分发) in the CUDA book.

## Load balancing {#负载均衡}

If the router always favors a few experts, those experts get overloaded, the others learn nothing, and parameters are wasted. So training must encourage load balancing:

- **Auxiliary loss** (Switch Transformer): $\mathcal{L}_{aux} = E \sum_{i=1}^{E} f_i P_i$, where $f_i$ is the share of tokens assigned to expert i and $P_i$ is the router's mean probability for expert i. It is minimized when everything is uniform;
- **Capacity limit**: in training, each expert gets a capacity cap, and tokens beyond it are dropped (they just take the residual path);
- **Auxiliary-loss-free balancing** (DeepSeek-V3): add a bias to each expert's routing score, used only to choose the top-k and not in the weights; during training the bias of an overloaded expert is lowered and that of an idle one raised. This avoids the auxiliary loss interfering with model quality.

Simulate the routing to see it: the more biased the router, the busier the busy experts, the larger the auxiliary loss, and the more tokens exceed capacity:

<div class="aig-widget" data-widget="moe-route"></div>

```python
def aux_loss(probs, idx, n_experts):
    f = torch.bincount(idx.flatten(), minlength=n_experts).float() / idx.numel()   # actual share of assignments
    P = probs.mean(dim=0)                                                            # mean routing probability
    return n_experts * (f * P).sum()

_, idx, probs = moe.route(x)
print(f"随机初始化的路由器，辅助损失 = {aux_loss(probs, idx, 16).item():.3f}（完全均衡时约为 1）")

# extreme imbalance: every token picks the first 4 experts
skewed = torch.zeros(50, 16)
skewed[:, :4] = 0.25
bad_idx = torch.arange(4).repeat(50, 1)
print(f"极端不均衡时，辅助损失 = {aux_loss(skewed, bad_idx, 16).item():.3f}")
assert aux_loss(skewed, bad_idx, 16) > aux_loss(probs, idx, 16)
```

## Total and active parameters {#总参数与激活参数}

Build the model from its official config on the "meta device" (shapes only, no memory allocated) and you can count the parameters exactly:

```python
import transformers
from transformers import AutoModelForCausalLM

def total_and_active(cfg, n_experts, top_k, moe_layers, expert_params):
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(cfg)
    total = sum(p.numel() for p in model.parameters())
    return total, total - (n_experts - top_k) * expert_params * moe_layers   # subtract the experts a token does not use

ds = transformers.DeepseekV3Config()          # the defaults are DeepSeek-V3's config
ds_total, ds_active = total_and_active(ds, ds.n_routed_experts, ds.num_experts_per_tok,
                                       ds.num_hidden_layers - ds.first_k_dense_replace,
                                       3 * ds.hidden_size * ds.moe_intermediate_size)
mx = transformers.MixtralConfig()             # the defaults are Mixtral-8x7B's config
mx_total, mx_active = total_and_active(mx, mx.num_local_experts, mx.num_experts_per_tok,
                                       mx.num_hidden_layers, 3 * mx.hidden_size * mx.intermediate_size)
print(f"DeepSeek-V3: 总参数 {ds_total / 1e9:.1f}B，每个 token 激活 {ds_active / 1e9:.1f}B")
print(f"Mixtral-8x7B: 总参数 {mx_total / 1e9:.1f}B，每个 token 激活 {mx_active / 1e9:.1f}B")
assert round(ds_total / 1e9) == 671 and round(ds_active / 1e9) == 38
assert round(mx_total / 1e9, 1) == 46.7 and round(mx_active / 1e9, 1) == 12.9
```

DeepSeek-V3 has 671 billion parameters, of which each token activates only about 37.5 billion (officially stated as 37B): the compute of a 37B dense model with a knowledge capacity close to that of a 671B model.

!!! inference "Inference view"
    MoE changes many trade-offs in inference:

    - **Memory goes by total parameters, compute by active parameters**: DeepSeek-V3's FP8 weights are about 700 GB, needing at least one 8-GPU H200 machine or a multi-node deployment, yet the compute per token is only that of a 37B model;
    - **MoE suffers at small batch sizes**: in decode the tokens of a batch are spread over many experts, each expert gets very few tokens, and the work degenerates into matrix-vector products; moreover, the more experts are activated, the more weights must be read. At batch 1 the weights of 8 experts are read, while with a large batch nearly all 256 experts' weights are read. So **serving MoE models needs larger batches** to amortize the weight reads;
    - **Expert parallelism (EP)**: spread the experts over many GPUs; each layer needs two all-to-alls (send tokens to the GPUs that hold their experts, then gather the results back), and DeepEP is a communication library designed for this;
    - **Load imbalance**: routing at inference time is uncontrolled, so the GPUs holding popular experts become bottlenecks; hence redundant experts (replicating popular experts on several GPUs) and re-placing experts by load (such as DeepSeek's open-source EPLB);
    - **Fused MoE kernels**: grouping, the grouped GEMM, the activation function and the weighted combination are fused into a few kernels.

!!! interview "How to explain it"
    Explain MoE along "structure → computation → deployment": router + top-k selection + many small experts + weighted combination, with DeepSeekMoE adding fine-grained and shared experts; total parameters decide memory and active parameters decide compute (671B / 37B for DeepSeek-V3); the skeleton of the implementation is grouping by expert (sort, count, segmented GEMM, scatter back), which is the fused MoE kernel; training needs load balancing (auxiliary loss, capacity limits, the auxiliary-loss-free bias). At inference time each expert gets very few tokens in decode and mostly reads weights, so you need larger batches, expert parallelism and all-to-all, and you must handle load imbalance.

## Exercises {#练习}

**1. Compute.** An MoE model has 64 experts, each token picks 8, each expert has 3 × 2048 × 1408 parameters, and there are 24 MoE layers. How many parameters do all the experts have together? How many expert parameters does each token activate?

??? success "Answer"
    ```python
    per_expert = 3 * 2048 * 1408
    total = 64 * per_expert * 24
    active = 8 * per_expert * 24
    print(f"{total / 1e9:.2f}B 总，{active / 1e9:.2f}B 激活")   # 13.29B total, 1.66B active
    ```

**2. Food for thought.** In `forward_grouped`, what happens if an expert gets no tokens at all? If routing is severely imbalanced, how does that affect inference performance?

??? success "Answer"
    The code skips the expert when `n == 0`, so nothing breaks. A real grouped GEMM kernel must also handle "an empty group" correctly. With imbalanced routing, the experts with many tokens take longer, the whole MoE layer waits for the slowest expert (or GPU) to finish, and the other experts (GPUs) sit idle; under expert parallelism, the GPUs of popular experts also receive more all-to-all data. That is why inference systems track expert load and balance it with redundant experts and re-placement.

## Summary {#小结}

- [x] An MoE layer = router + top-k selection + many small FFN experts + weighted combination; DeepSeekMoE adds fine-grained and shared experts.
- [x] Computing grouped by expert (sort, count, segmented GEMM, scatter back) is the skeleton of the fused MoE kernel.
- [x] Training needs load balancing: an auxiliary loss, capacity limits, or DeepSeek-V3's auxiliary-loss-free bias adjustment.
- [x] Total parameters decide memory and active parameters decide compute; 671B / 37B for DeepSeek-V3.
- [x] MoE inference needs larger batches, expert parallelism and all-to-all communication, and must handle load imbalance.
