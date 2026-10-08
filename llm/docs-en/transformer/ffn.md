# Feed-forward networks and SwiGLU

<p class="lead">Attention moves information between tokens; the feed-forward network (FFN, also called the MLP) "processes" the information of each token on its own. It has the simplest structure, yet it holds about two thirds of a Transformer layer's parameters and compute, and it is where the largest matrix multiplications of inference live.</p>

!!! question "Self-test: if you can answer these, skip this chapter"
    1. What are the FFN's input and output shapes? What is its biggest difference from attention?
    2. How is SwiGLU computed? Why does it have three matrices?
    3. Why is LLaMA-7B's intermediate dimension 11008 rather than 4 × 4096?
    4. What share of a Transformer layer's parameters does the FFN hold?
    5. How do inference engines usually implement SwiGLU?

??? success "Answers (try first, then expand to compare)"
    1. Both are `[B, T, d]`. The FFN computes each token independently and exchanges no information between tokens; attention is the only place where tokens look at each other.
    2. $\mathrm{down}(\mathrm{silu}(\mathrm{gate}(x)) \odot \mathrm{up}(x))$: the gate branch goes through SiLU and multiplies the up branch elementwise as a "gate", and down projects back to d dimensions, hence three matrices.
    3. With three matrices, to keep the parameter count equal to the old FFN with two matrices and an intermediate dimension of 4d, the intermediate dimension is set to $\frac{2}{3} \times 4d = \frac{8}{3}d \approx 10923$, rounded up to a multiple of 256 to give 11008.
    4. Without GQA, attention is about $4d^2$ and the FFN about $8d^2$, so the FFN is about two thirds; in models with GQA it is higher (about 60% for this chapter's Qwen3-0.6B, about 70% for LLaMA-3-8B).
    5. Merge gate and up into one `gate_up_proj` GEMM, then use a fused kernel that does SiLU and the elementwise multiplication together (`silu_and_mul`), and finally the down GEMM.

## The classic two-layer MLP {#经典的两层-mlp}

The original Transformer's FFN first projects up to $d_{ff}$ (usually 4d), applies an activation function, and projects back down to d:

$$
\text{FFN}(x) = \text{Act}(x W_1)\, W_2
$$

It **computes each token independently**, with no interaction between tokens. So the FFN can be seen as a "lookup + processing" module applied to every token: research has found that each row of $W_1$ acts like a "pattern detector", and the matching column of $W_2$ is the "content" written back to the residual stream when that pattern is detected; much factual knowledge is stored in the FFN weights.

## Activation functions {#激活函数}

```pycon
>>> import torch
>>> import torch.nn.functional as F
>>> x = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0])
>>> F.relu(x)
tensor([0., 0., 0., 1., 3.])
>>> F.gelu(x)
tensor([-0.0041, -0.1587,  0.0000,  0.8413,  2.9959])
>>> F.silu(x)                     # SiLU(x) = x * sigmoid(x), also called Swish
tensor([-0.1423, -0.2689,  0.0000,  0.7311,  2.8577])
```

GELU and SiLU are "smooth versions" of ReLU: continuously differentiable around 0, and not entirely 0 for negative inputs. Modern large models almost all use SiLU or GELU.

## Gating: SwiGLU {#门控swiglu}

The **GLU (gated linear unit)** family multiplies two projections: one goes through an activation function and acts as a "gate" controlling how much of the other gets through. **SwiGLU**, used by LLaMA, Qwen, DeepSeek and others:

$$
\text{SwiGLU}(x) = \big(\text{SiLU}(x W_{gate}) \odot x W_{up}\big)\, W_{down}
$$

There are three matrices: `gate_proj` and `up_proj` (both d → $d_{ff}$) and `down_proj` ($d_{ff}$ → d). Experiments show that the gated structure works better for the same parameter count.

What "gate" means is clearest when you draw the output of one intermediate neuron, $\text{SiLU}(a) \cdot b$, as a surface over $a$ (the gate's input) and $b$ (the content), and compare it with ungated GELU, hard-cut ReGLU and the bilinear form without any nonlinearity:

<div class="aig-widget" data-widget="swiglu3d"></div>

To keep the parameter count equal to the classic two-layer 4d MLP ($2 \times 4d^2 = 8d^2$), the three-matrix structure sets the intermediate dimension to about $\frac{8}{3}d$: $3 \times \frac{8}{3}d \times d = 8d^2$. LLaMA-7B has d = 4096, $\frac{8}{3} \times 4096 ≈ 10923$, rounded up to a multiple of 256 to give 11008. That is only a starting point, though, and many models choose other ratios: Qwen2.5-0.5B uses 4864 / 896 ≈ 5.4 times, and Qwen3-0.6B 3072 / 1024 = 3 times.

The implementation below is identical to `Qwen2MLP` in transformers:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP

class SwiGLU(nn.Module):
    def __init__(self, d, d_ff):
        super().__init__()
        self.gate_proj = nn.Linear(d, d_ff, bias=False)
        self.up_proj = nn.Linear(d, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

torch.manual_seed(0)
cfg = Qwen2Config(hidden_size=896, intermediate_size=4864, hidden_act="silu")
ref = Qwen2MLP(cfg)
ours = SwiGLU(896, 4864)
ours.load_state_dict(ref.state_dict())               # same parameter names, so it loads directly
x = torch.randn(2, 5, 896)
assert torch.allclose(ours(x), ref(x), atol=1e-6)
print({k: tuple(v.shape) for k, v in ours.state_dict().items()})
```

## Shares of parameters and compute {#参数与计算量的占比}

Work out the parameters of each part of one Qwen3-0.6B layer:

```pycon
>>> d, d_ff, n_h, n_kv, d_h = 1024, 3072, 16, 8, 128
>>> attn = d * n_h * d_h + 2 * d * n_kv * d_h + n_h * d_h * d      # the four projections q, k, v, o (not counting QK-Norm's 2 × 128 parameters)
>>> mlp = 3 * d * d_ff
>>> attn, mlp, round(mlp / (attn + mlp), 3)
(6291456, 9437184, 0.6)
```

In this model the FFN holds 60% of a layer's parameters. The attention part is larger than "4d²" because Qwen3's heads × head dimension (16 × 128 = 2048) is twice the hidden dimension, so the q and o projections have 2d² parameters each; K and V use GQA (8 heads; see [attention variants](attention-variants.md)), which wins some of that back. In LLaMA-7B, which does not use GQA, attention is $4d^2$ and the FFN about $8d^2$, so the FFN is about two thirds. **Since each parameter contributes 2 operations per token, the share of parameters is essentially the share of compute** (the $T^2$ term of the attention scores themselves is counted separately).

Try the configs of a few models to see how the parameters of a layer and of the whole model are divided, and at what context length the $T^2$ term of the attention scores turns into the largest part:

<div class="aig-widget" data-widget="param-share"></div>

!!! inference "Inference view"
    - **The FFN is the largest GEMM**: in decode it decides how many weights are read, and in prefill it accounts for most of the compute;
    - **gate and up are fused into one matrix multiplication**: `gate_proj` and `up_proj` take the same input, so inference engines concatenate the two weights into one `[2·d_ff, d]` matrix (called `gate_up_proj` in vLLM), get both results from one GEMM, and compute `silu(gate) * up` in one fused kernel (`SiluAndMul` in vLLM), saving a kernel launch and the read and write of an intermediate result;
    - **Tensor parallelism**: `gate_up_proj` is split along the output dimension (column parallel), so each GPU computes part of the intermediate dimension and the activation can be computed locally; `down_proj` is split along the input dimension (row parallel), so each GPU gets a partial sum, followed by one all-reduce. The whole FFN needs just one communication;
    - **MoE**: replace one large FFN with many small FFNs (experts), and use only a few of them for each token; see [mixture of experts](moe.md).

!!! interview "How to explain it"
    On the FFN: it computes each token independently and, without GQA, holds about two thirds of a layer's parameters and compute (higher in models with GQA); SwiGLU = `down(silu(gate(x)) * up(x))`, three matrices, with an intermediate dimension of about 8d/3 rounded up (LLaMA-7B's 11008). At inference time gate and up are merged into one GEMM, and the activation and multiplication are fused into one kernel; under tensor parallelism gate/up are split by columns and down by rows, so the whole FFN needs only one all-reduce.

## Exercises {#练习}

**1. Fuse gate and up.** Rewrite `SwiGLU` to use one `gate_up_proj` (`nn.Linear(d, 2 * d_ff)`), copy the weights from `ours` above, and check that the output matches.

??? success "Answer"
    ```python
    class FusedSwiGLU(nn.Module):
        def __init__(self, d, d_ff):
            super().__init__()
            self.gate_up_proj = nn.Linear(d, 2 * d_ff, bias=False)
            self.down_proj = nn.Linear(d_ff, d, bias=False)

        def forward(self, x):
            gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
            return self.down_proj(F.silu(gate) * up)

    fused = FusedSwiGLU(896, 4864)
    with torch.no_grad():
        fused.gate_up_proj.weight.copy_(torch.cat([ours.gate_proj.weight, ours.up_proj.weight], dim=0))
        fused.down_proj.weight.copy_(ours.down_proj.weight)
    assert torch.allclose(fused(x), ours(x), atol=1e-6)
    ```

    The weights are concatenated by rows (output dimension): the first half of the output is gate, the second half up. This is how vLLM puts the two matrices together when loading Hugging Face weights.

**2. Compute.** LLaMA-3-8B has d = 4096, $d_{ff}$ = 14336 and 32 layers. How many parameters do the FFNs of all layers have together? What share of the 8.03 billion total is that?

??? success "Answer"
    3 × 4096 × 14336 ≈ 176 million per layer, about 5.64 billion over 32 layers: 70%.

    ```python
    ffn = 32 * 3 * 4096 * 14336
    assert ffn == 5_637_144_576
    print(f"{ffn / 8_030_261_248:.1%}")   # 70.2%
    ```

## Summary {#小结}

- [x] The FFN computes each token independently and is the part of a Transformer layer with the most parameters and compute.
- [x] Modern large models use SwiGLU: `down(silu(gate(x)) * up(x))`, three matrices, with an intermediate dimension usually no smaller than 8d/3.
- [x] The share of parameters roughly equals the share of compute; GQA shrinks attention and raises the FFN's share.
- [x] At inference time gate and up are fused into one GEMM and the activation and multiplication into one kernel; under tensor parallelism the whole FFN needs only one all-reduce.
