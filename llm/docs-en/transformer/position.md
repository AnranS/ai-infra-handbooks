# Positional encoding and RoPE

<p class="lead">Attention by itself does not know the order of tokens. To attention without position information, "the cat chases the dog" and "the dog chases the cat" look the same. Positional encoding injects order into the model. Nearly all of today's large models use rotary position embedding (RoPE), and its design directly affects long-context extension, how the KV cache is stored, and how inference kernels are written.</p>

!!! question "Self-test: if you can answer these, skip this chapter"
    1. Why does attention need positional encoding?
    2. What transformation does RoPE apply to Q and K? Why do we say it encodes relative position?
    3. What does raising `rope_theta` (the base) from 10000 to 1000000 do?
    4. Why does the same model's RoPE come in two forms (adjacent pairs of dimensions vs. the two halves)?
    5. Is the K stored in the KV cache before or after rotation?

??? success "Answers (try first, then expand to compare)"
    1. Attention is permutation-equivariant in its input: shuffle the tokens and the output is just shuffled the same way, so the model cannot tell order by itself, and position information must be injected explicitly.
    2. Treat every two dimensions of q and k as a complex number and rotate it by an angle $m\theta_i$ according to its position m. After rotation, $q_m^\top k_n$ depends only on $n - m$, so it encodes relative position.
    3. The frequencies are $\theta_i = \text{base}^{-2i/d}$; a larger base makes the low-frequency dimensions rotate more slowly, so they do not "come back to the start" over longer distances, and the model can tell apart positions further away. This is used for long contexts.
    4. One pairs adjacent dimensions, the other pairs the first half with the second half. The two are mathematically equivalent and differ only in how the dimensions are ordered, but the form must match the weights' dimension order (when converting weights from one format to another, the q and k projection matrices are permuted accordingly).
    5. After: each K is rotated by its own position before it is written to the cache, and is later dotted directly with new queries.

<!-- comic ../assets/comics/rope.webp is in Chinese; put it back once the English version exists -->

## Attention is "blind" to order {#注意力对顺序视而不见}

Without a mask and without position information, attention is **permutation-equivariant** in its input: shuffle the input tokens and the output is shuffled the same way, with every token getting exactly the same result:

```python
import math
import torch

torch.manual_seed(0)
T, d = 6, 16
x = torch.randn(T, d)
Wq, Wk, Wv = (torch.randn(d, d) for _ in range(3))

def attn(x):
    q, k, v = x @ Wq, x @ Wk, x @ Wv
    return (q @ k.T / math.sqrt(d)).softmax(-1) @ v

perm = torch.randperm(T)
assert torch.allclose(attn(x)[perm], attn(x[perm]), atol=1e-5)   # shuffling the input = shuffling the output
```

Language obviously depends on order, so the model must be told explicitly where each token is. (The causal mask itself provides a little implicit position information, but nowhere near enough.)

## Early approaches: absolute positional encoding {#早期方案绝对位置编码}

- **Sinusoidal positional encoding** (the original Transformer): give position m a fixed vector (sines and cosines of different frequencies) and add it to the word embedding;
- **Learned absolute position embeddings** (GPT-2, BERT): learn a vector for each position, just like word embeddings.

Their problem: the model learns absolute facts such as "position 5", whereas what matters more in language is often **relative** distance ("the previous word", "three words back"); learned position embeddings also cannot handle sequences longer than those seen in training.

## RoPE: encoding position by rotation {#rope用旋转编码位置}

Rotary Position Embedding (Su Jianlin et al., 2021) is a very elegant idea: **instead of adding the position to the vector, rotate q and k by an angle that depends on the position**.

![Figure: RoPE rotates each pair of dimensions by position, so the dot product keeps only the relative position](../assets/figures/rope.svg){.aig-svg}

Start with two dimensions. Rotate the query vector at position m by an angle $m\theta$, and the key vector at position n by $n\theta$:

$$
q'_m = R(m\theta)\, q,\qquad k'_n = R(n\theta)\, k,\qquad
R(\alpha) = \begin{pmatrix} \cos\alpha & -\sin\alpha \\ \sin\alpha & \cos\alpha \end{pmatrix}
$$

Rotation matrices satisfy $R(a)^\top R(b) = R(b - a)$, so their dot product is:

$$
q'^\top_m k'_n = q^\top R(m\theta)^\top R(n\theta)\, k = q^\top R\big((n - m)\theta\big)\, k
$$

It **depends only on the relative position $n - m$**, not on the absolute positions. And it is multiplicative, so it does not mix position information into content information the way addition does.

For a $d_h$-dimensional vector, split it into $d_h/2$ pairs and rotate each pair independently, with pair i using its own frequency:

$$
\theta_i = \text{base}^{-2i/d_h},\qquad i = 0, 1, \ldots, d_h/2 - 1
$$

The first pairs rotate fast (high frequency, sensitive to short distances) and the last pairs rotate slowly (low frequency, able to distinguish very long distances). `base` is the `rope_theta` in the config file.

### Implementation {#实现}

In practice there are two equivalent ways to form the pairs:

- **Adjacent dimensions**: (0, 1), (2, 3), …; each pair can be seen as a complex number, and the rotation is multiplication by $e^{i m\theta}$. Meta's original LLaMA code is written this way;
- **The two halves**: dimension i pairs with dimension i + d_h/2, implemented with `rotate_half`. Hugging Face transformers, Qwen and others use this form.

```python title="rope.py"
"""rope.py —— 旋转位置编码的两种等价实现。"""

import torch


def inv_freq(head_dim: int, base: float = 10000.0) -> torch.Tensor:
    return 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))   # [d/2]


def rope_half(x: torch.Tensor, pos: torch.Tensor, base: float = 10000.0) -> torch.Tensor:
    """"前后两半一组"（transformers 的写法）。x: [..., T, d]，pos: [T]"""
    freqs = pos.float()[:, None] * inv_freq(x.shape[-1], base)[None, :]   # [T, d/2]
    cos, sin = torch.cat([freqs, freqs], -1).cos(), torch.cat([freqs, freqs], -1).sin()
    x1, x2 = x.chunk(2, dim=-1)
    rotated = torch.cat([-x2, x1], dim=-1)                                 # rotate_half
    return x * cos + rotated * sin


def rope_complex(x: torch.Tensor, pos: torch.Tensor, base: float = 10000.0) -> torch.Tensor:
    """"相邻两维一组"（原始 LLaMA 的写法）：把 (x0, x1) 看成复数 x0 + i·x1，乘以 e^{i·m·θ}。"""
    freqs = pos.float()[:, None] * inv_freq(x.shape[-1], base)[None, :]
    rot = torch.polar(torch.ones_like(freqs), freqs)                       # e^{i m θ}
    xc = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    return torch.view_as_real(xc * rot).flatten(-2)
```

Check RoPE's key properties:

```python
from rope import rope_complex, rope_half

torch.manual_seed(0)
d = 64
q, k = torch.randn(d), torch.randn(d)

def score(m, n, rope):
    qm = rope(q[None, :], torch.tensor([m]))[0]
    kn = rope(k[None, :], torch.tensor([n]))[0]
    return (qm @ kn).item()

# 1) the dot product depends only on relative position: (5, 3), (105, 103), (1005, 1003) give the same result
for rope in (rope_half, rope_complex):
    s = [score(m, m - 2, rope) for m in (5, 105, 1005)]
    assert max(s) - min(s) < 1e-3, s

# 2) rotation does not change a vector's length
x = torch.randn(10, d)
assert torch.allclose(rope_half(x, torch.arange(10)).norm(dim=-1), x.norm(dim=-1), atol=1e-4)

# 3) the two forms are equivalent: only the order of the dimensions differs
perm = torch.cat([torch.arange(0, d, 2), torch.arange(1, d, 2)])   # adjacent-pair layout -> two-halves layout
x = torch.randn(7, d)
pos = torch.arange(7)
assert torch.allclose(rope_complex(x, pos)[:, perm], rope_half(x[:, perm], pos), atol=1e-5)
```

Point 3 shows that the two forms differ only by a **permutation** of the q and k dimensions. This has a practical consequence: when converting Meta-format LLaMA weights to the Hugging Face format, the conversion script permutes the output dimensions of `q_proj` and `k_proj`. **The weights and the RoPE form must match**, or the model runs but outputs garbage.

## Frequencies and "wavelengths" {#频率与波长}

The number of positions it takes pair i to turn a full circle (its wavelength) is $2\pi / \theta_i$:

```pycon
>>> import math
>>> from rope import inv_freq
>>> for base in (10_000, 1_000_000):
...     f = inv_freq(64, base)
...     print(base, [round(2 * math.pi / f[i].item()) for i in (0, 8, 16, 24, 31)])
...
10000 [6, 63, 628, 6283, 47117]
1000000 [6, 199, 6283, 198692, 4080185]
```

The fastest pair turns a full circle every 6 tokens, while the slowest takes tens of thousands or even millions of tokens. Raising the base from 10000 to 1000000 (Qwen2.5's value; LLaMA-3 uses 500000) stretches the wavelengths of the low-frequency part even further, so the model can still tell positions apart over very long distances. This is one of the foundations of long-context support.

Draw how one pair of dimensions changes with position as a 3D helix, and you can see two things at once: "how fast it turns" (the wavelength) and "the angle between q and k depends only on m − n". Drag to rotate, and pull i from 0 to 63 to watch the helix go from a spring to a straight line:

<div class="aig-widget" data-widget="rope-helix"></div>

## Extending the context {#长上下文扩展}

A model trained at 4K length may be used at 32K or even 128K at inference time; positions (rotation angles) beyond the training range were never seen, and quality drops sharply. A few common extension methods:

| Method | How | Notes |
| --- | --- | --- |
| Position interpolation (PI) | scale position m to m / s, "squeezing" every position back into the training range | simple, but compressing the high frequencies lowers short-range resolution, so it usually needs a little fine-tuning |
| NTK-aware scaling | increase the base, mainly stretching the wavelengths of the low frequencies while leaving the high frequencies nearly unchanged | works to some extent without fine-tuning |
| YaRN | treat frequencies in bands: no interpolation for high frequencies, interpolation for low ones, a transition in between, plus a temperature correction on attention scores | Qwen2.5 uses it to extend the context to 128K, written in the config as `rope_scaling: {"type": "yarn", "factor": 4.0, ...}` |
| LLaMA-3.1's approach | band-wise frequency scaling similar to YaRN | `rope_type: "llama3"` in the config |

All of these only change how the cos/sin tables are computed (and possibly the attention scaling), leaving the model architecture unchanged. **The inference engine must implement the same RoPE variant as the config**, or long-text quality degrades noticeably.

Another approach is **ALiBi**: instead of rotating vectors, add a negative bias proportional to distance directly to the attention scores, so distant tokens are "penalized". BLOOM, MPT and others used it; today's mainstream models mostly use RoPE.

!!! inference "Inference view"
    - **The KV cache stores K after rotation**: each token's K is rotated by its position when it is written to the cache and never changes afterwards. During decode only the new token's q and k need rotating;
    - Because K has been rotated by absolute position, **a cache can be reused only if the same text appears at the same positions**. Prefix caching satisfies this naturally (what is shared is always a prefix starting at position 0); dropping tokens in the middle or "shifting" the cache to other positions requires reprocessing the positions;
    - RoPE needs little computation but a fair amount of memory traffic, so inference engines usually fuse it with the operations after the QKV projection into one kernel (sometimes together with writing the KV cache) instead of launching it separately;
    - To compress the KV cache, DeepSeek's MLA splits a small set of dimensions off for RoPE; see [attention variants](attention-variants.md#mla多头潜在注意力).

!!! interview "How to explain it"
    On RoPE: every two dimensions of q and k are rotated by position, so the dot product depends only on relative position; different pairs use different frequencies, and a larger base (`rope_theta`) makes the low frequencies rotate more slowly, for long contexts; the two forms (adjacent pairs, two halves) are mathematically equivalent but must match the weights' dimension order, a common bug when bringing up a new model; the KV cache stores K after rotation, so cache reuse requires the same positions; long-context extensions such as PI, NTK and YaRN only change how cos/sin are computed, and at inference time RoPE is often fused with neighboring operations.

## Exercises {#练习}

**1. Rotate by hand.** The 2D vector q = (1, 0), θ = π/2, at position m = 1. What does q become after RoPE? What about k = (1, 0) at position n = 3? What is their dot product, and does it agree with $q^\top R((n-m)\theta)k$?

??? success "Answer"
    q' = R(π/2)(1, 0) = (0, 1); k' = R(3π/2)(1, 0) = (0, −1); the dot product is −1.
    Directly: $R(2 \cdot \pi/2) = R(\pi)$, and $q^\top R(\pi) k = (1, 0) \cdot (−1, 0) = −1$, which agrees.

    ```python
    import math, torch
    def R(a): return torch.tensor([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    q = k = torch.tensor([1.0, 0.0])
    lhs = (R(math.pi / 2) @ q) @ (R(3 * math.pi / 2) @ k)
    rhs = q @ R(math.pi) @ k
    assert abs(lhs.item() + 1) < 1e-6 and abs(rhs.item() + 1) < 1e-6
    ```

**2. Food for thought.** What happens if an inference engine uses one RoPE form in prefill and the other in decode?

??? success "Answer"
    The cached K was rotated with one way of pairing the dimensions and the new token's q with the other, so their relative-position relationship is broken and the attention scores come out wrong. The usual symptom: the first few tokens look fine (they come from the last position of prefill), and soon the output turns repetitive or meaningless. Bugs like this are well hidden; inference engine tests usually compare the logits of "step-by-step generation with a cache" against "computing the whole sequence without a cache", and the [KV cache](../inference/kv-cache.md) chapter checks our implementation this way.

## Summary {#小结}

- [x] Attention is permutation-equivariant in its input, so position information must be injected explicitly.
- [x] RoPE rotates q and k by position, so the dot product depends only on relative position; different pairs of dimensions use different frequencies, and a larger base makes the low frequencies slower.
- [x] The two implementations (adjacent pairs / two halves) are equivalent but must match the weights' dimension order.
- [x] Long-context extensions (PI, NTK, YaRN) only change how cos/sin are computed, and the inference engine must implement what the config says.
- [x] The KV cache stores K after rotation, so cache reuse requires the same positions; RoPE is often fused with other operations into one kernel.
