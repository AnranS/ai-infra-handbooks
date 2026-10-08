# Pretraining and scaling laws

<p class="lead">Working on inference does not require knowing how to train large models, but it does require knowing where models come from: the training objective, the scale of the data, compute estimates, and why today's small models are "overtrained". This helps you understand the limits of a model's abilities, and shows what training and inference have in common, and where they differ, in memory, parallelism and precision.</p>

!!! question "Self-test: if you can answer these, skip this chapter"
    1. Roughly how many FLOPs does it take to train a model of N parameters on D tokens? Why is the factor 6?
    2. What does the Chinchilla law say? Why was LLaMA-3-8B trained on far more data than "optimal"?
    3. With mixed-precision AdamW training, how many bytes of memory does each parameter take?
    4. What is MFU?
    5. What are the common kinds of training parallelism? How do they differ from parallelism in inference?

??? success "Answers (try first, then expand to compare)"
    1. About $6ND$. The forward pass does one multiply-add (2 operations) per parameter per token, and the backward pass computes the gradients with respect to the activations and with respect to the weights, one equally large matrix multiplication each (4 operations), for 6 in total.
    2. For a given compute budget, parameters and data should grow in proportion, with an optimum of about 20 tokens per parameter. But inference cost depends only on model size, and training a small model longer makes it stronger at the same inference cost, so LLaMA-3-8B was trained on about 15T tokens (about 1900 tokens per parameter).
    3. About 16 bytes: bf16 parameters 2 + gradients 2, an fp32 master copy 4, and Adam's two moments 4 each.
    4. Model FLOPs utilization: the model computation actually completed (counted at $6N$ per token, excluding recomputation) ÷ the hardware's peak compute; large-scale training typically reaches 35% to 55%.
    5. Data parallelism (including ZeRO / FSDP), tensor parallelism, pipeline parallelism, context parallelism and expert parallelism. Training must also hold gradients and optimizer states and handle the backward pass and lots of activations; inference has none of these and cares about latency and the KV cache, commonly using TP, EP, DP attention and PD disaggregation.

## What pretraining does {#预训练做什么}

There is only one goal: [next-token prediction](../basics/language-model.md#训练目标交叉熵) on a huge amount of text, minimizing the cross-entropy.

- **Data**: web pages, books, code, papers, math and more, deduplicated, quality-filtered and filtered for harmful content, then mixed in set proportions. Mainstream models are now trained on over ten trillion (10^13) tokens: LLaMA-3 on more than 15T, Qwen2.5 on 18T, DeepSeek-V3 on 14.8T;
- **Sequence packing**: join documents end to end and cut them into training samples of fixed length (say 4096 or 8192) to avoid wasting padding;
- **Length in stages**: train on most of the data with a shorter context first, then extend to 32K or 128K in a final "long-context stage" (together with RoPE adjustments; see [positional encoding](../transformer/position.md#长上下文扩展));
- **Optimizer**: AdamW; the learning rate warms up and then decays along a cosine, or follows a "warmup-stable-decay" (WSD) schedule; gradient clipping; each step's batch is typically millions of tokens.

Below, `mini_llm.py` trains for a few dozen steps on a short text to walk through the whole process:

```python
import math
import torch
import torch.nn.functional as F
from mini_llm import Config, Transformer

torch.manual_seed(0)
text = "北京是中国的首都。上海是中国最大的城市。" * 40
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
data = torch.tensor([stoi[c] for c in text])

cfg = Config(vocab_size=len(chars), hidden_size=64, intermediate_size=172, num_hidden_layers=2,
             num_attention_heads=4, num_key_value_heads=2, tie_word_embeddings=True)
model = Transformer(cfg)
opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.1)
steps, warmup, seq_len = 60, 6, 32
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: (s + 1) / warmup if s < warmup else 0.5 * (1 + math.cos(math.pi * (s - warmup) / (steps - warmup))))

losses = []
for step in range(steps):
    starts = torch.randint(0, len(data) - seq_len - 1, (16,))
    x = torch.stack([data[s:s + seq_len] for s in starts])          # [16, 32]
    y = torch.stack([data[s + 1:s + seq_len + 1] for s in starts])  # targets shifted by one
    loss = F.cross_entropy(model(x).view(-1, cfg.vocab_size), y.view(-1))
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)          # gradient clipping
    opt.step()
    sched.step()
    losses.append(loss.item())
print(f"loss: {losses[0]:.2f} -> {losses[-1]:.2f}")
assert losses[-1] < losses[0] / 3
```

Real pretraining scales this loop up to thousands of GPUs and trillions of tokens, plus data pipelines, parallelism, fault tolerance and monitoring.

## How much compute training needs {#训练需要多少算力}

As [shown earlier](../basics/math-torch.md#矩阵乘法线性层), the forward pass takes about 2N operations per token (N is the parameter count). The backward pass computes two gradients, with respect to the inputs and with respect to the weights, about twice the forward cost, or 4N. So:

$$
C_{\text{train}} \approx 6ND
$$

```python
def train_flops(params, tokens):
    return 6 * params * tokens

C = train_flops(8.03e9, 15e12)                      # LLaMA-3-8B scale: 8 billion parameters, 15 trillion tokens
h100_bf16 = 989e12                                  # dense BF16 peak of an H100 SXM
gpu_hours = C / (0.4 * h100_bf16) / 3600             # assume 40% compute utilization
print(f"{C:.2e} FLOPs，约 {gpu_hours / 1e3:.0f}k 个 H100 小时")
```

**MFU (Model FLOPs Utilization)** = the effective compute actually achieved / the hardware peak. Large-scale training usually reaches an MFU of 35% to 55%. The prefill phase of inference is measured the same way; the decode phase is memory bound and better measured by bandwidth utilization.

## Scaling laws {#scaling-law}

**Kaplan et al. (2020)** found that a model's loss falls smoothly as a **power law** in parameters, data and compute, so small experiments can predict the results of large-scale training.

**Chinchilla (Hoffmann et al., 2022)** answered: for a compute budget C, how should parameters and data be split to minimize the loss? The conclusion is that both should grow in proportion, **with the optimal amount of data about 20 times the parameter count**. By this standard, a 70B model trained on 1.4T tokens is "enough".

The conclusion comes from a fitted loss function $L(N, D) = E + A/N^{\alpha} + B/D^{\beta}$: draw it as a surface, and the compute budget $C = 6ND$ is a line on it; the lowest point along the line is the optimal split. Drag to rotate, and move the budget:

<div class="aig-widget" data-widget="scaling3d"></div>

Yet today's models generally go far beyond this ratio: LLaMA-3-8B used 15T tokens, about 1900 times its parameter count. The reason is inference:

!!! inference "Inference view"
    The Chinchilla optimum considers only the cost of **training**. But once released, a model is called trillions of times, and inference costs far exceed training. **Training a small model for longer** ("overtraining") is not optimal for training efficiency, but it yields a smaller, cheaper model for inference at the same capability. This is why small models of 7B or 8B are trained on over ten trillion tokens, and also why MoE is popular: it gets a large capacity with relatively few active parameters.

## Training memory {#训练的显存}

Mixed-precision training with AdamW needs per parameter:

| Item | Bytes |
| --- | --- |
| BF16 weights (used in forward and backward) | 2 |
| BF16 gradients | 2 |
| FP32 master weights (for the optimizer update) | 4 |
| Adam's first moment m (FP32) | 4 |
| Adam's second moment v (FP32) | 4 |
| **Total** | **16** |

For a 7B model this alone is 112 GB, not counting activations. Inference needs only the weights themselves (2 bytes per parameter in BF16) plus the KV cache.

```python
def train_state_gb(params, bytes_per_param=16):
    return params * bytes_per_param / 2**30

for name, n in [("0.6B", 0.596e9), ("7B", 7e9), ("70B", 70e9)]:
    print(f"{name}: 训练状态约 {train_state_gb(n):.0f} GB，推理权重（BF16）约 {n * 2 / 2**30:.0f} GB")
```

So large-model training has to split these states across many GPUs:

| Parallelism | What it splits | Notes |
| --- | --- | --- |
| Data parallelism + ZeRO / FSDP | optimizer states, gradients and weights split across the data-parallel GPUs, gathered temporarily when needed | the most common in training |
| Tensor parallelism (TP) | each matrix across several GPUs | the same as TP in inference, limited to one node |
| Pipeline parallelism (PP) | by layer | the batch must be cut into micro-batches to fill the pipeline |
| Context parallelism (CP) | along the sequence length | long-context training |
| Expert parallelism (EP) | MoE experts on different GPUs | the same as EP in inference |

There is also **activation recomputation** (not saving some intermediates in the forward pass and recomputing them in the backward pass), which trades compute for memory.

## Multi-token prediction {#多-token-预测}

DeepSeek-V3 added a **multi-token prediction (MTP)** objective in training: besides predicting the next token, an extra small module predicts the token after it. This makes the training signal denser; more importantly, at inference time the module can serve directly as the **draft model for speculative decoding**, "guessing" several tokens at once for the main model to verify; see [serving](../inference/serving.md#投机解码). It is a good example of a training design that directly serves inference speed.

!!! interview "How to explain it"
    Estimating a training run: training compute is about 6ND (2N forward, 4N backward); the Chinchilla optimum is about 20 tokens per parameter, but to lower inference cost small models are generally "overtrained" (LLaMA-3-8B used about 15T tokens); mixed-precision AdamW takes about 16 bytes per parameter (inference needs only 2), so training cannot do without ZeRO / FSDP, TP and PP; MFU measures compute utilization. Connecting training-side designs (MTP, GQA, MoE) to inference cost earns extra credit.

## Exercises {#练习}

**1. Estimate.** Using the Chinchilla-optimal ratio (20 tokens per parameter), how large a model should a compute budget of 1e24 FLOPs train, and on how much data?

??? success "Answer"
    C = 6ND and D = 20N, so C = 120N² and N = √(C/120) ≈ 9.1e10 (about 91B parameters), D ≈ 1.8e12 (about 1.8T tokens).

    ```python
    import math
    C = 1e24
    N = math.sqrt(C / 120)
    print(f"N ≈ {N / 1e9:.0f}B, D ≈ {20 * N / 1e12:.1f}T")
    assert 90e9 < N < 92e9
    ```

**2. Food for thought.** A training batch is usually millions of tokens, while a decode batch in inference is only tens to hundreds of tokens. Why is GPU utilization usually much higher in training than in inference?

??? success "Answer"
    In training, the M dimension of each matrix multiplication (batch × sequence length) is in the millions, so each weight read is reused millions of times: fully compute bound, making full use of the Tensor Cores. In the decode phase of inference, each request has only 1 token per step, and all the requests in a batch add up to only tens to hundreds of tokens, so weight reads cannot be amortized enough: memory bound. The core ways to make decode more efficient (larger batches, quantization, speculative decoding verifying several tokens at once) all essentially raise the number of tokens computed per weight read, or reduce the bytes read.

## Summary {#小结}

- [x] Pretraining = minimizing next-token cross-entropy over more than ten trillion tokens; training compute is about 6ND.
- [x] The Chinchilla optimum is about 20 tokens per parameter, but to lower inference cost small models are generally "overtrained".
- [x] Mixed-precision AdamW training takes about 16 bytes per parameter, far more than inference's 2; it needs parallelism such as ZeRO/FSDP, TP and PP.
- [x] MFU measures compute utilization; training designs such as MTP can be used directly to speed up inference.
