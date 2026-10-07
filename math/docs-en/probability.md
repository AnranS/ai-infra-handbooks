# Probability and sampling

<p class="lead">A language model outputs a probability distribution, and generating text means sampling from it again and again. Temperature, top-p, the proof that speculative decoding is correct, the importance sampling ratios in RL, and the error bars on load-test results all rest on the same set of probability tools. This chapter starts from discrete distributions, derives the sampling algorithms used in inference one by one, and checks each result with Monte Carlo experiments and real model distributions.</p>

!!! question "Self-test: if you can answer these, skip this chapter"
    1. What are the ways to sample from a discrete distribution? Why does the Gumbel-max trick work?
    2. If you estimate a probability from n samples, roughly how large is the error?
    3. How is the probability of a sequence computed from the conditional probabilities of each step? Why does greedy decoding not necessarily find the most probable sequence?
    4. What is the acceptance rate of speculative sampling equal to?
    5. What is importance sampling for? Why is the estimate unreliable when the weights have a large variance?

??? success "Answers (try first, then expand to compare)"
    1. Inverse CDF (prefix sums plus a binary search with one uniform random number), Gumbel-max ($\arg\max_i (\log p_i + G_i)$ with independent Gumbel noise $G_i$), the exponential race ($\arg\max_i p_i / E_i$), the alias method and others. Gumbel-max works because, after adding Gumbel noise, the probability that element i is the largest is exactly the softmax probability $p_i$.
    2. The standard error is about $\sqrt{p(1-p)/n}$ and falls as $1/\sqrt{n}$: 10 times the precision takes 100 times the samples.
    3. The probability of a sequence is the product of the conditional probabilities of each step (a sum in log space). Greedy picks the maximum at every step, but choosing a slightly less likely token early can lead to a far more likely continuation, so greedy does not necessarily find the sequence with the highest overall probability.
    4. The expected acceptance rate is $\sum_x \min(p(x), q(x)) = 1 - \mathrm{TV}(p, q)$: the closer the draft distribution $q$ is to the target distribution $p$, the more is accepted.
    5. It estimates an expectation under distribution $p$ using samples from distribution $q$, by multiplying each sample by the weight $p/q$. When the two distributions differ a lot, a few samples get huge weights, the variance of the estimate explodes and the result is unreliable; truncating the weights (trading bias for variance) is common.

## Discrete distributions, expectation and variance {#离散分布期望与方差}

At every position, a language model gives a **discrete distribution** $p(x)$ over the vocabulary, with $\sum_x p(x) = 1$. For any function $f$, the **expectation** is $\mathbb{E}_p[f] = \sum_x p(x) f(x)$ and the **variance** is $\text{Var}[f] = \mathbb{E}[f^2] - \mathbb{E}[f]^2$.

Logits become a distribution through softmax, and temperature $T$ divides the logits by $T$ before the softmax: $p_T(x) \propto e^{z_x / T}$. $T < 1$ makes the distribution peakier, $T > 1$ flatter. Look at the effect of temperature at one position of a real model:

```python
import copy
import math
import re
import torch
from transformers import AutoTokenizer
from mini_llm import Transformer

torch.set_num_threads(16)
path = "models/Qwen3-0.6B"
tok = AutoTokenizer.from_pretrained(path)
model = Transformer.from_pretrained(path)
raw = open("docs/assets/sample-passage.txt").read()   # frozen sample text: a snapshot of the language-model chapter, so editing that chapter no longer changes the numbers here
ids = tok(re.sub(r"[#*`>|\-\[\]()!]", "", raw)).input_ids[:400]          # a passage from the LLM book
with torch.no_grad():
    logits = model(torch.tensor([ids]))[0]                                 # logits at every position

for T in (0.5, 1.0, 1.5):
    p = (logits[49] / T).softmax(-1).sort(descending=True).values           # position 50
    print(f"T = {T}：最可能的 token 概率 {p[0]:.3f}，凑够 90% 概率需要 {int((p.cumsum(0) < 0.9).sum()) + 1} 个 token")
```

```text title="output"
T = 0.5：最可能的 token 概率 0.302，凑够 90% 概率需要 4 个 token
T = 1.0：最可能的 token 概率 0.146，凑够 90% 概率需要 110 个 token
T = 1.5：最可能的 token 概率 0.044，凑够 90% 概率需要 6708 个 token
```

At the same position, going from temperature 0.5 to 1.5 turns the distribution from "4 tokens hold 90%" into "more than six thousand tokens share 90% of the probability". That is why high-temperature sampling easily "talks nonsense": the total probability of the many unlikely tokens becomes substantial, which is the problem truncations such as top-p and min-p solve (see [decoding and sampling](llm://inference/decoding/)).

## Look at the shape first: what temperature, top-k and top-p each do {#先看形状温度top-ktop-p-各自在做什么}

At every step the model outputs a list of logits, and softmax turns them into probabilities: $p_i = e^{z_i / T} / \sum_j e^{z_j / T}$. Temperature divides inside the exponent, so what it changes is "whether the gaps are amplified or flattened": $T < 1$ makes the large ones larger (closer to greedy), $T > 1$ spreads the distribution out (more diverse). top-k and top-p cut off the tail before sampling.

Drag the three sliders below to see the shape of the same logits under different settings, and how the entropy changes with them:

<div class="aig-widget" data-widget="softmax"></div>

Three sentences to remember how they divide the work: **temperature changes the shape, top-k and top-p change the candidate set**; top-k keeps a fixed number, so with a peaked distribution it keeps a pile of almost impossible words; top-p keeps by cumulative probability, only one or two when the distribution is peaked and more when it is flat, which is why it is more common. $T \to 0$ is equivalent to greedy decoding: deterministic, but prone to repetition.

## Sampling algorithms {#采样算法}

Given a distribution $p$, how do you generate a random sample from it? Three methods:

1. **Inverse CDF**: compute the cumulative distribution $F(x) = \sum_{y \le x} p(y)$, draw a uniform random number $u \sim U(0,1)$, and return the first $x$ with $F(x) \ge u$. Intuitive, but it needs prefix sums and a search;
2. **Gumbel-max**: add independent Gumbel noise $g = -\log(-\log u)$ to every logit and take the argmax. $\arg\max_x (z_x + g_x)$ picks $x$ with exactly probability $\text{softmax}(z)_x$;
3. **The exponential race**: take $\arg\max_x p(x) / E_x$ with $E_x$ exponentially distributed with rate 1. $E_x / p(x)$ is exponential with rate $p(x)$, and the probability that the smallest of several independent exponentials is $x$ equals $p(x) / \sum_y p(y)$. It is really the same thing as Gumbel-max: $-\log E$ has a Gumbel distribution. vLLM's sampler uses this method (see [batched sampling](serving://engine/sampler-api/#批量采样) in the inference systems book).

The last two need only elementwise operations and one argmax, with no prefix sums and no CPU-GPU synchronization, which suits batched execution on a GPU. A Monte Carlo experiment confirms that all three give the same distribution:

```python
torch.manual_seed(0)
p = torch.tensor([0.5, 0.25, 0.15, 0.07, 0.03])
n = 200_000
u = torch.rand(n, 1)
inverse_cdf = torch.searchsorted(p.cumsum(0), u).squeeze(1).clamp(max=len(p) - 1)
gumbel = (p.log() - torch.log(-torch.log(torch.rand(n, len(p))))).argmax(-1)
race = (p / torch.empty(n, len(p)).exponential_()).argmax(-1)
for name, samples in [("逆 CDF", inverse_cdf), ("Gumbel-max", gumbel), ("指数竞赛", race)]:
    freq = torch.bincount(samples, minlength=len(p)).float() / n
    print(f"{name:10s} {[round(f, 3) for f in freq.tolist()]}  与 p 的最大偏差 {(freq - p).abs().max():.4f}")
```

```text title="output"
逆 CDF      [0.499, 0.25, 0.151, 0.07, 0.03]  与 p 的最大偏差 0.0014
Gumbel-max [0.501, 0.249, 0.151, 0.07, 0.03]  与 p 的最大偏差 0.0015
指数竞赛       [0.501, 0.249, 0.15, 0.069, 0.03]  与 p 的最大偏差 0.0006
```

## Monte Carlo error {#蒙特卡洛的误差}

If you estimate the probability $p$ of an event by its frequency $\hat{p}$ in $n$ independent samples, the standard deviation of $\hat p$ is $\sqrt{p(1-p)/n}$: **the error falls with the square root of the number of samples**, so 100 times the samples only shrinks the error 10 times. The 200,000 samples above give an error of about 0.001, consistent with the deviations we saw.

```python
p_true = 0.3
for n in (100, 10_000):
    estimates = (torch.rand(2000, n) < p_true).float().mean(1)     # repeat the experiment 2000 times, sampling n each time
    print(f"n = {n:6d}：估计值的标准差 {estimates.std():.4f}，理论值 {math.sqrt(p_true * (1 - p_true) / n):.4f}")
```

```text title="output"
n =    100：估计值的标准差 0.0458，理论值 0.0458
n =  10000：估计值的标准差 0.0046，理论值 0.0046
```

This result is everywhere in inference work: the acceptance rate of speculative decoding, the P99 latency of a load test and accuracy on an evaluation set are all "quantities estimated from finite samples", and all have error bars. With only 200 questions in an evaluation set, a 70% accuracy has a standard error of about 3 percentage points, so a 2-point difference between two approaches does not show which is better (for the statistics of performance measurement, see [the math of performance and serving](performance-math.md)).

## The probability of a sequence {#序列的概率}

A language model defines the probability of a whole sequence with the **chain rule**:

$$
p(x_1, \dots, x_T) = \prod_{t=1}^{T} p(x_t \mid x_{<t}), \qquad \log p(x_{1:T}) = \sum_{t} \log p(x_t \mid x_{<t})
$$

In practice everything is computed in log space; otherwise multiplying hundreds of probabilities below 1 underflows to 0. An important consequence: **picking the most probable token at every step (greedy) does not necessarily give the most probable sequence**:

```pycon
>>> # step 1: A has probability 0.6, B 0.4; after A the best second step is only 0.5, after B it is 0.9
>>> paths = {("A", "a1"): 0.6 * 0.5, ("A", "a2"): 0.6 * 0.5, ("B", "b1"): 0.4 * 0.9, ("B", "b2"): 0.4 * 0.1}
>>> max(paths, key=paths.get), round(max(paths.values()), 2)          # the most probable sequence
(('B', 'b1'), 0.36)
>>> round(paths[("A", "a1")], 2)                                     # the sequence greedy decoding finds
0.3
```

Beam search eases this by keeping several candidate paths at once, but chat models usually sample rather than use beam search: the most probable sequence is often repetitive, dull text.

## Rejection sampling and speculative decoding {#拒绝采样与投机解码}

**Rejection sampling**: you want to sample from distribution $p$ but can only conveniently sample from $q$. Take a sample $x$ from $q$, accept it with a certain probability, and when you reject it, make up for it some other way, so that the final result follows $p$ exactly. Speculative sampling is a special case: the draft model proposes $x \sim q$, which is accepted with probability $\min(1, p(x)/q(x))$, and on rejection a new sample is drawn from $\text{norm}(\max(0, p - q))$. The [inference serving](llm://inference/serving/#投机解码) chapter verifies with Monte Carlo that the output distribution equals $p$.

Its **acceptance rate** is

$$
\sum_x q(x) \min\!\left(1, \frac{p(x)}{q(x)}\right) = \sum_x \min(p(x), q(x)) = 1 - \text{TV}(p, q)
$$

where $\text{TV}(p, q) = \frac{1}{2}\sum_x |p(x) - q(x)|$ is the **total variation distance** between the two distributions. So the more the draft model "resembles" the target model, the higher the acceptance rate. Measure it on a real model: the target is Qwen3-0.6B in FP32, the draft is its INT4-quantized version, and the acceptance rate is computed at 400 positions of a text:

```python
from quant import fake_quant_int

draft = copy.deepcopy(model)
for layer in draft.layers:
    a, f = layer.self_attn, layer.mlp
    for lin in (a.q_proj, a.k_proj, a.v_proj, a.o_proj, f.gate_proj, f.up_proj, f.down_proj):
        lin.weight.data = fake_quant_int(lin.weight.data, 4, "group")
with torch.no_grad():
    p_target, q_draft = logits.softmax(-1), draft(torch.tensor([ids]))[0].softmax(-1)
accept = torch.minimum(p_target, q_draft).sum(-1)
tv = 0.5 * (p_target - q_draft).abs().sum(-1)
print(f"接受率：平均 {accept.mean():.2f}，中位数 {accept.median():.2f}，最低 {accept.min():.2f}")
print(f"1 - 总变差距离 的平均值：{(1 - tv).mean():.2f}")
assert torch.allclose(accept, 1 - tv, atol=1e-3)
```

```text title="output"
接受率：平均 0.70，中位数 0.70，最低 0.03
1 - 总变差距离 的平均值：0.70
```

On average each draft token is accepted with probability about 70%, exactly "1 − total variation distance"; but at a few positions the two models' distributions differ a lot and the acceptance rate drops to 3%.

## Importance sampling {#重要性采样}

You want $\mathbb{E}_p[f]$, but your samples come from another distribution $q$. Using

$$
\mathbb{E}_p[f] = \sum_x p(x) f(x) = \sum_x q(x) \frac{p(x)}{q(x)} f(x) = \mathbb{E}_q\!\left[w(x) f(x)\right], \quad w = \frac{p}{q}
$$

you just multiply each sample by its **importance weight** $w$. The ratio $\pi_\theta / \pi_\text{old}$ in PPO and GRPO for RL, and the ratio that corrects for "the inference engine and the training framework disagreeing on probabilities" (see [inference inside RL training](serving://topics/rl-rollout/#问题二训练与推理的概率不一致) in the inference systems book), are both importance weights.

The problem is that when $q$ and $p$ differ a lot, a few samples get huge weights and the variance of the estimate explodes. One measure is the **effective sample size** $\text{ESS} = (\sum w)^2 / \sum w^2$:

```python
torch.manual_seed(0)
values = torch.arange(10).float()                                   # f(x) = x
p = torch.softmax(torch.linspace(0, 2, 10), 0)                      # target distribution
true_mean = (p * values).sum().item()
for shift in (0.0, 1.0, 3.0):
    q = torch.softmax(torch.linspace(0, 2, 10) - shift * torch.linspace(0, 1, 10), 0)   # sampling distributions less and less like p
    x = torch.multinomial(q, 1000, replacement=True)
    w = p[x] / q[x]
    estimate = (w * values[x]).mean().item()
    ess = (w.sum() ** 2 / (w ** 2).sum()).item()
    clipped = (w.clamp(max=2.0) * values[x]).mean().item()           # truncated weights: smaller variance, but biased
    print(f"偏移 {shift}：估计 {estimate:.2f}（真值 {true_mean:.2f}），有效样本数 {ess:6.0f} / 1000，截断后估计 {clipped:.2f}")
```

```text title="output"
偏移 0.0：估计 6.18（真值 6.20），有效样本数   1000 / 1000，截断后估计 6.18
偏移 1.0：估计 6.11（真值 6.20），有效样本数    922 / 1000，截断后估计 6.11
偏移 3.0：估计 6.60（真值 6.20），有效样本数    489 / 1000，截断后估计 4.98
```

The further $q$ is from $p$, the smaller the effective sample size and the less stable the estimate; capping the weights at some limit (PPO's clip, TIS) controls the variance but introduces bias. This is the mathematical background of the various "ratio clipping" schemes in RL algorithms.

!!! interview "In an interview"
    Questions about sampling: on a GPU, sampling from a discrete distribution usually uses Gumbel-max or the exponential race (`argmax(log p + Gumbel)`, `argmax(p / Exp)`), with no prefix sums or synchronization; the probability of a sequence is a product of conditional probabilities, and greedy does not necessarily find the most probable sequence; the acceptance rate of speculative sampling is $1 - \mathrm{TV}(p, q)$, higher the closer the draft distribution; importance sampling weights by $p/q$ and truncates when the weights' variance is large, which is what TIS does to correct the probability mismatch between the inference side and the training side in RL training.

## Exercises {#练习}

**1. The acceptance rate after top-k.** If both the target model and the draft model apply top-k truncation before sampling, how does the acceptance rate change?

??? success "How to think about it"
    The acceptance rate is still $\sum_x \min(p'(x), q'(x))$ between the two truncated distributions. If their top-k sets largely overlap, truncation moves the tail's probability onto the head and the two distributions are often closer, so the acceptance rate may rise; if the top-k sets differ a lot, the overlap shrinks and the acceptance rate falls. Note what correctness requires: the $p$ used for verification and the $q$ used for drafting must both be **the distributions actually sampled from** (including temperature and truncation); otherwise the output distribution no longer equals the target distribution.

**2. How many samples do you need?** You want to compare the mean accepted length of two speculative decoding configurations, which are about 2.5 and 2.6, and the standard deviation of the accepted length per verification is about 1.2. How many verifications does each configuration need at least, for the difference to exceed 2 standard errors?

??? success "Answer"
    The standard error of the difference of two means is about $\sqrt{2} \times 1.2 / \sqrt{n}$. Requiring $0.1 > 2 \times \sqrt{2} \times 1.2 / \sqrt{n}$ gives $n > (2 \times 1.7 / 0.1)^2 \approx 1150$. Each configuration needs more than a thousand verifications (that is, several thousand generated tokens); a comparison with fewer is not reliable.

## Summary {#小结}

- [x] Temperature changes how peaked the distribution is; at high temperature the total probability of the many unlikely tokens becomes substantial and needs truncating.
- [x] Inverse CDF, Gumbel-max and the exponential race all sample exactly from a discrete distribution, and the last two suit batched execution on a GPU.
- [x] Monte Carlo error falls as $1/\sqrt{n}$; every quantity estimated from samples has an error bar.
- [x] The probability of a sequence is a product of conditional probabilities (a sum in log space); greedy does not necessarily find the most probable sequence.
- [x] The acceptance rate of speculative sampling = $1 - \text{TV}(p, q)$; importance sampling weights by $p/q$ and truncates when the weights' variance is large, trading bias for variance.
