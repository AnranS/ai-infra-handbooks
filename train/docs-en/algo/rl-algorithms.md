# Reinforcement-learning algorithms in depth: from PPO to GRPO, DAPO and GSPO

<p class="lead">Post-training a reasoning model is almost always reinforcement learning with verifiable rewards: sample a group of answers, score them right or wrong, and raise the probability of the right ones. After GRPO in 2024 came a string of improvements within a year: DAPO, Dr. GRPO, GSPO, CISPO. What each changes is usually one or two places in the loss function: how the advantage is computed, how the loss is averaged over tokens, whether the importance ratio is per token or per sequence, and what gets clipped. These details decide directly whether training collapses and whether the entropy does, and they also decide what the inference engine has to provide to the training side (the log probabilities at sampling time, for instance). This chapter computes the effect of each change in a numerical experiment.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. How is GRPO's advantage computed? What happens when a group of answers is all right or all wrong?
    2. What is the difference between averaging within each answer and then over answers, and averaging over all tokens at once? Why did DAPO switch to the latter?
    3. What are the properties of the KL divergence's k1, k2 and k3 estimators? Which does GRPO use?
    4. Why does GSPO use a sequence-level importance ratio? Why a geometric mean rather than a product?
    5. When the training and inference sides' probabilities disagree, how do TIS and MIS each correct for it? What does that cost?

??? success "Answers for the self-test (answer first, then open this)"
    1. Sample a group of answers to one question, and the advantage is the reward minus the group's mean divided by the group's standard deviation, with no critic. When all are right or all wrong the rewards within the group are identical, the advantages are all 0, and the group carries no learning signal (DAPO's dynamic sampling filters such groups out).
    2. Averaging by answer dilutes each token's weight in a long answer, under-penalising a long wrong answer and encouraging verbose errors; averaging over all tokens gives every token the same weight. That is exactly why DAPO switched.
    3. k1 $= -\log r$: unbiased but high-variance, with nearly half the samples negative. k2 $= \frac{1}{2}(\log r)^2$: non-negative and low-variance but biased. k3 $= (r-1) - \log r$: unbiased, non-negative and low-variance. GRPO uses k3.
    4. A per-token ratio is noisy and clipping throws away many tokens' gradients; the reward was given to the whole sequence anyway, so a sequence-level ratio is more consistent. But the product of the token ratios diverges exponentially with the length, so the geometric mean is used (the mean of the log ratios, exponentiated), which is independent of the length, with the clipping done at the sequence level.
    5. TIS truncates the training-to-inference probability ratio at a ceiling before multiplying it into the loss (biased, but with controlled variance); MIS simply drops the tokens or sequences whose ratios are anomalous. The costs: truncation introduces bias and dropping wastes samples, and both need the inference engine to return the log probabilities from sampling.

A six-panel strip first, then the chapter:

<!-- comic ../assets/comics/rl-algorithms.webp is in Chinese; put it back once the English version exists -->

## From PPO to GRPO {#从-ppo-到-grpo}

PPO's objective is: sample from the old policy $\pi_{\text{old}}$, compute an importance ratio $r_t = \pi_\theta(o_t) / \pi_{\text{old}}(o_t)$ per token, and maximise

$$\min\big(r_t A_t,\ \mathrm{clip}(r_t, 1-\epsilon, 1+\epsilon) A_t\big)$$

where the advantage $A_t$ is estimated by a value model (a critic) as large as the policy. **GRPO** (DeepSeekMath) does away with the critic: sample $G$ answers to one question and use the group's mean (and standard deviation) as the baseline, $A_i = (R_i - \mathrm{mean}(R)) / \mathrm{std}(R)$, with every token of one answer sharing that advantage; it then adds a KL penalty against a reference model. Dropping the critic halves the memory and the computation, at the price of several samples per question.

What each improvement changes:

| Algorithm | What it changes | Why |
| --- | --- | --- |
| **Dr. GRPO** | removes the normalisation by answer length and by standard deviation | the two introduce a length bias and a difficulty bias respectively |
| **DAPO** | the clipping's upper bound widened to $1 + 0.28$ (Clip-Higher); groups that are all right or all wrong filtered out and resampled (dynamic sampling); averaging over all tokens; a growing penalty on over-long answers; the KL removed | to prevent entropy collapse, to give every batch a usable signal, and to treat a long answer's tokens like a short one's |
| **GSPO** | the importance ratio and the clipping both move to **the sequence level**, with the ratio the geometric mean of the per-token ratios | the per-token ratio is noisy, especially for a mixture-of-experts model, and training is unstable |
| **CISPO** | clips the importance weight itself rather than the objective (the weight carries no gradient) | a clipped token still contributes a gradient rather than being thrown away entirely |

## How the advantage and the loss are averaged {#优势与损失的平均方式}

For one group of answers, here is each token's weight in the loss:

Change the rewards and the lengths and watch each token's weight move:

<div class="aig-widget" data-widget="grpo-adv"></div>

```python title="loss_aggregation.py"
import torch

# 4 answers sampled for one prompt (GRPO's group), where a reward of 1 means correct
rewards = torch.tensor([1.0, 0.0, 0.0, 1.0])
lengths = torch.tensor([200, 50, 2000, 400])                  # the answers differ greatly in length

adv = rewards - rewards.mean()
adv_std = adv / (rewards.std() + 1e-6)
print("组内优势（减均值）：", adv.tolist(), " 再除以标准差：", [round(a, 2) for a in adv_std.tolist()])

# each token's gradient weight = the advantage x that token's coefficient in the loss
seq_mean = adv_std / lengths / len(lengths)                   # GRPO: average within each answer first, then over the answers
tok_mean = adv_std / lengths.sum()                            # DAPO: average over all of the tokens together
print("回答   奖励  长度   每个 token 的权重（序列平均）  （token 平均）  整条回答的总权重（序列平均）  （token 平均）")
for i in range(4):
    print(f"{i:4d} {rewards[i]:5.0f} {lengths[i]:5d}   {seq_mean[i]:+26.2e}  {tok_mean[i]:+13.2e}"
          f"  {seq_mean[i] * lengths[i]:+27.3f}  {tok_mean[i] * lengths[i]:+13.3f}")

# an all-right or all-wrong group: the advantages are all 0 and the group contributes nothing to the gradient
for group in ([1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]):
    g = torch.tensor(group)
    print(f"奖励 {group}：优势 {(g - g.mean()).tolist()}")
```

```text title="output"
组内优势（减均值）： [0.5, -0.5, -0.5, 0.5]  再除以标准差： [0.87, -0.87, -0.87, 0.87]
回答   奖励  长度   每个 token 的权重（序列平均）  （token 平均）  整条回答的总权重（序列平均）  （token 平均）
   0     1   200                    +1.08e-03      +3.27e-04                       +0.217         +0.065
   1     0    50                    -4.33e-03      -3.27e-04                       -0.217         -0.016
   2     0  2000                    -1.08e-04      -3.27e-04                       -0.217         -0.654
   3     1   400                    +5.41e-04      +3.27e-04                       +0.217         +0.131
奖励 [1.0, 1.0, 1.0, 1.0]：优势 [0.0, 0.0, 0.0, 0.0]
奖励 [0.0, 0.0, 0.0, 0.0]：优势 [0.0, 0.0, 0.0, 0.0]
```

- **A group that is all right or all wrong has advantages of 0 throughout**, contributing nothing to the gradient while still costing the sampling and the forward pass. Later in training most questions are either always right or always wrong, so such groups grow more common and the effective batch shrinks. DAPO's **dynamic sampling** filters them out and keeps sampling until a batch is full.
- **Averaging by sequence** (the original GRPO) gives each answer the same total weight: a wrong answer of 2000 tokens gives each of its tokens a fortieth of what a wrong answer of 50 tokens gives. The model therefore penalises long wrong answers very lightly and tends to write more and more. **Averaging by token** (DAPO) treats all tokens alike, so a long wrong answer is penalised more in total.
- Dividing by the group's standard deviation amplifies the advantage for nearly-all-right and nearly-all-wrong groups (where the standard deviation is small), and Dr. GRPO argues this introduces a bias by question difficulty and recommends subtracting the mean alone.

## The KL penalty: three estimators {#kl-惩罚三种估计量}

GRPO estimates the KL of the current policy $q$ against the reference model $p$ at each token. Only the sampled token is available, so a single-sample estimator is needed. With $r = p(x) / q(x)$ and $x \sim q$:

The three curves drawn out:

<div class="aig-widget" data-widget="kl-estimators"></div>

```python title="kl_estimators.py"
import torch

torch.manual_seed(0)
V = 1000
p_ref = torch.randn(V).softmax(-1)                            # the reference model's distribution at one position
q = (p_ref.log() + 0.3 * torch.randn(V)).softmax(-1)          # the current policy: not far from the reference model
exact = (q * (q / p_ref).log()).sum()                         # the exact value of KL(q ‖ p_ref)

x = torch.multinomial(q, 100_000, replacement=True)           # the tokens sampled from the current policy (the rollout)
r = p_ref[x] / q[x]
estimators = {"k1 = -log r": -r.log(), "k2 = (log r)² / 2": r.log() ** 2 / 2, "k3 = (r - 1) - log r": (r - 1) - r.log()}
print(f"精确的 KL：{exact:.4f}")
for name, v in estimators.items():
    print(f"{name:22s} 均值 {v.mean():.4f}，标准差 {v.std():.4f}，负值的比例 {(v < 0).float().mean():.0%}")
```

```text title="output"
精确的 KL：0.0502
k1 = -log r            均值 0.0497，标准差 0.3221，负值的比例 44%
k2 = (log r)² / 2      均值 0.0531，标准差 0.0744，负值的比例 0%
k3 = (r - 1) - log r   均值 0.0505，标准差 0.0697，负值的比例 0%
```

- **k1** $= -\log r$ is unbiased but high-variance, and nearly half the samples are negative, so as a loss term it rewards departing from the reference model on individual tokens.
- **k2** $= \frac{1}{2}(\log r)^2$ is non-negative and low-variance, but biased.
- **k3** $= (r - 1) - \log r$ is unbiased, non-negative and low-variance (since $\mathbb{E}_q[r - 1] = 0$, it amounts to adding a zero-mean control variate to k1). This is what GRPO uses.

Reinforcement learning for reasoning models often wants the policy as far from the initial model as possible (to learn new ways of reasoning), and DAPO and others simply drop the KL term. If the KL is kept, mind its gradient: differentiating k3 directly as a loss term and taking the expectation over $q$'s samples gives the gradient of the forward KL$(p \| q)$, $-\sum_x p(x) \nabla \log q(x)$, rather than of the reverse KL$(q \| p)$ it estimates. One quantity is estimated numerically and another is optimised, a detail that recent analyses have pointed out.

## The importance ratio: per token or per sequence {#重要性比按-token-还是按序列}

PPO and GRPO compute and clip the ratio per token. After an update, every token's log ratio carries a little noise (finite samples, the learning rate, and in a mixture of experts the routing changing as well):

```python title="ratios.py"
import torch

torch.manual_seed(0)
# simulating a batch of answers: sampled from the old policy, and after a few updates each token's log ratio log(π/π_old) carries a little noise
# (in a mixture of experts the same token may route to different experts under the old and new policies, which is noisier still)
for sigma in (0.02, 0.2):
    print(f"每个 token 的 log 比率 ~ N(0, {sigma}²)：")
    for L in (100, 1000, 4000):
        log_r = torch.randn(512, L) * sigma                   # 512 answers, each of length L
        clipped = ((log_r.exp() < 0.8) | (log_r.exp() > 1.28)).float().mean()   # per-token clipping (ε_low 0.2, ε_high 0.28)
        product = log_r.sum(1).exp()                          # the strict sequence-level importance weight: the product of the per-token ratios
        geo = log_r.mean(1).exp()                             # GSPO: the geometric mean, normalised by the length
        print(f"  L={L:4d}：超出裁剪范围的 token {clipped:5.1%}；比率的乘积在 [{product.min():.0e}, {product.max():.0e}]；"
              f"几何平均在 [{geo.min():.4f}, {geo.max():.4f}]")
```

```text title="output"
每个 token 的 log 比率 ~ N(0, 0.02²)：
  L= 100：超出裁剪范围的 token  0.0%；比率的乘积在 [5e-01, 2e+00]；几何平均在 [0.9932, 1.0057]
  L=1000：超出裁剪范围的 token  0.0%；比率的乘积在 [2e-01, 6e+00]；几何平均在 [0.9983, 1.0018]
  L=4000：超出裁剪范围的 token  0.0%；比率的乘积在 [2e-02, 6e+01]；几何平均在 [0.9990, 1.0010]
每个 token 的 log 比率 ~ N(0, 0.2²)：
  L= 100：超出裁剪范围的 token 24.0%；比率的乘积在 [2e-03, 5e+02]；几何平均在 [0.9392, 1.0639]
  L=1000：超出裁剪范围的 token 24.1%；比率的乘积在 [9e-10, 1e+07]；几何平均在 [0.9794, 1.0164]
  L=4000：超出裁剪范围的 token 24.1%；比率的乘积在 [3e-17, 5e+22]；几何平均在 [0.9905, 1.0132]
```

- Under per-token clipping, enough noise puts a quarter of the tokens outside the clipping range with their gradients thrown away, regardless of how long the answer is.
- By definition an answer's importance weight should be **the product of all of its tokens' ratios**. But the product diverges exponentially with the length: at 4000 tokens it spans 39 orders of magnitude, which is unusable.
- **GSPO** uses the geometric mean $\big(\pi_\theta(y) / \pi_{\text{old}}(y)\big)^{1/|y|}$: normalised by the length, it stays near 1 and gets steadier the longer the answer. It clips at the sequence level with a very narrow range (the paper uses $3 \times 10^{-4}$ and $4 \times 10^{-4}$). The fraction of answers clipped out entirely is actually far larger than the fraction of tokens GRPO clips, and yet training is more efficient, which says the per-token ratio is too noisy and the gradients left carry little useful information. Qwen3's mixture-of-experts models train with it and no longer need routing replay or other measures to stabilise the mixture.

**CISPO** (MiniMax-M1) takes another angle: the objective is written as $\mathrm{sg}\big(\mathrm{clip}(r_t)\big) \cdot A_t \cdot \log \pi_\theta(o_t)$, clipping the importance weight (which carries no gradient) so that every token's gradient survives. PPO-style clipping throws away any token whose ratio leaves the range entirely, and those tokens are often the pivotal words in a chain of reasoning ("however", "wait, let me check that again"), so CISPO argues that keeping their gradients matters for learning to reflect.

## When the training and inference sides disagree {#训练端与推理端不一致}

Reinforcement learning's samples come from the inference engine and the probabilities are recomputed on the training side. The two implementations differ numerically (kernels, precision, batch composition), so the same token's probability differs (the inference-systems handbook measures up to 10% in bf16 on Qwen3-0.6B, see [Inference in reinforcement-learning training](serving://topics/rl-rollout/)). So the training is really off-policy: the samples come from the inference side's distribution $\mu$ while the gradient to estimate is the training-side policy $\pi$'s.

```python title="mismatch.py"
import torch

torch.manual_seed(0)
V = 200
logits = torch.randn(V) * 2
pi = logits.softmax(-1)                                        # the training side's policy
noise = 0.2 * torch.randn(V)
noise[logits.argsort()[-8:-5]] -= 3                            # a few tokens' probabilities are badly underestimated by the inference side (a kernel's precision problem, say)
mu = (logits + noise).softmax(-1)                              # the distribution the inference engine actually sampled from
A = torch.randn(V)                                             # each action's advantage
true_grad = (pi * A)[:, None] * (torch.eye(V) - pi[None, :])  # the exact policy gradient E_π[A ∇log π], differentiated with respect to the logits
true_grad = true_grad.sum(0)

w = pi / mu
print(f"重要性比 π/μ：中位数 {w.median():.2f}，最大 {w.max():.1f}，最小 {w.min():.2f}")


def estimate(kind, n=256, C=2.0, gen=None):
    a = torch.multinomial(mu, n, replacement=True, generator=gen)   # only the inference side's samples are available
    score = torch.eye(V)[a] - pi                               # the gradient of ∇ log π(a) with respect to the logits
    ratio = w[a]
    weight = {"不修正": torch.ones(n), "重要性采样": ratio, "TIS（截断到 2）": ratio.clamp(max=C),
              "MIS（比率超过 2 的丢掉）": ratio * (ratio <= C)}[kind]
    return (weight * A[a])[:, None].mul(score).mean(0)


gen = torch.Generator().manual_seed(1)
for kind in ("不修正", "重要性采样", "TIS（截断到 2）", "MIS（比率超过 2 的丢掉）"):
    est = torch.stack([estimate(kind, gen=gen) for _ in range(400)])
    bias = (est.mean(0) - true_grad).norm() / true_grad.norm()
    noise = (est - est.mean(0)).norm(dim=1).mean() / true_grad.norm()
    print(f"{kind:18s} 偏差 {bias:5.1%}，单个 batch 的噪声 {noise:5.1%}")
```

```text title="output"
重要性比 π/μ：中位数 0.95，最大 19.7，最小 0.56
不修正                偏差 42.5%，单个 batch 的噪声 30.3%
重要性采样              偏差  2.4%，单个 batch 的噪声 54.8%
TIS（截断到 2）         偏差 23.6%，单个 batch 的噪声 30.6%
MIS（比率超过 2 的丢掉）    偏差 26.8%，单个 batch 的噪声 29.6%
```

- **No correction**: the gradient has a large systematic bias which accumulates, quietly taking the training off course and collapsing it in the worst case.
- **Importance sampling** (multiplying by $\pi / \mu$): unbiased, but a few samples with very large ratios nearly double the variance.
- **TIS** (truncated importance sampling, with the ratio truncated at $C$) and **MIS** (masked importance sampling, dropping the samples whose ratio exceeds $C$): trading some bias for controlled variance. In a real system the ratios are overwhelmingly close to 1, so truncation and dropping affect only a few anomalous samples and the bias is far smaller than in this deliberately exaggerated example.

This requires the inference engine to **return each token's log probability from sampling**. The bias of asynchronous reinforcement learning (where inference uses weights a few versions behind, see [Asynchronous reinforcement learning](serving://frontier/rl-async/) in the inference-systems handbook) is the same class of problem, corrected the same way with a cap on how many versions behind it may be. The more fundamental answer is to make both sides compute identically: batch-invariant deterministic kernels on the inference side and the same implementation on the training side (see [Deterministic inference](serving://topics/deterministic/) in the inference-systems handbook).

!!! interview "How to explain it"
    To explain what is wrong with GRPO and how the later algorithms fixed it, go by the loss function's four parts: **the advantage** (the group mean subtracted; an all-right or all-wrong group carries no signal, hence DAPO's dynamic sampling; dividing by the standard deviation adds a difficulty bias, which Dr. GRPO removes); **the averaging** (averaging by answer shrinks each token's weight in a long answer, hence averaging by token); **the importance ratio and clipping** (the per-token ratio is noisy, hence GSPO's sequence-level geometric mean; clipping throws away pivotal tokens' gradients, hence CISPO clipping the weight; too tight an upper bound collapses the entropy, hence DAPO's Clip-Higher); and **the KL** (the k3 estimator; often dropped in reasoning reinforcement learning). Then add the systems point: a disagreement between the training and inference sides' probabilities needs a TIS or MIS correction, which requires the inference engine to return log probabilities.

## Exercises {#练习}

**1. Why does Clip-Higher prevent entropy collapse?** Two tokens with old probabilities of 0.01 and 0.9, both with a positive advantage. At $\epsilon = 0.2$, how high can each rise in one update? And with the upper bound widened to $0.28$?

??? success "Answer"
    The ratio's ceiling $1 + \epsilon$ limits the relative change: 0.01 reaches at most 0.012 (0.0128 after widening), while 0.9 reaches at most 1.08 and is effectively unconstrained (a probability cannot exceed 1). A high-probability token is easily pushed higher while a low-probability exploratory token rises only a little each time, so the policy becomes more and more certain and the entropy falls quickly. Widening the upper bound lets low-probability tokens rise faster and the entropy fall more slowly. The lower bound is not widened, to stop a low-probability token being pushed to near 0 in one step and never sampled again.

**2. Why does the original GRPO make answers longer and longer?** Explain with this chapter's weight table.

??? success "Answer"
    Averaging by answer fixes a wrong answer's total penalty regardless of its length, so spread over the tokens it is inversely proportional to the length: for the same mistake, the longer the answer, the smaller the penalty per token, and the model has no pressure to shorten a wrong answer; meanwhile a short correct answer gives each of its tokens a larger reward. The two together push wrong answers toward being longer. Averaging by token (DAPO) or normalising by a fixed maximum length (Dr. GRPO) removes the bias.

**3. A mixture-of-experts model doing reinforcement learning has its loss and reward collapse halfway through training. Where would you look?**

??? success "Answer"
    Start with the difference between the training and inference sides' log probabilities (for a mixture of experts the routing can differ between the two for the same token, so the difference is far larger than for a dense model) and with the importance ratios' distribution: is the ratios' tail lengthening, and is the fraction of clipped tokens rising? The matching measures: switch to a sequence-level ratio (GSPO), pass the inference side's routing back to the training side (routing replay), add a TIS or MIS correction or drop the anomalous samples, and make the inference side's numbers more consistent (deterministic kernels, an fp32 lm_head). Then look at whether the entropy fell quickly just before the collapse (Clip-Higher, a lower learning rate), and whether the reward is being gamed (reward hacking, a format reward being exploited).

## Summary {#小结}

- [x] GRPO replaces the critic with a group baseline; an all-right or all-wrong group carries no signal (DAPO's dynamic sampling), and dividing by the standard deviation adds a difficulty bias (which Dr. GRPO removes).
- [x] Averaging by answer shrinks each token's weight in a long answer and encourages verbose errors; DAPO switched to averaging by token.
- [x] The KL's k3 estimator is unbiased, non-negative and low-variance; reasoning reinforcement learning often drops the KL term entirely.
- [x] The per-token ratio is noisy and throws gradients away, while the sequence-level product diverges with the length, so GSPO uses the geometric mean with sequence-level clipping; CISPO clips the weight and keeps every token's gradient; Clip-Higher prevents entropy collapse.
- [x] A disagreement between the training and inference sides' probabilities makes the reinforcement learning off-policy: importance sampling is unbiased but high-variance, while TIS and MIS truncate or drop the anomalous ratios, and both need the inference engine to return log probabilities.
