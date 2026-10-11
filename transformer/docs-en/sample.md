# 11. Sampling and stopping

<p class="lead">So far we have always taken the top score (greedy). A real chat model <strong>samples from the distribution</strong>: the temperature decides how sharp the distribution is, top-k / top-p cut off the long tail, the repetition and presence penalties keep it from going round in circles, and it stops at a stop token. This chapter writes all of that as <code>sample()</code> and <code>generate()</code> — every later chapter generates with them.</p>

!!! note "What this chapter runs"
    `ch11_sample.py`, about 30 seconds.

@@files@@

## From scores to one token {#从分数到一个-token}

```python title="qwen/sample.py"
"""采样：从 logits 里挑下一个 token。温度、top-k、top-p、重复惩罚，以及什么时候停。"""
import torch

from qwen.cache import State


def sample(logits, temperature=1.0, top_k=0, top_p=1.0, repetition_penalty=1.0, presence_penalty=0.0, generated=()):
    """logits: [词表] 的分数 -> 一个 token id。temperature=0 就是贪心（永远取最高分）。"""
    logits = logits.float().clone()
    if generated and (repetition_penalty != 1.0 or presence_penalty):
        seen = torch.tensor(sorted(set(generated)), device=logits.device)
        s = logits[seen]
        s = torch.where(s > 0, s / repetition_penalty, s * repetition_penalty)             # repetition penalty (multiplicative): seen tokens get positive scores lowered and negative scores pushed further down
        logits[seen] = s - presence_penalty                                                # presence penalty (subtractive): a flat deduction for anything seen; Qwen's model card recommends this one
    if temperature <= 0:
        return int(logits.argmax())
    logits = logits / temperature                                                          # the lower the temperature, the sharper the distribution
    if top_k > 0:
        kth = logits.topk(top_k).values[-1]
        logits[logits < kth] = float("-inf")                                               # keep only the k highest scores
    if top_p < 1.0:
        sorted_logits, order = logits.sort(descending=True)
        probs = sorted_logits.softmax(-1)
        drop = probs.cumsum(-1) - probs > top_p                                            # drop everything after the cumulative probability passes p (keep the one that crosses the line)
        sorted_logits[drop] = float("-inf")
        logits = torch.full_like(logits, float("-inf")).scatter(0, order, sorted_logits)
    return int(torch.multinomial(logits.softmax(-1), 1))


def generate(model, ids, max_new_tokens=200, stop=(), on_token=None, **kw):
    """ids: [1, T] 的提示词 -> 生成的 token 列表。kw 原样传给 sample；每生成一个 token 调一次 on_token。"""
    state = State(model)
    logits = model(ids, state)[0, -1]                                                      # prefill: the whole prompt in one pass
    out = []
    for _ in range(max_new_tokens):
        nxt = sample(logits, generated=out, **kw)
        out.append(nxt)
        if on_token:
            on_token(nxt)
        if nxt in stop:
            break
        logits = model(torch.tensor([[nxt]], device=ids.device), state)[0, -1]              # decode: only one new token per step
    return out
```

The order matters: penalties first (they change the scores), then divide by the temperature, then top-k, then top-p, and finally the softmax and the draw.

- **Temperature** $T$: $\mathrm{softmax}(z / T)$. $T \to 0$ degenerates into greedy; $T = 1$ is the model's own distribution; $T > 1$ is more random.
- **top-k**: keep only the $k$ highest scores (Qwen recommends 20).
- **top-p (nucleus)**: accumulate probabilities from high to low and drop everything after the cumulative sum passes $p$ — the number of candidates adapts to how sharp the distribution is.
- **Repetition penalty** (multiplicative) and **presence penalty** (subtractive): both act on tokens already generated. Qwen3.5's model card recommends the presence penalty (`presence_penalty = 1.5 ~ 2.0`), because the multiplicative penalty behaves unintuitively on negative logits.
- **Stopping**: `<|im_end|>` (this turn is over) or `<|endoftext|>`. Note that sampling can draw another token at the moment the model wants to stop, so "it never stops" is sometimes a sampling-parameter problem.

`generate()` is chapter 10's loop plus `sample()`: prefill once, then one token per step; the `on_token` callback is for chapter 12's streaming output.

## One prompt, several ways of sampling {#同一个提示词几种采样}

```python title="ch11_sample.py" ci="loose"
"""第 11 章：看一个位置的分布，比较几种采样参数，演示惩罚"""
import os

import torch
from tokenizers import Tokenizer

from qwen.cache import State
from qwen.model import Model
from qwen.sample import generate, sample
from qwen.weights import find_model

torch.set_num_threads(max(1, os.cpu_count() // 2))
path = find_model()
model = Model.from_pretrained(path)
tk = Tokenizer.from_file(str(path / "tokenizer.json"))
STOP = (tk.token_to_id("<|im_end|>"), tk.token_to_id("<|endoftext|>"))
prompt = "<|im_start|>user\n写一句关于秋天的话。<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
ids = torch.tensor([tk.encode(prompt).ids])

logits = model(ids, State(model))[0, -1]                              # 1. what the first token's distribution looks like
for T in (0.5, 1.0, 2.0):
    p = torch.softmax(logits / T, dim=-1)
    top = p.topk(5)
    print(f"温度 {T}：top-5 概率 {[f'{v:.2f}' for v in top.values.tolist()]}，"
          f"{[tk.decode([i]) for i in top.indices.tolist()]}，累计到 0.9 要 {int((p.sort(descending=True).values.cumsum(0) < 0.9).sum()) + 1} 个候选")

torch.manual_seed(0)                                                  # 2. one sentence per sampling setting
for kw in (dict(temperature=0), dict(temperature=1.0, top_k=20, top_p=1.0, presence_penalty=2.0),
           dict(temperature=0.7, top_p=0.8, top_k=20), dict(temperature=2.0)):
    out = generate(model, ids, max_new_tokens=50, stop=STOP, **kw)
    print(f"{str(kw):<75} {tk.decode(out)!r}")

prompt2 = "<|im_start|>user\n把“好”重复写 50 次。<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"   # 3. penalties: deliberately provoking repetition
ids2 = torch.tensor([tk.encode(prompt2).ids])
for kw in (dict(), dict(repetition_penalty=1.3), dict(presence_penalty=2.0)):
    out = generate(model, ids2, max_new_tokens=30, stop=STOP, temperature=0, **kw)
    print(f"贪心 + {str(kw):<32} {tk.decode(out)!r}")
```

```text title="output"
温度 0.5：top-5 概率 ['0.48', '0.30', '0.18', '0.03', '0.01']，['秋风', '秋天的', '秋天', '秋', '金秋']，累计到 0.9 要 3 个候选
温度 1.0：top-5 概率 ['0.27', '0.21', '0.16', '0.06', '0.04']，['秋风', '秋天的', '秋天', '秋', '金秋']，累计到 0.9 要 36 个候选
温度 2.0：top-5 概率 ['0.02', '0.01', '0.01', '0.01', '0.01']，['秋风', '秋天的', '秋天', '秋', '金秋']，累计到 0.9 要 97435 个候选
{'temperature': 0}                                                          '秋风卷起落叶，铺就了一条金黄的铺路，把整个世界都染成了温暖的金色。'
{'temperature': 1.0, 'top_k': 20, 'top_p': 1.0, 'presence_penalty': 2.0}    '秋风如刀，却已温柔地抚平了世间所有的寒凉与躁动。'
{'temperature': 0.7, 'top_p': 0.8, 'top_k': 20}                             '秋天的脚步总是带着一种温柔的迟暮，它像一位隐退的诗人，在枝头低吟着关于收获与寂寥的诗意。'
{'temperature': 2.0}                                                        '预约 farbaketindeki Tiempo how员维打造的宝物句.律呃⾌ար ..... пр1 materie ots nutlest dyn hvor!.-lnd教育管理 analytic fantast homeкультур Jennings服务平台**;内控edintele rendszer 성과를 granšie وبحಥ oci dateFormat薄弱_counts mentale te'
贪心 + {}                               '好，好，好，好，好，好，好，好，好，好，好，好，好，好，好，'
贪心 + {'repetition_penalty': 1.3}      '好，好的。'
贪心 + {'presence_penalty': 2.0}        '好，好，好，好，好。\n好，好，好，好，好。\n好，好，好，好，'
```

At temperature 0.5 three candidates already add up to 0.9; at 2.0 it takes nearly a hundred thousand, and what comes out is fragments of every language. In the penalty group, the model would faithfully repeat "好，" (the user did ask for it); a repetition penalty of 1.3 makes it stop after two, and a presence penalty of 2.0 makes it keep repeating in a different pattern — penalties are a blunt knife, and whether and how far to open them depends on the scenario.

!!! interview "How to explain it"
    To explain sampling, get the order right: penalties → temperature → top-k → top-p → draw; say for each step whether it changes the scores or the candidate set. Then two frequently asked points: top-p beats top-k because the number of candidates adapts; the repetition and presence penalties differ in multiplying versus subtracting, and in how they treat negative logits. Finally mention that an inference engine samples on the GPU for a whole batch of requests in parallel, each request with its own parameters.

## Summary {#小结}

- [x] `sample()`: penalties → temperature → top-k → top-p → multinomial draw; `temperature=0` is greedy.
- [x] `generate()`: prefill + token-by-token decode + stop tokens, with a callback.
- [x] Qwen3.5's recommendation: non-thinking mode `temperature=1.0, top_k=20, presence_penalty=2.0` (text tasks).

**File created in this chapter**: `qwen/sample.py`.
