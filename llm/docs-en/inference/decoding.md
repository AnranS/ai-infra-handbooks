# Decoding and sampling

<p class="lead">At every step the model outputs a probability distribution, and how the next token is chosen from it is the decoding strategy. Greedy, temperature, top-k, top-p, min-p, repetition penalty: these parameters directly decide the quality and diversity of the output, and they are the parameters users tune most in inference service APIs. This chapter implements them from scratch and checks each one against the transformers implementation.</p>

!!! question "Self-test: if you can answer these, skip this chapter"
    1. What does temperature do to the distribution? What is it equivalent to as the temperature approaches 0?
    2. What is the difference between top-k and top-p? Which tokens exactly does top-p = 0.9 keep?
    3. How does min-p work? What makes it better than top-p?
    4. What is the difference between repetition penalty and presence penalty?
    5. Why can the same model with the same parameters produce different output in different frameworks?

??? success "Answers (try first, then expand to compare)"
    1. The logits are divided by the temperature: below 1 the distribution gets sharper, above 1 flatter; as the temperature approaches 0, the probability concentrates on the largest token, which is equivalent to greedy (argmax).
    2. top-k keeps only the k most likely tokens; top-p adds up probabilities from largest to smallest and keeps the smallest set whose cumulative probability just reaches p. top-p = 0.9 means "the few tokens that just add up to at least 90%", and the number changes with the distribution.
    3. It keeps only tokens with probability at least $\text{min\_p} \times p_{max}$: the threshold adapts to the largest probability, cutting more when the model is confident and keeping more when it is not, which is steadier than top-p when sampling at high temperature.
    4. Repetition penalty scales a token's logit multiplicatively depending on whether it has appeared (positive values divided by the penalty, negative ones multiplied by it); presence penalty subtracts a fixed value from tokens that have appeared, regardless of how many times (it is frequency penalty that is proportional to the count).
    5. Different default sampling parameters (`generation_config.json`), different implementations and ordering of penalties and truncation, different tokenization or chat templates, and floating-point differences from batching and kernels all make outputs differ.

<!-- comic ../assets/comics/decoding.webp is in Chinese; put it back once the English version exists -->

## From logits to the next token {#从-logits-到下一个-token}

At every step the model outputs logits for the last position (as long as the vocabulary). A decoding strategy does two things:

1. **Transform the logits**: scale (temperature), truncate (top-k, top-p, min-p, setting unwanted tokens to −∞), penalize (repetition penalty);
2. **Choose**: take the maximum (greedy), or sample randomly by the post-softmax probabilities.

| Strategy | How | Effect |
| --- | --- | --- |
| Greedy | take the most likely token | deterministic and stable, but prone to repetition and lacking diversity |
| Temperature T | divide the logits by T | T < 1 makes the distribution sharper (more conservative), T > 1 flatter (more diverse); T → 0 is equivalent to greedy |
| top-k | keep only the k most likely tokens | cuts off the long tail of unlikely tokens |
| top-p (nucleus sampling) | add up probabilities from high to low and keep the smallest set whose cumulative probability just reaches p | the number of candidates adapts to the distribution: few when the model is sure, many when it is not |
| min-p | keep only tokens with probability at least "maximum probability × min_p" | a relative threshold, more robust than top-p at high temperature |
| Repetition penalty | for tokens already seen: positive logits divided by the penalty, negative ones multiplied by it | suppresses parroting (the transformers definition) |
| Presence/frequency penalty | subtract a fixed value / "count × coefficient" from the logits of tokens already seen | the OpenAI API definition, also supported by vLLM and other engines |

See what temperature, top-k and top-p each do to the same distribution by trying them (the same tool as in the [probability and sampling](math://probability/) chapter):

<div class="aig-widget" data-widget="softmax"></div>

## Implementation {#实现}

```python title="sampling.py"
"""sampling.py —— 常用采样策略的实现。logits: [B, V]。"""

import torch


def apply_repetition_penalty(logits, prev_ids, penalty):
    """transformers 的定义：出现过的 token，正 logit 除以 penalty，负 logit 乘以 penalty。"""
    if penalty == 1.0:
        return logits
    score = logits.gather(1, prev_ids)
    score = torch.where(score > 0, score / penalty, score * penalty)
    return logits.scatter(1, prev_ids, score)


def top_k_filter(logits, k):
    if k <= 0 or k >= logits.shape[-1]:
        return logits
    kth = logits.topk(k, dim=-1).values[..., -1:]            # the k-th largest value
    return logits.masked_fill(logits < kth, float("-inf"))


def top_p_filter(logits, p):
    if p >= 1.0:
        return logits
    sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
    probs = sorted_logits.softmax(dim=-1)
    cum_before = probs.cumsum(dim=-1) - probs                 # cumulative probability of the tokens ranked before it
    remove_sorted = cum_before >= p                           # the tokens before it already reach p, so it is not needed
    remove = remove_sorted.scatter(1, sorted_idx, remove_sorted)
    return logits.masked_fill(remove, float("-inf"))


def min_p_filter(logits, min_p):
    if min_p <= 0.0:
        return logits
    probs = logits.softmax(dim=-1)
    threshold = min_p * probs.max(dim=-1, keepdim=True).values
    return logits.masked_fill(probs < threshold, float("-inf"))


def sample_next(logits, prev_ids=None, temperature=1.0, top_k=0, top_p=1.0, min_p=0.0,
                repetition_penalty=1.0, generator=None):
    """按 transformers 的顺序：重复惩罚 -> 温度 -> top-k -> top-p -> min-p -> 采样。"""
    if prev_ids is not None:
        logits = apply_repetition_penalty(logits, prev_ids, repetition_penalty)
    if temperature == 0.0:
        return logits.argmax(dim=-1)                          # temperature 0 means greedy
    logits = logits / temperature
    logits = min_p_filter(top_p_filter(top_k_filter(logits, top_k), top_p), min_p)
    return torch.multinomial(logits.softmax(dim=-1), 1, generator=generator).squeeze(-1)
```

Check each against the transformers logits processors:

```python
import torch
from transformers.generation.logits_process import (MinPLogitsWarper, RepetitionPenaltyLogitsProcessor,
                                                    TopKLogitsWarper, TopPLogitsWarper)
from sampling import apply_repetition_penalty, min_p_filter, top_k_filter, top_p_filter

torch.manual_seed(0)
logits = torch.randn(4, 1000) * 3
prev = torch.randint(0, 1000, (4, 30))

def same(a, b):
    return torch.equal(a.isinf(), b.isinf()) and torch.allclose(a[~a.isinf()], b[~b.isinf()])

for k in (1, 10, 50):
    assert same(top_k_filter(logits, k), TopKLogitsWarper(k)(prev, logits))
for p in (0.1, 0.5, 0.9, 0.99):
    assert same(top_p_filter(logits, p), TopPLogitsWarper(p)(prev, logits))
for m in (0.05, 0.1, 0.5):
    assert same(min_p_filter(logits, m), MinPLogitsWarper(m)(prev, logits))
for r in (1.1, 1.5):
    assert same(apply_repetition_penalty(logits, prev, r), RepetitionPenaltyLogitsProcessor(r)(prev, logits))
print("top-k / top-p / min-p / 重复惩罚 与 transformers 一致")
```

## What they do {#看看它们的效果}

Take a concrete distribution and see how many candidates each truncation keeps:

```pycon
>>> import torch
>>> from sampling import min_p_filter, top_k_filter, top_p_filter
>>> logits = torch.tensor([[3.0, 2.5, 2.0, 1.0, 0.5, 0.0, -1.0, -2.0]])
>>> [round(p, 3) for p in logits.softmax(-1)[0].tolist()]
[0.441, 0.268, 0.162, 0.06, 0.036, 0.022, 0.008, 0.003]
>>> for name, out in [("top_k=3", top_k_filter(logits, 3)), ("top_p=0.8", top_p_filter(logits, 0.8)),
...                   ("min_p=0.1", min_p_filter(logits, 0.1)), ("temp=0.5 后 top_p=0.8", top_p_filter(logits / 0.5, 0.8))]:
...     print(name, int((~out.isinf()).sum()))
...
top_k=3 3
top_p=0.8 3
min_p=0.1 4
temp=0.5 后 top_p=0.8 2
```

- top-p = 0.8: 0.441 + 0.268 = 0.709 is not enough; adding the third, 0.162, reaches 0.871, so 3 are kept;
- min-p = 0.1: the threshold is 0.441 × 0.1 ≈ 0.044, and the 4 tokens at or above it are kept;
- Lowering the temperature to 0.5 before top-p: the distribution gets sharper and the first two already exceed 0.8, so only 2 are kept. **The order of temperature and truncation affects the result**; transformers and vLLM both apply temperature first, then truncation.

Generate with different parameters on a real model (with a fixed random seed, so the results are reproducible):

```python
from transformers import AutoTokenizer
from mini_llm import KVCache, Transformer
from sampling import sample_next

path = "models/Qwen3-0.6B"
tok = AutoTokenizer.from_pretrained(path)
model = Transformer.from_pretrained(path)
msgs = [{"role": "user", "content": "写一句关于秋天的诗。"}]
prompt_ids = tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False), return_tensors="pt").input_ids

@torch.no_grad()
def run(max_new_tokens=40, seed=0, **params):
    g = torch.Generator().manual_seed(seed)
    cache = KVCache(model.cfg.num_hidden_layers)
    logits = model(prompt_ids, cache)
    ids = prompt_ids
    for _ in range(max_new_tokens):
        nxt = sample_next(logits[:, -1], prev_ids=ids, generator=g, **params)
        if nxt.item() == tok.eos_token_id:
            break
        ids = torch.cat([ids, nxt[:, None]], dim=1)
        logits = model(nxt[:, None], cache)
    return tok.decode(ids[0, prompt_ids.shape[1]:])

settings = {
    "贪心": dict(temperature=0.0),
    "T=0.7, top_p=0.8": dict(temperature=0.7, top_p=0.8),
    "T=1.5, 不截断": dict(temperature=1.5),
    "T=1.5, min_p=0.1": dict(temperature=1.5, min_p=0.1),
}
for name, params in settings.items():
    print(f"[{name}] {run(**params)}")
```

Running it in this handbook's environment gives:

```text
[贪心] 秋风起，枫叶红，山色染金黄，岁月静好。
[T=0.7, top_p=0.8] 秋风送爽，枫叶染红了大地。
[T=1.5, 不截断] 枫叶染晚秋愈旺品尝；啦啦黑夜นิยม
[T=1.5, min_p=0.1] 枫叶染金秋，寒蝉鸣不息。
```

(The prompt asks for a line of poetry about autumn. Greedy gives "Autumn wind rises, maple leaves turn red, the hills are dyed gold, the years are quiet and good"; T=0.7 with top_p=0.8 gives "The autumn wind brings coolness, maple leaves redden the land"; T=1.5 with min_p=0.1 gives "Maple leaves dye the golden autumn, the cold cicadas sing without end".)

At high temperature without truncation, many irrelevant tokens in the long tail get a chance to be picked, and once a wrong one is chosen, what follows drifts further and further (here the very first line contains words that make no sense and Thai characters); the same high temperature with min-p truncation still gives fluent output, with more variety than at low temperature. That is what truncation strategies are for.

## A model's own default parameters {#模型自带的默认参数}

Many models give recommended sampling parameters in `generation_config.json`:

```pycon
>>> import json
>>> json.load(open("models/Qwen3-0.6B/generation_config.json"))  # doctest: +NORMALIZE_WHITESPACE
{'bos_token_id': 151643, 'do_sample': True, 'eos_token_id': [151645, 151643], 'pad_token_id': 151643, 'temperature': 0.6, 'top_k': 20, 'top_p': 0.95, 'transformers_version': '4.51.0'}
```

transformers' `generate` uses these values by default; whether engines such as vLLM and SGLang read them depends on the version and launch flags. **When comparing two frameworks' outputs or running evaluations, first make sure the sampling parameters really match** (in the [previous chapter](../transformer/build-llm.md#验证二加载真实的-qwen3-06b), unless sampling was explicitly turned off, the default parameters changed the result of "greedy" generation; Qwen2.5's config also has a default repetition penalty of 1.1, which applies even under greedy decoding). Note also that there are two `eos_token_id`s: generation should stop at either one.

## Other common decoding controls {#其他常见的解码控制}

- **Stop conditions**: an end token, a specified stop string, or the maximum length. Stop strings must be matched on the detokenized text, including when they span several tokens;
- **Beam search**: keep several of the highest-scoring candidate sequences at once. Common in machine translation, rarely used in chat (it tends to produce bland, repetitive text, and costs as many times more as there are beams);
- **Logit bias**: add or subtract a value directly to certain tokens' logits, for example to forbid a word;
- **Structured output (constrained decoding)**: require the output to follow a JSON Schema or a grammar. At each step, compute from the current state "which tokens are legal" and set the logits of illegal ones to −∞. Libraries such as xgrammar, Outlines and llguidance compute this mask efficiently, and vLLM and SGLang integrate them.

!!! inference "Inference view"
    - **Sampling must run efficiently on the GPU too**: the vocabulary has well over a hundred thousand entries, top-p needs sorting, and every request in a batch may have different parameters. Libraries such as FlashInfer implement top-k/top-p with rejection-sampling-based methods to avoid a full sort;
    - **Penalties need state**: repetition/frequency penalties need to know which tokens each request has generated and how many times each appeared, so inference engines keep these counts per request;
    - **Reproducibility**: a fixed seed only guarantees the same "random numbers", but GPU results can differ slightly depending on the batch composition (different batch sizes make kernels choose different reduction orders), which makes sampling diverge. Strict reproducibility requires batch-invariant kernels, at some cost in performance;
    - **The cost of constrained decoding**: the legal-token mask is computed on the CPU and must overlap with the GPU's forward pass, or it slows decode down.

!!! interview "In an interview"
    The standard answer to sampling-parameter questions: decoding = transform the logits (temperature, repetition penalty, top-k / top-p / min-p truncation) + choose (greedy or sampling), and the order affects the result; temperature approaching 0 is equivalent to greedy; top-k keeps a fixed number, top-p adapts by cumulative probability, and min-p uses a threshold relative to the largest probability, which is steadier at high temperature. Then the engineering: the same model gives different outputs in different frameworks, commonly because the defaults in `generation_config.json` are not aligned, penalties are implemented differently, or batching introduces floating-point differences; inference engines must sample in batches on the GPU with each request's own parameters.

## Exercises {#练习}

**1. top-p by hand.** With probabilities [0.5, 0.2, 0.15, 0.1, 0.05], which tokens does top-p = 0.75 keep? And top-p = 0.5?

??? success "Answer"
    top-p = 0.75: 0.5 is not enough, adding 0.2 gives 0.7, still not enough, adding 0.15 gives 0.85, which reaches it: the first 3 are kept. top-p = 0.5: the first one alone reaches exactly 0.5, so only 1 is kept (the cumulative probability before the second token is already 0.5, so it is not needed).

    ```python
    probs = torch.tensor([[0.5, 0.2, 0.15, 0.1, 0.05]])
    assert int((~top_p_filter(probs.log(), 0.75).isinf()).sum()) == 3
    assert int((~top_p_filter(probs.log(), 0.5).isinf()).sum()) == 1
    ```

**2. Implement presence and frequency penalties.** Follow the OpenAI definition: `logits[t] -= presence_penalty * (count[t] > 0) + frequency_penalty * count[t]`, where count is the number of times token t has appeared in the generated part.

??? success "Answer"
    ```python
    def apply_presence_frequency(logits, generated_ids, presence=0.0, frequency=0.0):
        counts = torch.zeros_like(logits)
        counts.scatter_add_(1, generated_ids, torch.ones_like(generated_ids, dtype=logits.dtype))
        return logits - presence * (counts > 0).to(logits.dtype) - frequency * counts

    lg = torch.zeros(1, 5)
    out = apply_presence_frequency(lg, torch.tensor([[1, 1, 3]]), presence=0.5, frequency=0.2)
    assert torch.allclose(out, torch.tensor([[0.0, -0.9, 0.0, -0.7, 0.0]]))
    ```

    Note the difference from transformers' repetition penalty: here it is subtraction, and usually only the **generated part** is counted, not the prompt. Frameworks do not define these parameters exactly the same way, so check carefully when migrating.

## Summary {#小结}

- [x] Decoding = transform the logits (temperature, truncation, penalties) + choose (greedy or sampling).
- [x] top-k keeps a fixed number, top-p adapts by cumulative probability, min-p uses a relative threshold; temperature comes first, then truncation.
- [x] A model's `generation_config.json` carries default sampling parameters, which must be aligned when comparing frameworks.
- [x] Stop conditions, structured output and logit bias are also decoding controls.
- [x] Inference engines must efficiently apply each request's own sampling parameters on the GPU and keep the state that penalties need.

For the related math (how temperature changes entropy, why the sampling algorithms are correct, Monte Carlo error), see [probability and sampling](math://probability/) and [information theory](math://information-theory/).
