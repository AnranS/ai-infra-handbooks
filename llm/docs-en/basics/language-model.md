# Language models: from probabilities to the next word

<p class="lead">However large the model and however complex its architecture, a large language model does exactly one thing: given the preceding text, compute the probability distribution of the next token. Training makes that distribution more and more accurate; inference draws words from it again and again. Once you see this, everything that follows (including why inference is hard to optimize) has something to stand on.</p>

!!! question "Self-test: if you can answer these, skip this chapter"
    1. What does "autoregressive" mean? How is the probability of a sentence factored?
    2. What does one forward pass of the model output? What is its shape?
    3. How are the cross-entropy loss and perplexity related?
    4. Why can training compute the loss at every position of a whole sentence in one forward pass?
    5. Why is "autoregressive" the root cause of slow inference?

??? success "Answers (try first, then expand to compare)"
    1. Each token's probability depends only on the tokens before it: $P(x_1 \dots x_T) = \prod_t P(x_t \mid x_{<t})$, and generation produces them one after another.
    2. One vocabulary-sized logits vector per position, of shape `[B, T, V]`; after softmax it is the probability distribution of the "next token".
    3. Perplexity = exp(mean cross-entropy), and the cross-entropy is the mean negative log-likelihood per token; perplexity can be read as how many "equally likely" options the model hesitates between at each step.
    4. In training the whole sentence is known, and the causal mask ensures each position only sees the tokens before it, so one forward pass computes the predictions and losses of all positions at once (teacher forcing).
    5. Each new token depends on all previous tokens, so tokens can only be generated one step at a time; every step reads all the weights but computes just one token, leaving most of the GPU's compute idle. That makes decode serial and memory bound.

## A language model is a conditional probability function {#语言模型就是一个条件概率函数}

Split a piece of text into a sequence of tokens $x_1, x_2, \ldots, x_T$ (how to split it is covered in [tokenization](tokenization.md)). By the chain rule of probability, the probability of the whole text factors as:

$$
P(x_1, \ldots, x_T) = \prod_{t=1}^{T} P(x_t \mid x_1, \ldots, x_{t-1})
$$

A **language model** is a function that can compute each factor on the right: it takes the preceding tokens and outputs the probability distribution of the next token over the whole vocabulary. Predicting one token at a time, conditioned on everything before it, is called **autoregressive**. GPT, LLaMA, Qwen and DeepSeek are all models of this kind.

Generating text means doing the same thing over and over:

1. Feed the existing tokens into the model and get the distribution of the next token;
2. Pick a token from the distribution with some strategy (the most likely one, or a random sample; see [decoding and sampling](../inference/decoding.md));
3. Append it to the input and go back to step 1, until an end-of-sequence token is generated or the length limit is reached.

## A look at a real model's output {#看一眼真实模型的输出}

Let's use Qwen3-0.6B to see what "the distribution of the next token" looks like (how to download the model is on the [home page](../index.md#准备环境)):

```pycon
>>> import torch
>>> from transformers import AutoModelForCausalLM, AutoTokenizer
>>> path = "models/Qwen3-0.6B"
>>> tok = AutoTokenizer.from_pretrained(path)
>>> model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).eval()
>>> ids = tok("中国的首都是", return_tensors="pt").input_ids
>>> ids
tensor([[105538,  59975, 100132]])
>>> with torch.no_grad():
...     logits = model(ids).logits
...
>>> logits.shape
torch.Size([1, 3, 151936])
```

The input is 3 tokens ("中国" China, "的" 's, "首都是" capital is), and the output has shape `[batch, sequence length, vocabulary size]` = `[1, 3, 151936]`. Note that 3 is the sequence length (three input positions), not the length of the logits: **every position** outputs a vector of length 151936, called the **logits** (unnormalized scores). Every word in the vocabulary gets a score, rating "the next token after this position is this word". Softmax turns the 151936 scores of the last position (after "首都是") into probabilities, and `topk(5)` just picks the 5 most likely of those 151936 candidates to print:

```pycon
>>> probs = logits[0, -1].softmax(dim=-1)
>>> top = probs.topk(5)
>>> for p, i in zip(top.values, top.indices):
...     print(repr(tok.decode(i)), round(p.item(), 3))
'____' 0.122
'北京' 0.094
'城市' 0.048
'位于' 0.043
'建' 0.037
```

Interestingly, the most likely token is not "北京" (Beijing) but the underscores "____". Because we did not use a chat template, the model treats "中国的首都是" ("the capital of China is") as the stem of a fill-in-the-blank question, which is common in the training text it has seen. The model is only imitating the statistics of its training data, and that is worth remembering. How it behaves with a chat template is in the [tokenization](tokenization.md#特殊-token-与对话模板) chapter.

!!! inference "Inference view"
    At every step the model scores the **whole vocabulary**, so the last layer (the LM head) is a `hidden_size × vocab_size` matrix multiplication. For Qwen3-0.6B this matrix has 1024 × 151936 ≈ 156 million parameters (shared with the word embedding), more than a quarter of the whole model. The bigger the vocabulary, the more this layer costs, which is why inference engines often optimize the logits computation and sampling separately.

## The training objective: cross-entropy {#训练目标交叉熵}

How does the model learn to give good distributions? On a huge amount of text, make it **assign as high a probability as possible to the token that actually comes next at every position**. Formally, minimize the negative log-likelihood, that is, the **cross-entropy loss**:

$$
\mathcal{L} = -\frac{1}{T-1} \sum_{t=1}^{T-1} \log P_\theta(x_{t+1} \mid x_1, \ldots, x_t)
$$

Its exponential is the **perplexity**: $\text{PPL} = e^{\mathcal{L}}$. Intuitively, a perplexity of 10 means that at each position the model "hesitates" between 10 candidates on average.

Compute it on a real model. A transformers model computes the loss automatically when given `labels` (it shifts the labels by one position internally):

```pycon
>>> text = "北京是中国的首都，也是全国的政治和文化中心。"
>>> ids = tok(text, return_tensors="pt").input_ids
>>> with torch.no_grad():
...     out = model(ids, labels=ids)
>>> ids.shape, round(out.loss.item(), 3), round(torch.exp(out.loss).item(), 2)
(torch.Size([1, 12]), 3.703, 40.58)
```

Compute it by hand to see the loss of each token:

```pycon
>>> with torch.no_grad():
...     logp = model(ids).logits[0, :-1].log_softmax(dim=-1)   # the output at position t predicts token t+1
>>> nll = -logp.gather(1, ids[0, 1:, None]).squeeze(1)          # pick out the negative log-probability of the true token
>>> round(nll.mean().item(), 3)
3.703
>>> for t, v in zip(ids[0, 1:], nll):
...     print(tok.decode(t), round(v.item(), 2))
是中国 17.86
的 1.64
首都 1.93
， 0.29
也是 2.35
全国 4.6
的政治 3.22
和 5.15
文化 3.16
中心 0.0
。 0.53
```

You can see what the model "knows": after "政治和文化" ("political and cultural"), "中心" ("center") is a foregone conclusion (loss 0.0); after seeing "北京是中国的" ("Beijing is China's"), it is also confident about "首都" ("capital", 1.93); while what follows "和" ("and") is genuinely uncertain (5.15). The first position has a huge loss (17.86): the model has only seen "北京" and does not expect the next token to be "是中国" at all, and this single term pulls the mean loss from about 2.3 up to 3.7. **Perplexity is sensitive to a few surprising tokens**, so compare models on text that is long and varied enough.

Lay out the distribution at each position and "conditional probability function" and "cross-entropy" stop being abstract: drag the position to see which candidates the model considers most likely at each step and where the true next token ranks (the data is the result of the computation above):

<div class="aig-widget" data-widget="nextword"></div>

## Training can be parallel, inference must be serial {#训练可以并行推理只能串行}

Notice the computation above: **one forward pass** gives the predictions of all 11 positions. That is because in training the whole sentence is known, and the model uses a **causal mask** (each position can only see the tokens before it; see [attention](../transformer/attention.md#因果掩码)) to ensure the output at position t depends only on the first t tokens. So all positions can be computed at once, which is called **teacher forcing**.

Inference is different: token t+1 is unknown until token t has been generated, so **tokens can only be generated one by one**. Generating 500 tokens takes 500 forward passes in sequence.

![Figure: training computes the predictions of all positions in one forward pass; inference can only generate one token at a time](../assets/figures/train-vs-infer.svg){.aig-svg}

!!! inference "Inference view"
    This asymmetry, "parallel in training, serial in inference", is the root of nearly every problem in inference optimization:

    - Generating each token reads **all the weights once** but computes only one token, so the computation is tiny and the GPU spends most of its time waiting on memory: this is the **memory bottleneck** of the decode phase;
    - Hence the **KV cache** (do not recompute past tokens; see [KV cache](../inference/kv-cache.md)), **batching** (many requests share one read of the weights), **speculative decoding** (verify several tokens at once; see [serving](../inference/serving.md#投机解码)) and **quantization** (read fewer bytes).

## A tiny language model {#一个迷你语言模型}

To walk through "model → logits → loss → training → sampling" end to end, the code below trains the simplest possible language model in PyTorch: a **bigram model**, where the next character depends only on the current one. Its "parameters" are just a `vocab × vocab` table of logits:

```python
import torch
import torch.nn.functional as F

torch.manual_seed(0)
corpus = "春眠不觉晓处处闻啼鸟夜来风雨声花落知多少床前明月光疑是地上霜举头望明月低头思故乡"
chars = sorted(set(corpus))
stoi = {c: i for i, c in enumerate(chars)}
data = torch.tensor([stoi[c] for c in corpus])
V = len(chars)

logits_table = torch.zeros(V, V, requires_grad=True)     # row i: logits of the next character when the current one is i
opt = torch.optim.Adam([logits_table], lr=0.1)
for step in range(300):
    logits = logits_table[data[:-1]]                     # [T-1, V]: each position's scores for the next character
    loss = F.cross_entropy(logits, data[1:])             # compare with the true next character
    opt.zero_grad()
    loss.backward()
    opt.step()

print(f"vocab={V}, final loss={loss.item():.3f}, perplexity={loss.exp().item():.2f}")
assert loss.item() < 0.5

# generate: start from "床" and sample each next character from the distribution
idx = stoi["床"]
out = ["床"]
for _ in range(9):
    probs = logits_table[idx].softmax(dim=-1)
    idx = torch.multinomial(probs, 1).item()
    out.append(chars[idx])
print("".join(out))
```

This model is extremely limited: it only looks at the previous character, and since "明" can be followed by "月" or by other characters, it cannot tell contexts apart. **What a Transformer does, in essence, is extend "look only at the previous character" to "look at all the previous characters, and learn which ones to focus on"**. The architecture gets more and more complex, but the main thread never changes: tokens in, logits of the next token out, training with cross-entropy, and generation by sampling from the distribution.

!!! interview "In an interview"
    Asked "how does a large model generate text", make it clear in three sentences: the model outputs vocabulary-sized logits at every position, and after softmax they are the conditional probability of the next token; in training the whole sentence is known, and the causal mask lets one forward pass compute the cross-entropy at every position (perplexity is its exponential); in inference tokens can only be generated one by one, and every step reads all the weights. Then land the point: "parallel training, serial inference" makes decode slow and memory bound, and the KV cache, batching and speculative decoding all start from there.

## Exercises {#练习}

**1. Probabilities and loss.** If the model gives the correct next token probabilities 0.5, 0.25 and 0.125, what are the mean cross-entropy loss and the perplexity over these three positions?

??? success "Answer"
    Loss = −(ln 0.5 + ln 0.25 + ln 0.125) / 3 = (0.693 + 1.386 + 2.079) / 3 ≈ 1.386, that is, ln 4. Perplexity = e^1.386 = 4. Intuitively, the three positions are like guessing right among 2, 4 and 8 candidates respectively, and the geometric mean is exactly 4.

    ```python
    import math
    probs = [0.5, 0.25, 0.125]
    loss = -sum(math.log(p) for p in probs) / len(probs)
    assert abs(loss - math.log(4)) < 1e-12 and abs(math.exp(loss) - 4) < 1e-9
    ```

**2. Understanding the shapes.** A model has vocabulary size V and takes token ids of shape `[B, T]`. What is the shape of the output logits? To compute the training loss, which values do you take from it?

??? success "Answer"
    The logits have shape `[B, T, V]`. The logits at position t predict token t+1, so take the logits of the first T−1 positions (`[B, T-1, V]`) and compute the cross-entropy against tokens 2 through T (`[B, T-1]`). Inference only cares about the logits of the last position (`[B, V]`), and the logits of earlier positions are wasted, so inference engines usually compute the LM head for the last position only during prefill.

**3. Food for thought.** Why do we say a large model "only imitates the statistics of its training data"? Does that contradict the reasoning ability it shows?

??? success "Approach"
    The training objective really is just predicting the next token. But to do that well on a huge and varied body of text, the model has to form internal representations of grammar, facts and logical structure: to accurately predict the next step of a math problem, for example, the most effective way is to "learn" to solve that kind of problem. So "predict the next token" is an extremely general learning signal. Keep its limits in mind too: the model reproduces patterns common in its training data (like the fill-in-the-blank format above), and it will give plausible-looking wrong answers when it has nothing to go on. Post-training (SFT, RLHF) steers the pretrained model's behavior toward what we want; see [post-training](../training/post-training.md).

## Summary {#小结}

- [x] A language model computes $P(x_t \mid x_{<t})$; one forward pass outputs a vocabulary-sized logits vector at every position.
- [x] The training objective is cross-entropy (negative log-likelihood), and perplexity is its exponential.
- [x] In training the whole sentence is known, and the causal mask lets all positions be computed in parallel; in inference tokens can only be generated one by one.
- [x] The autoregressive nature of inference makes decode serial and memory bound, which is where optimizations such as the KV cache, batching and speculative decoding start.
