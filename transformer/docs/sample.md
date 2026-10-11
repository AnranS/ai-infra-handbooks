# （十一）采样与停止

<p class="lead">到目前为止我们一直取最高分（贪心）。真正的对话模型是<strong>按分布采样</strong>的：温度决定分布多尖，top-k / top-p 砍掉长尾，重复惩罚和出现惩罚压住"车轱辘话"，遇到停止 token 就停。这一章把这些写成 <code>sample()</code> 和 <code>generate()</code>——后面所有章都用它生成。</p>

!!! note "这一章跑什么"
    `ch11_sample.py`，约 30 秒。

@@files@@

## 从分数到一个 token

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
        s = torch.where(s > 0, s / repetition_penalty, s * repetition_penalty)             # 重复惩罚（乘法）：出现过的词正分压低、负分压得更低
        logits[seen] = s - presence_penalty                                                # 出现惩罚（减法）：出现过就统一扣分，Qwen 官方推荐用这个
    if temperature <= 0:
        return int(logits.argmax())
    logits = logits / temperature                                                          # 温度越低，分布越尖
    if top_k > 0:
        kth = logits.topk(top_k).values[-1]
        logits[logits < kth] = float("-inf")                                               # 只留分数最高的 k 个
    if top_p < 1.0:
        sorted_logits, order = logits.sort(descending=True)
        probs = sorted_logits.softmax(-1)
        drop = probs.cumsum(-1) - probs > top_p                                            # 累计概率超过 p 之后的都去掉（刚好跨线的那个留下）
        sorted_logits[drop] = float("-inf")
        logits = torch.full_like(logits, float("-inf")).scatter(0, order, sorted_logits)
    return int(torch.multinomial(logits.softmax(-1), 1))


def generate(model, ids, max_new_tokens=200, stop=(), on_token=None, **kw):
    """ids: [1, T] 的提示词 -> 生成的 token 列表。kw 原样传给 sample；每生成一个 token 调一次 on_token。"""
    state = State(model)
    logits = model(ids, state)[0, -1]                                                      # prefill：整段提示词一次算完
    out = []
    for _ in range(max_new_tokens):
        nxt = sample(logits, generated=out, **kw)
        out.append(nxt)
        if on_token:
            on_token(nxt)
        if nxt in stop:
            break
        logits = model(torch.tensor([[nxt]], device=ids.device), state)[0, -1]              # decode：每次只算一个新 token
    return out
```

顺序有讲究：先惩罚（改分数），再除温度，再 top-k，再 top-p，最后 softmax 采样。

- **温度** $T$：$\mathrm{softmax}(z / T)$。$T \to 0$ 退化成贪心；$T = 1$ 是模型原本的分布；$T > 1$ 更随机。
- **top-k**：只保留分数最高的 $k$ 个（Qwen 推荐 20）。
- **top-p（nucleus）**：按概率从高到低累加，累计超过 $p$ 之后的都去掉——候选的个数随分布的尖锐程度自动变化。
- **重复惩罚**（乘法）与**出现惩罚**（减法）：都作用在已经生成过的 token 上。Qwen3.5 的模型卡推荐用出现惩罚（`presence_penalty = 1.5 ~ 2.0`），因为乘法惩罚对负分 logit 的行为不直观。
- **停止**：`<|im_end|>`（这一轮说完了）或 `<|endoftext|>`。注意采样可能在模型想停的时候抽到别的 token，所以"停不下来"有时是采样参数的问题。

`generate()` 就是第 10 章的循环加上 `sample()`：prefill 一次，之后每步一个 token；`on_token` 回调给第 12 章的流式输出用。

## 同一个提示词，几种采样

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

logits = model(ids, State(model))[0, -1]                              # 1. 第一个 token 的分布长什么样
for T in (0.5, 1.0, 2.0):
    p = torch.softmax(logits / T, dim=-1)
    top = p.topk(5)
    print(f"温度 {T}：top-5 概率 {[f'{v:.2f}' for v in top.values.tolist()]}，"
          f"{[tk.decode([i]) for i in top.indices.tolist()]}，累计到 0.9 要 {int((p.sort(descending=True).values.cumsum(0) < 0.9).sum()) + 1} 个候选")

torch.manual_seed(0)                                                  # 2. 几种采样参数各生成一句
for kw in (dict(temperature=0), dict(temperature=1.0, top_k=20, top_p=1.0, presence_penalty=2.0),
           dict(temperature=0.7, top_p=0.8, top_k=20), dict(temperature=2.0)):
    out = generate(model, ids, max_new_tokens=50, stop=STOP, **kw)
    print(f"{str(kw):<75} {tk.decode(out)!r}")

prompt2 = "<|im_start|>user\n把“好”重复写 50 次。<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"   # 3. 惩罚：故意诱导重复
ids2 = torch.tensor([tk.encode(prompt2).ids])
for kw in (dict(), dict(repetition_penalty=1.3), dict(presence_penalty=2.0)):
    out = generate(model, ids2, max_new_tokens=30, stop=STOP, temperature=0, **kw)
    print(f"贪心 + {str(kw):<32} {tk.decode(out)!r}")
```

```text title="输出"
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

温度 0.5 时 3 个候选就累计到 0.9；温度 2.0 时要近十万个，生成出来的就是各种语言的碎片。惩罚的那组：模型本来会老老实实地重复"好，"（用户确实要求了），重复惩罚 1.3 让它两次就停，出现惩罚 2.0 让它换个花样接着重复——惩罚是一把钝刀，该不该开、开多大要看场景。

!!! interview "怎么讲清楚"
    讲采样把顺序说对：惩罚 → 温度 → top-k → top-p → 采样；每一步改的是分数还是候选集。再说两个常被问到的点：top-p 比 top-k 好在候选数自适应；重复惩罚和出现惩罚的区别是乘法还是减法、以及对负分 logit 的影响。最后提一句推理引擎里采样是在 GPU 上对一批请求并行做的，每个请求参数可以不同。

## 小结

- [x] `sample()`：惩罚 → 温度 → top-k → top-p → 多项式采样；`temperature=0` 即贪心。
- [x] `generate()`：prefill + 逐 token 解码 + 停止 token，带回调。
- [x] Qwen3.5 推荐：非思考模式 `temperature=1.0, top_k=20, presence_penalty=2.0`（文本任务）。

**这一章新建的文件**：`qwen/sample.py`。
