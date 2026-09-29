# 新模型接入与精度对齐

<p class="lead">一个新模型发布的当天，推理团队要让它在自己的引擎里跑起来，而且要跑得<b>对</b>——输出和官方参考实现一致，精度评测和模型卡上的数字吻合。这件事听上去是"照着写一遍"，实际上最花时间的是找出那些看不出来的差别：归一化多了个 1、某几层的 RoPE 用了另一个底数、滑窗只在长提示词时才生效。这一章走一遍完整的流程：找差别、照着参考实现写、逐层对齐定位 bug、用 KL 和精度评测验收，最后落到 vLLM 和 SGLang 的代码结构上。例子是把 Gemma 3（270M）接入一个"只会 Qwen 结构"的实现。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. 拿到一个新模型，你会先看哪些东西，来判断它和已支持的架构差在哪？
    2. 输出对不上时，怎样最快定位到是哪一层、哪个模块错了？
    3. 为什么测试用的提示词必须足够长、足够多样？哪些 bug 只在特定长度下才出现？
    4. FP32 下完全对齐、BF16 下有差异，差多少算正常？top-1 一致率 100% 就说明没有 bug 吗？
    5. 在 vLLM 或 SGLang 里加一个新模型，要写哪些东西？

## 流程

| 步骤 | 做什么 | 验收 |
| --- | --- | --- |
| 1. 找差别 | 对比新模型的 `config.json`、权重名和参考实现（transformers 里的 `modeling_*.py`）与最接近的已支持架构 | 列出一张"差异清单" |
| 2. 实现 | 复用已有的层，只补差异；映射权重名、合并 QKV / gate_up 投影、按张量并行切分 | 权重全部加载，没有缺失和多余 |
| 3. 逐层对齐 | FP32、同样的输入，逐层比较隐藏状态，找到第一处对不上的层，再往模块里缩小 | 每层的最大差在浮点误差以内 |
| 4. 端到端 | 下一个 token 分布的 KL 散度、top-1 一致率、贪心生成逐 token 一致；覆盖长提示词、batch 混合长短请求、prefill + decode 两条路径 | KL 在同精度噪声的量级 |
| 5. 精度评测 | 起服务跑 GSM8K、MMLU 等数据集，和模型卡、参考框架的数字比 | 差距在 1 个点左右以内 |
| 6. 性能 | CUDA Graph、合适的注意力后端和 kernel、量化 | 压测达到预期 |

## 第一步：找差别

先看 config 和权重名比"已经支持的 Qwen 结构"多了什么：

```python
import json

import torch
from safetensors import safe_open

PATH = "models/gemma-3-270m"
new = json.load(open(f"{PATH}/config.json"))
old = json.load(open("models/Qwen3-0.6B/config.json"))
print("config 里新出现的字段：", sorted(set(new) - set(old) - {"_sliding_window_pattern", "use_bidirectional_attention", "pad_token_id"}))
layer0 = lambda path: {k.split("layers.0.")[1] for k in safe_open(f"{path}/model.safetensors", "pt").keys() if "layers.0." in k}
print("第 0 层多出来的权重：", sorted(layer0(PATH) - layer0("models/Qwen3-0.6B")))
print("层类型：", "".join("S" if t == "sliding_attention" else "F" for t in new["layer_types"]), "（S = 滑窗，F = 全注意力）")
```

```text title="输出"
config 里新出现的字段： ['attn_logit_softcapping', 'final_logit_softcapping', 'hidden_activation', 'layer_types', 'query_pre_attn_scalar', 'rope_local_base_freq']
第 0 层多出来的权重： ['post_feedforward_layernorm.weight', 'pre_feedforward_layernorm.weight']
层类型： SSSSSFSSSSSFSSSSSF （S = 滑窗，F = 全注意力）
```

清单上的每一项都对应参考实现（`transformers/models/gemma3/modeling_gemma3.py`）里的一段代码，逐项读过去：

- `query_pre_attn_scalar`：注意力分数的缩放是 $1/\sqrt{256}$，而不是 $1/\sqrt{\text{head\_dim}}$（这个模型里两者恰好相等，换个尺寸就不等了）；
- `hidden_activation`：MLP 的激活是 tanh 近似的 GELU，不是 SiLU；
- `layer_types` 与 `rope_local_base_freq`：每 6 层里 5 层是滑窗注意力（窗口 512），1 层是全注意力；两种层的 RoPE 底数不同；
- 多出来的两个归一化：每层有 4 个 RMSNorm（注意力和 MLP 的前后各一个，"三明治"结构）；
- 读代码还会发现 config 里看不出来的：RMSNorm 乘的是 $(1 + w)$ 而不是 $w$（权重初始化为 0），词嵌入要乘 $\sqrt{\text{hidden\_size}}$，输出层和词嵌入共享权重。

## 第二步：照着参考实现写

按上面的清单写一个 Gemma 3，其中有两个细节第一版常常漏掉，用开关 `fixes` 控制，下一节靠逐层对齐把它们找出来：

```python
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


class GemmaRMSNorm(nn.Module):
    """Gemma 的 RMSNorm：权重初始化为 0，乘的是 (1 + weight)"""

    def __init__(self, dim, eps):
        super().__init__()
        self.eps, self.weight = eps, nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        x32 = x.float()
        return (x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps) * (1 + self.weight.float())).type_as(x)


def rope(x, theta):
    """x: [heads, T, D]；与 LLaMA 相同的"前后两半配对"写法"""
    D, T = x.shape[-1], x.shape[-2]
    inv = 1.0 / theta ** (torch.arange(0, D, 2).float() / D)
    f = torch.arange(T).float()[:, None] * inv
    cos, sin = torch.cat([f, f], -1).cos().to(x.dtype), torch.cat([f, f], -1).sin().to(x.dtype)
    x1, x2 = x.chunk(2, -1)
    return x * cos + torch.cat([-x2, x1], -1) * sin


class Gemma3(nn.Module):
    """按 config 和参考实现写的 Gemma 3 文本模型；fixes 控制两个"第一版里漏掉的细节"，演示怎么靠逐层对齐把它们找出来"""

    def __init__(self, cfg, fixes=()):
        super().__init__()
        self.cfg, self.fixes = cfg, set(fixes)
        H, hd, eps = cfg["hidden_size"], cfg["head_dim"], cfg["rms_norm_eps"]
        self.nh, self.nkv, self.hd = cfg["num_attention_heads"], cfg["num_key_value_heads"], hd
        self.embed_tokens = nn.Embedding(cfg["vocab_size"], H)
        self.layers = nn.ModuleList()
        for _ in range(cfg["num_hidden_layers"]):
            layer = nn.Module()
            layer.self_attn = nn.Module()
            for name, out in (("q_proj", self.nh * hd), ("k_proj", self.nkv * hd), ("v_proj", self.nkv * hd)):
                setattr(layer.self_attn, name, nn.Linear(H, out, bias=False))
            layer.self_attn.o_proj = nn.Linear(self.nh * hd, H, bias=False)
            layer.self_attn.q_norm, layer.self_attn.k_norm = GemmaRMSNorm(hd, eps), GemmaRMSNorm(hd, eps)
            layer.mlp = nn.Module()
            layer.mlp.gate_proj, layer.mlp.up_proj = nn.Linear(H, cfg["intermediate_size"], bias=False), nn.Linear(H, cfg["intermediate_size"], bias=False)
            layer.mlp.down_proj = nn.Linear(cfg["intermediate_size"], H, bias=False)
            for name in ("input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm", "post_feedforward_layernorm"):
                setattr(layer, name, GemmaRMSNorm(H, eps))                    # 每层 4 个归一化："三明治"结构
            self.layers.append(layer)
        self.norm = GemmaRMSNorm(H, eps)

    def attention(self, i, x):
        a, T = self.layers[i].self_attn, x.shape[0]
        sliding = self.cfg["layer_types"][i] == "sliding_attention"
        q = a.q_norm(a.q_proj(x).view(T, self.nh, self.hd).transpose(0, 1))          # QK-Norm 在 RoPE 之前
        k = a.k_norm(a.k_proj(x).view(T, self.nkv, self.hd).transpose(0, 1))
        v = a.v_proj(x).view(T, self.nkv, self.hd).transpose(0, 1)
        theta = self.cfg["rope_local_base_freq"] if sliding and "local_rope" in self.fixes else self.cfg["rope_theta"]
        q, k = rope(q, theta), rope(k, theta)
        k, v = k.repeat_interleave(self.nh // self.nkv, 0), v.repeat_interleave(self.nh // self.nkv, 0)
        scores = q @ k.transpose(-1, -2) * self.cfg["query_pre_attn_scalar"] ** -0.5
        pos = torch.arange(T)
        mask = pos[None, :] <= pos[:, None]                                          # 因果
        if sliding and "sliding_window" in self.fixes:
            mask &= pos[None, :] > pos[:, None] - self.cfg["sliding_window"]         # 只看最近 sliding_window 个 token
        out = scores.masked_fill(~mask, float("-inf")).softmax(-1, dtype=torch.float32).to(v.dtype) @ v
        return a.o_proj(out.transpose(0, 1).reshape(T, -1))

    def forward(self, ids):
        x = self.embed_tokens(ids) * torch.tensor(self.cfg["hidden_size"] ** 0.5, dtype=self.embed_tokens.weight.dtype)
        for i, l in enumerate(self.layers):
            x = x + l.post_attention_layernorm(self.attention(i, l.input_layernorm(x)))
            m = l.mlp
            h = l.pre_feedforward_layernorm(x)
            x = x + l.post_feedforward_layernorm(m.down_proj(F.gelu(m.gate_proj(h), approximate="tanh") * m.up_proj(h)))
            self.outputs.append(x)
        return self.norm(x[-32:]) @ self.embed_tokens.weight.T                       # 输出层与词嵌入共享权重；只算最后 32 个位置

    def run(self, ids):
        self.outputs = []
        with torch.no_grad():
            return self(ids), self.outputs
```

## 第三步：逐层对齐

在参考实现的每个解码层上挂 hook，和自己的实现逐层比较；第一处超过阈值的层，就是 bug 所在（或者它的上游）。再看那一层里从第几个 token 开始出错，往往就能直接猜出原因：

```python
from safetensors.torch import load_file

state = {k.removeprefix("model."): v.float() for k, v in load_file(f"{PATH}/model.safetensors").items()}
tok = AutoTokenizer.from_pretrained(PATH)
ref = AutoModelForCausalLM.from_pretrained(PATH, dtype=torch.float32, attn_implementation="eager").eval()


def reference(ids):
    outs = []
    hooks = [l.register_forward_hook(lambda m, i, o: outs.append((o[0] if isinstance(o, tuple) else o)[0])) for l in ref.model.layers]
    with torch.no_grad():
        logits = ref.lm_head(ref.model(ids[None]).last_hidden_state[0, -32:])
    for h in hooks:
        h.remove()
    return logits, outs


def check(model, ids, name):
    lo, ours = model.run(ids)
    lt, theirs = reference(ids)
    diffs = [(a - b).abs().max().item() for a, b in zip(ours, theirs)]
    bad = next((i for i, d in enumerate(diffs) if d > 1e-3), None)
    if bad is None:
        top1 = (lo.argmax(-1) == lt.argmax(-1)).float().mean().item()
        print(f"{name}：{len(ids)} 个 token，18 层全部对齐（逐层最大差都小于 1e-3），最后 32 个位置的 top-1 一致率 {top1:.0%}")
    else:
        rows = ((ours[bad] - theirs[bad]).abs().amax(-1) > 1e-3).nonzero()
        print(f"{name}：{len(ids)} 个 token，第 {bad} 层（{new['layer_types'][bad]}）开始对不上，出错的位置从第 {rows[0].item()} 个 token 开始")


short = tok("Gemma 3 uses local sliding-window attention on five of every six layers.", return_tensors="pt").input_ids[0]
long = tok(" ".join(f"Item {i}: the quick brown fox jumps over the lazy dog." for i in range(40)), return_tensors="pt").input_ids[0]
model = Gemma3(new)
model.load_state_dict(state)
check(model, short, "第一版，短提示词")
model.fixes.add("local_rope")                  # 对照参考实现：滑窗层的 RoPE 用 rope_local_base_freq（1 万），全注意力层才用 rope_theta（100 万）
check(model, short, "修正 RoPE 后，短提示词")
check(model, long, "修正 RoPE 后，长提示词")
model.fixes.add("sliding_window")              # 超过 512 个 token 才暴露：滑窗层只能看最近 512 个 token
check(model, long, "再加上滑窗，长提示词")
```

```text title="输出"
第一版，短提示词：18 个 token，第 0 层（sliding_attention）开始对不上，出错的位置从第 1 个 token 开始
修正 RoPE 后，短提示词：18 个 token，18 层全部对齐（逐层最大差都小于 1e-3），最后 32 个位置的 top-1 一致率 100%
修正 RoPE 后，长提示词：591 个 token，第 0 层（sliding_attention）开始对不上，出错的位置从第 512 个 token 开始
再加上滑窗，长提示词：591 个 token，18 层全部对齐（逐层最大差都小于 1e-3），最后 32 个位置的 top-1 一致率 100%
```

- **第一版在第 0 层就错了，而且从第 1 个 token 开始**：第 0 个位置没问题，说明和位置有关——RoPE 在位置 0 是恒等变换。第 0 层是滑窗层，回头对照参考实现，滑窗层的 RoPE 用 `rope_local_base_freq`（1 万），只有全注意力层用 `rope_theta`（100 万）；
- **修好之后短提示词完全对齐，长提示词还是错，而且恰好从第 512 个 token 开始**：512 就是滑窗的大小。第一版没有实现滑窗，在提示词短于 512 时和全注意力一模一样——这类 bug 用短提示词永远测不出来；
- 每处修正都只改一行，但如果只看最终输出，很难想到是这两行。

逐层对齐是接新模型时最有效的工具。几个经验：用 FP32 比，才能把"实现错了"和"精度不同"分开；参考实现用最朴素的 eager 注意力（`attn_implementation="eager"`），避免和它自己的优化 kernel 的差异混在一起；对不上时先看"从哪一层、从哪个位置开始"，位置本身就是线索（位置 0 正常 → RoPE；某个整数位置开始 → 窗口、块、分块 prefill 的边界）。

## 第四步：端到端指标

逐层对齐之后，还要看最终的输出分布。常用两个指标：下一个 token 分布的 **KL 散度**，和 **top-1 一致率**。拿刚才的长提示词，比较最后 64 个位置：

```python
def last_logits(m, ids, n=64):
    """只算最后 n 个位置的 logits（词表有 26 万，全算太占内存）"""
    with torch.no_grad():
        x = m.embed_tokens(ids) * torch.tensor(m.cfg["hidden_size"] ** 0.5, dtype=m.embed_tokens.weight.dtype)
        for i, l in enumerate(m.layers):
            x = x + l.post_attention_layernorm(m.attention(i, l.input_layernorm(x)))
            h = l.pre_feedforward_layernorm(x)
            x = x + l.post_feedforward_layernorm(l.mlp.down_proj(F.gelu(l.mlp.gate_proj(h), approximate="tanh") * l.mlp.up_proj(h)))
        return (m.norm(x[-n:]) @ m.embed_tokens.weight.T).float()


with torch.no_grad():
    ref_logp = ref.lm_head(ref.model(long[None]).last_hidden_state[0, -64:]).log_softmax(-1)


def agreement(m):
    """与参考实现比：下一个 token 分布的 KL 散度、top-1 一致率"""
    logp = last_logits(m, long).log_softmax(-1)
    kl = (ref_logp.exp() * (ref_logp - logp)).sum(-1).mean().item()
    return ("< 1e-6" if kl < 1e-6 else f"{kl:.1e}"), (logp.argmax(-1) == ref_logp.argmax(-1)).float().mean().item()


for name, setup in [("两处都修正", lambda: None), ("漏了滑窗", lambda: model.fixes.discard("sliding_window"))]:
    setup()
    kl, top1 = agreement(model)
    print(f"FP32、{name}：与参考实现的 KL {kl}，top-1 一致率 {top1:.0%}")
```

```text title="输出"
FP32、两处都修正：与参考实现的 KL < 1e-6，top-1 一致率 100%
FP32、漏了滑窗：与参考实现的 KL 1.8e-02，top-1 一致率 100%
```

**漏了滑窗时，top-1 一致率仍然是 100%**——被截断的注意力只看不到 512 个 token 以前的内容，在这段重复的文本上并不影响最可能的下一个 token，但分布已经变了。top-1 一致率很迟钝，KL 才灵敏。再看部署时的精度：

```python
model.fixes.add("sliding_window")
model.to(torch.bfloat16)                      # 部署时的精度：同样的实现，只是换成 BF16
kl, top1 = agreement(model)
print(f"BF16、两处都修正：与参考实现的 KL {kl}，top-1 一致率 {top1:.0%}")
```

在一台 x86 开发机上的输出（BF16 的数值和 CPU 的指令集有关，只看量级）：

```text
BF16、两处都修正：与参考实现的 KL 1.3e-05，top-1 一致率 100%
```

同一个正确的实现，换成 BF16 之后 KL 在 $10^{-5}$ 量级；漏了滑窗的 FP32 实现是 $10^{-2}$，大了三个数量级。所以验收的标准不是"KL 为 0"，而是"和同精度的正常噪声同一量级"：先测一个已知正确的模型在同样精度下的 KL 作为基线，新模型的 KL 明显高出基线就要回去逐层查。

测试输入要覆盖所有**边界**，否则像滑窗这样的 bug 会一直藏着：

- 长度：超过滑窗、分块 prefill 的块大小、KV 块（页）大小、压缩块（DeepSeek-V4 的 128）、最大上下文的提示词；
- 路径：一次 prefill 的结果和"prefill 一部分 + 逐个 decode"的结果要一致（后者走 KV Cache 的增量路径，和前者是两套代码）；
- batch：同一个请求单独跑和混在长短不一的 batch 里跑，结果应该一致（见[确定性推理](../topics/deterministic.md)）；
- 并行：张量并行切分之后（TP=2、4、8）与单卡一致，尤其是 KV 头数小于 TP 时头的复制；
- 其他：CUDA Graph 补齐的 batch、前缀缓存命中之后接着算、量化之后与 BF16 基线的 KL。

## 第五步：精度评测

对齐的是"和参考实现一样"，精度评测验证的是"和模型发布时的能力一样"——它能发现参考实现里没有的问题，比如对话模板、停止符、采样参数的默认值、分词器的特殊 token。做法是把模型用 OpenAI 兼容接口起成服务，用评测框架去调：

```bash
vllm serve google/gemma-3-270m-it --port 8000
lm_eval --model local-completions \
  --model_args model=google/gemma-3-270m-it,base_url=http://localhost:8000/v1/completions,num_concurrent=32 \
  --tasks gsm8k --num_fewshot 5
```

- 和谁比：模型卡上的数字（注意评测设置：few-shot 数、是否用对话模板、贪心还是采样），以及同一设置下参考框架（transformers 或另一个推理框架）跑出来的数字；
- 差多少算正常：生成类任务（GSM8K）受采样和截断影响，差 1 个点左右都常见；多次运行或多个任务一起看。明显偏低时，先查对话模板和停止符，再查数值；
- 量化模型：和同一模型的 BF16 基线比任务精度，再在一个校准集上比 KL 和 top-1。

## 在推理框架里落地

- **vLLM**：在 `vllm/model_executor/models/` 下加一个模型文件，用框架的并行层（`QKVParallelLinear`、`MergedColumnParallelLinear`、`RowParallelLinear`）和统一的 `Attention` 层（滑窗是每层的一个参数）；`load_weights` 里用 `stacked_params_mapping` 把 HF 的 `q_proj/k_proj/v_proj` 装进合并后的 `qkv_proj`、`gate_proj/up_proj` 装进 `gate_up_proj`；在 `registry.py` 里把 config 的 `architectures` 名字映射到这个类。测试在 `tests/models/` 下，`tests/models/utils.py` 里的 `check_logprobs_close` 就是"和 HF 比 logprobs"；
- **SGLang**：在 `srt/models/` 下加一个模型文件，文件末尾的 `EntryClass` 声明它对应的类；并行层、注意力（`RadixAttention`）和权重加载的写法与 vLLM 类似；
- 新模型带来的新结构（新的注意力、新的 MoE 路由、新的缓存类型）往往要改引擎本身，而不只是加一个模型文件——[新一代开源模型](../frontier/new-models.md)一章列出了 DeepSeek-V4 这一代需要的改动。

!!! interview "面试怎么答"
    被问到"新模型发布当天怎么让它在引擎里跑对"，按流程讲，并给出一个自己踩过的例子：先对比 config、权重名和参考实现，列差异清单（归一化的 $1+w$、嵌入缩放、每层的 RoPE 底数、滑窗、激活函数、缩放因子）；复用已有的层补差异；FP32 下逐层挂 hook 和参考实现比，从"第几层、第几个位置开始出错"推断原因（位置 0 正常说明和 RoPE 有关，从 512 开始说明是滑窗）；端到端看 KL 而不只看 top-1，以同精度的正常噪声为基线；测试覆盖长度、prefill 与 decode 两条路径、batch 组成、张量并行；最后起服务跑 GSM8K 这类评测，先排查对话模板和停止符。

## 练习

**1. 如果漏掉的是 RMSNorm 里的"1 +"，逐层对齐会看到什么？**

??? success "参考答案"
    RMSNorm 是每层的第一个操作，第 0 层就会错，而且**所有位置、包括第 0 个 token 都错**——这和 RoPE 的 bug（位置 0 正常）很好区分。更细一点，在第 0 层里 hook `input_layernorm` 的输出，会发现它和参考实现差了一个接近整体缩放的因子（Gemma 的权重是在 0 附近学出来的，$w$ 和 $1+w$ 差得很远），第一步就能定位到归一化。可以把示例里的 `(1 + self.weight.float())` 改成 `self.weight.float()` 试一试。

**2. 为什么用 FP32 做逐层对齐，而不是直接用部署时的 BF16？**

??? success "参考答案"
    BF16 只有 8 位尾数，两个都正确的实现（例如矩阵乘的累加顺序不同）逐层也会有 $10^{-2}$ 量级的差，而且越往后层越大，和"真 bug 造成的差"混在一起，阈值很难定。FP32 下两个正确的实现只差 $10^{-6}$ 左右，任何实现错误都会远远超出，逐层比较才有分辨力。BF16 的问题（溢出、某些算子必须用 FP32 累加）放到端到端阶段，用 KL 对比同精度的基线来查。

**3. 一个新模型在你的引擎里 GSM8K 比模型卡低了 8 个点，逐层对齐却完全通过，你会查什么？**

??? success "参考答案"
    逐层对齐通过说明"给定 token 序列时，模型计算是对的"，问题多半在 token 序列本身或解码过程：对话模板（系统提示、角色标记、是否加了 BOS、thinking 模式的开关）、分词器的特殊 token 和停止符（没停下来或停早了）、采样参数的默认值（`generation_config.json` 里的 temperature、top_p、重复惩罚）、最大生成长度截断了推理过程、评测的 few-shot 设置和答案抽取方式。其次才是只在服务里才走到的路径：分块 prefill、前缀缓存、CUDA Graph 补齐、投机解码。

## 小结

- [x] 接新模型的流程：找差别 → 复用已有的层补差异 → FP32 逐层对齐 → 端到端 KL 与贪心一致 → 精度评测 → 性能。
- [x] 差异清单来自 config、权重名和参考实现三处；有些差别（$1+w$ 的归一化、嵌入缩放）config 里看不出来，只有读代码才知道。
- [x] 逐层对齐从"第几层、第几个位置开始出错"推断原因；测试输入要跨过所有长度边界，覆盖 prefill 与 decode、batch 组成和张量并行。
- [x] top-1 一致率很迟钝，KL 才灵敏；验收标准是"和同精度的正常噪声同一量级"。
- [x] 精度评测发现的是模板、停止符、采样默认值这类问题；在 vLLM / SGLang 里接入要写模型文件、权重映射和注册，新结构往往还要改引擎。
