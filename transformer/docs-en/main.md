# Step 0 · main.py: run the finish line first

<p class="lead">The model this book writes by hand is Qwen3.5-0.8B, the smallest of the newest Qwen generation (3.5 / 3.6 / 3.8 share one architecture). Before writing anything, run the finish line: keep the model itself as a black box for now (the Hugging Face implementation) and <strong>write the decode loop yourself</strong>, to see exactly what "one token after another" means. Then open the weight file and count: 488 tensors, each of which corresponds to a file some later chapter writes.</p>

!!! note "What this chapter runs"
    `main.py` (about 10 seconds: load the weights and generate 40 tokens) and `map.py` (1 second; it only reads the weight file's index).
    Download the weights first (1.7 GB): `hf download Qwen/Qwen3.5-0.8B --local-dir models/Qwen3.5-0.8B`; the environment is on the [home page](index.md#怎么跑).

## One token after another {#一个-token-接一个-token}

A language model does one thing: given all the tokens so far, it scores every token in the vocabulary (the logits); the highest score is what it thinks comes next. Generating a sentence means repeating this a few dozen times, appending the new token and computing again. The `main.py` below is the skeleton of the whole book: everything the next sixteen chapters do amounts to replacing the `model` on line 15 with our own:

```python title="main.py"
"""第 0 步：先把终点跑起来。模型本身先当黑盒（Hugging Face 的实现），解码循环自己写。"""
import os
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_num_threads(max(1, os.cpu_count() // 2))          # on CPU more threads is not better (measured in chapter 15)


def find_model(name="Qwen3.5-0.8B"):
    """从当前目录往上找 models/<name>"""
    for d in [Path.cwd(), *Path.cwd().parents]:
        if (d / "models" / name).is_dir():
            return d / "models" / name
    raise FileNotFoundError(f"先下载权重：hf download Qwen/{name} --local-dir models/{name}")


path = find_model()
tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).eval()
print(type(model).__name__, f"{sum(p.numel() for p in model.parameters()) / 1e6:.0f}M 参数")

# 1. text -> tokens: apply the chat template (chapter 12 writes it by hand), then split into token ids
messages = [{"role": "user", "content": "用一句话介绍一下 Transformer。"}]
ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False,
                                    return_tensors="pt", return_dict=True)["input_ids"]
print(f"提示词 {ids.shape[1]} 个 token：", tokenizer.decode(ids[0]).replace("\n", "\\n"))
STOP = {tokenizer.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")}   # the two "I am done" tokens

# 2. the decode loop: feed the whole sequence, take the last position's scores, append the top token, repeat
t0 = time.time()
out = ids
with torch.no_grad():
    for _ in range(40):
        logits = model(out).logits[0, -1]                    # [vocab]: scores of the next token
        nxt = logits.argmax()                                 # greedy: always the highest score
        out = torch.cat([out, nxt.view(1, 1)], dim=1)
        if nxt.item() in STOP:
            break
new = out[0, ids.shape[1]:]
print(f"生成 {len(new)} 个 token，{time.time() - t0:.1f} 秒")
print(tokenizer.decode(new, skip_special_tokens=True))

# 3. compare with HF's own generate: the same greedy search should not differ by a single token
ref = model.generate(ids, max_new_tokens=40, do_sample=False, eos_token_id=list(STOP))
print("和 model.generate 一样：", ref[0].tolist() == out[0].tolist())
```

```text title="output"
Qwen3_5ForCausalLM 752M 参数
提示词 17 个 token： <|im_start|>user\n用一句话介绍一下 Transformer。<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n
生成 38 个 token，6.3 秒
Transformer 是一种基于深度学习的神经网络架构，通过自注意力机制解决了传统卷积神经网络在处理长距离依赖和复杂语义关系时的局限性，从而实现了强大的语言理解与生成能力。
和 model.generate 一样： True
```

Three things are worth seeing clearly right now:

1. **The prompt is not the sentence you typed** but a sequence of tokens wrapped in a template: `<|im_start|>user\n…<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n`. Who is speaking, whether they are done, whether to think first — all of it is carried by these special tokens. Chapter 12 writes this template by hand.
2. **The model only ever gives "the scores of the next token"**; one forward pass computes the whole sequence and we keep only the last position. This loop recomputes the whole sequence at every step, so it gets slower the longer it generates — exactly the problem chapter 10 solves (the KV cache and the memory table).
3. It stops because the model itself emits `<|im_end|>`. Not because a length ran out, but because the model gave that token the highest score in the vocabulary.

## The map of the book: 488 tensors {#全书地图488-个张量}

The weight file is a 1.7 GB safetensors file. It carries its own index (`model.safetensors.index.json`), so the name and shape of every tensor can be read without loading anything. Grouped by name:

```python title="map.py"
"""全书地图：权重文件里有什么。只读目录和形状，不加载数据。"""
import json
import re
from collections import OrderedDict
from pathlib import Path

from safetensors import safe_open


def find_model(name="Qwen3.5-0.8B"):
    for d in [Path.cwd(), *Path.cwd().parents]:
        if (d / "models" / name).is_dir():
            return d / "models" / name
    raise FileNotFoundError(f"先下载权重：hf download Qwen/{name} --local-dir models/{name}")


path = find_model()
cfg = json.loads((path / "config.json").read_text())["text_config"]
print(f"{cfg['num_hidden_layers']} 层：{cfg['layer_types'].count('linear_attention')} 层线性注意力 + "
      f"{cfg['layer_types'].count('full_attention')} 层全注意力，排法是", " ".join("L" if t.startswith("linear") else "A" for t in cfg["layer_types"]))
print(f"hidden {cfg['hidden_size']}，前馈 {cfg['intermediate_size']}，词表 {cfg['vocab_size']}，"
      f"注意力 {cfg['num_attention_heads']} 个 query 头 / {cfg['num_key_value_heads']} 组 KV × {cfg['head_dim']} 维，"
      f"线性注意力 {cfg['linear_num_value_heads']} 个头 × {cfg['linear_value_head_dim']} 维")

groups = OrderedDict()                                          # layer numbers in the names become N; tensors of the same kind form a group
with safe_open(next(path.glob("*.safetensors")), "pt") as f:
    for name in f.keys():
        key = re.sub(r"\.\d+\.", ".N.", name)
        shape = tuple(f.get_slice(name).get_shape())
        g = groups.setdefault(key, {"shape": shape, "n": 0, "params": 0})
        g["n"] += 1
        g["params"] += __import__("math").prod(shape)

parts = {"model.language_model.": "语言模型（这本书）", "model.visual.": "视觉编码器（不碰）", "mtp.": "MTP 草稿头（第 14 章）"}
for prefix, title in parts.items():
    rows = [(k, g) for k, g in groups.items() if k.startswith(prefix)]
    total = sum(g["params"] for _, g in rows)
    print(f"\n== {title}：{sum(g['n'] for _, g in rows)} 个张量，{total / 1e6:.1f}M 参数")
    if prefix != "model.language_model.":
        continue
    for k, g in rows:
        print(f"  {k[len(prefix):]:<48} {str(g['shape']):<14} ×{g['n']:<3} {g['params'] / 1e6:7.1f}M")
```

```text title="output"
24 层：18 层线性注意力 + 6 层全注意力，排法是 L L L A L L L A L L L A L L L A L L L A L L L A
hidden 1024，前馈 3584，词表 248320，注意力 8 个 query 头 / 2 组 KV × 256 维，线性注意力 16 个头 × 128 维

== 语言模型（这本书）：320 个张量，752.4M 参数
  embed_tokens.weight                              (248320, 1024) ×1     254.3M
  layers.N.input_layernorm.weight                  (1024,)        ×24      0.0M
  layers.N.linear_attn.A_log                       (16,)          ×18      0.0M
  layers.N.linear_attn.conv1d.weight               (6144, 1, 4)   ×18      0.4M
  layers.N.linear_attn.dt_bias                     (16,)          ×18      0.0M
  layers.N.linear_attn.in_proj_a.weight            (16, 1024)     ×18      0.3M
  layers.N.linear_attn.in_proj_b.weight            (16, 1024)     ×18      0.3M
  layers.N.linear_attn.in_proj_qkv.weight          (6144, 1024)   ×18    113.2M
  layers.N.linear_attn.in_proj_z.weight            (2048, 1024)   ×18     37.7M
  layers.N.linear_attn.norm.weight                 (128,)         ×18      0.0M
  layers.N.linear_attn.out_proj.weight             (1024, 2048)   ×18     37.7M
  layers.N.mlp.down_proj.weight                    (1024, 3584)   ×24     88.1M
  layers.N.mlp.gate_proj.weight                    (3584, 1024)   ×24     88.1M
  layers.N.mlp.up_proj.weight                      (3584, 1024)   ×24     88.1M
  layers.N.post_attention_layernorm.weight         (1024,)        ×24      0.0M
  layers.N.self_attn.k_norm.weight                 (256,)         ×6       0.0M
  layers.N.self_attn.k_proj.weight                 (512, 1024)    ×6       3.1M
  layers.N.self_attn.o_proj.weight                 (1024, 2048)   ×6      12.6M
  layers.N.self_attn.q_norm.weight                 (256,)         ×6       0.0M
  layers.N.self_attn.q_proj.weight                 (4096, 1024)   ×6      25.2M
  layers.N.self_attn.v_proj.weight                 (512, 1024)    ×6       3.1M
  norm.weight                                      (1024,)        ×1       0.0M

== 视觉编码器（不碰）：153 个张量，100.6M 参数

== MTP 草稿头（第 14 章）：15 个张量，20.5M 参数
```

This table is the table of contents of the book. Keep three things in mind while reading it:

- **The vocabulary is a third of the model.** `embed_tokens` is 248320 × 1024 = 254M parameters, and the output layer shares it (`tie_word_embeddings`), so there is no `lm_head` in the file at all. Small models are all like this: the vocabulary is sized for the big ones.
- **18 linear-attention layers + 6 full-attention layers**, one full-attention layer after every three linear ones. The linear-attention group (`linear_attn.*`) has 7 tensors with unfamiliar names — `A_log`, `dt_bias`, `conv1d`, `in_proj_a/b`; they are all of chapters 7 and 8, and the biggest difference between this generation of Qwen and Llama-style models.
- **The full attention's `q_proj` is [4096, 1024]** rather than 8 × 256 = 2048 rows: the extra half is the "gate". Chapter 6.

Two parts of the weight file are not written in this book: `model.visual.*` is the vision encoder (all the small Qwen3.5 models are multimodal), and `mtp.*` is a one-layer "draft head" that chapter 14 uses for speculative decoding.

![Figure: the 24-layer pattern (one full-attention layer after every three linear ones) and the structure of one layer](assets/figures/layer-pattern.svg){.aig-svg}

## The road ahead {#后面怎么走}

| Stage | Chapters | What | How it is checked |
| --- | --- | --- | --- |
| 1 · Taking the black box apart | 1–9 | one file per chapter, replacing the parts of the HF model with our own | after every replacement, compare with the original logits (differences around 1e-5); chapter 9's whole model no longer imports transformers |
| 2 · From a model to an engine | 10–14 | the two kinds of state, sampling, the chat template, batching, speculative decoding | stateful == stateless token by token; batch == single; template == HF token by token |
| 3 · Faster and bigger | 15–16 | threads, bf16, devices; how the same architecture grows to 27B and MoE | real timings; MoE checked against transformers with random weights |

Every chapter ends by listing the files it created. The finished package looks like this:

<div class="filetree" markdown>

```text
qwen/
├── weights.py     （一）读 config.json 和 safetensors
├── harness.py     （一）替换骨架：把自己写的模块装进 HF 模型里对答案
├── embedding.py   （二）嵌入层、共用权重的输出层
├── norm.py        （三）零中心 RMSNorm、带门的 RMSNorm、L2 归一化
├── mlp.py         （四）SwiGLU
├── rope.py        （五）旋转位置编码（只转 1/4 维）
├── attention.py   （六）门控注意力：GQA + QK-Norm + 输出门      ← 第 10 章加状态和 mask
├── deltanet.py    （七）Gated DeltaNet：逐 token 递推            ← 第 10 章加状态和 mask
├── chunk.py       （八）Gated DeltaNet：分块并行
├── layer.py       （九）解码层                                   ← 第 10 章传状态
├── model.py       （九）整模型：从这里起不再 import transformers  ← 第 10 章加状态
├── cache.py       （十）KV cache 与记忆表
├── sample.py      （十一）温度 / top-k / top-p / 重复惩罚，generate()
├── chat.py        （十二）聊天模板、流式输出、思考模式
├── batch.py       （十三）左填充 + mask 的批处理
├── mtp.py         （十四）草稿头 + 投机解码
└── moe.py         （十六）混合专家层（对 MoE 版 Qwen3.5 的权重命名）
```

</div>

!!! interview "How to explain it"
    To explain "how a large model generates text", do not start from attention; start from this loop: the model is a function from a token sequence to scores over the vocabulary, and generation is "take the top score (or sample) → append → compute again". Then add two sentences: the prompt is a token sequence wrapped in a template, and stopping relies on a special token. Once the listener has this frame, every later part has a place to go.

## Summary {#小结}

- [x] Our greedy loop matches `model.generate` token for token: the model = a function that "scores the next token".
- [x] The 0.8B language model has 752M parameters in 320 tensors: a third is the vocabulary; 18 linear-attention layers + 6 full-attention layers.
- [x] Every group of tensors in the weight file corresponds to a later chapter; `model.visual.*` is left alone, `mtp.*` waits for chapter 14.
