# 第 0 步 · main.py：先把终点跑起来

<p class="lead">这本书要手写的是 Qwen3.5-0.8B——Qwen 最新一代（3.5 / 3.6 / 3.8 是同一套架构）里最小的那个。写之前先把终点跑起来：模型本身暂时当黑盒（直接用 Hugging Face 的实现），<strong>解码循环自己写</strong>，看清楚"一个 token 接一个 token"到底是怎么回事。然后把权重文件打开数一遍：488 个张量，每一个都对应后面某一章要写的文件。</p>

!!! note "这一章跑什么"
    `main.py`（约 10 秒：加载权重 + 生成 40 个 token）和 `map.py`（1 秒，只读权重文件的目录）。
    需要先下载权重（1.7 GB）：`hf download Qwen/Qwen3.5-0.8B --local-dir models/Qwen3.5-0.8B`；环境见[首页](index.md#怎么跑)。

## 一个 token 接一个 token

语言模型只做一件事：给定前面所有的 token，给词表里每个 token 打分（logits），分最高的就是它认为的下一个。生成一句话，就是把这件事重复几十次，每次把新 token 接在后面再算一遍。下面这段 `main.py` 就是整本书的骨架——后面十六章做的全部事情，只是把第 15 行的 `model` 换成自己写的：

```python title="main.py"
"""第 0 步：先把终点跑起来。模型本身先当黑盒（Hugging Face 的实现），解码循环自己写。"""
import os
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_num_threads(max(1, os.cpu_count() // 2))          # CPU 上线程不是越多越好（第 15 章量过）


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

# 1. 文字 -> token：套上聊天模板（第 12 章手写它），再切成 token id
messages = [{"role": "user", "content": "用一句话介绍一下 Transformer。"}]
ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False,
                                    return_tensors="pt", return_dict=True)["input_ids"]
print(f"提示词 {ids.shape[1]} 个 token：", tokenizer.decode(ids[0]).replace("\n", "\\n"))
STOP = {tokenizer.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")}   # 两个"说完了"的 token

# 2. 解码循环：整段序列喂进去，取最后一个位置的分数，分最高的 token 接在后面，再来一遍
t0 = time.time()
out = ids
with torch.no_grad():
    for _ in range(40):
        logits = model(out).logits[0, -1]                    # [词表]：下一个 token 的分数
        nxt = logits.argmax()                                 # 贪心：永远取最高分
        out = torch.cat([out, nxt.view(1, 1)], dim=1)
        if nxt.item() in STOP:
            break
new = out[0, ids.shape[1]:]
print(f"生成 {len(new)} 个 token，{time.time() - t0:.1f} 秒")
print(tokenizer.decode(new, skip_special_tokens=True))

# 3. 和 HF 自己的 generate 对一下：同样的贪心，应该一个 token 都不差
ref = model.generate(ids, max_new_tokens=40, do_sample=False, eos_token_id=list(STOP))
print("和 model.generate 一样：", ref[0].tolist() == out[0].tolist())
```

```text title="输出"
Qwen3_5ForCausalLM 752M 参数
提示词 17 个 token： <|im_start|>user\n用一句话介绍一下 Transformer。<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n
生成 38 个 token，6.3 秒
Transformer 是一种基于深度学习的神经网络架构，通过自注意力机制解决了传统卷积神经网络在处理长距离依赖和复杂语义关系时的局限性，从而实现了强大的语言理解与生成能力。
和 model.generate 一样： True
```

三件事值得现在就看清楚：

1. **提示词不是你打的那句话**，而是套了模板的一串 token：`<|im_start|>user\n…<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n`。谁在说、说完没有、要不要先想，全靠这些特殊 token。第 12 章会手写这个模板。
2. **模型每次只给出"下一个 token 的分数"**，一次前向算的是整段序列，我们只取最后一个位置。这个循环每一步都把整段重新算一遍，所以越生成越慢——这就是第 10 章要解决的问题（KV cache 和记忆表）。
3. 停下来靠的是模型自己吐出 `<|im_end|>`。它不是"字数到了"，而是模型在词表里给这个 token 打了最高分。

## 全书地图：488 个张量

权重文件是一个 1.7 GB 的 safetensors。它自带目录（`model.safetensors.index.json`），不用加载就能看到每个张量的名字和形状。按名字归一下类：

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

groups = OrderedDict()                                          # 名字里的层号换成 N，同类张量归成一组
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

```text title="输出"
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

这张表就是全书的目录。读的时候对着它想三件事：

- **词表占了三分之一。** `embed_tokens` 是 248320 × 1024 = 254M 参数，而输出层和它共用（`tie_word_embeddings`），所以文件里根本没有 `lm_head`。小模型都这样：词表是按大模型的规格配的。
- **18 层线性注意力 + 6 层全注意力**，每 3 层线性接 1 层全注意力。线性注意力那一组（`linear_attn.*`）有 7 个张量，名字很陌生——`A_log`、`dt_bias`、`conv1d`、`in_proj_a/b`；它们是第 7、8 章的全部内容，也是这代 Qwen 和 Llama 式模型最大的区别。
- **全注意力的 `q_proj` 是 [4096, 1024]** 而不是 8 × 256 = 2048 行：多出来的一半是"门"。第 6 章讲。

权重文件里还有两部分这本书不会手写：`model.visual.*` 是看图用的视觉编码器（Qwen3.5 的小模型全是多模态的），`mtp.*` 是一个只有一层的"草稿头"，第 14 章拿它做投机解码。

![图：24 层的排法（每 3 层线性注意力接 1 层全注意力）与一层的结构](assets/figures/layer-pattern.svg){.aig-svg}

## 后面怎么走

| 阶段 | 章 | 做什么 | 怎么验证 |
| --- | --- | --- | --- |
| 一 · 拆黑盒 | 1–9 | 一章一个文件，把 HF 模型里的零件逐个换成自己写的 | 每换一块，和原来的 logits 对一次（误差 1e-5 量级）；第 9 章整模型不再 import transformers |
| 二 · 从模型到引擎 | 10–14 | 两种状态、采样、聊天模板、批处理、投机解码 | 带状态 == 不带状态逐 token 一致；批处理 == 单条；模板和 HF 逐 token 一致 |
| 三 · 更快、更大 | 15–16 | 线程、bf16、设备；同一套架构怎么变到 27B 和 MoE | 真实计时；MoE 和 transformers 用随机权重对 |

每一章末尾都会列出那一章新建的文件。整个包最后长这样：

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

!!! interview "怎么讲清楚"
    讲"大模型是怎么生成文字的"，不要从注意力讲起，从这个循环讲起：模型是一个函数，输入一串 token，输出词表上的分数；生成就是"取最高分（或按分布采样）→ 接在后面 → 再算一遍"。然后再说两句：提示词是套了模板的 token 序列；停止靠的是一个特殊 token。听的人有了这个框架，后面的每个零件才知道往哪放。

## 小结

- [x] 自己写的贪心循环和 `model.generate` 逐 token 一致：模型 = "给下一个 token 打分"的函数。
- [x] 0.8B 的语言模型部分 752M 参数、320 个张量：词表占 1/3，18 层线性注意力 + 6 层全注意力。
- [x] 权重文件里每一组张量都对应后面的一章；`model.visual.*` 不碰，`mtp.*` 留到第 14 章。
