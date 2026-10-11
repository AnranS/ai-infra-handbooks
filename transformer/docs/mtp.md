# （十四）MTP：用自带的草稿头做投机解码

<p class="lead">权重文件里的 <code>mtp.*</code> 是一个只有一层的"草稿头"（20M 参数）：拿主模型最后一层的输出加上下一个 token 的嵌入，猜<strong>再下一个</strong> token。训练时它是辅助任务（Multi-Token Prediction），推理时拿来做投机解码：草稿头先猜一个，主模型一次验证两个 token，猜中就白赚一个。这一章把它写出来，量接受率和加速比，顺便碰一个混合架构特有的坑：递推状态不能像 KV cache 那样截断回退。</p>

!!! note "这一章跑什么"
    `ch14_mtp.py`，约 40 秒。

## 草稿头

```python title="qwen/mtp.py"
"""多 token 预测（MTP）：权重里自带一个只有一层的"草稿头"，拿主模型最后一层的输出加上下一个 token 的嵌入，猜再下一个 token。

推理时用它做投机解码：草稿头先猜一个，主模型一次验证两个 token；猜中就白赚一个 token，
猜错就退回到验证前的状态，只认主模型自己算出的那一个。
"""
import torch
import torch.nn as nn

from qwen.cache import AttentionState, State
from qwen.layer import DecoderLayer
from qwen.norm import RMSNorm
from qwen.rope import rope_tables
from qwen.weights import find_model, load_weights


class MTP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        H = cfg.hidden_size
        self.cfg = cfg
        self.fc = nn.Linear(2 * H, H, bias=False)                             # 把"主模型的输出"和"下一个 token 的嵌入"拼起来压回 H 维
        self.pre_fc_norm_embedding = RMSNorm(H, cfg.rms_norm_eps)
        self.pre_fc_norm_hidden = RMSNorm(H, cfg.rms_norm_eps)
        self.layers = nn.ModuleList([DecoderLayer(cfg, "full_attention")])     # 只有一层，是全注意力
        self.norm = RMSNorm(H, cfg.rms_norm_eps)

    @classmethod
    def from_pretrained(cls, cfg, path=None, dtype=torch.float32):
        mtp = cls(cfg)
        mtp.load_state_dict(load_weights(path or find_model(), part="mtp.", dtype=dtype))
        return mtp.to(dtype).eval()

    @torch.no_grad()
    def forward(self, emb_next, hidden, start, state):
        """emb_next: 位置 t+1 的 token 的嵌入；hidden: 主模型在位置 t 的输出（都是 [B, T, H]）-> 位置 t+2 的预测（过 lm_head 之前）"""
        x = self.fc(torch.cat([self.pre_fc_norm_embedding(emb_next), self.pre_fc_norm_hidden(hidden)], dim=-1))
        cos, sin = rope_tables(self.cfg, torch.arange(start, start + x.shape[1], device=x.device))
        return self.norm(self.layers[0](x, cos, sin, state))


@torch.no_grad()
def forward_hidden(model, ids, state):
    """和 Model.forward 一样，只是把最后一层的输出（过最终 RMSNorm 之后）也带出来给草稿头用"""
    start = state.pos
    x = model.embed_tokens(ids)
    cos, sin = rope_tables(model.cfg, torch.arange(start, start + ids.shape[1], device=ids.device))
    for i, layer in enumerate(model.layers):
        x = layer(x, cos, sin, state.layers[i])
    state.pos += ids.shape[1]
    hidden = model.norm(x)
    return model.lm_head(hidden), hidden


def snapshot(state):
    """验证前记下状态：KV cache 只记长度（回退就是截断），记忆表和卷积窗口要复制一份"""
    return state.pos, [s.k.shape[2] if isinstance(s, AttentionState) and s.k is not None else
                       (None if isinstance(s, AttentionState) else (s.conv.clone(), s.S.clone())) for s in state.layers]


def rollback(state, snap):
    state.pos, saved = snap
    for s, v in zip(state.layers, saved):
        if isinstance(s, AttentionState):
            if v is not None:
                s.k, s.v = s.k[:, :, :v], s.v[:, :, :v]
        else:
            s.conv, s.S = v


@torch.no_grad()
def generate_mtp(model, mtp, ids, max_new_tokens=100, stop=()):
    """贪心 + 一个草稿 token 的投机解码。返回 (token 列表, 验证轮数, 猜中次数)"""
    state, draft_state = State(model), AttentionState()
    logits, hidden = forward_hidden(model, ids, state)                       # 主模型 prefill
    tok = int(logits[0, -1].argmax())                                         # 主模型给出的下一个 token（已确定）
    nxt_emb = model.embed_tokens(torch.tensor([ids[0, 1:].tolist() + [tok]], device=ids.device))   # 草稿头 prefill：每个位置配上它的下一个 token
    draft_logits = model.lm_head(mtp(nxt_emb, hidden, 0, draft_state))
    out, rounds, hits = [tok], 0, 0
    while len(out) < max_new_tokens and tok not in stop:
        draft = int(draft_logits[0, -1].argmax())                             # 草稿：猜 tok 后面那个
        snap = snapshot(state)
        logits, hidden = forward_hidden(model, torch.tensor([[tok, draft]], device=ids.device), state)   # 一次验证两个
        rounds += 1
        tok2 = int(logits[0, 0].argmax())                                     # 主模型自己算的"tok 后面那个"
        if tok2 == draft:                                                      # 猜中：draft 和主模型在 draft 之后的预测都收下
            hits += 1
            tok3 = int(logits[0, 1].argmax())
            out += [draft, tok3]
            nxt_emb = model.embed_tokens(torch.tensor([[draft, tok3]], device=ids.device))
            draft_logits = model.lm_head(mtp(nxt_emb, hidden, state.pos - 2, draft_state))
            tok = tok3
            if draft in stop:
                out.pop()
                break
        else:                                                                  # 猜错：退回验证前，只重算 tok 这一个
            rollback(state, snap)
            draft_state.k, draft_state.v = draft_state.k[:, :, :state.pos], draft_state.v[:, :, :state.pos]
            logits, hidden = forward_hidden(model, torch.tensor([[tok]], device=ids.device), state)
            out.append(tok2)
            nxt_emb = model.embed_tokens(torch.tensor([[tok2]], device=ids.device))
            draft_logits = model.lm_head(mtp(nxt_emb, hidden, state.pos - 1, draft_state))
            tok = tok2
    return out, rounds, hits
```

草稿头的输入是两样东西拼起来：位置 $t$ 的主模型输出 $h_t$（过了最终 RMSNorm），和位置 $t+1$ 的 token 的嵌入 $e_{t+1}$。各自归一化后拼成 2048 维，`fc` 压回 1024，过一层全注意力解码层（有自己的 KV cache），再归一化、过共用的输出层，得到位置 $t+2$ 的预测。它比主模型多知道一个 token（$t+1$ 是主模型刚刚确定的），所以猜得准。

投机解码的流程（贪心版）：

1. 主模型确定了 token $a$；草稿头根据 $(h, e_a)$ 猜 $b$。
2. 主模型一次算 $[a, b]$ 两个位置：第一个位置给出"$a$ 后面应该是什么"$= a'$，第二个位置给出"$b$ 后面应该是什么"$= c$。
3. 若 $a' = b$：猜中，收下 $b$ 和 $c$，一次前向出了两个 token；否则只收 $a'$，并把状态退回到验证前。

**回退是混合架构的坑。** KV cache 回退就是截断（`k[:, :, :n]`），但线性注意力的记忆表已经被 $b$ 改写了，没法"减掉"。这里的做法是验证前把 18 张记忆表和卷积窗口复制一份（19 MB，几毫秒），猜错就恢复——vLLM / SGLang 对 Mamba、DeltaNet 这类层也是类似的处理。

## 接受率和加速比

```python title="ch14_mtp.py" ci="loose"
"""第 14 章：草稿头 + 投机解码，和普通贪心对答案、比速度"""
import os
import time

from qwen.chat import Chat, build_prompt
from qwen.model import Model
from qwen.mtp import MTP, generate_mtp
from qwen.sample import generate
from qwen.weights import find_model

import torch
torch.set_num_threads(max(1, os.cpu_count() // 2))
path = find_model()
model = Model.from_pretrained(path)
mtp = MTP.from_pretrained(model.cfg, path)
chat = Chat(model, path)
print(f"草稿头 {sum(p.numel() for p in mtp.parameters()) / 1e6:.1f}M 参数：{[k for k, _ in mtp.named_children()]}")

for q in ("用一句话介绍一下 Transformer。", "写一段 200 字的短文，介绍长江。"):
    ids = chat.encode(build_prompt([{"role": "user", "content": q}]))
    t = time.time()
    plain = generate(model, ids, max_new_tokens=120, stop=chat.stop, temperature=0)
    t_plain = time.time() - t
    t = time.time()
    spec, rounds, hits = generate_mtp(model, mtp, ids, max_new_tokens=120, stop=chat.stop)
    t_spec = time.time() - t
    print(f"\n{q}")
    print(f"  普通贪心：{len(plain)} 个 token，{t_plain:.2f} 秒（{len(plain) / t_plain:.1f} tok/s）")
    print(f"  投机解码：{len(spec)} 个 token，{t_spec:.2f} 秒（{len(spec) / t_spec:.1f} tok/s）；验证 {rounds} 轮，猜中 {hits} 次（{hits / rounds:.0%}）")
    print(f"  两者逐 token 一致：{spec[:len(plain)] == plain[:len(spec)]}")
    print(f"  {chat.tk.decode(spec)[:60]}…")
```

```text title="输出"
草稿头 20.5M 参数：['fc', 'pre_fc_norm_embedding', 'pre_fc_norm_hidden', 'layers', 'norm']

用一句话介绍一下 Transformer。
  普通贪心：38 个 token，2.09 秒（18.2 tok/s）
  投机解码：38 个 token，2.89 秒（13.1 tok/s）；验证 25 轮，猜中 13 次（52%）
  两者逐 token 一致：True
  Transformer 是一种基于深度学习的神经网络架构，通过自注意力机制解决了传统卷积神经网络在处理长距离依赖和复杂语…

写一段 200 字的短文，介绍长江。
  普通贪心：120 个 token，7.00 秒（17.1 tok/s）
  投机解码：120 个 token，6.06 秒（19.8 tok/s）；验证 63 轮，猜中 56 次（89%）
  两者逐 token 一致：True
  长江，中国第一长河，发源于青藏高原，流经青海、西藏、四川、云南、贵州、江西、湖北、湖南、广东、福建、浙江、上海、江苏、安…
```

两点观察：

- **结果和普通贪心完全一致**——投机解码不改变输出，只改变算法。这是它和"用小模型代替大模型"的本质区别。
- **接受率看文本**：写短文这种"顺"的任务接受率近 90%，加速 10% 多；短回答接受率只有一半，反而变慢。CPU 上加速有限，因为一次算 2 个 token 的成本（这台机器约 1.35 倍）比 GPU 上高得多——GPU 上 2 个 token 和 1 个几乎一样贵，同样的接受率能换来 1.5 倍以上。官方给 vLLM 的推荐是一次猜 2 个（`num_speculative_tokens=2`），SGLang 是 3 步 4 个草稿 token，都是在接受率和验证成本之间折中。

!!! interview "怎么讲清楚"
    讲投机解码，三句话：小模型猜、大模型并行验证、输出分布不变（贪心下逐 token 相同，采样下用拒绝采样保证）。MTP 头是"自带的草稿模型"，它比独立小模型准，因为它看到了主模型的隐状态。再说混合架构的特殊处理：KV cache 可以截断，递推状态要快照或重算——这是面向这代模型的推理引擎都要解决的问题。

## 小结

- [x] 草稿头：`fc(cat(norm(e_{t+1}), norm(h_t)))` → 一层注意力 → 归一化 → 共用输出层。
- [x] 投机解码：猜一个、验证两个；猜中收两个，猜错回退；输出和贪心一致。
- [x] 接受率随文本变化（50% ~ 90%）；递推状态用快照回退。

**这一章新建的文件**：`qwen/mtp.py`。
