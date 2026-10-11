# 14. MTP: speculative decoding with the built-in draft head

<p class="lead">The <code>mtp.*</code> tensors in the weight file are a one-layer "draft head" (20M parameters): given the main model's last-layer output plus the embedding of the next token, it guesses the token <strong>after next</strong>. In training it is an auxiliary objective (Multi-Token Prediction); in inference it serves speculative decoding: the draft head guesses one token, the main model verifies two tokens in one pass, and a correct guess is a free token. This chapter writes it, measures the acceptance rate and the speed-up, and runs into a trap peculiar to the hybrid architecture: a recurrent state cannot be truncated and rolled back like a KV cache.</p>

!!! note "What this chapter runs"
    `ch14_mtp.py`, about 40 seconds.

@@files@@

## The draft head {#草稿头}

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
        self.fc = nn.Linear(2 * H, H, bias=False)                             # concatenate "the main model's output" and "the next token's embedding" and project back to H dims
        self.pre_fc_norm_embedding = RMSNorm(H, cfg.rms_norm_eps)
        self.pre_fc_norm_hidden = RMSNorm(H, cfg.rms_norm_eps)
        self.layers = nn.ModuleList([DecoderLayer(cfg, "full_attention")])     # a single layer, full attention
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
    logits, hidden = forward_hidden(model, ids, state)                       # main model prefill
    tok = int(logits[0, -1].argmax())                                         # the next token from the main model (already decided)
    nxt_emb = model.embed_tokens(torch.tensor([ids[0, 1:].tolist() + [tok]], device=ids.device))   # draft-head prefill: every position paired with its next token
    draft_logits = model.lm_head(mtp(nxt_emb, hidden, 0, draft_state))
    out, rounds, hits = [tok], 0, 0
    while len(out) < max_new_tokens and tok not in stop:
        draft = int(draft_logits[0, -1].argmax())                             # draft: guess the token after tok
        snap = snapshot(state)
        logits, hidden = forward_hidden(model, torch.tensor([[tok, draft]], device=ids.device), state)   # verify two at once
        rounds += 1
        tok2 = int(logits[0, 0].argmax())                                     # the main model's own "token after tok"
        if tok2 == draft:                                                      # hit: accept draft and the main model's prediction after draft
            hits += 1
            tok3 = int(logits[0, 1].argmax())
            out += [draft, tok3]
            nxt_emb = model.embed_tokens(torch.tensor([[draft, tok3]], device=ids.device))
            draft_logits = model.lm_head(mtp(nxt_emb, hidden, state.pos - 2, draft_state))
            tok = tok3
            if draft in stop:
                out.pop()
                break
        else:                                                                  # miss: roll back to before the verification and recompute only tok
            rollback(state, snap)
            draft_state.k, draft_state.v = draft_state.k[:, :, :state.pos], draft_state.v[:, :, :state.pos]
            logits, hidden = forward_hidden(model, torch.tensor([[tok]], device=ids.device), state)
            out.append(tok2)
            nxt_emb = model.embed_tokens(torch.tensor([[tok2]], device=ids.device))
            draft_logits = model.lm_head(mtp(nxt_emb, hidden, state.pos - 1, draft_state))
            tok = tok2
    return out, rounds, hits
```

The draft head's input is two things concatenated: the main model's output $h_t$ at position $t$ (after the final RMSNorm) and the embedding $e_{t+1}$ of the token at position $t+1$. Each is normalized, they are concatenated into 2048 dimensions, `fc` projects back to 1024, a full-attention decoder layer (with its own KV cache) follows, then a normalization and the shared output layer, giving the prediction for position $t+2$. It knows one token more than the main model ($t+1$ was just decided by the main model), which is why it guesses well.

The speculative decoding loop (greedy version):

1. The main model has decided token $a$; from $(h, e_a)$ the draft head guesses $b$.
2. The main model computes the two positions $[a, b]$ in one pass: the first position gives "what should follow $a$" $= a'$, the second gives "what should follow $b$" $= c$.
3. If $a' = b$: a hit; accept both $b$ and $c$ — one forward pass produced two tokens. Otherwise accept only $a'$ and roll the state back to before the verification.

![Figure: one round of speculative decoding with the draft head](assets/figures/speculative.svg){.aig-svg}

**Rolling back is the hybrid architecture's trap.** Rolling back a KV cache is a truncation (`k[:, :, :n]`), but the linear attention's memory table has already been rewritten by $b$ and cannot be "subtracted". The approach here is to copy the 18 memory tables and convolution windows before verification (19 MB, a few milliseconds) and restore them on a miss — vLLM / SGLang handle Mamba- and DeltaNet-style layers in a similar way.

## Acceptance rate and speed-up {#接受率和加速比}

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

```text title="output"
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

Two observations:

- **The result is identical to plain greedy decoding** — speculative decoding changes the algorithm, not the output. That is what sets it apart from "replacing the big model with a small one".
- **The acceptance rate depends on the text**: a "smooth" task like writing a short essay accepts nearly 90% and gains a little over 10%; a short answer accepts only half and actually gets slower. The gain on a CPU is limited because computing 2 tokens at once costs more here (about 1.35× on this machine) than on a GPU — on a GPU 2 tokens cost almost the same as 1, and the same acceptance rate buys 1.5× or more. The official recommendation for vLLM is to guess 2 tokens at a time (`num_speculative_tokens=2`) and for SGLang 3 steps with 4 draft tokens, both compromises between the acceptance rate and the verification cost.

!!! interview "How to explain it"
    Speculative decoding in three sentences: a small model guesses, the large model verifies in parallel, the output distribution is unchanged (identical token by token under greedy decoding, guaranteed by rejection sampling when sampling). The MTP head is a "built-in draft model", more accurate than an independent small model because it sees the main model's hidden state. Then the hybrid architecture's special handling: a KV cache can be truncated, a recurrent state has to be snapshotted or recomputed — every inference engine targeting this generation of models has to solve this.

## Summary {#小结}

- [x] The draft head: `fc(cat(norm(e_{t+1}), norm(h_t)))` → one attention layer → normalization → the shared output layer.
- [x] Speculative decoding: guess one, verify two; accept two on a hit, roll back on a miss; the output is identical to greedy decoding.
- [x] The acceptance rate varies with the text (50% ~ 90%); the recurrent state is rolled back from a snapshot.

**File created in this chapter**: `qwen/mtp.py`.
