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
