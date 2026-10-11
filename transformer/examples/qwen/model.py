"""整模型：嵌入 -> 24 个解码层 -> 最后的 RMSNorm -> 输出层。从这个文件起，前向里没有一行 Hugging Face 的代码。"""
import torch
import torch.nn as nn

from qwen.embedding import Embedding, LMHead
from qwen.layer import DecoderLayer
from qwen.norm import RMSNorm
from qwen.rope import rope_tables
from qwen.weights import Config, find_model, load_weights


class Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(DecoderLayer(cfg, kind) for kind in cfg.layer_types)
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.lm_head = LMHead(self.embed_tokens.weight)                       # 和嵌入层共用权重

    @classmethod
    def from_pretrained(cls, path=None, dtype=torch.float32):
        path = path or find_model()
        model = cls(Config.load(path))
        missing, unexpected = model.load_state_dict(load_weights(path, dtype=dtype), strict=False, assign=True)   # assign=True：不复制，直接接管
        assert not unexpected and missing == ["lm_head.weight"], (missing, unexpected)   # 文件里没有 lm_head：它就是嵌入层
        model.lm_head.weight = model.embed_tokens.weight                                  # 接管之后嵌入表换了对象，输出层重新指过去
        return model.to(dtype).eval()

    @torch.no_grad()
    def forward(self, ids, state=None, mask=None):
        """ids: [B, T] -> logits [B, T, 词表]。带 state 时 ids 是接在已有 token 后面的新 token；
        mask: [B, T]，批处理时标出哪些是填充位（第 13 章），不传就全是真 token。"""
        if mask is None:
            mask = torch.ones_like(ids, dtype=torch.bool)
        if state is not None:                                                     # 整段对话的 mask 要攒起来：注意力的 key 包括以前的 token
            state.mask = mask if state.mask is None else torch.cat([state.mask, mask], dim=1)
        kv_mask = state.mask if state is not None else mask
        positions = (kv_mask.cumsum(1) - 1).clamp(min=0)[:, -ids.shape[1]:]       # 每行各数各的位置：填充位不算
        x = self.embed_tokens(ids)
        cos, sin = rope_tables(self.cfg, positions)
        for i, layer in enumerate(self.layers):
            x = layer(x, cos, sin, state.layers[i] if state is not None else None, mask, kv_mask)
        if state is not None:
            state.pos += ids.shape[1]
        return self.lm_head(self.norm(x))
