"""两种状态。全注意力层：KV cache，每个 token 追加一份 key/value，随长度增长；
线性注意力层：一张记忆表 S 加一个 3 个 token 的卷积窗口，不管读了多少 token 都是固定大小。
"""
import torch


class AttentionState:
    def __init__(self):
        self.k = self.v = None                                     # [B, kv 头, 已有 token 数, head_dim]，第一次 prefill 时创建

    def numel(self):
        return 0 if self.k is None else self.k.numel() + self.v.numel()


class DeltaState:
    def __init__(self, cfg, batch, device, dtype):
        conv_dim = 2 * cfg.linear_num_key_heads * cfg.linear_key_head_dim + cfg.linear_num_value_heads * cfg.linear_value_head_dim
        self.conv = torch.zeros(batch, conv_dim, cfg.linear_conv_kernel_dim - 1, device=device, dtype=dtype)   # 最近 3 个 token 的卷积输入
        self.S = torch.zeros(batch, cfg.linear_num_value_heads, cfg.linear_key_head_dim, cfg.linear_value_head_dim,
                             device=device, dtype=torch.float32)                                                # 记忆表：[B, 头, dk, dv]，永远 fp32

    def numel(self):
        return self.conv.numel() + self.S.numel()


class State:
    """一次对话的全部状态：每层一个，外加已经处理了多少个 token（下一个 token 的位置）"""

    def __init__(self, model, batch=1):
        p = next(model.parameters())
        self.layers = [AttentionState() if kind == "full_attention" else DeltaState(model.cfg, batch, p.device, p.dtype)
                       for kind in model.cfg.layer_types]
        self.pos = 0                                                       # 已经处理了多少个 token
        self.mask = None                                                   # [B, pos]：批处理时哪些位置是真 token

    def megabytes(self):
        return sum(s.numel() for s in self.layers) * 4 / 2 ** 20
