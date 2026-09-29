import re

import numpy as np

_MERGE = {".q_proj": (".qkv_proj", "q", ("q", "k", "v")), ".k_proj": (".qkv_proj", "k", ("q", "k", "v")),
          ".v_proj": (".qkv_proj", "v", ("q", "k", "v")), ".gate_proj": (".gate_up_proj", "gate", ("gate", "up")),
          ".up_proj": (".gate_up_proj", "up", ("gate", "up"))}
_EXPERT = re.compile(r"^(?P<prefix>.+\.experts)\.(?P<idx>\d+)\.(?P<name>.+)$")


def load_weights(items, num_experts=0):
    items = list(items)                   # 一次性读进来：不是流式的
    for name, arr in items:
        yield name, arr
