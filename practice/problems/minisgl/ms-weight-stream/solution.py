import re

import numpy as np

_MERGE = {".q_proj": (".qkv_proj", "q", ("q", "k", "v")), ".k_proj": (".qkv_proj", "k", ("q", "k", "v")),
          ".v_proj": (".qkv_proj", "v", ("q", "k", "v")), ".gate_proj": (".gate_up_proj", "gate", ("gate", "up")),
          ".up_proj": (".gate_up_proj", "up", ("gate", "up"))}
_EXPERT = re.compile(r"^(?P<prefix>.+\.experts)\.(?P<idx>\d+)\.(?P<name>.+)$")


def load_weights(items, num_experts=0):
    merge_buf, expert_buf = {}, {}
    for name, arr in items:
        info = next(((name.replace(suf, fused), slot, slots) for suf, (fused, slot, slots) in _MERGE.items()
                     if suf in name), None)
        if info is None:
            out = (name, arr)
        else:
            merged, slot, slots = info
            merge_buf.setdefault(merged, {})[slot] = arr
            if not all(s in merge_buf[merged] for s in slots):
                continue
            parts = merge_buf.pop(merged)
            out = (merged, np.concatenate([parts[s] for s in slots], axis=0))
        m = _EXPERT.match(out[0]) if num_experts else None
        if m is None:
            yield out
            continue
        packed = f"{m['prefix']}.{m['name'].removesuffix('.weight')}"
        slots_e = expert_buf.setdefault(packed, {})
        slots_e[int(m["idx"])] = out[1]
        if len(slots_e) < num_experts:
            continue
        yield packed, np.stack([slots_e[i] for i in range(num_experts)])
        del expert_buf[packed]
    if merge_buf or expert_buf:
        raise ValueError(f"有没凑齐的组：{sorted(merge_buf) + sorted(expert_buf)}")
