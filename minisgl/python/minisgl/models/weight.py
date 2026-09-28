"""流式权重加载：一边从 safetensors 读，一边按 TP 切分、合并 q/k/v 与 gate/up、打包 MoE 专家。

峰值内存只多出一个完整张量加上一小块合并缓冲区，而不是整个 checkpoint。
"""

from __future__ import annotations

import glob
import re
from typing import Dict, Iterator, Tuple

import safetensors
import torch
from minisgl.distributed import get_tp_info
from minisgl.utils import cached_load_hf_config, div_ceil, download_hf_weight

from .config import ModelConfig

_SPLIT_DIM_0 = [".q_proj", ".k_proj", ".v_proj", ".gate_proj", ".up_proj"]  # 列并行：切输出维
_SPLIT_DIM_1 = [".o_proj", ".down_proj"]  # 行并行：切输入维

# 分开存的投影 -> 合并后的名字，以及合并时的拼接顺序
_MERGE_GROUPS = {
    ".q_proj": (".qkv_proj", "q", ("q", "k", "v")),
    ".k_proj": (".qkv_proj", "k", ("q", "k", "v")),
    ".v_proj": (".qkv_proj", "v", ("q", "k", "v")),
    ".gate_proj": (".gate_up_proj", "gate", ("gate", "up")),
    ".up_proj": (".gate_up_proj", "up", ("gate", "up")),
}
_EXPERT_PATTERN = re.compile(r"^(?P<prefix>.+\.experts)\.(?P<idx>\d+)\.(?P<name>.+)$")


def shard_tensor(key: str, value: torch.Tensor, rank: int, size: int,
                 num_kv_heads: int) -> torch.Tensor:
    """取出第 rank 个分片。"""
    if size == 1:
        return value
    if any(s in key for s in _SPLIT_DIM_0):
        is_kv = ".k_proj" in key or ".v_proj" in key
        if is_kv and num_kv_heads < size:  # KV 头比 rank 少：几个 rank 共用（复制）同一个 KV 头
            head_dim = value.shape[0] // num_kv_heads
            head = rank * num_kv_heads // size
            return value[head * head_dim:(head + 1) * head_dim].clone()
        return value.chunk(size, dim=0)[rank].clone()
    if any(s in key for s in _SPLIT_DIM_1):
        return value.chunk(size, dim=1)[rank].clone()
    if "lm_head" in key or "embed_tokens" in key:  # 词表并行：按行切，最后一段可能短一些
        per = div_ceil(value.shape[0], size)
        return value[rank * per:min((rank + 1) * per, value.shape[0])].clone()
    return value  # 其余（各种 norm、MoE 路由）每个 rank 一份完整的


def _merge_info(key: str) -> Tuple[str, str, Tuple[str, ...]] | None:
    for suffix, (fused, slot, slots) in _MERGE_GROUPS.items():
        if suffix in key:
            return key.replace(suffix, fused), slot, slots
    return None


def load_weight(model_path: str, device: torch.device) -> Iterator[Tuple[str, torch.Tensor]]:
    config = ModelConfig.from_hf(cached_load_hf_config(model_path))
    tp = get_tp_info()
    folder = download_hf_weight(model_path)
    files = sorted(glob.glob(f"{folder}/*.safetensors"))
    merge_buf: Dict[str, Dict[str, torch.Tensor]] = {}
    expert_buf: Dict[str, Dict[int, torch.Tensor]] = {}
    for file in files:
        with safetensors.safe_open(file, framework="pt", device=str(device)) as f:
            for name in f.keys():
                tensor = shard_tensor(name, f.get_tensor(name), tp.rank, tp.size,
                                      config.num_kv_heads)
                if (info := _merge_info(name)) is None:
                    out = (name, tensor)
                else:
                    merged, slot, slots = info
                    merge_buf.setdefault(merged, {})[slot] = tensor
                    if not all(s in merge_buf[merged] for s in slots):
                        continue  # 等同一组的其他投影都读到了再合并
                    parts = merge_buf.pop(merged)
                    out = (merged, torch.cat([parts[s] for s in slots], dim=0))

                if config.is_moe and (m := _EXPERT_PATTERN.match(out[0])) is not None:
                    packed = f"{m['prefix']}.{m['name'].removesuffix('.weight')}"
                    slots_e = expert_buf.setdefault(packed, {})
                    slots_e[int(m["idx"])] = out[1]
                    if len(slots_e) < config.num_experts:
                        continue  # 等齐所有专家再打包成 [E, ...]
                    yield packed, torch.stack([slots_e[i] for i in range(config.num_experts)])
                    del expert_buf[packed]
                else:
                    yield out
    assert not merge_buf, f"Incomplete merge groups: {list(merge_buf)}"
    assert not expert_buf, f"Incomplete expert tensors: {list(expert_buf)}"
