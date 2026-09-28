"""第 1 章：跟踪一个请求在 prefill 和 decode 中各个长度字段的变化。"""

import torch
from minisgl.core import Req, SamplingParams

req = Req(input_ids=torch.arange(6, dtype=torch.int32), table_idx=0, cached_len=2, output_len=3,
          uid=0, sampling_params=SamplingParams(max_tokens=3), cache_handle=None)


def show(stage: str) -> None:
    print(f"{stage:<18} cached_len={req.cached_len}  device_len={req.device_len}  "
          f"extend_len={req.extend_len}  remain_len={req.remain_len}  can_decode={req.can_decode}")


show("接纳时（命中 2 个）")
for step in ("prefill 之后", "decode 1 之后", "decode 2 之后"):
    req.complete_one()
    req.append_host(torch.tensor([100], dtype=torch.int32))  # 假装采样出了 token 100
    show(step)
print("最终 input_ids:", req.input_ids.tolist())
