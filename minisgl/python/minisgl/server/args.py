from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import List, Tuple

import torch
from minisgl.distributed import DistributedInfo
from minisgl.scheduler import SchedulerConfig
from minisgl.utils import init_logger


@dataclass(frozen=True)
class ServerArgs(SchedulerConfig):
    server_host: str = "127.0.0.1"
    server_port: int = 1919
    num_tokenizer: int = 0  # 0：分词和反分词共用一个进程
    silent_output: bool = False

    @property
    def share_tokenizer(self) -> bool:
        return self.num_tokenizer == 0

    @property
    def zmq_frontend_addr(self) -> str:  # tokenizer -> API server
        return "ipc:///tmp/minisgl_3" + self._unique_suffix

    @property
    def zmq_tokenizer_addr(self) -> str:  # API server -> tokenizer
        if self.share_tokenizer:
            return self.zmq_detokenizer_addr  # 共用进程：两个方向的消息进同一个队列
        return "ipc:///tmp/minisgl_4" + self._unique_suffix

    @property
    def tokenizer_create_addr(self) -> bool:
        return self.share_tokenizer

    @property
    def backend_create_detokenizer_link(self) -> bool:
        return not self.share_tokenizer

    @property
    def frontend_create_tokenizer_link(self) -> bool:
        return not self.share_tokenizer


def parse_args(args: List[str], run_shell: bool = False) -> Tuple[ServerArgs, bool]:
    from minisgl.attention import validate_attn_backend
    from minisgl.kvcache import SUPPORTED_CACHE_MANAGER

    p = argparse.ArgumentParser(description="mini-sglang server")
    p.add_argument("--model-path", "--model", required=True)
    p.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    p.add_argument("--device", default="auto", help="auto / cpu / cuda")
    p.add_argument("--tp-size", "--tp", dest="tp_size", type=int, default=1)
    p.add_argument("--max-running-requests", dest="max_running_req", type=int,
                   default=ServerArgs.max_running_req)
    p.add_argument("--max-seq-len-override", type=int, default=None)
    p.add_argument("--memory-ratio", type=float, default=ServerArgs.memory_ratio)
    p.add_argument("--dummy-weight", dest="use_dummy_weight", action="store_true")
    p.add_argument("--host", dest="server_host", default=ServerArgs.server_host)
    p.add_argument("--port", dest="server_port", type=int, default=ServerArgs.server_port)
    p.add_argument("--cuda-graph-max-bs", "--graph", type=int, default=None)
    p.add_argument("--num-tokenizer", type=int, default=0)
    p.add_argument("--max-prefill-length", dest="max_extend_tokens", type=int,
                   default=ServerArgs.max_extend_tokens)
    p.add_argument("--num-pages", dest="num_page_override", type=int, default=None)
    p.add_argument("--page-size", type=int, default=1)
    p.add_argument("--attention-backend", "--attn", type=validate_attn_backend, default="auto")
    p.add_argument("--cache-type", default="radix", choices=SUPPORTED_CACHE_MANAGER.supported_names())
    p.add_argument("--shell-mode", "--shell", action="store_true")
    kwargs = vars(p.parse_args(args)).copy()

    run_shell |= kwargs.pop("shell_mode")
    if run_shell:  # 交互模式只有一个用户：单请求、只录 batch size 为 1 的 graph
        kwargs.update(cuda_graph_max_bs=1, max_running_req=1, silent_output=True)
    kwargs["model_path"] = os.path.expanduser(kwargs["model_path"])
    dtype = kwargs.pop("dtype")
    if dtype == "auto":
        from minisgl.utils import cached_load_hf_config

        dtype = cached_load_hf_config(kwargs["model_path"]).dtype
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    kwargs["dtype"] = dtype_map[dtype] if isinstance(dtype, str) else dtype
    kwargs["tp_info"] = DistributedInfo(0, kwargs.pop("tp_size"))
    kwargs["distributed_port"] = kwargs["server_port"] + 1
    result = ServerArgs(**kwargs)
    init_logger(__name__).info(f"Parsed arguments: {result}")
    return result, run_shell
