from __future__ import annotations

from datetime import timedelta
from typing import Any, Dict, NamedTuple, Tuple

import torch
import torch.distributed as dist
from minisgl.attention import create_attention_backend
from minisgl.core import Batch, Context, Req, reset_global_ctx, set_global_ctx
from minisgl.distributed import reset_tp_info, set_tp_info
from minisgl.kvcache import create_kvcache_pool
from minisgl.layers import set_rope_device
from minisgl.models import create_model, load_weight
from minisgl.moe import create_moe_backend
from minisgl.utils import (
    create_event,
    create_stream,
    div_even,
    get_free_memory,
    init_logger,
    is_cuda,
    set_stream,
    synchronize,
    torch_dtype,
)

from .config import EngineConfig
from .graph import GraphRunner
from .sample import BatchSamplingArgs, Sampler

logger = init_logger(__name__)


class ForwardOutput(NamedTuple):
    next_tokens_gpu: torch.Tensor  # 留在设备上：下一轮的输入直接从这里取，不必等拷回 CPU
    next_tokens_cpu: torch.Tensor  # 异步拷回 CPU 的副本：调度器用它判断是否结束、发给前端
    copy_done_event: Any  # 拷贝完成的事件；CPU 上是空对象


def _align_up_32(num: int) -> int:
    return (num + 31) // 32 * 32


def _gib(size: int) -> str:
    return f"{size / (1 << 30):.2f} GiB"


class Engine:
    """一个 TP rank 上的计算引擎：持有模型、KV 池、page table、注意力后端、采样器和 CUDA Graph。"""

    def __init__(self, config: EngineConfig):
        set_tp_info(rank=config.tp_info.rank, size=config.tp_info.size)
        self.config = config = _adjust_config(config)
        self.device = _pick_device(config)
        if is_cuda(self.device):
            torch.cuda.set_device(self.device)
        else:
            _set_cpu_threads(config.tp_info.size)
        torch.manual_seed(42)
        self.stream = create_stream(self.device)
        set_stream(self.stream)
        self.dtype = config.dtype
        self.ctx = Context(config.page_size)
        set_global_ctx(self.ctx)

        self.tp_cpu_group = self._init_communication(config)
        init_free_memory = self._sync_get_memory()[1]
        logger.info_rank0(f"Free memory before loading model: {_gib(init_free_memory)}")

        # ======================= 模型：先在 meta 设备上建空壳，再把权重"装"进去
        set_rope_device(self.device)
        with torch.device("meta"), torch_dtype(config.dtype):
            self.model = create_model(config.model_config)
        self.model.load_state_dict(self._load_weight_state_dict(config))

        # ======================= KV 池：剩下的显存都给它
        self.num_pages = self._determine_num_pages(init_free_memory, config)
        num_tokens = self.num_pages * config.page_size
        self.ctx.kv_cache = self.kv_cache = create_kvcache_pool(
            model_config=config.model_config,
            num_pages=self.num_pages + 1,  # 多出的 1 页给 dummy 请求，它的写入落在这里
            page_size=config.page_size,
            device=self.device,
            dtype=self.dtype,
        )

        # ======================= page table：每行对应一个请求，存每个 token 的 KV 位置
        self.max_seq_len = min(config.max_seq_len, num_tokens)
        aligned_max_seq_len = _align_up_32(self.max_seq_len)
        self.ctx.page_table = self.page_table = torch.zeros(
            (config.max_running_req + 1, aligned_max_seq_len),  # 最后一行给 dummy 请求
            dtype=torch.int32, device=self.device,
        )

        # ======================= 注意力后端、MoE 后端、采样器
        self.ctx.attn_backend = self.attn_backend = create_attention_backend(
            config.attention_backend, config.model_config)
        if config.model_config.is_moe:
            self.ctx.moe_backend = self.moe_backend = create_moe_backend(config.moe_backend)
        self.sampler = Sampler(self.device, config.model_config.vocab_size)

        # ======================= dummy 请求与 CUDA Graph
        self.dummy_req = Req(
            input_ids=torch.tensor([0], dtype=torch.int32),
            table_idx=config.max_running_req,
            cached_len=0,
            output_len=1,
            uid=-1,
            sampling_params=None,  # type: ignore[arg-type]
            cache_handle=None,  # type: ignore[arg-type]
        )
        self.page_table[self.dummy_req.table_idx].fill_(num_tokens)  # 指向 dummy 页
        self.graph_runner = GraphRunner(
            stream=self.stream,
            device=self.device,
            model=self.model,
            attn_backend=self.attn_backend,
            cuda_graph_bs=config.cuda_graph_bs,
            cuda_graph_max_bs=config.cuda_graph_max_bs,
            free_memory=init_free_memory,
            max_seq_len=aligned_max_seq_len,
            vocab_size=config.model_config.vocab_size,
            dummy_req=self.dummy_req,
        )

    def _init_communication(self, config: EngineConfig) -> Any:
        """TP>1 时建立进程组。GPU 上张量走 NCCL，另建一个 gloo 组做 CPU 侧的同步；CPU 上全用 gloo。"""
        if config.tp_info.size == 1:
            return None
        backend = "nccl" if is_cuda(self.device) else "gloo"
        dist.init_process_group(
            backend=backend,
            rank=config.tp_info.rank,
            world_size=config.tp_info.size,
            timeout=timedelta(seconds=config.distributed_timeout),
            init_method=config.distributed_addr,
        )
        return dist.new_group(backend="gloo") if backend == "nccl" else dist.group.WORLD

    def _load_weight_state_dict(self, config: EngineConfig) -> Dict[str, torch.Tensor]:
        if config.use_dummy_weight:
            return {k: torch.randn(v.shape, dtype=v.dtype, device=self.device) * 0.02
                    for k, v in self.model.state_dict().items()}
        return {k: v.to(self.dtype) for k, v in load_weight(config.model_path, self.device)}

    def _determine_num_pages(self, old_free_memory: int, config: EngineConfig) -> int:
        cache_per_page = (
            2  # K 和 V
            * config.model_config.head_dim
            * div_even(config.model_config.num_kv_heads, config.tp_info.size, allow_replicate=True)
            * config.page_size
            * self.dtype.itemsize
            * config.model_config.num_layers
        )
        num_pages = config.num_page_override
        if num_pages is None:
            new_free_memory = self._sync_get_memory()[1]
            model_memory = old_free_memory - new_free_memory
            available = int(config.memory_ratio * old_free_memory) - model_memory
            if not is_cuda(self.device):
                available = min(available, config.cpu_kv_cache_bytes)
            num_pages = available // cache_per_page
        assert num_pages > 1, "Not enough memory for KV cache"
        logger.info_rank0(f"KV cache: {num_pages * config.page_size} tokens, "
                          f"{_gib(num_pages * cache_per_page)}")
        return num_pages

    def _sync_get_memory(self) -> Tuple[int, int]:
        """各 rank 可用显存的最小值和最大值（相差太大说明有别的进程在占用 GPU）。"""
        synchronize(self.device)
        if is_cuda(self.device):
            torch.cuda.empty_cache()
        free = get_free_memory(self.device)
        if self.tp_cpu_group is None:
            return free, free
        t = torch.tensor([free, -free], dtype=torch.int64)
        dist.all_reduce(t, op=dist.ReduceOp.MIN, group=self.tp_cpu_group)
        return int(t[0]), -int(t[1])

    def forward_batch(self, batch: Batch, args: BatchSamplingArgs) -> ForwardOutput:
        with self.ctx.forward_batch(batch):
            if self.graph_runner.can_use_cuda_graph(batch):
                logits = self.graph_runner.replay(batch)
            else:
                logits = self.model.forward()
        for req in batch.reqs:
            req.complete_one()
        next_tokens_gpu = self.sampler.sample(logits[: batch.size], args).to(torch.int32)
        next_tokens_cpu = next_tokens_gpu.to("cpu", non_blocking=True)
        copy_done_event = create_event(self.device)
        copy_done_event.record(self.stream)
        return ForwardOutput(next_tokens_gpu, next_tokens_cpu, copy_done_event)

    def shutdown(self) -> None:
        self.graph_runner.destroy_cuda_graphs()
        if dist.is_initialized():
            dist.destroy_process_group()
        # 清掉进程级的全局状态，这样同一个进程里可以再建一个引擎（测试里常用）
        reset_global_ctx()
        reset_tp_info()


def _set_cpu_threads(tp_size: int) -> None:
    """CPU 上默认的线程数等于逻辑核数，小矩阵乘法会因为线程同步开销慢上几十倍。
    这里按"物理核数 / TP 进程数"设置（没有设置 OMP_NUM_THREADS 时）。"""
    import os

    if "OMP_NUM_THREADS" not in os.environ:
        torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2 // tp_size))


def _pick_device(config: EngineConfig) -> torch.device:
    if config.device != "auto":
        return torch.device(config.device)
    if torch.cuda.is_available():
        return torch.device(f"cuda:{config.tp_info.rank}")
    return torch.device("cpu")


def _adjust_config(config: EngineConfig) -> EngineConfig:
    """解析 auto 选项。EngineConfig 是冻结的 dataclass，这里用 replace 生成新对象。"""
    from dataclasses import replace

    on_cuda = _pick_device(config).type == "cuda"
    # FlashInfer 和 FlashAttention 都只有半精度的 kernel：fp32（与 Hugging Face 逐 token 对比
    # 时才用）在 CUDA 上也只能走 PyTorch 实现
    fast_kernels = on_cuda and config.dtype in (torch.float16, torch.bfloat16)
    changes: Dict[str, Any] = {}
    if config.attention_backend == "auto":
        if not fast_kernels:
            changes["attention_backend"] = "torch"
        else:
            major, _ = torch.cuda.get_device_capability()
            changes["attention_backend"] = "fa,fi" if major == 9 else "fi"
    if config.moe_backend == "auto":
        changes["moe_backend"] = "fused" if fast_kernels else "torch"
    return replace(config, **changes) if changes else config
