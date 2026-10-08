"""nano_engine.py —— 一个最小但完整的推理引擎：调度器 + 模型执行器 + 采样。

调度思路与 vLLM V1 相同：没有"prefill 阶段"和"decode 阶段"之分，每个请求只记录
num_computed（KV 已在缓存中的 token 数）和 num_tokens（提示词 + 已生成）。每一步，调度器在
token 预算内给请求分配 token，让 num_computed 追上 num_tokens。分块 prefill、decode、抢占后的重算
都是这个规则的特例。
"""

import math
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum

import torch

from paged import BlockPool, PagedKVCache
from runner import ModelRunner, build_batch


@dataclass
class SamplingParams:
    max_tokens: int = 64
    temperature: float = 0.0          # 0 表示贪心
    top_p: float = 1.0
    top_k: int = 0                    # 0 表示不限制
    seed: int | None = None
    stop_token_ids: tuple[int, ...] = ()
    ignore_eos: bool = False


class Status(Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"


class Request:
    def __init__(self, request_id: str, prompt_ids: list[int], params: SamplingParams):
        self.request_id, self.prompt_ids, self.params = request_id, list(prompt_ids), params
        self.output_ids: list[int] = []
        self.num_computed = 0            # 已写入 KV Cache 的 token 数
        self.block_ids: list[int] = []   # 块表
        self.status = Status.WAITING
        self.finish_reason: str | None = None
        self.num_preemptions = 0
        self.num_cached_tokens = 0       # 前缀缓存命中的 token 数
        self.generator = torch.Generator().manual_seed(params.seed) if params.seed is not None else None
        self.arrival_time = time.perf_counter()
        self.token_times: list[float] = []

    @property
    def token_ids(self) -> list[int]:
        return self.prompt_ids + self.output_ids

    @property
    def num_tokens(self) -> int:
        return len(self.prompt_ids) + len(self.output_ids)


@dataclass
class SchedulerOutput:
    scheduled: list[tuple[Request, int]]   # (请求, 本步为它计算的 token 数)
    num_preempted: int


class Scheduler:
    def __init__(self, pool: BlockPool, block_size: int, max_num_batched_tokens: int = 256,
                 max_num_seqs: int = 16, enable_chunked_prefill: bool = True):
        self.pool, self.block_size = pool, block_size
        self.max_num_batched_tokens, self.max_num_seqs = max_num_batched_tokens, max_num_seqs
        self.enable_chunked_prefill = enable_chunked_prefill
        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []

    def add_request(self, req: Request) -> None:
        self.waiting.append(req)

    def has_unfinished(self) -> bool:
        return bool(self.waiting or self.running)

    def schedule(self) -> SchedulerOutput:
        budget = self.max_num_batched_tokens
        scheduled: list[tuple[Request, int]] = []
        num_preempted = 0

        # 1. 先调度正在运行的请求（decode 或未完成的分块 prefill），按到达顺序
        i = 0
        while i < len(self.running) and budget > 0:
            req = self.running[i]
            n = min(req.num_tokens - req.num_computed, budget)
            while not self._allocate(req, req.num_computed + n):
                victim = self.running.pop()            # 显存不够：抢占优先级最低（最后到达）的请求
                self._preempt(victim)
                num_preempted += 1
                if victim is req:
                    break
            else:
                scheduled.append((req, n))
                budget -= n
                i += 1
                continue
            break                                      # 当前请求自己被抢占了，本轮不再调度运行队列

        # 2. 再从等待队列取新请求（本步发生过抢占就不接新请求，避免抖动）
        while self.waiting and budget > 0 and not num_preempted and len(self.running) < self.max_num_seqs:
            req = self.waiting[0]
            self._lookup_prefix_cache(req)
            n = req.num_tokens - req.num_computed
            if not self.enable_chunked_prefill and n > budget:
                self._release_prefix_hit(req)
                break
            n = min(n, budget)
            if not self._allocate(req, req.num_computed + n):
                self._release_prefix_hit(req)
                break
            self.waiting.popleft()
            req.status = Status.RUNNING
            self.running.append(req)
            scheduled.append((req, n))
            budget -= n
        return SchedulerOutput(scheduled, num_preempted)

    def _allocate(self, req: Request, num_tokens: int) -> bool:
        """保证请求的块表能容纳 num_tokens 个 token。"""
        need = math.ceil(num_tokens / self.block_size) - len(req.block_ids)
        if need <= 0:
            return True
        blocks = self.pool.allocate(need)
        if blocks is None:
            return False
        req.block_ids += blocks
        return True

    def _preempt(self, req: Request) -> None:
        """重算式抢占：释放全部块，回到等待队列最前面；下次调度时从头计算（提示词 + 已生成的 token）。"""
        self.pool.free(req.block_ids)
        req.block_ids, req.num_computed = [], 0
        req.status = Status.WAITING
        req.num_preemptions += 1
        self.waiting.appendleft(req)

    def _lookup_prefix_cache(self, req: Request) -> None:
        if getattr(self.pool, "enable_caching", False) and req.num_computed == 0:
            # 至少留最后一个 token 不命中：必须真正算一次前向，才能得到下一个 token 的 logits
            hit = self.pool.lookup(req.token_ids[: req.num_tokens - 1])
            self.pool.touch(hit)
            req.block_ids = hit
            req.num_computed = req.num_cached_tokens = len(hit) * self.block_size

    def _release_prefix_hit(self, req: Request) -> None:
        if req.block_ids:
            self.pool.free(req.block_ids)
            req.block_ids, req.num_computed, req.num_cached_tokens = [], 0, 0

    def finish(self, req: Request, reason: str) -> None:
        req.status, req.finish_reason = Status.FINISHED, reason
        self.running.remove(req)
        self.pool.free(req.block_ids)
        req.block_ids = []


def greedy_sample(logits: torch.Tensor, reqs: list[Request]) -> list[int]:
    return logits.argmax(-1).tolist()


class LLMEngine:
    def __init__(self, model, eos_token_id: int | None = None, num_blocks: int = 256, block_size: int = 16,
                 max_num_batched_tokens: int = 256, max_num_seqs: int = 16, enable_chunked_prefill: bool = True,
                 pool: BlockPool | None = None, sample_fn=greedy_sample):
        cfg = model.cfg
        self.block_size, self.eos_token_id, self.sample_fn = block_size, eos_token_id, sample_fn
        self.pool = pool if pool is not None else BlockPool(num_blocks)
        self.kv = PagedKVCache(cfg.num_hidden_layers, self.pool.num_blocks, block_size, cfg.num_key_value_heads, cfg.hd)
        self.runner = ModelRunner(model, self.kv)
        self.scheduler = Scheduler(self.pool, block_size, max_num_batched_tokens, max_num_seqs, enable_chunked_prefill)
        self.step_log: list[dict] = []
        self._next_id = 0

    def add_request(self, prompt_ids: list[int], params: SamplingParams | None = None) -> Request:
        req = Request(str(self._next_id), prompt_ids, params or SamplingParams())
        self._next_id += 1
        self.scheduler.add_request(req)
        return req

    def step(self) -> list[Request]:
        """调度 → 构造批次 → 前向 → 采样 → 更新状态。返回本步产生了新 token 的请求。"""
        out = self.scheduler.schedule()
        if not out.scheduled:
            return []
        items, sampled = [], []
        for req, n in out.scheduled:
            need_logits = req.num_computed + n == req.num_tokens   # 算到了最后一个 token，才需要采样
            items.append((req.token_ids[req.num_computed:req.num_computed + n], req.num_computed, req.block_ids,
                          need_logits))
            if need_logits:
                sampled.append(req)
        logits = self.runner.forward(build_batch(items, self.block_size))
        next_tokens = self.sample_fn(logits, sampled) if sampled else []

        for req, n in out.scheduled:
            req.num_computed += n
            if hasattr(self.pool, "cache_blocks"):
                self.pool.cache_blocks(req.token_ids, req.block_ids, req.num_computed)
        now = time.perf_counter()
        for req, tok in zip(sampled, next_tokens):
            req.output_ids.append(tok)
            req.token_times.append(now)
            p = req.params
            if (tok == self.eos_token_id and not p.ignore_eos) or tok in p.stop_token_ids:
                self.scheduler.finish(req, "stop")
            elif len(req.output_ids) >= p.max_tokens:
                self.scheduler.finish(req, "length")
        self.step_log.append({"num_tokens": sum(n for _, n in out.scheduled), "num_reqs": len(out.scheduled),
                              "num_prefill_tokens": sum(n for r, n in out.scheduled if n > 1),
                              "num_preempted": out.num_preempted, "free_blocks": self.pool.num_free(),
                              "waiting": len(self.scheduler.waiting)})
        return sampled

    def generate(self, prompts: list[list[int]], params=None) -> list[list[int]]:
        """params 可以是一个 SamplingParams（所有请求共用），也可以是每个请求一个的列表。"""
        if not isinstance(params, list):
            params = [params] * len(prompts)
        reqs = [self.add_request(p, sp) for p, sp in zip(prompts, params)]
        while self.scheduler.has_unfinished():
            self.step()
        return [r.output_ids for r in reqs]
