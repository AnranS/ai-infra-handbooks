import math
from collections import deque
from dataclasses import dataclass
from enum import Enum


class Status(Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"


class Request:
    def __init__(self, request_id, prompt_ids, max_tokens):
        self.request_id, self.prompt_ids, self.max_tokens = request_id, list(prompt_ids), max_tokens
        self.output_ids = []
        self.num_computed = 0
        self.block_ids = []
        self.status = Status.WAITING
        self.num_preemptions = 0

    @property
    def num_tokens(self):
        return len(self.prompt_ids) + len(self.output_ids)


class BlockPool:
    def __init__(self, num_blocks):
        self.num_blocks = num_blocks
        self.free_queue = deque(range(num_blocks))

    def num_free(self):
        return len(self.free_queue)

    def allocate(self, n):
        if n > len(self.free_queue):
            return None
        return [self.free_queue.popleft() for _ in range(n)]

    def free(self, blocks):
        self.free_queue.extend(blocks)


@dataclass
class SchedulerOutput:
    scheduled: list
    num_preempted: int


class Scheduler:
    def __init__(self, pool, block_size, max_num_batched_tokens=256, max_num_seqs=16, enable_chunked_prefill=True):
        self.pool, self.block_size = pool, block_size
        self.max_num_batched_tokens, self.max_num_seqs = max_num_batched_tokens, max_num_seqs
        self.enable_chunked_prefill = enable_chunked_prefill
        self.waiting = deque()
        self.running = []

    def add_request(self, req):
        self.waiting.append(req)

    def has_unfinished(self):
        return bool(self.waiting or self.running)

    def schedule(self) -> SchedulerOutput:
        budget = self.max_num_batched_tokens
        scheduled = []
        num_preempted = 0
        i = 0
        while i < len(self.running) and budget > 0:
            req = self.running[i]
            n = min(req.num_tokens - req.num_computed, budget)
            while not self._allocate(req, req.num_computed + n):
                victim = self.running.pop()
                self._preempt(victim)
                num_preempted += 1
                if victim is req:
                    break
            else:
                scheduled.append((req, n))
                budget -= n
                i += 1
                continue
            break
        while self.waiting and budget > 0 and not num_preempted and len(self.running) < self.max_num_seqs:
            req = self.waiting[0]
            n = req.num_tokens - req.num_computed
            if not self.enable_chunked_prefill and n > budget:
                break
            n = min(n, budget)
            if not self._allocate(req, req.num_computed + n):
                break
            self.waiting.popleft()
            req.status = Status.RUNNING
            self.running.append(req)
            scheduled.append((req, n))
            budget -= n
        return SchedulerOutput(scheduled, num_preempted)

    def _allocate(self, req, num_tokens):
        need = math.ceil(num_tokens / self.block_size) - len(req.block_ids)
        if need <= 0:
            return True
        blocks = self.pool.allocate(need)
        if blocks is None:
            return False
        req.block_ids += blocks
        return True

    def _preempt(self, req):
        self.pool.free(req.block_ids)
        req.block_ids, req.num_computed = [], 0
        req.status = Status.WAITING
        req.num_preemptions += 1
        self.waiting.appendleft(req)

    def finish(self, req):
        req.status = Status.FINISHED
        self.running.remove(req)
        self.pool.free(req.block_ids)
        req.block_ids = []
