from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from minisgl.core import SamplingParams
from minisgl.distributed import DistributedInfo
from minisgl.message import BaseBackendMsg, DetokenizeMsg, UserMsg
from minisgl.scheduler import Scheduler, SchedulerConfig


class RequestAllFinished(Exception):
    pass


@dataclass
class RequestStatus:
    uid: int
    input_ids: List[int]
    output_ids: List[int]


class LLM(Scheduler):
    """离线推理接口：不起任何进程，直接复用调度器的主循环。

    它覆盖了调度器的两个 IO 方法：receive_msg 从待处理列表里取请求（按 prefill 预算分批放入），
    send_result 把生成的 token 记下来。所有请求处理完、调度器又要阻塞等待新消息时，
    抛出 RequestAllFinished 跳出 run_forever。
    """

    def __init__(self, model_path: str, dtype: torch.dtype = torch.bfloat16, **kwargs):
        config = SchedulerConfig(model_path=model_path, tp_info=DistributedInfo(0, 1),
                                 dtype=dtype, offline_mode=True, **kwargs)
        super().__init__(config)
        self.pending_requests: List[Tuple[List[int] | str, SamplingParams]] = []
        self.status_map: Dict[int, RequestStatus] = {}
        self.counter = 0

    def _tokenize_one(self, prompt: List[int] | str) -> torch.Tensor:
        if isinstance(prompt, str):
            return self.tokenizer.encode(prompt, return_tensors="pt").view(-1).to(torch.int32)
        return torch.tensor(prompt, dtype=torch.int32)

    def offline_receive_msg(self, blocking: bool = False) -> List[BaseBackendMsg]:
        if blocking and not self.pending_requests:
            raise RequestAllFinished()
        results: List[BaseBackendMsg] = []
        added = total_len = 0
        for prompt, sp in self.pending_requests:
            if total_len >= self.prefill_budget:
                break
            input_ids = self._tokenize_one(prompt)
            total_len += len(input_ids)
            uid = self.counter + added
            added += 1
            results.append(UserMsg(uid=uid, input_ids=input_ids, sampling_params=sp))
            self.status_map[uid] = RequestStatus(uid, input_ids.tolist(), [])
        self.counter += added
        self.pending_requests = self.pending_requests[added:]
        return results

    def offline_send_result(self, reply: List[DetokenizeMsg]) -> None:
        for msg in reply:
            status = self.status_map[msg.uid]
            if not (msg.finished and msg.next_token == self.eos_token_id):
                status.output_ids.append(msg.next_token)

    def generate(self, prompts: List[str] | List[List[int]],
                 sampling_params: List[SamplingParams] | SamplingParams
                 ) -> List[Dict[str, str | List[int]]]:
        self.pending_requests, self.status_map, self.counter = [], {}, 0
        if isinstance(sampling_params, SamplingParams):
            sampling_params = [sampling_params] * len(prompts)
        # 每次都复制一份：调度器可能会修改 max_tokens
        self.pending_requests = [(p, SamplingParams(**vars(sp)))
                                 for p, sp in zip(prompts, sampling_params)]
        try:
            self.run_forever()
        except RequestAllFinished:
            pass
        return [{"text": self.tokenizer.decode(self.status_map[i].output_ids),
                 "token_ids": self.status_map[i].output_ids} for i in range(len(prompts))]
