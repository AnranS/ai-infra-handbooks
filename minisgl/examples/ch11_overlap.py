"""第 11 章：重叠调度的事件顺序，以及按官方逻辑判断结束时出现的两个问题。"""

from typing import List, Set

import torch
from minisgl.core import Req, SamplingParams
from minisgl.env import ENV
from minisgl.llm import LLM
from minisgl.message import DetokenizeMsg
from minisgl.scheduler.prefill import ChunkedReq

PATH = "models/Qwen3-0.6B"
KW = dict(dtype=torch.float32, max_running_req=8, num_page_override=512, max_seq_len_override=128)


def trace_events(overlap: bool) -> None:
    ENV.DISABLE_OVERLAP_SCHEDULING.value = not overlap
    llm = LLM(PATH, **KW)
    step = {"n": 0}
    orig_forward, orig_process = llm._forward, llm._process_last_data

    def forward(fi):
        step["n"] += 1
        fi.batch.step = step["n"]
        print(f"    发射第 {step['n']} 轮（{fi.batch.phase}）")
        return orig_forward(fi)

    def process(data):
        if data is not None:
            print(f"    处理第 {data[0].batch.step} 轮的结果")
        return orig_process(data)

    llm._forward, llm._process_last_data = forward, process
    llm.generate(["The capital of France is"], SamplingParams(max_tokens=3, ignore_eos=True))
    llm.shutdown()


print("普通循环：")
trace_events(overlap=False)
print("重叠循环：")
trace_events(overlap=True)


class UpstreamLogicLLM(LLM):
    """把 _process_last_data 换成官方的判断方式（其余不变），用来复现问题。"""

    def _process_last_data(self, last_data) -> None:
        if last_data is None:
            return
        batch, (_, next_tokens_cpu, copy_done) = last_data[0].batch, last_data[1]
        copy_done.synchronize()
        reply: List[DetokenizeMsg] = []
        new_finished: Set[Req] = set()
        with self.cache_manager.lazy_free_region():
            for i, req in enumerate(batch.reqs):
                if isinstance(req, ChunkedReq):
                    continue
                req.append_host(next_tokens_cpu[i].unsqueeze(0))
                token = int(next_tokens_cpu[i].item())
                finished = not req.can_decode  # 官方：按 can_decode 判断
                if not req.sampling_params.ignore_eos:
                    finished |= token == self.eos_token_id
                reply.append(DetokenizeMsg(uid=req.uid, next_token=token, finished=finished))
                if finished and req not in self.finished_reqs:  # 官方：只防重复释放
                    self.decode_manager.remove_req(req)
                    self._free_req_resources(req)
                    new_finished.add(req)
                elif batch.is_prefill:
                    self.cache_manager.cache_req(req, finished=False)
        self.finished_reqs = new_finished
        self.send_result(reply)


def replies_of(cls, prompt: str, params: SamplingParams, eos: int | None = None):
    ENV.DISABLE_OVERLAP_SCHEDULING.value = False
    llm = cls(PATH, **KW)
    if eos is not None:
        llm.eos_token_id = eos
    replies = []
    orig = llm.offline_send_result
    llm.send_result = lambda reply: (replies.extend(reply), orig(reply))
    llm.generate([prompt], params)
    llm.shutdown()
    return [(m.next_token, m.finished) for m in replies]


print("\n问题一：max_tokens=3，发给 detokenizer 的 (token, finished)：")
for cls in (UpstreamLogicLLM, LLM):
    print(f"  {cls.__name__:<17}", replies_of(cls, "The capital of France is",
                                          SamplingParams(max_tokens=3, ignore_eos=True)))
print("问题二：把 eos 设成第 3 个输出 token（576），max_tokens=8：")
for cls in (UpstreamLogicLLM, LLM):
    print(f"  {cls.__name__:<17}", replies_of(cls, "The capital of France is",
                                          SamplingParams(max_tokens=8), eos=576))
