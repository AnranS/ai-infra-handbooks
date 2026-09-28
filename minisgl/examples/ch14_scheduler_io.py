"""第 14 章：两个"调度器 rank"怎样收到完全相同的消息序列。

不加载模型，只用调度器的 IO 部分：主进程扮演 tokenizer 发 3 条消息，
rank 0 从 ZMQ 收到后广播"本轮条数"并转发原始字节，rank 1 按条数从 PUB/SUB 取。
"""

import multiprocessing as mp
import time

import torch
import torch.distributed as dist
from minisgl.core import SamplingParams
from minisgl.distributed import DistributedInfo
from minisgl.message import BaseBackendMsg, UserMsg
from minisgl.scheduler.config import SchedulerConfig
from minisgl.scheduler.io import SchedulerIOMixin
from minisgl.utils import ZmqPushQueue


class IOOnly(SchedulerIOMixin):
    def run_when_idle(self) -> None:
        pass


def rank_main(rank: int, suffix: str, out: mp.Queue) -> None:
    dist.init_process_group("gloo", init_method="tcp://127.0.0.1:29555", rank=rank, world_size=2)
    config = SchedulerConfig(model_path="unused", tp_info=DistributedInfo(rank, 2),
                             dtype=torch.float32, _unique_suffix=suffix)
    io = IOOnly(config, dist.group.WORLD)
    dist.barrier()  # 对应 sync_all_ranks：订阅者都连上之后才开始
    out.put(("ready", rank))
    got = []
    while len(got) < 3:
        got += [m.uid for m in io.receive_msg(blocking=not got)]
    out.put((rank, got))
    dist.barrier()


if __name__ == "__main__":
    mp.set_start_method("spawn")
    suffix = f".example{int(time.time())}"
    out: mp.Queue = mp.Queue()
    procs = [mp.Process(target=rank_main, args=(r, suffix, out)) for r in range(2)]
    for p in procs:
        p.start()
    for _ in range(2):
        out.get()
    config = SchedulerConfig(model_path="unused", tp_info=DistributedInfo(0, 2), dtype=torch.float32,
                             _unique_suffix=suffix)
    push = ZmqPushQueue(config.zmq_backend_addr, create=False, encoder=BaseBackendMsg.encoder)
    for uid in range(3):
        push.put(UserMsg(uid=uid, input_ids=torch.tensor([1, 2], dtype=torch.int32),
                         sampling_params=SamplingParams()))
    results = dict(out.get() for _ in range(2))
    for rank in (0, 1):
        print(f"rank {rank} 收到的 uid 序列: {results[rank]}")
    for p in procs:
        p.join()
    push.stop()
