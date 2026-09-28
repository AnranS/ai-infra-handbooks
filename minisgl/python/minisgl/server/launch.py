from __future__ import annotations

import logging
import multiprocessing as mp
import sys
from dataclasses import replace
from typing import TYPE_CHECKING

from minisgl.distributed import DistributedInfo
from minisgl.utils import init_logger

if TYPE_CHECKING:
    from .args import ServerArgs


def _run_scheduler(args: ServerArgs, ack_queue: mp.Queue) -> None:
    import torch
    from minisgl.scheduler import Scheduler

    with torch.inference_mode():
        scheduler = Scheduler(args)
        scheduler.sync_all_ranks()  # 所有 rank 都初始化完（包括 ZMQ 订阅）再开始
        if args.tp_info.is_primary():
            ack_queue.put("Scheduler is ready")
        if args.silent_output:
            logging.disable(logging.INFO)
        try:
            scheduler.run_forever()
        except KeyboardInterrupt:
            scheduler.shutdown()


def start_backend(server_args: ServerArgs) -> None:
    """启动所有后端进程：每个 TP rank 一个调度器进程，一个 detokenizer，若干 tokenizer。"""
    from minisgl.tokenizer import tokenize_worker

    logger = init_logger(__name__, "initializer")
    mp.set_start_method("spawn", force=True)  # CUDA 不支持 fork 出来的子进程
    ack_queue: mp.Queue = mp.Queue()
    world_size = server_args.tp_info.size
    for i in range(world_size):
        mp.Process(target=_run_scheduler,
                   args=(replace(server_args, tp_info=DistributedInfo(i, world_size)), ack_queue),
                   name=f"minisgl-TP{i}-scheduler").start()
    common = dict(tokenizer_path=server_args.model_path, backend_addr=server_args.zmq_backend_addr,
                  frontend_addr=server_args.zmq_frontend_addr, local_bs=1, ack_queue=ack_queue)
    n = server_args.num_tokenizer
    mp.Process(target=tokenize_worker, name="minisgl-detokenizer",
               kwargs=dict(common, addr=server_args.zmq_detokenizer_addr,
                           create=server_args.tokenizer_create_addr, tokenizer_id=n)).start()
    for i in range(n):
        mp.Process(target=tokenize_worker, name=f"minisgl-tokenizer-{i}",
                   kwargs=dict(common, addr=server_args.zmq_tokenizer_addr,
                               create=server_args.tokenizer_create_addr, tokenizer_id=i)).start()
    for _ in range(n + 2):  # rank 0 调度器 + n 个 tokenizer + 1 个 detokenizer
        logger.info(ack_queue.get())


def launch_server(run_shell: bool = False) -> None:
    from .api_server import run_api_server
    from .args import parse_args

    server_args, run_shell = parse_args(sys.argv[1:], run_shell)
    run_api_server(server_args, lambda: start_backend(server_args), run_shell=run_shell)
