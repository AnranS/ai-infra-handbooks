import os

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard

dist.init_process_group("gloo")
rank, world = dist.get_rank(), dist.get_world_size()


def make_model():
    torch.manual_seed(0)
    return torch.nn.Sequential(torch.nn.Linear(64, 256), torch.nn.GELU(), torch.nn.Linear(256, 64))


model = make_model()
fully_shard(model, mesh=init_device_mesh("cpu", (world,)))      # 4 路切分：每个参数按第 0 维切成 4 片
dcp.save({"model": model.state_dict()}, checkpoint_id="ckpt")   # 每个 rank 只写自己的分片，外加一份元数据

if rank == 0:
    print(f"{world} 个 rank 保存，第一层权重每片 {tuple(model[0].weight.to_local().shape)}")
    print("checkpoint 目录：", sorted(os.listdir("ckpt")))
dist.destroy_process_group()
