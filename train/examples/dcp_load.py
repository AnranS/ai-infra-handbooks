import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard

dist.init_process_group("gloo")
rank, world = dist.get_rank(), dist.get_world_size()


def make_model(seed):
    torch.manual_seed(seed)
    return torch.nn.Sequential(torch.nn.Linear(64, 256), torch.nn.GELU(), torch.nn.Linear(256, 64))


ref = make_model(0)                                             # 保存时的模型
model = make_model(1)                                           # 故意用不同的初始化：参数要靠加载
fully_shard(model, mesh=init_device_mesh("cpu", (world,)))      # 换成 2 路切分
state = {"model": model.state_dict()}
dcp.load(state, checkpoint_id="ckpt")                           # 按元数据找到每一片需要的那部分，原地读入
model.load_state_dict(state["model"])

same = all(torch.equal(p.full_tensor(), q) for p, q in zip(model.parameters(), ref.parameters()))
if rank == 0:
    print(f"{world} 个 rank 加载，第一层权重每片 {tuple(model[0].weight.to_local().shape)}，与原模型一致：{same}")
    dcp_to_torch_save("ckpt", "full.pt")                        # 离线合并成一个普通的 state_dict，便于转换成推理格式
    full = torch.load("full.pt")["model"]
    print("合并后的完整权重：", {k: tuple(v.shape) for k, v in full.items()})
dist.destroy_process_group()
