import torch
import torch.distributed as dist


class MyDDP(torch.nn.Module):
    """从零实现的数据并行：梯度按反向的顺序分桶，桶满就异步 all-reduce，与剩下的反向计算重叠。"""

    def __init__(self, module, bucket_bytes=64 * 1024):
        super().__init__()
        self.module = module
        self.world = dist.get_world_size()
        for p in module.parameters():                       # 所有 rank 从同样的初始参数开始
            dist.broadcast(p.data, src=0)
        params = [p for p in module.parameters() if p.requires_grad][::-1]   # 反向时大致按逆序产生梯度
        self.buckets, cur, size = [], [], 0
        for p in params:
            cur.append(p)
            size += p.numel() * p.element_size()
            if size >= bucket_bytes:
                self.buckets.append(cur)
                cur, size = [], 0
        if cur:
            self.buckets.append(cur)
        self.bucket_of = {p: i for i, b in enumerate(self.buckets) for p in b}
        self.pending, self.launched_in_backward = {}, 0
        for p in params:
            p.register_post_accumulate_grad_hook(self._on_grad_ready)

    def forward(self, *args):
        self.ready = [0] * len(self.buckets)
        self.handles = []
        return self.module(*args)

    def _on_grad_ready(self, p):
        i = self.bucket_of[p]
        self.ready[i] += 1
        if self.ready[i] == len(self.buckets[i]):           # 桶里的梯度都到齐了：立刻发起异步 all-reduce
            flat = torch.cat([q.grad.flatten() for q in self.buckets[i]])
            self.handles.append((dist.all_reduce(flat, async_op=True), flat, i))
            self.launched_in_backward += 1

    def finish_gradient_sync(self):
        for work, flat, i in self.handles:                  # 优化器更新之前，等所有桶完成
            work.wait()
            flat /= self.world
            offset = 0
            for q in self.buckets[i]:
                q.grad.copy_(flat[offset:offset + q.numel()].view_as(q.grad))
                offset += q.numel()
