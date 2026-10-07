# Data parallelism and DDP: bucketing and overlapping communication

<p class="lead">Data parallelism is the simplest kind and the most widely used: every card holds a complete model, handles different data, and averages the gradients after the backward pass. The difficulty is not the averaging but keeping the communication from slowing the training down. This chapter writes a DDP from scratch: bucketing the gradients, all-reducing asynchronously during the backward pass, overlapping with the computation, and verifying across 2 processes that it stays step for step with single-process training.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. Each card computes the average loss on its own data and backpropagates. Why is all-reducing the gradients to an average equivalent to training on the whole batch?
    2. Why not synchronise all the gradients in one go once the backward pass has finished?
    3. What problem does bucketing the gradients solve? What goes wrong with buckets that are too large, and too small?
    4. Under gradient accumulation, why do the intermediate micro-steps not need synchronising?
    5. What decides data parallelism's scaling efficiency?

??? success "Answers for the self-test (answer first, then open this)"
    1. The gradient of the whole batch's average loss equals the average of each share's average-loss gradients (when the shares are the same size). Every card's parameters start out the same and are updated each step with the same average gradient, so they stay the same, which is equivalent to training on the whole batch.
    2. The backward pass computes from the last layer forward, so the later layers' gradients were ready long ago; waiting until it has all finished puts the communication entirely on the critical path. Synchronising as you go hides most of the communication behind the backward computation still to come.
    3. Merging many small gradient tensors into larger buckets before the all-reduce cuts the number of collectives and the fixed cost, while letting communication start as soon as a bucket is full. Too large: it takes a long time to fill, there is less chance to overlap, and the last bucket is exposed for longer. Too small: many collectives, and the latency is a large share of each.
    4. The intermediate steps only accumulate gradients locally and only the last step updates the parameters; synchronising the accumulated gradients once at the last step gives exactly the same result as synchronising every step, saving the communication in between (`no_sync`).
    5. The ratio of each card's computation time to the gradient communication time: the communication volume depends only on the parameter count and is fixed, so the larger each card's batch (the more computation), the better the communication is hidden and the higher the efficiency; efficiency drops when there are many cards and each one's batch is small.

A six-panel strip first, then the chapter:

<!-- comic ../assets/comics/ddp.webp is in Chinese; put it back once the English version exists -->

## Why averaging is equivalent {#为什么求平均就等价}

Let the global batch have $B$ samples split evenly across $n$ cards. Card $k$ computes the average loss $L_k$ of its own $B/n$ samples, and the global average loss is $L = \frac{1}{n}\sum_k L_k$. Gradients are linear, so

$$\nabla L = \frac{1}{n}\sum_k \nabla L_k$$

Every card computes its own gradient, the all-reduce sums them and dividing by $n$ gives exactly the gradient of training the whole batch on one card. Each card then applies the same update with the same gradient, so the parameters stay in step (provided they start the same: broadcast once from rank 0 at the beginning).

## Overlapping communication with the backward pass {#通信与反向重叠}

The naive implementation waits for the backward pass to finish and then all-reduces each parameter's gradient. Two problems:

- **The communication is entirely exposed**: the network card only starts work after the backward pass has ended, with the GPU waiting.
- **Too many small messages**: a model has hundreds of parameter tensors, and all-reducing each one separately lets the latency term (the previous chapter's $\alpha$) add up.

What DDP does instead:

1. The backward pass produces gradients roughly in order from the last layer to the first; split the parameters into **buckets** in that order (PyTorch defaults to 25 MB per bucket).
2. Register a hook on each parameter that tells DDP when its gradient is ready; as soon as a bucket's gradients are all in, **fire that bucket's all-reduce asynchronously**.
3. The backward pass carries on with the earlier layers while the network card moves the later layers' gradients.
4. Before the optimizer updates, wait for every bucket's communication to finish.

Most of the communication is then hidden behind the backward computation, and only the last bucket (the first layer's gradients) is exposed.

Dial the layer count, the bucket size and the bandwidth and watch how much communication stays exposed:

<div class="aig-widget" data-widget="ddp-overlap"></div>

## Implementing it from scratch {#从零实现}

```python title="my_ddp.py"
import torch
import torch.distributed as dist


class MyDDP(torch.nn.Module):
    """从零实现的数据并行：梯度按反向的顺序分桶，桶满就异步 all-reduce，与剩下的反向计算重叠。"""

    def __init__(self, module, bucket_bytes=64 * 1024):
        super().__init__()
        self.module = module
        self.world = dist.get_world_size()
        for p in module.parameters():                       # every rank starts from the same initial parameters
            dist.broadcast(p.data, src=0)
        params = [p for p in module.parameters() if p.requires_grad][::-1]   # the backward pass produces gradients in roughly reverse order
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
        if self.ready[i] == len(self.buckets[i]):           # the bucket's gradients are all in: fire the asynchronous all-reduce at once
            flat = torch.cat([q.grad.flatten() for q in self.buckets[i]])
            self.handles.append((dist.all_reduce(flat, async_op=True), flat, i))
            self.launched_in_backward += 1

    def finish_gradient_sync(self):
        for work, flat, i in self.handles:                  # before the optimizer updates, wait for every bucket to finish
            work.wait()
            flat /= self.world
            offset = 0
            for q in self.buckets[i]:
                q.grad.copy_(flat[offset:offset + q.numel()].view_as(q.grad))
                offset += q.numel()
```

- `register_post_accumulate_grad_hook` (from PyTorch 2.1) is called after the gradient has been accumulated into `.grad`, which is exactly the moment this parameter's gradient is ready.
- `async_op=True` has `all_reduce` return a handle immediately with the communication proceeding in the background; `wait()` is what blocks.
- A bucket's gradients are concatenated into one contiguous tensor (`torch.cat`) before the communication, and the concatenation is itself a copy. PyTorch's DDP goes further: it allocates each bucket's gradients in one contiguous buffer from the start (`gradient_as_bucket_view`), which saves that copy.

Training 3 steps across two processes, against both single-process training on the full batch and PyTorch's own DDP:

```python title="ddp_check.py" torchrun="2"
import torch
import torch.distributed as dist

from my_ddp import MyDDP

dist.init_process_group("gloo")
rank, world = dist.get_rank(), dist.get_world_size()


def make_model():
    torch.manual_seed(0)
    return torch.nn.Sequential(torch.nn.Linear(64, 256), torch.nn.GELU(), torch.nn.Linear(256, 256), torch.nn.GELU(),
                               torch.nn.Linear(256, 10))


torch.manual_seed(123)
X, Y = torch.randn(32, 64), torch.randint(0, 10, (32,))   # the global batch: 32 samples
local = slice(rank * 32 // world, (rank + 1) * 32 // world)   # each rank takes its own share

ref = make_model()                                          # the single-process reference: trained on the full batch
ddp = MyDDP(make_model())
official = torch.nn.parallel.DistributedDataParallel(make_model())   # PyTorch's own DDP, as a control
opt_ref = torch.optim.SGD(ref.parameters(), lr=0.1)
opt = torch.optim.SGD(ddp.parameters(), lr=0.1)
opt_off = torch.optim.SGD(official.parameters(), lr=0.1)
loss_fn = torch.nn.CrossEntropyLoss()

for step in range(3):
    opt_ref.zero_grad()
    loss_fn(ref(X), Y).backward()
    opt_ref.step()

    opt.zero_grad()
    loss = loss_fn(ddp(X[local]), Y[local])                # each rank computes only its own share's average loss
    loss.backward()
    ddp.finish_gradient_sync()                              # averaging the gradients: equivalent to averaging over the global batch
    opt.step()

    opt_off.zero_grad()
    loss_fn(official(X[local]), Y[local]).backward()        # the official DDP synchronises inside the backward pass automatically
    opt_off.step()

def max_diff(m):
    return max((a - b).abs().max().item() for a, b in zip(ref.parameters(), m.parameters()))


if rank == 0:
    print(f"{world} 个 rank，{len(ddp.buckets)} 个梯度桶，每步在反向过程中发起 {ddp.launched_in_backward // 3} 次 all-reduce")
    print("训练 3 步后，自己写的 DDP 与单进程一致：", max_diff(ddp.module) < 1e-6)
    print("训练 3 步后，官方 DDP 与单进程一致：", max_diff(official.module) < 1e-6)
dist.destroy_process_group()
```

```text title="output"
2 个 rank，2 个梯度桶，每步在反向过程中发起 2 次 all-reduce
训练 3 步后，自己写的 DDP 与单进程一致： True
训练 3 步后，官方 DDP 与单进程一致： True
```

## The details in practice {#实践中的细节}

**Gradient accumulation**: when memory only holds a very small micro-batch, do several forward and backward passes in a row, accumulating the gradients before updating. The intermediate gradients do not need synchronising; all-reduce the accumulated gradients only on the last backward pass, which divides the communication by the number of accumulation steps. PyTorch's DDP wraps the intermediate backward passes in `with model.no_sync():`.

**The bucket size**: too small and there are many collectives with latency a large share of each; too large and the first bucket takes a long time to fill, there is less chance to overlap, and the communication left exposed at the end is longer. The default 25 MB is an empirical value, often tuned to 50 to 200 MB for large models.

**Reducing the volume**: sending the gradients in bf16 (a compression hook registered through `register_comm_hook`) halves the volume; low-rank compression like PowerSGD is more aggressive but affects convergence and is rarely used in large-model training.

**Scaling efficiency**: the computation per step is proportional to each card's batch, while the communication is a fixed $2\Psi$ or so (the parameters' size in bytes). As cards are added and each one's batch shrinks, the computation shrinks while the communication does not, and sooner or later the communication cannot be hidden. At that point either grow the global batch (limited by convergence) or switch to another kind of parallelism.

**Partitioning the data**: each rank reads different data, usually with `DistributedSampler` splitting the dataset by rank and shuffling with the same seed each epoch, so the ranks neither overlap nor miss anything.

!!! interview "How to answer in an interview"
    On DDP: every card computes the average gradient of its own data and the all-reduce averages them, which is equivalent to training on the whole batch (with the initial parameters broadcast beforehand). Not waiting for the backward pass to end is about overlap: DDP buckets the gradients in backward order, all-reduces a bucket asynchronously as soon as it is full, in parallel with the rest of the backward pass, and only one bucket's communication is left exposed; too small a bucket means many collectives with a large latency share, too large means less chance to overlap. Gradient accumulation's intermediate steps need no synchronisation (`no_sync`). Scaling efficiency comes down to the ratio of each card's computation to the fixed gradient communication.

## Exercises {#练习}

1. Add gradient accumulation to `MyDDP`: a backward pass inside `with ddp.no_sync():` only accumulates gradients without starting any communication, and the next backward pass after leaving it synchronises normally. Verify that accumulating 2 steps of half the data each matches one step on all of it.

??? success "The key points"
    Check a `self.sync_enabled` flag at the start of `_on_grad_ready`, and have `no_sync()` use `contextlib.contextmanager` to set it to `False` on entry and restore it on exit.
    Mind the loss scaling: each micro-step's loss has to be divided by the number of accumulation steps (or the gradients divided at the end) to match the average loss of one step on all of the data.
    PyTorch's DDP also skips the bucket preparation under `no_sync`; the equivalent under FSDP is more complicated, because the parameters themselves are partitioned (the next chapter).

2. A 7B model (about 14 GB of bf16 gradients) does data parallelism on 64 cards at 50 GB/s per card between nodes, with 0.8 seconds of backward computation per step. Can the gradient synchronisation be hidden entirely?

??? success "Answer"
    A ring all-reduce sends about $2 \times 14 = 28$ GB per card, which at 50 GB/s takes about 0.56 seconds, less than the backward pass's 0.8, so in theory it can be hidden. But the last bucket's communication is necessarily exposed, and real network utilisation does not reach 100%, so there is not much headroom.
    This is also why large-model training always pairs data parallelism with ZeRO (the same communication per step, but the memory saved allows a larger batch, which lengthens the computation) and with tensor parallelism inside a node (which reduces the data-parallel degree across nodes).

## Summary {#小结}

- [x] Data parallelism: each card computes the gradient of its own data and the all-reduce averages them, which is equivalent to training on the whole batch; broadcast the initial parameters beforehand.
- [x] DDP buckets the gradients in backward order and all-reduces a bucket asynchronously as soon as it is full, overlapping with the rest of the backward pass; only the last bucket's communication is exposed.
- [x] Gradient accumulation synchronises only on the last step; the bucket size trades latency against the chance to overlap.
- [x] Scaling efficiency comes down to the ratio of each card's computation to the fixed gradient communication.
