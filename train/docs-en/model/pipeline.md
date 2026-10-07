# Pipeline parallelism: from GPipe to zero bubble

<p class="lead">Pipeline parallelism cuts the model into stages by depth, each card taking several consecutive layers, with the activations passed point-to-point between neighbouring cards. Its communication volume is small and it can cross nodes, but it has an inherent problem: while the pipeline fills and drains, some card is always waiting, and that is the bubble. This chapter first draws GPipe's and 1F1B's timelines with a scheduling simulator, then really runs 1F1B across several processes and lines its gradients up against a single process, and finally covers how interleaving, zero bubble and DualPipe go on squeezing the bubble.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. Why does a pipeline cut a batch into several micro-batches?
    2. Are GPipe's and 1F1B's bubbles the same size? Where do they differ?
    3. Under a 1F1B schedule, how many micro-batches' activations does the first stage hold at most?
    4. What does interleaved 1F1B (the virtual pipeline) trade for a smaller bubble?
    5. Which two parts does zero-bubble scheduling split the backward pass into? Why does that fill the bubble?

??? success "Answers for the self-test (answer first, then open this)"
    1. With only one micro-batch, exactly one stage works at any moment and the utilization is $1/p$; cut into several micro-batches, the next one enters the first stage as soon as the previous one leaves, several stages work at once, and only the bubble at the start and the end is left.
    2. The bubbles are the same size, $(p-1)/(m+p-1)$. The difference is activation memory: GPipe does all of the forward passes first, so each stage holds $m$ copies of activations at once, while 1F1B alternates one forward and one backward after a warm-up and releases each as soon as its backward pass is done.
    3. At most $p$ copies (the stage count), independently of the micro-batch count $m$.
    4. Each card takes several non-adjacent segments of layers (virtual stages), which lengthens the pipeline and shortens each segment's computation, shrinking the bubble to $1/v$ of what it was; the price is $v$ times the point-to-point communication.
    5. Into the gradient with respect to the input (B, which the next stage is waiting for and has to be computed promptly) and the gradient with respect to the weights (W, which nobody is waiting for and can be deferred). Moving W into what would otherwise be idle bubble fills it in.

A six-panel strip first, then the chapter:

<!-- comic ../assets/comics/pipeline.webp is in Chinese; put it back once the English version exists -->

## Where the bubble comes from {#气泡从哪里来}

With one micro-batch, the $p$ stages run one after another, exactly one card works at any moment and the utilization is $1/p$. Cut the batch into $m$ micro-batches and the next one can enter stage 0 once the previous one has left, so several stages work at once. But the start has to wait for the pipeline to fill and the end for it to drain, and those two periods are the **bubble**.

![Figure: the GPipe and 1F1B schedules (the backward pass drawn as twice the forward pass's duration)](../assets/figures/pipeline-1f1b.svg){.aig-svg}

The simulator below lays out each stage's timeline by the dependencies (stage $s$'s forward pass waits for stage $s-1$'s, and its backward pass waits for stage $s+1$'s). A digit is the forward pass of that micro-batch (taking 1), a letter is its backward pass (taking 2), and `.` is idle:

```python title="pp_sim.py"
def schedule(kind, p, m):
    """每个 stage 按什么顺序执行前向 (F, i) 和反向 (B, i)"""
    orders = []
    for s in range(p):
        if kind == "gpipe":
            orders.append([("F", i) for i in range(m)] + [("B", i) for i in range(m)])
        else:                                   # 1F1B: p-s-1 forward passes as a warm-up, then alternating one forward and one backward
            warm = min(p - s - 1, m)
            order = [("F", i) for i in range(warm)]
            for i in range(m - warm):
                order += [("F", warm + i), ("B", i)]
            order += [("B", i) for i in range(m - warm, m)]
            orders.append(order)
    return orders


def simulate(kind, p, m, tf=1, tb=2):
    orders, done, t_free = schedule(kind, p, m), {}, [0] * p
    pos, rows = [0] * p, [[] for _ in range(p)]
    live, peak = [0] * p, [0] * p
    while any(pos[s] < len(orders[s]) for s in range(p)):
        for s in range(p):
            if pos[s] == len(orders[s]):
                continue
            op, i = orders[s][pos[s]]
            dep = ("F", s - 1, i) if op == "F" else (("B", s + 1, i) if s < p - 1 else ("F", s, i))
            if op == "F" and s == 0:
                dep = None
            if dep is not None and dep not in done:
                continue
            start = max(t_free[s], done.get(dep, 0))
            end = start + (tf if op == "F" else tb)
            rows[s] += ["."] * (start - len(rows[s])) + [str(i) if op == "F" else chr(ord("a") + i)] * (end - start)
            done[(op, s, i)], t_free[s] = end, end
            live[s] += 1 if op == "F" else -1
            peak[s] = max(peak[s], live[s])
            pos[s] += 1
    total = max(t_free)
    busy = p * m * (tf + tb)
    return rows, total, 1 - busy / (p * total), peak


for kind in ("gpipe", "1f1b"):
    rows, total, bubble, peak = simulate(kind, p=4, m=8)
    print(f"{kind}：总时间 {total}，气泡占比 {bubble:.1%}，每个 stage 同时保存的激活份数 {peak}")
    for s, r in enumerate(rows):
        print(f"  stage {s} |{''.join(r).ljust(total, '.')}|")
print("理论气泡占比 (p-1)/(m+p-1) =", f"{3 / 11:.1%}")
```

```text title="output"
gpipe：总时间 33，气泡占比 27.3%，每个 stage 同时保存的激活份数 [8, 8, 8, 8]
  stage 0 |01234567.........aabbccddeeffgghh|
  stage 1 |.01234567......aabbccddeeffgghh..|
  stage 2 |..01234567...aabbccddeeffgghh....|
  stage 3 |...01234567aabbccddeeffgghh......|
1f1b：总时间 33，气泡占比 27.3%，每个 stage 同时保存的激活份数 [4, 3, 2, 1]
  stage 0 |0123......aa4bb5cc6dd7ee.ff.gg.hh|
  stage 1 |.012....aa3bb4cc5dd6ee7ff.gg.hh..|
  stage 2 |..01..aa2bb3cc4dd5ee6ff7gg.hh....|
  stage 3 |...0aa1bb2cc3dd4ee5ff6gg7hh......|
理论气泡占比 (p-1)/(m+p-1) = 27.3%
```

Two conclusions:

- **GPipe's and 1F1B's bubbles are the same size**, $(p-1)/(m+p-1)$. The direct way to shrink the bubble is a larger $m$, but $m$ is limited by the global batch.
- **The difference is memory**: GPipe does all of the forward passes first, so each stage holds $m$ micro-batches' activations at once; 1F1B does one forward and then one backward after the warm-up, and a micro-batch's activations are released as soon as its backward pass is done, so the first stage holds at most $p$ copies. $m$ can be far larger than $p$, which is why 1F1B is the default schedule in every modern training framework.

Drag p and m yourself and compare the two schedules' timelines, bubbles and activation counts:

<div class="aig-widget" data-widget="pipeline"></div>

## Really running 1F1B {#真跑一遍-1f1b}

Each process is one stage taking two layers. The forward pass's activations are sent to the next stage asynchronously with `isend` and received from the previous one with `recv`; the backward pass passes the gradients the other way. At the end, each stage's parameter gradients are compared with a single process's on the full batch:

```python title="pp_1f1b.py" torchrun="4"
import torch
import torch.distributed as dist

dist.init_process_group("gloo")
stage, p = dist.get_rank(), dist.get_world_size()
M, MB, H = 4, 2, 16                                     # the micro-batch count, the samples per micro-batch, the hidden dimension


def make_layers():
    torch.manual_seed(0)
    return [torch.nn.Sequential(torch.nn.Linear(H, H), torch.nn.Tanh()) for _ in range(2 * p)]


torch.manual_seed(1)
X, Y = torch.randn(M * MB, H), torch.randn(M * MB, H)
layers = make_layers()

# the single-process reference: the whole model and the whole batch (the loss is the batch's average, which equals the average of the micro-batches' average losses)
ref = torch.nn.Sequential(*make_layers())
torch.nn.functional.mse_loss(ref(X), Y).backward()
ref_grads = [q.grad for q in ref.parameters()]

# the pipeline: stage s takes layers 2s and 2s+1
mine = torch.nn.Sequential(*layers[2 * stage:2 * stage + 2])
first, last = stage == 0, stage == p - 1
inputs, outputs, trace, pending = {}, {}, [], []


def forward(i):
    if first:
        x = X[i * MB:(i + 1) * MB]
    else:
        x = torch.empty(MB, H)
        dist.recv(x, stage - 1)                          # receive the previous stage's activations
        x.requires_grad_()
    y = mine(x)
    inputs[i], outputs[i] = x, y
    if not last:
        pending.append(dist.isend(y.detach(), stage + 1))   # send to the next stage asynchronously, without blocking
    trace.append(f"F{i}")


def backward(i):
    y = outputs.pop(i)
    if last:
        (torch.nn.functional.mse_loss(y, Y[i * MB:(i + 1) * MB]) / M).backward()
    else:
        g = torch.empty(MB, H)
        dist.recv(g, stage + 1)                          # receive the gradient the next stage passes back
        y.backward(g)
    x = inputs.pop(i)
    if not first:
        pending.append(dist.isend(x.grad, stage - 1))    # pass the gradient with respect to the input back to the previous stage
    trace.append(f"B{i}")


warm = min(p - stage - 1, M)                             # 1F1B: a few forward passes as a warm-up, then alternating one forward and one backward
for i in range(warm):
    forward(i)
for i in range(M - warm):
    forward(warm + i)
    backward(i)
for i in range(M - warm, M):
    backward(i)
for w in pending:
    w.wait()

ok = all(torch.allclose(q.grad, r, atol=1e-6) for q, r in zip(mine.parameters(), ref_grads[4 * stage:4 * stage + 4]))
traces = [None] * p
dist.all_gather_object(traces, " ".join(trace))
flags = torch.tensor([int(ok)])
dist.all_reduce(flags, op=dist.ReduceOp.MIN)
if stage == 0:
    for s, t in enumerate(traces):
        print(f"stage {s} 的执行顺序：{t}")
    print("各 stage 的参数梯度与单进程一致：", bool(flags.item()))
dist.destroy_process_group()
```

```text title="output"
stage 0 的执行顺序：F0 F1 F2 F3 B0 B1 B2 B3
stage 1 的执行顺序：F0 F1 F2 B0 F3 B1 B2 B3
stage 2 的执行顺序：F0 F1 B0 F2 B1 F3 B2 B3
stage 3 的执行顺序：F0 B0 F1 B1 F2 B2 F3 B3
各 stage 的参数梯度与单进程一致： True
```

A few implementation points:

- Sending uses the **asynchronous** `isend`: with a blocking `send`, stage 0 waiting for stage 1 to receive the second activation could meet stage 1 waiting for stage 0 to receive the first gradient, and the two would deadlock waiting for each other.
- The last stage's loss is divided by the micro-batch count $M$, so that accumulating each micro-batch's gradients equals the gradient of the whole batch's average loss.
- Here $M = p = 4$, so stage 0's warm-up happens to complete all 4 forward passes and its order matches GPipe's; with a larger $M$ a steady alternating phase appears in the middle.
- In a real framework each stage also has tensor and data parallelism inside it, and the activations' shapes have to be agreed between stages beforehand (or sent once first).

## Squeezing the bubble further {#进一步压缩气泡}

**Interleaved 1F1B** (Megatron's virtual pipeline): instead of one consecutive run of layers, each card takes $v$ non-consecutive segments (with 4 cards and 16 layers, card 0 takes layers 0, 4, 8 and 12). A micro-batch goes round the cards $v$ times, each segment's computation takes $1/v$ of what it did, filling and draining are $v$ times faster, and the bubble becomes $\frac{p-1}{v\,m + p - 1}$. The price is $v$ times the point-to-point communication.

**Zero bubble**: a backward pass actually contains two parts, the gradient with respect to the input $\partial L/\partial x$ (B, which the previous stage is waiting for) and the gradient with respect to the weights $\partial L/\partial W$ (W, which nobody is waiting for and only has to be done before the optimizer update). Splitting W out and deferring it, and using it to fill what would otherwise be idle bubble, can in theory squeeze the bubble to nearly zero.

**DualPipe** (DeepSeek-V3): micro-batches enter from both ends of the pipeline at once (a bidirectional pipeline), and the computation and communication within each chunk, especially expert parallelism's all-to-all, are carefully overlapped. It targets the specific case where expert parallelism across nodes communicates heavily, at the price of each card holding two copies of the parameters.

!!! interview "How to answer in an interview"
    On pipelines: cutting into $m$ micro-batches is what lets $p$ stages work at once, and the bubble from filling and draining is $(p-1)/(m+p-1)$; GPipe's and 1F1B's bubbles are the same size and the difference is memory, since 1F1B has the first stage hold at most $p$ copies of activations against GPipe's $m$, which is why it is the default. Squeezing further: interleaving shrinks the bubble by $v$ at $v$ times the point-to-point communication, zero bubble splits the backward pass into the input gradient and the weight gradient and fills the bubble with the latter, and DualPipe runs the pipeline in both directions while overlapping computation and communication. In the implementation, sending has to be asynchronous so that the stages do not deadlock waiting for each other.

## Exercises {#练习}

1. With this chapter's simulator, fix $p = 4$ and take $m$ as 4, 8, 16 and 32, recording 1F1B's bubble fraction and comparing it against the formula $(p-1)/(m+p-1)$.

??? success "Answer"
    The bubble fractions are about 42.9%, 27.3%, 15.8% and 8.6%, matching the formula. Doubling $m$ roughly halves the bubble; but the larger $m$ is, the smaller each micro-batch (with the global batch fixed) and the lower the matrix multiplies' efficiency, so in practice $m$ is usually 4 to 8 times $p$.

2. Pipeline parallelism communicates far less than tensor parallelism, so why is it rarely used on its own?

??? success "Answer"
    A pipeline passes each micro-batch's activations between neighbouring stages once (and the gradients once), about $sbh$ elements, far less than tensor parallelism's four communications per layer; and it can cross nodes.
    But it has a bubble, which has to be amortised over many micro-batches, and the micro-batch count is limited by the global batch; the stages also have to be load-balanced (the embedding and output layers make the first and last stages heavier). So it is usually combined with tensor parallelism (within a node) and data parallelism (the outermost), to cut by depth a model that will not fit in one tensor-parallel group.

## Summary {#小结}

- [x] A pipeline cuts the batch into micro-batches so that several stages work at once; filling and draining create a bubble of $(p-1)/(m+p-1)$.
- [x] GPipe and 1F1B have the same bubble, and 1F1B takes each stage's concurrent activations from $m$ copies down to at most $p$, which makes it the default schedule.
- [x] Interleaving shrinks the bubble by $v$ at $v$ times the point-to-point communication; zero bubble moves the weight gradient into the bubble; DualPipe runs the pipeline in both directions while overlapping computation and communication.
- [x] In the implementation, sending has to be asynchronous so that stages do not deadlock waiting for each other, and the loss is scaled by the micro-batch count.
