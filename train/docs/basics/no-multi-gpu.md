# 没有多卡怎么练：在 CPU 上跑真的多进程

<p class="lead">本书后面每一章都在讲"多卡怎么切、怎么通信"，但学这些并不需要真的有多卡。多卡里真正需要多卡的只有"性能和运维"那一小块，占大头的设计与正确性全都能在一台笔记本上用 <code>torchrun</code> + gloo 跑真的多进程验证——不是伪造的模拟，是同一套 <code>torch.distributed</code> API、同一套切分逻辑，只是把 NCCL 换成了 gloo。这一章讲清楚怎么跑、能验证到什么程度、哪些必须上真卡，以及租到卡之后照着跑的清单。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. 在 CPU 上用 gloo 跑 4 个进程，哪些集合通信原语是可用的？哪些不可用？
    2. 张量并行的实现写错了，你怎么在没有 GPU 的情况下发现？
    3. 哪些结论必须在真的多卡上才能得到？为什么 CPU 上测不出来？
    4. `torchrun --standalone --nproc-per-node=4` 和手工设置 `MASTER_ADDR` / `RANK` 有什么区别？
    5. 租到 8 张卡只有 4 小时，你会按什么顺序测？

??? success "自测参考答案（先自己答，再展开对照）"
    1. 现在的 PyTorch 里 gloo 已经支持 `all_reduce`、`all_gather`、`all_gather_into_tensor`、`reduce_scatter_tensor`、`all_to_all_single`、`broadcast`、`barrier` 和点对点的 `send` / `recv` / `isend` / `irecv`——本书所有多进程示例用到的都在里面。真正缺的是 GPU 专有能力：NCCL 的通信器、CUDA stream 上的异步通信、GPUDirect / NVLink 直连、`reduce_scatter` 的 GPU 快路径。
    2. 写一个**单进程参考实现**当黄金对照：同一份权重、同一份输入，单进程算一遍完整结果，多进程按切分算一遍，逐元素 `allclose`。切分写错（切错维、忘了 all-reduce、RoPE 位置对不上）几乎一定会在这里暴露。
    3. 实测带宽与理论的差距、NVLink 与 PCIe 的差、拓扑（NUMA、PIX/PXB/SYS）的影响、通信与计算重叠的真实收益、scaling 曲线拐点、一个 rank 挂掉之后的行为。这些依赖真实的链路和 NCCL 实现，CPU 上的 gloo 走的是本机回环，测出来的数字没有意义。
    4. `--standalone` 让 torchrun 自己在本机拉起 rendezvous（随机端口、单机多进程），不用管 `MASTER_ADDR` / `MASTER_PORT` / `RANK` / `WORLD_SIZE`，也不会和别人的端口撞。手工设环境变量只在多机或者要精确控制端口时才需要。
    5. 先跑基线（nccl-tests 各消息大小的总线带宽 + 拓扑），再跑"有对照组的实验"（同一个模型 TP=1/2/4/8 的 scaling、开关通信重叠），最后才是探索性的调参。所有脚本和对照组都在 CPU 上先写好调通，上卡只负责跑和记。

## 先说结论：哪些能在 CPU 上验证

| 要学的东西 | CPU 上能不能练 | 怎么练 |
| --- | --- | --- |
| 集合通信原语的语义 | ✅ 完全可以 | gloo 后端，4 个进程 |
| 环形 all-reduce 的实现与通信量 | ✅ | 点对点 `isend` / `irecv` 手写一遍，统计字节数 |
| DDP 的分桶与通信重叠**逻辑** | ✅ | 看桶怎么划、什么时候触发通信 |
| ZeRO-1/2/3 的切分与 all-gather 时机 | ✅ | 和单进程 Adam 逐元素对齐 |
| 张量并行 / 序列并行的切法 | ✅ | 和单进程前向逐元素对齐 |
| 流水线并行的 1F1B 调度与气泡 | ✅ | 数气泡，和公式对 |
| 上下文并行（Ulysses / Ring Attention） | ✅ | 和单进程注意力逐元素对齐 |
| 专家并行的 all-to-all 与负载不均 | ✅ | 统计每个专家收到多少 token |
| 通信量与时间模型（α-β） | ✅ | 纯算，不用跑 |
| **实测带宽、拓扑、重叠收益、scaling 曲线** | ❌ | 必须真卡 |
| **NCCL 调参、多机 RDMA、故障恢复** | ❌ | 必须真卡（而且要多机） |

一句话：**逻辑和数值正确性在 CPU 上能 100% 验证，性能数字一个都不能信。**

## 怎么跑：torchrun + gloo

```bash
# 4 个进程，单机，torchrun 自己搞定 rendezvous
torchrun --standalone --nproc-per-node=4 your_script.py

# macOS 上还要让 gloo 走回环网卡，否则主机名会解析到外部地址、连不上
GLOO_SOCKET_IFNAME=lo0 torchrun --standalone --nproc-per-node=4 your_script.py
# Linux 上是 lo
```

脚本里只要把后端从 `nccl` 换成 `gloo`，其余一行不用改：

```python title="probe.py" torchrun="4"
import torch
import torch.distributed as dist

dist.init_process_group("gloo")
rank, n = dist.get_rank(), dist.get_world_size()
checks = []


def try_op(name, fn):
    try:
        fn()
        checks.append((name, "可用"))
    except Exception as e:
        checks.append((name, type(e).__name__))


x = torch.ones(4 * n) * (rank + 1)
try_op("all_reduce", lambda: dist.all_reduce(x.clone()))
try_op("all_gather_into_tensor", lambda: dist.all_gather_into_tensor(torch.zeros(4 * n * n), x.clone()))
try_op("reduce_scatter_tensor", lambda: dist.reduce_scatter_tensor(torch.zeros(4), x.clone()))
try_op("all_to_all_single", lambda: dist.all_to_all_single(torch.zeros(4 * n), x.clone()))
try_op("broadcast", lambda: dist.broadcast(x.clone(), 0))
try_op("barrier", dist.barrier)
if rank == 0:
    print(f"后端 {dist.get_backend()}，{n} 个进程")
    for name, ok in checks:
        print(f"  {name:24s} {ok}")
dist.destroy_process_group()
```

```text title="输出"
后端 gloo，4 个进程
  all_reduce               可用
  all_gather_into_tensor   可用
  reduce_scatter_tensor    可用
  all_to_all_single        可用
  broadcast                可用
  barrier                  可用
```

本书所有带 `torchrun` 标记的示例用的就是这些原语，所以它们在任何一台机器上都能直接跑——CI 里也是这样验证的。

!!! note "CPU 上跑多进程要注意两件事"
    - **限制线程数**：`OMP_NUM_THREADS=2`。不限的话每个进程都会按核数开满线程，4 个进程互相抢核，慢到像是卡死。
    - **把规模调小**：CPU 上做的是"验证逻辑"，不是"跑出性能"，隐藏维几十、batch 个位数就够。想验证的是切分对不对，不是快不快。

## 最重要的一招：用单进程参考实现做黄金对照

多卡代码最常见的 bug 不是崩溃，而是**悄悄算错**：切错了维度、少了一次 all-reduce、RoPE 的位置偏移没跟着切、归一化在错误的轴上做。这类 bug 在真卡上也很难发现，因为 loss 照样往下掉，只是掉得慢一点。

对策非常简单：同一份权重、同一份输入，**单进程算一遍完整结果**，多进程按切分算一遍，逐元素比。下面用一个两层 MLP 演示列并行 + 行并行：

```python title="tp_check.py" torchrun="4"
import torch
import torch.distributed as dist

dist.init_process_group("gloo")
rank, P = dist.get_rank(), dist.get_world_size()
torch.manual_seed(0)                                   # 每个 rank 造出同一份完整权重，模拟"从同一个 checkpoint 切"

B, d, h = 8, 64, 256
x = torch.randn(B, d)
W1 = torch.randn(d, h) / d ** 0.5                      # 第一层：按列切（列并行）
W2 = torch.randn(h, d) / h ** 0.5                      # 第二层：按行切（行并行）

ref = torch.relu(x @ W1) @ W2                          # 单进程参考实现：黄金对照

W1_local = W1.chunk(P, dim=1)[rank]                    # 每个 rank 只拿自己那片
W2_local = W2.chunk(P, dim=0)[rank]
y = torch.relu(x @ W1_local) @ W2_local                # 列并行之后不需要通信，行并行之后要 all-reduce
dist.all_reduce(y)

err = (y - ref).abs().max().item()
if rank == 0:
    print(f"张量并行度 {P}，每个 rank 只持有 {W1_local.shape[1]} / {h} 个隐藏维")
    print(f"和单进程结果逐元素一致：{torch.allclose(y, ref, atol=1e-5)}（最大绝对误差 < 1e-5：{err < 1e-5}）")
    print("整个 MLP 只在最后 all-reduce 一次：这就是张量并行的通信量")
dist.destroy_process_group()
```

```text title="输出"
张量并行度 4，每个 rank 只持有 64 / 256 个隐藏维
和单进程结果逐元素一致：True（最大绝对误差 < 1e-5：True）
整个 MLP 只在最后 all-reduce 一次：这就是张量并行的通信量
```

把 `dist.all_reduce(y)` 注释掉，或者把 `chunk(P, dim=1)` 改成 `dim=0`，误差立刻变成 $10^{-1}$ 量级——这就是这套对照的价值。**本书后面每一章的多进程示例都带一个单进程参考**，读的时候可以重点看它是怎么写的。

这套方法覆盖面比想象中大：

- **ZeRO**：和单进程 AdamW 比参数更新后的结果；
- **序列并行 / 上下文并行**：和单进程完整注意力比输出和 log-sum-exp；
- **流水线并行**：和单进程逐层前向比每个 stage 的激活，再数一遍气泡数对不对；
- **MoE**：和单进程"按 router 结果分发"的实现比每个专家收到的 token 集合。

## 用算的代替用试的

性能问题在 CPU 上测不出来，但很多性能**结论**是可以算出来的。α-β 模型只有两个参数：每步的固定开销 $\alpha$ 和链路带宽 $\beta$，一次通信的时间是 $\alpha + S / \beta$。

```python title="commtime.py"
import math

ALPHA = 5e-6            # 每一步通信的固定开销（握手、同步），秒
BETA = 200e9            # 单向链路带宽，字节 / 秒（按 NVLink 一个方向 200 GB/s 估）


def ring(nbytes, n):    # 环形 all-reduce：2(n-1) 步，每步只发 1/n
    return 2 * (n - 1) * (ALPHA + nbytes / n / BETA)


def tree(nbytes, n):    # 树形：2*log2(n) 步，每步发整份
    return 2 * math.ceil(math.log2(n)) * (ALPHA + nbytes / BETA)


print("一次 all-reduce 要多久（α-β 模型，α = 5 μs，β = 200 GB/s）")
print(f"{'卡数':>4} {'消息大小':>10} {'环形':>12} {'树形':>12}  更快的")
for n in (4, 8, 16, 64):
    for name, nbytes in (("4 KB", 4 << 10), ("4 MB", 4 << 20), ("1 GB", 1 << 30)):
        t_ring, t_tree = ring(nbytes, n), tree(nbytes, n)
        print(f"{n:>4} {name:>10} {t_ring * 1e6:>9.1f} μs {t_tree * 1e6:>9.1f} μs  "
              f"{'环形' if t_ring < t_tree else '树形'}")

print()
print("通信占一步训练的多少（7B 模型，bf16 梯度 14 GB，一步前反向按 300 ms 算）")
grad = 14 * (1 << 30)
for n in (8, 16, 64, 256):
    t = ring(grad, n)
    print(f"  {n:>3} 卡数据并行：all-reduce {t * 1e3:>6.1f} ms，占一步的 {t / (0.3 + t):>5.1%}")
```

```text title="输出"
一次 all-reduce 要多久（α-β 模型，α = 5 μs，β = 200 GB/s）
  卡数       消息大小           环形           树形  更快的
   4       4 KB      30.0 μs      20.1 μs  树形
   4       4 MB      61.5 μs     103.9 μs  环形
   4       1 GB    8083.1 μs   21494.8 μs  环形
   8       4 KB      70.0 μs      30.1 μs  树形
   8       4 MB     106.7 μs     155.8 μs  环形
   8       1 GB    9465.2 μs   32242.3 μs  环形
  16       4 KB     150.0 μs      40.2 μs  树形
  16       4 MB     189.3 μs     207.8 μs  环形
  16       1 GB   10216.3 μs   42989.7 μs  环形
  64       4 KB     630.0 μs      60.2 μs  树形
  64       4 MB     671.3 μs     311.7 μs  树形
  64       1 GB   11199.6 μs   64484.5 μs  环形

通信占一步训练的多少（7B 模型，bf16 梯度 14 GB，一步前反向按 300 ms 算）
    8 卡数据并行：all-reduce  131.6 ms，占一步的 30.5%
   16 卡数据并行：all-reduce  141.1 ms，占一步的 32.0%
   64 卡数据并行：all-reduce  148.6 ms，占一步的 33.1%
  256 卡数据并行：all-reduce  152.3 ms，占一步的 33.7%
```

两个结论一眼就出来了，而且不需要任何硬件：

- **小消息用树、大消息用环**，交叉点随卡数往右移——这正是 NCCL 在 `NCCL_ALGO` 里做的选择；
- **环形 all-reduce 的时间几乎不随卡数增长**（8 卡到 256 卡只多了 16%），所以数据并行能扩展；但那 30% 的通信占比不会自己消失，得靠**通信与反向重叠**（见[数据并行与 DDP](../data/ddp.md)）藏掉。

## 有一张卡能多补什么

如果手头有一张消费级卡（哪怕是笔记本上的），还能补上这些：

- **同一张卡上起 2～4 个 rank 跑 NCCL**。性能数字没有意义（都在一张卡上抢 SM），但 NCCL 的 API 语义、通信器创建、通信组（`new_group`）、stream 上的异步性都是真的，能发现"CPU 上不报错、NCCL 上报错"的一类问题。
- **pinned memory、H2D / D2H 与计算重叠、CUDA IPC**。推理里 PD 分离传 KV 本质上就是这套东西（见[推理系统手册的 PD 分离](serving://distributed/pd-disagg/)）。
- **单卡上的显存账本与 OOM**。激活占多少、重计算省多少、碎片长什么样，这些在一张卡上就能量清楚（见[显存账本](overview.md)）。
- **混合精度与数值稳定**。bf16 / fp16 的溢出、loss scaling、梯度裁剪，单卡上全部可复现（见[混合精度与 FP8 训练](../practice/mixed-precision.md)）。

## 必须真卡的清单：租到卡照着跑

剩下的只能上真机。好消息是这张清单不长，**一次 8 卡、4～6 小时就能走完**，按小时租的成本很低。关键纪律只有一条：**脚本、对照组、要记录的指标全部在 CPU 上写好调通，上卡只负责跑和记**。

| 测什么 | 怎么测 | 看什么 | 异常的样子 |
| --- | --- | --- | --- |
| ① 拓扑基线 | `nvidia-smi topo -m` | 卡与卡之间是 NV#、PIX、PXB 还是 SYS | 本该 NVLink 的对出现 SYS，说明插槽或虚拟化有问题 |
| ② 通信基线 | `nccl-tests` 的 `all_reduce_perf`，消息从 8 B 扫到 8 GB | 总线带宽（busbw）曲线 | 大消息达不到链路峰值的 70%，或曲线有台阶 |
| ③ 算法选择 | 同上，分别锁 `NCCL_ALGO=Ring` / `Tree` | 交叉点在哪个消息大小 | 和上面 α-β 模型算出来的位置差一个数量级 |
| ④ 单卡基线 | 固定模型，TP=1，测 tokens/s 与显存 | 作为 scaling 的分母 | — |
| ⑤ TP scaling | 同一模型 TP=2/4/8 | 相对单卡的加速比、通信占比 | 加速比在 TP=4 就压不上去，通常是通信没重叠或拓扑跨了 NUMA |
| ⑥ 重叠收益 | 开 / 关通信重叠各跑一次 | 一步的耗时差 | 开了反而更慢：桶太小、通信被切碎 |
| ⑦ 数据并行 scaling | DDP 2/4/8 卡，固定全局 batch | 每卡 tokens/s 的衰减 | 衰减超过 α-β 模型的预测，看是不是梯度没分桶 |
| ⑧ ZeRO 各级 | ZeRO-1/2/3 各跑一次 | 每卡显存、一步耗时 | ZeRO-3 慢超过 1.5 倍，通常是参数 all-gather 没和前向重叠 |
| ⑨ 故障行为 | 训练中 `kill` 掉一个 rank | 多久超时、报什么错、怎么恢复 | 挂起不报错：`NCCL_ASYNC_ERROR_HANDLING` 没开 |
| ⑩ profile 一步 | Nsight Systems 抓 10 步 | 通信 kernel 和计算 kernel 有没有重叠 | 时间线上通信和计算完全串行 |

前四项是基线，必须先跑；⑤～⑧是有对照组的实验，是简历上真正能写的东西；⑨⑩看时间。

!!! tip "租卡前的准备清单"
    - 所有脚本 `git push` 好，上卡只 `git clone`，不要在卡上现写；
    - 数据集、权重先传到对象存储或用能直连的镜像源，别让下载吃掉机时；
    - 写一个 `run_all.sh` 串起全部实验，每项把结果追加到同一个 CSV；
    - 准备好"预期值"——每一项跑之前先用 α-β 模型算出该是多少，跑完立刻对，差得远就地查，不要留到下机之后；
    - 结束前 `nvidia-smi -q` 和 `nccl-tests` 的原始输出一起存下来，后面复盘要用。

## 常见的坑

- **端口被占 / 上一次的僵尸进程还在**：`--standalone` 会随机选端口，但上一轮残留的进程还占着显存和文件锁。先 `pkill -f torch.distributed` 再跑。
- **gloo 连不上**：主机名解析到了外网地址。设 `GLOO_SOCKET_IFNAME=lo`（macOS 是 `lo0`）。
- **慢到像卡死**：没设 `OMP_NUM_THREADS`，每个进程开满线程互相抢核。
- **死锁**：某个 rank 走了不同的分支，少调用了一次集合通信；或者先 `send` 再 `recv` 排成一个环。集合通信必须**所有 rank 都调用、顺序一致**；点对点要么用 `isend` / `irecv` 再统一 `wait`，要么按奇偶错开。
- **随机性不一致**：各 rank 的 `torch.manual_seed` 不同导致初始权重不同，结果对不上。要么统一种子，要么从 rank 0 `broadcast` 一份。
- **只有 rank 0 打印**：不加 `if rank == 0` 的话输出会交错成一团，看不出对错。

!!! interview "面试怎么答"
    被问到"你有多卡经验吗"，没有就直说没有，但要立刻补上你**验证过什么、怎么验证的**：用 gloo 在 CPU 上从零实现并验证了环形 all-reduce、ZeRO-2、1F1B、Ring Attention，全部和单进程参考实现逐元素对齐；通信量和时间用 α-β 模型算过，知道小消息为什么要换树形算法、为什么环形的时间几乎不随卡数增长；真卡上补测过 NCCL 的总线带宽曲线和 TP 的 scaling，发现了什么、原因是什么。这套回答比"用过 DeepSpeed"有说服力得多，因为它证明你理解的是机制而不是命令行。

## 练习

1. 把 `tp_check.py` 里的 `dist.all_reduce(y)` 删掉，观察误差变成多少；再把 `W1` 的切分从 `dim=1` 改成 `dim=0`，看误差又是多少。解释这两种错误在数学上分别错在哪里。

??? success "参考答案"
    删掉 all-reduce：每个 rank 只算出了"部分和"（自己那 64 个隐藏维的贡献），结果大约只有正确值的 $1/P$，误差和输出本身同量级。
    把 `W1` 按行切：`x @ W1_local` 的形状直接对不上（`x` 的列数是 64，`W1_local` 的行数变成 16），会直接报形状错误——这类错误反而是"好错误"，因为立刻暴露。真正危险的是形状恰好对得上、结果却错的情况，比如把 `W2` 也按列切，那时形状没问题但数学错了。

2. 用 `commtime.py` 的模型回答：64 张卡做 ZeRO-3，每步要 all-gather 参数两次（前向、反向各一次）再 reduce-scatter 梯度一次。7B 模型 bf16 参数 14 GB，通信一共多久？比纯数据并行多多少？

??? success "参考答案"
    ZeRO-3 的三次通信各自的量级都和一次 all-reduce 的"半程"相当：reduce-scatter 与 all-gather 各是 $(n-1)/n \cdot S$，所以三次合计约 $1.5 \times$ 一次 all-reduce。按上面 64 卡、14 GB 的数字，一次 all-reduce 约 149 ms，ZeRO-3 约 223 ms，多出约 50%。这正是"能用低一级就别上高一级"的原因；代价换来的是每卡显存从 $16\Psi$ 降到 $16\Psi/N$。

3. 设计一个在 CPU 上就能做的实验，验证"流水线并行的气泡率是 $(P-1)/(M+P-1)$"（$P$ 是 stage 数，$M$ 是 microbatch 数）。你需要记录什么？

??? success "参考答案"
    不需要真的算模型：每个 rank 用 `time.sleep` 模拟一个固定耗时的 stage，按 1F1B 的顺序收发激活和梯度，记录每个 rank 从第一次开始工作到最后一次结束之间**空闲的总时长**。气泡率 = 空闲时长 / 总时长。把 $M$ 从 1 扫到 16，画出来应该和公式重合。这个实验完全不依赖 GPU，因为气泡是**调度**造成的，不是算力造成的。

## 小结

- [x] 多卡的设计与正确性可以在 CPU 上 100% 验证：`torchrun --standalone` + gloo，同一套 `torch.distributed` API，本书所有多进程示例都这样跑。
- [x] 最有效的方法是**单进程参考实现做黄金对照**，逐元素比——切分写错几乎一定会在这里暴露。
- [x] 性能结论很多可以**算**出来：α-β 模型能解释为什么小消息用树、大消息用环，以及环形 all-reduce 为什么几乎不随卡数变慢。
- [x] 必须真卡的只有十项，一次 8 卡、几小时就能走完；纪律是脚本先在 CPU 上调通，上卡只跑和记。
