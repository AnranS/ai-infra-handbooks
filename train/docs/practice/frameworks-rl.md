# 训练框架与 RL 训练系统

<p class="lead">前面十章从零实现了各种并行，实际训练用的是把它们组装好的框架：Megatron-LM、DeepSpeed、PyTorch 原生的 FSDP2 与 torchtitan。这一章先把这几个框架的定位、以及它们共同依赖的分布式 checkpoint 讲清楚；然后转到 RL 训练——它把训练框架和推理引擎绑在一起，是推理工程师最常接触训练侧的地方。我们从训练端的视角跑一个完整的 GRPO 循环，再实现 RL 系统里最容易出错的一步：把训练端切分的权重重新切分成推理引擎的布局。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. Megatron-LM、DeepSpeed、FSDP2 / torchtitan 的定位有什么不同？
    2. 分布式 checkpoint 为什么能用 4 张卡保存、2 张卡加载？
    3. 一个 GRPO 训练步从训练端看包含哪几步？为什么训练端要重算一遍 log 概率？
    4. 训练端 TP=2、推理端 TP=4 且用了融合的 `qkv_proj`，权重怎样转换？最容易犯什么错？
    5. RL 框架为什么普遍采用"单控制器"来编排？

## 训练框架地图

| 框架 | 定位 | 并行能力 | 适合 |
| --- | --- | --- | --- |
| **Megatron-LM / Megatron-Core**（NVIDIA） | 大规模预训练的事实标准；Megatron-Core 是可复用的库，NeMo 等上层框架基于它 | TP + SP、交错 1F1B 的 PP、CP、EP（MoE 并行折叠）、分布式优化器（ZeRO-1）；通过 Transformer Engine 做 FP8 | 百 B 以上、上千卡；模型要按它的模块写 |
| **DeepSpeed**（微软） | ZeRO 的原始实现，和 HuggingFace 生态集成好 | ZeRO-1/2/3、CPU / NVMe 卸载（ZeRO-Offload / Infinity）、PP、Ulysses 序列并行 | 微调、中等规模训练、显存紧张时的卸载 |
| **PyTorch 原生：FSDP2 + torchtitan** | PyTorch 自己的分布式栈；torchtitan 是把它们组合起来的参考实现 | FSDP2（基于 DTensor）、`parallelize_module` 做 TP、`torch.distributed.pipelining` 做 PP、实验性的 CP；和 `torch.compile`、Float8 训练配合 | 希望留在原生 PyTorch 里、模型代码改动最少 |
| **HF Transformers Trainer / Accelerate / TRL** | 高层训练循环，底下调用 FSDP 或 DeepSpeed | 取决于底层 | 微调和后训练的快速实验 |

几个框架在概念上完全对应本书的章节：Megatron 的 `ColumnParallelLinear` / `RowParallelLinear` 就是[张量并行](../model/tensor-sequence.md)里的 $f$ / $g$ 算子，分布式优化器就是 [ZeRO-1](../data/zero-fsdp.md)，DeepSpeed 的 ZeRO-3 和 FSDP 做的是同一件事。选框架时真正要考虑的是：模型规模需要哪几种并行、团队愿意为性能付出多少模型改写的代价、以及和下游（推理引擎、RL 框架）的对接是否方便。

## 分布式 checkpoint 与重新切分

大模型的 checkpoint 不能由 rank 0 收集完整的权重再写盘——70B 的模型状态有 1 TB。所有框架都用**分布式 checkpoint**：每个 rank 只写自己的分片，再写一份元数据，记录每个张量的全局形状和每一片在哪个文件的哪个位置。加载时按元数据计算"我这个 rank 需要全局张量的哪一块"，从对应的文件里读出来。于是保存和加载可以用**不同的并行布局**——训练中途换卡数、从预训练切到微调、导出给推理，都依赖这一点。

PyTorch 的实现是 `torch.distributed.checkpoint`（DCP），Megatron 的 `dist_checkpointing` 思路相同。先用 4 个进程、4 路 FSDP 切分保存：

```python title="dcp_save.py" torchrun="4"
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
```

```text title="输出"
4 个 rank 保存，第一层权重每片 (64, 64)
checkpoint 目录： ['.metadata', '__0_0.distcp', '__1_0.distcp', '__2_0.distcp', '__3_0.distcp']
```

再用 2 个进程、2 路切分加载同一个 checkpoint，并离线合并成一个普通的 `state_dict`：

```python title="dcp_load.py" torchrun="2"
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
```

```text title="输出"
2 个 rank 加载，第一层权重每片 (128, 64)，与原模型一致：True
合并后的完整权重： {'0.weight': (256, 64), '0.bias': (256,), '2.weight': (64, 256), '2.bias': (64,)}
```

导出给推理引擎还要多一步**格式转换**：Megatron 的参数名、融合方式（比如把 Q、K、V 交错存在一个张量里）和 HuggingFace 格式不同，要按名字映射、拆分或拼接，再保存成 safetensors。推理引擎加载时又会按自己的方式融合和切分——这正是下面 RL 权重同步要在线完成的事。

## 从训练端看 RL 的一步

RL 训练的算法见大模型手册的[后训练](llm://training/post-training/#推理模型用可验证的奖励做强化学习)，推理引擎侧的问题见推理系统手册的 [RL 训练中的推理](serving://topics/rl-rollout/)。这里从训练端看一个 GRPO 步骤要做的事，用一个最小的例子把它们串起来：一个只看上一个 token 的"语言模型"，任务是"每个 token 等于上一个 token + 1"；推理引擎的替身用 bf16 权重采样，训练端保留 fp32 主权重：

```python title="grpo_toy.py"
import torch

V, T, G, PROMPTS = 16, 6, 8, 32          # 词表大小、生成长度、每个 prompt 采样几条（GRPO 的组）、每步几个 prompt


class Policy(torch.nn.Module):
    """最小的"语言模型"：下一个 token 只看上一个 token"""

    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(V, 32)
        self.head = torch.nn.Linear(32, V)

    def forward(self, tok):
        return self.head(torch.tanh(self.emb(tok)))


def reward(prompts, seq):
    """规则奖励：每个"等于上一个 token + 1"的位置得分，满分 1"""
    prev = torch.cat([prompts[:, None], seq[:, :-1]], 1)
    return (seq == (prev + 1) % V).float().mean(1)


class Engine:
    """推理引擎的替身：bf16 权重，只负责采样，并顺手返回每个 token 的 logprob"""

    def __init__(self):
        self.model = Policy().to(torch.bfloat16)

    def load_weights(self, state):
        self.model.load_state_dict(state)                 # 权重同步：fp32 主权重 → bf16 推理权重

    @torch.no_grad()
    def generate(self, prompts, gen):
        tok, logps = prompts[:, None], []
        for _ in range(T):
            lp = self.model(tok[:, -1]).float().log_softmax(-1)
            nxt = torch.multinomial(lp.exp(), 1, generator=gen)
            logps.append(lp.gather(1, nxt))
            tok = torch.cat([tok, nxt], 1)
        return tok[:, 1:], torch.cat(logps, 1)


def token_logp(model, prompts, seq):
    inp = torch.cat([prompts[:, None], seq[:, :-1]], 1)
    return model(inp).log_softmax(-1).gather(2, seq[..., None]).squeeze(-1)


torch.manual_seed(0)
gen = torch.Generator().manual_seed(0)
policy, engine = Policy(), Engine()
opt = torch.optim.Adam(policy.parameters(), lr=1e-2)
for step in range(1, 41):
    engine.load_weights(policy.state_dict())                              # ① 同步权重
    prompts = torch.randint(0, V, (PROMPTS,), generator=gen).repeat_interleave(G)
    seq, engine_logp = engine.generate(prompts, gen)                      # ② rollout
    r = reward(prompts, seq).view(PROMPTS, G)                             # ③ 打分
    adv = ((r - r.mean(1, keepdim=True)) / (r.std(1, keepdim=True) + 1e-6)).view(-1, 1)   # ④ 组内相对优势
    with torch.no_grad():
        old_logp = token_logp(policy, prompts, seq)                       # ⑤ 训练端重算 old logprob
    for _ in range(2):                                                    # ⑥ PPO 式的裁剪更新，每批数据用两遍
        ratio = (token_logp(policy, prompts, seq) - old_logp).exp()
        loss = -torch.min(ratio * adv, ratio.clamp(0.8, 1.2) * adv).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    if step % 8 == 0 or step == 1:
        gap = (engine_logp - old_logp).abs()
        print(f"step {step:>2}：平均奖励 {r.mean():.2f}，推理引擎与训练端的 logprob 差：平均 {gap.mean():.4f}，最大 {gap.max():.4f}")
```

```text title="输出"
step  1：平均奖励 0.07，推理引擎与训练端的 logprob 差：平均 0.0008，最大 0.0032
step  8：平均奖励 0.38，推理引擎与训练端的 logprob 差：平均 0.0020，最大 0.0081
step 16：平均奖励 0.83，推理引擎与训练端的 logprob 差：平均 0.0010，最大 0.0172
step 24：平均奖励 0.98，推理引擎与训练端的 logprob 差：平均 0.0004，最大 0.0217
step 32：平均奖励 0.99，推理引擎与训练端的 logprob 差：平均 0.0001，最大 0.0185
step 40：平均奖励 1.00，推理引擎与训练端的 logprob 差：平均 0.0001，最大 0.0393
```

六个编号的步骤就是真实系统里的六个阶段，每一步在大模型上都变成了一个系统问题：

- **① 同步权重**：真实模型有几十到几百 GB，训练端和推理端的切分方式还不一样（下一节）；
- **② rollout**：占一步的大半时间，长尾回答让 GPU 大部分时间只在跑很小的 batch；
- **③ 打分**：数学题用规则比对答案，代码题要在沙箱里跑测试，智能体任务要和环境多轮交互——它们通常是 CPU 密集或 I/O 密集的独立服务；
- **④ 优势**：GRPO 用组内均值做基线，省掉了 PPO 的价值模型（critic）；代价是每个 prompt 要采 $G$ 条，全对或全错的组优势为 0、没有学习信号；
- **⑤ 重算 log 概率**：训练端对所有回答做一次前向。即使权重完全相同，推理引擎和训练端的数值也不一样——上面最大的差在训练后期反而变大了：策略越来越确定，logits 的数值越来越大，bf16 的相对误差换算成绝对误差也越大。真实模型、长序列、不同的 kernel 下差异要大得多（推理系统手册在一个 0.6B 模型上实测，bf16 下同一个 token 的概率可以相差 10%），所以 PPO 比率的分母必须用训练端重算的值，并用推理端返回的 logprob 做重要性采样修正；带 KL 惩罚时还要用参考模型再做一次前向；
- **⑥ 更新**：就是前面十章的内容：FSDP 或 Megatron 的前向、反向和优化器步骤。

## 权重的重新切分

训练端和推理端对同一个模型的切分方式几乎总是不同的：训练端可能是 FSDP（按参数平铺切）或 Megatron 的 TP=2 × PP=4，推理端是 TP=8、融合了 `qkv_proj` 和 `gate_up_proj`，还可能有 EP。每次同步都要把前者转成后者。

下面用一个带 GQA 的注意力层演示：训练端 Megatron 风格 TP=2（Q、K、V 分开存，各自按头切），推理端 vLLM / SGLang 风格——Q、K、V 融合成一个矩阵，再按 TP 切；KV 头只有 2 个，推理端 TP=4 时每个 KV 头要**复制**到 2 个 rank 上：

```python title="weight_sync.py"
import torch

torch.manual_seed(0)
H, HEADS, KV_HEADS, D = 64, 8, 2, 8               # 隐藏维度、Q 头数、KV 头数（GQA）、头维度
GROUP = HEADS // KV_HEADS                         # 每个 KV 头服务 4 个 Q 头
Wq, Wk, Wv = torch.randn(HEADS * D, H), torch.randn(KV_HEADS * D, H), torch.randn(KV_HEADS * D, H)
Wo = torch.randn(H, HEADS * D)
x = torch.randn(5, H)


def attention(q, k, v):
    """q: [T, 头数, D]，k/v: [T, KV 头数, D]；每 GROUP 个 Q 头共用一个 KV 头"""
    rep = q.shape[1] // k.shape[1]
    k, v = k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)
    s = torch.einsum("thd,shd->hts", q, k) / D**0.5
    s = s.masked_fill(torch.ones(len(q), len(q)).triu(1).bool(), float("-inf"))
    return torch.einsum("hts,shd->thd", s.softmax(-1), v).reshape(len(q), -1)


ref = attention((x @ Wq.T).view(5, HEADS, D), (x @ Wk.T).view(5, KV_HEADS, D), (x @ Wv.T).view(5, KV_HEADS, D)) @ Wo.T

# ---- 训练端：Megatron 风格 TP=2，q/k/v 各自按头切（列并行），o 按输入维切（行并行）
TRAIN_TP = 2
train = [dict(q=Wq.chunk(TRAIN_TP)[r], k=Wk.chunk(TRAIN_TP)[r], v=Wv.chunk(TRAIN_TP)[r], o=Wo.chunk(TRAIN_TP, 1)[r])
         for r in range(TRAIN_TP)]


def gather(name, dim=0):
    return torch.cat([t[name] for t in train], dim)      # 第 1 步：把训练端的分片拼回完整权重


# ---- 推理端：q/k/v 融合成一个 qkv_proj，按 TP 切；KV 头比 TP 度数少时，每个 KV 头复制到多个 rank 上
def reshard(tp):
    q, k, v, o = gather("q"), gather("k"), gather("v"), gather("o", 1)
    q_per, kv_per = HEADS // tp, max(1, KV_HEADS // tp)
    shards = []
    for r in range(tp):
        h0 = r * q_per
        kv0 = h0 // GROUP                                 # 这几个 Q 头对应的第一个 KV 头
        shards.append(dict(qkv=torch.cat([q[h0 * D:(h0 + q_per) * D], k[kv0 * D:(kv0 + kv_per) * D],
                                          v[kv0 * D:(kv0 + kv_per) * D]]),
                           o=o[:, h0 * D:(h0 + q_per) * D]))
    return shards


def reshard_naive(tp):
    qkv = torch.cat([gather("q"), gather("k"), gather("v")])      # 先拼出完整的 qkv，再按行均分
    return [dict(qkv=w, o=o) for w, o in zip(qkv.chunk(tp), gather("o", 1).chunk(tp, 1))]


def infer_forward(shards):
    tp = len(shards)
    q_per, kv_per = HEADS // tp, max(1, KV_HEADS // tp)
    out = 0
    for w in shards:                                              # 每个 rank 算自己的头，最后 all-reduce（这里直接相加）
        q, k, v = (x @ w["qkv"].T).split([q_per * D, kv_per * D, kv_per * D], -1)
        out = out + attention(q.view(5, q_per, D), k.view(5, kv_per, D), v.view(5, kv_per, D)) @ w["o"].T
    return out


for tp in (2, 4):
    for name, fn in (("正确", reshard), ("朴素", reshard_naive)):
        try:
            err = f"最大误差 {(infer_forward(fn(tp)) - ref).abs().max():.1e}"
        except RuntimeError:
            err = "形状对不上，报错"
        print(f"推理 TP={tp}，{name}的重切分：{err}")

for r in range(4):                                                # 每个推理 rank 只需要一个训练 rank 的数据：可以点对点直传
    heads = range(r * 2, r * 2 + 2)
    src = sorted({h // (HEADS // TRAIN_TP) for h in heads})
    print(f"推理 rank {r} ← 训练 rank {src}：Q 头 {heads[0]}～{heads[-1]}，KV 头 {heads[0] // GROUP}")
```

```text title="输出"
推理 TP=2，正确的重切分：最大误差 4.6e-05
推理 TP=2，朴素的重切分：最大误差 3.2e+02
推理 TP=4，正确的重切分：最大误差 6.1e-05
推理 TP=4，朴素的重切分：形状对不上，报错
推理 rank 0 ← 训练 rank [0]：Q 头 0～1，KV 头 0
推理 rank 1 ← 训练 rank [0]：Q 头 2～3，KV 头 0
推理 rank 2 ← 训练 rank [1]：Q 头 4～5，KV 头 1
推理 rank 3 ← 训练 rank [1]：Q 头 6～7，KV 头 1
```

几个要点：

- **融合权重不能"先拼后切"**：融合矩阵的每个 TP 分片是"本 rank 的 Q 头 + 对应的 K 头 + 对应的 V 头"，而不是完整矩阵的连续一段。TP=2 时朴素做法形状恰好对得上，结果却完全错误、不报任何错——这类 bug 在 RL 里的表现是"奖励不涨"或"rollout 输出乱码"，很难直接定位。`gate_up_proj` 同理。推理引擎的每个层都有自己的 `weight_loader` 处理这些映射，RL 框架调用的就是它；
- **GQA 的 KV 头复制**：推理 TP 度数大于 KV 头数时，KV 头要复制到多个 rank；训练端没有这种复制，转换时要按映射取；
- **不必先聚合**：最后四行说明，每个推理 rank 需要的数据只来自一个训练 rank。按映射直接点对点发送，就不需要先在某处拼出完整权重——大规模系统（尤其是分离部署）都朝这个方向优化；
- **量化推理**：推理端如果用 FP8，每次同步还要在线量化（重新计算缩放因子）。

同步的开销可以粗算：70B 的 bf16 权重是 141 GB。共置部署时，训练和推理在同一批卡上，按层聚合、通过 CUDA IPC 把张量句柄交给推理进程，数据只在节点内的 NVLink 和显存里移动；分离部署时，推理实例 TP=8，每张卡要收 $141 / 8 \approx 17.6$ GB，走 50 GB/s 的网卡至少 0.35 秒，而且每个推理实例都要一份——要用广播树或流水线式的传输，并与下一轮 rollout 重叠。

## 共置还是分离

| | 共置（同一批卡，训练和推理轮流用） | 分离（训练和推理各用一批卡） |
| --- | --- | --- |
| 利用率 | 两个阶段都能用满所有卡 | 两边的卡数要按负载配比，配错了一边空等 |
| 切换开销 | 每步都要腾显存：推理引擎释放 KV cache（sleep），训练端把优化器状态甚至参数卸载到 CPU | 没有 |
| 权重同步 | 同机，CUDA IPC，快 | 跨机传输，需要 RDMA / NCCL 广播 |
| 并行布局 | 训练和推理各自选最优布局，但共享同样的卡数 | 两边可以用完全不同的卡数和布局 |
| 同步 / 异步 | 天然同步（on-policy） | 可以让 rollout 和训练流水起来（异步，off-policy），要限制策略的滞后步数并修正 |

共置的代价主要在切换：训练端每张卡的模型状态和激活，与推理端的权重和 KV cache 不能同时驻留。分离的代价在配比和同步：rollout 慢了训练端等，训练慢了推理端等，于是有了异步 RL——推理端不停地生成，训练端凑够一批就更新，推理端在生成过程中换上新权重。

## RL 框架的结构

RL 训练有多个模型（actor、参考模型，PPO 还有 critic 和奖励模型）、多个阶段、训练和推理两种引擎。纯 SPMD（每个 rank 跑同一份脚本，就像本书的 torchrun 例子）很难编排这么多角色，所以主流 RL 框架都用**单控制器**：一个驱动进程（通常基于 Ray）按顺序调用各个"工作组"的方法，把数据在它们之间传递；每个工作组内部仍是 SPMD（FSDP、Megatron、推理引擎的 TP 组）。算法逻辑写在驱动进程里，像写单机程序一样；分布式细节封装在工作组里。

| 框架 | 训练后端 | 推理后端 | 特点 |
| --- | --- | --- | --- |
| **verl** | FSDP / FSDP2、Megatron | vLLM、SGLang | HybridFlow：单控制器编排 + 工作组内 SPMD；默认共置（hybrid engine），在同一批卡上切换训练与推理并在内部完成重新切分 |
| **OpenRLHF** | DeepSpeed | vLLM | 基于 Ray，actor、critic、参考模型、奖励模型可以分开放置，也可以共置 |
| **slime** | Megatron | SGLang | 把 Megatron 训练和 SGLang rollout 直接连起来的轻量框架，支持共置和分离、同步和异步 |
| **AReaL** | FSDP、Megatron | SGLang、vLLM | 全异步：rollout 不等训练，生成过程中可以中断换权重；训练时按样本的滞后程度修正 |
| **TRL** | Accelerate（FSDP / DeepSpeed） | vLLM（可选） | HuggingFace 的后训练库，`GRPOTrainer` 等，适合中小规模 |

这些框架对推理引擎提出的要求——返回采样时的 logprobs、释放和恢复显存、在线更新权重、中断和续写请求、确定性采样——在推理系统手册的 [RL 训练中的推理](serving://topics/rl-rollout/) 一章里有逐项的讨论。

!!! interview "面试怎么答"
    RL 训练系统题：GRPO 的一步是同步权重 → rollout → 打分 → 组内归一化的优势 → 训练端重算 log 概率 → 裁剪更新；重算是因为推理端和训练端的数值不一致，要靠它和重要性采样修正。权重从训练端送到推理端要处理融合的权重（`qkv_proj`、`gate_up_proj`）、GQA 的 KV 头复制和不同的 TP 度数，"先拼后切"会静默出错，按映射点对点直传能省掉聚合。部署上：共置省卡、同步简单但要切换显存；分离可以异步、布局自由但要跨机同步权重。框架层面讲清 Megatron-Core、DeepSpeed、FSDP2 各自的定位，以及 RL 框架用单控制器编排多个 SPMD 工作组。

## 练习

1. 在 `weight_sync.py` 的基础上，写出 MLP 的 `gate_up_proj` 从训练端 TP=2 转到推理端 TP=4 的重新切分。朴素做法错在哪里？

??? success "参考答案"
    推理端第 $r$ 个 rank 的 `gate_up_proj` 分片应该是 `cat([gate 的第 r 片, up 的第 r 片])`，其中"第 r 片"是按 TP=4 对 `gate`、`up` 各自的输出维（中间维度）切分得到的：

    ```python
    gate, up = gather("gate"), gather("up")                  # 先拼回 [I, H]
    shards = [torch.cat([g, u]) for g, u in zip(gate.chunk(4), up.chunk(4))]
    ```

    朴素做法 `torch.cat([gate, up]).chunk(4)` 会让 rank 0、1 分到的全是 gate，rank 2、3 分到的全是 up，形状完全对得上，结果却是错的——激活函数 $\text{SiLU}(\text{gate}) \cdot \text{up}$ 里相乘的两个分量不再对应。`down_proj` 是行并行，按输入维切 4 份即可。

2. 为什么 PPO 比率的分母要用训练端重算的 `old_logp`，而不直接用推理引擎返回的 `engine_logp`？推理引擎返回的值又有什么用？

??? success "参考答案"
    PPO 的比率 $\pi_\theta / \pi_{\text{old}}$ 衡量的是"更新了几步之后策略变了多少"，裁剪也是基于这个比率。如果分母用推理端的值，更新之前比率就已经不等于 1——两者的数值差异被当成了策略的变化，裁剪会在错误的地方生效。训练端重算保证第一次更新前比率恰好是 1。
    推理引擎返回的 logprob 代表"样本实际是从哪个分布采出来的"。训练端和推理端的分布不同，本质上是 off-policy 的，所以用 $\pi_{\text{old}} / \pi_{\text{rollout}}$ 做重要性采样修正（通常截断，比如 TIS），或者丢掉差异过大的样本。异步 RL 里策略滞后带来的偏差也用同一套方法处理。

3. 一个 7B 模型做 GRPO，64 张卡。你倾向于共置还是分离？如果是一个 600B 的 MoE 模型、上千张卡呢？

??? success "参考答案"
    7B、64 卡：倾向**共置**。模型小，训练状态和推理的 KV cache 在 80 GB 的卡上都容易腾挪，切换开销小；共置时所有卡在两个阶段都有活干，也不用费心配比。
    600B MoE、上千卡：倾向**分离**（通常配合异步）。训练端的模型状态巨大，每步卸载和恢复的代价高；训练和推理的最优并行布局差别很大（训练端 PP × EP，推理端大规模 EP + DP 注意力），在同一批卡上很难兼顾；rollout 的长尾在这个规模上更严重，异步能把它和训练重叠。代价是要有高效的跨机权重同步，以及对策略滞后的修正。

## 小结

- [x] Megatron-Core 是大规模预训练的标准，DeepSpeed 以 ZeRO 和卸载见长，FSDP2 + torchtitan 是 PyTorch 原生的组合；它们的构件都对应本书的各章。
- [x] 分布式 checkpoint 每个 rank 写自己的分片加一份元数据，加载时按元数据重新切分，保存和加载可以用不同的并行布局。
- [x] GRPO 的一步：同步权重 → rollout → 打分 → 组内优势 → 训练端重算 log 概率 → 裁剪更新；训练端与推理端的数值差异要靠重算和重要性采样处理。
- [x] 训练端到推理端的权重转换要处理融合权重、GQA 的 KV 头复制和不同的 TP 度数；"先拼后切"会静默出错；按映射点对点直传可以省掉聚合。
- [x] 共置省卡、同步简单但要切换显存；分离可以异步、布局自由但要跨机同步权重。RL 框架用单控制器编排多个 SPMD 工作组。
