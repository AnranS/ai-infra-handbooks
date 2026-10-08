# RL 推理系统：异步 rollout、策略陈旧度与权重同步

<p class="lead"><a href="../../topics/rl-rollout/">RL 训练中的推理</a>一章讲了 rollout 的长尾、训练与推理的概率不一致和显存切换；分布式训练手册的<a href="train://practice/frameworks-rl/">训练框架与 RL 训练系统</a>从训练端跑通了一个 GRPO 循环，并实现了权重的重新切分。这一章把它们放到大规模的系统里算账：同步、一步异步、全异步三种流水各能跑多快，异步带来的"陈旧"落在哪些样本上；万亿参数的权重每一步怎样在几秒内送到所有推理实例；MoE 模型为什么还要把推理时的路由结果带回训练端。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. 同步 RL 的 rollout 为什么利用率很低？"一步异步"能解决多少？
    2. 全异步 RL 里，哪些样本最"陈旧"？这会带来什么偏差？
    3. 推理引擎在更新权重时，正在生成的请求怎么办？
    4. 万亿参数的权重要同步给几十个推理实例，怎样做到时间与实例数无关？
    5. MoE 模型的"路由重放"解决什么问题？推理引擎要提供什么？

??? success "自测参考答案（先自己答，再展开对照）"
    1. rollout 的回答长度是长尾分布：每一批都要等最长的那几条写完，大部分时间推理池只剩少数请求在跑（本章模拟里约 70% 的时间空着）。一步异步让生成下一批和训练这一批重叠，只能藏住训练的时间，长尾本身没变，只快了一点。
    2. 长回答：它们写得久，开始时用的权重早已过时（平均落后 2 个版本以上），短回答只落后约 0.4 个版本；设了陈旧度上限，被丢掉的也是长回答——长推理链既更陈旧又更容易被丢，模型可能被推向短回答。
    3. 暂停 → 更新 → 恢复：vLLM 可以选择中止正在进行的请求、等它们完成，或者冻结在队列里、更新后继续（这时一条回答的前后部分来自不同的权重，要逐 token 记录采样时的 logprob）；更新之后通常要清空前缀缓存。
    4. 每个推理 rank 直接收自己需要的那一片（先算好训练端和推理端切分之间的映射），发送端用多张卡并行发；实例之间分块流水接力：收到一块就转发给下一个实例，总时间几乎与实例数无关（本章估算 64 个实例约 2.6 秒）。
    5. 推理端和训练端数值上的微小差异，会让 MoE 的 top-k 路由选中不同的专家，变成完全不同的计算路径；路由重放记录推理时每个 token 每一层选中的专家，训练端直接使用。推理引擎要能返回每个 token 的路由结果（vLLM、SGLang 的 `--enable-return-routed-experts`）。

## 三种流水

![图：三种 RL 流水——同步、一步异步、全异步](../assets/figures/rl-pipelines.svg){.aig-svg}

rollout 的长尾让同步 RL 的大部分时间花在"等最后几条回答写完"上。模拟一个 256 路并发的推理池、每步要 512 个样本（比如 64 个问题 × 8 个回答）、训练一步 60 秒，回答长度服从长尾分布（中位数约 2000 token、截断在 16K），比较三种流水：

- **同步**：生成一整批 → 训练 → 更新权重 → 再生成下一批；
- **一步异步**：用上一版权重生成第 $i+1$ 批的同时训练第 $i$ 批，样本固定落后一个版本；
- **全异步**：推理池的槽位一空就用当前最新的权重开始新样本；训练端凑够一批就训练、更新版本号。

```python
import heapq
import random

SLOTS, SPEED, BATCH, TRAIN, HOURS = 256, 30.0, 512, 60.0, 2.0   # rollout 并发槽位、每条序列每秒生成的 token、每步样本数、每步训练秒数、模拟时长


def length(rng):
    return min(rng.lognormvariate(7.6, 1.0), 16384)            # 回答长度：中位数约 2000，长尾截断在 16K


def sync_like(overlap):
    """同步（overlap=False）与一步异步（overlap=True）：每步生成一整批，批内用连续批处理"""
    rng, t, steps, busy = random.Random(0), 0.0, 0, 0.0
    while t < HOURS * 3600:
        slots = [0.0] * SLOTS                                   # 每个槽位空闲的时刻（相对本批开始）
        for _ in range(BATCH):
            d = length(rng) / SPEED
            s = heapq.heappop(slots) if len(slots) == SLOTS else 0.0
            heapq.heappush(slots, s + d)
            busy += d
        rollout = max(slots)                                    # 这一批要等最长的那条生成完
        t += max(rollout, TRAIN) if overlap else rollout + TRAIN
        steps += 1
    return steps, busy / (t * SLOTS), (1.0, 1) if overlap else (0.0, 0)


def fully_async(max_stale=None):
    """全异步：槽位一空就用最新的权重开始新样本；训练端凑够一批就更新，版本号加一"""
    rng, version, steps, t, busy, done, stale, wasted = random.Random(0), 0, 0, 0.0, 0.0, [], [], 0.0
    running = [(n / SPEED, 0, n) for n in (length(rng) for _ in range(SLOTS))]   # (完成时刻, 开始时的版本, 长度)
    heapq.heapify(running)
    next_train = None
    while t < HOURS * 3600:
        finish, ver, n = heapq.heappop(running)
        if next_train is not None and next_train <= finish:     # 训练先结束：版本更新
            t, version, steps, next_train = next_train, version + 1, steps + 1, None
            heapq.heappush(running, (finish, ver, n))
        else:
            t = finish
            m = length(rng)
            busy += m / SPEED
            heapq.heappush(running, (t + m / SPEED, version, m))
            done.append((ver, n))
        if max_stale is not None:                               # 太陈旧的样本直接丢弃（按当前版本判断）
            keep = [x for x in done if version - x[0] <= max_stale]
            wasted, done = wasted + len(done) - len(keep), keep
        if next_train is None and len(done) >= BATCH:            # 凑够一批，开始训练
            batch, done = done[:BATCH], done[BATCH:]
            stale += [(version - v, n) for v, n in batch]
            next_train = t + TRAIN
    total = len(stale) + wasted
    lag = [x for x, _ in stale]
    by_len = [sum(x for x, n in stale if cond(n)) / max(1, sum(cond(n) for _, n in stale))
              for cond in (lambda n: n < 1000, lambda n: n > 8000)]
    return steps, busy / (t * SLOTS), (sum(lag) / len(lag), max(lag)), wasted / total, by_len


for name, res in [("同步", sync_like(False)), ("一步异步（生成下一批与训练重叠）", sync_like(True))]:
    steps, util, (avg, mx) = res
    print(f"{name}：{HOURS:.0f} 小时 {steps} 步，rollout 槽位利用率 {util:.0%}，样本陈旧度平均 {avg:.1f}、最大 {mx}")
for name, bound in [("全异步（不限陈旧度）", None), ("全异步（最多落后 2 个版本）", 2)]:
    steps, util, (avg, mx), wasted, (short, long_) = fully_async(bound)
    print(f"{name}：{HOURS:.0f} 小时 {steps} 步，rollout 槽位利用率 {util:.0%}，样本陈旧度平均 {avg:.1f}、最大 {mx}，丢弃 {wasted:.0%}")
    print(f"  其中短回答（<1K token）平均陈旧 {short:.1f} 个版本，长回答（>8K token）平均陈旧 {long_:.1f} 个版本")
```

```text title="输出"
同步：2 小时 11 步，rollout 槽位利用率 30%，样本陈旧度平均 0.0、最大 0
一步异步（生成下一批与训练重叠）：2 小时 12 步，rollout 槽位利用率 33%，样本陈旧度平均 1.0、最大 1
全异步（不限陈旧度）：2 小时 34 步，rollout 槽位利用率 100%，样本陈旧度平均 0.8、最大 4，丢弃 0%
  其中短回答（<1K token）平均陈旧 0.4 个版本，长回答（>8K token）平均陈旧 2.2 个版本
全异步（最多落后 2 个版本）：2 小时 33 步，rollout 槽位利用率 100%，样本陈旧度平均 0.7、最大 2，丢弃 2%
  其中短回答（<1K token）平均陈旧 0.4 个版本，长回答（>8K token）平均陈旧 1.8 个版本
```

- **同步**：每一批都要等最长的那条写完（约 9 分钟），推理池 70% 的时间空着；
- **一步异步**只把训练的 60 秒藏到了生成后面，长尾本身没有变短，只快了一步——训练时间远短于 rollout 时就是这样；
- **全异步**让推理池一直满载，同样的时间多跑了两倍的训练步；平均陈旧度不到一个版本，看起来代价很小。

但陈旧度**不是均匀分布**的：短回答很快写完、很快被训练用掉，平均只落后 0.4 个版本；长回答要写很久，开始时用的权重早已过时，平均落后 2 个版本以上。加上陈旧度上限后，被丢弃的恰恰也是长回答。对推理模型的 RL，这是一种系统性的偏差：**长的推理链条既更陈旧、又更容易被丢掉**，模型可能因此偏向短回答。常见的对策：

- **修正而不是丢弃**：用"采样时的策略"与"当前策略"的概率比做重要性加权（并截断），把陈旧样本的贡献修正回来——AReaL 等全异步系统采用的"解耦"的 PPO 目标就是把行为策略和近端策略分开处理；
- **按长度均衡地组批**：不要"先写完的先训练"，而是等到各长度段都有一定数量的样本；
- **可中断的生成**：权重更新时打断正在生成的长回答，换上新权重继续写（partial rollout 的一种）。这让一条回答的不同部分来自不同版本的策略，必须逐 token 记录采样时的 logprob，训练端才能正确修正；打断之后，已经生成的前缀的 KV 是旧权重算出来的，要么丢弃后用新权重重算（多一次 prefill），要么保留继续用（引入额外的不一致）。

## 权重同步的时间账

每一步训练之后，新权重都要送到所有推理实例。一个万亿参数的模型，推理端用 FP8 就是每个实例 1 TB：

```python
W = 1e12                                      # 万亿参数模型，推理端用 FP8：每个推理实例要收 1 TB
NIC, NVLINK, PER_INST = 50e9, 450e9, 8         # 每卡网卡带宽、NVLink 带宽、每个推理实例 8 张卡

print(f"共置（同一批卡）：机内 NVLink 聚合后交给推理进程，每卡约 {W / PER_INST / NVLINK:.2f} s")
print("分离部署，推理实例数：      1        8       64")
naive = [n * W / NIC for n in (1, 8, 64)]                       # 训练端一张卡依次发给每个实例
star = [n * W / (PER_INST * NIC) for n in (1, 8, 64)]           # 训练端 8 张卡并行发，但每个实例单独发一遍
tree = [W / (PER_INST * NIC) * (1 + 0.05 * (n > 1)) for n in (1, 8, 64)]   # 分块流水的广播：实例之间接力转发
for name, row in (("单卡依次发送", naive), ("8 卡并行、逐实例发送", star), ("8 卡并行 + 实例间流水接力", tree)):
    print(f"  {name}：" + "  ".join(f"{x:>7.1f} s" for x in row))
```

```text title="输出"
共置（同一批卡）：机内 NVLink 聚合后交给推理进程，每卡约 0.28 s
分离部署，推理实例数：      1        8       64
  单卡依次发送：   20.0 s    160.0 s   1280.0 s
  8 卡并行、逐实例发送：    2.5 s     20.0 s    160.0 s
  8 卡并行 + 实例间流水接力：    2.5 s      2.6 s      2.6 s
```

- **共置**最简单：数据只在机内移动，通过 CUDA IPC 交给推理进程，不到一秒；
- **分离部署**时，朴素的做法（汇总到一张卡、依次发给每个实例）要几十分钟，完全不可用；让推理实例的 8 张卡各收 1/8、发送端也用多张卡并行，一个实例只要 2.5 秒；再让实例之间**分块流水接力**（收到一块就转发给下一个实例，和 [ring / pipeline 广播](../comm/nccl.md)是同一个思路），总时间几乎与实例数无关（模型里给接力加了 5% 的开销）；
- 这一切的前提是每个推理 rank 能直接拿到自己需要的那一片——训练端和推理端的切分不同，要先算好"谁发给谁"的映射（见[权重的重新切分](train://practice/frameworks-rl/#权重的重新切分)），而不是先汇总成完整的权重；
- 推理端如果用 FP8 或更低的精度，还要在发送前（训练端）或接收后（推理端）做一次量化。

**推理引擎这一侧**要配合的是"暂停 → 更新 → 恢复"：vLLM 的 `pause_generation(mode=...)` 可以选择中止正在进行的请求（`abort`）、等它们完成（`wait`），或者冻结在队列里、恢复后继续（`keep`），并可以选择是否清空 KV 和前缀缓存；SGLang 提供 `/pause_generation` 和 `/continue_generation` 接口，加上 `update_weights_from_distributed`、`update_weights_from_tensor` 等更新权重的接口。更新之后，前缀缓存里旧权重算出的 KV 一般要清空，否则新请求会命中"旧策略"的缓存。热更新在引擎内部要处理的格式转换、原地更新和 CUDA Graph 的问题，见[权重热更新](../ops/weight-update.md)。

## MoE 的路由重放

[RL 训练中的推理](../topics/rl-rollout.md#问题二训练与推理的概率不一致)一章测过：推理端和训练端对同一个序列算出的概率在 bf16 下会有差异。MoE 模型把这个问题放大了：两边的数值只要有一点不同，路由器选出的 top-k 专家就可能不一样——一个 token 在推理时走了专家 3，训练端重算时却走了专家 7，概率差异就不再是"一点数值误差"，而是换了一条计算路径。

**路由重放**（routing replay）的做法是：rollout 时记录每个 token 在每一层选中的专家，训练端重算 logprob 和反向传播时直接使用这些专家，而不是重新路由。推理引擎要提供的是"返回每个 token 的路由结果"：vLLM 和 SGLang 都有 `--enable-return-routed-experts` 选项（实现分别在 vLLM 的 `model_executor/layers/fused_moe/routed_experts_capturer.py` 和 SGLang 的 `srt/state_capturer/routed_experts.py`）。数据量不大：61 层、每层 8 个专家编号，每个 token 约 500 字节，一条 16K 的回答约 8 MB。

!!! interview "怎么讲清楚"
    讲"怎样提高 RL 训练的效率"，先用数字说明瓶颈：同步 RL 里 rollout 被长尾拖住，推理池七成时间空闲，一步异步只能藏住训练时间；全异步让推理池满载、步数翻几倍，代价是陈旧度，而且陈旧集中在长回答上，需要重要性修正、按长度组批、逐 token 记录 logprob。再讲权重同步：共置走 NVLink 秒级以内；分离部署按映射点对点发送、实例间流水接力，万亿参数也能做到几秒且与实例数无关；推理端要支持暂停、更新、恢复和清缓存。最后提 MoE 的路由重放，说明你了解 MoE RL 特有的不稳定来源。

## 练习

**1. 什么时候一步异步就够了？** 在本章的模型里，把训练时间从 60 秒改成 600 秒（比如模型更大、每步样本更多），同步和一步异步的差距会怎样变化？

??? success "参考答案"
    同步的一步约为"rollout 时间 + 训练时间"，一步异步约为"两者中较大的那个"。训练时间 60 秒、rollout 约 9 分钟时，一步异步只省下 60 秒，几乎没有收益；训练时间和 rollout 时间相当（都在 10 分钟左右）时，一步异步能把每一步缩短近一半，就很划算。所以一步异步适合"训练和生成差不多重"的场景；生成远重于训练时（推理模型的长回答），要靠全异步或 partial rollout 去掉长尾本身。

**2. 为什么更新权重后要清空前缀缓存？** 如果不清空，会发生什么？有没有例外？

??? success "参考答案"
    前缀缓存里的 KV 是用旧权重算出来的。新请求命中这些缓存时，前缀部分的 KV 来自旧策略、之后的 token 来自新策略，得到的概率既不是旧策略的、也不是新策略的，重要性修正也无从谈起，还会让同一个问题的多个回答因为命中与否而分布不同。例外：共享的系统提示词等"与策略无关"的内容仍然会随着权重变化而变化（KV 依赖权重），所以严格来说都要清空；有的系统为了效率选择接受这部分误差（例如只在一个版本内复用前缀），但必须清楚这是一个近似。

## 小结

- [x] 同步 RL 被长尾拖住，推理池大部分时间空闲；一步异步只能藏住训练时间；全异步让推理池满载，训练步数成倍增加。
- [x] 全异步的陈旧度集中在长回答上，设上限时被丢弃的也是长回答，会让模型偏向短回答；对策是重要性修正、按长度组批、逐 token 记录 logprob 的可中断生成。
- [x] 权重同步：共置走 NVLink，秒级以内；分离部署按切分映射点对点发送、实例间流水接力，时间与实例数无关；推理端需要暂停 / 更新 / 恢复和清空缓存。
- [x] MoE 的路由在两端可能不同，路由重放记录推理时的专家选择、训练时直接使用，vLLM 和 SGLang 都能返回每个 token 的路由结果。
