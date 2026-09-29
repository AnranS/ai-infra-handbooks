# 大规模 EP 的容错、弹性扩缩与排障

<p class="lead">专家并行把一个 MoE 模型摊到几十、几百张卡上，每一层都要在所有卡之间做一次 all-to-all。这让整个 EP 组变成一个整体：任何一张卡坏了、慢了，所有卡都跟着停。卡越多，这种事越常见，每次的代价也越大。这一章先算清楚故障的频率和影响范围，再读 SGLang 的弹性 EP：通信不被坏卡挂住、按健康的卡重新放置专家、故障卡恢复后重新加入、在线扩容；接着用一个估算看冗余专家能不能兜住坏卡，对比 vLLM 的做法，最后整理一份排障清单。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. 为什么 EP 组越大，单卡故障的代价越大？
    2. 一张卡坏了，SGLang 的弹性 EP 怎样让其他卡继续服务？
    3. 为什么只有热门专家有冗余副本时，坏一张卡几乎一定会"丢专家"？丢了的专家从哪里补回来？
    4. SGLang 和 vLLM 的弹性扩容有什么不同？
    5. 一个 EP 组整体卡住、所有卡都不出结果，怎么找到是哪张卡的问题？

??? success "自测参考答案（先自己答，再展开对照）"
    1. 单卡故障的次数只和卡的总数有关，和怎么分组无关；但每一层的 all-to-all 把整个 EP 组绑在一起，一张卡坏了整组都停，所以每次故障停掉的卡数就是组的大小。组越大，每次故障损失的卡·小时越多，停掉的在途请求也越多。
    2. 用 Mooncake（或 NIXL）作为弹性 EP 的通信后端，和坏卡通信不会一直挂住，坏掉的 rank 在 `active_ranks` 里被标成 0；各卡发现活跃集合变了，就触发 EPLB 在健康的卡上重新放置专家，并把这一批重新算一遍；故障卡修好后以 recover 模式重新加入各个通信组，从一张健康的卡同步专家的放置信息。
    3. 冗余副本只给最热的一小部分专家（比如 256 个专家里的 32 个），其余专家只有一份；一张卡上放着好几个只有一份的专家，这张卡一坏它们就没有了。重新放置时，还在别的卡上的专家通过卡间传输搬过去，所有副本都没了的专家要从外面加载：SGLang 可以打开专家备份（`--enable-elastic-expert-backup`），每台机器在内存里备份一部分专家的权重，用 Mooncake Transfer Engine 通过 RDMA 读回来；否则从磁盘重新读。
    4. SGLang 预留最大规模（`--max-ep-size`），`/scale_elastic_ep` 发起扩容后，新的 rank 以 scale 模式加入可扩展的全局通信组，专家的放置信息同步过去后提交，只支持扩大；vLLM 的 `--enable-elastic-ep` 以数据并行的引擎为单位扩缩（`/scale_elastic_ep` 的参数是新的 DP 规模，可大可小），用无状态的 NCCL 组重建通信，扩缩期间新请求返回 503，可以选择先把在途请求排空。
    5. 看每个 rank 最后在做什么：各 rank 的日志停在哪一步、用 py-spy 打印每个进程的调用栈、打开 NCCL 的调试日志或 PyTorch 的 flight recorder，找出没有进入这次集合通信（或者已经退出）的那个 rank；再看那张卡和它的网卡：GPU 的 Xid 错误、ECC 错误、网卡的链路错误和降速。

## 一张卡坏了，整组都停

MoE 层的 dispatch 和 combine 是所有 EP rank 之间的 all-to-all（见[专家并行](../distributed/expert-parallel.md)和 [NVSHMEM 与 DeepEP](../comm/nvshmem-deepep.md)）：每个 rank 都要等所有 rank 的数据到齐才能往下走。所以只要一个 rank 出了问题——进程崩溃、GPU 报错、网卡掉线——其他 rank 就会卡在这次通信上，直到超时。没有特别处理时，唯一的办法是整组重启：重新加载几百 GB 的权重、捕获 CUDA Graph、预热，在途的请求全部失败。

故障有多频繁？训练集群公开过比较完整的统计，可以借来估个量级：

```python
# Llama 3 论文公开的数字：16384 张 H100 预训练 54 天，意外中断 419 次（约 78% 是硬件问题）
per_gpu_hour = 419 / (54 * 24 * 16384)
TOTAL, MONTH_H = 2048, 30 * 24                  # 一个 2048 张卡的推理服务，按月统计
RESTART_MIN, PAUSE_MIN = 20, 1                  # 假设：整组重启（重新加载权重、捕获图、预热）20 分钟；弹性恢复时整组停顿 1 分钟
failures = TOTAL * per_gpu_hour * MONTH_H
print(f"每张卡每小时的意外故障率约 {per_gpu_hour:.2e}；{TOTAL} 张卡每月约 {failures:.0f} 次单卡故障，和怎么分组无关")
print("EP 组大小   组数   每组平均多久坏一次   每月损失的卡·小时：整组重启 / 弹性恢复")
for gpus in (16, 64, 128, 256):
    mtbf_days = 1 / (gpus * per_gpu_hour) / 24
    lost_restart = failures * gpus * RESTART_MIN / 60        # 一张卡坏了，整组都停
    lost_elastic = failures * gpus * PAUSE_MIN / 60
    print(f"{gpus:8d} {TOTAL // gpus:6d} {mtbf_days:14.1f} 天 {lost_restart:18.0f} / {lost_elastic:.0f}")
```

```text title="输出"
每张卡每小时的意外故障率约 1.97e-05；2048 张卡每月约 29 次单卡故障，和怎么分组无关
EP 组大小   组数   每组平均多久坏一次   每月损失的卡·小时：整组重启 / 弹性恢复
      16    128          132.0 天                155 / 8
      64     32           33.0 天                621 / 31
     128     16           16.5 天               1241 / 62
     256      8            8.2 天               2483 / 124
```

单卡故障的次数只取决于卡的总数；但 EP 组越大，每次故障连带停掉的卡越多。256 张卡一组时，同样的服务每月因重启损失的卡·小时是 16 张卡一组的 16 倍。能不能让一张卡坏了只影响这一张卡（和很短的一次停顿），就是弹性 EP 要解决的问题。表里的重启和停顿时长都是假设，真实部署要按自己的模型大小、存储和网络实测。

## SGLang 的弹性 EP

SGLang 的弹性 EP 用于 DP 注意力 + 大规模 EP 的部署（每个 rank 既是一个数据并行的注意力 rank，也是一个专家并行的 rank），代码在 `srt/elastic_ep/`。

### 不让通信被坏卡挂住

`--elastic-ep-backend` 指定一个能容错的通信后端（`mooncake` 或 `nixl`）。以 Mooncake 为例：各个通信组和 EP 的 dispatch / combine 都带着同一份 `active_ranks`（`ElasticEPState` 里的张量，每个 rank 一个 0/1）和超时。某个 rank 超时没有响应，就在 `active_ranks` 里被置 0，之后的通信跳过它，而不是一直挂住。

### 继续服务：在健康的卡上重新放置专家

模型每次前向之后，都会检查活跃集合和上一次的快照是否一样（`maybe_rebalance_after_rank_fault`）。不一样就说明有卡坏了：

1. 立刻让 EPLB 按剩下的健康卡重新计算专家的放置（EPLB 的算法见[大规模 EP 部署](ep-deploy.md)）；
2. `expert_location_updater` 把还在某张卡上的专家通过卡间传输搬到新的位置；
3. 所有副本都在坏卡上的专家（`p2p_missing_logical_experts`），只能从外面加载：打开了 `--enable-elastic-expert-backup` 时，从内存里的专家备份读（每台机器在主机内存里备份 1/节点数 的专家权重，通过 Mooncake Transfer Engine 用 RDMA 读取）；否则从磁盘重新读这些专家的权重；
4. 用新的放置把这一批重新算一遍，继续服务。

### 恢复：故障卡重新加入

修好（或换掉）的卡以 `--elastic-ep-join-mode recover` 启动（旧参数 `--elastic-ep-rejoin` 是它的别名），重新加入全局通信组和启动时建立的各个并行组。健康的 rank 在 `maybe_join_ep_ranks` 里发现有 rank 待恢复，就等对端就绪、在每个通信组上执行恢复，然后从一个健康的 rank 广播专家的放置信息，把活跃集合复位。

### 扩容：在线加卡

启动时用 `--max-ep-size` 预留能扩到的最大规模（活跃集合和通信缓冲区按它预先分配）。扩容的过程：

1. 调用 `POST /scale_elastic_ep`，参数 `{"new_ep_size": N}`：N 必须大于当前规模、不超过 `--max-ep-size`，而且上一次扩容已经完成（否则返回 409）；
2. 新的机器以 `--elastic-ep-join-mode scale` 启动（要求 `--node-rank 1`），用 `--elastic-ep-join-rank-offset` 告诉它当前的 EP 规模，用 `--elastic-ep-initial-size` 告诉它原部署启动时的 EP 规模（专家在每个 rank 上的存储布局由它决定）；它们在全局的 TCPStore 里登记自己要加入的目标规模；
3. 原有的 rank 在每一步检查：登记的目标和请求的 N 一致，就把新 rank 纳入可扩展的全局通信组，扩展 EPLB 的元数据、广播专家的放置、更新 DP 注意力的规模，过一道全局屏障后提交；
4. 状态依次是 `waiting_for_cohort` → `pending` → `joining` → `configuring_data_plane` → `syncing_new_world` → `serving_expanded`；`GET /is_scaling_elastic_ep` 可以查询；超过 `--elastic-ep-scale-timeout`（默认 600 秒）没完成就标记失败。

### 限制

- 只能扩大，不能缩小；扩容之后不再支持故障恢复（代码里直接报"Restart the expanded deployment"）；
- 扩容进行中会暂停 EPLB 的定期重新均衡；
- 活跃集合的检查在前向的路径上，会带来主机与设备之间的同步（代码里的注释也提到了这一点）。

## 冗余够不够：一个估算

弹性 EP 能不能"无缝"扛住坏卡，取决于坏卡上的专家在别处有没有副本。下面随机放置专家的副本（同一个专家的副本不放在同一张卡上），随机坏掉 k 张卡，看有多少专家在所有卡上都没了副本。前两种是[大规模 EP 部署](ep-deploy.md)一章里 prefill 和 decode 的配置（只给最热的专家加副本），第三种是每个专家都放两份：

```python
import random

E = 256                                              # DeepSeek-V3 的路由专家数


def place(gpus, slots_per_gpu, replicas, rng):
    """把各专家的副本放到卡上：同一个专家的副本不放在同一张卡上。返回每张卡上的专家列表"""
    copies = [e for e in range(E) for _ in range(replicas[e])]
    assert len(copies) == gpus * slots_per_gpu
    while True:
        rng.shuffle(copies)
        cards = [copies[g * slots_per_gpu:(g + 1) * slots_per_gpu] for g in range(gpus)]
        if all(len(set(c)) == len(c) for c in cards):
            return cards


def lost_after(cards, k, rng):
    """随机坏 k 张卡，数一数有多少个专家的副本全在坏卡上"""
    dead = set(rng.sample(range(len(cards)), k))
    alive = {e for g, c in enumerate(cards) if g not in dead for e in c}
    return E - len(alive)


rng = random.Random(0)
configs = [  # (说明, 卡数, 每卡槽位, 多出来的副本数：给最热的那些专家各加一份)
    ("EP32，每卡 9 个，32 个冗余", 32, 9, 32),
    ("EP320，每卡 1 个，64 个冗余", 320, 1, 64),
    ("EP64，每卡 8 个，每个专家 2 份", 64, 8, 256),
]
print("坏 k 张卡之后，至少有一个专家在所有卡上都没了副本的概率（平均丢几个）：")
for name, gpus, slots, extra in configs:
    replicas = [2 if e < extra else 1 for e in range(E)]
    cards = place(gpus, slots, replicas, rng)
    cells = []
    for k in (1, 2, 4):
        lost = [lost_after(cards, k, rng) for _ in range(4000)]
        cells.append(f"k={k}: {sum(x > 0 for x in lost) / len(lost):4.0%}（{sum(lost) / len(lost):.2f}）")
    print(f"  {name}  " + "  ".join(cells))
```

```text title="输出"
坏 k 张卡之后，至少有一个专家在所有卡上都没了副本的概率（平均丢几个）：
  EP32，每卡 9 个，32 个冗余  k=1: 100%（6.98）  k=2: 100%（14.00）  k=4: 100%（28.40）
  EP320，每卡 1 个，64 个冗余  k=1:  59%（0.59）  k=2:  84%（1.20）  k=4:  98%（2.40）
  EP64，每卡 8 个，每个专家 2 份  k=1:   0%（0.00）  k=2:  12%（0.13）  k=4:  56%（0.78）
```

- **只给热门专家加副本时，坏一张卡几乎一定丢专家**：EP32 每张卡上有 7 个左右只有一份的专家，坏一张卡丢 7 个；EP320 每卡一个专家，坏的那张卡有六成概率放的是只有一份的专家；
- **每个专家两份**也只能保证坏一张卡时不丢；坏两张卡就有一成多的概率丢，坏四张卡时过半；
- 所以"从别处补回丢掉的专家"是必须的一环：内存里的专家备份把这一步从读磁盘（几百 GB 的检查点里挑出这些专家）变成一次 RDMA 读，恢复时间差一个量级以上。代价是每台机器要拿出主机内存存备份。

## vLLM 的弹性 EP 与故障处理

vLLM 走的是另一条路：

- **`--enable-elastic-ep`**：以数据并行的引擎为单位扩缩。它要求打开 EPLB（`--enable-eplb`），不支持流水线并行，也不支持外部或混合的 DP 负载均衡（扩缩要由唯一的 API 服务器和引擎客户端协调）；`--elastic-ep-max-dp-size` 设置能扩到的最大 DP 规模；
- **`POST /scale_elastic_ep`**：参数是新的 DP 规模 `new_data_parallel_size`（可大可小）和 `drain_timeout`；扩缩期间一个中间件让新请求直接返回 503，设置 `VLLM_ELASTIC_EP_DRAIN_REQUESTS=1` 时先等在途请求排空（超时返回 408），然后用无状态的 NCCL 组重建 DP/EP 的通信、重新放置专家；
- **`--enable-fault-tolerance`**：某个 DP 引擎出错时，先中止它上面的请求、把状态报成不健康，等待上层下发恢复指令（比如重试，会重建它的 DP 通信组），超过 `engine_recovery_timeout_sec`（默认 120 秒）才真正报错退出。

两者的取舍：SGLang 把重点放在"坏一张卡时其他卡不停"和"在线加卡"，依赖能容错的通信后端和专家备份；vLLM 以 DP 引擎为粒度扩缩，机制更通用（普通的 NCCL），代价是扩缩时要短暂停止接收请求。

## 排障清单

大规模 EP 的问题大多表现为"整组都慢"或"整组都卡住"，要先找出是哪一个 rank、哪一张卡、哪一条链路：

| 现象 | 常见原因 | 看什么 |
| --- | --- | --- |
| 所有 rank 都卡住，不出结果 | 某个 rank 进程崩溃、GPU 报错，或者各 rank 的执行路径不一致（比如某个 rank 多做了一次集合通信） | 每个 rank 的日志停在哪一步；`py-spy dump` 打印每个进程的调用栈；`NCCL_DEBUG=INFO`；PyTorch 的 NCCL flight recorder（`TORCH_NCCL_TRACE_BUFFER_SIZE` 等）在超时时记录各 rank 最近的集合通信，对比谁没进入这次调用 |
| 整组变慢，但各 rank 都在跑 | 某张卡或某块网卡降速：GPU 降频、PCIe / NVLink 降级、IB 链路错误；或者专家负载失衡 | 每个 rank 的 dispatch、combine 耗时，找出总是最后到的那个；`nvidia-smi` 的时钟与 Xid 错误；网卡的链路状态和错误计数；专家负载统计（SGLang 的 `/start_expert_distribution_record` 和 `/dump_expert_distribution_record`） |
| 某个 DP rank 的队列越积越长 | DP 注意力的负载不均：长上下文请求集中到了少数 rank | 各 DP rank 的 token 数和 KV 占用；路由策略（见[大规模 EP 部署](ep-deploy.md#dp-attention-的负载最慢的-rank-决定速度)） |
| 重新放置专家之后某张卡 OOM | 冗余专家或新搬来的专家占了显存 | 每卡的专家数和显存余量；冗余专家的数量；KV 池的大小留没留余量 |
| 故障恢复之后输出乱码或 NaN | 专家的放置信息和实际加载的权重不一致；恢复路径没有重新加载丢失的专家 | 恢复日志里的放置信息广播和权重加载；对一批固定输入做逐 token 比对（见[新模型接入与精度对齐](../ops/new-model.md)） |

还有两条经验：一是**先找最慢的那一个**——集合通信的耗时由最慢的 rank 决定，看平均值会掩盖问题；二是**区分"坏"和"慢"**——进程崩溃、GPU 掉卡容易发现，而降频、链路降速、单个 GPU 的 ECC 错误不会让进程退出，只会让整组悄悄变慢，要靠持续的指标监控（每个 rank 的通信耗时、GPU 的时钟和错误计数）才能发现。

!!! interview "面试怎么答"
    被问"几百张卡的 EP 部署，一张卡坏了怎么办"：先讲为什么难——每层的 all-to-all 把整个 EP 组绑在一起，一张卡坏了整组卡在通信上，传统做法只能整组重启；故障次数由卡的总数决定，但每次故障的影响范围等于组的大小。再讲弹性 EP 的三步：通信后端能容错（和坏卡通信不挂住、标记出失联的 rank），EPLB 在健康的卡上重新放置专家（还在的专家卡间搬运，丢了的专家从内存备份用 RDMA 读回或从磁盘加载），故障卡修好后重新加入通信组并同步专家放置。补充冗余的局限（只给热门专家加副本时坏一张卡必然丢专家）、在线扩容（预留最大规模、新 rank 加入、提交），以及排障的方法：先找最慢或没进入集合通信的 rank，区分"坏"和"慢"。

## 练习

**1. 影响范围。** 用本章的估算，同样 2048 张卡，EP 组从 64 张卡扩大到 256 张卡，每月因整组重启损失的卡·小时变成多少倍？如果重启时间从 20 分钟缩短到 5 分钟呢？

??? success "参考答案"
    损失的卡·小时 = 故障次数 × 组大小 × 重启时长。故障次数不变，组大小从 64 变成 256，损失变成 4 倍（621 → 2483 卡·小时）。重启时间缩短到 1/4，损失也缩小到 1/4，正好抵消：256 张一组、5 分钟重启，和 64 张一组、20 分钟重启的损失相同。这说明两条路都有用——缩小影响范围（弹性 EP）和缩短恢复时间（更快的权重加载，比如从内存或邻近节点读）。

**2. 冗余与备份。** 为什么不干脆把每个专家都放两份，省掉专家备份？

??? success "参考答案"
    一是显存：DeepSeek-V3 每个专家在 58 个 MoE 层各有一份，每份 FP8 权重约 44 MB，256 个专家全部多放一份要多占约 650 GB 显存（256 × 58 × 44 MB），会挤掉大量 KV Cache；二是保证不了：本章的估算里，每个专家两份时坏两张卡仍有一成多的概率丢专家，坏四张卡时过半。冗余副本的主要用途是负载均衡（给热门专家分流），兜底还得靠备份：备份放在主机内存里，不占显存，丢了的专家通过 RDMA 读回来。

**3. 卡住了。** 一个 72 张卡的 EP 组所有请求都不出结果，GPU 利用率显示 100%。你会怎么一步步排查？

??? success "参考答案"
    GPU 利用率 100% 但不出结果，典型是 NCCL / DeepEP 的 kernel 在自旋等待某个对端。步骤：① 看各 rank 的日志最后一行，找出和别人不一样的 rank（比如有一个 rank 报了错或者早就没有新日志）；② 对所有 rank 做 `py-spy dump`，看每个进程停在哪个调用上，找出没有进入这次集合通信、或者执行路径不同（多做或少做了一次通信）的 rank；③ 如果开了 flight recorder，对比各 rank 最近的集合通信序号；④ 找到可疑的卡后看硬件：`nvidia-smi` 的 Xid 和 ECC 错误、网卡的链路状态；⑤ 没有弹性 EP 时，隔离这张卡、整组重启；有弹性 EP 时，确认它被标成失联、专家已经重新放置，再以 recover 模式把修好的卡加回来。

## 小结

- [x] 每层的 all-to-all 把 EP 组绑成一个整体：故障次数由卡的总数决定，每次故障的影响范围等于组的大小。
- [x] SGLang 的弹性 EP：能容错的通信后端（Mooncake / NIXL）+ `active_ranks` 标记失联的 rank → EPLB 在健康的卡上重新放置专家、重算这一批 → 故障卡以 recover 模式重新加入；`--max-ep-size` + `/scale_elastic_ep` 在线扩容。
- [x] 只给热门专家加副本时坏一张卡必然丢专家，要靠内存里的专家备份（RDMA 读回）或磁盘补回。
- [x] vLLM：`--enable-elastic-ep` 以 DP 引擎为单位扩缩（扩缩期间 503、可排空请求），`--enable-fault-tolerance` 让出错的引擎等待恢复指令。
- [x] 排障：先找最慢或没进入集合通信的 rank，区分"坏"和"慢"，持续监控每个 rank 的通信耗时和硬件错误。
