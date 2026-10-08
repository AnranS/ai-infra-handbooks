# KV 传输引擎与分布式 KV 存储：Mooncake、NIXL 与 3FS

<p class="lead">PD 分离要把 KV 从 prefill 实例搬到 decode 实例，分层缓存要把 KV 在显存、内存、远端内存和 SSD 之间搬来搬去。每个推理框架都自己写一遍 RDMA 不现实，于是出现了专门的<b>传输引擎</b>（Mooncake Transfer Engine、NVIDIA NIXL）和<b>分布式 KV 存储</b>（Mooncake Store、LMCache、3FS）。这一章讲它们的抽象和设计取舍，并用一个多轮对话的模拟说明：跨实例共享的 KV 池到底能省多少 prefill。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. 一个 KV 传输引擎要解决哪些问题？为什么不直接用 NCCL？
    2. Mooncake Transfer Engine 的 segment 和批量传输接口是怎样的？
    3. 分布式 KV 存储用什么做键？为什么是"前缀链式"的哈希？
    4. 已经有缓存感知路由了，为什么还需要跨实例共享的 KV 池？
    5. 3FS 是什么？它在推理里起什么作用？

??? success "自测参考答案（先自己答，再展开对照）"
    1. 要在不同的机器、不同的介质（显存、内存、SSD）之间点对点地搬运大量大小不一的小块：注册内存、交换元数据（地址、密钥）、批量读写、完成通知，还要处理多网卡、拓扑和故障。NCCL 是为固定成员的集合通信设计的，通信组要事先建立、双方都要参与，不适合这种动态、点对点、多介质的传输。
    2. segment 是一段注册好、可以被远端访问的内存（显存、内存或文件），用名字标识，元数据（地址、密钥、所在的网卡）登记在元数据服务里；传输时提交一批读写请求（本地地址、目标 segment 和偏移、长度），引擎自动选择网卡和路径、并行执行，最后查询完成状态。
    3. 用前缀的链式哈希做键：第 i 个块的键 = hash(第 i−1 个块的键, 第 i 个块的 token)。这样同一个键就代表同一个完整的前缀（KV 取决于整个前缀），不同实例对同样的前缀会算出同样的键，可以在集群里全局查找。
    4. 缓存感知路由要把请求发到缓存了它的实例，会造成热点，实例的缓存容量也有限；共享的 KV 池把"命中缓存"和"路由到哪里"解耦：随机路由下重复计算从约一半降到只剩新增的内容，本地缓存紧张时比缓存感知路由更稳。
    5. DeepSeek 开源的分布式文件系统：SSD 加 RDMA 的分离式架构，提供很高的聚合带宽。推理里它作为 KV 缓存最下面的一层，为长上下文和大量会话提供容量；前提是读回来比重算快。

## 为什么需要传输引擎

![图：KV Cache 的分层——越往下容量越大、带宽越低](../assets/figures/kv-tiers.svg){.aig-svg}

KV 传输和训练中的集合通信很不一样：

- **点对点、动态**：哪个 prefill 实例传给哪个 decode 实例，每个请求都不同；实例会随时加入、退出、故障。NCCL 的通信组是启动时建立的静态集合，成员变化就要重建，不适合；
- **介质多样**：源和目标可能是显存、CPU 内存，也可能是本机或远端的 SSD；
- **要用满多张网卡**：一台机器 8 张网卡，一个大传输要切片、分到多张网卡上并行；选哪张网卡要看 GPU 与网卡的 PCIe 拓扑（[互联](interconnect.md#gpudirect让数据不绕道-cpu)一章）；
- **小块多**：KV 按页存放，要合并、批量提交，避开网卡的消息速率上限（[RDMA](rdma.md#请求数与消息速率)一章）；
- **要能容错**：某张网卡或某条链路坏了，切换到其他路径重试。

传输引擎把这些封装成一个简单的接口：注册内存、按"目标 + 偏移"批量读写、查询完成状态。推理框架只需要说"把这几段 KV 写到那台机器的那几个位置"。

## Mooncake Transfer Engine

Mooncake 是一套以 KV Cache 为中心的分离式推理架构（最初为大规模在线对话服务设计），开源了其中的传输引擎和 KV 存储；SGLang、vLLM 的 PD 分离都可以用它作为传输后端。传输引擎的核心抽象：

- **segment**：一段已注册、可被远程访问的连续地址空间——可以是某台机器上的一块内存或显存（RAM segment），也可以是一块 NVMe 盘上的文件（通过 NVMe-oF 访问）。每个 segment 有名字和 ID；
- **元数据服务**：segment 的地址、`rkey`、所在机器的网卡信息登记在一个元数据服务里（etcd、Redis 或 HTTP 服务），其他节点按名字查到后就能直接访问——这就是上一章"带外交换地址和密钥"的工程化版本；
- **批量传输**：先申请一个批次（`allocateBatchID`），再提交一组请求（`submitTransfer`），每个请求是 `{操作: READ/WRITE, 本地地址, 目标 segment, 目标偏移, 长度}`，最后轮询每个请求的状态（`getTransferStatus`）；
- **拓扑感知与多网卡**：引擎读取本机的拓扑，为每段内存选择"最近"的网卡；大请求被切成多个分片，分散到多张网卡并行传输；某条路径出错时换一张网卡重试。

NVIDIA 的 **NIXL**（NVIDIA Inference Xfer Library）是同类的库，Dynamo 和 vLLM 的 NIXL connector 用的就是它：每个进程是一个 agent，注册显存、内存、文件等各类内存到不同的后端（UCX、GPUDirect Storage 等），agent 之间交换元数据后，用描述符列表发起批量传输，并可以附带通知。两者的抽象几乎一一对应：注册 → 交换元数据 → 批量读写 → 完成通知。

## 分布式 KV 存储

有了传输引擎，就可以把 KV 存到"别处"：

| 层级 | 介质 | 容量 | 典型实现 |
| --- | --- | --- | --- |
| L1 | 本卡显存 | 几十 GB | 推理引擎自己的 KV 池与前缀缓存 |
| L2 | 本机 CPU 内存 | 几百 GB～TB | SGLang HiCache、vLLM 的 CPU 卸载 |
| L3 | 远端机器的内存池 | 集群内存总和 | Mooncake Store、LMCache |
| L3 / L4 | SSD（本地或分布式文件系统） | TB～PB | 3FS、本地 NVMe |

**Mooncake Store** 把集群里许多机器的空闲内存（以及 SSD）组织成一个分布式的对象存储：一个 master 服务管理元数据、分配空间、决定淘汰；数据本身通过传输引擎直接在客户端和存放节点之间传输，不经过 master；热门对象可以有多个副本。**LMCache** 是以 KV 为中心的缓存层，支持本地内存、磁盘和远端服务器等多种后端，通过 vLLM 的 KV connector 接入。

**键**：各层之间、各实例之间要能认出"同一段 KV"，普遍的做法是沿用前缀缓存的**前缀链式哈希**：第 $i$ 块的键是 $\text{hash}(\text{第 } i-1 \text{ 块的键}, \text{第 } i \text{ 块的 token})$。这样一个块的键就代表了"从开头到这一块为止的整个前缀"：两个请求只要前缀相同，块键就相同；前缀里任何一个 token 不同，此后所有块的键都不同——而 KV 恰恰依赖整个前缀，所以这正是想要的语义。

下面的模拟验证了这一点，再看共享 KV 池在多轮对话里的作用。负载是 300 个会话、3 种系统提示词，每个会话 2～6 轮，每轮用户说 128 个 token、模型回答 256 个 token，各会话的轮次交错到达 4 个实例。比较两种路由（随机、缓存感知——同一个会话总是发到同一个实例）和有无共享 KV 池：

```python
import random
from collections import OrderedDict

BLOCK = 16


def block_keys(tokens):
    """前缀链式哈希：第 i 块的键 = hash(第 i-1 块的键, 第 i 块的 token)；只有完整的块才有键"""
    keys, h = [], 0
    for i in range(0, len(tokens) - len(tokens) % BLOCK, BLOCK):
        h = hash((h, tuple(tokens[i:i + BLOCK])))
        keys.append(h)
    return keys


class LRU:
    def __init__(self, cap):
        self.cap, self.d = cap, OrderedDict()

    def __contains__(self, k):
        if k in self.d:
            self.d.move_to_end(k)
            return True
        return False

    def add(self, keys):
        for k in keys:
            self.d[k] = True
            self.d.move_to_end(k)
        while len(self.d) > self.cap:
            self.d.popitem(last=False)


a = list(range(1000, 1048))                      # 三个完整的块
b = a[:40] + [7] + a[41:]                        # 第 41 个 token 不同
print("相同前缀的块键一致：", block_keys(a)[:2] == block_keys(b)[:2], "；改动之后的块全部不同：", block_keys(a)[2] != block_keys(b)[2])

# 多轮对话负载：3 种系统提示词（各 512 token）× 300 个会话，每轮追加 128 个用户 token 和 256 个回答 token
rng = random.Random(0)
systems = [[rng.randrange(50000) for _ in range(512)] for _ in range(3)]
convs = [{"hist": list(rng.choice(systems)), "turns": rng.randint(2, 6)} for _ in range(300)]
events = [c for c in convs for _ in range(c["turns"])]
rng.shuffle(events)                              # 各个会话的轮次交错到达


def simulate(n_inst, local_cap, pool_cap, sticky):
    rng = random.Random(1)
    local = [LRU(local_cap) for _ in range(n_inst)]
    pool = LRU(pool_cap) if pool_cap else None
    for c in convs:
        c["cur"] = list(c["hist"])
        c["home"] = rng.randrange(n_inst)
    total = computed = 0
    for c in events:
        c["cur"] += [rng.randrange(50000) for _ in range(128)]           # 用户的新消息
        inst = c["home"] if sticky else rng.randrange(n_inst)            # 缓存感知路由 vs 随机路由
        keys = block_keys(c["cur"])
        hit = 0
        for k in keys:                                                   # 从头开始找最长的命中前缀：本地没有就去共享池
            if k in local[inst] or (pool is not None and k in pool):
                hit += 1
            else:
                break
        total += len(c["cur"])
        computed += len(c["cur"]) - hit * BLOCK
        local[inst].add(keys)
        if pool is not None:
            pool.add(keys)
        c["cur"] += [rng.randrange(50000) for _ in range(256)]           # 模型的回答，下一轮成为历史
    return computed / total


print("需要重新计算的 prefill token 占比（每个实例的本地缓存 8000 块 / 2000 块）：")
for name, sticky, pool_cap in [("随机路由", False, 0), ("随机路由 + 共享 KV 池", False, 400_000),
                               ("缓存感知路由", True, 0), ("缓存感知路由 + 共享 KV 池", True, 400_000)]:
    print(f"  {name}：" + " / ".join(f"{simulate(4, cap, pool_cap, sticky):.0%}" for cap in (8000, 2000)))
```

```text title="输出"
相同前缀的块键一致： True ；改动之后的块全部不同： True
需要重新计算的 prefill token 占比（每个实例的本地缓存 8000 块 / 2000 块）：
  随机路由：48% / 56%
  随机路由 + 共享 KV 池：24% / 24%
  缓存感知路由：25% / 42%
  缓存感知路由 + 共享 KV 池：24% / 24%
```

24% 是这个负载的下限——每一轮新增的内容（用户的新消息，以及上一轮回答中没凑满一块的尾部）总要算。读这张表：

- **随机路由、只有本地缓存**时，一半的 prefill 在重复计算：会话的下一轮大概率落到别的实例上，只能命中系统提示词；
- **缓存感知路由**在本地缓存够大时几乎达到下限，但本地缓存一紧张（2000 块），长会话的历史被淘汰，重复计算回升到 42%；
- **共享 KV 池**在两种路由、两种本地容量下都达到了下限：它把"命中缓存"和"路由到哪个实例"解耦了。

这就是分布式 KV 池的价值：路由器可以更自由地做负载均衡（不必为了命中缓存把请求压在某个繁忙的实例上）；实例扩缩容、重启之后缓存不会丢；缓存容量从"单卡显存 + 单机内存"扩大到"整个集群的内存和 SSD"。代价是从远端读 KV 要花时间——读回来比重新算快才划算，这个判断见 [KV 分层缓存与卸载](../distributed/kv-offload.md#读回来还是重算)：长前缀、大模型时读回来远比重算便宜，短前缀、小模型时未必。

## 3FS：给 AI 负载设计的分布式文件系统

**3FS**（Fire-Flyer File System）是 DeepSeek 开源的分布式文件系统，专为 AI 训练和推理的负载设计：

- **分离式架构**：大量 SSD 和 RDMA 网络组成存储集群，计算节点通过 RDMA 直接读写，不关心数据在哪台存储机器上——把几千块 SSD 的带宽聚合起来；
- **强一致性**：数据用 CRAQ（Chain Replication with Apportioned Queries，链式复制，读可以分摊到链上的任意副本）复制；元数据服务无状态，元数据存在事务型键值库 FoundationDB 里；
- **用途**：训练数据的随机读取、checkpoint 的并行读写，以及推理的 **KVCache**——把 KV 缓存放到 SSD 上，用比内存便宜得多的容量换取更高的上下文缓存命中率。官方公布的数据是：180 个存储节点的集群，聚合读带宽约 6.6 TiB/s。

SGLang 的 HiCache 可以把 3FS 作为第三层存储后端。SSD 层的意义在于**容量**：一个 70B 模型每个 token 的 KV 是 320 KB，一百万个 token 就是 320 GB；要为成千上万个会话、长文档、智能体的长上下文保留缓存，只有 SSD 的容量和成本撑得住。

## 设计取舍

- **推还是拉、写回还是写穿**：PD 分离里 prefill 逐层推送能和计算重叠；分层缓存里写穿（算完就异步写一份到下层）便于共享，写回（淘汰时才写）写入量少；
- **粒度**：块越小，前缀共享越细，但元数据和请求数越多；存储层通常用比推理引擎更大的块（或把多个块打包成一个对象）；
- **一致性要求低**：KV 缓存是"丢了可以重算"的数据，所以存储可以激进地淘汰、不必强一致（3FS 的强一致性主要是为训练数据和 checkpoint 服务的）；
- **布局转换**：存储里的 KV 布局要和读取它的实例匹配（TP 度数、页大小、数据类型），否则读回来还要转换（见 [PD 分离](../distributed/pd-disagg.md#布局转换)）；
- **安全与隔离**：不同租户的 KV 不能互相命中。前缀链式哈希里通常会混入租户、模型版本、LoRA 适配器等信息，否则"相同前缀"可能来自不同的模型。

!!! source "源码对照"
    - **Mooncake**（`kvcache-ai/Mooncake`）：`mooncake-transfer-engine/`（传输引擎，C++，含 RDMA、TCP、NVMe-oF 等传输方式）、`mooncake-store/`（分布式 KV 存储）。
    - **NIXL**（`ai-dynamo/nixl`）：agent、内存注册与各后端插件；vLLM 的 `kv_connector/v1/nixl/` 是一个完整的接入示例（拉取式和推送式两种），同一目录下还有 `mooncake/`、`hf3fs/`、`lmcache_connector.py` 等。
    - **SGLang**：PD 传输在 `srt/disaggregation/`（`mooncake/`、`nixl/` 等子目录）；分层缓存的第三层存储在 `srt/mem_cache/storage/`。
    - **3FS**（`deepseek-ai/3FS`）：文件系统本体，以及用于 KVCache 的客户端接口。

!!! interview "怎么讲清楚"
    谈到 PD 分离或多轮对话的缓存，可以主动展开这一层：传输引擎负责"注册内存、带外交换元数据、批量读写、多网卡与拓扑感知"；分布式 KV 存储用前缀链式哈希做全局键，把命中缓存和路由解耦；SSD 层（3FS 这类）提供容量。给出"随机路由一半的 prefill 是重复计算，共享池能压到只算新增内容"这类量化结论，再补一句"读回来要比重算快才划算"，就把这一层讲完整了。

## 练习

**1. 为什么不用 NCCL 传 KV？** 列出至少三个理由。

??? success "参考答案"
    ① NCCL 的通信组是静态的，实例加入、退出、故障都要重建通信组，而 PD 分离中的配对关系每个请求都在变；② NCCL 面向集合通信和显存之间的传输，不直接支持内存、SSD 等多种介质；③ NCCL 的收发需要两端同时调用（双边语义），而 KV 推送更适合单边写、对端无感知；④ 容错：NCCL 出错通常意味着整个通信组失效，传输引擎可以换一条路径重试单个请求。实际上 vLLM 早期也有基于 NCCL 的 PD 传输实现，后来的主流方案都换成了 NIXL、Mooncake 这类传输引擎。

**2. 块键里要混入什么？** 一个多租户的推理平台，同时服务同一个基座模型的多个 LoRA 版本。设计 KV 块的全局键时，除了 token，还要考虑什么？

??? success "参考答案"
    至少要包括：模型与权重版本（换了权重，KV 就不同）、LoRA 适配器的 ID（LoRA 作用在 Q/K/V 投影上时 KV 不同）、KV 的数据类型和量化方式、影响 KV 的其他输入（多模态输入的图片哈希、位置编码的偏移等），以及租户隔离的需要（不希望不同租户通过缓存命中时间推测出彼此的提示词）。一个常见的做法是把这些信息放进第一块的哈希种子里，之后的链式哈希自然继承。

## 小结

- [x] KV 传输是点对点、动态、多介质、小块多的，NCCL 不合适；传输引擎提供"注册内存 → 交换元数据 → 批量读写 → 完成通知"的接口，并负责多网卡、拓扑感知和容错。
- [x] Mooncake Transfer Engine 的核心是 segment、元数据服务和批量传输；NIXL 用 agent 和后端插件提供同样的能力。
- [x] 分布式 KV 存储（Mooncake Store、LMCache、3FS）以前缀链式哈希为全局键，把缓存容量扩大到集群内存和 SSD。
- [x] 共享 KV 池把命中缓存和路由解耦：随机路由下重复计算从约一半降到只剩新增内容；本地缓存紧张时比缓存感知路由更稳。
- [x] 3FS 用 SSD + RDMA 的分离式架构提供高聚合带宽，SSD 层为长上下文和大量会话提供容量；读回来要比重算快才划算。
