# 多机多卡与分离式部署

<p class="lead">一个 671B 的 MoE 模型要 16 张卡、跨两台机器，这时"一个副本 = 一个 Pod"的假设就不成立了：一个副本是一组必须同时存在、同时就绪、同时重启的 Pod。再往前一步，PD 分离让 prefill 和 decode 变成两个独立伸缩的池子，中间还要传 KV。这一章讲清楚 Kubernetes 上表达这些形态的机制——LeaderWorkerSet、Headless Service 的成员发现、xPyD 的配比与路由——以及 Gateway API Inference Extension 这套正在成形的标准。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. 为什么多机多卡的推理实例不能用普通 Deployment？
    2. LeaderWorkerSet 解决了什么问题？扩缩容的单位是什么？
    3. 一组里的一个 worker 挂了，为什么通常要整组重启？
    4. PD 分离在 Kubernetes 上怎么表达？配比怎么定？
    5. 推理网关和普通七层负载均衡的区别在哪？

??? success "自测参考答案（先自己答，再展开对照）"
    1. Deployment 的副本是**彼此独立、可以单独增删**的 Pod，而多机实例的成员要互相知道对方地址、一起建立通信组、一起启动一起退出。用 Deployment 表达会出现"扩容扩出半个实例""一个成员重启后通信组坏掉"这类问题。
    2. **LeaderWorkerSet（LWS）** 把"1 个 leader + N-1 个 worker"打包成一个副本单位：一起创建、按组编号、组内有稳定的域名、支持整组重启、扩缩容以组为单位。它是 Kubernetes SIG 为多机推理专门做的 CRD，vLLM、SGLang 的多机部署都推荐用它。
    3. 因为集合通信的成员是启动时固定的：一个进程没了，其余进程会卡在 all-reduce 上直到超时；即使重启了那一个，它也无法重新加入已经建好的通信组。所以策略通常是 `RecreateGroupOnHostFailure`——整组重启，代价是一次故障影响 N 个 Pod。
    4. 两个独立的工作负载（各自是 Deployment 或 LWS）加一个全局调度器/网关：prefill 池、decode 池、以及负责"把请求先送到 P、再把 KV 交给 D"的路由。配比（xPyD）由两边的处理能力决定，本章的模型给出随输入/输出长度变化的理想比值，生产上再按实测队列长度动态调。
    5. 普通七层负载均衡按连接或请求数轮询；推理网关要理解**请求的代价**（输入长度、预估输出长度）和**实例的状态**（队列深度、KV 缓存命中、正在跑的 batch），所以需要模型感知的路由。Gateway API Inference Extension（InferencePool / InferenceModel）就是把这套语义标准化。

## 多机实例：LeaderWorkerSet

```yaml
apiVersion: leaderworkerset.x-k8s.io/v1
kind: LeaderWorkerSet
metadata: {name: vllm}
spec:
  replicas: 2                          # 两个副本 = 两组
  leaderWorkerTemplate:
    size: 4                            # 每组 4 个 Pod（1 leader + 3 worker）
    restartPolicy: RecreateGroupOnHostFailure    # 一个成员挂了，整组重启
    leaderTemplate:
      spec:
        containers:
          - name: leader
            image: vllm/vllm-openai:v0.11.0
            command: ["sh", "-c", "ray start --head --block & vllm serve $MODEL --tensor-parallel-size 8 --pipeline-parallel-size 4"]
            resources: {limits: {nvidia.com/gpu: 8}}
            ports: [{containerPort: 8000}]
    workerTemplate:
      spec:
        containers:
          - name: worker
            image: vllm/vllm-openai:v0.11.0
            command: ["sh", "-c", "ray start --address=$LWS_LEADER_ADDRESS:6379 --block"]
            resources: {limits: {nvidia.com/gpu: 8}}
```

```python title="lws.py"
"""多机多卡的副本组（LeaderWorkerSet 的模型）与 PD 分离的配比。

一个"副本"不再是一个 Pod，而是"1 个 leader + (size-1) 个 worker"的一组 Pod：
一起创建、一起就绪、一起重启、一起被扩缩容。
"""


def group_pods(name, replicas, size):
    """返回每个副本组的 Pod 名字：<name>-<组号> 是 leader，<name>-<组号>-<序号> 是 worker"""
    out = []
    for g in range(replicas):
        pods = [f"{name}-{g}"] + [f"{name}-{g}-{i}" for i in range(1, size)]
        out.append(pods)
    return out


def gpus_needed(replicas, size, gpus_per_pod):
    return replicas * size * gpus_per_pod


def restart_blast_radius(size, policy):
    """一个 worker 挂掉时受影响的 Pod 数：整组重启 vs 只重启这一个"""
    return size if policy == "RecreateGroupOnHostFailure" else 1


def pd_ratio(prefill_ms, decode_ms_per_token, out_tokens, prefill_gpus, decode_gpus,
             decode_batch=64, prefill_batch=1):
    """PD 分离的配比：让两边的"每秒能处理多少个请求"匹配。

    prefill 实例受算力限制，一次基本只服务一个请求，每个请求占用 prefill_ms；
    decode 实例受带宽限制，一步同时推进 decode_batch 个请求，所以每个请求的实际占用是
    out_tokens * decode_ms_per_token / decode_batch。两边每秒的处理能力相等时配比最优。
    """
    per_request_prefill = prefill_ms / prefill_batch
    per_request_decode = out_tokens * decode_ms_per_token / decode_batch
    ideal = per_request_decode / per_request_prefill
    actual = decode_gpus / prefill_gpus
    bottleneck = "prefill" if actual > ideal else "decode"
    return round(ideal, 2), round(actual, 2), bottleneck


def kv_transfer_ms(kv_bytes, link_gbs):
    """PD 分离要把 KV 从 prefill 实例搬到 decode 实例"""
    return kv_bytes / (link_gbs * 1e9) * 1000
```

```python
from lws import group_pods, gpus_needed, kv_transfer_ms, pd_ratio, restart_blast_radius

print("LeaderWorkerSet：replicas=2、size=4（每组 1 leader + 3 worker），每个 Pod 8 张卡")
for i, pods in enumerate(group_pods("vllm", replicas=2, size=4)):
    print(f"  第 {i} 组：leader={pods[0]}，worker={pods[1:]}")
print("  总卡数：", gpus_needed(2, 4, 8), "张（扩缩容以'组'为单位，不会扩出半个实例）")

print("\n一个 worker 挂掉时的影响面：")
for policy in ("RecreateGroupOnHostFailure", "Default"):
    print(f"  {policy:28s} 受影响 {restart_blast_radius(4, policy)} 个 Pod")
print("  多机推理必须整组重启：通信组的成员变了，剩下的进程会卡在集合通信上")

print("\nPD 分离的配比：理想的 decode : prefill 实例数之比")
print("（prefill 每 1K 输入 75 ms、一次一个请求；decode 每 token 20 ms、一步推进 64 个请求）")
print("  输入 \\ 输出      100 token    500 token   2000 token")
for in_tokens in (1024, 4096, 16384):
    prefill_ms = in_tokens / 1024 * 75
    row = ""
    for out_tokens in (100, 500, 2000):
        ideal, _, _ = pd_ratio(prefill_ms, 20, out_tokens, 1, 1)
        row += f"{ideal:11.2f}  "
    print(f"  {in_tokens:5d} token  {row}")
print("  输出越长、输入越短，就越需要更多 decode 实例；反过来长上下文的场景 prefill 更吃紧。")
print("  实际配比还要按实测的队列长度动态调整（xPyD 的 x、y 通常在 1:1 到 1:4 之间）。")

print("\nKV 传输的代价（一个 4K 上下文的请求，GQA 8 头、128 KB/token）：")
kv_bytes = 4096 * 128 * 1024
for name, gbs in (("同机 NVLink", 450), ("机间 400G RDMA", 50), ("机间 TCP", 5)):
    print(f"  {name:16s} {kv_transfer_ms(kv_bytes, gbs):7.1f} ms")
print("  这段时间要么和 prefill 重叠（逐层传），要么直接算进 TTFT")
```

```text title="输出"
LeaderWorkerSet：replicas=2、size=4（每组 1 leader + 3 worker），每个 Pod 8 张卡
  第 0 组：leader=vllm-0，worker=['vllm-0-1', 'vllm-0-2', 'vllm-0-3']
  第 1 组：leader=vllm-1，worker=['vllm-1-1', 'vllm-1-2', 'vllm-1-3']
  总卡数： 64 张（扩缩容以'组'为单位，不会扩出半个实例）

一个 worker 挂掉时的影响面：
  RecreateGroupOnHostFailure   受影响 4 个 Pod
  Default                      受影响 1 个 Pod
  多机推理必须整组重启：通信组的成员变了，剩下的进程会卡在集合通信上

PD 分离的配比：理想的 decode : prefill 实例数之比
（prefill 每 1K 输入 75 ms、一次一个请求；decode 每 token 20 ms、一步推进 64 个请求）
  输入 \ 输出      100 token    500 token   2000 token
   1024 token         0.42         2.08         8.33  
   4096 token         0.10         0.52         2.08  
  16384 token         0.03         0.13         0.52  
  输出越长、输入越短，就越需要更多 decode 实例；反过来长上下文的场景 prefill 更吃紧。
  实际配比还要按实测的队列长度动态调整（xPyD 的 x、y 通常在 1:1 到 1:4 之间）。

KV 传输的代价（一个 4K 上下文的请求，GQA 8 头、128 KB/token）：
  同机 NVLink            1.2 ms
  机间 400G RDMA        10.7 ms
  机间 TCP             107.4 ms
  这段时间要么和 prefill 重叠（逐层传），要么直接算进 TTFT
```

几个要点：

- **组内成员靠稳定域名互相找到**：LWS 会建一个 Headless Service，组内 Pod 的域名形如 `vllm-0.vllm.default.svc.cluster.local`，leader 的地址通过 `LWS_LEADER_ADDRESS` 注入 worker。多机推理框架（Ray、`torch.distributed`、SGLang 的 `--dist-init-addr`）都需要这样一个稳定的会合点。
- **扩缩容以组为单位**：`replicas: 2` 表示两个完整实例，不会出现"扩出半个实例占着卡等另一半"的情况。要真正保证一组的卡能同时拿到，还要配合上一章的 gang 调度（Kueue / Volcano）。
- **整组重启的影响面**：`RecreateGroupOnHostFailure` 下一个 worker 挂掉会重启 4 个 Pod。这是对的（通信组坏了），但意味着**故障影响面随并行规模放大**，所以多机实例的可用性要靠"多组副本"来保证，而不是靠"组内自愈"。
- **拓扑要对齐**：一组的 Pod 最好落在同一个机架/同一个 NVLink 域内，用 `topologySpreadConstraints` 或 LWS 的 `subGroupPolicy`、节点标签来约束，否则跨交换机的 all-reduce 会拖慢每一步。

## 一个请求怎么找到实例：Service 与网关

| 机制 | 作用 | 推理场景的注意点 |
| --- | --- | --- |
| **ClusterIP Service** | 四层负载均衡（iptables/IPVS 轮询） | 对长连接和流式响应不均衡（见[计算机基础：负载均衡](root://cs/net/load-balance/)） |
| **Headless Service**（`clusterIP: None`） | 直接解析出所有 Pod IP | 多机成员发现、以及让网关自己做路由 |
| **Ingress / Gateway API** | 七层路由、TLS、灰度 | 要确认支持流式响应（别缓冲） |
| **InferencePool / InferenceModel**（Gateway API Inference Extension） | 把"一池推理实例"和"模型名 → 后端"抽象成标准对象 | 网关按队列深度、KV 命中、LoRA 亲和路由 |

推理流量的特殊性决定了网关不能只做轮询：

- **请求代价差几十倍**：一个 200 token 的输入和一个 128K 的输入占用完全不同；
- **实例状态差别很大**：有的实例队列里还排着十几个请求，有的刚空出来；
- **缓存亲和有巨大收益**：同一个会话/同一个系统提示词发给同一个实例，能省掉整段 prefill（见[前缀缓存](../engine/prefix-cache.md)）。

所以社区把这套语义标准化成了 **Gateway API Inference Extension**：`InferencePool` 描述一组同构的推理实例，`InferenceModel` 把模型名（含 LoRA 适配器）映射到池子并声明优先级；网关侧的 Endpoint Picker 按实例上报的指标（队列长度、KV 使用率、已加载的适配器）选实例。llm-d、vLLM production stack、SGLang Model Gateway 都在这个方向上。

## PD 分离在 Kubernetes 上的形状

PD 分离把一个请求拆成两段：prefill 实例算完整个提示词、把 KV 交给 decode 实例，decode 实例负责逐 token 生成（原理见[PD 分离与 KV 传输](../distributed/pd-disagg.md)）。落到 Kubernetes 上是三组对象：

```text
┌──────────────┐   1. 请求      ┌───────────────┐
│   网关/调度   │ ────────────▶ │  prefill 池   │  (Deployment 或 LWS，算力型卡)
│  (EPP/路由)   │ ◀──────────── │               │
└──────┬───────┘   2. KV 就绪   └───────┬───────┘
       │ 3. 转交                        │ KV 传输（RDMA / NVLink / 共享存储）
       ▼                                ▼
┌──────────────┐                ┌───────────────┐
│  decode 池    │ ◀──────────── │  KV 传输引擎   │  (NIXL / Mooncake / LMCache)
└──────────────┘                └───────────────┘
```

上面程序的后两段给出两个关键的账：

- **配比随输入输出长度变化**：输入 1K、输出 2000 时理想的 decode:prefill 是 8.3 : 1，而输入 16K、输出 100 时只要 0.03 : 1（prefill 才是瓶颈）。所以 xPyD 的配比不是拍脑袋定的常数，要按业务的长度分布算，再按实测队列长度动态调整（见[分离式架构的全局调度](../frontier/disagg-sched.md)）。
- **KV 传输的代价决定部署边界**：4K 上下文的 KV 约 512 MB，走同机 NVLink 1.2 ms、机间 RDMA 10.7 ms、机间 TCP 107 ms。所以 PD 分离几乎必须配 RDMA（或者把 P 和 D 放在同一台机器上按 SM 切分，见 [PD 复用](../frontier/pd-multiplex.md)），而且要逐层传输与 prefill 计算重叠。

部署上的具体建议：

- **两个池子分别伸缩**：prefill 池按"输入 token 速率"扩容，decode 池按"并发请求数/队列长度"扩容，两个 HPA/ScaledObject 分开配。
- **异构卡按长处分工**：算力强的卡（H100）做 prefill，带宽高、算力弱的卡（H20）做 decode，这正是[硬件速查](../career/hardware.md)里屋脊点那张表的用处。
- **网络要先打通**：RDMA 需要 SR-IOV 或 RoCE 的 CNI 插件、`IPC_LOCK` 权限、以及足够的 hugepages；这部分往往比推理配置本身更难调。
- **故障域**：prefill 挂了可以重试（请求还没开始生成），decode 挂了正在生成的回答就断了——decode 池的可用性要求更高。

## 有状态的那些东西

推理服务"大部分无状态"，但有几样东西是有状态的，部署时要单独考虑：

| 状态 | 放哪 | 说明 |
| --- | --- | --- |
| 模型权重 | PVC（ReadOnlyMany）或本地 NVMe + 预热 | 别每次都从对象存储拉；用 initContainer 或 DaemonSet 预热到节点本地盘 |
| KV Cache（跨请求复用） | 实例内存 + 外部 KV 存储 | Mooncake Store / LMCache；跨实例复用要配缓存感知路由 |
| LoRA 适配器 | ConfigMap 太小，用 PVC 或对象存储 | 动态加载，配合适配器亲和路由（见[多 LoRA 服务](../ops/multi-lora.md)） |
| 会话与对话历史 | 外部存储（Redis 等） | 不要放在实例里，否则实例一换用户就丢上下文 |
| 指标与日志 | Prometheus / 日志系统 | Pod 会被删，落盘的东西要及时送出去 |

一个常见误区是用 **StatefulSet** 部署推理服务。StatefulSet 提供稳定的名字和存储，代价是**按序启停**（滚动更新时一个一个来，发布很慢）。推理服务真正需要的是"一组 Pod 一起管理"，这正是 LWS 做的事；除非确实需要每个实例绑定独立的持久卷，否则不要用 StatefulSet。

!!! interview "面试怎么答"
    被问多机部署：先说 Deployment 不合适的原因——副本成员要互相发现、一起就绪、一起重启，扩缩容要以"组"为单位。再讲 LeaderWorkerSet：1 leader + N-1 worker 打包成一个副本，Headless Service 提供稳定域名当会合点，`RecreateGroupOnHostFailure` 保证通信组一致，代价是故障影响面等于组大小，所以可用性靠多组副本。配合 gang 调度保证整组资源同时到位，用拓扑约束让一组落在同一个 NVLink 域。再讲 PD 分离：两个独立伸缩的池子加一个模型感知的网关，配比按输入输出长度算（输出越长越需要 decode 实例），KV 传输几乎必须走 RDMA（4K 上下文的 KV 约 512 MB，RDMA 10 ms、TCP 100 ms），异构卡按屋脊点分工。最后提一句 Gateway API Inference Extension 把 InferencePool / InferenceModel 和按队列深度、KV 命中路由标准化了。

## 练习

**1. 选部署形态。** 三个服务：（a）0.6B 小模型，单卡，20 个副本；（b）70B 模型，TP=4 单机；（c）671B MoE，TP=8 + PP=2 跨两机。各自用什么对象部署？

??? success "参考答案"
    （a）Deployment，单 Pod 单卡，配 HPA 按队列长度扩缩；（b）Deployment 也可以——一个 Pod 申请 4 张卡，进程内部做 TP，不涉及跨 Pod 通信；（c）LeaderWorkerSet，`size: 2`（每个 Pod 8 张卡，两个 Pod 组成一个实例），配 gang 调度和拓扑约束，`RecreateGroupOnHostFailure`。

    判断标准很简单：**跨 Pod 的通信组** → LWS；**单 Pod 内部多卡** → 普通 Deployment。

**2. 算 PD 配比。** 业务的输入平均 2K token、输出平均 300 token。prefill 每 1K 输入 75 ms，decode 每 token 20 ms、一步推进 64 个请求。理想的 P:D 配比是多少？如果引入前缀缓存让 60% 的输入被命中呢？

??? success "参考答案"
    prefill 每请求 150 ms；decode 每请求 300 × 20 / 64 = 93.75 ms。理想 decode:prefill = 93.75 / 150 = 0.63，即大约 3 个 prefill 配 2 个 decode。

    前缀缓存命中 60% 后，prefill 的有效工作量降到 60 ms，比值变成 93.75 / 60 = 1.56，即 2 个 prefill 配 3 个 decode——**缓存命中率直接改变配比**。这也是为什么 xPyD 要按实测动态调整，而不是写死。

**3. 一个 worker 挂了。** LWS 的一组有 4 个 Pod，其中一个节点故障。写出 `RecreateGroupOnHostFailure` 和不整组重启两种策略下会发生什么，以及为什么推理要选前者。

??? success "参考答案"
    不整组重启：那一个 Pod 被重建，但其余 3 个进程还在原来的通信组里等它——新进程无法加入已建立的 NCCL 通信组，3 个老进程会卡在集合通信上直到超时（几分钟），期间这个实例完全不可用，而且可能一直卡着不恢复。

    整组重启：4 个 Pod 一起删掉重建，几分钟后（要重新加载权重）恢复服务。虽然影响面是 4 个 Pod，但**状态是确定的**。推理服务必须选后者，并靠多组副本 + PDB 保证整体容量。

**4. 网关该看什么指标。** 你要给一个 20 实例的池子写路由逻辑。列出三个必须采集的实例侧指标，并说明各自怎么用。

??? success "参考答案"
    （1）**待处理请求数 / 队列深度**：直接反映忙闲，作为主要的负载指标（比连接数准得多）；（2）**KV Cache 使用率**：接近满时这个实例马上要开始抢占或拒绝，要提前避开；（3）**前缀缓存命中信息**（实例缓存了哪些前缀，或路由器维护的近似视图）：用来做缓存亲和路由，命中能省掉整段 prefill。

    组合方式：先按缓存亲和挑候选，再在候选里按队列深度选最空的，并设负载上限避免热点（有界负载的一致性哈希，见[计算机基础：一致性哈希](root://cs/dist/hash-shard/)）。另外要采集 TTFT/TPOT 分位数用于告警和 HPA。

## 小结

- [x] 跨 Pod 的通信组用 LeaderWorkerSet：一组一起创建、一起就绪、整组重启，扩缩容以组为单位。
- [x] 组内靠 Headless Service 的稳定域名会合；配 gang 调度保证整组资源同时到位；拓扑约束让一组落在同一 NVLink 域。
- [x] 整组重启的影响面等于组大小，可用性靠多组副本而不是组内自愈。
- [x] PD 分离 = 两个独立伸缩的池子 + 模型感知网关；配比随输入输出长度和缓存命中率变化，KV 传输几乎必须走 RDMA。
- [x] 推理网关要按请求代价和实例状态路由，Gateway API Inference Extension 正在把这套语义标准化。
- [x] 权重、KV、适配器、会话各有各的去处；别用 StatefulSet 部署无状态推理服务。
