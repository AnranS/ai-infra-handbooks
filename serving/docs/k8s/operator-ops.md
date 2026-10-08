# Operator、配额与排障

<p class="lead">平台做到一定规模，一定会出现"每上线一个模型都要拼十几个 YAML"的问题——这时就该把这套模式固化成一个自定义对象和一个控制器，让业务方只写 `model: Qwen3-0.6B, replicas: 3`。这一章从零做一个 InferenceService CRD（在真实集群上创建过），把控制器的 reconcile 逻辑写成能跑的代码；再讲多租户要用的配额与优先级；最后把前面几章的排障线索汇成一张可以照着走的表。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. CRD 和 Operator 分别是什么？为什么推理平台需要它们？
    2. 写控制器时，`spec` 和 `status` 的分工是什么？为什么要用 `status` 子资源？
    3. 控制器怎么知道"哪些子对象是我创建的"？
    4. ResourceQuota 和 LimitRange 各管什么？
    5. 一个 Pod 处于 `Pending`、`CrashLoopBackOff`、`Running 但 0/1` 时，分别先看什么？

??? success "自测参考答案（先自己答，再展开对照）"
    1. **CRD**（CustomResourceDefinition）向 API server 注册一种新的对象类型，之后就能像操作内置对象一样 `kubectl get/apply` 它；**Operator** 是给这种对象配的控制器，负责把它翻译成 Deployment、Service、HPA 等实际资源。推理平台需要它们，是因为"上线一个模型"背后有一整套固定模式（部署 + 服务 + 自动扩缩 + 监控 + 网关注册），把模式沉淀成一个对象，业务方就只需要填几个字段。
    2. `spec` 是**用户写的期望**，`status` 是**控制器写的实际情况**。用 `status` 子资源（`subresources: {status: {}}`）把两者的更新权限分开：控制器更新 status 不会误改 spec，用户改 spec 也不会覆盖 status，还能避免两边互相触发更新风暴。
    3. 靠 **ownerReferences**：创建子对象时把自定义资源写成它的 owner。这样一来，删除自定义资源时子对象会被**级联删除**（垃圾回收器负责），控制器也能通过 owner 反查自己管的对象。
    4. **ResourceQuota** 是命名空间级别的总量上限（一共能用多少 CPU、内存、GPU，能建多少个 Deployment）；**LimitRange** 是单个 Pod/容器级别的默认值与上下限（没写 requests 时给个默认值、限制单个容器最多要多少）。开了 ResourceQuota 之后，**不写 requests 的 Pod 会被直接拒绝**，所以两者通常一起用。
    5. `Pending` 看 `describe` 里的 `FailedScheduling`（调度问题）；`CrashLoopBackOff` 看 `logs --previous` 和容器的 `Last State`（启动或探针问题）；`Running 但 0/1` 看就绪探针的配置和 `/health` 的实际返回（服务还没准备好，或探针配错）。

## 自定义资源：让上线一个模型只写五行

![图：Kubernetes 的控制器模式——声明期望状态，控制器不断比较期望与实际](../assets/figures/k8s-reconcile.svg){.aig-svg}

```yaml title="crd.yaml"
apiVersion: apiextensions.k8s.io/v1
kind: CustomResourceDefinition
metadata:
  name: inferenceservices.ai.example.com
spec:
  group: ai.example.com
  scope: Namespaced
  names: {plural: inferenceservices, singular: inferenceservice, kind: InferenceService, shortNames: [isvc]}
  versions:
    - name: v1alpha1
      served: true
      storage: true
      subresources: {status: {}}          # 把 status 分成独立的子资源
      schema:
        openAPIV3Schema:                  # API server 会按这个 schema 校验用户写的对象
          type: object
          properties:
            spec:
              type: object
              required: [model, replicas]
              properties:
                model: {type: string}
                replicas: {type: integer, minimum: 0}
                gpusPerReplica: {type: integer, default: 1}
            status:
              type: object
              properties:
                readyReplicas: {type: integer}
                phase: {type: string}
      additionalPrinterColumns:           # kubectl get 时多显示几列
        - {name: Model, type: string, jsonPath: .spec.model}
        - {name: Replicas, type: integer, jsonPath: .spec.replicas}
        - {name: Ready, type: integer, jsonPath: .status.readyReplicas}
        - {name: Phase, type: string, jsonPath: .status.phase}
```

注册之后，业务方上线一个模型只需要：

```yaml
apiVersion: ai.example.com/v1alpha1
kind: InferenceService
metadata: {name: qwen3}
spec:
  model: Qwen/Qwen3-0.6B
  replicas: 3
  gpusPerReplica: 1
```

```bash
kubectl apply -f crd.yaml
kubectl apply -f isvc.yaml
kubectl get isvc
```

```text title="输出（本机示例）"
customresourcedefinition.apiextensions.k8s.io/inferenceservices.ai.example.com created
inferenceservice.ai.example.com/qwen3 created

NAME    MODEL             REPLICAS   READY   PHASE
qwen3   Qwen/Qwen3-0.6B   3
```

`READY` 和 `PHASE` 是空的——因为还没有控制器去填它们。CRD 只是"定义了一种对象"，真正干活的是 Operator。

## 控制器的 reconcile

```python title="isvc_controller.py"
"""一个 InferenceService Operator 的 reconcile 逻辑（与 controller-runtime 的 Reconcile 同构）。

真实 Operator 里这段函数的输入是 API server 的对象，输出是对子对象的增删改；
这里把集群抽象成字典，方便把"该建什么、该改什么、状态怎么写回"这套逻辑单独跑通。
"""


def desired_children(spec):
    """从自定义资源的 spec 推导出应该存在的子对象"""
    name, replicas = spec["name"], spec["replicas"]
    return {
        f"deploy/{name}": {"replicas": replicas, "image": spec["image"],
                           "gpus": spec["gpusPerReplica"]},
        f"svc/{name}": {"selector": name, "port": 8000},
        f"hpa/{name}": {"min": max(1, replicas // 2), "max": replicas * 3,
                        "metric": "num_requests_waiting"},
    }


def reconcile(spec, actual):
    """比较期望与实际，返回要执行的动作列表和新的 status（幂等：只看当前状态）"""
    want = desired_children(spec)
    actions = []
    for key, cfg in want.items():
        if key not in actual:
            actions.append(("create", key, cfg))
        elif any(actual[key].get(k) != v for k, v in cfg.items()):
            actions.append(("update", key, cfg))     # 只比较自己管的字段，别碰子控制器写的状态
    for key in actual:                                   # 多余的子对象要删掉（比如副本数改成 0）
        if key not in want:
            actions.append(("delete", key, None))
    ready = actual.get(f"deploy/{spec['name']}", {}).get("ready", 0)
    phase = "Ready" if ready >= spec["replicas"] and not actions else \
            "Progressing" if actions or ready < spec["replicas"] else "Unknown"
    return actions, {"readyReplicas": ready, "phase": phase}


def apply(actions, actual):
    """把动作施加到"集群"上，模拟子控制器随后把 Pod 拉起来"""
    for op, key, cfg in actions:
        if op == "delete":
            actual.pop(key, None)
        else:
            actual[key] = dict(cfg)
    for key, obj in actual.items():
        if key.startswith("deploy/"):
            obj["ready"] = obj["replicas"]               # 假设 Pod 都能就绪
    return actual
```

```python
from isvc_controller import apply, desired_children, reconcile

spec = {"name": "qwen3", "replicas": 3, "image": "vllm/vllm-openai:v0.11.0", "gpusPerReplica": 1}
actual = {}

print("第一次 reconcile（集群里什么都没有）：")
actions, status = reconcile(spec, actual)
for op, key, cfg in actions:
    print(f"  {op:6s} {key}")
print("  status：", status)
actual = apply(actions, actual)

print("\n第二次 reconcile（已经收敛）：")
actions, status = reconcile(spec, actual)
print("  动作：", actions or "无")
print("  status：", status)

print("\n把 replicas 改成 6 之后：")
spec["replicas"] = 6
actions, status = reconcile(spec, actual)
for op, key, cfg in actions:
    print(f"  {op:6s} {key} -> {cfg}")
actual = apply(actions, actual)
print("  再跑一次：", reconcile(spec, actual)[1])

print("\n有人手动删掉了 Service：")
actual.pop("svc/qwen3")
actions, _ = reconcile(spec, actual)
print("  动作：", [(op, key) for op, key, _ in actions], "（自愈）")
```

```text title="输出"
第一次 reconcile（集群里什么都没有）：
  create deploy/qwen3
  create svc/qwen3
  create hpa/qwen3
  status： {'readyReplicas': 0, 'phase': 'Progressing'}

第二次 reconcile（已经收敛）：
  动作： 无
  status： {'readyReplicas': 3, 'phase': 'Ready'}

把 replicas 改成 6 之后：
  update deploy/qwen3 -> {'replicas': 6, 'image': 'vllm/vllm-openai:v0.11.0', 'gpus': 1}
  update hpa/qwen3 -> {'min': 3, 'max': 18, 'metric': 'num_requests_waiting'}
  再跑一次： {'readyReplicas': 6, 'phase': 'Ready'}

有人手动删掉了 Service：
  动作： [('create', 'svc/qwen3')] （自愈）
```

这段逻辑和用 controller-runtime（Go）或 kopf（Python）写出来的 `Reconcile()` 是同构的，四个要点都体现在输出里：

- **幂等**：第二次 reconcile 没有任何动作，`phase` 变成 `Ready`；
- **只比较自己管的字段**：子控制器写回的 `ready` 不参与比较，否则会陷入"每次都觉得要更新"的死循环；
- **自愈**：有人手动删掉 Service，下一次 reconcile 就把它建回来；
- **status 写回**：控制器把观测到的实际情况写进 `status`，用户和上层系统据此判断是否可用。

真实 Operator 还要处理几件事：

| 事项 | 做法 |
| --- | --- |
| 子对象归属 | 创建时写 `ownerReferences`，删除自定义资源时级联删除子对象 |
| 清理外部资源 | 用 **finalizer**：删除时先执行清理（注销网关路由、释放外部存储），完成后再移除 finalizer |
| 并发冲突 | 更新用乐观锁（`resourceVersion`），冲突就重新读取再试 |
| 限速与重试 | workqueue 的指数退避；失败的对象过一会再进队列 |
| 版本演进 | CRD 多版本 + conversion webhook；`storage: true` 的那个版本是实际存储格式 |
| 校验与默认值 | schema 里的 `required`、`default`，复杂规则用 validating/mutating webhook |

生态里已有的推理 Operator 可以直接用或参考：**KServe**（标准的 InferenceService，支持多框架、金丝雀、Serverless）、**KubeAI**、**llm-d**（面向分离式架构）、**LeaderWorkerSet**（多机组）、**Kueue**（队列与配额）。自研的价值通常在"接上公司内部的存储、网关、监控和审批流程"。

## 多租户：配额、优先级与隔离

```yaml
apiVersion: v1
kind: ResourceQuota
metadata: {name: team-a, namespace: default}
spec:
  hard:
    requests.nvidia.com/gpu: "8"
    requests.cpu: "64"
    requests.memory: 256Gi
    count/deployments.apps: "10"
```

```bash
kubectl describe quota team-a
```

```text title="输出（本机示例）"
Name:                    team-a
Namespace:               default
Resource                 Used  Hard
--------                 ----  ----
count/deployments.apps   1     10
requests.cpu             150m  64
requests.memory          96Mi  256Gi
requests.nvidia.com/gpu  1     8
```

超了会在**创建 Pod 时**被拒绝，错误信息在 ReplicaSet 的事件里（不在 Deployment 上，这点很容易找错地方）：

```text title="输出（本机示例）"
Warning  FailedCreate  5s  replicaset-controller  Error creating: pods "big-6dd76597ff-nfbnn" is forbidden: exceeded quota: team-a, requested: requests.cpu=100, used: requests.cpu=150m, limited: requests.cpu=64
```

配套的几个机制：

- **LimitRange**：给没写 requests 的容器补默认值（开了 ResourceQuota 之后，不写 requests 的 Pod 会被直接拒绝，报错是 `must specify requests.cpu`），也可以限制单个容器的上下限。
- **优先级与抢占**：在线推理高优先级、离线任务低优先级（见[调度器](scheduling.md#抢占与优先级)）。
- **Kueue**：比 ResourceQuota 更适合 GPU——它在**准入层**排队，资源不够时任务排队等待而不是创建失败，还支持队列之间的借用与回收。
- **命名空间 + NetworkPolicy**：把不同团队隔开，限制跨命名空间的网络访问。

一个常见的取舍：ResourceQuota 是"硬上限、超了就拒绝"，对批处理任务不友好（任务直接失败而不是排队）；Kueue 是"排队 + 配额借用"，更适合 GPU 集群的公平共享。生产上常见的是两者叠加：ResourceQuota 兜底防止失控，Kueue 做日常调度。

## 排障：一张表走完

把前几章的线索汇总成一条动线。先定位**卡在哪一层**：

```bash
kubectl get isvc,deploy,rs,pods -l app=vllm      # 从上到下，看哪一层的数字对不上
kubectl describe <卡住的那一层>                   # Events 在最下面
kubectl logs <pod> --previous                    # 容器崩溃时看上一个实例的日志
kubectl get events --sort-by=.lastTimestamp | tail -30
```

| 现象 | 先看 | 常见原因 |
| --- | --- | --- |
| 自定义资源有、Deployment 没有 | Operator 的日志 | 控制器崩了、RBAC 权限不足、schema 校验失败 |
| Deployment 有、ReplicaSet 的 Pod 建不出来 | `describe rs` 的事件 | **超配额**、LimitRange 要求、准入 webhook 拒绝 |
| Pod `Pending` | `describe pod` 的 `FailedScheduling` | GPU 不够、污点/亲和不匹配、碎片化、PVC 没绑定 |
| Pod `ContainerCreating` 很久 | `describe pod` 的事件 | 拉镜像慢（几十 GB）、挂卷失败、device plugin 没就绪 |
| Pod `CrashLoopBackOff` | `logs --previous`、`Last State` | 参数错、显存不足、权重路径不对、**liveness 探针太严** |
| Pod `Running` 但 `0/1` | 就绪探针配置、`/health` 返回 | 还在加载权重（缺 startupProbe）、探针路径或端口错 |
| Pod `OOMKilled` | `describe` 的 `Last State`、监控 | 内存 limits 按稳态设的、忘了加载峰值、`/dev/shm` 太小 |
| Pod 频繁被驱逐 | 节点事件、`kubectl top node` | 节点内存压力、QoS 是 BestEffort、节点磁盘压力 |
| 服务能访问但慢/抖动 | 指标（TTFT、队列长度）、CPU 节流 | CPU limits 太小被节流、探针误摘端点、路由不均 |
| 滚动发布卡住 | `rollout status`、`describe rs` | 新 Pod 起不来（上面各条）、PDB 挡住、资源不够 surge |
| GPU 相关异常 | `nvidia-smi`、DCGM 指标、节点事件 | 掉卡、ECC 错误、驱动与镜像不匹配、MIG 配置变更 |

两个容易忽略的排查点：

- **事件会过期**（默认保留 1 小时），事后排障要靠日志系统和监控，所以生产集群要把事件也采集走。
- **看 `kubectl get pod -o yaml` 的 `status.conditions`**：`PodScheduled`、`Initialized`、`ContainersReady`、`Ready` 四个条件能精确告诉你卡在哪一步。

!!! interview "怎么讲清楚"
    讲 Operator：CRD 注册对象类型、Operator 写控制器把它翻译成 Deployment/Service/HPA；spec 是用户的期望、status 是控制器写回的实际，用 status 子资源分开更新权限；子对象靠 ownerReferences 归属和级联删除，外部资源清理靠 finalizer；reconcile 要幂等、只比较自己管的字段、用 workqueue 限速重试。推理平台的价值在于把"上线一个模型"的十几个 YAML 收敛成五行。多租户讲 ResourceQuota（硬上限、超了创建失败、错误在 ReplicaSet 事件里）、LimitRange（默认值）、优先级抢占、以及 Kueue 的准入排队更适合 GPU。排障给动线：从自定义资源往下看哪一层数字对不上 → describe 看事件 → logs --previous；并背下 Pending/CrashLoop/0-1 READY/OOMKilled 各自先看什么。

## 练习

**1. 设计一个 CRD。** 你要让业务方能一行上线一个带 LoRA 的模型服务，支持多个适配器、按队列长度扩缩容、并自动注册到网关。列出 `spec` 里该有哪些字段，以及控制器要创建哪些子对象。

??? success "参考答案"
    `spec`：`baseModel`（基座模型）、`adapters`（适配器列表：名字 + 存储路径）、`replicas` 或 `autoscaling: {min, max, targetQueueLength}`、`gpusPerReplica`、`resources`（可选覆盖）、`gateway: {route, priority}`。

    子对象：Deployment（或 LWS）、Service、ScaledObject（KEDA）、ConfigMap（适配器清单）、HTTPRoute / InferenceModel（网关注册）、ServiceMonitor（监控采集）、PodDisruptionBudget。status 里写 `readyReplicas`、`loadedAdapters`、`endpoint`、`conditions`。

**2. 找错地方了。** 一个 Deployment 的 `AVAILABLE` 一直是 0，`kubectl describe deploy` 的事件只有一条 `ScalingReplicaSet`。接下来该看哪里？

??? success "参考答案"
    看 **ReplicaSet**：`kubectl describe rs -l app=<name>`。Pod 创建失败（配额、LimitRange、准入 webhook、PSA 安全策略）的事件挂在 ReplicaSet 上，Deployment 上只会显示"我已经让 RS 扩容了"。

    如果 ReplicaSet 显示 Pod 已经创建，再往下看 Pod 的事件和日志。这条"从上到下逐层看数字对不对"的动线，比一上来就看 Pod 日志有效得多。

**3. 配额设计。** 一个团队有 16 张 GPU 的配额，他们同时跑在线推理（8 卡）和离线评测（想用剩下的 8 卡，但高峰期要让出来）。怎么配？

??? success "参考答案"
    （1）ResourceQuota 设 `requests.nvidia.com/gpu: "16"` 作为硬上限；（2）在线服务用高优先级的 `PriorityClass` 并开启抢占，离线评测用低优先级；（3）离线任务用 Kueue 的队列管理，配额设成"可借用"——高峰期在线扩容时抢占离线 Pod，离线任务自动回到队列等待；（4）离线任务实现检查点和幂等，被抢占后能续跑；（5）给在线服务配 PDB，避免运维操作把它一次拿走太多。

**4. 完整排障。** 一个新上线的模型服务，`kubectl get pods` 显示 3 个 Pod 全是 `Running 1/1`，但网关返回 503。给出排查顺序。

??? success "参考答案"
    Pod 都健康，说明问题在 Pod 之外：（1）**Service 的端点**：`kubectl get endpoints <svc>`，空的说明标签选择器不匹配（Service 的 `selector` 和 Pod 的 labels 对不上）或者 readiness 没通过；（2）**端口**：Service 的 `targetPort` 和容器实际监听的端口是否一致，容器是否监听在 `0.0.0.0` 而不是 `127.0.0.1`；（3）**网关配置**：HTTPRoute/Ingress 指向的 Service 名和命名空间是否正确，路径重写规则是否把 `/v1/...` 改坏了；（4）**从集群内部直连**验证：`kubectl run -it --rm curl --image=curlimages/curl -- curl http://<svc>:8000/health`，能通就说明问题在网关层；（5）**网关日志**：看它选了哪个后端、返回了什么。

## 小结

- [x] CRD 定义对象、Operator 写控制器把它翻译成实际资源；推理平台用它把"上线一个模型"收敛成几行。
- [x] spec 是期望、status 是实际，用 status 子资源分开；子对象靠 ownerReferences 级联删除，外部清理靠 finalizer。
- [x] reconcile 要幂等、只比较自己管的字段，否则会陷入更新死循环。
- [x] ResourceQuota 是硬上限（错误在 ReplicaSet 事件里），LimitRange 补默认值，GPU 集群更适合用 Kueue 做准入排队。
- [x] 排障动线：从上到下看哪一层数字对不上 → describe 看事件 → logs --previous；Pending/CrashLoop/0-1 READY/OOMKilled 各有固定的第一落点。
