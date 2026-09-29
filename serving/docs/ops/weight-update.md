# 权重热更新：不重启地换权重

<p class="lead">有些场景要求推理引擎在不重启的情况下换掉权重：RL 训练每一步都要把新策略送进 rollout 引擎，线上服务要把微调后的新版本换上去，弹性 EP 要把丢失的专家补回来。重启一次要重新加载几百 GB 的权重、重新捕获 CUDA Graph、预热，热更新则希望只花传输权重的时间。这一章讲清楚热更新的三个难点：在途请求和缓存怎么处理（一致性），检查点格式和 kernel 用的格式不一样（加载后处理），以及 CUDA Graph 记住的是地址（只能原地更新）；然后看 vLLM 和 SGLang 提供了哪些接口、权重从哪里来。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. 更新权重时，正在生成的请求有哪几种处理方式？各自的代价是什么？
    2. 为什么不能把检查点里的张量直接拷进模型的参数？
    3. 为什么更新权重必须"原地拷贝"，而不能让参数指向一块新的显存？
    4. 更新之后前缀缓存要不要清空？为什么？
    5. vLLM 和 SGLang 分别用什么方式把权重送进推理引擎？

??? success "自测参考答案（先自己答，再展开对照）"
    1. 中止（abort）：更新马上开始，但已生成的内容作废、要重新来；等待（wait）：不浪费，但要等最长的请求结束，长尾会让更新等上几分钟；冻结（vLLM 的 keep、SGLang 的 in_place）：不等也不浪费，但这些请求前半段是旧权重生成的，KV 也是旧权重算的；退回队列（SGLang 的 retract）：更新后用新权重重新 prefill，KV 是新的，但已经生成的 token 仍来自旧权重。
    2. 检查点是训练端的格式（完整的浮点张量、按模块分开），kernel 用的是处理过的格式：按张量并行切分、QKV / gate_up 合并、FP8 分块量化和缩放系数、int4 重排、MoE 专家的冗余副本……必须走模型自己的加载流程（`load_weights` 和加载后处理），否则要么形状对不上，要么被悄悄地转换成错误的值（比如浮点拷进 int8 缓冲区被截断）。
    3. CUDA Graph 捕获时记下的是每个张量的地址，重放时只读这些地址；kernel 里预先准备的描述符（比如 TMA 描述符）也是一样。参数指向新显存后，模型的 Python 代码看到的是新权重，但重放的图还在读旧的那块，结果是错的而且不报错。所以更新要把新值拷进原来的存储，地址不变。
    4. 一般要清空：前缀缓存里的 KV 是旧权重算出来的，新请求命中它就等于用旧权重处理了前缀。对 RL 来说这会让 rollout 混进旧策略；对线上换版本来说，新旧版本的输出会混在一起。
    5. vLLM：权重传输引擎（`nccl` 广播、`ipc` 共享显存、`sparse_nccl` 只传变化的元素、`sharded_rdt` 每个 worker 只拉自己那一片），按层分组、打包进固定大小的缓冲区流水传输，每层收齐后执行加载后处理并原地拷贝。SGLang：`/update_weights_from_disk`（从磁盘换）、`/update_weights_from_tensor` 和 `/update_weights_from_ipc`（共置时传张量或显存句柄）、`/init_weights_update_group` + `/update_weights_from_distributed`（NCCL 广播），也集成了专门的 checkpoint-engine。

## 什么时候需要热更新

| 场景 | 多久一次 | 最在意什么 |
| --- | --- | --- |
| RL 训练的权重同步 | 每个训练步（几十秒到几分钟） | 更新要快（传输时间直接影响 GPU 利用率，见[异步 RL 的时间账](../frontier/rl-async.md#权重同步的时间账)），生成的数据要知道来自哪个版本的策略 |
| 线上换成微调后的新版本 | 几天到几周 | 不中断服务、能灰度、能回滚 |
| 弹性 EP 补回丢失的专家 | 故障时 | 只更新一部分权重，越快越好（见[大规模 EP 的容错](../moe/ep-elastic.md)） |

架构变了（层数、隐藏维度、量化方式）只能重启，用[发布策略](deploy.md#发布策略)滚动替换实例；架构不变、只是权重变了，就可以热更新。多 LoRA 服务里动态加载适配器（见[多 LoRA](multi-lora.md)）是更轻的一种形式：底座不动，只换小的适配器矩阵。

## 难点一：在途请求和缓存

更新发生时，引擎里有正在生成的请求。一次前向中途换权重肯定不行（一半层是旧的、一半是新的），所以更新总是在两步之间进行，先让调度器暂停。暂停时怎么处理在途请求，有四种做法（vLLM 的 `pause_generation(mode=...)` 支持 `abort`、`wait`、`keep`，SGLang 的 `/pause_generation` 支持 `abort`、`retract`、`in_place`）：

```python
import random

rng = random.Random(0)
STEP_MS, PROMPT, N = 25, 2048, 64                # 假设：decode 一步 25 ms，每个请求的提示词 2048 token，64 个在途请求
done = [rng.randint(1, 4000) for _ in range(N)]                                   # 已经生成的 token
left = [min(16000, int(rng.lognormvariate(6.5, 1.2))) for _ in range(N)]          # 还要生成的 token（长尾）


def cell(text, width):
    """按显示宽度右对齐（汉字占两格）"""
    return " " * (width - sum(2 if ord(ch) > 0x2E7F else 1 for ch in str(text))) + str(text)


rows = [  # 做法, 更新前要等多久, 要重新 decode 的 token, 要重新 prefill 的 token, 新旧权重混合的请求数, KV 是否和新权重一致
    ("abort：中止在途请求", 0, sum(done), N * PROMPT, 0, "是"),
    ("wait：等在途请求做完", max(left) * STEP_MS / 1000, 0, 0, 0, "是"),
    ("keep：冻结，接着用旧 KV", 0, 0, 0, N, "否"),
    ("retract：退回队列，重算 KV", 0, 0, N * PROMPT + sum(done), N, "是"),
]
print(f"{N} 个在途请求：已生成 {sum(done)} 个 token；还要生成的最多 {max(left)} 个，中位数 {sorted(left)[N // 2]} 个")
print(cell("做法", 28) + cell("更新前要等", 12) + cell("重新 decode", 13) + cell("重新 prefill", 14)
      + cell("混合版本的请求", 16) + cell("KV 用新权重", 13))
for name, wait_s, redecode, reprefill, mixed, fresh in rows:
    print(cell(name, 28) + cell(f"{wait_s:.0f} s", 12) + cell(redecode, 13) + cell(reprefill, 14)
          + cell(mixed, 16) + cell(fresh, 13))
```

```text title="输出"
64 个在途请求：已生成 139822 个 token；还要生成的最多 16000 个，中位数 466 个
                        做法  更新前要等  重新 decode  重新 prefill  混合版本的请求  KV 用新权重
         abort：中止在途请求         0 s       139822        131072               0           是
        wait：等在途请求做完       400 s            0             0               0           是
     keep：冻结，接着用旧 KV         0 s            0             0              64           否
  retract：退回队列，重算 KV         0 s            0        270894              64           是
```

- **abort**：更新马上开始，代价是已经生成的十几万个 token 作废，这些请求要从头来过；
- **wait**：不浪费，但长尾让更新等了 400 秒——等待期间新请求也进不来，GPU 越来越空。RL 里这正是"同步 RL 被长尾拖慢"的问题（见 [RL 训练中的推理](../topics/rl-rollout.md)）；
- **keep / in_place**：最快，也不浪费，但这 64 个请求的前半段是旧权重生成的，而且旧 KV 会继续被新权重使用。对 RL 来说，一条轨迹混了两个版本的策略，训练端要按 token 记录版本、做重要性采样修正，或者丢掉这些轨迹；
- **retract**：把请求退回等待队列、释放 KV，更新后用新权重重新 prefill（prefill 是并行的，比重新 decode 快得多），KV 和新权重一致，但已经生成的 token 仍来自旧权重。

另外，**前缀缓存里的 KV 都是旧权重算的**。vLLM 的 `pause_generation` 默认 `clear_cache=True`，把 KV 和前缀缓存一起清掉；SGLang 在 retract 模式下可以清空缓存、之后自动重算。给生成结果打上版本号也很重要：vLLM 和 SGLang 都能设置权重版本（`update_weight_version`），RL 框架据此判断每条数据来自哪个版本的策略。

## 难点二：检查点格式不等于 kernel 格式

训练端送来的是检查点格式的权重：完整的浮点张量，按模块命名。推理引擎里的参数则是加载时处理过的：按张量并行切分，QKV 和 gate_up 合并成一个矩阵，FP8 量化并按块算好缩放系数，int4 权重重排成 kernel 需要的布局，MoE 专家按 EPLB 复制到多张卡上……直接拷贝要么形状对不上，要么更糟——形状对上了，值被悄悄转换错。下面模拟一个 int8 量化的层（真实系统里是 FP8 分块缩放等加载后处理）：

```python
import torch

def quantize(w):
    """按行量化成 int8：每行一个缩放系数（真实系统里是 FP8 分块缩放、int4 重排、QKV 合并等"加载后处理"）"""
    scale = w.abs().amax(dim=1, keepdim=True) / 127
    return torch.round(w / scale).to(torch.int8), scale


class Int8Linear:
    """kernel 用的格式：int8 权重 + 缩放系数。CUDA Graph 捕获的是这两块存储"""
    def __init__(self, w):
        self.qweight, self.scale = quantize(w)

    def __call__(self, inp):
        return inp @ (self.qweight.float() * self.scale).T


torch.manual_seed(1)
w0, w1 = torch.randn(64, 64) * 0.05, torch.randn(64, 64) * 0.05     # 旧权重、新的检查点（checkpoint 格式：浮点）
lin = Int8Linear(w0)
ptrs = (lin.qweight.data_ptr(), lin.scale.data_ptr())
x = torch.randn(8, 64)
ref = x @ w1.T


def rel_err():
    return ((lin(x) - ref).norm() / ref.norm()).item()


lin.qweight.copy_(w1)                            # 错误：浮点直接拷进 int8 缓冲区，被悄悄截断成 0，缩放系数也还是旧的
print(f"直接拷贝检查点：相对误差 {rel_err():.3f}")
q, s = quantize(w1)                              # 正确：先按加载时的方式重新处理，再原地拷贝进原来的两块存储
lin.qweight.copy_(q)
lin.scale.copy_(s)
print(f"重新量化后原地拷贝：相对误差 {rel_err():.4f}（只剩量化误差），地址没变：",
      (lin.qweight.data_ptr(), lin.scale.data_ptr()) == ptrs)
```

```text title="输出"
直接拷贝检查点：相对误差 1.000
重新量化后原地拷贝：相对误差 0.0063（只剩量化误差），地址没变： True
```

所以更新必须走和加载时相同的流程：模型的 `load_weights` 负责切分、合并、复制到各个专家副本，加载后处理（vLLM 里是量化方法的 `process_weights_after_loading`）负责量化和重排。vLLM 的"按层重新加载"（`vllm/model_executor/model_loader/reload/layerwise.py`）把这件事做得很仔细：

1. 更新开始前，记下每一层当前 kernel 用的张量；
2. 把各层的参数恢复成检查点格式的"空壳"（放在 meta 设备上，不占显存），并包装各个参数的 `weight_loader`，让处理推迟到这一层的权重全部到齐；
3. 某一层收齐后：在设备上建出这一层、加载缓存的权重、执行量化等处理，再把处理后的值**原地拷贝回第 1 步记下的那些张量**；
4. 最后处理注意力层（比如 KV 缩放系数）等需要延后的层。

一次只展开一层，额外的显存只有一层的大小；原地拷贝保证了 kernel 和 CUDA Graph 看到的地址不变——这正是下一个难点。

## 难点三：CUDA Graph 记住的是地址

CUDA Graph 捕获时，每个 kernel 的参数（包括指向权重的指针）都被记录下来，重放时只读这些地址（见[CUDA Graphs 与 torch.compile](../engine/graphs-compile.md)）。如果更新时让参数指向一块新分配的显存，模型的 Python 代码看到的是新权重，重放的图读的却还是旧的那块——结果错了，而且不报错。下面用一个记住了权重存储的闭包来模拟捕获的图：

```python
import torch

torch.manual_seed(0)
layer = torch.nn.Linear(4, 4, bias=False)
x = torch.randn(1, 4)


def capture(layer):
    """模拟 CUDA Graph：捕获时记下的是权重张量的存储（在 GPU 上就是一个固定的地址），之后重放只读这块存储"""
    w = layer.weight.detach()                    # 和参数共享同一块存储
    return lambda inp: inp @ w.T


graph = capture(layer)
new_w = torch.randn(4, 4)

# 错误的更新：让参数指向一块新的存储。模型本身用上了新权重，但捕获的图还在读旧的那块
layer.weight.data = new_w.clone()
print("重新绑定：模型输出 == 新权重的结果：", torch.allclose(layer(x), x @ new_w.T),
      "；图的输出 == 新权重的结果：", torch.allclose(graph(x), x @ new_w.T))

# 正确的更新：原地拷贝进原来的存储，地址不变，图和模型都看到新权重
graph = capture(layer)                           # 重新捕获（相当于重启后的状态）
newer_w = torch.randn(4, 4)
ptr = layer.weight.data_ptr()
layer.weight.data.copy_(newer_w)
print("原地拷贝：图的输出 == 新权重的结果：", torch.allclose(graph(x), x @ newer_w.T),
      "；存储地址没变：", layer.weight.data_ptr() == ptr)
```

```text title="输出"
重新绑定：模型输出 == 新权重的结果： True ；图的输出 == 新权重的结果： False
原地拷贝：图的输出 == 新权重的结果： True ；存储地址没变： True
```

`layer.weight.data = new` 之后，模型本身的输出是对的，"图"的输出却还是旧权重的结果；`layer.weight.data.copy_(new)` 把新值写进原来的存储，两边都对。

几条实践规则：

- **只原地更新**：`param.data.copy_(new_value)`，形状、dtype、设备都不变；
- **处理后的缓冲区也要原地更新**：量化的缩放系数、重排后的权重、预先建好的 TMA 描述符指向的缓冲区，都属于"图记住的地址"；
- **形状变了就必须重新捕获**：比如换了一个词表更大的版本，那就不是热更新了；
- **更新后释放临时显存**：加载和处理时的临时张量会留在 PyTorch 的缓存分配器里，SGLang 的 `/continue_generation` 默认先调用一次 `torch.cuda.empty_cache()` 再恢复推理。

## 权重从哪里来

**vLLM** 有一套权重传输引擎（`vllm/distributed/weight_transfer/`），流程是 `init_weight_transfer_engine` → `start_weight_update` → 若干次 `update_weights` → `finish_weight_update`，可以附带新的权重版本。后端（`WeightTransferConfig.backend`）有四种：

- `nccl`：训练端和推理端建一个 NCCL 组，整份权重广播过去；
- `ipc`：训练和推理共置在同一批卡上时，通过 CUDA IPC 句柄直接共享显存里的张量；
- `sparse_nccl`：只传变化了的元素（名字、形状和扁平下标），由模型自己的 `weight_loader` 映射到本 rank 的参数上；
- `sharded_rdt`：用 Ray 的直接传输，每个 worker 只拉取自己在张量并行、专家并行下需要的那一片，而不是完整的张量。第一次更新时用假张量走一遍 `load_weights`，记下每一片从哪里取、落到哪里，之后每次按记录重放。

传输时按解码层分组（`layerwise_groups`），打包进固定大小的缓冲区（默认 2 个 1 GB 的缓冲区轮流用），收一层、处理一层，传输和处理流水进行。

**SGLang** 的接口都是 HTTP 端点（也有对应的 Python 方法）：

- `/update_weights_from_disk`：从磁盘上的新检查点原地换权重，不重启服务，返回暂停的请求数；
- `/update_weights_from_tensor`、`/update_weights_from_ipc`：共置时传张量或 CUDA IPC 句柄，多个张量先打包进一个扁平的缓冲区（`srt/weight_sync/tensor_bucket.py`）；
- `/init_weights_update_group` + `/update_weights_from_distributed`：和训练端建一个 NCCL 组，按批广播；
- `/init_weights_send_group_for_remote_instance`：让一个已经在服务的实例（种子实例）把权重发给新启动的实例，新实例不用读磁盘（启动参数里用 `--remote-instance-weight-loader-seed-instance-ip` 等指定种子实例）；
- `srt/checkpoint_engine/`：集成了开源的 checkpoint-engine，服务先用假权重启动（`--load-format dummy`，并用 `--checkpoint-engine-wait-weights-before-ready` 等权重到了再就绪），由它以广播或点对点的方式把检查点推进来。

传输本身的时间账（共置用 NVLink、分离部署时多实例之间流水接力）见[异步 RL 的权重同步](../frontier/rl-async.md#权重同步的时间账)。

## 线上换版本的做法

RL 追求快，线上换版本追求稳。一个可行的流程：

1. **灰度**：先挑一个实例热更新（从磁盘或从已更新的实例拉取），用一组固定的请求对比输出（逐 token 比对和评测集，方法见[新模型接入与精度对齐](new-model.md)），再逐步扩大到其他实例；
2. **一致性**：换版本前暂停调度并清空前缀缓存；多轮对话的会话粘在某个实例上时，要决定这些会话是在旧版本上结束，还是整体切到新版本；
3. **回滚**：热更新不会同时保留两份权重（显存放不下），回滚就是再做一次热更新（从旧检查点），所以旧检查点要保留在能快速读到的地方；
4. **监控**：更新前后对比延迟、吞吐、输出长度和质量指标，异常时自动回滚。

!!! interview "面试怎么答"
    被问"怎么在不重启的情况下更新推理引擎的权重"：先说场景（RL 每步同步、线上换版本、补回专家），再讲三个难点。一致性：更新在两步之间进行，在途请求可以中止、等待、冻结或退回队列，各有代价（浪费、长尾等待、新旧版本混合、重算 prefill），前缀缓存要清空，生成结果要带版本号。格式：检查点格式要经过模型自己的 `load_weights` 和加载后处理（切分、合并、量化、重排）才能变成 kernel 格式，vLLM 按层展开、处理、再拷回。地址：CUDA Graph 记住的是指针，只能原地 `copy_`，不能换成新分配的显存。最后讲传输：共置用 CUDA IPC，分离部署用 NCCL 广播或只拉自己那一片，按层分组、打包进缓冲区流水传输。

## 练习

**1. 选哪种暂停方式。** 一个 RL 系统采用"部分 rollout"：每次更新时，没生成完的轨迹不作废，更新后继续生成，训练时对每个 token 做版本修正。它应该用哪种暂停方式？如果训练端要求每条轨迹只来自一个版本呢？

??? success "参考答案"
    部分 rollout 要的是"不浪费、不等待"，用 keep（vLLM）或 in_place（SGLang）最直接：请求原地冻结，更新后接着生成，每个 token 记录生成它的权重版本。它的代价是旧 KV 继续被新权重使用，这部分偏差也由训练端的版本修正吸收；如果希望 KV 和新权重一致，可以用 retract，代价是重新 prefill。如果要求每条轨迹只来自一个版本，就只能在 abort（重新生成）和 wait（等长尾）之间选，或者把没生成完的轨迹留在旧版本的引擎上做完（需要两套引擎或更复杂的调度）。

**2. 找 bug。** 某人实现的热更新是这样的：`for name, t in new_weights: model.get_parameter(name).data = t.to("cuda")`。测试时不开 CUDA Graph 一切正常，一开 CUDA Graph 输出就不对。原因是什么？还有什么别的问题？

??? success "参考答案"
    `.data = ...` 让参数指向新分配的显存，开了 CUDA Graph 时重放的图还在读旧的地址，所以不对；不开 CUDA Graph 时每次都重新发射 kernel、读的是参数当前的地址，所以"正常"。应该改成 `param.data.copy_(t)`。别的问题：直接按名字赋值绕过了模型的 `load_weights`，检查点里的 `q_proj`、`k_proj`、`v_proj` 在推理端早已合并成 `qkv_proj`，张量并行下每个 rank 只有一片，量化模型还需要重新量化——名字对不上会报错，对得上的也可能值不对；另外，旧显存在图还引用时被释放，可能被别的张量复用，导致更难查的错误。

**3. 更新多久。** 一个 32B 的 BF16 模型分在 4 张卡上（张量并行），和训练端共置。用 CUDA IPC 更新时，每张卡要拷贝多少数据？如果推理端是 FP8，训练端是 BF16，这一步还要做什么？

??? success "参考答案"
    32B × 2 字节 = 64 GB，分到 4 张卡上每卡 16 GB。共置时通过 IPC 句柄直接读训练端的显存，拷贝走 NVLink 或卡内显存带宽，每卡 16 GB 大约是秒级。推理端是 FP8 时，要在拷贝后（或者在训练端发送前）按推理端的量化方式重新量化：算每个块的缩放系数、转换成 FP8，再原地拷进 kernel 用的缓冲区；如果用了 DeepGEMM 之类要求特殊缩放格式（比如 UE8M0）的 kernel，还要按它的格式打包缩放系数。

## 小结

- [x] 热更新的场景：RL 每步同步权重、线上换版本、弹性 EP 补回专家；架构变了只能重启。
- [x] 一致性：更新在两步之间；在途请求可以中止、等待、冻结或退回队列，代价各不相同；前缀缓存要清空，结果要带版本号。
- [x] 格式：检查点要走 `load_weights` 和加载后处理（切分、合并、量化、重排）；vLLM 按层展开、处理、再原地拷回，额外显存只有一层。
- [x] 地址：CUDA Graph 记住的是指针，只能原地 `copy_`，处理后的缓冲区也一样。
- [x] 传输：vLLM 的 `nccl` / `ipc` / `sparse_nccl` / `sharded_rdt`，按层分组、打包流水；SGLang 的从磁盘、张量、IPC、分布式、远程实例更新和 checkpoint-engine。
