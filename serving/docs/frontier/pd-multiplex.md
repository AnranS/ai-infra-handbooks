# PD 复用：在一张卡上按 SM 切分 prefill 与 decode

<p class="lead">prefill 和 decode 放在同一张卡上会互相干扰：一个长提示词的 prefill 插进来，正在 decode 的请求这一步就要等它。分块 prefill 把干扰切成小段，PD 分离干脆把两者放到不同的卡上。还有第三条路：在同一张卡上按 SM 切开，prefill 和 decode 各用一部分 SM、同时运行。CUDA 的 green context 让一个流只在指定的 SM 上执行，SGLang 的 PD 复用（PD-Multiplexing）就建立在它之上。这一章讲清楚 green context 能隔离什么、不能隔离什么，读 SGLang 的实现（怎么切 SM、按什么选切分、为什么把 prefill 按层切开），再用一个估算模型比较它和分块 prefill 的 ITL 与 prefill 用时。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. 同一批 GPU 上，prefill 和 decode 有哪几种共存方式？各自付出什么代价？
    2. green context 是什么？它能隔离哪些资源，不能隔离哪些？
    3. SGLang 的 PD 复用怎样决定分给 decode 多少个 SM？
    4. 为什么 PD 复用要把 prefill 按层切开，一轮只算几层？
    5. 同样插进来一个长 prefill，PD 复用和分块 prefill 相比，ITL 和这个请求的 prefill 用时各有什么不同？

??? success "自测参考答案（先自己答，再展开对照）"
    1. 时间上交替：prefill 优先（decode 整段被卡住）或分块 prefill（每步混一块 prefill，decode 每步都变慢）；空间上切分：同一张卡上按 SM 分给 prefill 和 decode，同时运行（PD 复用），代价是两边都只有一部分算力、显存带宽和 L2 仍然共享；分到不同的卡上（PD 分离）：没有干扰，但要传 KV、要更多卡、要调 xPyD 配比。
    2. 一种轻量的 CUDA 上下文，绑定一部分 SM；在它上面创建的流，kernel 只在这部分 SM 上执行。能隔离的是 SM（计算单元、寄存器、共享内存）；不能隔离显存带宽、L2 缓存等共享资源，文档也明确说不同 green context 上的 kernel 不保证真的并发。
    3. 启动时把 SM 切成几档（H100 上 prefill : decode 从 112 : 20 到 72 : 60，粒度 8 个 SM，decode 至少 16 个），再加上"全给 prefill"和"全给 decode"两组普通流；运行时按正在 decode 的请求数选档：`档位 = decode 请求数 × 可切的档数 // decode_bs_divisor`（默认 36），限制在可切的范围内，decode 越多分到的 SM 越多；只有 decode 或只有 prefill 时用整卡。
    4. 调度循环每一轮都要同步 decode 流、处理 decode 结果，prefill 如果一次算完所有层，一轮就被它拖得很长，切分也没法跟着 decode 的负载调整。按层切开后，每轮只提交 `split_forward_token_budget // prefill 的 token 数` 层（默认预算 65536），prefill 和 decode 交替推进，prefill 结束后再换一档切分。
    5. PD 复用下 decode 一直有自己的 SM，ITL 只比独占整卡时略慢（本章的估算里 9 ms 变成 9～15 ms），分块 prefill 的 ITL 在 prefill 期间变成 19～74 ms；代价是 prefill 只用到一部分算力，用时变长（估算里 291 ms 变成 480 ms）。

## 三种共存方式

| | 时间上交替（prefill 优先 / 分块 prefill） | 空间上切分（PD 复用） | 分到不同的卡（PD 分离） |
| --- | --- | --- | --- |
| decode 受的干扰 | prefill 优先时整段卡住；分块时每步都变慢 | 小：有自己的 SM，只和 prefill 争带宽、L2 | 没有 |
| 新请求的 prefill 用时 | 最快（用整卡） | 变长（只有一部分 SM） | 取决于 prefill 实例的负载 |
| KV 传输 | 不需要 | 不需要（同一块显存） | 需要（RDMA / NVLink） |
| 部署规模 | 一张卡起 | 一张卡起 | 至少一个 prefill 实例 + 一个 decode 实例 |
| 主要的调节手段 | token 预算、分块大小 | 切分档位、prefill 每轮的层数 | xPyD 配比、路由 |

[调度器](../engine/scheduler.md)和 [PD 分离](../distributed/pd-disagg.md)两章讲过前后两种。PD 复用介于两者之间：不用多买卡、不用传 KV，又能让 decode 不被 prefill 卡住。它依赖的硬件能力是 green context。

![图：同一张卡上 prefill 与 decode 的三种共存方式](../assets/figures/pd-multiplex-sm.svg){.aig-svg}

## green context：给流划一块 SM

**green context** 是 CUDA 12.4 起驱动 API 提供的一种轻量上下文：它只拥有设备的一部分资源（目前主要是 SM）。在它上面创建的流，提交的 kernel 只在这部分 SM 上执行。用法分五步（驱动 API，CUDA 13 的运行时 API 里也有了对应的 `cudaGreenCtxCreate` 等函数）：

```cpp
CUdevResource all, parts[2], rest;
cuDeviceGetDevResource(dev, &all, CU_DEV_RESOURCE_TYPE_SM);             // 1. 拿到整卡的 SM 资源
unsigned n = 1;
cuDevSmResourceSplitByCount(parts, &n, &all, &rest, 0, 104);            // 2. 切出一份 104 个 SM，剩下的在 rest
CUdevResourceDesc desc;
cuDevResourceGenerateDesc(&desc, &parts[0], 1);                         // 3. 生成资源描述
CUgreenCtx g;
cuGreenCtxCreate(&g, desc, dev, CU_GREEN_CTX_DEFAULT_STREAM);           // 4. 创建 green context
CUstream prefill_stream;
cuGreenCtxStreamCreate(&prefill_stream, g, CU_STREAM_NON_BLOCKING, 0);  // 5. 在它上面建流：这个流的 kernel 只用这 104 个 SM
// 用 rest 同样建第二个 green context 和 decode_stream
```

几个限制要清楚：

- **切分有粒度**：计算能力 7.x、8.x 上 SM 数要是 2 的倍数，9.0 及以上要是 8 的倍数（按 CUDA 头文件里的说明）；
- **只隔离 SM**：显存带宽、L2 缓存、拷贝引擎、工作队列仍然共享。CUDA 的文档说得很明确：即使两个 green context 的 SM 不相交，它们上面的 kernel 也不保证真的并发执行；CUDA 13 另外提供了工作队列的配置（`cudaDevWorkqueueConfigScopeGreenCtxBalanced`），尽量让不同 green context 的提交互不阻塞；
- **不是严格上限**：文档列了两种 kernel 会用到比分配的更多 SM 的情况（MPS 限制了线程百分比时；Hopper 上加载了动态并行的模块时，会多用 2 个 SM），但不会更少。

decode 受访存限制，只要有足够多的 SM 同时发出访存请求就能接近带宽上限；prefill 受算力限制，SM 越多越快。所以"给 decode 一小部分 SM、其余给 prefill"在带宽上并不吃亏，这正是 PD 复用的出发点。

## SGLang 的 PD 复用

SGLang 用 `--enable-pdmux` 打开 PD 复用，配置文件由 `--pdmux-config-path` 指定（YAML，字段有 `sm_group_num`、`manual_divisions`、`split_forward_token_budget`、`decode_bs_divisor`）。代码在 `srt/multiplex/`：`pdmux_context.py` 负责切分 SM、创建流，`multiplexing_mixin.py` 是调度循环 `event_loop_pdmux`。下面把它的切分和选择规则原样写成 Python（与 SGLang 0.5.20 的 `divide_sm` 在 40～200 个 SM、计算能力 7～9 的所有组合上逐一对比过，结果相同）：

```python title="pdmux_policy.py"
"""pdmux_policy.py —— SGLang PD 复用（srt/multiplex/pdmux_context.py、multiplexing_mixin.py）的切分与选择规则。"""

ARCH = {6: (1, 1), 7: (2, 2), 8: (4, 2), 9: (8, 8)}     # 计算能力大版本 → (每份最少 SM 数, SM 数的粒度)


def divide_sm(total_sms, major, groups):
    """候选的 (prefill SM, decode SM) 切分：prefill 不少于一半，decode 至少 16 个，按粒度取值，均匀挑 groups 个"""
    min_per_part, multiple = ARCH[major]
    cand = [x for x in range(min_per_part, total_sms - min_per_part + 1, multiple)
            if x >= total_sms - x and total_sms - x >= 16]
    if len(cand) >= groups:
        cand = cand[::max(1, len(cand) // groups)][:groups]
    return [(x, total_sms - x) for x in reversed(cand)]            # prefill 分得多的排在前面


def stream_groups(total_sms, major, sm_group_num=8):
    """第 0 组：全部 SM 给 prefill（普通流）；中间 sm_group_num-2 组：green context 切分；最后一组：全部给 decode"""
    return [(total_sms, 0)] + divide_sm(total_sms, major, sm_group_num - 2) + [(0, total_sms)]


def choose(groups, decode_bs, has_prefill, decode_bs_divisor=36):
    """调度器按正在 decode 的请求数选一组：decode 越多，分给 decode 的 SM 越多"""
    n = len(groups)
    if decode_bs and has_prefill:
        return max(1, min(n - 2, decode_bs * (n - 2) // decode_bs_divisor))
    return n - 1 if decode_bs else 0


def prefill_layers_per_step(extend_tokens, num_layers, token_budget=65536):
    """prefill 按层切开：每轮只算 token_budget // extend_tokens 层（至少 1 层），和 decode 交替推进"""
    return min(num_layers, max(1, token_budget // extend_tokens))
```

```python
from pdmux_policy import choose, prefill_layers_per_step, stream_groups

for name, sms, major in (("H100 SXM", 132, 9), ("A100", 108, 8), ("H20", 78, 9)):
    print(f"{name}（{sms} 个 SM）：", stream_groups(sms, major))
groups = stream_groups(132, 9)
print("H100 上 decode 请求数 → 选中的组 (prefill SM, decode SM)：")
for bs in (1, 6, 12, 18, 24, 30, 36, 64):
    k = choose(groups, bs, has_prefill=True)
    print(f"  {bs:3d} → 第 {k} 组 {groups[k]}")
print("只有 decode：", groups[choose(groups, 32, False)], " 只有 prefill：", groups[choose(groups, 0, True)])
for tokens in (512, 8192, 32768, 131072):
    print(f"prefill {tokens:6d} 个 token：每轮 {prefill_layers_per_step(tokens, 36):2d} 层（共 36 层）")
```

```text title="输出"
H100 SXM（132 个 SM）： [(132, 0), (112, 20), (104, 28), (96, 36), (88, 44), (80, 52), (72, 60), (0, 132)]
A100（108 个 SM）： [(108, 0), (84, 24), (78, 30), (72, 36), (66, 42), (60, 48), (54, 54), (0, 108)]
H20（78 个 SM）： [(78, 0), (56, 22), (48, 30), (40, 38), (0, 78)]
H100 上 decode 请求数 → 选中的组 (prefill SM, decode SM)：
    1 → 第 1 组 (112, 20)
    6 → 第 1 组 (112, 20)
   12 → 第 2 组 (104, 28)
   18 → 第 3 组 (96, 36)
   24 → 第 4 组 (88, 44)
   30 → 第 5 组 (80, 52)
   36 → 第 6 组 (72, 60)
   64 → 第 6 组 (72, 60)
只有 decode： (0, 132)  只有 prefill： (132, 0)
prefill    512 个 token：每轮 36 层（共 36 层）
prefill   8192 个 token：每轮  8 层（共 36 层）
prefill  32768 个 token：每轮  2 层（共 36 层）
prefill 131072 个 token：每轮  1 层（共 36 层）
```

- **切分**：`divide_sm` 在满足粒度的取值里挑 prefill 分得不少于一半、decode 至少 16 个 SM 的切法，均匀取 `sm_group_num - 2` 档（默认 6 档）。每一档用 `sgl_kernel.spatial.create_greenctx_stream_by_value` 建一对 green context 流；再在两头加上两组普通流：第 0 组整卡给 prefill，最后一组整卡给 decode；
- **选档**（`adjust_stream_groups`）：同时有 decode 和 prefill 时，按正在 decode 的请求数线性选档，32 个请求时选中 80 : 52；也可以在配置里用 `manual_divisions` 直接写"decode 请求数达到多少用哪一档"。只有 decode 或只有 prefill 时用整卡。每换一档都要先同步两个流，所以只在 prefill 批次开始、结束这类时机调整；
- **decode 用哪套注意力后端**：每一档有自己的 decode 注意力后端和 CUDA Graph 状态（`decode_attn_backend_group`），换档时 `update_decode_attn_backend` 切过去；
- **prefill 按层切开**：新的 prefill 批次标成 `ForwardMode.SPLIT_PREFILL`，模型要实现 `forward_split_prefill(..., split_interval)`，只算 `[start, end)` 这几层（Qwen、Llama、Gemma 等常用模型都实现了）。每轮算 `split_forward_token_budget // prefill 的 token 数` 层：8K token 的 prefill 一轮 8 层，128K token 一轮 1 层；
- **主循环**：每一轮在 decode 流上跑一步 decode，在 prefill 流上提交接下来的几层 prefill，然后只同步 decode 流、处理 decode 的结果；prefill 的最后几层提交后记一个事件，所有 TP rank 通过 CPU 侧的 all-reduce 确认事件都完成了，才把这批请求并进 decode 批次，再换一档切分。

目前的限制（启动参数检查里写着）：不能和流水线并行（`pp_size` 必须为 1）、分块 prefill（`chunked_prefill_size` 必须为 -1）、PD 分离、重叠调度一起用；在 torch 2.7 及以后的版本上，green context 和 CUDA Graph 一起用可能有性能下降，启动时会打印警告。

## 值不值：一个估算

用一个简单的模型比较几种做法：H100 上跑 Qwen3-8B，正在 decode 32 个请求（上下文各 2048），这时插进来一个 8192 token 的 prefill。decode 一步读一遍权重和 KV；prefill 受算力限制，时间和分到的 SM 数成反比。decode 只用一部分 SM 时带宽打多少折，是这个模型里最不确定的一项，用 `S_SAT`（跑满带宽需要的 SM 数）取三档：

```python
from pdmux_policy import choose, stream_groups

# 假设：H100 SXM、Qwen3-8B、BF16；带宽用到 85%，prefill 的 MFU 50%；decode 只用一部分 SM 时，
# 带宽按 min(1, SM 数 / S_SAT) 打折，S_SAT 是跑满带宽所需的 SM 数（不确定，取三档）
SMS, BW, FLOPS = 132, 3.35e12 * 0.85, 989e12 * 0.5
LINEAR = 36 * (2 * 4096 * 4096 + 2 * 4096 * 1024 + 3 * 4096 * 12288) + 4096 * 151936   # 各层线性层 + lm_head 的参数
KV_PER_TOKEN = 36 * 2 * 1024 * 2                                                  # 字节：36 层 × K、V × 8 头 × 128 维 × BF16
DECODE_BS, CONTEXT, PROMPT = 32, 2048, 8192


def decode_step(sms, s_sat):
    """decode 一步：读一遍权重和 32 个请求的 KV"""
    nbytes = 2 * LINEAR + DECODE_BS * CONTEXT * KV_PER_TOKEN
    return nbytes / (BW * min(1.0, sms / s_sat))


def prefill_flops(tokens, start=0):
    """prefill 从第 start 个 token 算到 start + tokens：线性层 2·参数·token，外加因果注意力"""
    attn = 2 * 2 * 36 * 4096 * (tokens * start + tokens * tokens / 2)
    return 2 * LINEAR * tokens + attn


def fmt(sec):
    return f"{sec * 1e3:.0f} ms"


def cell(text, width):
    """按显示宽度右对齐（汉字占两格）"""
    return " " * (width - sum(2 if ord(ch) > 0x2E7F else 1 for ch in text)) + text


groups = stream_groups(SMS, 9)
p_sm, d_sm = groups[choose(groups, DECODE_BS, has_prefill=True)]    # SGLang 为 32 个 decode 请求选的切分
base = decode_step(SMS, 66)                                         # 用全部 SM 时 S_SAT 不起作用
rows = [("只有 decode（参考）", fmt(base), "-", "-")]
t = prefill_flops(PROMPT) / FLOPS
rows.append(("prefill 优先", f"有一步 {fmt(t + base)}", fmt(t), "0"))
for chunk in (2048, 512):
    steps = [max((prefill_flops(chunk, k) + 2 * LINEAR * DECODE_BS) / FLOPS, base) for k in range(0, PROMPT, chunk)]
    rows.append((f"分块 prefill {chunk}", f"{fmt(sum(steps) / len(steps))} × {len(steps)} 步", fmt(sum(steps)), str(len(steps))))
t = prefill_flops(PROMPT) / (FLOPS * p_sm / SMS)
for s_sat in (44, 66, 88):
    itl = decode_step(d_sm, s_sat)
    rows.append((f"PD 复用 {p_sm}/{d_sm}，S_SAT={s_sat}", fmt(itl), fmt(t), f"{t / itl:.0f}"))

print(f"decode {DECODE_BS} 个请求（上下文 {CONTEXT}）时，插进一个 {PROMPT} token 的 prefill：")
print(cell("做法", 26) + cell("decode 的 ITL", 16) + cell("prefill 用时", 14) + cell("期间每个 decode 请求出的 token", 34))
for name, itl, pf, n in rows:
    print(cell(name, 26) + cell(itl, 16) + cell(pf, 14) + cell(n, 34))
```

```text title="输出"
decode 32 个请求（上下文 2048）时，插进一个 8192 token 的 prefill：
                      做法   decode 的 ITL  prefill 用时    期间每个 decode 请求出的 token
       只有 decode（参考）            9 ms             -                                 -
              prefill 优先   有一步 299 ms        291 ms                                 0
         分块 prefill 2048    74 ms × 4 步        295 ms                                 4
          分块 prefill 512   19 ms × 16 步        306 ms                                16
   PD 复用 80/52，S_SAT=44            9 ms        480 ms                                55
   PD 复用 80/52，S_SAT=66           11 ms        480 ms                                43
   PD 复用 80/52，S_SAT=88           15 ms        480 ms                                33
```

- **prefill 优先**：这个 prefill 最快做完，但 decode 有一步卡了 300 ms，对 ITL 的 SLO 是灾难；
- **分块 prefill**：块越小，每步 decode 越快，但 prefill 期间每一步都被拖慢（2048 的块每步 74 ms，512 的块每步 19 ms），而且块太小时 prefill 本身的效率下降；
- **PD 复用**：decode 始终在自己的 52 个 SM 上走，ITL 只从 9 ms 变成 9～15 ms，prefill 期间每个 decode 请求能多出几十个 token；代价是 prefill 只用了 80 个 SM，用时从 291 ms 变成 480 ms。

所以 PD 复用是用新请求的 prefill 用时（TTFT）换 decode 的 ITL 平稳。ITL 的 SLO 很紧、提示词又长的负载最适合它；如果 TTFT 更要紧，分块 prefill 或者给 prefill 分更多 SM 的档位更合适。选档的依据正是 decode 的负载：decode 请求越多，需要的带宽和算力越多，分给 decode 的 SM 也越多。

## 什么时候用

- **规模小、不值得做 PD 分离**时：一两台机器、每台就那么几张卡，PD 复用不用传 KV，也不用为 prefill、decode 各配一组实例；
- **提示词长短差别大**时：偶尔的长提示词不会再卡住所有 decode；
- **和 PD 分离并不矛盾**：分离之后，decode 实例上仍然可能有少量 prefill（比如投机解码的草稿或重算），prefill 实例上也可能做一些 decode；一张卡内部怎么分配 SM，是另一个层次的问题；
- **别指望完全隔离**：带宽和 L2 是共享的。prefill 里的注意力读 KV、MoE 的专家权重读取都会占带宽，decode 的 ITL 仍会有波动，要用真实负载压测（见[压测、SLO 与容量规划](../perf/benchmark.md)）。

!!! interview "面试怎么答"
    被问"不做 PD 分离，怎么减少 prefill 对 decode 的干扰"：时间上交替的办法是分块 prefill（每步混一块 prefill，用 token 预算控制每步的时长）；空间上切分的办法是 PD 复用：用 CUDA 的 green context 把 SM 切成两份，prefill 和 decode 各在自己的流上同时跑。decode 受访存限制，少量 SM 就能接近带宽上限，所以分给它一小部分 SM 在带宽上不吃亏。SGLang 的实现：启动时按 8 个 SM 的粒度切出几档，运行时按 decode 请求数选档，prefill 按层切开、每轮只算几层，和 decode 交替推进，结束后再换档。代价是 prefill 变慢（TTFT 换 ITL），而且 green context 只隔离 SM，显存带宽和 L2 仍然共享。

## 练习

**1. H20 上的切分。** H20 有 78 个 SM，计算能力 9.0。按 SGLang 的规则能切出几档？为什么比 H100 少？

??? success "参考答案"
    粒度是 8，候选的 prefill SM 数是 8、16、……、64，还要满足 prefill 不少于一半（≥ 39，即 40 起）、decode 至少 16 个（prefill ≤ 62），只剩 40、48、56 三个，所以只有 (56, 22)、(48, 30)、(40, 38) 三档，加上两头的两组普通流一共 5 组。SM 少了，满足"decode 至少 16 个、prefill 不少于一半"的取值就少，档位自然变少；选档公式里的"可切的档数"也跟着变成 3。

**2. 选档的公式。** 默认配置下（H100，6 档），decode 请求数为多少时第一次选到 72 : 60 这一档？如果希望 decode 请求数到 24 就用这一档，可以怎么改配置？

??? success "参考答案"
    档位 = decode 请求数 × 6 // 36，要等于 6（上限），请求数至少 36。希望 24 个请求就用最后一档，可以把 `decode_bs_divisor` 改成 24（档位 = 请求数 × 6 // 24，24 个请求时正好是 6）；或者用 `manual_divisions` 为每一档直接写阈值，最后一档的阈值写 24。

**3. 按层切 prefill 的开销。** 一个 131072 token 的 prefill，每轮只算 1 层。对 36 层的模型，这意味着什么？会带来哪些额外开销？

??? success "参考答案"
    这个 prefill 要跨 36 轮调度循环才能做完，每轮只提交一层；层与层之间的隐藏状态和残差（各 131072 × 4096 × 2 字节 ≈ 1 GB）要一直留在显存里等下一轮。开销包括：每轮一次 CPU 调度和 decode 流的同步；prefill 流上的 kernel 在层与层之间可能出现空隙；中间激活占的显存。好处是每一轮的时间有上限，decode 不会被一个巨大的 prefill 拖住，切分也能在 prefill 进行中保持稳定。

## 小结

- [x] prefill 和 decode 在同一批卡上有三种共存方式：时间上交替（分块 prefill）、空间上切分（PD 复用）、分到不同的卡（PD 分离）。
- [x] green context 让一个流只在指定的 SM 上执行（9.0 起按 8 个 SM 切分）；只隔离 SM，带宽和 L2 仍共享，也不保证真的并发。
- [x] SGLang 启动时切出几档 SM 切分，运行时按 decode 请求数选档；prefill 按层切开，每轮算 `split_forward_token_budget // token 数` 层，和 decode 交替推进。
- [x] PD 复用用 prefill 用时（TTFT）换 decode 的 ITL 平稳；适合小规模部署、ITL 要求严、提示词长短差别大的负载。
