# Hot weight updates: changing weights without restarting

<p class="lead">Some scenarios require an inference engine to swap its weights without restarting: RL training sends the new policy into the rollout engine every step, production services swap in a fine-tuned new version, and elastic EP restores lost experts. A restart means reloading hundreds of GB of weights, recapturing CUDA Graphs and warming up again, while a hot update hopes to spend only the time to transfer the weights. This chapter explains the three difficulties of hot updates: what to do with in-flight requests and caches (consistency), the checkpoint format differing from the format kernels use (post-load processing), and CUDA Graphs remembering addresses (updates must be in place); then it looks at the interfaces vLLM and SGLang offer and where the weights come from.</p>

!!! question "Self-test: if you can answer these, skip this chapter"
    1. When updating weights, what are the ways to handle requests mid-generation? What does each cost?
    2. Why can't tensors from the checkpoint be copied straight into the model's parameters?
    3. Why must weight updates be "in-place copies", rather than pointing parameters at new GPU memory?
    4. Should the prefix cache be cleared after an update? Why?
    5. How do vLLM and SGLang each get weights into the inference engine?

??? success "Answers (try first, then expand to compare)"
    1. Abort: the update starts immediately, but what was generated is discarded and must be redone; wait: nothing wasted, but it waits for the longest request to finish, and the long tail can keep the update waiting for minutes; freeze (vLLM's keep, SGLang's in_place): neither waiting nor waste, but the first half of these requests was generated with the old weights, and their KV was computed with the old weights too; retract to the queue (SGLang's retract): re-prefill with the new weights after the update, so the KV is new, but already generated tokens still come from the old weights.
    2. The checkpoint is in the training side's format (full floating-point tensors, separated by module), while kernels use a processed format: split for tensor parallelism, QKV / gate_up merged, FP8 block quantization with scales, int4 reordering, redundant replicas of MoE experts... The update must go through the model's own loading flow (`load_weights` and post-load processing); otherwise either the shapes don't match, or values are silently converted wrong (for example, floats copied into an int8 buffer get truncated).
    3. CUDA Graph capture records each tensor's address, and replay reads only those addresses; descriptors prepared in advance inside kernels (such as TMA descriptors) are the same. Once a parameter points at new memory, the model's Python code sees the new weights, but the replayed graph still reads the old memory, producing wrong results with no error. So updates must copy the new values into the original storage, keeping the address unchanged.
    4. Generally yes: the KV in the prefix cache was computed with the old weights, and a new request hitting it effectively processes its prefix with the old weights. For RL this mixes the old policy into rollouts; for a production version change, outputs of the old and new versions get mixed.
    5. vLLM: weight transfer engines (`nccl` broadcast, `ipc` shared GPU memory, `sparse_nccl` sending only changed elements, `sharded_rdt` with each worker pulling only its own shard), grouped by layer and packed into fixed-size buffers for pipelined transfer, with each layer post-processed and copied in place once it is complete. SGLang: `/update_weights_from_disk` (swap from disk), `/update_weights_from_tensor` and `/update_weights_from_ipc` (passing tensors or memory handles when co-located), `/init_weights_update_group` + `/update_weights_from_distributed` (NCCL broadcast), and an integration of the dedicated checkpoint-engine.

## When hot updates are needed {#什么时候需要热更新}

| Scenario | How often | What matters most |
| --- | --- | --- |
| Weight sync in RL training | every training step (tens of seconds to minutes) | updates must be fast (transfer time directly affects GPU utilization; see [the time budget of asynchronous RL](../frontier/rl-async.md#权重同步的时间账)), and generated data must know which policy version it came from |
| Switching production to a fine-tuned new version | every few days to weeks | no service interruption, gradual rollout, rollback |
| Elastic EP restoring lost experts | on failures | update only part of the weights, the faster the better (see [fault tolerance for large-scale EP](../moe/ep-elastic.md)) |

If the architecture changes (number of layers, hidden size, quantization scheme), only a restart will do, with instances replaced by a [release strategy](deploy.md#发布策略); if the architecture stays and only the weights change, a hot update works. Dynamically loading adapters in multi-LoRA serving (see [multi-LoRA](multi-lora.md)) is a lighter form: the base stays, and only the small adapter matrices change.

![Figure: changing weights without restarting: pause scheduling between two steps, convert the format, then write in place into the same GPU memory](../assets/figures/weight-update-flow.svg){.aig-svg}

## Difficulty one: in-flight requests and caches {#难点一在途请求和缓存}

When an update happens, the engine has requests mid-generation. Swapping weights halfway through a forward pass is certainly out (half the layers old, half new), so updates always happen between two steps, with the scheduler paused first. There are four ways to handle in-flight requests during the pause (vLLM's `pause_generation(mode=...)` supports `abort`, `wait` and `keep`; SGLang's `/pause_generation` supports `abort`, `retract` and `in_place`):

```python
import random

rng = random.Random(0)
STEP_MS, PROMPT, N = 25, 2048, 64                # assumptions: 25 ms per decode step, 2048-token prompts, 64 in-flight requests
done = [rng.randint(1, 4000) for _ in range(N)]                                   # tokens already generated
left = [min(16000, int(rng.lognormvariate(6.5, 1.2))) for _ in range(N)]          # tokens still to generate (long-tailed)


def cell(text, width):
    """按显示宽度右对齐（汉字占两格）"""
    return " " * (width - sum(2 if ord(ch) > 0x2E7F else 1 for ch in str(text))) + str(text)


rows = [  # approach, wait before updating, tokens to re-decode, tokens to re-prefill, requests mixing old and new weights, whether KV matches the new weights
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

```text title="output"
64 个在途请求：已生成 139822 个 token；还要生成的最多 16000 个，中位数 466 个
                        做法  更新前要等  重新 decode  重新 prefill  混合版本的请求  KV 用新权重
         abort：中止在途请求         0 s       139822        131072               0           是
        wait：等在途请求做完       400 s            0             0               0           是
     keep：冻结，接着用旧 KV         0 s            0             0              64           否
  retract：退回队列，重算 KV         0 s            0        270894              64           是
```

- **abort**: the update starts immediately, at the cost of discarding over a hundred thousand generated tokens, and these requests start over;
- **wait**: nothing wasted, but the long tail kept the update waiting 400 seconds, during which new requests can't get in either and the GPU grows ever idler. In RL this is exactly the problem of "synchronous RL held back by the long tail" (see [inference in RL training](../topics/rl-rollout.md));
- **keep / in_place**: fastest and wasting nothing, but the first half of these 64 requests was generated with the old weights, and the old KV keeps being used with the new weights. For RL, a trajectory mixes two policy versions, so the training side must record versions per token and correct with importance sampling, or drop these trajectories;
- **retract**: return requests to the waiting queue and free their KV, then re-prefill with the new weights after the update (prefill is parallel, much faster than re-decoding), so the KV matches the new weights, though already generated tokens still come from the old weights.

Also, **all the KV in the prefix cache was computed with the old weights**. vLLM's `pause_generation` defaults to `clear_cache=True`, clearing the KV and the prefix cache together; SGLang can clear the cache in retract mode and recompute automatically afterwards. Tagging generated results with a version number matters too: both vLLM and SGLang can set a weight version (`update_weight_version`), from which RL frameworks tell which policy version each piece of data came from.

## Difficulty two: the checkpoint format is not the kernel format {#难点二检查点格式不等于-kernel-格式}

What the training side sends is weights in checkpoint format: full floating-point tensors, named by module. The inference engine's parameters were processed at load time: split for tensor parallelism, QKV and gate_up merged into single matrices, FP8 quantized with block scales computed, int4 weights reordered into the layout kernels need, MoE experts replicated across GPUs by EPLB... A direct copy either fails to match shapes or, worse, matches shapes while values are silently converted wrong. Below we simulate an int8-quantized layer (in real systems it is post-load processing like FP8 block scaling):

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
w0, w1 = torch.randn(64, 64) * 0.05, torch.randn(64, 64) * 0.05     # old weights, the new checkpoint (checkpoint format: float)
lin = Int8Linear(w0)
ptrs = (lin.qweight.data_ptr(), lin.scale.data_ptr())
x = torch.randn(8, 64)
ref = x @ w1.T


def rel_err():
    return ((lin(x) - ref).norm() / ref.norm()).item()


lin.qweight.copy_(w1)                            # wrong: copying floats straight into the int8 buffer silently truncates them to 0, and the scales are still the old ones
print(f"直接拷贝检查点：相对误差 {rel_err():.3f}")
q, s = quantize(w1)                              # right: reprocess the way loading does, then copy in place into the two original buffers
lin.qweight.copy_(q)
lin.scale.copy_(s)
print(f"重新量化后原地拷贝：相对误差 {rel_err():.4f}（只剩量化误差），地址没变：",
      (lin.qweight.data_ptr(), lin.scale.data_ptr()) == ptrs)
```

```text title="output"
直接拷贝检查点：相对误差 1.000
重新量化后原地拷贝：相对误差 0.0063（只剩量化误差），地址没变： True
```

So updates must follow the same flow as loading: the model's `load_weights` handles splitting, merging and copying to each expert replica, and post-load processing (in vLLM, the quantization method's `process_weights_after_loading`) handles quantization and reordering. vLLM's "layerwise reload" (`vllm/model_executor/model_loader/reload/layerwise.py`) does this carefully:

1. Before the update starts, record the tensors each layer's kernels currently use;
2. Restore each layer's parameters to checkpoint-format "shells" (on the meta device, taking no GPU memory), and wrap each parameter's `weight_loader` so processing is deferred until all of the layer's weights have arrived;
3. Once a layer is complete: materialize the layer on the device, load the cached weights, run processing such as quantization, then **copy the processed values in place back into the tensors recorded in step 1**;
4. Finally, handle layers that need deferring, such as attention layers (for example the KV scales).

Only one layer is expanded at a time, so the extra memory is only one layer's worth; the in-place copy keeps the addresses that kernels and CUDA Graphs see unchanged, which is the next difficulty.

## Difficulty three: CUDA Graphs remember addresses {#难点三cuda-graph-记住的是地址}

When a CUDA Graph is captured, each kernel's arguments (including pointers to the weights) are recorded, and replay reads only those addresses (see [CUDA Graphs and torch.compile](../engine/graphs-compile.md)). If an update points parameters at newly allocated memory, the model's Python code sees the new weights, but the replayed graph still reads the old memory: the result is wrong, with no error. Below, a closure that remembers the weights' storage simulates a captured graph:

```python
import torch

torch.manual_seed(0)
layer = torch.nn.Linear(4, 4, bias=False)
x = torch.randn(1, 4)


def capture(layer):
    """模拟 CUDA Graph：捕获时记下的是权重张量的存储（在 GPU 上就是一个固定的地址），之后重放只读这块存储"""
    w = layer.weight.detach()                    # shares the parameter's storage
    return lambda inp: inp @ w.T


graph = capture(layer)
new_w = torch.randn(4, 4)

# wrong update: point the parameter at new storage. The model itself uses the new weights, but the captured graph still reads the old storage
layer.weight.data = new_w.clone()
print("重新绑定：模型输出 == 新权重的结果：", torch.allclose(layer(x), x @ new_w.T),
      "；图的输出 == 新权重的结果：", torch.allclose(graph(x), x @ new_w.T))

# right update: copy in place into the original storage; the address is unchanged, and both the graph and the model see the new weights
graph = capture(layer)                           # recapture (equivalent to the state after a restart)
newer_w = torch.randn(4, 4)
ptr = layer.weight.data_ptr()
layer.weight.data.copy_(newer_w)
print("原地拷贝：图的输出 == 新权重的结果：", torch.allclose(graph(x), x @ newer_w.T),
      "；存储地址没变：", layer.weight.data_ptr() == ptr)
```

```text title="output"
重新绑定：模型输出 == 新权重的结果： True ；图的输出 == 新权重的结果： False
原地拷贝：图的输出 == 新权重的结果： True ；存储地址没变： True
```

After `layer.weight.data = new`, the model's own output is right, but the "graph" still outputs results from the old weights; `layer.weight.data.copy_(new)` writes the new values into the original storage, and both are right.

A few practical rules:

- **Update only in place**: `param.data.copy_(new_value)`, with shape, dtype and device unchanged;
- **Processed buffers must be updated in place too**: quantization scales, reordered weights, and the buffers pointed to by prebuilt TMA descriptors all count as "addresses the graph remembers";
- **A changed shape requires recapture**: for example, switching to a version with a larger vocabulary is no longer a hot update;
- **Free temporary memory after updating**: temporary tensors from loading and processing stay in PyTorch's caching allocator, so SGLang's `/continue_generation` calls `torch.cuda.empty_cache()` once by default before resuming inference.

## Where the weights come from {#权重从哪里来}

**vLLM** has a weight transfer engine (`vllm/distributed/weight_transfer/`), with the flow `init_weight_transfer_engine` → `start_weight_update` → several `update_weights` calls → `finish_weight_update`, optionally attaching a new weight version. There are four backends (`WeightTransferConfig.backend`):

- `nccl`: the training and inference sides build one NCCL group and broadcast the full weights;
- `ipc`: when training and inference are co-located on the same GPUs, share tensors in GPU memory directly through CUDA IPC handles;
- `sparse_nccl`: send only the changed elements (names, shapes and flattened indices), mapped onto this rank's parameters by the model's own `weight_loader`;
- `sharded_rdt`: Ray direct transport, with each worker pulling only the shard it needs under tensor and expert parallelism rather than full tensors. The first update walks through `load_weights` with fake tensors, recording where each shard comes from and where it lands, and every later update replays the record.

Transfers are grouped by decoder layer (`layerwise_groups`) and packed into fixed-size buffers (two 1 GB buffers used in turn by default): receive a layer, process a layer, with transfer and processing pipelined.

**SGLang**'s interfaces are all HTTP endpoints (with corresponding Python methods):

- `/update_weights_from_disk`: swap the weights in place from a new checkpoint on disk without restarting the service, returning the number of paused requests;
- `/update_weights_from_tensor`, `/update_weights_from_ipc`: pass tensors or CUDA IPC handles when co-located, packing multiple tensors into one flat buffer first (`srt/weight_sync/tensor_bucket.py`);
- `/init_weights_update_group` + `/update_weights_from_distributed`: build an NCCL group with the training side and broadcast in batches;
- `/init_weights_send_group_for_remote_instance`: have an already serving instance (a seed instance) send its weights to a newly started instance, so the new one doesn't read from disk (the seed is specified at startup with flags like `--remote-instance-weight-loader-seed-instance-ip`);
- `srt/checkpoint_engine/`: integrates the open-source checkpoint-engine; the service starts with dummy weights (`--load-format dummy`, with `--checkpoint-engine-wait-weights-before-ready` to become ready only once weights arrive), and the engine pushes the checkpoint in by broadcast or point-to-point.

The time budget of the transfer itself (NVLink when co-located, pipelined relays between instances when disaggregated) is in [weight sync in asynchronous RL](../frontier/rl-async.md#权重同步的时间账).

## Changing versions in production {#线上换版本的做法}

RL pursues speed; production version changes pursue stability. A workable flow:

1. **Gradual rollout**: hot-update one instance first (from disk or by pulling from an already updated instance), compare outputs on a fixed set of requests (token-by-token comparison and evaluation sets, methods in [onboarding new models and aligning accuracy](new-model.md)), then gradually extend to the other instances;
2. **Consistency**: pause scheduling and clear the prefix cache before changing versions; when multi-turn sessions stick to an instance, decide whether those sessions finish on the old version or switch to the new one as a whole;
3. **Rollback**: a hot update doesn't keep two sets of weights at once (they won't fit in memory), so rolling back is another hot update (from the old checkpoint), which means the old checkpoint must stay somewhere it can be read quickly;
4. **Monitoring**: compare latency, throughput, output length and quality metrics before and after the update, rolling back automatically on anomalies.

!!! interview "How to explain it"
    To explain "how do you update an inference engine's weights without restarting": first the scenarios (per-step sync in RL, production version changes, restoring experts), then the three difficulties. Consistency: updates happen between two steps; in-flight requests can be aborted, waited for, frozen or retracted to the queue, each with a cost (waste, long-tail waiting, mixed old and new versions, recomputing prefill); the prefix cache must be cleared, and generated results should carry a version number. Format: the checkpoint format must go through the model's own `load_weights` and post-load processing (splitting, merging, quantization, reordering) to become the kernel format, with vLLM expanding, processing and copying back layer by layer. Addresses: CUDA Graphs remember pointers, so only in-place `copy_` works, never swapping in newly allocated memory. Finally the transfer: CUDA IPC when co-located, NCCL broadcast or pulling only your own shard when disaggregated, grouped by layer and packed into buffers for pipelined transfer.

## Exercises {#练习}

**1. Which pause mode?** An RL system uses "partial rollout": at each update, unfinished trajectories are not discarded but continue generating after the update, and training corrects each token by version. Which pause mode should it use? And if the training side requires each trajectory to come from a single version?

??? success "Answer"
    Partial rollout wants "no waste, no waiting", so keep (vLLM) or in_place (SGLang) is most direct: requests freeze in place and continue after the update, with each token recording the weight version that generated it. The cost is that the old KV keeps being used with the new weights, a bias the training side's version correction also absorbs; if you want the KV to match the new weights, use retract, at the cost of re-prefilling. If each trajectory must come from a single version, you can only choose between abort (regenerate) and wait (wait out the long tail), or finish unfinished trajectories on an old-version engine (needing two engines or more complex scheduling).

**2. Find the bug.** Someone implemented a hot update as `for name, t in new_weights: model.get_parameter(name).data = t.to("cuda")`. Tests pass without CUDA Graphs, but outputs are wrong once CUDA Graphs are on. Why? What other problems are there?

??? success "Answer"
    `.data = ...` points the parameter at newly allocated memory; with CUDA Graphs on, the replayed graph still reads the old address, so it's wrong; without CUDA Graphs, kernels are launched afresh every time and read the parameter's current address, so it looks "fine". It should be `param.data.copy_(t)`. Other problems: assigning directly by name bypasses the model's `load_weights`; the checkpoint's `q_proj`, `k_proj` and `v_proj` were long since merged into `qkv_proj` on the inference side, each rank holds only a shard under tensor parallelism, and quantized models need re-quantization; mismatched names raise errors, and matched ones may still have wrong values; also, the old memory is freed while the graph still references it and may be reused by other tensors, causing even harder-to-find bugs.

**3. How long does an update take?** A 32B BF16 model is split across 4 GPUs (tensor parallelism), co-located with the training side. When updating via CUDA IPC, how much data does each GPU copy? If the inference side uses FP8 while training uses BF16, what else must this step do?

??? success "Answer"
    32B × 2 bytes = 64 GB, so 16 GB per GPU across 4 GPUs. When co-located, IPC handles read the training side's GPU memory directly, copying over NVLink or on-GPU memory bandwidth, so 16 GB per GPU takes on the order of a second. With FP8 on the inference side, re-quantize after copying (or before sending on the training side) the way the inference side quantizes: compute each block's scale, convert to FP8, then copy in place into the buffers the kernels use; with kernels like DeepGEMM that require a special scale format (such as UE8M0), also pack the scales in that format.

## Summary {#小结}

- [x] Scenarios for hot updates: per-step weight sync in RL, production version changes, and elastic EP restoring experts; a changed architecture requires a restart.
- [x] Consistency: updates happen between two steps; in-flight requests can be aborted, waited for, frozen or retracted to the queue, each at a different cost; clear the prefix cache and tag results with a version number.
- [x] Format: checkpoints must go through `load_weights` and post-load processing (splitting, merging, quantization, reordering); vLLM expands, processes and copies back in place layer by layer, with only one layer of extra memory.
- [x] Addresses: CUDA Graphs remember pointers, so only in-place `copy_` works, for processed buffers too.
- [x] Transfer: vLLM's `nccl` / `ipc` / `sparse_nccl` / `sharded_rdt`, grouped by layer and pipelined through packed buffers; SGLang's updates from disk, tensors, IPC, distributed groups and remote instances, plus checkpoint-engine.
