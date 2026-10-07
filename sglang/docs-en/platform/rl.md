# The RL training loop: weight synchronisation and the rollout interface

<p class="lead">The README's first sentence lists "RL rollouts" alongside agentic workloads and large-scale serving as the three things it is optimised for. In reinforcement-learning training an inference engine has to swap its weights every few steps, give its memory back while training runs and take it again to generate — needs that grew from "update weights without restarting" in August 2024 into <code>update_weights_from_tensor</code>, <code>release_memory_occupation</code>, <code>verl_engine.py</code> and <code>weight_sync/</code>. This chapter reads that set of interfaces in order, to see what an inference engine conceded in order to be embedded by a training framework.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. What special demands does RL training make of an inference engine? Which interface meets each?
    2. How do `update_weights_from_distributed` and `update_weights_from_tensor` differ? Which deployment shape suits each?
    3. Why does "release the memory and resume" have to be compatible with CUDA graphs?
    4. What does `verl_engine.py`'s SPMD mean? Why does the engine have to run inside a training framework's process?

??? success "Answers for the self-test (answer first, then open this)"
    1. Swapping the weights (after each training step), giving up the memory (the weights' and the KV's memory has to go back to the training process while it trains), running in the training framework's parallel layout (on the same machines and cards as the training processes), and the throughput of batch rollouts. The interfaces: `update_weights_from_disk` / `from_distributed` / `from_tensor`, `release_memory_occupation` / `resume_memory_occupation`, `verl_engine.py`'s SPMD mode, and the offline `Engine.generate`.
    2. `from_distributed`: the training and inference processes join one torch.distributed group, the training side broadcasts the parameters and the inference side receives them by name, which suits a separated deployment (training cards and inference cards apart). `from_tensor`: tensors are handed to the engine directly (in the same process, or through serialised buckets), which suits a colocated deployment; bucketing (`tensor_bucket.py`) was added in 2025 to cut the round trips for small tensors.
    3. A CUDA graph records fixed memory addresses when captured; if a release followed by an allocation returns different addresses the graph is invalid. #2630 uses something like torch_memory_saver to free the physical memory while the virtual addresses stay put, mapping back to the same addresses on resume so the graph can still be replayed.
    4. SPMD (single program, multiple data): each training rank holds an inference worker of its own, with no separate HTTP service or scheduler process; the engine is called as an ordinary object inside the processes torchrun started, and the parallel layout follows the training framework's. Updating the weights then needs no cross-process transfer, and the memory can be switched between training and inference directly.

A six-panel strip first, then the chapter:

<!-- comic ../assets/comics/rl.webp is in Chinese; put it back once the English version exists -->

## One set of interfaces, in order {#一组接口的时间线}

```bash title="rl-commits.sh"
for h in cd10654e7e 983bfcf386 fd28640dc5 9183c23eca 923f518337 e3e0bc50a9 bc92107b03 ce32bc2ba9 89588179cf 96a5e4dd79 21028b5507; do
  git log -1 --date=short --format='%ad  %h  %s' $h | cut -c1-96
done
```

```text title="output"
2024-08-20  cd10654e7e  [Feat] Support update weights without restart server (#1157)
2024-12-01  983bfcf386  Online weight updates from torch.distributed (#2279)
2024-12-29  fd28640dc5  Add `update_weights_from_tensor` (#2631)
2025-01-02  9183c23eca  Speed up `update_weights_from_tensor` (#2695)
2025-01-14  923f518337  CUDA-graph-compatible releasing and resuming KV cache and model weight m
2025-03-01  e3e0bc50a9  [Feature] SPMD for SGLang + Verl (#3852)
2025-04-13  bc92107b03  Support server based rollout in Verlengine (#4848)
2025-07-26  ce32bc2ba9  Extract update_weights from RL Engine to SGLang to keep simplicity and f
2025-08-05  89588179cf  [1/3] Optimize Slime Update Weights: Remove QWen3MOE Load Weight Overhea
2025-10-24  96a5e4dd79  [Feature] Support loading weights from ckpt engine worker (#11755)
2025-12-10  21028b5507  [RL] support weight reload for low-bit rollout (#9650)
```

- **2024-08-20 #1157**: `update_weights` (reloading from disk) without restarting the service — the earliest demand came from swapping a model version in production.
- **2024-12-01 #2279**: `update_weights_from_distributed` — the training process broadcasts the parameters over torch.distributed and the inference process receives them; the first time an RL framework could update online.
- **2024-12-29 #2631 and 2025-01-02 #2695**: `update_weights_from_tensor` and its speedup — no network when colocated.
- **2025-01-14 #2630**: releasing and resuming the KV cache's and the weights' memory, compatible with CUDA graphs — giving the memory to the training framework while it trains.
- **2025-03-01 #3852**: SPMD plus veRL, `entrypoints/verl_engine.py` — the engine running inside the training process.
- **2025-04-13 #4848**: server-based rollout (the training framework calling a remote engine over HTTP).
- **2025-07-26, the #8xxx series**: a `weight_sync/` directory — the weight-update tools (bucketed transfer) extracted from the RL engines into SGLang itself; August brought optimisations to slime's update path (removing Qwen3-MoE's loading overhead, avoiding device synchronisations).
- **2025-10-24 #11755**: loading weights from a checkpoint engine's worker; **2025-12-10 #9650**: reloading weights for low-bit rollouts.

## Three ways to update the weights {#三种更新权重的方式}

```bash title="update-weights-api.sh"
REF=${REF:-29f6d408c0}
echo "io_struct.py 里和权重更新有关的请求类型："
git show "$REF:python/sglang/srt/managers/io_struct.py" | grep -E '^class (Update|Init|Get|Release|Resume|Destroy).*(Weight|Memory|Parameter).*:' | sed 's/^class //; s/[(:].*//' | tr '\n' ' '; echo
echo "weight_sync/：$(git ls-tree -r --name-only "$REF" -- python/sglang/srt/weight_sync | sed 's|.*/||' | tr '\n' ' ')"
git show "$REF:python/sglang/srt/weight_sync/tensor_bucket.py" | grep -E '^class |^    def ' | sed 's/^ *//; s/(.*//' | tr '\n' ' '; echo
```

```text title="output"
io_struct.py 里和权重更新有关的请求类型：
UpdateWeightFromDiskReqInput UpdateWeightFromDiskReqOutput UpdateWeightsFromDistributedReqInput UpdateWeightsFromDistributedReqOutput UpdateWeightsFromTensorReqInput UpdateWeightsFromTensorReqOutput InitWeightsSendGroupForRemoteInstanceReqInput UpdateWeightsFromIPCReqInput UpdateWeightsFromIPCReqOutput InitWeightsSendGroupForRemoteInstanceReqOutput InitWeightsUpdateGroupReqInput InitWeightsUpdateGroupReqOutput DestroyWeightsUpdateGroupReqInput DestroyWeightsUpdateGroupReqOutput UpdateWeightVersionReqInput UpdateWeightVersionReqOutput GetWeightsByNameReqInput GetWeightsByNameReqOutput ReleaseMemoryOccupationReqInput ReleaseMemoryOccupationReqOutput ResumeMemoryOccupationReqInput ResumeMemoryOccupationReqOutput 
weight_sync/：tensor_bucket.py utils.py 
class FlattenedTensorMetadata: class FlattenedTensorBucket: def __init__ def get_flattened_tensor def get_metadata def reconstruct_tensors 
```

What the three have in common is that they all cross [chapter six](../service/processes.md)'s process structure: a request goes from the `TokenizerManager` to every scheduler process, and the scheduler passes it to `TpModelWorker` → `ModelRunner` to replace the parameters; what differs is how the data arrives: read from disk, received from a torch.distributed group, or carried in the request as serialised tensors (several at a time after bucketing). `update_weights_from_tensor` over ZMQ has to serialise whole weights, so `tensor_bucket.py` packs the small tensors and sends one large buffer with a `FlattenedTensorBucket` — one of the results of the #8751 "Optimize Slime Update Weights" series.

## Giving up memory: release and resume {#让出显存release--resume}

#2630's title, "CUDA-graph-compatible releasing and resuming KV cache and model weight memory", names the difficulty: when training and inference are colocated, a training step needs the memory and the inference engine has to free the KV pool's and the weights' memory and take it back afterwards; but a CUDA graph records addresses, and reallocating invalidates it. The approach is a memory saver that keeps the virtual addresses fixed (`torch_memory_saver`): a release returns only the physical pages, and a resume maps back to the same virtual addresses. `--enable-memory-saver` turns it on, and the `/release_memory_occupation` and `/resume_memory_occupation` interfaces are called before and after training. The `memory_saver_adapter.configure_subprocess()` in [chapter six](../service/processes.md)'s `_launch_scheduler_processes` is where that layer is put on as the scheduler processes start.

## Running inside the training framework's process {#在训练框架的进程里运行}

```bash title="verl-engine.sh"
REF=${REF:-29f6d408c0}
git show --stat=100 --format='%ad  %an  %s' --date=short e3e0bc50a9 | grep -v '^$' | cut -c1-96
echo "-- verl_engine.py @ v0.4.6 的方法："; git show v0.4.6:python/sglang/srt/entrypoints/verl_engine.py | grep -E '^class |^    def ' | sed 's/^ *//; s/(.*//' | tr '\n' ' '; echo
echo "-- verl_engine.py 的最后一个提交：$(git log -1 --date=short --format='%ad  %h  %s' "$REF" -- python/sglang/srt/entrypoints/verl_engine.py | cut -c1-80)"
```

```text title="output"
2025-03-01  fzyzcjy  [Feature] SPMD for SGLang + Verl (#3852)
 .github/workflows/pr-test.yml                               |   6 +
 examples/runtime/engine/offline_batch_inference_torchrun.py |  81 +++++++
 python/sglang/srt/entrypoints/engine.py                     |  19 +-
 python/sglang/srt/entrypoints/verl_engine.py                | 145 +++++++++++++
 python/sglang/srt/managers/data_parallel_controller.py      |   8 +-
 python/sglang/srt/managers/io_struct.py                     |   2 +
 python/sglang/srt/managers/scheduler.py                     |   5 +-
 python/sglang/srt/managers/tp_worker.py                     |   5 +-
 python/sglang/srt/model_executor/model_runner.py            |  45 +++-
 python/sglang/srt/models/gemma.py                           |   6 -
 python/sglang/srt/models/gemma2.py                          |   7 -
 python/sglang/srt/server_args.py                            |   8 +
 python/sglang/srt/utils.py                                  |   1 -
 python/sglang/test/runners.py                               | 364 ++++++++++++++++++++---------
 python/sglang/test/test_programs.py                         |   2 +-
 test/lang/test_srt_backend.py                               |   2 +-
 test/srt/models/test_generation_models.py                   |  61 ++----
 test/srt/test_update_weights_from_tensor.py                 |  28 +++
 test/srt/test_verl_engine.py                                | 297 ++++++++++++++++++++++++++
 19 files changed, 890 insertions(+), 202 deletions(-)
-- verl_engine.py @ v0.4.6 的方法：
class VerlEngine: def __init__ def generate def update_weights_from_tensor def release_memory_occupation def resume_memory_occupation def shutdown 
-- verl_engine.py 的最后一个提交：2025-06-19  1ab6be1b26  Purge VerlEngine (#7326)
```

`VerlEngine`'s idea: the training framework (veRL) starts N processes with torchrun for FSDP or Megatron training, each of which builds an SGLang `Engine` whose `tp_size` matches the training's parallel layout, with the `device_mesh` passed straight in; generation calls `generate` SPMD-style across the processes (the input broadcast among the ranks, the output gathered), and the weight update happens in-process with `update_weights_from_tensor`. That turns "the inference engine" into an object inside a training program rather than a service. April's #4848 added the opposite shape too: the training framework calling a remote SGLang server over HTTP for its rollouts (suiting training cards and inference cards apart). Three and a half months later, #7326 "Purge VerlEngine" of 2025-06-19 deleted the wrapper class (the script's last line): the capabilities SPMD needed (constructing an `Engine` inside an external process group, broadcasting the input by rank) went into `Engine` itself and the framework-specific wrapper went back to veRL's own repository — the engine keeps only the general interface and no longer maintains a class for one particular framework.

![Figure: the inference engine inside the RL training loop](../assets/figures/sgl-rl-loop.svg){.aig-svg}

## Design trade-offs {#设计取舍}

- **Three update paths coexisting.** Each matches a deployment shape (separated / colocated / from disk), with one interface (all going through `io_struct`'s requests, the scheduler and `ModelRunner`) and different transports. The price is keeping all three in step on the special weights for quantization and MoE experts (the #8751 series, #13323's ue8m0 scale).
- **The engine is embeddable, but no wrapper for one framework.** `Engine` does not depend on HTTP and can run SPMD inside someone else's processes; `VerlEngine` lived three and a half months before being deleted, with the general capability staying in `Engine` and the framework-specific part staying in the framework. The price is that the engine has to tolerate "the parallel layout decided outside" and "being asked to give up memory at any time".
- **Pull the tools back into the project.** `weight_sync/` came from an RL engine's code (slime) and, pulled into SGLang, is shared by every framework rather than written once per framework.

## What happened afterwards {#后来怎么样了}

- H2 2025: slime, AREAL and veRL are all on the roadmap's (issue #7736) collaboration list; the update path for MoE expert weights, reloading weights for FP8 and low-bit rollouts (#9650), and loading directly from a checkpoint engine (`checkpoint_engine/`).
- 2026: `weight_cache/` (#27139, 2026-07-25) lets the engine restore its weights quickly from a cache after a restart; a `ray/` directory provides engine orchestration on Ray.
- The distributed-training handbook's [training frameworks and RL systems chapter](train://practice/frameworks-rl/) and the inference-systems handbook's [RL rollout chapter](serving://topics/rl-rollout/) cover the same loop from the training side and the systems side.

## Exercises {#练习}

**1. What bucketing buys.** Read the baseline commit's `weight_sync/tensor_bucket.py` and say how `FlattenedTensorBucket` packs several tensors into one buffer, how the receiver restores them, and why this matters particularly for a MoE model.

??? success "A way to approach it"
    It records each tensor's name, shape, dtype and offset in order and concatenates them into a one-dimensional buffer sent at once; the receiver slices by the metadata and `view`s each back into shape. A MoE has hundreds or thousands of expert weights, many of them small, and without bucketing each would take its own round trip.

**2. What is left after a release.** After `release_memory_occupation`, what memory is still held in the engine's process? Why not free the CUDA graphs and recapture them?

??? success "A way to approach it"
    The CUDA context, the graphs' memory pool and a little workspace remain; recapturing takes tens of seconds to minutes, and RL switches once per training step, which is unaffordable.

**3. What SPMD costs.** Read v0.4.6's `verl_engine.py` and find the broadcasts and gathers in `generate` that make every rank return the same result.

??? success "A way to approach it"
    The input is broadcast within the tp group (only rank 0 really receives the data) and the output is identical on every rank; note that `update_weights_from_tensor` is called on each rank separately.

!!! interview "How to answer in an interview"
    "How does an inference engine support RL training?" — Answer in three parts: swapping the weights (from disk, over torch.distributed, or bucketed tensors, matching three deployment shapes), giving up the memory (a release and resume that keeps the virtual addresses, which is what lets it coexist with CUDA graphs), and being embeddable (running SPMD inside the training processes, or serving remotely). Then say these interfaces were added step by step from 2024-08 to 2025-07, and that the tools from the RL frameworks (the bucketed transfer) were pulled back into the engine itself.

## Summary {#小结}

- [x] From "swap the weights without a restart" in 2024-08 to 2025's three update paths, memory release and resume, and SPMD embedding, RL's demands shaped a set of dedicated interfaces.
- [x] Compatibility between CUDA graphs and freeing memory rests on a memory saver that keeps the virtual addresses fixed.
- [x] `verl_engine.py` (2025-03 → 06) made the engine an object inside a training program and its capabilities then merged into `Engine`; `weight_sync/` pulled the frameworks' tools back into the project for slime, AREAL and veRL to share.
