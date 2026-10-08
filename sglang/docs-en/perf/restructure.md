# The big directory reorganisation: mem_cache, model_executor and the attention backends

<p class="lead">In the two months from late July to early October 2024, SGLang's runtime was cut up afresh: the four directories <code>mem_cache/</code>, <code>model_executor/</code>, <code>sampling/</code> and <code>layers/attention/</code> appeared in that time, <code>InputMetadata</code> was renamed <code>ForwardBatch</code>, the batch data structure split into three — a scheduling layer, a worker layer and a forward layer — and attention went from "one class choosing a kernel with if-else" to a set of pluggable backends. Not one of these changes came with a performance number, yet they decided where every new feature of the next two years would hang. This chapter reads the reorganisation as a story of how module boundaries get discovered.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. Which layer do `ScheduleBatch`, `ModelWorkerBatch` and `ForwardBatch` each belong to, and what does each hold? Why three?
    2. What methods does the `AttentionBackend` interface have? Why are half of them about CUDA graphs?
    3. What did #807 of 2024-07-29 move where? Which parts of today's directory structure come from it?
    4. What problem does moving sampling into `sampling/` solve, and what problem does moving the allocation state to the CPU (#1557) solve?

??? success "Answers for the self-test (answer first, then open this)"
    1. `ScheduleBatch` (`managers/schedule_batch.py`) lives in the scheduler: the list of request objects, the prefix-matching results and the metadata for admission and retraction, all Python objects on the CPU side. `ModelWorkerBatch` is the minimum necessary set the scheduler hands the worker (input_ids, seq_lens, slots, sampling information…), passable across threads or processes and carrying none of the scheduler's private state. `ForwardBatch` (`model_executor/forward_batch_info.py`) is the tensors and attention metadata the worker prepares on the GPU for one forward pass. Three structures for three owners, so that under overlapped scheduling the scheduling thread and the forward thread each modify their own without interfering.
    2. `init_forward_metadata` (prepare the metadata before each forward pass), `init_cuda_graph_state`, `init_forward_metadata_capture_cuda_graph`, `init_forward_metadata_replay_cuda_graph`, `get_cuda_graph_seq_len_fill_value`, and `forward_decode` / `forward_extend`. A CUDA graph requires fixed buffers and a replayable metadata update, and each backend does it differently (FlashInfer's indptr and indices buffers, Triton's kv_indptr), so the interface leaves "how to fill it while capturing" and "how to fill it while replaying" to the backend.
    3. `managers/controller/` was flattened into `managers/` (`controller_single.py`, `controller_multi.py`, `tp_worker.py`, `schedule_batch.py`, `policy_scheduler.py`), `radix_cache.py` and `memory_pool.py` went into `mem_cache/`, and `model_runner.py` and `cuda_graph_runner.py` into `model_executor/`. Today's `mem_cache/`, `model_executor/`, `managers/schedule_batch.py` and `managers/schedule_policy.py` all come from it.
    4. `sampling/` (2024-08-08) split the sampling parameters, the penalties and the batch sampling information out of `infer_batch.py` so that sampling could go into a CUDA graph (#1201) and support more parameters; #1557 moved the slots' free state from a GPU tensor (scanned with `torch.nonzero`) to a free list on the CPU, so allocation no longer needs a GPU synchronisation.

A six-panel strip first, then the chapter:

<!-- comic ../assets/comics/restructure.webp is in Chinese; put it back once the English version exists -->

## Two months of commits {#两个月的提交清单}

```bash title="restructure-commits.sh"
for h in cdcbde5fc3 87e8c090e9 ab7875941b 75ce37f401 3a6e8b6d78 fec185ce0c 3efa798116 f86c1e611f 3f0fe08d37 36d5acfca5 63ba2f8d7b 99ec439da4 f202ed9712 4ae0969c0a 32eb6e96f2; do
  git log -1 --date=short --format='%ad  %h  %s' $h | cut -c1-96
done
```

```text title="output"
2024-07-29  cdcbde5fc3  Code structure refactor (#807)
2024-08-06  87e8c090e9  Organize code (rename, movement) (#953)
2024-08-08  ab7875941b  feat: frequency, min_new_tokens, presence, and repetition penalties (#97
2024-08-26  75ce37f401  Move sampler into CUDA graph (#1201)
2024-09-10  3a6e8b6d78  [Minor] move triton attention kernels into a separate folder (#1379)
2024-09-11  fec185ce0c  Refactor attention backend (#1381)
2024-09-12  3efa798116  Support cuda graph in the triton attention backend (#1401)
2024-09-29  f86c1e611f  Move scheduler code from tp_worker.py to scheduler.py (#1538)
2024-09-29  3f0fe08d37  Let ModelRunner take InputMetadata as input, instead of ScheduleBatch (#
2024-09-30  36d5acfca5  Rename InputMetadata -> ForwardBatch (#1543)
2024-09-30  63ba2f8d7b  Clean up batch data structures: Introducing ModelWorkerBatch (#1544)
2024-09-30  99ec439da4  Organize Attention Backends (#1547)
2024-10-01  f202ed9712  [Refactor] Simplify io_struct and tokenizer_manager (#1549)
2024-10-02  4ae0969c0a  Move status check in the memory pool to CPU (#1557)
2024-10-03  32eb6e96f2  Organize sampling batch info better (#1562)
```

`srt/`'s top-level directories in three versions:

```bash title="srt-dirs.sh"
for t in v0.2.0 v0.3.0 v0.4.0; do
  echo "== $t"
  git ls-tree -r --name-only $t -- python/sglang/srt | grep '\.py$' | awk -F/ '{print (NF > 4 ? $4 "/" : $4)}' | sort | uniq -c | sort -rn | awk '{printf "   %3d %s\n", $1, $2}'
done
```

```text title="output"
== v0.2.0
    22 models/
    11 managers/
     9 layers/
     4 constrained/
     2 openai_api/
     2 model_loader/
     1 utils.py
     1 server_args.py
     1 server.py
     1 sampling_params.py
     1 model_config.py
     1 mm_utils.py
     1 memory_pool.py
     1 hf_transformers_utils.py
     1 flush_cache.py
     1 conversation.py
== v0.3.0
    25 models/
    12 layers/
     8 sampling/
     8 managers/
     5 mem_cache/
     4 constrained/
     3 model_executor/
     2 openai_api/
     2 configs/
     1 utils.py
     1 server_args.py
     1 server.py
     1 model_config.py
     1 mm_utils.py
     1 hf_transformers_utils.py
     1 conversation.py
== v0.4.0
    41 models/
    26 layers/
    13 distributed/
    11 managers/
     8 sampling/
     6 configs/
     5 mem_cache/
     5 constrained/
     4 model_loader/
     3 model_executor/
     3 lora/
     2 openai_api/
     2 metrics/
     1 utils.py
     1 server_args.py
     1 server.py
     1 model_parallel.py
     1 mm_utils.py
     1 hf_transformers_utils.py
     1 conversation.py
     1 _custom_ops.py
```

v0.2.0 has only `managers/`, `models/`, `layers/`, `constrained/` and a scattering of top-level files; v0.3.0 adds `mem_cache/`, `model_executor/`, `sampling/` and `configs/`; v0.4.0 adds `distributed/`, `lora/`, `metrics/` and `model_loader/`. Seen from today's directories, August and September 2024 are when the skeleton settled.

## Cut one: directories by lifetime (#807, #953) {#第一刀按生命周期分目录807953}

[Chapter six](../service/processes.md) covered #807's flattening of `controller/`. Its deeper meaning is dividing the directories by an **object's lifetime**:

| Directory | What it holds | Lifetime |
| --- | --- | --- |
| `managers/` | the processes, the scheduling, the requests and the batches | follows a request |
| `mem_cache/` | the radix tree, the memory pool | the service's whole life, shared across requests |
| `model_executor/` | `ModelRunner`, CUDA graphs, the forward metadata | follows one forward pass |
| `layers/` | the layers inside a model (attention, linear, sampling…) | stateless, or weights only |
| `sampling/` (08-08) | the sampling parameters, the penalties, the batch sampling information | follows a batch, but independent of the scheduling |

A week later #953 "Organize code (rename, movement)" moved `InputMetadata` into `model_executor/forward_batch_info.py` — it would not be renamed for another two months, but its place was right first.

## Cut two: the attention backends (#1379, #1381, #1547) {#第二刀注意力后端137913811547}

The first version's `RadixAttention` switched between two sets of kernels with `if use_flashinfer` ([chapter two](../origins/first-commit.md)), and MLA's arrival added another set of branches. #1381 "Refactor attention backend" of 11 September shrank the 175-line `RadixAttention` to under 50 and created `attention_backend.py` (383 lines) to take over everything kernel-related; #1547 of 30 September split that into a directory: `layers/attention/__init__.py` defines the interface, `flashinfer_backend.py` and `triton_backend.py` are one implementation each, and the Triton kernels moved into `triton_ops/`. The interface at v0.4.0:

```python title="python/sglang/srt/layers/attention/__init__.py @ v0.4.0 L11-82" linenums="11"
class AttentionBackend(ABC):
    """The base class of attention backends"""

    @abstractmethod
    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Init the metadata for a forward pass."""
        raise NotImplementedError()

    def init_cuda_graph_state(self, max_bs: int):
        """Init the global shared states for cuda graph."""
        raise NotImplementedError()

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor] = None,
    ):
        """Init the metadata for a forward pass for capturing a cuda graph."""
        raise NotImplementedError()

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: Optional[torch.Tensor] = None,
    ):
        """Init the metadata for a forward pass for replying a cuda graph."""
        raise NotImplementedError()

    def get_cuda_graph_seq_len_fill_value(self):
        """Get the fill value for padded seq lens. Typically, it is 0 or 1."""
        raise NotImplementedError()

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
    ):
        """Run forward on an attention layer."""
        if forward_batch.forward_mode.is_decode():
            return self.forward_decode(q, k, v, layer, forward_batch)
        else:
            return self.forward_extend(q, k, v, layer, forward_batch)

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
    ):
        """Run a forward for decode."""
        raise NotImplementedError()

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
    ):
        """Run a forward for extend."""
        raise NotImplementedError()
```

Four of the eight methods (including the dispatching `forward`) are about CUDA graphs, which is no accident: [chapter nine](../service/v02.md) explained that a CUDA graph requires fixed buffers, and attention is the only layer that needs metadata varying with the batch (each request's KV positions and lengths). Leaving "how to prepare the metadata while capturing and how to update it while replaying" to the backend means `CudaGraphRunner` does not have to know a backend's details. `forward` only dispatches by mode to `forward_decode` or `forward_extend`, and the `RadixAttention` layer is left with nothing but a call to `forward_batch.attn_backend.forward(q, k, v, self, forward_batch)`.

With that interface in place, #1401 of 12 September gave the Triton backend CUDA graph support too, and v0.4.0 added `torch_native_backend.py` (pure PyTorch, for debugging and platforms without kernels) and `double_sparsity_backend.py` (a sparse-attention experiment). How the backend count grew afterwards:

```bash title="attention-backends.sh"
REF=${REF:-29f6d408c0}
for t in v0.4.0 v0.4.6 v0.5.0rc0 "$REF"; do
  printf '%-11s %3d 个后端文件：' "$t" "$(git ls-tree -r --name-only "$t" -- python/sglang/srt/layers/attention | grep -c '_backend\.py$')"
  git ls-tree -r --name-only "$t" -- python/sglang/srt/layers/attention | grep '_backend\.py$' | sed 's|.*/||; s|_backend\.py||' | tr '\n' ' ' | cut -c1-110; echo
done
```

```text title="output"
v0.4.0        4 个后端文件：double_sparsity flashinfer torch_native triton 

v0.4.6        8 个后端文件：base_attn double_sparsity flashattention flashinfer flashinfer_mla flashmla torch_native triton 

v0.5.0rc0    17 个后端文件：aiter ascend base_attn cutlass_mla double_sparsity dual_chunk_flashattention flashattention flashinfer flashin

29f6d408c0   34 个后端文件：aiter base_attn cutedsl_mla deepseek_v4 deepseek_v4_trtllm dots_hybrid dsa_topk paged_mqa_logits dsa flashatte
```

## Cut three: three batches (#1538, #1541, #1543, #1544) {#第三刀三份批次1538154115431544}

Four consecutive commits of 29 and 30 September completed the batch structure's layering. First #1538 moved the scheduling code into `scheduler.py`; #1541 had `ModelRunner` receive an `InputMetadata` rather than a `ScheduleBatch` — the execution layer no longer sees the scheduling layer's objects; #1543 renamed `InputMetadata` to `ForwardBatch`; and #1544 introduced the intermediate `ModelWorkerBatch`. The head of v0.4.0's `ForwardBatch`:

```python title="python/sglang/srt/model_executor/forward_batch_info.py @ v0.4.0 L50-64,85-118"
class ForwardMode(IntEnum):
    # Prefill a new sequence. This is deprecated now. "EXTEND" covers this case.
    PREFILL = auto()
    # Extend a sequence. The KV cache of the beginning part of the sequence is already computed (e.g., system prompt).
    EXTEND = auto()
    # Decode one token.
    DECODE = auto()
    # Contains both EXTEND and DECODE when doing chunked prefill.
    MIXED = auto()
    # No sequence to forward. For data parallel attention, some workers wil be IDLE if no sequence are allocated.
    IDLE = auto()

    # A dummy first batch to start the pipeline for overlap scheduler.
    # It is now used for triggering the sampling_info_done event for the first prefill batch.
    DUMMY_FIRST = auto()
...
@dataclass
class ForwardBatch:
    """Store all inputs of a forward pass."""

    # The forward mode
    forward_mode: ForwardMode
    # The batch size
    batch_size: int
    # The input ids
    input_ids: torch.Tensor
    # The indices of requests in the req_to_token_pool
    req_pool_indices: torch.Tensor
    # The sequence length
    seq_lens: torch.Tensor
    # The indices of output tokens in the token_to_kv_pool
    out_cache_loc: torch.Tensor

    # The sum of all sequence lengths
    seq_lens_sum: int

    # For logprob
    return_logprob: bool = False
    top_logprobs_nums: Optional[List[int]] = None

    # Position information
    positions: torch.Tensor = None

    # For extend
    extend_num_tokens: Optional[int] = None
    extend_seq_lens: Optional[torch.Tensor] = None
    extend_prefix_lens: Optional[torch.Tensor] = None
    extend_start_loc: Optional[torch.Tensor] = None
    extend_prefix_lens_cpu: Optional[List[int]] = None
    extend_seq_lens_cpu: Optional[List[int]] = None
```

`ForwardMode` has three values the first version lacked: `MIXED` (chunked prefill and decode in one batch), `IDLE` (under DP attention a rank with no work still has to keep step) and `DUMMY_FIRST` (for starting overlapped scheduling). Every field of `ForwardBatch` is a GPU tensor or metadata an attention backend needs. The middle layer, `ModelWorkerBatch`:

```python title="python/sglang/srt/managers/schedule_batch.py @ v0.4.0 L1136-1168" linenums="1136"
@dataclasses.dataclass
class ModelWorkerBatch:
    # The batch id
    bid: int
    # The forward mode
    forward_mode: ForwardMode
    # The input ids
    input_ids: torch.Tensor
    # The indices of requests in the req_to_token_pool
    req_pool_indices: torch.Tensor
    # The sequence length
    seq_lens: torch.Tensor
    # The indices of output tokens in the token_to_kv_pool
    out_cache_loc: torch.Tensor

    # The sum of all sequence lengths
    seq_lens_sum: int

    # The memory pool operation records
    req_to_token_pool_records: Optional[List[Tuple[Tuple, torch.Tensor]]]

    # For logprob
    return_logprob: bool
    top_logprobs_nums: Optional[List[int]]

    # For DP attention
    global_num_tokens: Optional[List[int]]
    can_run_dp_cuda_graph: bool

    # For extend
    extend_num_tokens: Optional[int]
    extend_seq_lens: Optional[List[int]]
    extend_prefix_lens: Optional[List[int]]
```

It is the product of `ScheduleBatch.get_model_worker_batch()`, carrying only the fields the worker needs, and it can go into a queue for another thread — exactly what overlapped scheduling ([chapter 12](overlap.md)) needed three weeks later: the scheduling thread keeps modifying the `ScheduleBatch` while the forward thread takes only the `ModelWorkerBatch`.

![Figure: the three batches and where the attention backend sits](../assets/figures/sgl-batch-trio.svg){.aig-svg}

## Cut four: the allocation state onto the CPU (#1557) {#第四刀分配状态上-cpu1557}

The first version's `TokenToKVPool.alloc` scanned for free slots on the GPU with `torch.nonzero` ([chapter two](../origins/first-commit.md)). #1557 "Move status check in the memory pool to CPU" of 2 October changed the free slots into an index tensor on the CPU: allocation is a slice and freeing is a concatenation, with no GPU kernel or synchronisation. This is another precondition for overlapped scheduling — allocation in the scheduling thread must not wait for the GPU. The same week #1549 simplified `io_struct` and the `TokenizerManager`, #1562 tidied `SamplingBatchInfo`, and #1555 and #1556 took the regex FSM and the `ScheduleBatch` out of the sampling information. That group of "simplification" commits deletes more lines than it adds.

## Design trade-offs {#设计取舍}

The reorganisation brought no new feature, so why spend two months on it? Seen from the later history, it bought three things:

1. **A new backend adds a file without touching the scheduling.** FlashAttention 3 (2025-03), FlashMLA, TRT-LLM MLA, AITER, Ascend, CPU… each is one file under `layers/attention/` plus a registration, with `scheduler.py` untouched.
2. **Overlapped scheduling changed only two classes.** `Scheduler.event_loop_overlap` and `TpModelWorkerClient`, because the batch was already in three layers.
3. **The memory pool's implementation can be swapped.** Once `mem_cache/` stood alone, the MLA pool, the tiered cache, page sizes above 1 and the SWA pool were all classes added in that directory.

The price is more concepts: anyone reading the source has to work out the relationship between three batches, two pools and one backend first — [the inference-systems handbook's source reading](serving://source/sglang/) is written around exactly these concepts.

## What happened afterwards {#后来怎么样了}

- `layers/attention/` has 139 files and over twenty backends at the baseline commit, plus combiners like `hybrid_attn_backend`.
- `ForwardBatch` in `model_executor/forward_batch_info.py` has several times as many fields (multimodal, DP attention, speculative decoding and PD disaggregation each added their own), split with mixins from 2025.
- `managers/schedule_batch.py` is nearly 4000 lines, with `Req` and `ScheduleBatch` still at its centre.
- `mem_cache/` grew from 5 files to 156 (the script at the end of [chapter three](../origins/radix-v1.md)).

## Exercises {#练习}

**1. Compare the three batches' fields.** Count the fields of `ScheduleBatch`, `ModelWorkerBatch` and `ForwardBatch` at v0.4.0 and find the ones that appear in all three.

??? success "A way to approach it"
    `git show v0.4.0:python/sglang/srt/managers/schedule_batch.py | sed -n '471,560p'` shows `ScheduleBatch`'s dataclass fields, `ModelWorkerBatch` starts at line 1137, and `ForwardBatch` at line 86 of `forward_batch_info.py`. `input_ids`, `req_pool_indices`, `seq_lens`, `out_cache_loc` and `forward_mode` are in all three: they are the "primary keys" carried all the way from scheduling to the forward pass.

**2. How the backend interface changed.** Compare the method lists in `layers/attention/base_attn_backend.py` (or `__init__.py`) at v0.4.0 and at the baseline commit, and say which feature each new method serves.

??? success "A way to approach it"
    `git show 29f6d408c0:python/sglang/srt/layers/attention/base_attn_backend.py | grep 'def '`. The new ones usually concern speculative decoding (the draft and verify metadata), paged KV (the page size), PD disaggregation and DP attention.

**3. The week that deleted more than it added.** Use `git log --shortstat --since=2024-09-28 --until=2024-10-05 -- python/sglang/srt` to total that week's insertions and deletions.

??? success "A way to approach it"
    Add up each commit's insertions and deletions; in a refactoring week the deletions approach or exceed the additions, which says this is paying down technical debt rather than adding features.

!!! interview "How to explain it"
    "How should an inference engine's modules be divided?" — Use this reorganisation: directories by lifetime (the request, the batch, the service-level cache, one forward pass, the stateless layers), the batch split by owner into a scheduling layer, a worker layer and a forward layer, and the attention backend leaving "how the metadata is prepared" to the implementation behind an interface. Then say what those boundaries bought later (a new backend only adds a file, overlapped scheduling only changed two classes).

## Summary {#小结}

- [x] 2024-07-29 → 10-03: `mem_cache/`, `model_executor/`, `sampling/` and `layers/attention/` appeared, `InputMetadata` became `ForwardBatch`, and the batch split into three layers.
- [x] Half of the `AttentionBackend` interface's methods serve CUDA graphs, leaving "how the metadata is prepared" to the backend.
- [x] Moving the allocation state to the CPU and giving sampling its own directory cleared the way for overlapped scheduling three weeks later.
