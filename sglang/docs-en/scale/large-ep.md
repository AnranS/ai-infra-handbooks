# Large-scale EP: DeepEP, EPLB and two-batch overlap

<p class="lead">DeepSeek's open-source week of February 2025 released DeepEP, DeepGEMM and EPLB, and SGLang connected all three within three months, reproducing DeepSeek's own deployment shape on 96 H100s on 5 May: PD disaggregation, prefill at EP32 and decode at EP72, 52.3k input and 22.3k output tokens per second per node. This chapter follows those three months' three threads — how DeepEP's two dispatch modes match prefill and decode, how two-batch overlap (TBO) hides the communication behind the computation, and how EPLB balances the load with redundant experts — and where each lands in the code: <code>layers/moe/token_dispatcher/</code>, <code>two_batch_overlap.py</code> and <code>eplb/</code>.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. Which phase does each of DeepEP's normal and low-latency dispatch suit? Why does decode want the latter?
    2. How does two-batch overlap work? Does it overlap the same thing as [chapter 12](../perf/overlap.md)'s overlapped scheduling?
    3. Why does EPLB need "redundant experts"? When is rebalancing static and when dynamic?
    4. What do the blog's 1.49 times, 2.54 times and 27 to 35% each improve?

??? success "Answers for the self-test (answer first, then open this)"
    1. Normal mode does the all-to-all over the actual token count, so the shapes are dynamic: high throughput but no CUDA graph, which suits prefill (long inputs, large batches). Low-latency mode allocates fixed-size RDMA buffers in advance, so the shapes are fixed, the latency is low and CUDA graphs work, which suits decode (a few tokens per step, relying on graph replay to keep the launch overhead down).
    2. Cut a batch into two micro-batches and run them alternately: while one is computing attention or the MLP the other is doing its dispatch or combine communication, so the communication and the computation overlap. Chapter 12 overlaps the CPU's scheduling with the GPU's computation; this overlaps communication with computation on the GPU, and both can be on at once.
    3. The experts' load is uneven: the card holding a popular expert becomes the bottleneck. EPLB rearranges the expert-to-card mapping by the load it measured and replicates the popular experts (256 experts expanded into 288 "physical experts"), each copy taking a share of the traffic. Static: computed once at startup from a historical distribution. Dynamic: re-measured and rebalanced periodically while running (in three phases: load, transfer the weights asynchronously, copy between devices).
    4. With EPLB on, prefill's throughput is 1.49 times and decode's 2.54 times; TBO raises prefill's throughput 27 to 35% and decode's (at 256 tokens per card) 25.5%.

A six-panel strip first, then the chapter:

<!-- comic ../assets/comics/large-ep.webp is in Chinese; put it back once the English version exists -->

## Three months of commits {#三个月的提交}

```bash title="large-ep-commits.sh"
for h in 21463e321a c553e1604c f44db16c8e 4c56e5dbee ca75741e86 23c764b18a febe21ce03 77e929a1a2 f0653886a5 cba1cdbc46 ccfe5c009d 7a80f56513 0d47788025 32fa1e9cc2; do
  git log -1 --date=short --format='%ad  %h  %s' $h | cut -c1-96
done
```

```text title="output"
2025-02-26  21463e321a  Expert Parallelism (EP) Support for DeepSeek V3/R1 (#3602)
2025-03-10  c553e1604c  DeepGemm integrate to sgl-kernel (#4165)
2025-03-19  f44db16c8e  [Feature] Integrate DeepEP into SGLang (#4232)
2025-03-21  4c56e5dbee  Set deepgemm to the default value in the hopper architecture. (#4613)
2025-03-23  ca75741e86  Support async in DeepEP (#4610)
2025-04-02  23c764b18a  [Feature] Support DeepEP Low Latency (#4767)
2025-04-04  febe21ce03  Small refactor DeepEPDispatcher into subclasses (#4994)
2025-04-04  77e929a1a2  Support async DeepEP by splitting into two stages (#4995)
2025-05-20  f0653886a5  Expert distribution recording without overhead for EPLB (#4957)
2025-05-20  cba1cdbc46  Support DeepSeek EPLB algorithm with static distributions (#6387)
2025-05-21  ccfe5c009d  Support redundant experts in expert parallel (#6461)
2025-05-22  7a80f56513  Support dynamically rebalancing experts using EPLB (#6469)
2025-05-25  0d47788025  Support overlapping two batches (#4068)
2025-07-31  32fa1e9cc2  [4/N] MoE Refactor: Unified Triton Kernel for FusedMoE and EPMoE (#8515)
```

#3602 of 26 February first made DeepSeek-V3 and R1 run under EP ([chapter 13](../perf/multi-gpu.md)'s plain EP plus the MoE's weight sharding); DeepGEMM entered sgl-kernel on 10 March, #4232 connected DeepEP on 19 March, #4767 supported the low-latency mode on 2 April, and a set of restructurings followed in early April (`DeepEPMode`, `DeepEPDispatcher` subclassed, two asynchronous phases); the EPLB series (the statistics, the static distribution, redundant experts, dynamic rebalancing) began on 20 May; and #4068 brought TBO on 25 May. The directories:

```bash title="moe-dirs.sh"
REF=${REF:-29f6d408c0}
for t in v0.4.0 v0.4.6 v0.5.0rc0 "$REF"; do
  printf '%-11s layers/moe %3d 个文件，eplb %2d 个，two_batch_overlap.py %4d 行\n' "$t" "$(git ls-tree -r --name-only "$t" -- python/sglang/srt/layers/moe | grep -c '\.py$')" "$(git ls-tree -r --name-only "$t" -- python/sglang/srt/eplb | grep -c '\.py$')" "$(git show "$t:python/sglang/srt/two_batch_overlap.py" 2>/dev/null | wc -l)"
done
echo "今天 batch_overlap/：$(git ls-tree -r --name-only "$REF" -- python/sglang/srt/batch_overlap | sed 's|.*/||' | tr '\n' ' ')"
```

```text title="output"
v0.4.0      layers/moe   0 个文件，eplb  0 个，two_batch_overlap.py    0 行
v0.4.6      layers/moe  10 个文件，eplb  0 个，two_batch_overlap.py    0 行
v0.5.0rc0   layers/moe  18 个文件，eplb 11 个，two_batch_overlap.py  980 行
29f6d408c0  layers/moe  64 个文件，eplb 14 个，two_batch_overlap.py    0 行
今天 batch_overlap/：operations.py operations_strategy.py single_batch_overlap.py two_batch_overlap.py 
```

## DeepEP: each mode where it belongs {#deepep两种模式各用其所}

v0.5.0rc0's dispatcher wraps both modes in one interface:

```python title="python/sglang/srt/layers/moe/token_dispatcher/deepep.py @ v0.5.0rc0 L87-151" linenums="87"
class DeepEPDispatchMode(IntEnum):
    NORMAL = auto()
    LOW_LATENCY = auto()


class DeepEPBuffer:
    _buffer = None
    _dispatch_mode: Optional[DeepEPDispatchMode] = None
    _hidden_size: Optional[int] = None
    _num_max_dispatch_tokens_per_rank: Optional[int] = None
    _num_experts: Optional[int] = None

    @classmethod
    def get_deepep_buffer(
        cls,
        group: dist.ProcessGroup,
        hidden_size: int,
        param_bytes: int,
        deepep_mode: DeepEPMode,
        num_max_dispatch_tokens_per_rank: int = None,
        num_experts: int = None,
    ):
        if cls._buffer is not None:
            return cls._buffer

        cls._hidden_size = hidden_size
        cls._num_max_dispatch_tokens_per_rank = num_max_dispatch_tokens_per_rank
        cls._num_experts = num_experts

        num_nvl_bytes, num_rdma_bytes = 0, 0
        if deepep_mode.enable_normal():
            hidden_bytes = hidden_size * param_bytes
            for config in (
                DeepEPConfig.get_instance().normal_dispatch_config
                or Buffer.get_dispatch_config(group.size()),
                DeepEPConfig.get_instance().normal_combine_config
                or Buffer.get_combine_config(group.size()),
            ):
                num_nvl_bytes = max(
                    config.get_nvl_buffer_size_hint(hidden_bytes, group.size()),
                    num_nvl_bytes,
                )
                num_rdma_bytes = max(
                    config.get_rdma_buffer_size_hint(hidden_bytes, group.size()),
                    num_rdma_bytes,
                )
        if deepep_mode.enable_low_latency():
            assert num_max_dispatch_tokens_per_rank is not None
            assert num_experts is not None and num_experts % group.size() == 0
            num_rdma_bytes = max(
                Buffer.get_low_latency_rdma_size_hint(
                    num_max_dispatch_tokens_per_rank,
                    hidden_size,
                    group.size(),
                    num_experts,
                ),
                num_rdma_bytes,
            )

        if deepep_mode == DeepEPMode.NORMAL:
            num_qps_per_rank = DeepEPConfig.get_instance().num_sms // 2
        elif deepep_mode in [DeepEPMode.LOW_LATENCY, DeepEPMode.AUTO]:
            num_qps_per_rank = num_experts // group.size()
        else:
            raise NotImplementedError
```

`DeepEPBuffer` allocates the communication buffers by mode: normal mode needs both NVLink and RDMA buffers (two tiers, within and between nodes), while low-latency mode uses `get_low_latency_rdma_size_hint` to preallocate fixed-size RDMA buffers from the maximum token count, the hidden size and the expert count. `DeepEPMode.AUTO` takes normal during prefill and low-latency during decode — which is the implementation of the blog's "PD disaggregation lets both modes exist": in one server the two modes would share memory and buffers, and deployed separately each keeps only one. DeepGEMM's two kernels match: the contiguous layout handles the dynamic shapes after a normal dispatch (needing a Triton permute kernel to order the tokens by expert) and the masked layout handles low-latency's fixed shapes (marking the valid tokens with a mask), and the latter can go into a CUDA graph.

![Figure: large-scale EP's two configurations, for prefill and for decode](../assets/figures/sgl-large-ep.svg){.aig-svg}

## TBO: hiding communication behind computation {#tbo通信藏到计算后面}

`two_batch_overlap.py` is 980 lines at v0.5.0rc0. At its heart is cutting a batch in half:

```python title="python/sglang/srt/two_batch_overlap.py @ v0.5.0rc0 L42-75" linenums="42"
def get_token_num_per_seq(
    forward_mode: ForwardMode,
    spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]] = None,
):
    if forward_mode.is_target_verify():
        return spec_info.draft_token_num
    elif forward_mode.is_decode():
        return 1
    elif forward_mode.is_idle():
        return 0
    else:
        # For extend, we should not use `token_num_per_seq`.
        return None


# TODO: may smartly disable TBO when batch size is too small b/c it will slow down
def compute_split_seq_index(
    forward_mode: "ForwardMode",
    num_tokens: int,
    extend_lens: Optional[Sequence[int]],
    token_num_per_seq: Optional[int],
) -> Optional[int]:
    if forward_mode == ForwardMode.EXTEND:
        assert extend_lens is not None
        return _split_extend_seqs(extend_lens)
    elif forward_mode.is_target_verify() or forward_mode.is_decode():
        assert token_num_per_seq is not None
        return (num_tokens // token_num_per_seq) // 2
    elif forward_mode.is_idle():
        assert num_tokens == 0
        return 0
    else:
        raise NotImplementedError()

```

The split point balances the token counts (prefill splits by the sequences' cumulative length, decode halves the request count), `TboForwardBatchPreparer` derives two sub-batches from one `ForwardBatch` (each with its own attention metadata, positions and slots), `TboCudaGraphRunnerPlugin` handles the split under CUDA graphs (a fixed ratio at capture time), and `TboDPAttentionPreparer` makes the ranks under DP attention agree on whether TBO is on and how to split. The model side is written as "operations plus yield points": each layer is cut into attention, dispatch, MLP and combine stages, and the two micro-batches' stages interleave — A computes attention while B does its dispatch communication. The blog's numbers: 27 to 35% on prefill's throughput and 25.5% on decode's at 256 tokens per card; another benefit is halving the peak memory (only half a batch's activations at a time). Prefill has one more optimisation in the launch order: submit the GPU computation first and then make the dispatch call that blocks the CPU, so the GPU does not wait.

## EPLB: redundant experts and rebalancing {#eplb冗余专家与重平衡}

```bash title="eplb-files.sh"
REF=${REF:-29f6d408c0}
for f in $(git ls-tree -r --name-only v0.5.0rc0 -- python/sglang/srt/eplb | grep '\.py$'); do printf '%5d  %s\n' "$(git show "v0.5.0rc0:$f" | wc -l)" "${f#python/sglang/srt/}"; done
echo "今天：$(git ls-tree -r --name-only "$REF" -- python/sglang/srt/eplb | grep -c '\.py$') 个文件"
```

```text title="output"
    0  eplb/__init__.py
   63  eplb/eplb_algorithms/__init__.py
  223  eplb/eplb_algorithms/deepseek.py
  276  eplb/eplb_algorithms/deepseek_vec.py
   94  eplb/eplb_manager.py
    1  eplb/eplb_simulator/__init__.py
   51  eplb/eplb_simulator/reader.py
  927  eplb/expert_distribution.py
  463  eplb/expert_location.py
  109  eplb/expert_location_dispatch.py
  575  eplb/expert_location_updater.py
今天：14 个文件
```

EPLB has three pieces: `expert_distribution.py` records how many tokens were routed to each expert (#4957 made the recording "free" by accumulating on the kernel side), `eplb_algorithms/deepseek.py` is DeepSeek's open-sourced assignment algorithm (hierarchical: group by node first and assign within a group, so that the experts on one node come from the same group where possible), and `eplb_manager.py` triggers a rebalance while running: the measured distribution → a new expert-to-card mapping → the weights moved to the target cards asynchronously → a device-to-device copy to switch over. Redundant experts (#6461) free the physical expert count from the logical one: 256 logical experts in 288 slots, with the popular ones in several places. The blog's case study shows the balance (the mean over the maximum) correlating strongly with the throughput, which is where EPLB's 1.49 and 2.54 times come from.

## Design trade-offs {#设计取舍}

- **Connect DeepSeek's three libraries rather than build them.** DeepEP, DeepGEMM and the EPLB algorithm are all connected directly (DeepGEMM through sgl-kernel, the EPLB algorithm copied into `eplb_algorithms/`), and what SGLang does is the scheduling and the integration — unlike [chapter eight](../service/borrow-vllm.md)'s "borrow then own" curve, this time it is "integrate and build the runtime around it".
- **TBO at the ForwardBatch layer.** It does not change the model's computation graph, only how the batch is cut and how the stages interleave; the price is 980 lines of splitting logic that has to be made compatible with the attention metadata, CUDA graphs, DP attention and speculative decoding one by one.
- **PD disaggregation as the precondition.** Two DeepEP modes, two DeepGEMM layouts and different EP sizes are all configured per role; a unified deployment could do none of it.

## What happened afterwards {#后来怎么样了}

- #8515 "Unified Triton Kernel for FusedMoE and EPMoE" of 2025-07-31 unified the two MoE kernel paths; `layers/moe/` has over a hundred files at the baseline commit (CUTLASS MoE, W4A8, fused MoE for every vendor's hardware).
- `two_batch_overlap.py` moved into a `batch_overlap/` directory in H2 2025, integrated with speculative decoding and the overlap on PD's decode side.
- Elastic EP arrived in 2025-10 (#11837, roadmap #7736): an EP group can gain and lose cards while running, in an `elastic_ep/` directory.
- The limitations the blog listed (a TTFT of 2 to 5 seconds, an ITL of about 100 ms, MTP's compatibility with DP attention, Blackwell support) were worked through in the later versions.

## Exercises {#练习}

**1. The buffer's size.** Read v0.5.0rc0's `DeepEPBuffer.get_deepep_buffer` and say which parameters decide low-latency mode's RDMA buffer, and why normal mode does not need the token count in advance.

??? success "A way to approach it"
    `get_low_latency_rdma_size_hint(num_max_dispatch_tokens_per_rank, hidden, num_ranks, num_experts)`: reserved by the maximum tokens per rank; normal mode exchanges the shapes at each dispatch according to the actual token count.

**2. The split point.** How does `compute_split_seq_index` split a prefill? Will the two micro-batches have equal token counts?

??? success "A way to approach it"
    It finds the position whose cumulative token count is closest to half (`_split_array_by_balanced_sum`) without splitting a sequence, so the halves are usually not exactly equal; decode halves by request count.

**3. Verify the blog's numbers.** Find the load-testing script or configuration for DeepSeek's large-scale EP under `benchmark/` and list the parameters needed to reproduce the blog (the EP size, the DeepEP mode, TBO, EPLB, MTP).

??? success "A way to approach it"
    `git ls-tree -r --name-only 29f6d408c0 benchmark/ | grep -i 'deepseek\|disagg\|ep'`, then read the launch commands in the README.

!!! interview "How to explain it"
    "How is DeepSeek-V3 deployed at high throughput?" — Go by role: PD disaggregation, with prefill on normal dispatch plus contiguous GEMM plus a large EP, and decode on low-latency dispatch plus masked GEMM plus CUDA graphs; TBO hides the dispatch and combine behind the other micro-batch's computation; EPLB balances the load with redundant experts. Give the blog's numbers (52.3k / 22.3k per node, $0.20 per million) and where they come from (EPLB 1.49x / 2.54x, TBO 27 to 35%). Add that SGLang's approach was "connect DeepSeek's three open-source libraries and build the scheduling and integration itself".

## Summary {#小结}

- [x] 2025-02 → 05: EP support → DeepGEMM → DeepEP (normal / low-latency) → EPLB → TBO, and the 96-card blog post of 5 May.
- [x] The two dispatch modes and the two GEMM layouts are separated by the prefill and decode roles, with PD disaggregation as the precondition.
- [x] TBO cuts two micro-batches at the ForwardBatch layer and interleaves them; EPLB combines measurement, redundant experts and dynamic rebalancing.
