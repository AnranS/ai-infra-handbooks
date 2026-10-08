# The first code release: what the runtime looked like in ten thousand lines

<p class="lead">The commit <code>22085081bb</code> of 8 January 2024 is SGLang's genesis: 51 Python files and a little over ten thousand lines, of which the SRT runtime is only 31 files and 6400 lines. Yet SGLang's skeleton today — three kinds of process, ZMQ messages, the extend forward mode, a two-level KV pool paged by token, the scheduler's main loop — is all in those ten thousand lines. This chapter walks the first runtime through from the processes to the operators, with every piece of code cut verbatim by commit id, and by the end you will know which skeleton the next three years' hundreds of thousands of lines grew on.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. What kinds of process does the first version have? What do they communicate with? Which process runs the model?
    2. What are `ForwardMode`'s three values? Why is `EXTEND` needed rather than just prefill and decode?
    3. Why is the first version's KV pool split into the two levels `ReqToTokenPool` and `TokenToKVPool`? What is the page size?
    4. What does one iteration of the first version's scheduling loop do? How does it decide whether a new request can enter the batch?

??? success "Answers for the self-test (answer first, then open this)"
    1. The main process runs FastAPI and the `TokenizerManager` (tokenizing, collecting results); one router process runs `RouterManager`'s event loop and drives, through rpyc, one `ModelRpcServer` model process per TP rank (where both the scheduling and the execution happen); and one detokenizer process runs `DetokenizerManager`. Tokenizer → router → detokenizer → tokenizer communicate over ZMQ PUSH/PULL (TCP), and the router talks to the model processes over rpyc.
    2. `PREFILL`, `EXTEND`, `DECODE`. `EXTEND` is the forward pass for "the prefix is already in the cache, compute only what is new": with a prefix cache, a request's prefill usually only has to compute the suffix that missed, and attention has to read both the prefix's KV from the cache and the KV just computed, which needs its own Triton kernel (`extend_attention_fwd`).
    3. `ReqToTokenPool` is a `[request slot, context length]` table recording which slot holds the KV of a request's i-th token; `TokenToKVPool` stores every layer's K and V by slot and manages the slots with reference counts. The page size is 1 token, so that the radix tree can split a node at any token and any prefix can be reused.
    4. Each iteration first tries to form a new fill (extend) batch: it matches the prefix of every request in the waiting queue, sorts them by the policy, and admits them one by one by whether "the new tokens plus the tokens expected still to be generated (times an estimated ratio)" will fit; if no batch can be formed, it runs 10 decode steps in a row on the running batch. On admission it increments the reference count of the tree nodes that hit, so they cannot be evicted.

A six-panel strip first, then the chapter:

<!-- comic ../assets/comics/first-commit.webp is in Chinese; put it back once the English version exists -->

## How the 51 files are spread {#51-个文件的分布}

The runtime's files by line count:

```python title="srt-lines.py"
import subprocess
REF0 = "22085081bb"
files = subprocess.run(["git", "ls-tree", "-r", "--name-only", REF0, "--", "python/sglang/srt"], capture_output=True, text=True).stdout.split()
rows = []
for f in files:
    if f.endswith(".py"):
        rows.append((subprocess.run(["git", "show", f"{REF0}:{f}"], capture_output=True, text=True).stdout.count("\n"), f.removeprefix("python/sglang/srt/")))
for n, f in sorted(rows, reverse=True):
    print(f"{n:5d}  {f}")
print(f"{sum(n for n, _ in rows):5d}  合计（{len(rows)} 个文件）")
```

```text title="output"
  586  constrained/regex.py
  497  managers/router/model_rpc.py
  458  managers/router/model_runner.py
  385  constrained/fsm.py
  378  models/mixtral.py
  371  layers/extend_attention.py
  326  managers/router/infer_batch.py
  324  layers/token_attention.py
  316  models/llama2.py
  266  constrained/tokenizer.py
  222  server.py
  220  managers/router/radix_cache.py
  219  managers/tokenizer_manager.py
  217  utils.py
  213  models/llava.py
  181  layers/context_flashattention_nopad.py
  164  hf_transformers_utils.py
  158  layers/radix_attention.py
  138  server_args.py
  111  memory_pool.py
   88  managers/io_struct.py
   85  managers/detokenizer_manager.py
   81  sampling_params.py
   79  layers/get_selected_logprob.py
   77  layers/logits_processor.py
   73  managers/router/scheduler.py
   71  managers/router/manager.py
   41  constrained/fsm_cache.py
   27  model_config.py
   12  managers/openai_protocol.py
 6384  合计（30 个文件）
```

By responsibility they fall into five blocks: the processes and their communication (`server.py`, `managers/`), the scheduling and batching (`managers/router/model_rpc.py`, `infer_batch.py`, `scheduler.py`, `radix_cache.py`), the execution (`managers/router/model_runner.py`, `memory_pool.py`, `layers/`), the models (`models/`'s three files: Llama 2, Mixtral, LLaVA) and constrained decoding (`constrained/`). The three largest files are the regex FSM adapted from Outlines (586 lines), the scheduling main loop `model_rpc.py` (497) and `model_runner.py` (458).

![Figure: the first code release's processes and data flow](../assets/figures/sgl-v0-processes.svg){.aig-svg}

## Processes: three roles and two kinds of communication {#进程三类角色与两种通信}

`server.py`'s `launch_server` brings the processes up in one go: allocate ports, build the `TokenizerManager` in the main process, fork the router and detokenizer processes, and wait through a `Pipe` for each to report "init ok":

```python title="python/sglang/srt/server.py @ 22085081bb L82-116" linenums="82"
    # Allocate ports
    can_use_ports = alloc_usable_network_port(
        num=4 + server_args.tp_size, used_list=(server_args.port,)
    )
    port_args = PortArgs(
        tokenizer_port=can_use_ports[0],
        router_port=can_use_ports[1],
        detokenizer_port=can_use_ports[2],
        nccl_port=can_use_ports[3],
        model_rpc_ports=can_use_ports[4:],
    )

    # Launch processes
    tokenizer_manager = TokenizerManager(server_args, port_args)
    pipe_router_reader, pipe_router_writer = mp.Pipe(duplex=False)
    pipe_detoken_reader, pipe_detoken_writer = mp.Pipe(duplex=False)

    proc_router = mp.Process(
        target=start_router_process,
        args=(
            server_args,
            port_args,
            pipe_router_writer,
        ),
    )
    proc_router.start()
    proc_detoken = mp.Process(
        target=start_detokenizer_process,
        args=(
            server_args,
            port_args,
            pipe_detoken_writer,
        ),
    )
    proc_detoken.start()
```

The main process runs uvicorn (HTTP) at the same time. Once the `TokenizerManager` has a `/generate` request it tokenizes it, PUSHes a `TokenizedGenerateReqInput` over ZMQ to the router process, and waits for the result in `rid_to_state` (one `asyncio.Event` per request). The router process's `RouterManager` is only 71 lines and two coroutines: one receives requests into a list, the other loops handing what has accumulated to the model process for one step:

```python title="python/sglang/srt/managers/router/manager.py @ 22085081bb L31-46" linenums="31"
    async def loop_for_forward(self):
        while True:
            next_step_input = list(self.recv_reqs)
            self.recv_reqs = []
            out_pyobjs = await self.model_client.step(next_step_input)

            for obj in out_pyobjs:
                self.send_to_detokenizer.send_pyobj(obj)

            # await for a while to accept input requests
            await asyncio.sleep(0.001)

    async def loop_for_recv_requests(self):
        while True:
            recv_req = await self.recv_from_tokenizer.recv_pyobj()
            self.recv_reqs.append(recv_req)
```

The line `await asyncio.sleep(0.001)` is worth noting: after each step it yields for a millisecond to pick up new requests, which is the plainest possible interleaving of scheduling and receiving. Behind `model_client.step` is an rpyc remote call: `ModelRpcClient` starts one process per TP rank running a `ModelRpcServer`, and `exposed_step` performs the scheduling and the forward pass inside the model process. So the first version's "scheduler" actually lives in the model process and the router process is only a porter — a structure restructured repeatedly later ([chapter six](../service/processes.md)) — but "scheduling and model execution in the same process" survives into today's `Scheduler`.

The detokenizer process turns token ids back into strings and PUSHes them to the main process, so the three processes form a ring: tokenizer → router → detokenizer → tokenizer. Why give detokenizing its own process? Because streaming has to decode every one or two tokens generated, which would slow the GPU down inside the scheduling process and fight the HTTP event loop for the GIL inside the main one. This three-process structure has lasted from the first version to today.

## One step: fill or decode {#一步fill-或者-decode}

Having received the requests, `ModelRpcServer.exposed_step` calls `forward_step`, which is the first version's scheduling main loop:

```python title="python/sglang/srt/managers/router/model_rpc.py @ 22085081bb L139-174" linenums="139"
    def forward_step(self):
        new_batch = self.get_new_fill_batch()

        if new_batch is not None:
            # Run new fill batch
            self.forward_fill_batch(new_batch)

            if not new_batch.is_empty():
                if self.running_batch is None:
                    self.running_batch = new_batch
                else:
                    self.running_batch.merge(new_batch)
        else:
            # Run decode batch
            if self.running_batch is not None:
                # Run a few decode batches continuously for reducing overhead
                for _ in range(10):
                    self.forward_decode_batch(self.running_batch)

                    if self.running_batch.is_empty():
                        self.running_batch = None
                        break

        if self.running_batch is not None and self.tp_rank == 0:
            if self.decode_forward_ct >= 20:
                self.decode_forward_ct = 0
                num_used = self.max_total_num_token - (
                    self.token_to_kv_pool.available_size()
                    + self.tree_cache.evictable_size()
                )
                logger.info(
                    f"#running-req: {len(self.running_batch.reqs)}, "
                    f"#token: {num_used}, "
                    f"token usage: {num_used / self.max_total_num_token:.2f}, "
                    f"#queue-req: {len(self.forward_queue)}"
                )
```

The logic is direct: if a new fill batch can be formed, run it (an extend forward pass) and merge it into `running_batch`; otherwise run **10 decode steps in a row** on `running_batch`. "Ten steps in a row" is a trade-off peculiar to this period: returning to Python at every step to check for new requests costs something, so it simply does several steps at once, at the price of a new request waiting up to 10 decode steps to be handled. Once there was finer scheduling and overlapped execution, the number disappeared.

![Figure: one step of the first version's scheduling](../assets/figures/sgl-v0-step.svg){.aig-svg}

`get_new_fill_batch` decides who gets in. It matches the prefix of every request in the waiting queue, sorts them by the policy, and then estimates the space available:

```python title="python/sglang/srt/managers/router/model_rpc.py @ 22085081bb L236-246,264-285"
        available_size = (
            self.token_to_kv_pool.available_size() + self.tree_cache.evictable_size()
        )
        new_ratio = self.scheduler.new_token_estimation_ratio()
        if self.running_batch:
            available_size -= sum(
                [
                    (r.max_new_tokens() - len(r.output_ids)) * new_ratio
                    for r in self.running_batch.reqs
                ]
            )
...
            if (
                req.adjust_input_len + req.max_new_tokens() + new_batch_total_tokens
                < available_size
                and req.adjust_input_len + new_batch_input_tokens
                < self.max_prefill_num_token
            ):
                delta = self.tree_cache.inc_ref_counter(req.last_node)
                available_size += delta

                if not (
                    req.adjust_input_len + req.max_new_tokens() + new_batch_total_tokens
                    < available_size
                ):
                    delta = self.tree_cache.dec_ref_counter(req.last_node)
                    available_size += delta
                else:
                    self.token_to_kv_pool.add_refs(req.prefix_indices)
                    can_run_list.append(req)
                    new_batch_total_tokens += (
                        req.adjust_input_len + req.max_new_tokens()
                    )
                    new_batch_input_tokens += req.adjust_input_len
```

Two details settle the character of SGLang's scheduler:

- **The space available = the free slots + the nodes on the tree that can be evicted**, minus the tokens the running requests are "expected still to generate" (the remaining `max_new_tokens` times `new_token_estimation_ratio`, 0.4 by default). That is, it **estimates the future** and would rather admit fewer requests than run out of room halfway through decoding — the ancestor of today's `new_token_ratio`, and the opposite of vLLM's "admit first, preempt if it runs short".
- On admitting a request it first calls `inc_ref_counter(req.last_node)` to lock the tree nodes it hit (making them unevictable), and if the space turns out to be insufficient after the lock (because the locked nodes had been counted as evictable), it immediately unlocks and gives up.

The requests that get into the batch go to `forward_fill_batch`, and a finished request has its KV inserted back into the radix tree in `handle_finished_requests` ([the next chapter](radix-v1.md)).

## Batching: Req, Batch and the three forward modes {#批次reqbatch-与三种前向模式}

`infer_batch.py` defines the request and the batch. `ForwardMode` has only three values:

```python title="python/sglang/srt/managers/router/infer_batch.py @ 22085081bb L10-13" linenums="10"
class ForwardMode(Enum):
    PREFILL = auto()
    EXTEND = auto()
    DECODE = auto()
```

`EXTEND` is a notion the prefix cache brings: the request's first `len(prefix_indices)` tokens are already in the cache and only the rest is computed this time. `Batch.init_extend_batch` writes each request's prefix slots into the `req_to_token` table and then allocates slots for the new tokens:

```python title="python/sglang/srt/managers/router/infer_batch.py @ 22085081bb L104-131" linenums="104"
    def init_extend_batch(self, vocab_size: int, int_token_logit_bias: torch.Tensor):
        device = "cuda"
        bs = len(self.reqs)
        reqs = self.reqs
        input_ids = [r.input_ids[len(r.prefix_indices) :] for r in reqs]
        prefix_indices = [r.prefix_indices for r in reqs]

        # Handle prefix
        flatten_input_ids = []
        extend_lens = []
        prefix_lens = []
        seq_lens = []

        req_pool_indices = self.req_to_token_pool.alloc(bs)
        req_pool_indices_cpu = req_pool_indices.cpu().numpy()
        for i in range(bs):
            flatten_input_ids.extend(input_ids[i])
            extend_lens.append(len(input_ids[i]))

            if len(prefix_indices[i]) == 0:
                prefix_lens.append(0)
            else:
                prefix_lens.append(len(prefix_indices[i]))
                self.req_to_token_pool.req_to_token[req_pool_indices_cpu[i]][
                    : len(prefix_indices[i])
                ] = prefix_indices[i]

            seq_lens.append(prefix_lens[-1] + extend_lens[-1])
```

Note `input_ids = r.input_ids[len(r.prefix_indices):]` — only the suffix that missed goes into the model, `seq_lens` is the full length, and `prefix_lens` tells the attention kernel how much KV to read from the cache beforehand. During decode, `update_for_decode` allocates one slot per request and **tries for a contiguous range first** (`alloc_contiguous`): with a contiguous range it writes the KV as a slice, otherwise it falls back to writing by index — a small optimisation to make KV writes faster.

Sampling is in `Batch.sample` too: the temperature, the logit bias, the regex FSM's mask, top-p and top-k, and finally `torch.multinomial`. Every request's sampling parameters are packed into tensors in advance (`temperatures`, `top_ps`…), sliced or concatenated in step in `filter_batch` and `merge`. Today's `ScheduleBatch` plus `SamplingBatchInfo` were split out of this.

## Memory: a two-level pool paged by token {#内存按-token-分页的两级池}

`memory_pool.py` is only 111 lines, yet it is the source of today's hundred-odd files under `mem_cache/`:

```python title="python/sglang/srt/memory_pool.py @ 22085081bb L41-64" linenums="41"
class TokenToKVPool:
    def __init__(self, size, dtype, head_num, head_dim, layer_num):
        self.mem_state = torch.zeros((size,), dtype=torch.int16, device="cuda")
        self.alloc_ct = 0

        # [size, key/value, head_num, head_dim] for each layer
        self.kv_data = [
            torch.empty((size, 2, head_num, head_dim), dtype=dtype, device="cuda")
            for _ in range(layer_num)
        ]

    def get_key_buffer(self, layer_id):
        return self.kv_data[layer_id][:, 0]

    def get_value_buffer(self, layer_id):
        return self.kv_data[layer_id][:, 1]

    def alloc(self, need_size):
        select_index = torch.nonzero(self.mem_state == 0).squeeze(1)[:need_size]
        if select_index.shape[0] < need_size:
            return None

        self.add_refs(select_index)
        return select_index.to(torch.int32)
```

`ReqToTokenPool` is the request-to-slot mapping table (a `[size, max_context_len]` int32 tensor), `TokenToKVPool` stores every layer's K and V by slot (one `[size, 2, head_num, head_dim]` per layer), and `mem_state` is each slot's reference count. Several decisions:

- **The page size is 1.** One slot per token, so the radix tree can be cut anywhere; the price is large metadata and an attention kernel that has to address indirectly per token. Not until `Support page size > 1 (#4356)` of March 2025 were larger pages supported.
- **Allocation uses `torch.nonzero`.** `alloc` scans the whole `mem_state` on the GPU for free slots, O(N) but five lines to implement; it was later replaced by a free list on the CPU side (2024-10-02, `Move status check in the memory pool to CPU (#1557)`).
- **The reference counts are in the pool, not in the tree.** A tree node's `ref_counter` governs whether the node can be evicted and the pool's `mem_state` governs whether a slot can be reused, each at its own level — a division of labour that survives into today's `TokenToKVPoolAllocator`.

The pool's size is computed by `ModelRunner.profile_max_num_token`: take the memory available, subtract what lies outside `mem_fraction_static`, and divide by the KV bytes per token. The meaning of the `--mem-fraction-static` option has not changed from the first version to today.

## Execution: RadixAttention and two sets of kernels {#执行radixattention-与两套-kernel}

The attention layer in the model code is called `RadixAttention`, and it does no tree operations at all: it only picks a kernel according to the forward mode and writes this layer's K and V into the pool:

```python title="python/sglang/srt/layers/radix_attention.py @ 22085081bb L32-43,144-158"
        from sglang.srt.managers.router.model_runner import global_model_mode

        self.use_flashinfer = "flashinfer" in global_model_mode

        if self.use_flashinfer:
            self.prefill_forward = self.prefill_forward_flashinfer
            self.extend_forward = self.prefill_forward_flashinfer
            self.decode_forward = self.decode_forward_flashinfer
        else:
            self.prefill_forward = self.prefill_forward_triton
            self.extend_forward = self.extend_forward_triton
            self.decode_forward = self.decode_forward_triton
...
    def store_kv_cache(self, cache_k, cache_v, input_metadata: InputMetadata):
        key_buffer = input_metadata.token_to_kv_pool.get_key_buffer(self.layer_id)
        value_buffer = input_metadata.token_to_kv_pool.get_value_buffer(self.layer_id)
        if input_metadata.out_cache_loc is not None:
            key_buffer[input_metadata.out_cache_loc] = cache_k
            value_buffer[input_metadata.out_cache_loc] = cache_v
        elif input_metadata.out_cache_cont_start is not None:
            key_buffer[
                input_metadata.out_cache_cont_start : input_metadata.out_cache_cont_end
            ] = cache_k
            value_buffer[
                input_metadata.out_cache_cont_start : input_metadata.out_cache_cont_end
            ] = cache_v
        else:
            raise RuntimeError()
```

The three modes correspond to three Triton kernels: `context_attention_fwd` (a prefill with no prefix), `extend_attention_fwd` (an extend with a prefix, where q is only the new part and k and v are read from the pool through `req_to_token`) and `token_attention_fwd` (decode). The other path is FlashInfer (`--model-mode flashinfer`), using its paged prefill and decode wrappers, with the page size still 1. Two sets of kernels coexisting started on day one: the Triton version runs on any GPU and is easy to modify, the FlashInfer version is faster. Today there are a dozen or so backends under `layers/attention/`, but the division of labour — the attention layer only picks a backend and does not touch the cache — was settled here.

`ModelRunner.load_model` picks the model class from a hand-written table:

```python title="python/sglang/srt/managers/router/model_runner.py @ 22085081bb L240-258" linenums="240"
        # Select model class
        architectures = getattr(self.model_config.hf_config, "architectures", [])

        model_class = None
        for arch in architectures:
            if arch == "LlamaForCausalLM":
                model_class = LlamaForCausalLM
                break
            if arch == "MistralForCausalLM":
                model_class = LlamaForCausalLM
                break
            if arch == "LlavaLlamaForCausalLM":
                model_class = LlavaLlamaForCausalLM
                break
            if arch == "MixtralForCausalLM":
                model_class = MixtralForCausalLM
                break
        if model_class is None:
            raise ValueError(f"Unsupported architectures: {architectures}")
```

while the model definitions themselves borrow heavily from vLLM's layer implementations:

```python title="python/sglang/srt/models/llama2.py @ 22085081bb L12-31" linenums="12"
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    LinearMethodBase,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.parallel_utils.parallel_state import (
    get_tensor_model_parallel_world_size,
)
from vllm.model_executor.weight_utils import (
    default_weight_loader,
    hf_model_weights_iterator,
)
```

The parallel linear layers, RMSNorm, RoPE, the vocabulary-parallel embedding, the weight loaders and the tensor-parallel process group's initialisation, all `from vllm... import`. This is the first version's largest borrowing: write only what is genuinely new (the scheduling, the cache, the attention kernels) and take the rest off the shelf. Its price did not show until a year later ([chapter eight](../service/borrow-vllm.md)).

## Multimodal: a hash turns an image into a cacheable prefix {#多模态用哈希把图片变成可缓存的前缀}

The first version already supports LLaVA, and it fits the prefix cache rather neatly. When a request carries an image, the `TokenizerManager` computes the `pixel_values` and a hash of the image, and the scheduler replaces the `<image>` placeholder token in the prompt with a run of pad tokens derived from that hash:

```python title="python/sglang/srt/managers/router/model_rpc.py @ 22085081bb L180-192" linenums="180"
        req = Req(recv_req.rid)
        req.input_ids = recv_req.input_ids
        req.pixel_values = recv_req.pixel_values
        if req.pixel_values is not None:
            pad_value = [
                (recv_req.image_hash) % self.model_config.vocab_size,
                (recv_req.image_hash >> 16) % self.model_config.vocab_size,
                (recv_req.image_hash >> 32) % self.model_config.vocab_size,
                (recv_req.image_hash >> 64) % self.model_config.vocab_size,
            ]
            req.input_ids, req.image_offset = self.model_runner.model.pad_input_ids(
                req.input_ids, pad_value
            )
```

```python title="python/sglang/srt/models/llava.py @ 22085081bb L35-46" linenums="35"
    def pad_input_ids(self, input_ids, pad_value):
        pad_ids = pad_value * (
            (self.image_feature_len + len(pad_value)) // len(pad_value)
        )
        offset = input_ids.index(self.config.image_token_index)
        # old_len + pad_len - 1, because we need to remove image_token_id
        new_input_ids = (
            input_ids[:offset]
            + pad_ids[: self.image_feature_len]
            + input_ids[offset + 1 :]
        )
        return new_input_ids, offset
```

The same image is then always the same run of ids in the token sequence, so the radix tree can cache that image's KV exactly as it caches a text prefix (which is where the paper's LLaVA hit rate comes from). The actual visual features are filled into the right positions during the extend forward pass at `image_offset`. This trick of mapping an unhashable input onto a fixed token sequence has been used ever since, and today's `multimodal/` still uses a hash as the cache key.

## Design trade-offs {#设计取舍}

Put the first version back in the context of January 2024 and its trade-offs are clear:

| Trade-off | The first version's choice | Why |
| --- | --- | --- |
| Build or borrow | the scheduling, the cache and the attention kernels built; the model layers, weight loading and parallel communication borrowed from vLLM | spend the effort on the paper's novelty |
| Page size | 1 token | the tree can be cut anywhere, for the highest hit rate |
| Scheduling | estimate the future demand, admit conservatively, no preemption | simple to implement, and request lengths are similar in the paper's setting |
| Processes | scheduling and execution in one process, one helper process at each end | avoid an extra hop between the scheduler and the GPU |
| Communication | ZMQ plus rpyc | off the shelf and good enough; rpyc was removed half a year later |
| Streaming | push every 2 decode steps (`stream_interval`) | reduce the detokenizing and HTTP overhead |

Of these choices, "scheduling and execution in one process", "a two-level KV pool" and "three kinds of process" are alive today, while "10 decode steps in a row", rpyc, allocation by `torch.nonzero` and the hand-written model table were all replaced within a year.

## What happened afterwards {#后来怎么样了}

Map each of the first version's files onto its place at the baseline commit `29f6d408c0` and compare the sizes:

```bash title="then-and-now.sh"
REF=${REF:-29f6d408c0}
printf '%-46s %6s\n' '今天的文件' '行数'
for f in python/sglang/srt/entrypoints/http_server.py python/sglang/srt/managers/tokenizer_manager.py python/sglang/srt/managers/scheduler.py \
         python/sglang/srt/managers/schedule_batch.py python/sglang/srt/managers/schedule_policy.py python/sglang/srt/managers/tp_worker.py \
         python/sglang/srt/model_executor/model_runner.py python/sglang/srt/mem_cache/radix_cache.py python/sglang/srt/mem_cache/memory_pool.py \
         python/sglang/srt/layers/radix_attention.py python/sglang/srt/managers/detokenizer_manager.py; do
  printf '%-46s %6d\n' "${f#python/sglang/srt/}" "$(git show "$REF:$f" | wc -l)"
done
```

```text title="output"
今天的文件                                行数
entrypoints/http_server.py                       2942
managers/tokenizer_manager.py                    3944
managers/scheduler.py                            5969
managers/schedule_batch.py                       3993
managers/schedule_policy.py                      1645
managers/tp_worker.py                             746
model_executor/model_runner.py                   2438
mem_cache/radix_cache.py                          823
mem_cache/memory_pool.py                         5966
layers/radix_attention.py                         676
managers/detokenizer_manager.py                   561
```

| The first version (2024-01) | Today (2026-10) | The key commits in between |
| --- | --- | --- |
| `server.py` (FastAPI plus startup) | `entrypoints/http_server.py`, `entrypoints/engine.py` | `entrypoints/` established 2025-01-19, the OpenAI service rewritten 2025-06 |
| `managers/router/manager.py` plus `model_rpc.py` | `managers/scheduler.py` (nearly 6000 lines) plus `tp_worker.py` | #480 static DP (2024-05), #646 removing rpyc (2024-07), #807 the directory restructuring (2024-07), #1538 splitting out scheduler.py (2024-09) |
| `managers/router/infer_batch.py` | `managers/schedule_batch.py` plus `model_executor/forward_batch_info.py` | #807, #1543 `InputMetadata` → `ForwardBatch` (2024-09-30) |
| `managers/router/scheduler.py` (the policy) | `managers/schedule_policy.py` | renamed in #1543, restructured in #2571 (2025-01) |
| `memory_pool.py` | `mem_cache/memory_pool.py` plus `mem_cache/allocator/` | `mem_cache/` created 2024-08-01, #4356 page size > 1 (2025-03) |
| `layers/radix_attention.py` plus three Triton files | `layers/radix_attention.py` plus a dozen backends under `layers/attention/` | #1381 and #1547, the attention-backend abstraction (2024-09) |
| 3 files in `models/` | over 280 in `models/` | continuous |

## Exercises {#练习}

**1. Who is in which process.** Read `server.py`, `manager.py` and `model_rpc.py`'s `ModelRpcClient` at `22085081bb` and draw how many processes exist at `--tp-size 2` and what each one runs.

??? success "Answer"
    The main process (uvicorn plus the TokenizerManager), the router process (RouterManager plus ModelRpcClient), two model processes (one ModelRpcServer each, an rpyc ThreadedServer, each calling `init_process_group` to join the NCCL group in `exposed_init_model`) and the detokenizer process: 5 in all. For tp_size > 1 the `ModelRpcClient` uses a thread pool to call every rank's `exposed_step` concurrently and takes only rank 0's return value.

**2. The disappearance of the 10 decode steps.** Use `git log -S'range(10)'` or `-S'for _ in range(10)'` to find the commit that removed "10 decode steps in a row" and see what replaced it.

??? success "A way to approach it"
    `git log --date=short --format='%ad %h %s' -S'for _ in range(10)' -- python/sglang/srt/managers/` lists the commits that introduced and removed it; reading the removing commit's diff shows the main loop changed to one forward pass per step, handling new requests through `--stream-interval` and finer scheduling, which paved the way for overlapped scheduling later.

**3. How slot allocation evolved.** Find the commit that changed `TokenToKVPool.alloc` from `torch.nonzero` to a free list on the CPU side, and compare the two implementations' complexity and per-allocation cost.

??? success "A way to approach it"
    `git log --date=short --format='%ad %h %s' -S'torch.nonzero' -- python/sglang/srt/memory_pool.py python/sglang/srt/mem_cache/memory_pool.py`. #1557 of 2024-10-02 moved the free state to the CPU: a `nonzero` on the GPU is a kernel launch plus a synchronisation every time, while maintaining free indices on the CPU makes allocation an O(1) slice that no longer interrupts the GPU's stream.

!!! interview "How to explain it"
    To explain SGLang's architecture, the first version's skeleton is the surest route: three kinds of process (tokenizing, scheduling plus execution, detokenizing) joined in a ring by ZMQ; a scheduler that chooses at each step between forming a new extend batch and decoding the running batch, estimating the future demand on admission; KV managed at two levels, a request table plus a slot pool, with a page size of 1 to suit the radix tree; and an attention layer that only picks a kernel and writes the cache. Then add a sentence on what has changed since (overlapped scheduling, a configurable page size, dozens of attention backends) to show you know how it grew.

## Summary {#小结}

- [x] The first version is 51 files and ten thousand lines, with 6400 in the runtime; three kinds of process plus ZMQ plus rpyc, with scheduling and execution in one process.
- [x] One iteration of the main loop either forms an extend batch or runs 10 decode steps in a row; admission rests on the estimate "free plus evictable minus what is expected still to be generated".
- [x] The `EXTEND` forward mode, the two-level KV pool paged by token, and a `RadixAttention` that picks a kernel without touching the cache all serve the prefix cache.
- [x] The model layers are borrowed from vLLM wholesale; an image is mapped by hash onto a fixed run of tokens, so multimodal input can hit the prefix cache too.
