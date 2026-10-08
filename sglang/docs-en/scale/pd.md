# PD disaggregation: the disaggregation directory's design

<p class="lead">Prefill and decode want different things from a card: the first is large matrix multiplies, the second read bandwidth; on one card they interfere, and a decode's latency jitters as soon as a prefill arrives. Byron Hsu's #4654 "[PD] Release initial code" of 21 March 2025 split them into two kinds of server in 1410 lines: prefill computes and sends the KV to the decode server over RDMA. This chapter reads those 1410 lines' structure — an 81-line transfer interface, two scheduling-loop mixins each with its queues, and a load balancer of a few hundred lines of Python — then watches it connect to Mooncake, NIXL, overlapped scheduling, DP attention and DeepEP within a month.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. What is a request's full path under PD disaggregation? Which queues does each of the prefill and decode servers keep?
    2. What are `BaseKVSender`, `BaseKVReceiver` and `BaseKVBootstrapServer` each responsible for? Why is there a bootstrap?
    3. Why does the decode server have to pre-allocate? Why can `DecodePreallocQueue` and `DecodeTransferQueue` not be in the other order?
    4. Which three problems does the blog post say PD disaggregation solves?

??? success "Answers for the self-test (answer first, then open this)"
    1. A client's request goes to the load balancer first (`mini_lb.py` in the first version), which picks a prefill and a decode server, assigns the request a bootstrap room id and sends it to both. On the prefill side: `PrefillBootstrapQueue` (waiting for the decode side's handshake) → ordinary prefill scheduling → `send_kv_chunk` after each chunk is computed → an inflight queue waiting for the transfer to finish. On the decode side: `DecodePreallocQueue` (handshake, pre-allocate the KV slots) → `DecodeTransferQueue` (wait for the KV to arrive) → a "prebuilt extend" (the request's state set to already-prefilled) → the ordinary decode loop.
    2. The sender is on the prefill side, telling the far end how many KV slots there are in `init`, handing the KV indices to the transfer engine in `send` and checking progress with `poll`; the receiver is on the decode side, reporting the slots it pre-allocated in `init` and waiting for the data in `poll`; the bootstrap server is a small HTTP service that lets the two sides find each other by room id and exchange RDMA addresses and other metadata — without it, prefill does not know which machine's slots to send the KV to.
    3. The transfer is a one-sided RDMA write, so the sender needs the receiver's target addresses: the decode side must allocate the slots and tell prefill the addresses before prefill can start sending, and only once the transfer completes can the request enter a batch. Transferring before allocating is impossible, because there is no address.
    4. Prefill interrupting decode (jitter in decode's inter-token latency), the imbalance of prefill-to-decode ratios across ranks under DP attention, and DeepEP's two modes (normal suiting prefill, low-latency suiting decode) not being usable in one server at once.

A six-panel strip first, then the chapter:

<!-- comic ../assets/comics/pd.webp is in Chinese; put it back once the English version exists -->

## The 1410 lines' structure {#1410-行的结构}

```bash title="pd-initial.sh"
git show --stat=100 --format='%ad  %an  %s' --date=short c7c7dbebbe | grep -v '^$' | cut -c1-96
```

```text title="output"
2025-03-21  Byron Hsu  [PD] Release initial code (#4654)
 python/sglang/srt/disaggregation/conn.py        |  81 ++++++++
 python/sglang/srt/disaggregation/decode.py      | 495 +++++++++++++++++++++++++++++++++++++++++
 python/sglang/srt/disaggregation/mini_lb.py     | 285 +++++++++++++++++++++++++
 python/sglang/srt/disaggregation/prefill.py     | 249 ++++++++++++++++++++++
 python/sglang/srt/disaggregation/utils.py       |  44 ++++
 python/sglang/srt/managers/schedule_batch.py    |  22 +-
 python/sglang/srt/managers/scheduler.py         | 187 ++++++++++++++++-
 python/sglang/srt/managers/tokenizer_manager.py |  12 ++
 python/sglang/srt/mem_cache/memory_pool.py      |  13 ++
 python/sglang/srt/server_args.py                |  31 +++
 10 files changed, 1410 insertions(+), 9 deletions(-)
```

Five new files and four changes: `conn.py` is the transfer interface, `prefill.py` and `decode.py` are each a set of mixins (mixed into `Scheduler` and `ScheduleBatch`), `mini_lb.py` is the load balancer, and `scheduler.py` gained 187 lines to connect the mixins. Three weeks later #5328 "Add transfer backend abstraction" moved `conn.py` into `base/`, with Mooncake (#4880, 10 April) and NIXL (#5477, 21 April) each a subdirectory. The interface at v0.4.6:

```python title="python/sglang/srt/disaggregation/base/conn.py @ v0.4.6 L31-113" linenums="31"
class BaseKVManager(ABC):
    """Base class for managing transfers states"""

    @abstractmethod
    def __init__(
        self,
        args: KVArgs,
        disaggregation_mode: DisaggregationMode,
        server_args: ServerArgs,
    ): ...


class BaseKVSender(ABC):

    @abstractmethod
    def __init__(
        self, mgr: BaseKVManager, bootstrap_addr: str, bootstrap_room: int
    ): ...

    @abstractmethod
    def init(self, num_kv_indices: int, aux_index: Optional[int] = None):
        """
        Notify the decoder server about the kv indices length and aux index
        """
        ...

    @abstractmethod
    def send(self, kv_indices: npt.NDArray[np.int64]):
        """
        Send the kv cache at the given kv indices to the decoder server
        """
        ...

    @abstractmethod
    def poll(self) -> KVPoll:
        """
        Check the status of the kv cache transfer
        """
        ...

    @abstractmethod
    def failure_exception(self):
        """
        Raise an exception if the kv cache transfer fails
        """
        ...


class BaseKVReceiver(ABC):

    @abstractmethod
    def __init__(
        self,
        mgr: BaseKVManager,
        bootstrap_addr: str,
        bootstrap_room: Optional[int] = None,
    ): ...

    @abstractmethod
    def init(self, kv_indices: npt.NDArray[np.int64], aux_index: Optional[int] = None):
        """
        Notify the prefill server about the kv indices and aux index
        """
        ...

    @abstractmethod
    def poll(self) -> KVPoll:
        """
        Check the status of the kv cache transfer
        """
        ...

    @abstractmethod
    def failure_exception(self):
        """
        Raise an exception if the kv cache transfer fails
        """
        ...


class BaseKVBootstrapServer(ABC):
    @abstractmethod
    def __init__(self, port: int): ...
```

The four abstract classes come to 80 lines: the manager holds the transfer engine, the sender and receiver have three methods each (`init`, `send` or wait, `poll`), and the bootstrap server has only a port. Every transfer backend implements this one set — the same approach as last year's attention-backend abstraction ([chapter 10](../perf/restructure.md)).

![Figure: a request's path under PD disaggregation](../assets/figures/sgl-pd-flow.svg){.aig-svg}

## Two scheduling loops {#两个调度循环}

The prefill side's loop:

```python title="python/sglang/srt/disaggregation/prefill.py @ v0.4.6 L179-215" linenums="179"
    def event_loop_normal_disagg_prefill(self: Scheduler):
        """A normal scheduler loop for prefill worker in disaggregation mode."""

        while True:
            recv_reqs = self.recv_requests()
            self.process_input_requests(recv_reqs)
            self.waiting_queue.extend(
                self.disagg_prefill_bootstrap_queue.pop_bootstrapped()
            )
            self.process_prefill_chunk()
            batch = self.get_new_batch_prefill()

            # Handle DP attention
            if (
                self.server_args.enable_dp_attention
                or self.server_args.enable_sp_layernorm
            ):
                batch, _ = self.prepare_dp_attn_batch(batch)

            self.cur_batch = batch

            if batch:
                result = self.run_batch(batch)
                self.process_batch_result_disagg_prefill(batch, result)

            if len(self.disagg_prefill_inflight_queue) > 0:
                self.process_disagg_prefill_inflight_queue()

            if batch is None and len(self.disagg_prefill_inflight_queue) == 0:
                self.check_memory()
                self.new_token_ratio = self.init_new_token_ratio

            self.last_batch = batch
            # HACK (byronhsu): reset the batch_is_full flag because we never enter update_running_batch which resets it
            # Otherwise, it hangs under high concurrency
            self.running_batch.batch_is_full = False

```

Only three things differ from the ordinary loop: a received request goes into `PrefillBootstrapQueue` first and only `pop_bootstrapped` (the decode side has pre-allocated) moves it into the waiting queue; `process_batch_result_disagg_prefill` calls `send_kv_chunk` as soon as each chunk is computed — chunked prefill's chunk becomes the unit of transfer, without waiting for the whole prompt; and a request that has been sent goes into an inflight queue, with `process_disagg_prefill_inflight_queue` polling and only freeing its KV once the transfer completes. The decode side:

```python title="python/sglang/srt/disaggregation/decode.py @ v0.4.6 L457-498" linenums="457"
    def event_loop_normal_disagg_decode(self: Scheduler):
        """A normal scheduler loop for decode worker in disaggregation mode."""

        while True:
            recv_reqs = self.recv_requests()
            self.process_input_requests(recv_reqs)
            # polling and allocating kv cache
            self.process_decode_queue()
            batch = self.get_next_disagg_decode_batch_to_run()
            self.cur_batch = batch

            prepare_dp_attn_flag = (
                self.server_args.enable_dp_attention
                or self.server_args.enable_sp_layernorm
            )

            if batch:
                # Generate fake extend output.
                if batch.forward_mode.is_extend():
                    # Note: Logprobs should be handled on the prefill engine.
                    self.stream_output(batch.reqs, False)
                    if prepare_dp_attn_flag:
                        self._prepare_idle_batch_and_run(None)
                else:
                    if prepare_dp_attn_flag:
                        self.prepare_dp_attn_batch(batch)
                    result = self.run_batch(batch)
                    self.process_batch_result(batch, result)
            elif prepare_dp_attn_flag:
                batch, _ = self._prepare_idle_batch_and_run(None)

            if batch is None and (
                len(self.disagg_decode_transfer_queue.queue)
                + len(self.disagg_decode_prealloc_queue.queue)
                == 0
            ):
                # When the server is idle, do self-check and re-init some states
                self.check_memory()
                self.new_token_ratio = self.init_new_token_ratio

            self.last_batch = batch

```

`process_decode_queue` drives two queues: `DecodePreallocQueue` handshakes for a new request, pre-allocates the whole stretch of KV by `_allocatable_tokens` (the prompt and what is to be generated) and tells the far end the addresses; `DecodeTransferQueue` polls the receiver, and a request whose transfer is done goes through `get_new_prebuilt_batch` for a "prebuilt extend" — no actual forward pass, only setting the request's state to "the prefill is done and the last token is known" — and then enters decode like any other request. `event_loop_overlap_disagg_decode` existed in the first version the same day, but it was really reconciled with overlapped scheduling by #5608 and #5609 on 21 April.

## Connecting everything within a month {#一个月接上所有东西}

```bash title="pd-commits.sh"
for h in c7c7dbebbe 6d3b35fae9 4c31ae9f6d a9499885e9 ab4b5606e4 4dce1cc608 e65b9f21e3 bf98d2e377 711efe7814 e0673969b9 3f57b00a59; do
  git log -1 --date=short --format='%ad  %h  %s' $h | cut -c1-96
done
```

```text title="output"
2025-03-21  c7c7dbebbe  [PD] Release initial code (#4654)
2025-04-08  6d3b35fae9  [PD] Simplify mini LB (#4911)
2025-04-10  4c31ae9f6d  [PD] Support KV transfer with mooncake (#4880)
2025-04-13  a9499885e9  [PD] Add transfer backend abstraction (#5328)
2025-04-19  ab4b5606e4  [PD] Support page size > 1 (#5561)
2025-04-21  4dce1cc608  [PD] Add NIXL transfer backend  (#5477)
2025-04-21  e65b9f21e3  [PD] Support decode overlap schedule (#5608)
2025-04-21  bf98d2e377  [PD] Support prefill overlap + Ensure no race condition (#5609)
2025-04-23  711efe7814  Integrating PD disaggregation with DP attention and DeepEP (#5435)
2025-04-23  e0673969b9  [PD] Add support for dp attention with mooncake (#5530)
2025-04-21  3f57b00a59  Support PD bootstrap fields on /v1/chat/completions endpoint (#5488)
```

From 21 March to 23 April: the Mooncake transfer engine (RDMA scatter writes), the transfer backend abstraction, page sizes above 1, NIXL, overlapped scheduling for decode and prefill, integration with DP attention plus DeepEP, and the bootstrap fields on the OpenAI interface. The blog post of 5 May was written on this basis: DeepSeek-V3 on 96 H100s, prefill on 4 nodes (EP32) and decode on 9 (EP72), 52.3k input and 22.3k output tokens per second per node, and an output cost of $0.20 per million tokens. The blog's three motives — prefill interrupting decode, DP attention's imbalance, and DeepEP's two modes not coexisting — are exactly PD disaggregation's place in SGLang: it is not only a latency optimisation but a precondition for large-scale EP deployment ([the next chapter](large-ep.md)).

## Design trade-offs {#设计取舍}

- **The transfer is a one-sided RDMA write, so decode allocates first.** The receiver takes no part in the data plane and the sender writes to addresses directly; the price is that the decode side reserves a complete KV (including the output to come) for a request that has not arrived, with lower memory utilization than allocating on demand.
- **Sending by chunk.** Each prefill chunk is sent as soon as it is computed, overlapping the transfer with the computation; the price is that a request on the prefill side cannot be released until every chunk's transfer is done.
- **Mixins rather than subclasses.** `SchedulerDisaggregationPrefillMixin` mixes into the same `Scheduler`, so one body of code takes different loops by `--disaggregation-mode`; the price is `Scheduler`'s swelling method count.
- **A Python mini_lb first.** The 285-line load balancer only does round robin and bootstrap forwarding, which is enough to run; real PD routing (KV-aware, prefill-to-decode ratios) went into the Rust router later (the "PD Router" series from 2025-05, [chapter 21](../platform/gateway.md)).

## What happened afterwards {#后来怎么样了}

```bash title="pd-dir.sh"
REF=${REF:-29f6d408c0}
for t in v0.4.6 v0.5.0rc0 "$REF"; do
  printf '%-11s %3d 个文件：' "$t" "$(git ls-tree -r --name-only "$t" -- python/sglang/srt/disaggregation | grep -c '\.py$')"
  git ls-tree -r --name-only "$t" -- python/sglang/srt/disaggregation | sed 's|python/sglang/srt/disaggregation/||' | awk -F/ '{print (NF > 1 ? $1 "/" : $1)}' | sort -u | tr '\n' ' ' | cut -c1-130; echo
done
```

```text title="output"
v0.4.6       11 个文件：base/ decode.py mini_lb.py mooncake/ nixl/ prefill.py utils.py 

v0.5.0rc0    22 个文件：ascend/ base/ common/ decode.py decode_schedule_batch_mixin.py fake/ kv_events.py launch_lb.py mini_lb.py mooncake/ nixl/ prefill.

29f6d408c0   36 个文件：ascend/ base/ checksum.py common/ decode.py decode_hicache_mixin.py decode_kvcache_offload_manager.py decode_schedule_batch_mixin.
```

- The transfer backends extended from Mooncake and NIXL to Ascend and AMD's MoRI, plus a fake one for tests, with the shared staging buffers and packing logic in `common/`.
- `decode_kvcache_offload_manager.py` and `decode_hicache_mixin.py`: the decode side's KV can be offloaded and works with HiCache's storage tier (a prefill instance reusing a remote cache).
- `encoder/`: H2 2025 split multimodal visual encoding into its own service too (three stages, E/P/D).
- `role_switch.py`: an instance can switch between the prefill and decode roles, in preparation for elastic scheduling.
- The Q3 2025 roadmap's "PD CPU transfer #8210", PD with speculative decoding and PD with HiCache have all landed.

## Exercises {#练习}

**1. How much is pre-allocated.** Read v0.4.6's `DecodePreallocQueue._allocatable_tokens` and `_pre_alloc` and say how many slots the decode side reserves for a request, and how that relates to [chapter two](../origins/first-commit.md)'s `new_token_ratio` estimate.

??? success "A way to approach it"
    It reserves `len(prompt) + max_new_tokens` (page-aligned); because the KV needs an address before it can be received, it cannot reserve a fraction by an estimated coefficient as ordinary admission does, so the decode side's admission is more conservative than in the ordinary mode.

**2. What the handshake carries.** Use `git show v0.4.6:python/sglang/srt/disaggregation/mooncake/conn.py | grep -n 'def '` to read the Mooncake backend's sender and receiver, and list the fields exchanged at the handshake.

??? success "A way to approach it"
    The session id, the receiver's RDMA address and rkey, the KV pool's base address and each layer's offset, the pre-allocated slot indices, and where the auxiliary data (the last token and so on) goes.

**3. Verify the three motives.** The blog says DeepEP's normal mode suits prefill and low-latency suits decode. Find how `DeepEPMode.AUTO` is handled at the baseline commit and say how it picks a mode by role under PD disaggregation.

??? success "A way to approach it"
    `git grep -n 'DeepEPMode' 29f6d408c0 -- python/sglang/srt/layers/moe` finds the branch choosing normal or low_latency by `forward_mode` or `disaggregation_mode`.

!!! interview "How to explain it"
    "How is PD disaggregation implemented?" — Tell it by the data flow: the load balancer assigns a room id and sends to both sides; the decode side handshakes, pre-allocates the whole KV and tells prefill the addresses; prefill sends each chunk as it is computed with a one-sided RDMA write; and once it has arrived the decode side does a "prebuilt extend" and enters the ordinary loop. Then the abstraction: an 81-line sender / receiver / bootstrap interface makes Mooncake, NIXL and the rest pluggable. Finally the motive: not only latency isolation but the precondition for using each of DeepEP's two modes where it belongs.

## Summary {#小结}

- [x] #4654 (2025-03-21): 1410 lines, a transfer interface plus prefill and decode mixins plus mini_lb; the transfer backends got their own directory three weeks later.
- [x] Decode handshakes and pre-allocates first, prefill writes one-sidedly by chunk, and a prebuilt extend follows the transfer; Mooncake, NIXL, overlapped scheduling, DP attention and DeepEP all connected within a month.
- [x] The blog's three motives place PD disaggregation as a precondition for large-scale EP; it later extended to more transfer backends, KV offload, a separate encoder and role switching.
