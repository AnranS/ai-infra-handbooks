# The process model's two restructurings: removing rpyc, the controller and tp_worker

<p class="lead">The first version drove one model process per tensor-parallel rank with rpyc remote calls, with the scheduler lodging inside rank 0's model process. That structure was changed twice in the first half of 2024: May's static data parallelism turned <code>router/</code> into <code>controller/</code>, and July's removal of rpyc let rank 0 live in the control process itself and broadcast requests to the other ranks with torch.distributed; at the end of September the scheduling code moved out of <code>tp_worker.py</code> into a separate <code>scheduler.py</code>, settling today's arrangement of one scheduler process per rank. This chapter reads those three steps through the changes to the <code>managers/</code> directory, looking at what each one solved and what it left behind.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. What was rpyc responsible for in the first version? Once it was gone, how do requests get from rank 0 to the other ranks?
    2. Why does every TP rank run an identical copy of the scheduler, rather than having rank 0 alone schedule and send the batch to everyone?
    3. How do `controller_single` and `controller_multi` relate? At which layer are data-parallel requests dispatched?
    4. On what day did `scheduler.py` appear? Which file was it taken out of, and what was left there?

??? success "Answers for the self-test (answer first, then open this)"
    1. rpyc let the router process call each rank's `ModelRpcServer.exposed_step` as if it were a local object (serialising the arguments and going through a thread pool on every RPC). Afterwards, rank 0's `ModelTpServer` is instantiated inside the control process directly while the other ranks are each a process whose loop first calls `broadcast_recv_input` to wait for rank 0 to broadcast the pickled request list with torch.distributed (a gloo CPU group) and then runs its own `exposed_step`.
    2. Scheduling is deterministic: given the same input and the same state, every rank computes the same batch, so it is enough to broadcast "the requests that arrived" once at the start of each step instead of sending each rank the batch's metadata (which requests, how many tokens, which KV slots). The price is that the scheduling's CPU work is done once per rank.
    3. `controller_single` manages one group of TP ranks (one data-parallel replica); `controller_multi` manages several replicas and hands a received request to one replica's `controller_single` by round robin or shortest queue (through a `multiprocessing.Queue` at the time). Dispatch lives at the controller layer, decoupled from scheduling.
    4. #1538 of 2024-09-29, "Move scheduler code from tp_worker.py to scheduler.py": 887 lines moved into a new file and `tp_worker.py` was left with only the model execution's wrapper (`TpModelWorker`), which is also what made October 2024's overlapped scheduling possible as "a scheduling thread plus an execution thread".

A six-panel strip first, then the chapter:

<!-- comic ../assets/comics/processes.webp is in Chinese; put it back once the English version exists -->

## Telling the story through the directory {#用目录讲故事}

The `managers/` directory in four versions:

```bash title="managers-tree.sh"
REF=${REF:-29f6d408c0}
for t in v0.1.5 v0.2.0 v0.3.0 v0.4.0; do
  echo "== $t"; git ls-tree -r --name-only $t -- python/sglang/srt/managers | sed 's|python/sglang/srt/managers/||' | tr '\n' ' '; echo
done
echo "== $REF：$(git ls-tree -r --name-only "$REF" -- python/sglang/srt/managers | grep -c '\.py$') 个文件"
```

```text title="output"
== v0.1.5
detokenizer_manager.py io_struct.py openai_protocol.py router/infer_batch.py router/manager.py router/model_rpc.py router/model_runner.py router/radix_cache.py router/scheduler.py tokenizer_manager.py 
== v0.2.0
controller/cuda_graph_runner.py controller/infer_batch.py controller/manager_multi.py controller/manager_single.py controller/model_runner.py controller/radix_cache.py controller/schedule_heuristic.py controller/tp_worker.py detokenizer_manager.py io_struct.py tokenizer_manager.py 
== v0.3.0
controller_multi.py controller_single.py detokenizer_manager.py io_struct.py policy_scheduler.py schedule_batch.py tokenizer_manager.py tp_worker.py 
== v0.4.0
data_parallel_controller.py detokenizer_manager.py image_processor.py io_struct.py schedule_batch.py schedule_policy.py scheduler.py session_controller.py tokenizer_manager.py tp_worker.py tp_worker_overlap_thread.py 
== 29f6d408c0：53 个文件
```

The four snapshots are four process models. The key commits in between:

```bash title="process-commits.sh"
for h in 0463f7fb52 d774acad5c cdcbde5fc3 048685430d f86c1e611f 23cc66f7b6 b48edff67f; do
  git log -1 --date=short --format='%ad  %h  %s' $h | cut -c1-100
done
```

```text title="output"
2024-05-27  0463f7fb52  Support data parallelism (static) (#480)
2024-07-18  d774acad5c  Remove the dependency of rpyc (#646)
2024-07-29  cdcbde5fc3  Code structure refactor (#807)
2024-09-29  048685430d  Improve process creation (#1534)
2024-09-29  f86c1e611f  Move scheduler code from tp_worker.py to scheduler.py (#1538)
2024-10-11  23cc66f7b6  Add back data parallelism (#1635)
2024-10-20  b48edff67f  Split the overlapped version of TpModelWorkerClient into a separate file (#1
```

![Figure: the process model's three stages](../assets/figures/sgl-process-evolution.svg){.aig-svg}

## Step one: static data parallelism (#480, 2024-05-27) {#第一步静态数据并行4802024-05-27}

[Chapter two](../origins/first-commit.md)'s first version has one router process and `--tp-size` decides the number of model processes. May's #480 added `--dp-size`: `router/` was renamed `controller/`, `model_rpc.py` became `tp_worker.py` (the name's first appearance), `manager.py` became `manager_single.py`, and `manager_multi.py` and `dp_worker.py` were added. `ControllerMulti` hands a received request to one data-parallel replica:

```python title="python/sglang/srt/managers/controller/manager_multi.py @ v0.2.0 L124-135" linenums="124"
    def round_robin_scheduler(self, input_requests):
        for r in input_requests:
            self.workers[self.round_robin_counter].queue.put(r)
            self.round_robin_counter = (self.round_robin_counter + 1) % len(
                self.workers
            )

    def shortest_queue_scheduler(self, input_requests):
        for r in input_requests:
            queue_sizes = [worker.queue.qsize() for worker in self.workers]
            wid = np.argmin(queue_sizes)
            self.workers[wid].queue.put(r)
```

There are only two policies, round robin and shortest queue, with the queue length taken straight from `multiprocessing.Queue.qsize()`. It does not know what is in each replica's cache — the paper's idea of a router keeping a meta tree and dispatching by prefix hit was not implemented yet and would not arrive until October's Rust router (chapter 13). "Static" means the replica count is fixed at startup, with one group of GPUs per replica (`gpu_ids = range(dp_worker_id * tp_size, …)`).

This step still uses rpyc: `dp_worker.py`'s 102 lines are there to start an rpyc client for each replica.

## Step two: removing rpyc (#646, 2024-07-18) {#第二步去掉-rpyc6462024-07-18}

Two months later a commit whose title is six words deleted 551 lines and added 303: rank 0's `ModelTpServer` is no longer a remote object but instantiated inside the control process directly, while the other ranks are started as ordinary processes by `launch_tp_servers`:

```python title="python/sglang/srt/managers/controller/manager_single.py @ v0.2.0 L55-91" linenums="55"
        # Launch other tp ranks
        tp_size_local = server_args.tp_size // server_args.nnodes
        self.tp_procs = []
        if tp_size_local > 1:
            tp_rank_range = range(1, tp_size_local)
            self.tp_procs = launch_tp_servers(
                gpu_ids,
                tp_rank_range,
                server_args,
                port_args.nccl_ports[dp_worker_id],
                model_overide_args,
            )

        # Launch tp rank 0
        self.tp_server = ModelTpServer(
            gpu_ids[0],
            0,
            server_args,
            port_args.nccl_ports[dp_worker_id],
            model_overide_args,
        )
        self.tp_cpu_group = self.tp_server.model_runner.tp_group.cpu_group

    def loop_for_forward(self):
        while True:
            if not self.is_dp_worker:
                recv_reqs = self.recv_requests_from_zmq()
            else:
                recv_reqs = self.recv_requests_from_mp_queue()

            if self.tp_size > 1:
                broadcast_recv_input(recv_reqs, 0, self.tp_cpu_group)

            out_pyobjs = self.tp_server.exposed_step(recv_reqs)

            for obj in out_pyobjs:
                self.send_to_detokenizer.send_pyobj(obj)
```

Each round of `loop_for_forward`: take every request backed up in ZMQ without blocking (`recv_requests_from_zmq`), broadcast the batch to the other ranks when `tp_size > 1`, and then run `exposed_step` in this process. The other ranks' loop and the broadcast's implementation:

```python title="python/sglang/srt/managers/controller/tp_worker.py @ v0.2.0 L730-753,772-800"
def run_tp_server(
    gpu_id: int,
    tp_rank: int,
    server_args: ServerArgs,
    nccl_port: int,
    model_overide_args: dict,
):
    """Run a tensor parallel server."""
    try:
        model_server = ModelTpServer(
            gpu_id,
            tp_rank,
            server_args,
            nccl_port,
            model_overide_args,
        )
        tp_cpu_group = model_server.model_runner.tp_group.cpu_group

        while True:
            recv_reqs = broadcast_recv_input(None, tp_rank, tp_cpu_group)
            model_server.exposed_step(recv_reqs)
    except Exception:
        logger.error("Exception in run_tp_server:\n" + get_exception_traceback())
        raise
...
def broadcast_recv_input(data, rank, dist_group):
    """Broadcast inputs from rank=0 to all other ranks with torch.dist backend."""

    if rank == 0:
        if len(data) == 0:
            tensor_size = torch.tensor([0], dtype=torch.long)
            dist.broadcast(tensor_size, src=0, group=dist_group)
        else:
            serialized_data = pickle.dumps(data)
            size = len(serialized_data)
            tensor_data = torch.ByteTensor(list(serialized_data))
            tensor_size = torch.tensor([size], dtype=torch.long)

            dist.broadcast(tensor_size, src=0, group=dist_group)
            dist.broadcast(tensor_data, src=0, group=dist_group)
    else:
        tensor_size = torch.tensor([0], dtype=torch.long)
        dist.broadcast(tensor_size, src=0, group=dist_group)
        size = tensor_size.item()

        if size == 0:
            return []

        tensor_data = torch.empty(size, dtype=torch.uint8)
        dist.broadcast(tensor_data, src=0, group=dist_group)

        serialized_data = bytes(tensor_data.tolist())
        data = pickle.loads(serialized_data)
        return data
```

`broadcast_recv_input` uses torch.distributed's **CPU process group** (`tp_group.cpu_group`, the gloo backend): rank 0 broadcasts the length first and then the pickled bytes, and broadcasts a single 0 when there are no new requests. This is the embryo of `broadcast_pyobj` in today's `Scheduler.recv_requests`. Three design points:

1. **Scheduling runs again on every rank.** Each rank receives the same request list and runs an identical `exposed_step` of its own (prefix matching, admission, forming the batch, sampling). As long as the scheduling is deterministic (the same random seed), every rank computes the same batch and the NCCL collectives line up naturally during the forward pass. The price is CPU work times TP; the gain is one broadcast of "the input" per step rather than a broadcast of all of the batch's metadata.
2. **Rank 0 and the control logic in one process.** One RPC hop is gone, and so are rpyc's thread pool and serialisation.
3. **The control plane and the data plane go separate ways.** Requests are broadcast over gloo on the CPU and tensors communicate over NCCL on the GPU, without interfering.

`ControllerSingle` also supports `--nnodes`: `tp_size_local = tp_size // nnodes`, so under multi-node tensor parallelism each machine starts its own share of the ranks.

## Step three: flattening the directory and the birth of `scheduler.py` {#第三步目录压平与-schedulerpy-的诞生}

#807 "Code structure refactor" of 29 July flattened `controller/`: `controller_single.py`, `controller_multi.py`, `tp_worker.py`, `schedule_batch.py` (formerly `infer_batch.py`) and `policy_scheduler.py` went directly under `managers/`, `radix_cache.py` and `memory_pool.py` moved into a new `mem_cache/`, and `model_runner.py` and `cuda_graph_runner.py` into `model_executor/` — the direct source of today's directory structure ([chapter 10](../perf/restructure.md) is devoted to that reorganisation).

Two commits of 29 September completed the last step. #1534 "Improve process creation" redid the process startup; #1538 moved 853 lines of scheduling code out of `tp_worker.py` into a new `managers/scheduler.py` (887 lines), leaving `tp_worker.py` with only `TpModelWorker`: it holds the `ModelRunner` and is responsible for "given a batch, run one forward pass and sample once". By v0.4.0 (2024-12) `scheduler.py` is 1500 lines and the process entry point looks like this:

```python title="python/sglang/srt/managers/scheduler.py @ v0.4.0 L1464-1500" linenums="1464"
def run_scheduler_process(
    server_args: ServerArgs,
    port_args: PortArgs,
    gpu_id: int,
    tp_rank: int,
    dp_rank: Optional[int],
    pipe_writer,
):
    # set cpu affinity to this gpu process
    if get_bool_env_var("SGLANG_SET_CPU_AFFINITY"):
        set_gpu_proc_affinity(server_args.tp_size, server_args.nnodes, gpu_id)

    # [For Router] if env var "SGLANG_DP_RANK" exist, set dp_rank to the value of the env var
    if dp_rank is None and "SGLANG_DP_RANK" in os.environ:
        dp_rank = int(os.environ["SGLANG_DP_RANK"])

    if dp_rank is None:
        configure_logger(server_args, prefix=f" TP{tp_rank}")
    else:
        configure_logger(server_args, prefix=f" DP{dp_rank} TP{tp_rank}")

    suppress_other_loggers()
    parent_process = psutil.Process().parent()

    try:
        scheduler = Scheduler(server_args, port_args, gpu_id, tp_rank, dp_rank)
        pipe_writer.send(
            {"status": "ready", "max_total_num_tokens": scheduler.max_total_num_tokens}
        )
        if scheduler.enable_overlap:
            scheduler.event_loop_overlap()
        else:
            scheduler.event_loop_normal()
    except Exception:
        traceback = get_exception_traceback()
        logger.error(f"Scheduler hit an exception: {traceback}")
        parent_process.send_signal(signal.SIGQUIT)
```

One `Scheduler` process per TP rank, building a `TpModelWorker` (or the overlapping `TpModelWorkerClient`) inside it and choosing between `event_loop_normal` and `event_loop_overlap`. `controller_single` and `controller_multi` disappeared in this restructuring and data parallelism was removed for a while, added back on 11 October by #1635 "Add back data parallelism" in the form of `data_parallel_controller.py`: the DP controller is a process of its own that starts each replica's scheduler processes and dispatches requests, with the dispatch moved from a `multiprocessing.Queue` to ZMQ.

## Today: what processes exist on one node {#今天一个节点上有哪些进程}

At the baseline commit the processes are started by `_launch_scheduler_processes` in `entrypoints/engine.py` (`_launch_subprocesses` starts it first and then the detokenizer process and the HTTP service):

```python title="python/sglang/srt/entrypoints/engine.py @ 29f6d408c0 L888-947" linenums="888"
        use_dp_controller = (
            get_parallel().num_dp_ranks > 1 or get_exec().moe.ep_join_mode == "scale"
        )

        if not use_dp_controller:
            # Launch tensor parallel scheduler processes
            memory_saver_adapter = TorchMemorySaverAdapter.create(
                enable=get_exec().features.enable_memory_saver
            )
            scheduler_pipe_readers = []

            pp_rank_range, tp_rank_range, pp_size_per_node, tp_size_per_node = (
                _calculate_rank_ranges(get_parallel().node_rank)
            )

            for pp_rank in pp_rank_range:
                for tp_rank in tp_rank_range:
                    reader, writer = mp.Pipe(duplex=False)
                    gpu_id = (
                        get_device().base_gpu_id
                        + ((pp_rank % pp_size_per_node) * tp_size_per_node)
                        + (tp_rank % tp_size_per_node) * get_device().gpu_id_step
                    )

                    with maybe_reindex_device_id(gpu_id) as gpu_id:
                        proc = mp.Process(
                            target=run_scheduler_process_func,
                            args=(
                                server_args,
                                port_args,
                                gpu_id,
                                tp_rank,
                                pp_rank,
                                None,
                                writer,
                            ),
                        )
                        with (
                            memory_saver_adapter.configure_subprocess(),
                            numa_utils.configure_subprocess(server_args, gpu_id),
                        ):
                            proc.start()

                    scheduler_procs.append(proc)
                    scheduler_pipe_readers.append(reader)
        else:
            # Launch the data parallel controller
            reader, writer = mp.Pipe(duplex=False)
            scheduler_pipe_readers = [reader]
            proc = mp.Process(
                target=run_data_parallel_controller_process,
                kwargs=dict(
                    server_args=server_args,
                    port_args=port_args,
                    pipe_writer=writer,
                    run_scheduler_process_func=run_scheduler_process_func,
                ),
            )
            proc.start()
            scheduler_procs.append(proc)
```

Two nested loops: one scheduler process for each pipeline-parallel stage times each tensor-parallel rank (the `gpu_id` computed from pp_rank and tp_rank, each subprocess reporting readiness through a `Pipe`); when `dp_size > 1` it starts a `DataParallelController` process instead, which starts `dp_size` groups of schedulers; detokenizing is still its own process; and the main process holds the `TokenizerManager` and the HTTP or gRPC service. Against the first version, the triangle of "tokenize → schedule plus execute → detokenize" is unchanged, and what changed is that the middle corner went from "one router process plus N rpyc servers" to "N peer scheduler processes".

## Design trade-offs {#设计取舍}

| Design | How a request reaches each rank | Where the scheduling is | The price |
| --- | --- | --- | --- |
| The first version (rpyc) | the router process sends an RPC to each rank | rank 0's model process | one RPC serialisation plus a thread pool per step; the router process is an extra hop |
| v0.2 (broadcast) | rank 0 broadcasts a pickle over gloo | the control process (which includes rank 0) | the scheduling's CPU work times TP; pickling a large request (multimodal) is slow |
| v0.4 to now (peer schedulers) | the same, via `broadcast_pyobj` | each rank's Scheduler process | the same, plus one hop through the DP controller |

The decision to "schedule again on every rank" has never been overturned, because it turns the hardest problem (every rank's batch being identical) into the two easy conditions of identical input and determinism. The later overlapped scheduling, speculative decoding and PD disaggregation all rest on that premise; what really changed is the dispatch layer: in 2025 the Rust gateway took over routing between instances (chapter 21) and the `DataParallelController` handles the replicas within one instance.

## What happened afterwards {#后来怎么样了}

- 2024-10: `tp_worker_overlap_thread.py` (#1726) — `TpModelWorkerClient` moves the forward pass onto another thread so the scheduler thread no longer waits for the GPU ([chapter 12](../perf/overlap.md)).
- 2025-01: `entrypoints/engine.py` becomes the unified startup entry point, with the HTTP service, the offline `Engine` and the engine embedded in an RL framework all sharing one subprocess-launching path.
- 2025-05: pipeline parallelism arrives and the scheduler process count becomes PP x TP.
- H2 2025: the `DataParallelController` learns to dispatch by a load budget (`DPBudget`) and to add and remove replicas elastically (`add_elastic_workers`), and `scheduler.py` splits out `scheduler_components/` and a set of mixins (output handling, PP, control messages).

## Exercises {#练习}

**1. What is broadcast.** Read v0.2.0's `broadcast_recv_input` and explain why a "length" tensor is broadcast first. What happens when a request carries a 4 MB image?

??? success "Answer"
    `dist.broadcast` requires the receiver to know the tensor's size in advance, so the length goes first and the content second. An image travels inside the request object as `pixel_values` (a numpy array) and is pickled and broadcast byte by byte to every rank; a multimodal request therefore costs an extra serialisation and copy of a large object on the CPU, and later versions moved the image handling into the `TokenizerManager` and used a hash as the cache key to reduce the repeated transfers.

**2. Why DP disappeared for a while.** Use `git log --date=short --format='%ad %h %s' -- python/sglang/srt/managers/controller_multi.py` to find the commit that deleted `controller_multi.py`, and see how the first version of `data_parallel_controller.py` from the same period differs from it.

??? success "A way to approach it"
    `controller_multi.py` was deleted around #1534 and #1538 (2024-09-29); the `data_parallel_controller.py` created in #1635 (2024-10-11) no longer uses a `multiprocessing.Queue` but gives each replica a ZMQ port, with the controller itself a process; the load-balancing policies kept round robin and shortest queue.

**3. Count today's processes.** Read the baseline commit's `_launch_subprocesses` and write out how many processes `--tp-size 4 --pp-size 2 --dp-size 2` starts on one machine, and what each one is.

??? success "Answer"
    1 DP controller; PP x TP = 8 scheduler processes per DP replica, 16 for the two replicas; 1 detokenizer; 1 main process (HTTP plus the TokenizerManager). 19 processes in all (not counting any multimodal processing subprocesses).

!!! interview "How to explain it"
    "How does SGLang keep its multi-card scheduling consistent?" — The short version is "every rank runs an identical scheduler and only the new requests are broadcast each step", followed by why that works (determinism) and what it costs (repeated CPU work). Adding where it came from (the restructuring of 2024-07 that removed rpyc) and what followed (the overlapping thread, the DP controller, PP) shows you know it is a deliberate design and not an accident.

## Summary {#小结}

- [x] `router/` (rpyc) → `controller/` (static DP, #480) → rpyc removed, rank 0 into the control process, requests broadcast over gloo (#646) → the directory flattened (#807) → `scheduler.py` on its own (#1538).
- [x] "Every rank schedules again and only the input is broadcast" is a decision that runs through to today; the dispatch layer went from a `multiprocessing.Queue` to ZMQ to the Rust gateway.
- [x] Today: a main process (tokenizing plus HTTP), PP x TP scheduler processes and a detokenizer process, plus a controller process when DP is on.
