<!-- 这个目录由 tools/export_examples.py 生成，不要手改；改正文里的代码块 -->

# 《计算机基础手册》正文里的代码

这里的 85 个文件逐字取自正文的代码块，页面是唯一的源；`tools/export_examples.py` 负责导出，构建时会检查两边一致。

怎么跑见每一页正文；需要的环境见仓库根目录的 `env/setup-macos.sh`（Mac）或 `setup-gpu.sh`（NVIDIA 显卡）。

| 文件 | 出自 | 备注 |
| --- | --- | --- |
| `affinity.py` | [os/process-thread.md](https://anrans.github.io/ai-infra-handbooks/cs/os/process-thread) |  |
| `asyncio_demo.py` | [os/io-models.md](https://anrans.github.io/ai-infra-handbooks/cs/os/io-models) |  |
| `backpressure.py` | [net/tcp.md](https://anrans.github.io/ai-infra-handbooks/cs/net/tcp) |  |
| `backtrack.py` | [algo/dp-backtrack.md](https://anrans.github.io/ai-infra-handbooks/cs/algo/dp-backtrack) |  |
| `balance.py` | [net/load-balance.md](https://anrans.github.io/ai-infra-handbooks/cs/net/load-balance) |  |
| `bdp.py` | [net/tcp.md](https://anrans.github.io/ai-infra-handbooks/cs/net/tcp) |  |
| `binary.py` | [algo/array-string.md](https://anrans.github.io/ai-infra-handbooks/cs/algo/array-string) |  |
| `bounded.py` | [dist/hash-shard.md](https://anrans.github.io/ai-infra-handbooks/cs/dist/hash-shard) |  |
| `branch.c` | [arch/cpu.md](https://anrans.github.io/ai-infra-handbooks/cs/arch/cpu) |  |
| `breakeven.py` | [arch/evolution.md](https://anrans.github.io/ai-infra-handbooks/cs/arch/evolution) |  |
| `cache_latency.c` | [arch/cpu.md](https://anrans.github.io/ai-infra-handbooks/cs/arch/cpu) |  |
| `cache_sim.py` | [arch/cpu.md](https://anrans.github.io/ai-infra-handbooks/cs/arch/cpu) |  |
| `cfs.py` | [os/process-thread.md](https://anrans.github.io/ai-infra-handbooks/cs/os/process-thread) |  |
| `cgroup_limits.py` | [os/containers.md](https://anrans.github.io/ai-infra-handbooks/cs/os/containers) |  |
| `cold_warm.py` | [os/io-stack.md](https://anrans.github.io/ai-infra-handbooks/cs/os/io-stack) |  |
| `collective.py` | [arch/multi-gpu.md](https://anrans.github.io/ai-infra-handbooks/cs/arch/multi-gpu) |  |
| `complexity.py` | [algo/overview.md](https://anrans.github.io/ai-infra-handbooks/cs/algo/overview) |  |
| `ctxsw.c` | [os/process-thread.md](https://anrans.github.io/ai-infra-handbooks/cs/os/process-thread) |  |
| `ctxsw_kinds.py` | [os/perf-tools.md](https://anrans.github.io/ai-infra-handbooks/cs/os/perf-tools) |  |
| `cuda_ipc.py` | [os/ipc.md](https://anrans.github.io/ai-infra-handbooks/cs/os/ipc) | 需要 GPU，手册里只做语法检查 |
| `dp.py` | [algo/dp-backtrack.md](https://anrans.github.io/ai-infra-handbooks/cs/algo/dp-backtrack) |  |
| `epoll_echo.py` | [os/io-models.md](https://anrans.github.io/ai-infra-handbooks/cs/os/io-models) |  |
| `exp_bound.py` | [arch/gpu-sm.md](https://anrans.github.io/ai-infra-handbooks/cs/arch/gpu-sm) |  |
| `false_sharing.c` | [arch/cpu.md](https://anrans.github.io/ai-infra-handbooks/cs/arch/cpu) |  |
| `faults.py` | [os/virtual-memory.md](https://anrans.github.io/ai-infra-handbooks/cs/os/virtual-memory) |  |
| `fork_lock.py` | [os/process-thread.md](https://anrans.github.io/ai-infra-handbooks/cs/os/process-thread) |  |
| `graph.py` | [algo/tree-graph.md](https://anrans.github.io/ai-infra-handbooks/cs/algo/tree-graph) |  |
| `greedy.py` | [algo/sort-heap-greedy.md](https://anrans.github.io/ai-infra-handbooks/cs/algo/sort-heap-greedy) |  |
| `handshake.py` | [net/tcp.md](https://anrans.github.io/ai-infra-handbooks/cs/net/tcp) |  |
| `hbm.py` | [arch/gpu-memory.md](https://anrans.github.io/ai-infra-handbooks/cs/arch/gpu-memory) |  |
| `heaps.py` | [algo/sort-heap-greedy.md](https://anrans.github.io/ai-infra-handbooks/cs/algo/sort-heap-greedy) |  |
| `hot.c` | [os/perf-tools.md](https://anrans.github.io/ai-infra-handbooks/cs/os/perf-tools) |  |
| `ilp.c` | [arch/cpu.md](https://anrans.github.io/ai-infra-handbooks/cs/arch/cpu) |  |
| `in_flight.py` | [arch/gpu-memory.md](https://anrans.github.io/ai-infra-handbooks/cs/arch/gpu-memory) |  |
| `ipc_bench.py` | [os/ipc.md](https://anrans.github.io/ai-infra-handbooks/cs/os/ipc) |  |
| `latency_hiding.py` | [arch/gpu-sm.md](https://anrans.github.io/ai-infra-handbooks/cs/arch/gpu-sm) |  |
| `linked.py` | [algo/linked-stack-hash.md](https://anrans.github.io/ai-infra-handbooks/cs/algo/linked-stack-hash) |  |
| `lru.py` | [algo/linked-stack-hash.md](https://anrans.github.io/ai-infra-handbooks/cs/algo/linked-stack-hash) |  |
| `mini_safetensors.py` | [os/io-stack.md](https://anrans.github.io/ai-infra-handbooks/cs/os/io-stack) |  |
| `mlock.py` | [os/pinned-numa.md](https://anrans.github.io/ai-infra-handbooks/cs/os/pinned-numa) |  |
| `modulo.py` | [dist/hash-shard.md](https://anrans.github.io/ai-infra-handbooks/cs/dist/hash-shard) |  |
| `monotonic.py` | [algo/linked-stack-hash.md](https://anrans.github.io/ai-infra-handbooks/cs/algo/linked-stack-hash) |  |
| `multiplex.py` | [net/http-stream.md](https://anrans.github.io/ai-infra-handbooks/cs/net/http-stream) |  |
| `nagle.py` | [net/tcp.md](https://anrans.github.io/ai-infra-handbooks/cs/net/tcp) |  |
| `ns_demo.py` | [os/containers.md](https://anrans.github.io/ai-infra-handbooks/cs/os/containers) |  |
| `numa.py` | [os/pinned-numa.md](https://anrans.github.io/ai-infra-handbooks/cs/os/pinned-numa) |  |
| `numabw.c` | [os/pinned-numa.md](https://anrans.github.io/ai-infra-handbooks/cs/os/pinned-numa) |  |
| `odirect.py` | [os/io-stack.md](https://anrans.github.io/ai-infra-handbooks/cs/os/io-stack) |  |
| `operand_bw.py` | [arch/gpu-sm.md](https://anrans.github.io/ai-infra-handbooks/cs/arch/gpu-sm) |  |
| `partition.py` | [dist/replication.md](https://anrans.github.io/ai-infra-handbooks/cs/dist/replication) |  |
| `peaks.py` | [arch/gpu-sm.md](https://anrans.github.io/ai-infra-handbooks/cs/arch/gpu-sm) |  |
| `perf_profile.py` | [os/perf-tools.md](https://anrans.github.io/ai-infra-handbooks/cs/os/perf-tools) |  |
| `pickle_cost.py` | [os/ipc.md](https://anrans.github.io/ai-infra-handbooks/cs/os/ipc) |  |
| `pitfalls.py` | [algo/overview.md](https://anrans.github.io/ai-infra-handbooks/cs/algo/overview) |  |
| `poll_vs_epoll.py` | [os/io-models.md](https://anrans.github.io/ai-infra-handbooks/cs/os/io-models) |  |
| `prefix.py` | [algo/array-string.md](https://anrans.github.io/ai-infra-handbooks/cs/algo/array-string) |  |
| `proc_stat.py` | [os/perf-tools.md](https://anrans.github.io/ai-infra-handbooks/cs/os/perf-tools) |  |
| `proxy_buffer.py` | [net/http-stream.md](https://anrans.github.io/ai-infra-handbooks/cs/net/http-stream) |  |
| `queueing.py` | [net/load-balance.md](https://anrans.github.io/ai-infra-handbooks/cs/net/load-balance) |  |
| `quorum.py` | [dist/replication.md](https://anrans.github.io/ai-infra-handbooks/cs/dist/replication) |  |
| `raft.py` | [dist/replication.md](https://anrans.github.io/ai-infra-handbooks/cs/dist/replication) |  |
| `reach.py` | [os/virtual-memory.md](https://anrans.github.io/ai-infra-handbooks/cs/os/virtual-memory) |  |
| `rendezvous.py` | [dist/hash-shard.md](https://anrans.github.io/ai-infra-handbooks/cs/dist/hash-shard) |  |
| `reorder.c` | [arch/cpu.md](https://anrans.github.io/ai-infra-handbooks/cs/arch/cpu) |  |
| `retry.py` | [net/load-balance.md](https://anrans.github.io/ai-infra-handbooks/cs/net/load-balance) |  |
| `reuse.py` | [net/tcp.md](https://anrans.github.io/ai-infra-handbooks/cs/net/tcp) |  |
| `ridge.py` | [arch/evolution.md](https://anrans.github.io/ai-infra-handbooks/cs/arch/evolution) |  |
| `ring.py` | [dist/hash-shard.md](https://anrans.github.io/ai-infra-handbooks/cs/dist/hash-shard) |  |
| `share.py` | [os/process-thread.md](https://anrans.github.io/ai-infra-handbooks/cs/os/process-thread) |  |
| `sharing.py` | [arch/multi-gpu.md](https://anrans.github.io/ai-infra-handbooks/cs/arch/multi-gpu) |  |
| `shm_ring.py` | [os/ipc.md](https://anrans.github.io/ai-infra-handbooks/cs/os/ipc) |  |
| `simd.c` | [arch/cpu.md](https://anrans.github.io/ai-infra-handbooks/cs/arch/cpu) |  |
| `sse.py` | [net/http-stream.md](https://anrans.github.io/ai-infra-handbooks/cs/net/http-stream) |  |
| `strace_reads.py` | [os/perf-tools.md](https://anrans.github.io/ai-infra-handbooks/cs/os/perf-tools) |  |
| `throttle_sim.py` | [os/containers.md](https://anrans.github.io/ai-infra-handbooks/cs/os/containers) |  |
| `tlb.c` | [os/virtual-memory.md](https://anrans.github.io/ai-infra-handbooks/cs/os/virtual-memory) |  |
| `transfer.py` | [os/pinned-numa.md](https://anrans.github.io/ai-infra-handbooks/cs/os/pinned-numa) |  |
| `translate.py` | [os/virtual-memory.md](https://anrans.github.io/ai-infra-handbooks/cs/os/virtual-memory) |  |
| `traverse.py` | [algo/tree-graph.md](https://anrans.github.io/ai-infra-handbooks/cs/algo/tree-graph) |  |
| `trie.py` | [algo/tree-graph.md](https://anrans.github.io/ai-infra-handbooks/cs/algo/tree-graph) |  |
| `uring_read.c` | [os/io-models.md](https://anrans.github.io/ai-infra-handbooks/cs/os/io-models) |  |
| `vmsize.py` | [os/virtual-memory.md](https://anrans.github.io/ai-infra-handbooks/cs/os/virtual-memory) |  |
| `windows.py` | [algo/array-string.md](https://anrans.github.io/ai-infra-handbooks/cs/algo/array-string) |  |
| `write_fsync.py` | [os/io-stack.md](https://anrans.github.io/ai-infra-handbooks/cs/os/io-stack) |  |
| `zmq_pushpull.py` | [os/ipc.md](https://anrans.github.io/ai-infra-handbooks/cs/os/ipc) |  |
