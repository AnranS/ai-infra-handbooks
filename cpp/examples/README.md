<!-- 这个目录由 tools/export_examples.py 生成，不要手改；改正文里的代码块 -->

# 《C++ 进阶手册》正文里的代码

这里的 82 个文件逐字取自正文的代码块，页面是唯一的源；`tools/export_examples.py` 负责导出，构建时会检查两边一致。

怎么跑见每一页正文；需要的环境见仓库根目录的 `env/setup-macos.sh`（Mac）或 `setup-gpu.sh`（NVIDIA 显卡）。

| 文件 | 出自 | 备注 |
| --- | --- | --- |
| `a.cpp` | [basics/compile-ub.md](https://anrans.github.io/ai-infra-handbooks/cpp/basics/compile-ub) |  |
| `adapter_cache.cpp` | [basics/ownership.md](https://anrans.github.io/ai-infra-handbooks/cpp/basics/ownership) |  |
| `aligned.cpp` | [memory/layout.md](https://anrans.github.io/ai-infra-handbooks/cpp/memory/layout) |  |
| `aligned_allocator.cpp` | [memory/allocators.md](https://anrans.github.io/ai-infra-handbooks/cpp/memory/allocators) |  |
| `aligned_read.cpp` | [basics/compile-ub.md](https://anrans.github.io/ai-infra-handbooks/cpp/basics/compile-ub) |  |
| `aos_soa.cpp` | [memory/layout.md](https://anrans.github.io/ai-infra-handbooks/cpp/memory/layout) |  |
| `arena.cpp` | [memory/allocators.md](https://anrans.github.io/ai-infra-handbooks/cpp/memory/allocators) |  |
| `assertions.cpp` | [basics/compile-ub.md](https://anrans.github.io/ai-infra-handbooks/cpp/basics/compile-ub) |  |
| `atomic_max.cpp` | [concurrency/atomics.md](https://anrans.github.io/ai-infra-handbooks/cpp/concurrency/atomics) |  |
| `atomic_wait.cpp` | [concurrency/atomics.md](https://anrans.github.io/ai-infra-handbooks/cpp/concurrency/atomics) |  |
| `b.cpp` | [basics/compile-ub.md](https://anrans.github.io/ai-infra-handbooks/cpp/basics/compile-ub) |  |
| `bench_containers.cpp` | [basics/stl-perf.md](https://anrans.github.io/ai-infra-handbooks/cpp/basics/stl-perf) |  |
| `bits.cpp` | [basics/compile-ub.md](https://anrans.github.io/ai-infra-handbooks/cpp/basics/compile-ub) |  |
| `block_pool.cpp` | [memory/allocators.md](https://anrans.github.io/ai-infra-handbooks/cpp/memory/allocators) |  |
| `block_table.cpp` | [basics/move.md](https://anrans.github.io/ai-infra-handbooks/cpp/basics/move) |  |
| `blocking_queue.hpp` | [concurrency/threads.md](https://anrans.github.io/ai-infra-handbooks/cpp/concurrency/threads) |  |
| `bounded_queue.cpp` | [concurrency/threads.md](https://anrans.github.io/ai-infra-handbooks/cpp/concurrency/threads) |  |
| `build.sh` | [engineering/build-debug.md](https://anrans.github.io/ai-infra-handbooks/cpp/engineering/build-debug) |  |
| `build_pybind.sh` | [engineering/python-binding.md](https://anrans.github.io/ai-infra-handbooks/cpp/engineering/python-binding) |  |
| `build_torch_ext.sh` | [engineering/python-binding.md](https://anrans.github.io/ai-infra-handbooks/cpp/engineering/python-binding) |  |
| `caching_allocator.cpp` | [memory/allocators.md](https://anrans.github.io/ai-infra-handbooks/cpp/memory/allocators) |  |
| `concept_error.cpp` | [basics/templates.md](https://anrans.github.io/ai-infra-handbooks/cpp/basics/templates) |  |
| `constexpr_config.cpp` | [basics/templates.md](https://anrans.github.io/ai-infra-handbooks/cpp/basics/templates) |  |
| `counter.hpp` | [basics/compile-ub.md](https://anrans.github.io/ai-infra-handbooks/cpp/basics/compile-ub) |  |
| `counter_main.cpp` | [basics/compile-ub.md](https://anrans.github.io/ai-infra-handbooks/cpp/basics/compile-ub) |  |
| `custom_deleter.cpp` | [basics/ownership.md](https://anrans.github.io/ai-infra-handbooks/cpp/basics/ownership) |  |
| `cycle_leak.cpp` | [basics/ownership.md](https://anrans.github.io/ai-infra-handbooks/cpp/basics/ownership) |  |
| `dangling.cpp` | [basics/compile-ub.md](https://anrans.github.io/ai-infra-handbooks/cpp/basics/compile-ub) |  |
| `dangling_view.cpp` | [basics/stl-perf.md](https://anrans.github.io/ai-infra-handbooks/cpp/basics/stl-perf) |  |
| `dispatch.cpp` | [basics/templates.md](https://anrans.github.io/ai-infra-handbooks/cpp/basics/templates) |  |
| `dispatch96.cpp` | [basics/templates.md](https://anrans.github.io/ai-infra-handbooks/cpp/basics/templates) |  |
| `double_free.cpp` | [basics/value-raii.md](https://anrans.github.io/ai-infra-handbooks/cpp/basics/value-raii) |  |
| `false_sharing.cpp` | [memory/layout.md](https://anrans.github.io/ai-infra-handbooks/cpp/memory/layout) |  |
| `flag_barrier_allreduce.cpp` | [engineering/reading-code.md](https://anrans.github.io/ai-infra-handbooks/cpp/engineering/reading-code) |  |
| `forwarding.cpp` | [basics/move.md](https://anrans.github.io/ai-infra-handbooks/cpp/basics/move) |  |
| `if_constexpr.cpp` | [basics/templates.md](https://anrans.github.io/ai-infra-handbooks/cpp/basics/templates) |  |
| `include/kvpool/block_pool.hpp` | [engineering/build-debug.md](https://anrans.github.io/ai-infra-handbooks/cpp/engineering/build-debug) |  |
| `jthread_worker.cpp` | [concurrency/threads.md](https://anrans.github.io/ai-infra-handbooks/cpp/concurrency/threads) |  |
| `kvpool_py.cpp` | [engineering/python-binding.md](https://anrans.github.io/ai-infra-handbooks/cpp/engineering/python-binding) |  |
| `layout.cpp` | [memory/layout.md](https://anrans.github.io/ai-infra-handbooks/cpp/memory/layout) |  |
| `lifetime.cpp` | [basics/value-raii.md](https://anrans.github.io/ai-infra-handbooks/cpp/basics/value-raii) |  |
| `lock_order.cpp` | [concurrency/threads.md](https://anrans.github.io/ai-infra-handbooks/cpp/concurrency/threads) |  |
| `member_order.cpp` | [basics/value-raii.md](https://anrans.github.io/ai-infra-handbooks/cpp/basics/value-raii) |  |
| `misaligned.cpp` | [basics/compile-ub.md](https://anrans.github.io/ai-infra-handbooks/cpp/basics/compile-ub) |  |
| `move_buffer.cpp` | [basics/move.md](https://anrans.github.io/ai-infra-handbooks/cpp/basics/move) |  |
| `mutex_counter.cpp` | [concurrency/threads.md](https://anrans.github.io/ai-infra-handbooks/cpp/concurrency/threads) |  |
| `my_latch.cpp` | [concurrency/atomics.md](https://anrans.github.io/ai-infra-handbooks/cpp/concurrency/atomics) |  |
| `noexcept_realloc.cpp` | [basics/move.md](https://anrans.github.io/ai-infra-handbooks/cpp/basics/move) |  |
| `overflow.cpp` | [basics/compile-ub.md](https://anrans.github.io/ai-infra-handbooks/cpp/basics/compile-ub) |  |
| `pmr.cpp` | [memory/allocators.md](https://anrans.github.io/ai-infra-handbooks/cpp/memory/allocators) |  |
| `pool_demo.cpp` | [concurrency/lockfree-pool.md](https://anrans.github.io/ai-infra-handbooks/cpp/concurrency/lockfree-pool) |  |
| `pool_idle.cpp` | [concurrency/lockfree-pool.md](https://anrans.github.io/ai-infra-handbooks/cpp/concurrency/lockfree-pool) |  |
| `producer_consumer.cpp` | [concurrency/threads.md](https://anrans.github.io/ai-infra-handbooks/cpp/concurrency/threads) |  |
| `publish_acquire.cpp` | [concurrency/atomics.md](https://anrans.github.io/ai-infra-handbooks/cpp/concurrency/atomics) |  |
| `publish_relaxed.cpp` | [concurrency/atomics.md](https://anrans.github.io/ai-infra-handbooks/cpp/concurrency/atomics) |  |
| `race.cpp` | [concurrency/threads.md](https://anrans.github.io/ai-infra-handbooks/cpp/concurrency/threads) |  |
| `raii_buffer.cpp` | [basics/value-raii.md](https://anrans.github.io/ai-infra-handbooks/cpp/basics/value-raii) |  |
| `rehash.cpp` | [basics/stl-perf.md](https://anrans.github.io/ai-infra-handbooks/cpp/basics/stl-perf) |  |
| `rmsnorm_ext.cpp` | [engineering/python-binding.md](https://anrans.github.io/ai-infra-handbooks/cpp/engineering/python-binding) |  |
| `run_torch_ext.py` | [engineering/python-binding.md](https://anrans.github.io/ai-infra-handbooks/cpp/engineering/python-binding) |  |
| `scheduler_fixed.cpp` | [basics/compile-ub.md](https://anrans.github.io/ai-infra-handbooks/cpp/basics/compile-ub) |  |
| `scope_exit.cpp` | [basics/value-raii.md](https://anrans.github.io/ai-infra-handbooks/cpp/basics/value-raii) |  |
| `seq_state.cpp` | [memory/layout.md](https://anrans.github.io/ai-infra-handbooks/cpp/memory/layout) |  |
| `shared_prefix.cpp` | [basics/ownership.md](https://anrans.github.io/ai-infra-handbooks/cpp/basics/ownership) |  |
| `spinlock.cpp` | [concurrency/atomics.md](https://anrans.github.io/ai-infra-handbooks/cpp/concurrency/atomics) |  |
| `spsc_demo.cpp` | [concurrency/lockfree-pool.md](https://anrans.github.io/ai-infra-handbooks/cpp/concurrency/lockfree-pool) |  |
| `spsc_queue.hpp` | [concurrency/lockfree-pool.md](https://anrans.github.io/ai-infra-handbooks/cpp/concurrency/lockfree-pool) |  |
| `src/block_pool.cpp` | [engineering/build-debug.md](https://anrans.github.io/ai-infra-handbooks/cpp/engineering/build-debug) |  |
| `template_basics.cpp` | [basics/templates.md](https://anrans.github.io/ai-infra-handbooks/cpp/basics/templates) |  |
| `test_kvpool.py` | [engineering/python-binding.md](https://anrans.github.io/ai-infra-handbooks/cpp/engineering/python-binding) |  |
| `tests/test_block_pool.cpp` | [engineering/build-debug.md](https://anrans.github.io/ai-infra-handbooks/cpp/engineering/build-debug) |  |
| `thread_pool.hpp` | [concurrency/lockfree-pool.md](https://anrans.github.io/ai-infra-handbooks/cpp/concurrency/lockfree-pool) |  |
| `too_big.cpp` | [basics/templates.md](https://anrans.github.io/ai-infra-handbooks/cpp/basics/templates) |  |
| `top_p.cpp` | [basics/stl-perf.md](https://anrans.github.io/ai-infra-handbooks/cpp/basics/stl-perf) |  |
| `topk.cpp` | [basics/stl-perf.md](https://anrans.github.io/ai-infra-handbooks/cpp/basics/stl-perf) |  |
| `traverse.cpp` | [memory/layout.md](https://anrans.github.io/ai-infra-handbooks/cpp/memory/layout) |  |
| `two_resources.cpp` | [basics/value-raii.md](https://anrans.github.io/ai-infra-handbooks/cpp/basics/value-raii) |  |
| `unique_basics.cpp` | [basics/ownership.md](https://anrans.github.io/ai-infra-handbooks/cpp/basics/ownership) |  |
| `value_semantics.cpp` | [basics/value-raii.md](https://anrans.github.io/ai-infra-handbooks/cpp/basics/value-raii) |  |
| `vector_growth.cpp` | [basics/stl-perf.md](https://anrans.github.io/ai-infra-handbooks/cpp/basics/stl-perf) |  |
| `views.cpp` | [basics/stl-perf.md](https://anrans.github.io/ai-infra-handbooks/cpp/basics/stl-perf) |  |
| `weak_callback.cpp` | [basics/ownership.md](https://anrans.github.io/ai-infra-handbooks/cpp/basics/ownership) |  |
