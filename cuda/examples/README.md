<!-- 这个目录由 tools/export_examples.py 生成，不要手改；改正文里的代码块 -->

# 《CUDA 进阶手册》正文里的代码

这里的 81 个文件逐字取自正文的代码块，页面是唯一的源；`tools/export_examples.py` 负责导出，构建时会检查两边一致。

怎么跑见每一页正文；需要的环境见仓库根目录的 `env/setup-macos.sh`（Mac）或 `setup-gpu.sh`（NVIDIA 显卡）。

| 文件 | 出自 | 备注 |
| --- | --- | --- |
| `access_pattern.cu` | [basics/memory.md](https://anrans.github.io/ai-infra-handbooks/cuda/basics/memory) |  |
| `argmax.cu` | [kernels/reduction.md](https://anrans.github.io/ai-infra-handbooks/cuda/kernels/reduction) |  |
| `atomic_max_float.cu` | [basics/sync-warp.md](https://anrans.github.io/ai-infra-handbooks/cuda/basics/sync-warp) |  |
| `bank_conflict.cu` | [basics/memory.md](https://anrans.github.io/ai-infra-handbooks/cuda/basics/memory) |  |
| `bench_timing.cu` | [basics/first-kernel.md](https://anrans.github.io/ai-infra-handbooks/cuda/basics/first-kernel) |  |
| `block_sum.cu` | [basics/sync-warp.md](https://anrans.github.io/ai-infra-handbooks/cuda/basics/sync-warp) |  |
| `cluster_sum.cu` | [advanced/async-hopper.md](https://anrans.github.io/ai-infra-handbooks/cuda/advanced/async-hopper) |  |
| `common.cuh` | [basics/first-kernel.md](https://anrans.github.io/ai-infra-handbooks/cuda/basics/first-kernel) |  |
| `compact.cu` | [kernels/scan.md](https://anrans.github.io/ai-infra-handbooks/cuda/kernels/scan) |  |
| `conv1d.cu` | [basics/memory.md](https://anrans.github.io/ai-infra-handbooks/cuda/basics/memory) |  |
| `cublas_gemm.cu` | [tools/ecosystem.md](https://anrans.github.io/ai-infra-handbooks/cuda/tools/ecosystem) |  |
| `cuda_graph.cu` | [tools/streams.md](https://anrans.github.io/ai-infra-handbooks/cuda/tools/streams) |  |
| `cuda_graph_decode.py` | [framework/cuda-runtime.md](https://anrans.github.io/ai-infra-handbooks/cuda/framework/cuda-runtime) | 需要 GPU，手册里只做语法检查 |
| `custom_ops.py` | [framework/dispatcher.md](https://anrans.github.io/ai-infra-handbooks/cuda/framework/dispatcher) |  |
| `cute_algebra.cu` | [advanced/cute-layout.md](https://anrans.github.io/ai-infra-handbooks/cuda/advanced/cute-layout) |  |
| `cute_layout.cu` | [tools/ecosystem.md](https://anrans.github.io/ai-infra-handbooks/cuda/tools/ecosystem) |  |
| `decode_gaps.py` | [tools/pdl-megakernel.md](https://anrans.github.io/ai-infra-handbooks/cuda/tools/pdl-megakernel) |  |
| `device_query.cu` | [basics/first-kernel.md](https://anrans.github.io/ai-infra-handbooks/cuda/basics/first-kernel) |  |
| `dispatch_keys.py` | [framework/dispatcher.md](https://anrans.github.io/ai-infra-handbooks/cuda/framework/dispatcher) |  |
| `divergence.cu` | [basics/execution.md](https://anrans.github.io/ai-infra-handbooks/cuda/basics/execution) |  |
| `expand_repeat.py` | [framework/tensor.md](https://anrans.github.io/ai-infra-handbooks/cuda/framework/tensor) |  |
| `flash_attn.cu` | [advanced/attention.md](https://anrans.github.io/ai-infra-handbooks/cuda/advanced/attention) |  |
| `fused_add_rms_norm_ext.cu` | [tools/ecosystem.md](https://anrans.github.io/ai-infra-handbooks/cuda/tools/ecosystem) |  |
| `fx_fusion.py` | [framework/compilers.md](https://anrans.github.io/ai-infra-handbooks/cuda/framework/compilers) |  |
| `gemm.cu` | [kernels/gemm.md](https://anrans.github.io/ai-infra-handbooks/cuda/kernels/gemm) |  |
| `gemm_cp_async.cu` | [advanced/async-hopper.md](https://anrans.github.io/ai-infra-handbooks/cuda/advanced/async-hopper) |  |
| `gemv_w4.cu` | [advanced/quantization.md](https://anrans.github.io/ai-infra-handbooks/cuda/advanced/quantization) |  |
| `gqa_broadcast.py` | [framework/tensor.md](https://anrans.github.io/ai-infra-handbooks/cuda/framework/tensor) |  |
| `graph_break.py` | [framework/compile.md](https://anrans.github.io/ai-infra-handbooks/cuda/framework/compile) |  |
| `graph_walk.py` | [framework/autograd.md](https://anrans.github.io/ai-infra-handbooks/cuda/framework/autograd) |  |
| `grid_stride.cu` | [basics/first-kernel.md](https://anrans.github.io/ai-infra-handbooks/cuda/basics/first-kernel) |  |
| `histogram.cu` | [basics/sync-warp.md](https://anrans.github.io/ai-infra-handbooks/cuda/basics/sync-warp) |  |
| `inductor_fusion.py` | [framework/compile.md](https://anrans.github.io/ai-infra-handbooks/cuda/framework/compile) |  |
| `inference_mode.py` | [framework/autograd.md](https://anrans.github.io/ai-infra-handbooks/cuda/framework/autograd) |  |
| `layernorm.cu` | [kernels/softmax-norm.md](https://anrans.github.io/ai-infra-handbooks/cuda/kernels/softmax-norm) |  |
| `layout_algebra.py` | [advanced/cute-layout.md](https://anrans.github.io/ai-infra-handbooks/cuda/advanced/cute-layout) |  |
| `layout_basics.py` | [advanced/cute-layout.md](https://anrans.github.io/ai-infra-handbooks/cuda/advanced/cute-layout) |  |
| `layout_compose.py` | [advanced/cute-layout.md](https://anrans.github.io/ai-infra-handbooks/cuda/advanced/cute-layout) |  |
| `layout_core.py` | [advanced/cute-layout.md](https://anrans.github.io/ai-infra-handbooks/cuda/advanced/cute-layout) |  |
| `layout_mma.py` | [advanced/cute-layout.md](https://anrans.github.io/ai-infra-handbooks/cuda/advanced/cute-layout) |  |
| `layout_partition.py` | [advanced/cute-layout.md](https://anrans.github.io/ai-infra-handbooks/cuda/advanced/cute-layout) |  |
| `linear_gelu_saved.py` | [framework/autograd.md](https://anrans.github.io/ai-infra-handbooks/cuda/framework/autograd) |  |
| `log_aten_ops.py` | [framework/dispatcher.md](https://anrans.github.io/ai-infra-handbooks/cuda/framework/dispatcher) |  |
| `loops.py` | [framework/compiler-basics.md](https://anrans.github.io/ai-infra-handbooks/cuda/framework/compiler-basics) |  |
| `managed_prefetch.cu` | [tools/streams.md](https://anrans.github.io/ai-infra-handbooks/cuda/tools/streams) |  |
| `matrix_add.cu` | [basics/first-kernel.md](https://anrans.github.io/ai-infra-handbooks/cuda/basics/first-kernel) |  |
| `memory_debug.py` | [framework/cuda-runtime.md](https://anrans.github.io/ai-infra-handbooks/cuda/framework/cuda-runtime) | 需要 GPU，手册里只做语法检查 |
| `meta_model.py` | [framework/dispatcher.md](https://anrans.github.io/ai-infra-handbooks/cuda/framework/dispatcher) |  |
| `minilang.py` | [framework/compiler-basics.md](https://anrans.github.io/ai-infra-handbooks/cuda/framework/compiler-basics) |  |
| `mma_sync.cu` | [advanced/tensor-core.md](https://anrans.github.io/ai-infra-handbooks/cuda/advanced/tensor-core) |  |
| `nccl_allreduce.cu` | [tools/multi-gpu.md](https://anrans.github.io/ai-infra-handbooks/cuda/tools/multi-gpu) |  |
| `no_break.py` | [framework/compile.md](https://anrans.github.io/ai-infra-handbooks/cuda/framework/compile) |  |
| `nvtx_demo.cu` | [tools/profiling.md](https://anrans.github.io/ai-infra-handbooks/cuda/tools/profiling) |  |
| `occupancy.cu` | [basics/execution.md](https://anrans.github.io/ai-infra-handbooks/cuda/basics/execution) |  |
| `overlap.cu` | [tools/streams.md](https://anrans.github.io/ai-infra-handbooks/cuda/tools/streams) |  |
| `overlap_copy.py` | [framework/cuda-runtime.md](https://anrans.github.io/ai-infra-handbooks/cuda/framework/cuda-runtime) | 需要 GPU，手册里只做语法检查 |
| `paged_decode.cu` | [advanced/attention.md](https://anrans.github.io/ai-infra-handbooks/cuda/advanced/attention) |  |
| `passes.py` | [framework/compiler-basics.md](https://anrans.github.io/ai-infra-handbooks/cuda/framework/compiler-basics) |  |
| `pdl_chain.cu` | [tools/pdl-megakernel.md](https://anrans.github.io/ai-infra-handbooks/cuda/tools/pdl-megakernel) |  |
| `permute_stride.py` | [framework/tensor.md](https://anrans.github.io/ai-infra-handbooks/cuda/framework/tensor) |  |
| `recompile.py` | [framework/compile.md](https://anrans.github.io/ai-infra-handbooks/cuda/framework/compile) |  |
| `reduction.cu` | [kernels/reduction.md](https://anrans.github.io/ai-infra-handbooks/cuda/kernels/reduction) |  |
| `rmsnorm.cu` | [kernels/softmax-norm.md](https://anrans.github.io/ai-infra-handbooks/cuda/kernels/softmax-norm) |  |
| `saved_bytes.py` | [framework/autograd.md](https://anrans.github.io/ai-infra-handbooks/cuda/framework/autograd) |  |
| `scan.cu` | [kernels/scan.md](https://anrans.github.io/ai-infra-handbooks/cuda/kernels/scan) |  |
| `sdpa_ops.py` | [framework/dispatcher.md](https://anrans.github.io/ai-infra-handbooks/cuda/framework/dispatcher) |  |
| `show_graph.py` | [framework/compile.md](https://anrans.github.io/ai-infra-handbooks/cuda/framework/compile) |  |
| `softmax.cu` | [kernels/softmax-norm.md](https://anrans.github.io/ai-infra-handbooks/cuda/kernels/softmax-norm) |  |
| `storage_views.py` | [framework/tensor.md](https://anrans.github.io/ai-infra-handbooks/cuda/framework/tensor) |  |
| `swiglu_function.py` | [framework/autograd.md](https://anrans.github.io/ai-infra-handbooks/cuda/framework/autograd) |  |
| `timing.py` | [framework/cuda-runtime.md](https://anrans.github.io/ai-infra-handbooks/cuda/framework/cuda-runtime) | 需要 GPU，手册里只做语法检查 |
| `tl_flash.py` | [tools/tilelang.md](https://anrans.github.io/ai-infra-handbooks/cuda/tools/tilelang) | 需要 GPU，手册里只做语法检查 |
| `tl_matmul.py` | [tools/tilelang.md](https://anrans.github.io/ai-infra-handbooks/cuda/tools/tilelang) | 需要 GPU，手册里只做语法检查 |
| `tma_copy.cu` | [advanced/async-hopper.md](https://anrans.github.io/ai-infra-handbooks/cuda/advanced/async-hopper) |  |
| `transpose.cu` | [kernels/transpose.md](https://anrans.github.io/ai-infra-handbooks/cuda/kernels/transpose) |  |
| `triton_add.py` | [tools/triton.md](https://anrans.github.io/ai-infra-handbooks/cuda/tools/triton) |  |
| `triton_matmul.py` | [tools/triton.md](https://anrans.github.io/ai-infra-handbooks/cuda/tools/triton) |  |
| `triton_rmsnorm.py` | [tools/triton.md](https://anrans.github.io/ai-infra-handbooks/cuda/tools/triton) |  |
| `triton_softmax.py` | [tools/triton.md](https://anrans.github.io/ai-infra-handbooks/cuda/tools/triton) |  |
| `vector_add.cu` | [basics/first-kernel.md](https://anrans.github.io/ai-infra-handbooks/cuda/basics/first-kernel) |  |
| `wmma_gemm.cu` | [advanced/tensor-core.md](https://anrans.github.io/ai-infra-handbooks/cuda/advanced/tensor-core) |  |
