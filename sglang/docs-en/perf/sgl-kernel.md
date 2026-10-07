# sgl-kernel: where writing its own operators began

<p class="lead">On 30 November 2024 a nearly empty directory <code>sgl-kernel/</code> appeared in the repository; the next day it had a PyPI package and a 97-line warp-reduction example. A year and a half later it has over two hundred files covering kernels for all-reduce, attention, GEMM, MoE, quantization, sampling and speculative decoding, and wraps CUTLASS, FlashInfer, DeepGEMM, FlashMLA and others as one set of PyTorch operators; in July 2026 it moved into <code>python/sglang/kernels/</code>. This chapter covers why it appeared (removing the vLLM dependency, several kinds of hardware, kernels tailored to its own scheduling), how it is organised (TORCH_LIBRARY registration, directories by operator type, its own releases) and where "writing your own kernels" ends in this project.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. Where did SGLang's custom kernels come from before sgl-kernel? What moved in first afterwards?
    2. Why change from `PYBIND11_MODULE` to `TORCH_LIBRARY` in January 2025?
    3. Why does sgl-kernel have its own releases (its own PyPI package and version number)? What trouble does that bring?
    4. What is its relationship to FlashInfer, CUTLASS and DeepGEMM: replacement or wrapper?

??? success "Answers for the self-test (answer first, then open this)"
    1. Before: Triton kernels (written in `layers/`), FlashInfer (attention, sampling) and vLLM's `_custom_ops` (the CUDA operators for RMSNorm, the activations, quantization, all-reduce and so on). What moved in first were the FP8 quantization operators (#2370 "Move FP8 to SGLang", 2024-12-06), the custom all-reduce, and the MoE's alignment and top-k — all of them what used to depend on vLLM.
    2. `TORCH_LIBRARY` registers a kernel as a PyTorch operator (with a schema, traceable by `torch.compile`, capturable by a CUDA graph, and with a meta implementation for shape inference), while `PYBIND11_MODULE` only exposes a C++ function to Python, where the compiler cannot see it.
    3. Compiling the kernels takes tens of minutes to hours and depends on particular CUDA and torch versions, so as a separate package a user gets a prebuilt wheel with `pip install sgl-kernel` and the main package's releases are decoupled from the kernels'. The trouble is the version matrix: each version of the main package requires a particular range of sgl-kernel, upgrades have to be synchronised on both sides, and CI has to test several combinations.
    4. Wrapping mostly, replacing occasionally: CUTLASS's GEMMs, some of FlashInfer's kernels, DeepGEMM's grouped GEMM and FlashMLA's MLA decode are all exposed through sgl-kernel as uniform PyTorch operators; what is written in-house is mainly glue, fused operators (quantization plus a transpose, the MoE's alignment and gating) and the parts the external libraries do not cover (a lightweight attention decode, for instance).

A six-panel strip first, then the chapter:

<!-- comic ../assets/comics/sgl-kernel.webp is in Chinese; put it back once the English version exists -->

## Starting from a warp reduction {#从一个-warp-归约开始}

```bash title="sgl-kernel-commits.sh"
for h in 419a57e771 5c91a315d7 47eb139f81 84d96b3ae5 9286740eff 6b45a21d16 110e006673 c553e1604c 54b9a2de0a c32c4ef79c; do
  git log -1 --date=short --format='%ad  %h  %s' $h | cut -c1-96
done
```

```text title="output"
2024-11-30  419a57e771  minor: add sgl-kernel dir (#2261)
2024-12-01  5c91a315d7  feat: support sgl-kernel pypi (#2302)
2024-12-01  47eb139f81  feat: use warp reduce as a simple example (#2304)
2024-12-06  84d96b3ae5  Move FP8 to SGLang (#2370)
2025-01-26  9286740eff  feat: refactor sgl-kernel and use TORCH_LIBRARY instead of PYBIND11_MODU
2025-03-03  6b45a21d16  Reorganize c++ source files in sgl-kernel with multiple folders  (#4025)
2025-03-03  110e006673  Reorganize python source files in sgl-kernel with multiple files  (#4027
2025-03-10  c553e1604c  DeepGemm integrate to sgl-kernel (#4165)
2025-03-29  54b9a2de0a  remove setup for sgl-kernel (#4899)
2026-07-29  c32c4ef79c  [Kernel] Move sgl-kernel under sglang.kernels.aot (#32648)
```

#2261 "add sgl-kernel dir" on 30 November, #2302 "support sgl-kernel pypi" on 1 December, and #2304 the same day putting in the first kernel — a warp reduction as a teaching example:

```cuda title="sgl-kernel/src/sgl-kernel/csrc/warp_reduce_kernel.cu @ 47eb139f81 L1-39" linenums="1"
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#define FINAL_MASK 0xffffffff
#define BLOCK_SIZE 256

template <typename scalar_t>
__device__ __forceinline__ scalar_t add(scalar_t a, scalar_t b) {
  return a + b;
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t warpReduceSum(scalar_t val) {
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    val += __shfl_down_sync(FINAL_MASK, val, offset);
  }
  return val;
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t blockReduceSum(scalar_t val) {
  __shared__ scalar_t shared[32];
  int lane = threadIdx.x % 32;
  int wid = threadIdx.x / 32;

  val = warpReduceSum(val); // First reduce within warp

  if (lane == 0)
    shared[wid] = val; // Write reduced value to shared memory

  __syncthreads(); // Wait for all partial reductions

  // Read from shared memory only if that warp existed
  val = (threadIdx.x < (blockDim.x / 32)) ? shared[lane] : 0;

  if (wid == 0)
    val = warpReduceSum(val); // Final reduce within first warp
```

The example matters not for the kernel but for the skeleton it establishes: `csrc/` for the `.cu` files, `ops/__init__.py` for the Python-side wrappers, a `setup.py` compiling with `torch.utils.cpp_extension`, and a release to PyPI. Five days later #2370 "Move FP8 to SGLang" brought in the first real operators — exactly where [chapter eight](../service/borrow-vllm.md)'s removal of the vLLM dependency lands at the kernel level.

## Two reorganisations {#两次重组}

#3130 of 26 January 2025 changed the registration from `PYBIND11_MODULE` to `TORCH_LIBRARY`:

```cpp title="sgl-kernel/csrc/common_extension.cc @ v0.4.6 L1-40" linenums="1"
/* Copyright 2025 SGLang Team. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/all.h>
#include <torch/library.h>

#include "sgl_kernel_ops.h"

TORCH_LIBRARY_FRAGMENT(sgl_kernel, m) {
  /*
   * From csrc/allreduce
   */

  m.def("get_graph_buffer_ipc_meta", &get_graph_buffer_ipc_meta);
  m.def("register_graph_buffers", &register_graph_buffers);
  m.def("dispose", &dispose);
  m.def("meta_size", &meta_size);
  m.def("register_buffer", &register_buffer);

  m.def(
      "init_custom_ar(int[] ipc_tensors, Tensor rank_data, "
      "int rank, bool full_nvlink) -> int");
  m.impl("init_custom_ar", torch::kCUDA, &init_custom_ar);

  m.def(
      "all_reduce(int fa, Tensor inp, Tensor! out, int reg_buffer, "
      "int reg_buffer_sz_bytes) -> ()");
  m.impl("all_reduce", torch::kCUDA, &all_reduce);
```

Each operator has a schema (the parameter and return types) and an implementation dispatched by device (`m.impl(..., torch::kCUDA, ...)`), so `torch.compile` can trace it, a CUDA graph can capture it, and shape inference can rest on a meta implementation. #4025 and #4027 of 3 March split the C++ sources into directories by operator type and the Python wrappers into files by function; #4165 of 10 March integrated DeepGEMM; and #4899 of 29 March removed `setup.py` and moved entirely to a CMake build. The directory at v0.4.6:

```bash title="sgl-kernel-dirs.sh"
REF=${REF:-29f6d408c0}
for t in v0.4.6 v0.5.0rc0; do
  echo "== $t：$(git ls-tree -r --name-only $t -- sgl-kernel | wc -l) 个文件，.cu $(git ls-tree -r --name-only $t -- sgl-kernel | grep -c '\.cu$') 个"
  git ls-tree -r --name-only $t -- sgl-kernel/csrc | awk -F/ 'NF > 3 {print $3}' | sort | uniq -c | awk '{printf "   %3d %s\n", $1, $2}'
done
echo "== $REF：python/sglang/kernels/ 有 $(git ls-tree -r --name-only "$REF" -- python/sglang/kernels | wc -l) 个文件，.cu $(git ls-tree -r --name-only "$REF" -- python/sglang/kernels | grep -c '\.cu$') 个"
```

```text title="output"
== v0.4.6：128 个文件，.cu 29 个
     4 allreduce
     4 attention
    19 cpu
     3 cutlass_extensions
     3 elementwise
    12 gemm
     1 grammar
     5 moe
     5 speculative
== v0.5.0rc0：221 个文件，.cu 58 个
    11 allreduce
     9 attention
    23 cpu
     8 cutlass_extensions
     3 elementwise
    19 gemm
     1 grammar
     1 kvcacheio
    29 moe
     3 spatial
     5 speculative
== 29f6d408c0：python/sglang/kernels/ 有 1426 个文件，.cu 55 个
```

`allreduce/`, `attention/`, `elementwise/`, `gemm/`, `moe/`, `speculative/`, `cpu/` — the directory names are the kinds of operator SGLang needs. `cpu/` is the CPU backend added in early 2025 (contributed by Intel), which says sgl-kernel was never only for NVIDIA.

![Figure: where sgl-kernel sits in the stack](../assets/figures/sgl-kernel-layers.svg){.aig-svg}

## Why write them {#为什么自己写}

All three motives can be found in the commit history:

1. **Removing the dependency.** Of [chapter eight](../service/borrow-vllm.md)'s 26 commits, the 2025-03 wave (the MoE alignment #4164, the custom all-reduce #4210, the FP8 quantization #4215, the quantization module #4507) all lands in sgl-kernel. Without its own kernel package, removing the vLLM dependency is not even a possibility.
2. **Tailoring to its own scheduling.** The per-token quantization kernel that "accelerates per token quant by 20-28%" (#4215), fusing quantization with a transpose, the MoE's `moe_align_block_size` and `fused_gate` — what these fused operators gain depends on how the caller organises its data, and a general library will not do it for one engine.
3. **Wrapping external libraries.** CUTLASS's FP8 and INT8 GEMMs, DeepGEMM's grouped GEMM, FlashMLA and later some of FlashInfer's kernels are all unified into PyTorch operators through sgl-kernel, so the caller need not care which library is behind. #4449 of 16 March added FlashMLA as a submodule and was reverted the same day (#4470), to be connected a few days later another way — even how a third-party library is brought in was a matter of trial and error.

The boundary is clear too: attention's main force is still FlashInfer and FlashAttention ([chapter 19](../scale/attention-backends.md)) and sgl-kernel does not rewrite them; the Triton kernels stay on the Python side (`layers/attention/triton_ops/`, `fused_moe_triton/`) because they are quick to change. What sgl-kernel takes in is the part that has to be written in C++ or CUDA and released stably.

## What its own releases cost {#独立发版的代价}

sgl-kernel was a separate PyPI package from day one. The benefit is that a user installs a prebuilt wheel without waiting tens of minutes for a compile; the drawback is the version matrix: each version of the main package requires a particular range of sgl-kernel, and CUDA 12.x and 13.x, several torch versions, x86 and aarch64, and CUDA and ROCm each need their own wheel. Its release frequency:

```bash title="sgl-kernel-releases.sh"
REF=${REF:-29f6d408c0}
echo "2025 年里标题含 sgl-kernel 且含 bump/release/version 的提交：$(git log --date=short --format='%ad %s' "$REF" | grep '^2025' | grep -i 'sgl-kernel' | grep -icE 'bump|release|version')"
echo "sgl-kernel 相关提交总数（标题含 sgl-kernel）：$(git log --format=%s "$REF" | grep -ic 'sgl-kernel')"
```

```text title="output"
2025 年里标题含 sgl-kernel 且含 bump/release/version 的提交：112
sgl-kernel 相关提交总数（标题含 sgl-kernel）：491
```

#32648 "Move sgl-kernel under sglang.kernels.aot" of 29 July 2026 moved it into the main package's `python/sglang/kernels/` (`aot` meaning the ahead-of-time compiled part, as against just-in-time kernels like Triton's) — the top-level `sgl-kernel/` directory no longer exists at the baseline commit. That step narrowed the "one repository, two packages" version-matrix problem somewhat, though the prebuilt wheels are still released separately.

## Design trade-offs {#设计取舍}

| | Keep depending on vLLM's and FlashInfer's operators | Its own kernel package |
| --- | --- | --- |
| Development | free | C++ / CUDA, CMake and multi-platform wheels to maintain |
| Upgrade pace | bound by upstream | its own |
| Customisation | impossible | fused operators shaped to its own data layout |
| Several kinds of hardware | wait for upstream | ROCm and CPU connected itself |
| Installation | one package | two packages whose versions have to match |

SGLang's choice is "its own wrapper layer plus external libraries as the main kernels plus its own fusions and glue", different from vLLM putting `csrc/` in the main package to compile, and different again from FlashInfer as a pure kernel library.

## What happened afterwards {#后来怎么样了}

- 2025: FP4 and FP8 GEMMs for Blackwell (SM100), MXFP4, cutlass MLA, speculative decoding's kernels (`speculative/`), hip implementations for ROCm, and a `cpu/` backend that kept growing.
- The "[sgl-kernel] 1/N Refactor sglang cutlass 3x" series of 2025-08 reworked the CUTLASS wrappers.
- It moved into the main package's `python/sglang/kernels/` in 2026-07, with over 1400 files at the baseline commit (including CUTLASS and other third-party headers).

## Exercises {#练习}

**1. The first operator.** Read #2304's `warp_reduce_kernel.cu`, say which CUDA primitives it uses (`__shfl_down_sync` and so on), and why a reduction was chosen as the example.

??? success "A way to approach it"
    A reduction within a warp uses the shuffle instructions and is the smallest kernel you have to understand the hardware to write correctly, which makes it a good template; the CUDA handbook's [reduction chapter](cuda://kernels/reduction/) covers the same technique.

**2. The difference the registration makes.** Find an operator registered with `TORCH_LIBRARY` at v0.4.6, then find its Python wrapper, and describe the call path: Python → `torch.ops.sgl_kernel.xxx` → C++.

??? success "A way to approach it"
    `m.def("...")` and `m.impl(...)` in `common_extension.cc`, with the Python side calling `torch.ops.sgl_kernel.<name>` from files like `sgl_kernel/elementwise.py`; under `torch.compile` the operator appears as a single node.

**3. The version matrix.** Find the sgl-kernel version requirement in the baseline commit's `python/pyproject.toml`, then find the kernel package's own version under `python/sglang/kernels/`, and explain how the two correspond.

??? success "A way to approach it"
    `git show 29f6d408c0:python/pyproject.toml | grep -i kernel`; the main package declares `sgl-kernel==x.y.z` or a range, and the kernel package's version is in its own `pyproject.toml` or `version.py`; each has its own workflow in the release process.

!!! interview "How to answer in an interview"
    "Should an inference engine write its own kernels?" — Answer with sgl-kernel's history: leave the main force in attention and GEMM to dedicated libraries (FlashInfer, CUTLASS, DeepGEMM), write the fused operators and the glue yourself (quantization plus a transpose, the MoE alignment, sampling), and register them as PyTorch operators with `TORCH_LIBRARY` to work with `torch.compile` and CUDA graphs; separate releases buy an easy install at the price of a version matrix. That answer shows you know where "writing kernels" ends within a system.

## Summary {#小结}

- [x] The directory was created on 2024-11-30, the PyPI package and the warp-reduction example on 12-01, and the FP8 operators moved in on 12-06: the kernel-level landing point of removing the vLLM dependency.
- [x] `TORCH_LIBRARY` registration in 2025-01, directories by operator type plus DeepGEMM and a CMake build in March; the directory names are the kinds of operator SGLang needs.
- [x] Its place: wrap external libraries plus write the fusions and the glue; its own releases cost a version matrix, and it moved into the main package's `kernels/` in 2026-07.
