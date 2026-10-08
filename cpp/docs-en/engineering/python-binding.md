# pybind11 and PyTorch C++ extensions

<p class="lead">An inference framework's body is Python, but more and more of the hot path sinks into C++: prefix tree matching, the scheduler's inner loop, tokenization, custom operators. Two bridges connect the two sides: pybind11 (exposing any C++ class or function to Python) and PyTorch C++ extensions (writing operators that take a <code>torch.Tensor</code> and registering them as <code>torch.ops</code>). This chapter writes a complete example of each and covers what goes wrong most easily at the boundary: ownership, copying data and the GIL.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. Does passing a Python `list[int]` to a C++ function taking `const std::vector<int>&` copy? What about a numpy array?
    2. What Python exception does a C++ `std::invalid_argument` become?
    3. When do you release the GIL in C++? What can you not do once it is released?
    4. Writing an operator that takes a `torch::Tensor`, why check the device, the data type and contiguity first?
    5. What does registering an operator with `TORCH_LIBRARY` gain over exposing a function with pybind11 directly?

??? success "Answers (try it yourself first, then expand)"
    1. A `list[int]` is converted element by element and copied into a new `std::vector`; a numpy array can be accessed in place without a copy when the parameter is a `py::array_t` (as a `std::vector` it is still copied).
    2. `ValueError`. pybind11 translates the standard exceptions into the matching Python ones, so `std::out_of_range` becomes `IndexError` and any other `std::exception` becomes `RuntimeError`.
    3. For long-running pure C++ computation that touches no Python object, or for blocking I/O (`py::gil_scoped_release`), so that other Python threads can run. Once it is released, no Python object may be touched and no Python API called until the GIL is reacquired.
    4. An operator usually implements only particular devices, types and memory layouts: a CPU tensor passed to a CUDA kernel crashes, the wrong type interprets the memory wrongly, and a non-contiguous tensor accessed as if contiguous reads the wrong elements. Check and give a clear error, or call `contiguous()` first.
    5. Registered as `torch.ops` it has a schema, the dispatcher can dispatch it by device, it works with autograd, `torch.compile` (with a fake implementation) and CUDA Graphs, and it is called the same way from Python and TorchScript; a function exposed with pybind11 is a black box to all of these.

## The options {#几种方案}

| Option | Suits |
| --- | --- |
| pybind11 / nanobind | exposing C++ classes and functions whole to Python: a scheduler, a prefix tree, a tokenizer, a communication library's control plane. SGLang's `sgl-kernel` and Mooncake's Python interface both use it |
| a PyTorch C++ / CUDA extension | operators whose inputs and outputs are tensors: fused kernels, quantized GEMM, attention. It can be registered as `torch.ops.xxx` and works with `torch.compile` and CUDA Graphs |
| `ctypes` / `cffi` | calling an existing shared library that exports an `extern "C"` interface, with no binding code to write |

This chapter's examples are really compiled and run with g++ and a Python environment holding a CPU build of PyTorch, pybind11 and ninja.

## pybind11: exposing a C++ class to Python {#pybind11把-c-类暴露给-python}

![Figure: pybind11's boundary - type conversion, passing arrays without copying, releasing the GIL](../assets/figures/pybind-boundary.svg){.aig-svg}

Expose the previous chapter's block allocator plus two functions to Python:

```cpp title="kvpool_py.cpp" project="ext"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

class BlockPool {
 public:
  explicit BlockPool(std::int32_t n) : ref_(n, 0) {
    for (std::int32_t i = n - 1; i >= 0; --i) free_.push_back(i);
  }
  std::optional<std::vector<std::int32_t>> allocate(std::int32_t n) {
    if (n < 0 || n > num_free()) return std::nullopt;
    std::vector<std::int32_t> out;
    for (std::int32_t i = 0; i < n; ++i) {
      out.push_back(free_.back());
      free_.pop_back();
      ref_[out.back()] = 1;
    }
    return out;
  }
  void retain(std::int32_t b) { ++ref_.at(b); }
  void release(std::int32_t b) {
    if (ref_.at(b) == 0) throw std::logic_error("重复释放");
    if (--ref_[b] == 0) free_.push_back(b);
  }
  std::int32_t num_free() const { return static_cast<std::int32_t>(free_.size()); }

 private:
  std::vector<std::int32_t> free_, ref_;
};

// two token sequences' longest common prefix: pure C++ computation, with the GIL released so other Python threads can keep running
std::size_t common_prefix(const std::vector<std::int32_t>& a, const std::vector<std::int32_t>& b) {
  py::gil_scoped_release release;   // once released, no Python object may be touched
  std::size_t n = std::min(a.size(), b.size()), i = 0;
  while (i < n && a[i] == b[i]) ++i;
  return i;
}

// read the numpy array's memory directly without copying (when the dtype is not int32 or it is not contiguous, forcecast converts a copy first)
std::int64_t count_eos(py::array_t<std::int32_t, py::array::c_style | py::array::forcecast> tokens, std::int32_t eos) {
  auto r = tokens.unchecked<1>();   // a one-dimensional accessor without bounds checking
  std::int64_t n = 0;
  for (py::ssize_t i = 0; i < r.shape(0); ++i) n += r(i) == eos;
  return n;
}

PYBIND11_MODULE(kvpool_py, m) {
  m.doc() = "块分配器的 Python 绑定";
  py::class_<BlockPool>(m, "BlockPool")
      .def(py::init<std::int32_t>(), py::arg("num_blocks"))
      .def("allocate", &BlockPool::allocate, py::arg("n"))   // optional<vector<int>> becomes list[int] | None
      .def("retain", &BlockPool::retain)
      .def("release", &BlockPool::release)
      .def_property_readonly("num_free", &BlockPool::num_free)
      .def("__repr__", [](const BlockPool& p) { return "<BlockPool free=" + std::to_string(p.num_free()) + ">"; });
  m.def("common_prefix", &common_prefix, py::arg("a"), py::arg("b"));
  m.def("count_eos", &count_eos, py::arg("tokens"), py::arg("eos"));
}
```

```python title="test_kvpool.py" project="ext"
import numpy as np

import kvpool_py as kp

pool = kp.BlockPool(8)
blocks = pool.allocate(3)
print("分配：", blocks, pool)
print("不够时：", pool.allocate(10))
for b in blocks:
    pool.release(b)
try:
    pool.release(blocks[0])
except RuntimeError as e:                 # std::logic_error is translated into RuntimeError
    print("重复释放：", type(e).__name__, e)
print("公共前缀：", kp.common_prefix([1, 2, 3, 4], [1, 2, 9]))
print("EOS 个数：", kp.count_eos(np.array([5, 2, 7, 2, 2], dtype=np.int32), eos=2))
```

Compile it into a Python extension module (whose file name carries the current Python's extension suffix, `kvpool_py.cpython-312-x86_64-linux-gnu.so` say), then `import` it directly:

```bash title="build_pybind.sh" project="ext" run="yes"
PY="${PYTHON:-python3}"
EXT=$("$PY" -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
g++ -O2 -std=c++20 -shared -fPIC $("$PY" -m pybind11 --includes) kvpool_py.cpp -o "kvpool_py$EXT"
"$PY" test_kvpool.py
```

```text title="output"
分配： [0, 1, 2] <BlockPool free=5>
不够时： None
重复释放： RuntimeError 重复释放
公共前缀： 2
EOS 个数： 3
```

A proper project builds it with `pyproject.toml` + scikit-build-core (CMake underneath) or setuptools' `Pybind11Extension`, compiling automatically on `pip install .`.

### Three things at the boundary {#边界上的三件事}

**Type conversion copies.** With `#include <pybind11/stl.h>`, `std::vector`, `std::map` and `std::optional` convert to and from Python's `list`, `dict` and `None` automatically, and every call is a full copy. For a large array use `py::array_t` (numpy) or `py::buffer` and read the Python object's memory directly; when returning a large array, let the numpy array own the memory C++ allocated (with a `py::capsule` to free it) rather than copying again.

**Exceptions are translated.** A C++ exception crossing the boundary becomes a Python one: `std::invalid_argument` to `ValueError`, `std::out_of_range` to `IndexError`, `std::bad_alloc` to `MemoryError`, any other `std::exception` to `RuntimeError`. The other way round, an exception thrown in a Python callback is a `py::error_already_set` on the C++ side. **Never let an exception cross an `extern "C"` boundary** or escape a destructor.

**Ownership has to be stated.** When returning a pointer or a reference, pybind11 has to know whether the Python object owns it (`py::return_value_policy`). The safest approach is to return by value (copied or moved out) or to return a `std::shared_ptr` / `std::unique_ptr`, handing ownership to Python explicitly; when a reference to an internal object has to be returned, use `py::return_value_policy::reference_internal` so the parent outlives the reference.

### The GIL {#gil}

The Python interpreter lets only one thread execute Python code at a time (the GIL). C++ code called from Python **holds the GIL**:

- long-running pure C++ computation (prefix matching, tokenization, waiting on the network or the GPU) should release it with `py::gil_scoped_release`, or every Python thread is stuck; once released, **no** Python object may be created, read or destroyed;
- a C++ background thread calling back into Python takes the GIL first with `py::gil_scoped_acquire`;
- **the classic deadlock**: thread A holds the GIL and goes for the C++ lock `m`; thread B holds `m` and goes for the GIL (to call back into Python, say). Each waits for the other. The rule is: do not take the GIL while holding a C++ lock, and release the GIL before taking a C++ lock.

## A PyTorch C++ extension: writing an operator {#pytorch-c-扩展写一个算子}

When the inputs and outputs are tensors, use PyTorch's extension mechanism: `torch/extension.h` provides `at::Tensor` (`torch::Tensor`), the error-checking macros and the CPU thread pool, and `torch.utils.cpp_extension` does the compiling. Below is a CPU RMSNorm:

```cpp title="rmsnorm_ext.cpp" project="ext"
#include <torch/extension.h>

#include <cmath>

// y = x / sqrt(mean(x^2) + eps) * weight, normalized along the last dimension
torch::Tensor rmsnorm(torch::Tensor x, torch::Tensor weight, double eps) {
  TORCH_CHECK(x.device().is_cpu() && weight.device().is_cpu(), "这个示例只实现了 CPU 版本");
  TORCH_CHECK(x.scalar_type() == torch::kFloat32 && weight.scalar_type() == torch::kFloat32, "只支持 float32");
  TORCH_CHECK(x.size(-1) == weight.size(0), "weight 的长度必须等于 x 的最后一维");

  auto xc = x.contiguous();      // make the memory contiguous before walking it through a pointer
  auto wc = weight.contiguous(); // it has to go in a variable: a temporary tensor is released at the end of the statement
  auto y = torch::empty_like(xc);
  const int64_t d = xc.size(-1), rows = xc.numel() / d;
  const float* px = xc.data_ptr<float>();
  const float* pw = wc.data_ptr<float>();
  float* py = y.data_ptr<float>();

  at::parallel_for(0, rows, 16, [&](int64_t begin, int64_t end) {   // use PyTorch's own CPU thread pool
    for (int64_t r = begin; r < end; ++r) {
      const float* row = px + r * d;
      double ss = 0;
      for (int64_t i = 0; i < d; ++i) ss += double(row[i]) * row[i];
      const float inv = static_cast<float>(1.0 / std::sqrt(ss / d + eps));
      for (int64_t i = 0; i < d; ++i) py[r * d + i] = row[i] * inv * pw[i];
    }
  });
  return y;
}

// registered as torch.ops.demo.rmsnorm: with an explicit schema that torch.compile and CUDA Graphs both recognize
TORCH_LIBRARY(demo, m) { m.def("rmsnorm(Tensor x, Tensor weight, float eps) -> Tensor"); }
TORCH_LIBRARY_IMPL(demo, CPU, m) { m.impl("rmsnorm", &rmsnorm); }

// also expose an ordinary function through pybind11
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("rmsnorm", &rmsnorm, "RMSNorm（CPU）"); }
```

```python title="run_torch_ext.py" project="ext"
import torch
from torch.utils.cpp_extension import load

ext = load(name="rmsnorm_ext", sources=["rmsnorm_ext.cpp"], extra_cflags=["-O2"], verbose=False)


def reference(x, w, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


torch.manual_seed(0)
x = torch.randn(4, 7, 4096)
w = torch.rand(4096)
torch.testing.assert_close(ext.rmsnorm(x, w, 1e-6), reference(x, w), rtol=1e-5, atol=1e-5)
print("pybind11 接口与参考实现一致")
torch.testing.assert_close(torch.ops.demo.rmsnorm(x, w, 1e-6), reference(x, w), rtol=1e-5, atol=1e-5)
print("torch.ops.demo.rmsnorm 与参考实现一致")
xt = x.transpose(0, 1)   # a non-contiguous input
torch.testing.assert_close(ext.rmsnorm(xt, w, 1e-6), reference(xt, w), rtol=1e-5, atol=1e-5)
print("非连续的输入也正确")
try:
    ext.rmsnorm(x.double(), w, 1e-6)
except RuntimeError as e:
    print("类型检查：", str(e).splitlines()[0])
```

```bash title="build_torch_ext.sh" project="ext" run="yes"
TORCH_EXTENSIONS_DIR="$PWD/torch-ext" "${PYTHON:-python3}" run_torch_ext.py
```

```text title="output"
pybind11 接口与参考实现一致
torch.ops.demo.rmsnorm 与参考实现一致
非连续的输入也正确
类型检查： 只支持 float32
```

The checklist for writing a tensor operator:

- **check the device, the data type and the shape**, with `TORCH_CHECK` for a clear error message. Not checking means reading the wrong memory and computing a silently wrong result;
- **handle non-contiguous inputs**: a tensor after a `transpose` or a slice is not contiguous in memory, so call `.contiguous()` before walking it linearly through `data_ptr` (or access it by `stride`);
- **a temporary tensor's lifetime**: `weight.contiguous().data_ptr<float>()` happens to work when `weight` was contiguous already, and when it was not, the temporary `contiguous()` returned is released at the end of that line and the pointer dangles, the same problem as the temporaries in [move semantics](../basics/move.md) and [value semantics](../basics/value-raii.md);
- **compare against a reference implementation**: `torch.testing.assert_close` gives a sensible tolerance per data type;
- **register it as `torch.ops`**: `TORCH_LIBRARY` gives the operator a schema (the argument and return types, and which arguments are modified in place), which is what lets `torch.compile` treat it as an opaque but traceable node rather than breaking the graph there; an implementation for the `Meta` device that only computes the output's shape can be registered as well, for shape inference at compile time.

A CUDA version is written the same way: another `.cu` file holding the kernel and the launch code, registered in `TORCH_LIBRARY_IMPL(demo, CUDA, m)`, launching the kernel on PyTorch's current stream with `at::cuda::getCurrentCUDAStream()` (see the CUDA handbook's [the ecosystem: cuBLAS, CUTLASS and PyTorch](cuda://tools/ecosystem/)).

!!! interview "How to explain it"
    On bindings: pybind11 converting a `list[int]` into a `std::vector<int>` copies, and a large array goes through `py::array_t` reading numpy's memory directly; C++ exceptions are translated into Python ones automatically (`std::invalid_argument` becoming `ValueError`); long-running pure C++ computation releases the GIL (`py::gil_scoped_release`), after which no Python object may be touched, and the GIL is never taken while holding a C++ lock. A PyTorch operator checks the device, the data type, the shape and contiguity first (with the temporary a `contiguous()` returns kept alive until the computation ends) and is compared against a reference implementation; registering it as `torch.ops` with `TORCH_LIBRARY` is what lets `torch.compile` and CUDA Graphs handle it correctly (through its schema and a fake implementation), while a function exposed through pybind11 is a black box to them.

## Exercises {#练习}

1. Give `kvpool_py` an `allocate_many(counts)`: the argument is a numpy `int32` array (the blocks each request needs), committing only if every allocation succeeds and allocating none otherwise, returning `list[list[int]] | None`. Does it release the GIL for any of this? Why?

??? success "Answer"
    ```cpp
    std::optional<std::vector<std::vector<std::int32_t>>> allocate_many(
        BlockPool& pool, py::array_t<std::int32_t, py::array::c_style | py::array::forcecast> counts) {
      auto c = counts.unchecked<1>();
      std::int64_t total = 0;
      for (py::ssize_t i = 0; i < c.shape(0); ++i) total += c(i);
      if (total > pool.num_free()) return std::nullopt;       // check the total first: either all of them or nothing
      std::vector<std::vector<std::int32_t>> out;
      for (py::ssize_t i = 0; i < c.shape(0); ++i) out.push_back(*pool.allocate(c(i)));
      return out;
    }
    // the binding: .def("allocate_many", &allocate_many)
    ```

    It does **not release the GIL** here: `counts` is a Python object and reading its memory requires holding the GIL (unless the data is copied into a C++ `vector` first); and `BlockPool` has no lock, its thread safety resting precisely on "only the thread holding the GIL may call it". A C++ object with no internal lock, once touched inside a GIL-released region, can be modified by several Python threads at once.

2. A user reports that calling `ext.rmsnorm` (the version exposed through pybind11) inside a model compiled by `torch.compile` makes the compiler report a graph break, while `torch.ops.demo.rmsnorm` does not. Why?

??? success "Answer"
    To `torch.compile` (Dynamo), a function exposed through pybind11 is an arbitrary Python callable it cannot trace: it does not know which tensors the function reads and writes or what shape the output has, so it breaks the graph there and falls back to eager execution.
    `torch.ops.demo.rmsnorm` has a registered schema, so Dynamo knows its input and output types and whether it modifies in place; register a shape inference for `Meta` (or with `register_fake`) and the compiler can treat it as a node in the graph, with the operators before and after it still fused. That is why inference frameworks register their custom operators as `torch.ops`.

## Summary {#小结}

- [x] pybind11 exposes C++ classes and functions: STL containers convert automatically (with a copy), and a large array goes through `py::array_t` reading the memory directly; exceptions are translated into Python exceptions automatically.
- [x] Long-running pure C++ computation releases the GIL and touches no Python object afterwards; do not take the GIL while holding a C++ lock.
- [x] A tensor operator: check the device, the type and the shape, handle non-contiguous inputs, watch the lifetime of the temporary a `contiguous()` returns, and compare against a reference implementation.
- [x] Registering it as `torch.ops` with `TORCH_LIBRARY` is what makes it work well with `torch.compile` and CUDA Graphs.
