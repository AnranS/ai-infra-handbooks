# pybind11 与 PyTorch C++ 扩展

<p class="lead">推理框架的主体是 Python，但热路径越来越多地下沉到 C++：前缀树的匹配、调度器的内层循环、分词、自定义算子。连接两边的桥梁主要有两座：pybind11（把任意 C++ 类和函数暴露给 Python）和 PyTorch C++ 扩展（写接收 <code>torch.Tensor</code> 的算子，并注册成 <code>torch.ops</code>）。这一章各写一个完整的例子，并讲清楚跨语言边界上最容易出错的地方：所有权、数据拷贝和 GIL。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. Python 的 `list[int]` 传给一个参数是 `const std::vector<int>&` 的 C++ 函数，会发生拷贝吗？numpy 数组呢？
    2. C++ 抛出的 `std::invalid_argument` 在 Python 里变成什么异常？
    3. 什么时候要在 C++ 里释放 GIL？释放之后不能做什么？
    4. 写一个接收 `torch::Tensor` 的算子，为什么要先检查设备、数据类型和连续性？
    5. 用 `TORCH_LIBRARY` 注册算子，比直接用 pybind11 暴露函数多了什么好处？

??? success "自测参考答案（先自己答，再展开对照）"
    1. `list[int]` 会被逐个转换、拷贝成一个新的 `std::vector`；numpy 数组如果参数写成 `py::array_t` 就能直接访问它的内存，不拷贝（写成 `std::vector` 仍然会拷贝）。
    2. `ValueError`。pybind11 会把标准异常翻译成对应的 Python 异常，比如 `std::out_of_range` → `IndexError`，其他 `std::exception` → `RuntimeError`。
    3. 长时间运行、不接触 Python 对象的纯 C++ 计算或者阻塞的 I/O（`py::gil_scoped_release`），让其他 Python 线程可以运行。释放之后不能访问任何 Python 对象、也不能调用 Python API，直到重新获得 GIL。
    4. 算子通常只实现了特定的设备、类型和内存布局：CPU 张量传给 CUDA kernel 会崩溃，类型不对会按错误的方式解释内存，非连续的张量按连续的方式访问会读到错误的元素。检查之后给出清楚的报错，或者先 `contiguous()`。
    5. 注册成 `torch.ops` 后有 schema，dispatcher 能按设备分发，能和 autograd、`torch.compile`（需要 fake 实现）、CUDA Graph 配合，也可以在 Python 和 TorchScript 里统一调用；pybind11 暴露的函数对这些机制是黑盒。

## 几种方案

| 方案 | 适合 |
| --- | --- |
| pybind11 / nanobind | 把 C++ 类和函数完整地暴露给 Python：调度器、前缀树、分词器、通信库的控制面。SGLang 的 `sgl-kernel`、Mooncake 的 Python 接口都用它 |
| PyTorch C++ / CUDA 扩展 | 输入输出都是张量的算子：融合 kernel、量化 GEMM、注意力。能注册成 `torch.ops.xxx`，和 `torch.compile`、CUDA Graph 配合 |
| `ctypes` / `cffi` | 调用一个已有的、导出了 `extern "C"` 接口的动态库，不想写绑定代码 |

本章的例子用 g++ 和一个装了 CPU 版 PyTorch、pybind11 和 ninja 的 Python 环境实际编译运行。

## pybind11：把 C++ 类暴露给 Python

把上一章的块分配器、再加两个函数暴露给 Python：

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

// 两个 token 序列的最长公共前缀：纯 C++ 计算，期间释放 GIL，别的 Python 线程可以继续运行
std::size_t common_prefix(const std::vector<std::int32_t>& a, const std::vector<std::int32_t>& b) {
  py::gil_scoped_release release;   // 释放之后，不能再访问任何 Python 对象
  std::size_t n = std::min(a.size(), b.size()), i = 0;
  while (i < n && a[i] == b[i]) ++i;
  return i;
}

// 直接读 numpy 数组的内存，不拷贝（dtype 不是 int32 或不连续时，forcecast 会先转换出一份）
std::int64_t count_eos(py::array_t<std::int32_t, py::array::c_style | py::array::forcecast> tokens, std::int32_t eos) {
  auto r = tokens.unchecked<1>();   // 一维、不做边界检查的访问器
  std::int64_t n = 0;
  for (py::ssize_t i = 0; i < r.shape(0); ++i) n += r(i) == eos;
  return n;
}

PYBIND11_MODULE(kvpool_py, m) {
  m.doc() = "块分配器的 Python 绑定";
  py::class_<BlockPool>(m, "BlockPool")
      .def(py::init<std::int32_t>(), py::arg("num_blocks"))
      .def("allocate", &BlockPool::allocate, py::arg("n"))   // optional<vector<int>> 变成 list[int] | None
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
except RuntimeError as e:                 # std::logic_error 被翻译成 RuntimeError
    print("重复释放：", type(e).__name__, e)
print("公共前缀：", kp.common_prefix([1, 2, 3, 4], [1, 2, 9]))
print("EOS 个数：", kp.count_eos(np.array([5, 2, 7, 2, 2], dtype=np.int32), eos=2))
```

编译成一个 Python 扩展模块（文件名带上当前 Python 的扩展后缀，比如 `kvpool_py.cpython-312-x86_64-linux-gnu.so`），然后直接 `import`：

```bash title="build_pybind.sh" project="ext" run="yes"
PY="${PYTHON:-python3}"
EXT=$("$PY" -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
g++ -O2 -std=c++20 -shared -fPIC $("$PY" -m pybind11 --includes) kvpool_py.cpp -o "kvpool_py$EXT"
"$PY" test_kvpool.py
```

```text title="输出"
分配： [0, 1, 2] <BlockPool free=5>
不够时： None
重复释放： RuntimeError 重复释放
公共前缀： 2
EOS 个数： 3
```

正式的项目用 `pyproject.toml` + scikit-build-core（背后是 CMake）或 setuptools 的 `Pybind11Extension` 来构建，`pip install .` 时自动编译。

### 边界上的三件事

**类型转换会拷贝。**`#include <pybind11/stl.h>` 之后，`std::vector`、`std::map`、`std::optional` 和 Python 的 `list`、`dict`、`None` 自动互相转换——每次调用都是一次完整的拷贝。传大数组时用 `py::array_t`（numpy）或 `py::buffer`，直接读 Python 对象的内存；返回大数组时，让 numpy 数组持有 C++ 分配的内存（`py::capsule` 负责释放），避免再拷贝一次。

**异常会被翻译。**C++ 异常穿过边界时变成 Python 异常：`std::invalid_argument` → `ValueError`，`std::out_of_range` → `IndexError`，`std::bad_alloc` → `MemoryError`，其他 `std::exception` → `RuntimeError`。反过来，Python 回调里抛出的异常在 C++ 这边是 `py::error_already_set`。**绝对不要让异常穿过 `extern "C"` 的边界**或者从析构函数里抛出。

**所有权要说清楚。**返回指针或引用时，pybind11 需要知道 Python 对象是否拥有它（`py::return_value_policy`）。最安全的做法是：返回值类型（拷贝或移动出去），或者返回 `std::shared_ptr` / `std::unique_ptr`，让所有权明确地交给 Python；需要返回内部对象的引用时，用 `py::return_value_policy::reference_internal`，让父对象活得比返回的引用更久。

### GIL

Python 解释器同一时刻只允许一个线程执行 Python 代码（GIL）。C++ 代码被 Python 调用时**持有 GIL**：

- 长时间的纯 C++ 计算（前缀匹配、分词、等待网络或 GPU）应该用 `py::gil_scoped_release` 释放 GIL，否则所有 Python 线程都被卡住；释放之后**不能**再创建、读取或销毁任何 Python 对象；
- C++ 的后台线程要回调 Python 时，先用 `py::gil_scoped_acquire` 拿到 GIL；
- **死锁的经典形态**：线程 A 持有 GIL，去拿 C++ 的锁 `m`；线程 B 持有 `m`，去拿 GIL（比如要回调 Python）。两边互相等待。规则是：持有 C++ 锁的时候不要去拿 GIL；拿 C++ 锁之前先释放 GIL。

## PyTorch C++ 扩展：写一个算子

输入输出都是张量时，用 PyTorch 的扩展机制：`torch/extension.h` 里有 `at::Tensor`（`torch::Tensor`）、错误检查宏和 CPU 线程池，`torch.utils.cpp_extension` 负责编译。下面实现一个 CPU 版的 RMSNorm：

```cpp title="rmsnorm_ext.cpp" project="ext"
#include <torch/extension.h>

#include <cmath>

// y = x / sqrt(mean(x^2) + eps) * weight，沿最后一维归一化
torch::Tensor rmsnorm(torch::Tensor x, torch::Tensor weight, double eps) {
  TORCH_CHECK(x.device().is_cpu() && weight.device().is_cpu(), "这个示例只实现了 CPU 版本");
  TORCH_CHECK(x.scalar_type() == torch::kFloat32 && weight.scalar_type() == torch::kFloat32, "只支持 float32");
  TORCH_CHECK(x.size(-1) == weight.size(0), "weight 的长度必须等于 x 的最后一维");

  auto xc = x.contiguous();      // 按指针遍历之前，先保证内存是连续的
  auto wc = weight.contiguous(); // 必须存进变量：临时张量在语句结束时就释放了
  auto y = torch::empty_like(xc);
  const int64_t d = xc.size(-1), rows = xc.numel() / d;
  const float* px = xc.data_ptr<float>();
  const float* pw = wc.data_ptr<float>();
  float* py = y.data_ptr<float>();

  at::parallel_for(0, rows, 16, [&](int64_t begin, int64_t end) {   // 用 PyTorch 自己的 CPU 线程池
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

// 注册成 torch.ops.demo.rmsnorm：有明确的 schema，torch.compile、CUDA Graph 都能识别
TORCH_LIBRARY(demo, m) { m.def("rmsnorm(Tensor x, Tensor weight, float eps) -> Tensor"); }
TORCH_LIBRARY_IMPL(demo, CPU, m) { m.impl("rmsnorm", &rmsnorm); }

// 同时也用 pybind11 暴露一个普通函数
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
xt = x.transpose(0, 1)   # 非连续的输入
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

```text title="输出"
pybind11 接口与参考实现一致
torch.ops.demo.rmsnorm 与参考实现一致
非连续的输入也正确
类型检查： 只支持 float32
```

写张量算子的检查清单：

- **检查设备、数据类型、形状**，用 `TORCH_CHECK` 给出清楚的错误信息。不检查的后果是读错内存、静默地算出错误结果；
- **处理非连续的输入**：`transpose`、切片之后的张量在内存里不是连续的，按 `data_ptr` 线性遍历之前先 `.contiguous()`（或者按 `stride` 访问）；
- **临时张量的生命周期**：`weight.contiguous().data_ptr<float>()` 在 `weight` 本来就连续时碰巧能用，不连续时 `contiguous()` 返回的临时张量在这一行结束时就释放了，指针悬垂——和[移动语义](../basics/move.md)、[值语义](../basics/value-raii.md)里讲的临时对象是同一个问题；
- **用参考实现对拍**：`torch.testing.assert_close` 按数据类型给出合理的容差；
- **注册成 `torch.ops`**：`TORCH_LIBRARY` 给算子一个 schema（参数和返回值的类型、哪些参数会被原地修改），`torch.compile` 才能把它当成一个不透明但可追踪的节点，而不是在这里断开图；还可以为 `Meta` 设备注册一个只计算输出形状的实现，供编译期推导形状用。

CUDA 版本的写法相同：再写一个 `.cu` 文件放 kernel 和启动代码，在 `TORCH_LIBRARY_IMPL(demo, CUDA, m)` 里注册，用 `at::cuda::getCurrentCUDAStream()` 在 PyTorch 当前的 stream 上启动 kernel（见 CUDA 手册的[生态：cuBLAS、CUTLASS 与 PyTorch](cuda://tools/ecosystem/)）。

!!! interview "面试怎么答"
    绑定题：pybind11 把 `list[int]` 转成 `std::vector<int>` 会拷贝，大数组用 `py::array_t` 直接读 numpy 的内存；C++ 异常会自动翻译成 Python 异常（`std::invalid_argument` 变成 `ValueError`）；长时间的纯 C++ 计算要释放 GIL（`py::gil_scoped_release`），释放后不能碰 Python 对象，持有 C++ 锁时不要去拿 GIL。PyTorch 算子要先检查设备、数据类型、形状和连续性（`contiguous()` 返回的临时张量要保证活到计算结束），并用参考实现对拍；用 `TORCH_LIBRARY` 注册成 `torch.ops` 才能被 `torch.compile` 和 CUDA Graph 正确处理（有 schema、有 fake 实现），直接用 pybind11 暴露的函数对它们是黑盒。

## 练习

1. 给 `kvpool_py` 加一个 `allocate_many(counts)`：参数是一个 numpy 的 `int32` 数组（每个请求要的块数），全部分配成功才提交，否则一个都不分配，返回 `list[list[int]] | None`。要求：整个过程中释放 GIL 吗？为什么？

??? success "参考答案"
    ```cpp
    std::optional<std::vector<std::vector<std::int32_t>>> allocate_many(
        BlockPool& pool, py::array_t<std::int32_t, py::array::c_style | py::array::forcecast> counts) {
      auto c = counts.unchecked<1>();
      std::int64_t total = 0;
      for (py::ssize_t i = 0; i < c.shape(0); ++i) total += c(i);
      if (total > pool.num_free()) return std::nullopt;       // 先检查总量：要么全成功，要么不动
      std::vector<std::vector<std::int32_t>> out;
      for (py::ssize_t i = 0; i < c.shape(0); ++i) out.push_back(*pool.allocate(c(i)));
      return out;
    }
    // 绑定：.def("allocate_many", &allocate_many)
    ```

    这里**不释放 GIL**：`counts` 是一个 Python 对象，读它的内存时要持有 GIL（除非先把数据拷贝到 C++ 的 `vector` 里）；而且 `BlockPool` 没有加锁，它的线程安全恰恰依赖于"只有持有 GIL 的线程才能调用它"。一个没有内部锁的 C++ 对象，一旦在释放 GIL 的区间里被访问，多个 Python 线程就可能同时修改它。

2. 用户反馈：在 `torch.compile` 编译过的模型里调用 `ext.rmsnorm`（pybind11 暴露的版本）时，编译器报 graph break，而 `torch.ops.demo.rmsnorm` 没有这个问题。为什么？

??? success "参考答案"
    pybind11 暴露的函数对 `torch.compile`（Dynamo）来说是一个无法追踪的任意 Python 可调用对象，它不知道这个函数会读写哪些张量、输出的形状是什么，只能在这里把图断开，回退到 eager 执行。
    `torch.ops.demo.rmsnorm` 有注册过的 schema，Dynamo 知道它的输入输出类型、是否原地修改；再为 `Meta`（或用 `register_fake`）注册一个形状推导，编译器就能把它当成图里的一个节点，前后的算子也能继续被融合。这是推理框架都把自定义算子注册成 `torch.ops` 的原因。

## 小结

- [x] pybind11 暴露 C++ 类和函数：STL 容器自动转换（会拷贝），大数组用 `py::array_t` 直接读内存；异常自动翻译成 Python 异常。
- [x] 长时间的纯 C++ 计算释放 GIL，释放后不碰 Python 对象；持有 C++ 锁时不要去拿 GIL。
- [x] 张量算子：检查设备、类型、形状，处理非连续输入，注意 `contiguous()` 返回的临时张量的生命周期，用参考实现对拍。
- [x] 用 `TORCH_LIBRARY` 注册成 `torch.ops`，才能和 `torch.compile`、CUDA Graph 良好配合。
