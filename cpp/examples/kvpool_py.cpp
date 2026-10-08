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
