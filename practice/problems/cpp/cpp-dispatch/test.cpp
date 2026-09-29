#include <array>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "judge.hpp"
#include "user.cpp"

int main() {
  pj::run("example_head_dim", [] {
    for (int hd : {64, 96, 128, 256}) {
      int got = dispatch_head_dim(hd, [](auto d) {
        std::array<int, decltype(d)::value> regs{};   // 必须是编译期常量才能当数组大小
        return int(regs.size());
      });
      pj::require_eq(got, hd, "派发到的常量应该等于运行时的 head_dim");
    }
  });
  pj::run("unsupported_throws", [] {
    for (int hd : {0, 80, 100, 512}) {
      pj::require_throws<std::invalid_argument>([&] { dispatch_head_dim(hd, [](auto d) { return decltype(d)::value; }); },
                                                "head_dim=" + std::to_string(hd) + " 不支持");
    }
  });
  pj::run("dtype", [] {
    auto size_of = [](DType t) { return dispatch_dtype(t, [](auto tag) { return sizeof(typename decltype(tag)::type); }); };
    pj::require_eq(size_of(DType::F32), sizeof(float), "F32");
    pj::require_eq(size_of(DType::BF16), std::size_t(2), "BF16");
    bool is_bf16 = dispatch_dtype(DType::BF16, [](auto tag) { return std::is_same_v<typename decltype(tag)::type, bf16>; });
    bool is_f16 = dispatch_dtype(DType::F16, [](auto tag) { return std::is_same_v<typename decltype(tag)::type, f16>; });
    pj::require(is_bf16 && is_f16, "BF16 要派发到 bf16，F16 要派发到 f16");
  });
  pj::run("nested", [] {
    std::string got = dispatch_dtype(DType::BF16, [](auto tag) {
      using T = typename decltype(tag)::type;
      return dispatch_head_dim(96, [](auto d) {
        constexpr int D = decltype(d)::value;
        return std::to_string(sizeof(T) * D);   // 一个头的字节数
      });
    });
    pj::require_eq(got, std::string("192"), "嵌套派发：bf16 × 96");
  });
}
