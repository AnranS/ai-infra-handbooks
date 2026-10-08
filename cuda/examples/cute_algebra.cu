// cute_algebra.cu —— 用 CUTLASS 里真正的 CuTe 核对本章的布局代数：只在主机端运行，不需要 GPU
// 编译：nvcc -std=c++17 -arch=sm_80 -I${CUTLASS_DIR}/include cute_algebra.cu -o cute_algebra
#include <cstdio>
#include <cute/tensor.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/atom/mma_atom.hpp>
using namespace cute;

template <class T>
void show(const char* name, T const& x) {
  printf("%-20s", name);
  print(x);
  printf("\n");
}

int main() {
  // 1. 布局代数：形状、步长都用编译期常量 _N（即 Int<N>{}），化简才能在编译期完成
  show("coalesce", coalesce(make_layout(make_shape(_2{}, make_shape(_1{}, _6{})),
                                        make_stride(_1{}, make_stride(_6{}, _2{})))));
  show("coalesce (int)", coalesce(make_layout(make_shape(2, make_shape(1, 6)),   // 普通 int：只能展平
                                              make_stride(1, make_stride(6, 2)))));
  auto a = make_layout(make_shape(_6{}, _2{}), make_stride(_8{}, _2{}));
  auto b = make_layout(make_shape(_4{}, _3{}), make_stride(_3{}, _1{}));
  show("composition", composition(a, b));
  auto m = make_layout(make_shape(_4{}, _8{}), LayoutRight{});                 // 4x8 行优先
  show("transpose view", composition(m, make_layout(make_shape(_8{}, _4{}), LayoutRight{})));
  show("even rows", composition(m, make_layout(make_shape(_2{}, _8{}), make_stride(_2{}, _4{}))));
  show("complement", complement(make_layout(_4{}, _2{}), _24{}));
  show("logical_divide", logical_divide(make_layout(_24{}), make_layout(_4{}, _2{})));
  auto tile = make_layout(make_shape(_16{}, _8{}));                             // 16x8 列优先
  show("zipped_divide 2x2", zipped_divide(tile, make_shape(_2{}, _2{})));
  show("zipped_divide 16x2", zipped_divide(tile, make_shape(_16{}, _2{})));

  // 2. 线程划分：CuTe 教程 sgemm_sm80.cu 里 A 的拷贝，128 个线程按 16x8（k 方向相邻）排，每个线程 8 个 half
  using CopyA = decltype(make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, half_t>{},
                                         Layout<Shape<_16, _8>, Stride<_8, _1>>{}, Layout<Shape<_1, _8>>{}));
  show("TiledCopy tile", typename CopyA::Tiler_MN{});
  show("TiledCopy TV", typename CopyA::TiledLayout_TV{});

  // 3. Tensor Core：mma.sync.m16n8k16 的寄存器布局，(线程, 值) -> 块内列优先下标
  using Traits = MMA_Traits<SM80_16x8x16_F32F16F16F32_TN>;
  show("MMA A (thr,val)", typename Traits::ALayout{});
  show("MMA B (thr,val)", typename Traits::BLayout{});
  show("MMA C (thr,val)", typename Traits::CLayout{});
  return 0;
}
