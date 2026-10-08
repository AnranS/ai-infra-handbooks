// cute_layout.cu —— CuTe 的布局代数：只在主机端运行，不需要 GPU
// 编译：nvcc -std=c++17 -I${CUTLASS_DIR}/include cute_layout.cu -o cute_layout
#include <cstdio>
#include <cute/tensor.hpp>
using namespace cute;

int main() {
  // 1. Layout = Shape : Stride，把逻辑坐标映射成内存下标
  auto col_major = make_layout(make_shape(4, 8));                  // 默认列主序
  auto row_major = make_layout(make_shape(4, 8), LayoutRight{});   // 行主序
  print(col_major); printf("\n");
  print(row_major); printf("\n");
  printf("(1,2) -> col_major %d, row_major %d\n\n", int(col_major(1, 2)), int(row_major(1, 2)));

  // 2. 编译期常量用 Int<N>{}，打印时带下划线
  auto tile = make_layout(make_shape(Int<4>{}, Int<8>{}), LayoutRight{});
  print_layout(tile);

  // 3. 分块：把 8x8 的行主序矩阵切成 4x4 的块，得到 ((块内), (块号)) 的层次化布局
  auto mat = make_layout(make_shape(Int<8>{}, Int<8>{}), LayoutRight{});
  auto tiled = zipped_divide(mat, make_shape(Int<4>{}, Int<4>{}));
  print(tiled); printf("\n");
  printf("block (1,0), element (2,3) -> %d\n\n", int(tiled(make_coord(make_coord(2, 3), make_coord(1, 0)))));

  // 4. swizzle：第 r 行的列号与 r 做异或，打散 bank
  auto swizzled = composition(Swizzle<2, 0, 3>{}, tile);
  print_layout(swizzled);
  return 0;
}
