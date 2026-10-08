// g++ -std=c++20 counter_main.cpp a.cpp b.cpp && ./a.out
#include <cstdio>

#include "counter.hpp"

void bump_in_a();
void bump_in_b();

int main() {
  bump_in_a();
  bump_in_a();
  bump_in_b();
  std::printf("main 看到的 static_counter = %d\n", static_counter);
  std::printf("inline_counter = %d\n", inline_counter);
}
