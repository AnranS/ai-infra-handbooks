#include "counter.hpp"

void bump_in_a() {
  ++static_counter;
  ++inline_counter;
}
