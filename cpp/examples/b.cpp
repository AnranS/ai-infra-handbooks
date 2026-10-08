#include "counter.hpp"

void bump_in_b() {
  ++static_counter;
  ++inline_counter;
}
