#include <cstdio>
#include <vector>

struct Request {
  int id;
  int steps = 0;
};

int main() {
  std::vector<Request> queue = {{1}, {2}, {3}};
  std::size_t current = 0;         // 保存下标，而不是引用
  int processed = 0;
  for (int i = 0; i < 3; ++i) {
    queue.push_back({10 + i});
    queue[current].steps++;
    processed++;
  }
  std::printf("processed=%d queue=%zu\n", processed, queue.size());
}
