#include <cstdio>
#include <unordered_map>

int rehashes(bool reserve) {
  std::unordered_map<int, int> m;
  if (reserve) m.reserve(100000);
  int n = 0;
  auto buckets = m.bucket_count();
  for (int i = 0; i < 100000; ++i) {
    m[i] = i;
    if (m.bucket_count() != buckets) {
      ++n;
      buckets = m.bucket_count();
    }
  }
  return n;
}

int main() {
  std::printf("不预留：rehash %d 次\n", rehashes(false));
  std::printf("预留：rehash %d 次\n", rehashes(true));
}
