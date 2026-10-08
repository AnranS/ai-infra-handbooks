#include <cstdio>
#include <span>
#include <string>
#include <string_view>
#include <vector>

std::vector<std::string_view> split(std::string_view s, char sep) {
  std::vector<std::string_view> out;
  while (true) {
    auto pos = s.find(sep);
    out.push_back(s.substr(0, pos));   // 只是（指针, 长度），不拷贝字符
    if (pos == std::string_view::npos) break;
    s.remove_prefix(pos + 1);
  }
  return out;
}

// 接受任何连续的 float 序列：vector、数组、另一个 span 的一段
float mean(std::span<const float> xs) {
  float s = 0;
  for (float x : xs) s += x;
  return xs.empty() ? 0 : s / xs.size();
}

int main() {
  std::string line = "model=qwen3 tp=8 max_len=32768";
  for (auto kv : split(line, ' ')) std::printf("[%.*s] ", int(kv.size()), kv.data());
  std::printf("\n");
  std::vector<float> latencies = {12.5f, 13.0f, 40.0f, 12.0f};
  float arr[3] = {1, 2, 3};
  std::printf("mean(vector)=%.2f mean(前两个)=%.2f mean(数组)=%.2f\n", mean(latencies),
              mean(std::span(latencies).first(2)), mean(arr));
}
