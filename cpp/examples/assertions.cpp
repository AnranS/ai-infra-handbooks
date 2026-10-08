#include <vector>

int main() {
  std::vector<float> logits(128);
  int token = 128;                 // 词表大小是 128，合法下标是 0～127
  return logits[token] > 0;
}
