#include <cstdint>
#include <cstdio>

struct SeqStateBad {
  bool is_prefill;
  std::int64_t request_id;
  std::int16_t lora_slot;
  float temperature;
  bool finished;
  std::int32_t num_computed_tokens;
  double arrival_time;
  std::int8_t priority;
};

struct SeqState {
  std::int64_t request_id;
  double arrival_time;
  float temperature;
  std::int32_t num_computed_tokens;
  std::int16_t lora_slot;
  bool is_prefill;
  bool finished;
  std::int8_t priority;
};
static_assert(sizeof(SeqState) == 32, "布局变了：检查字段顺序");

int main() { std::printf("重排前 %zu 字节，重排后 %zu 字节\n", sizeof(SeqStateBad), sizeof(SeqState)); }
