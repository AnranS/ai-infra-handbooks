#include <cstddef>
#include <cstdint>
#include <cstdio>

struct Bad {                  // 字段顺序随手写
  bool done;
  double temperature;
  std::int32_t len;
  bool stream;
  std::int64_t id;
};

struct Good {                 // 按对齐要求从大到小排
  double temperature;
  std::int64_t id;
  std::int32_t len;
  bool done;
  bool stream;
};

int main() {
  std::printf("Bad：sizeof=%zu alignof=%zu\n", sizeof(Bad), alignof(Bad));
  std::printf("Good：sizeof=%zu alignof=%zu\n", sizeof(Good), alignof(Good));
  std::printf("Bad 的字段偏移：done=%zu temperature=%zu len=%zu stream=%zu id=%zu\n", offsetof(Bad, done),
              offsetof(Bad, temperature), offsetof(Bad, len), offsetof(Bad, stream), offsetof(Bad, id));
}
