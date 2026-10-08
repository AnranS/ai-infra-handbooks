// 用 CPU 线程模拟"写输入 → release 写对端的标志 → acquire 等自己的标志 → 读所有对端的输入"
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <thread>
#include <vector>

constexpr int kRanks = 4, kN = 8;

struct alignas(128) Flag {   // 每个标志独占 128 字节，和 vLLM 的 Signal 一样
  std::atomic<std::uint32_t> v{0};
};

struct Rank {
  float data[kN];            // 本 rank 的输入：相当于 IPC 映射出来、对端可以直接读的显存
  Flag start[kRanks];        // start[p]：rank p 在"开始屏障"写给我的标志
  Flag end[kRanks];          // end[p]：rank p 在"结束屏障"写给我的标志
};

// 两个屏障必须用不同的标志数组（at_end 选择用哪一组）
void barrier(std::vector<Rank>& ranks, bool at_end, int me, std::uint32_t flag) {
  auto slot = [&](int owner, int from) -> Flag& { return at_end ? ranks[owner].end[from] : ranks[owner].start[from]; };
  for (int p = 0; p < kRanks; ++p) slot(p, me).v.store(flag, std::memory_order_release);   // 通知每个对端
  for (int p = 0; p < kRanks; ++p) {
    while (slot(me, p).v.load(std::memory_order_acquire) != flag) {                         // 等每个对端
      std::this_thread::yield();   // GPU 上是空转；CPU 线程可能比核数多，让出时间片
    }
  }
}

int main() {
  std::vector<Rank> ranks(kRanks);
  std::vector<std::vector<float>> out(kRanks, std::vector<float>(kN));
  std::vector<std::thread> ts;
  for (int r = 0; r < kRanks; ++r) {
    ts.emplace_back([&, r] {
      for (std::uint32_t round = 1; round <= 3; ++round) {
        for (int i = 0; i < kN; ++i) ranks[r].data[i] = float(r + 1) * round;   // 写自己的输入
        barrier(ranks, false, r, round);    // 开始屏障：所有人的输入都写好了
        for (int i = 0; i < kN; ++i) {
          float s = 0;
          for (int p = 0; p < kRanks; ++p) s += ranks[p].data[i];              // 直接读对端的输入求和
          out[r][i] = s;
        }
        barrier(ranks, true, r, round);     // 结束屏障：所有人都读完了，下一轮才能覆盖输入
      }
    });
  }
  for (auto& t : ts) t.join();
  std::printf("第 3 轮 all-reduce 的结果：");
  for (int r = 0; r < kRanks; ++r) std::printf("%.0f ", out[r][0]);
  std::printf("\n");
}
