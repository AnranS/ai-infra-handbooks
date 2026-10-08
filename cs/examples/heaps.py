# 堆的三种典型用法：Top-K、多路归并、以及"动态取最小"的调度
import heapq
import random

rng = random.Random(0)
data = [rng.randrange(1000) for _ in range(20000)]

# 1. Top-K：维护一个大小为 k 的小顶堆，堆顶是"第 k 大"
def top_k(a, k):
    heap = []
    for x in a:
        if len(heap) < k:
            heapq.heappush(heap, x)
        elif x > heap[0]:                      # 比第 k 大还大，替换堆顶
            heapq.heapreplace(heap, x)
    return sorted(heap, reverse=True)


print("Top-5：", top_k(data, 5), "（和排序取前 5 一致：", sorted(data, reverse=True)[:5] == top_k(data, 5), "）")
print("复杂度：排序 O(n log n)，堆 O(n log k)——k 远小于 n 时堆明显更省")

# 2. 多路归并：k 条有序序列合成一条
streams = [sorted(rng.randrange(100) for _ in range(5)) for _ in range(4)]
print("\n四条有序序列：", streams)
print("归并结果：", list(heapq.merge(*streams)))

# 3. 调度：每次取"最早空闲"的机器，就是一个小顶堆
def schedule(tasks, machines):
    """tasks 是每个任务的耗时，返回全部完成的时刻"""
    heap = [0.0] * machines                    # 每台机器的空闲时刻
    heapq.heapify(heap)
    for cost in tasks:
        free_at = heapq.heappop(heap)          # 最早空闲的机器
        heapq.heappush(heap, free_at + cost)
    return max(heap)


tasks = [5, 3, 8, 2, 7, 1, 6]
for m in (1, 2, 3):
    print(f"{len(tasks)} 个任务（共 {sum(tasks)} 单位）用 {m} 台机器：{schedule(tasks, m):.0f} 单位完成，"
          f"理论下界 {max(max(tasks), sum(tasks) / m):.1f}")
print("\n把任务按耗时从大到小排序再调度（最长优先，LPT）：")
for m in (2, 3):
    print(f"  {m} 台：顺序到达 {schedule(tasks, m):.0f}，先排序 {schedule(sorted(tasks, reverse=True), m):.0f}")
