# 贪心：先按某个关键字排序，再一遍扫过去。难的不是代码，是证明"为什么这么选是对的"
def merge_intervals(intervals):
    """合并重叠区间：按起点排序，能接上就扩展右端点"""
    out = []
    for start, end in sorted(intervals):
        if out and start <= out[-1][1]:
            out[-1][1] = max(out[-1][1], end)
        else:
            out.append([start, end])
    return out


def max_non_overlapping(intervals):
    """最多能选几个互不重叠的区间：按**终点**排序，能选就选"""
    count, last_end = 0, float("-inf")
    for start, end in sorted(intervals, key=lambda t: t[1]):
        if start >= last_end:
            count += 1
            last_end = end
    return count


def min_rooms(meetings):
    """同时进行的最大数量（需要几个会议室 / 几个并发槽位）：把端点排序后扫描"""
    events = sorted([(s, 1) for s, _ in meetings] + [(e, -1) for _, e in meetings])
    cur = best = 0
    for _, delta in events:                    # 同一时刻先处理结束（-1 排在 +1 前面）
        cur += delta
        best = max(best, cur)
    return best


data = [[1, 3], [2, 6], [8, 10], [15, 18]]
print("区间：", data)
print("合并后：", merge_intervals(data))
print("最多能选几个互不重叠：", max_non_overlapping(data))
print("需要几个并发槽位：", min_rooms(data))
print()
print("三道题排序的关键字不同：合并按起点，选最多按终点，算并发按端点扫描。")
print("按起点选最多是错的——先来的可能很长，挡住后面好几个短的。")
print()

# 推理系统里的例子：按输出长度排序能减少"批次里的浪费"
reqs = [("A", 500), ("B", 20), ("C", 480), ("D", 30)]
for name, order in [("按到达顺序", reqs), ("按预估长度分组", sorted(reqs, key=lambda t: t[1]))]:
    batches = [order[:2], order[2:]]
    waste = sum(max(x[1] for x in b) * len(b) - sum(x[1] for x in b) for b in batches)
    print(f"{name}：分两批 {[[x[0] for x in b] for b in batches]}，浪费的 token 槽位 {waste}")
