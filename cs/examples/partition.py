# 网络分区时谁还能工作：只有拿到多数派的那一侧能选出领导者、能提交
def can_work(sizes, total):
    return [s > total // 2 for s in sizes]


cases = [
    ("5 个副本，3 : 2 分区", [3, 2], 5),
    ("5 个副本，2 : 2 : 1 三分区", [2, 2, 1], 5),
    ("4 个副本，两个机房各 2 台", [2, 2], 4),
    ("5 个副本：机房 A 2 台、机房 B 2 台、第三地仲裁 1 台，A 与其他断开", [2, 3], 5),
    ("3 个副本，1 台宕机后再分区 1 : 1", [1, 1], 3),
]
for name, sizes, total in cases:
    flags = can_work(sizes, total)
    who = "、".join(f"{s} 台那侧{'可以' if ok else '不行'}" for s, ok in zip(sizes, flags))
    print(f"{name}：{who}")
print()
print("故障切换要多久（心跳 h、选举超时随机取 [t, 2t]）：")
for h, t in [(50, 150), (100, 300), (500, 1500)]:
    print(f"  心跳 {h} ms、选举超时 {t}～{2 * t} ms：发现故障最多 {2 * t} ms，"
          f"加上一轮投票和日志追赶，通常 {t + h}～{2 * t + 2 * h} ms 内恢复写入")
