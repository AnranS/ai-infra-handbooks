---
title: 法定人数与 Raft 投票
chapter: dist/replication.md
difficulty: 中等
tags: [共识, Raft, 法定人数, 选举]
---
实现共识里最核心的三个判断：

1. `fresh_read(n, w, r)`：`n` 个副本、写 `w` 个、读 `r` 个时，读到的是否一定包含最新写入（`w + r > n`）；
2. `tolerated_failures(n)`：多数派配置下能容忍几台故障；
3. `can_elect(total, alive)`：`alive` 台存活时能否选出领导者；
4. `grant_vote(voter, candidate)`：Raft 的投票规则。两个参数都是字典 `{"term": 任期, "voted_for": 已投给谁或 None, "last_log_index": 日志长度, "last_log_term": 最后一条的任期, "id": 编号}`，`candidate` 还带 `"term"`（它发起选举后的新任期）。返回 `True` 表示投票，规则是**全部满足**才投：
   - 候选人的任期 **大于** 投票者的任期（等于时，只有还没投过票或已经投给同一个候选人才投）；
   - 候选人的日志至少和投票者一样新：先比最后一条的任期，任期相同再比日志长度。

```python
fresh_read(5, 3, 3)                      # True
tolerated_failures(5)                    # 2
can_elect(5, 2)                          # False：2 台不是多数派
grant_vote({"term": 1, "voted_for": None, "last_log_index": 3, "last_log_term": 1, "id": 0},
           {"term": 2, "last_log_index": 3, "last_log_term": 1, "id": 1})    # True
```

<!-- 题解 -->
`w + r > n` 保证读写集合必有交集。多数派（`w = r = n // 2 + 1`）读写对称、能容忍 `n // 2` 台故障；偶数副本不划算（4 台和 3 台一样只能容忍 1 台），所以集群通常是 3 或 5 台。

投票规则里的两条各解决一个问题：**任期**识别过期的领导者和候选人——任何人收到更大的任期就退回跟随者，旧主带着旧任期回来会被拒绝；**日志新旧检查**保证已提交的条目不会丢——被多数派复制过的条目，必然出现在多数派的日志里，而当选需要多数票，于是当选者的日志一定包含所有已提交的条目。

比较日志新旧时要**先比任期再比长度**：一个日志更长但最后一条任期更旧的节点，它多出来的那些条目必然是未提交的、要被覆盖的，不能因为"长"就认为它更新。
