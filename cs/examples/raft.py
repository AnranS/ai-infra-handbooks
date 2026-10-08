# Raft 选举与日志复制的最小模拟：离散时间、消息立刻送达，用来看清任期、投票和提交的规则
import random


class Node:
    def __init__(self, i, n):
        self.i, self.n = i, n
        self.term, self.voted_for, self.role = 0, None, "follower"
        self.log = []                                  # [(任期, 命令)]
        self.commit = 0                                # 已提交的日志条数
        self.timeout = 0                               # 还有几个时钟滴答就超时发起选举
        self.match = {}                                # 领导者视角：每个跟随者已复制到第几条


class Cluster:
    def __init__(self, n, seed=1):
        self.rng = random.Random(seed)
        self.nodes = [Node(i, n) for i in range(n)]
        self.down = set()
        for node in self.nodes:
            node.timeout = self.rng.randint(3, 6)      # 随机超时，避免所有人同时发起选举

    def alive(self):
        return [x for x in self.nodes if x.i not in self.down]

    def leader(self):
        return next((x for x in self.alive() if x.role == "leader"), None)

    def tick(self):
        lead = self.leader()
        if lead:                                       # 领导者发心跳，压住跟随者的超时
            for f in self.alive():
                if f is not lead:
                    f.role, f.term, f.timeout = "follower", lead.term, self.rng.randint(3, 6)
                    f.log = lead.log[:]                # 简化：一次把日志补齐
                    lead.match[f.i] = len(f.log)
            acked = sorted([len(lead.log)] + [lead.match.get(f.i, 0) for f in self.alive() if f is not lead])
            lead.commit = acked[len(self.alive()) // 2]   # 多数派已复制的位置就是提交点
            return
        for node in self.alive():                      # 没有领导者：等超时的人发起选举
            node.timeout -= 1
            if node.timeout > 0:
                continue
            node.term, node.role, node.voted_for = node.term + 1, "candidate", node.i
            votes = 1
            for other in self.alive():                 # 只投给日志不比自己旧的候选人
                if other is node:
                    continue
                up_to_date = (len(node.log), node.log[-1][0] if node.log else 0) >= \
                             (len(other.log), other.log[-1][0] if other.log else 0)
                if other.term < node.term and up_to_date:
                    other.term, other.voted_for = node.term, node.i
                    votes += 1
            if votes > self.n_total() // 2:            # 拿到多数票才当选
                node.role = "leader"
                node.match = {}
                return f"第 {node.term} 任期：节点 {node.i} 当选（{votes}/{self.n_total()} 票）"
            node.timeout = self.rng.randint(3, 6)
        return None

    def n_total(self):
        return len(self.nodes)

    def propose(self, cmd):
        lead = self.leader()
        if lead is None:
            return False
        lead.log.append((lead.term, cmd))
        return True


c = Cluster(5)
log = []
for step in range(30):
    if step == 12:
        old = c.leader()
        c.down.add(old.i)                              # 领导者宕机
        log.append(f"第 {step} 拍：领导者 节点 {old.i} 宕机，已提交 {old.commit} 条")
    msg = c.tick()
    if msg:
        log.append(f"第 {step} 拍：{msg}")
    if step in (6, 8, 10, 20, 22):
        c.propose(f"cmd{step}")
for line in log:
    print(line)
lead = c.leader()
print(f"\n最终：领导者是节点 {lead.i}（第 {lead.term} 任期），日志 {len(lead.log)} 条，已提交 {lead.commit} 条")
print("各节点日志：", {x.i: len(x.log) for x in c.nodes})
