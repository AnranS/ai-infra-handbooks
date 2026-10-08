# Replication and consensus: who gets to decide

<p class="lead">The previous chapter left a loose end: every client has to agree on which instances exist, or the routing falls apart. That kind of small state where everyone has to agree, the cluster membership, the sharding scheme, who the leader is, the configuration version, usually goes to a consensus system such as etcd. This chapter works through why consensus is needed, how a quorum guarantees you read the latest value, what Raft's election and log replication are doing (with a minimal simulation you can run), what split brain and leases are about, and which things in an inference cluster belong in a consensus system and which do not.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. In primary-replica replication, what is wrong with the simplest way of promoting a replica when the primary dies?
    2. What condition does "write W replicas, read R replicas" have to satisfy to guarantee you read the latest write?
    3. What is a Raft term for? Why does a candidate have to check whether the other node's log is newer than its own?
    4. Five replicas are split by the network into a side of 3 and a side of 2. What can each side do? Why is the replica count usually odd?
    5. Which state in an inference cluster belongs in etcd, and which absolutely does not?

??? success "Answers for the self-test (answer first, then open this)"
    1. Who decides the primary is dead, and who appoints the new one? If every client decides for itself, half of them think A is the primary and half think B is, both primaries accept writes, and the data forks (split brain). The old primary may also have had nothing worse than a network blip, and it still believes it is the primary when it comes back. Solving both needs consensus: a majority agreeing on who the leader is and which term we are in.
    2. W + R > N. Any write set and any read set then necessarily intersect, so at least one replica among those read holds the latest write. The most common configuration is a majority (W = R = floor(N/2)+1), where both reads and writes tolerate floor(N/2) failures.
    3. The term is a monotonically increasing logical clock used to recognise a stale leader: anyone receiving a message with a higher term than its own falls back to follower, and an old primary returning with an old term is rejected. Checking whose log is newer guarantees that committed entries are never lost: only a candidate whose log is at least as new can win a vote, so whoever is elected necessarily holds every committed entry.
    4. Only the side with 3 can elect a leader and commit (the majority is 3); the side with 2 can only read possibly stale data and cannot write. Odd counts are used because even ones are poor value: 4 replicas need a majority of 3 and still tolerate 1 failure, the same as 3 replicas, for one machine more, and they add the 2-against-2 partition where neither side can work.
    5. In it goes metadata that is small, critical and changes rarely: the instance membership list, the sharding scheme, the routing table's version, model versions and configuration, distributed locks and leader election. What absolutely does not go in is high-frequency data: per-request state, KV cache locations, metrics, queue depths. Those are written in volume and stale within moments, and putting them in a consensus system drags it down.

## Why consensus is needed {#为什么需要共识}

In a single-process system, who decides is not a question: there is one process and the state is in its memory. Make it multi-replica and the trouble starts:

- **Disagreement**: A thinks the primary is dead, B thinks it is not.
- **Split brain**: two nodes both think they are the primary, each accepts writes, and the data forks.
- **Reordering and replay**: an old primary comes back with stale commands once the network recovers.

Consensus algorithms (Paxos, Raft) solve one thing: **getting a majority to agree on the order of a series of operations**, and guaranteeing that once an operation has been declared committed it will not be overturned. With that, who the leader is, what the membership list currently is, and who holds this lock can all be expressed as writes to one log.

## A quorum: intersecting reads and writes {#法定人数读写交集}

First a case that does not need full consensus: multi-replica storage where a write succeeds on W replicas and a read takes the newest of R.

```python title="quorum.py"
# the quorum: write W replicas and read R replicas, and as long as W + R > N the latest write is always read
from itertools import combinations


def always_fresh(n, w, r):
    """任取一个写集合和一个读集合，是否总有交集"""
    return all(set(a) & set(b) for a in combinations(range(n), w) for b in combinations(range(n), r))


print("N   W   R   读一定最新   能容忍几台故障（写）")
for n, w, r in [(3, 2, 2), (3, 3, 1), (3, 1, 3), (3, 1, 1), (5, 3, 3), (5, 4, 2), (5, 2, 2)]:
    print(f"{n}   {w}   {r}   {'是' if always_fresh(n, w, r) else '否':6s}      {n - w}")
print()
print("多数派（W = R = ⌊N/2⌋+1）在不同副本数下的容错能力：")
for n in (1, 3, 5, 7):
    majority = n // 2 + 1
    print(f"  {n} 个副本：多数派 {majority}，能容忍 {n - majority} 台故障")
print()
print("为什么用奇数个副本：加一台不一定提高容错")
for n in range(2, 8):
    print(f"  {n} 个副本：多数派 {n // 2 + 1}，容忍 {n - (n // 2 + 1)} 台")
```

```text title="output"
N   W   R   读一定最新   能容忍几台故障（写）
3   2   2   是           1
3   3   1   是           0
3   1   3   是           2
3   1   1   否           2
5   3   3   是           2
5   4   2   是           1
5   2   2   否           3

多数派（W = R = ⌊N/2⌋+1）在不同副本数下的容错能力：
  1 个副本：多数派 1，能容忍 0 台故障
  3 个副本：多数派 2，能容忍 1 台故障
  5 个副本：多数派 3，能容忍 2 台故障
  7 个副本：多数派 4，能容忍 3 台故障

为什么用奇数个副本：加一台不一定提高容错
  2 个副本：多数派 2，容忍 0 台
  3 个副本：多数派 2，容忍 1 台
  4 个副本：多数派 3，容忍 1 台
  5 个副本：多数派 3，容忍 2 台
  6 个副本：多数派 4，容忍 2 台
  7 个副本：多数派 4，容忍 3 台
```

- **W + R > N** guarantees the write and read sets intersect, which guarantees reading the latest value. W=1 and R=1 is fastest but reads stale values, which is eventual consistency.
- **A majority** (W = R = floor(N/2)+1) is the most common configuration, symmetric between reads and writes, tolerating floor(N/2) failures.
- **An even replica count is poor value**: 4 replicas need a majority of 3 and still tolerate 1 failure, the same as 3. So a cluster is generally 3 or 5 machines.

A quorum only solves reading a value that is not stale; it does not order concurrent writes. Giving the operations a global order needs a consensus algorithm.

## Raft: election and log replication {#raft选举与日志复制}

![Figure: Raft, where a majority elects a leader and an entry commits once replicated to a majority](../assets/figures/raft-replication.svg){.aig-svg}

Raft splits consensus into three pieces: **electing a leader**, **replicating the log** and **the safety constraints**.

- Each node is in one of three roles: follower, candidate or leader. **The term** is a monotonically increasing logical clock, incremented at each election.
- A follower that receives no heartbeat from the leader within a random timeout increments the term, becomes a candidate and canvasses everyone for votes. It is elected only with **a majority of the votes**. The random timeout stops everyone starting an election at once and failing round after round.
- The leader accepts the client's writes, appends them to its own log, and replicates them to the followers; an entry **replicated to a majority** can be committed and applied to the state machine.
- Two safety constraints: a vote goes only to a candidate whose log is at least as new as your own; and a leader may only commit entries from its own term (committing earlier ones indirectly). Together these guarantee that committed content is never lost.

The minimal simulation below runs it in discrete time: a leader is elected, a few log entries are replicated, the leader dies, and a new election happens:

```python title="raft.py"
# a minimal simulation of Raft's election and log replication: discrete time, messages delivered at once, to make the rules for terms, votes and commits clear
import random


class Node:
    def __init__(self, i, n):
        self.i, self.n = i, n
        self.term, self.voted_for, self.role = 0, None, "follower"
        self.log = []                                  # [(term, command)]
        self.commit = 0                                # how many log entries are committed
        self.timeout = 0                               # how many ticks until the timeout starts an election
        self.match = {}                                # the leader's view: how far each follower has replicated


class Cluster:
    def __init__(self, n, seed=1):
        self.rng = random.Random(seed)
        self.nodes = [Node(i, n) for i in range(n)]
        self.down = set()
        for node in self.nodes:
            node.timeout = self.rng.randint(3, 6)      # a random timeout, so not everyone starts an election at once

    def alive(self):
        return [x for x in self.nodes if x.i not in self.down]

    def leader(self):
        return next((x for x in self.alive() if x.role == "leader"), None)

    def tick(self):
        lead = self.leader()
        if lead:                                       # the leader sends heartbeats, holding off the followers' timeouts
            for f in self.alive():
                if f is not lead:
                    f.role, f.term, f.timeout = "follower", lead.term, self.rng.randint(3, 6)
                    f.log = lead.log[:]                # simplified: bring the log up to date in one go
                    lead.match[f.i] = len(f.log)
            acked = sorted([len(lead.log)] + [lead.match.get(f.i, 0) for f in self.alive() if f is not lead])
            lead.commit = acked[len(self.alive()) // 2]   # the position a majority has replicated to is the commit point
            return
        for node in self.alive():                      # no leader: whoever times out starts an election
            node.timeout -= 1
            if node.timeout > 0:
                continue
            node.term, node.role, node.voted_for = node.term + 1, "candidate", node.i
            votes = 1
            for other in self.alive():                 # vote only for a candidate whose log is not older than your own
                if other is node:
                    continue
                up_to_date = (len(node.log), node.log[-1][0] if node.log else 0) >= \
                             (len(other.log), other.log[-1][0] if other.log else 0)
                if other.term < node.term and up_to_date:
                    other.term, other.voted_for = node.term, node.i
                    votes += 1
            if votes > self.n_total() // 2:            # elected only with a majority of the votes
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
        c.down.add(old.i)                              # the leader dies
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
```

```text title="output"
第 2 拍：第 1 任期：节点 1 当选（5/5 票）
第 12 拍：领导者 节点 1 宕机，已提交 3 条
第 14 拍：第 2 任期：节点 0 当选（4/5 票）

最终：领导者是节点 0（第 2 任期），日志 5 条，已提交 5 条
各节点日志： {0: 5, 1: 3, 2: 5, 3: 5, 4: 5}
```

Look at the last line: the dead old leader (node 1) stopped at 3 entries while the new leader and the others have 5. When the old node comes back it will find its term behind, fall back to follower and have its inconsistent part overwritten by the new leader. That is exactly the term and the log-recency check at work.

(This simulation leaves out a great deal of real Raft: messages can be lost and reordered, logs have to be compared entry by entry and backed up, there are snapshots and log compaction, and membership changes use joint consensus. It is only here to make the roles, the terms and the commit rule clear.)

## Partitions, split brain and leases {#分区脑裂与租约}

```python title="partition.py"
# who can still work during a network partition: only the side holding a majority can elect a leader and commit
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
```

```text title="output"
5 个副本，3 : 2 分区：3 台那侧可以、2 台那侧不行
5 个副本，2 : 2 : 1 三分区：2 台那侧不行、2 台那侧不行、1 台那侧不行
4 个副本，两个机房各 2 台：2 台那侧不行、2 台那侧不行
5 个副本：机房 A 2 台、机房 B 2 台、第三地仲裁 1 台，A 与其他断开：2 台那侧不行、3 台那侧可以
3 个副本，1 台宕机后再分区 1 : 1：1 台那侧不行、1 台那侧不行

故障切换要多久（心跳 h、选举超时随机取 [t, 2t]）：
  心跳 50 ms、选举超时 150～300 ms：发现故障最多 300 ms，加上一轮投票和日志追赶，通常 200～400 ms 内恢复写入
  心跳 100 ms、选举超时 300～600 ms：发现故障最多 600 ms，加上一轮投票和日志追赶，通常 400～800 ms 内恢复写入
  心跳 500 ms、选举超时 1500～3000 ms：发现故障最多 3000 ms，加上一轮投票和日志追赶，通常 2000～4000 ms 内恢复写入
```

- **Only the majority side can work**. The old leader on the minority side finds it is getting no heartbeat responses from a majority and steps down; clients on that side can only read possibly stale data. This is choosing CP in CAP terms: sacrifice availability during a partition to keep consistency.
- **Splitting evenly across two data centres is a trap**: neither side is a majority, and after the link drops nobody can write. The right approach is three data centres (2:2:1, with an arbiter replica in the third site), or accepting that the primary site is writable and the backup read-only.
- **How long failover takes** is set by the heartbeat interval and the election timeout. With a 50 ms heartbeat and a 150 to 300 ms election timeout, writes usually resume within a few hundred milliseconds; too short a timeout changes leader constantly on an occasional network blip, and too long makes recovery slow.

**A lease** is the most common pattern built on consensus: the leader holds a time-limited lease and, within it, may answer reads directly without going through a majority (much better read performance); it renews before the lease expires. A lease's safety depends on the clock: as long as the nodes' clock drift is bounded, the old primary's lease necessarily expires before the new primary starts serving. A distributed lock built on a lease needs the same care: **the lock can expire without the holder knowing**, so the protected operation either has to be fast enough or carry a version number of its own for validation (a fencing token).

## Consensus in an inference cluster {#推理集群里的共识}

| Purpose | Where it goes | Notes |
| --- | --- | --- |
| The instance membership list and health | etcd / Kubernetes | the router watches for changes and updates the consistent hash ring (the previous chapter) |
| The sharding scheme, the routing table's version | etcd | a change has to take effect atomically, so clients never see different versions |
| Leader election: who runs global scheduling, who runs a weight update | etcd leases plus a distributed lock | mind lock expiry, and have a long operation carry a version number for validation |
| Model versions, configuration, canary ratios | etcd / a configuration service | written rarely, agreed by everyone |
| Request state, KV cache locations | **not** in a consensus system | far too many writes; use dedicated KV storage, or have the router keep an approximate view |
| Metrics, queue depths | **not** in a consensus system | high-frequency, loseable, and only an approximation is needed; use the monitoring system or heartbeat reports |

One rule of thumb: **a consensus system is for storing state that is small and critical**. An etcd write has to be durable on a majority, so a few thousand per second is the ceiling and the latency is in milliseconds. The state of tens of thousands of requests per second in an inference cluster must never go in there.

Kubernetes is itself an example of this pattern: every object lives in etcd and controllers watch for changes and act; the deployment, scaling and service discovery of inference instances all go through it (see [Production deployment and operations](serving://ops/deploy/)). Synchronising multiple ranks inside an inference engine is another matter entirely: that uses collective communication and shared memory (see [Interprocess communication](../os/ipc.md) and [Multi-rank synchronisation](minisgl://serve/scheduler-io/)), because the processes are all on one machine or in one job and do not have to tolerate arbitrary node failures.

**One more word on consistency models.** Linearizability (a read always returns the latest committed value) costs the most; sequential, causal and eventual consistency relax it in turn. For the overwhelming majority of state in an inference service, eventual consistency is enough: a routing table a few hundred milliseconds out of date sends at most a few requests to the wrong instance, and a retry or a fallback fixes it. Only state that decides correctness, such as who the leader is, needs linearizability.

!!! interview "How to explain it"
    To explain distributed fundamentals: start with why consensus is needed, that multiple replicas have to avoid split brain and disagreement, and consensus is a majority agreeing on the order of a series of operations. The quorum: W + R > N makes reads and writes intersect, a majority configuration tolerates floor(N/2) failures, and an even replica count is poor value so use 3 or 5. Raft: the term is a logical clock for recognising a stale leader, random timeouts stop elections colliding, a majority of votes elects, and only an entry replicated to a majority counts as committed; the two safety constraints (vote only for a candidate whose log is not older, commit only entries from your own term) guarantee committed content is never lost. During a partition only the majority side can write, splitting evenly across two data centres is a trap so use three sites or an arbiter, and the failover time is set by the heartbeat and the election timeout, on the order of a few hundred milliseconds. A lease lets the leader read locally, but a lock can expire quietly, so use a fencing token. Finish on inference: the membership list, the sharding scheme and leader election go in etcd, request state and metrics never do, and multi-rank synchronisation inside an engine uses collective communication rather than consensus.

## Exercises {#练习}

**1. Configure the quorum.** A 5-replica metadata store has to read the latest value at all times and tolerate 2 simultaneous machine failures. What should W and R be? And how else could it be configured for a read-heavy, write-light workload where read latency comes first?

??? success "Answer"
    To tolerate 2 failures and still have writes succeed, W <= 3; to satisfy W + R > 5, take W = 3 and R = 3 (a majority).

    The read-first configuration is W = 5 and R = 1: a read asks one machine, the lowest possible latency; the price is that a write needs every replica, so one failure makes writes unavailable. A compromise is W = 4 and R = 2, where a write tolerates 1 failure and a read asks two. In practice the more common approach is to keep the majority configuration and add a lease so the leader can read locally.

**2. A two-site deployment.** A 3-replica etcd cluster with two machines in site A and one in site B. What happens when the link between A and B drops? And if site A loses power entirely? How would you improve it?

??? success "Answer"
    A and B losing the link: A's side has 2, which is a majority, so it keeps working; B's single machine can only read stale data. That is acceptable.

    Site A losing power entirely: only B's 1 machine is left, which is not a majority, so the cluster cannot be written at all. Service discovery and configuration changes stop and only cached reads work.

    The improvement: deploy across three sites (1:1:1), so any whole site failing still leaves 2; or 2:2:1 with an arbiter replica alone in the third site (it carries no read or write load but provides the deciding vote).

**3. The lock expires.** A hot weight-update task uses an etcd distributed lock to keep one instance updating at a time. The task normally takes 30 seconds, and the lock's lease is 10 seconds with automatic renewal. Write a scenario that goes wrong, and how to prevent it.

??? success "Answer"
    The scenario: the instance holding the lock hits a long garbage-collection pause or a disk stall, fails to renew for more than 10 seconds, the lock expires, and another instance takes it and starts updating; the first instance then comes out of its pause and carries on with the second half, so two instances are writing the same weights.

    The prevention: (1) **a fencing token**, where etcd returns a monotonically increasing version on each acquisition and the actual write target (object storage, a registry) validates it and rejects anything lower than what it has already seen; (2) make the operation idempotent and atomic (write to a temporary location and atomically rename or atomically switch a pointer as the last step); (3) re-check that you still hold the lock before the operation, although this only narrows the window rather than closing it. Renewal alone is not enough.

**4. Whether it belongs in etcd.** Decide where each of these four belongs: (a) each inference instance's current queue depth; (b) the model version currently in effect; (c) which machine a given prefix's KV blocks live on; (d) the ratio of prefill to decode instances under prefill-decode disaggregation.

??? success "Answer"
    (a) Not in etcd. It changes several times a second, can be lost, and only needs to be approximate. Use heartbeat reports or the monitoring system, with the router keeping an approximate view.

    (b) In etcd. Written rarely, has to be agreed by everyone, and a change has to take effect atomically.

    (c) Not in etcd. KV blocks number in the millions with very short lifetimes, so this is high-frequency data. It belongs in dedicated distributed KV storage (located by consistent hashing), or the instances report events and the router keeps an approximate index.

    (d) In etcd. The ratio is cluster-level configuration, changed rarely, but every scheduling component has to see the same version; the actual adjustment is decided by a controller from the monitoring metrics and written back to this configuration.

## Summary {#小结}

- [x] Multiple replicas have to solve split brain and disagreement, and consensus gets a majority to agree on the order of a series of operations.
- [x] W + R > N guarantees reading the latest value; a majority configuration tolerates floor(N/2) failures; an even replica count is poor value, so use 3 or 5.
- [x] Raft: the term recognises a stale leader, random timeouts stop failed elections, a majority of votes elects, and only replication to a majority counts as a commit; two safety constraints keep committed content from being lost.
- [x] During a partition only the majority side can write; splitting evenly across two data centres is a trap, so use three sites or an arbiter; the failover time is set by the heartbeat and the election timeout.
- [x] A lease lets the leader read locally, but a lock can expire quietly, so a long operation needs a fencing token or an idempotent atomic write.
- [x] The membership list, the sharding scheme, model versions and leader election go in a consensus system; request state, KV locations and metrics never do.
