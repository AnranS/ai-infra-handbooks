# Load balancing and queueing: who should the request go to

<p class="lead">An inference cluster always has a layer of load balancing in front of it. It decides which machine a request goes to, and it decides how bad the p99 looks. This chapter covers three things: how far apart the four common load-balancing algorithms are under a long tail of service times (the p99 in this chapter's simulation differs by nearly a factor of 4), why queueing theory says latency takes off once utilisation passes 80%, and how rate limiting, retries and circuit breaking should work together. It ends on what is particular to inference: requests whose cost differs by tens of times, long-lived connections that defeat layer-4 load balancing, and cache affinity fighting with load balancing.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. Which layer does each of layer-4 and layer-7 load balancing work at? Why can a service with streaming output not rely on layer 4 alone?
    2. Of round robin, random, least connections and power of two choices, which is best when service times have a long tail? Why?
    3. If a service's utilisation goes from 50% to 90%, roughly how much does the average queueing time grow? What effect does the machine count have on this?
    4. When a back end is briefly overloaded, what happens if clients retry? Is exponential backoff with jitter enough?
    5. How is load balancing for inference special compared with an ordinary web service?

??? success "Answers for the self-test (answer first, then open this)"
    1. Layer 4 looks at IP addresses and ports and forwards per connection, understanding nothing of HTTP; layer 7 parses HTTP and can route by path, header or body, and can retry and break circuits. A streaming service's connection lasts tens of seconds to minutes, and layer 4 chooses only once, at connection time, after which every request on that connection goes to the same back end, which easily goes uneven. Routing by request length or cache hit also needs layer 7.
    2. Usually least connections and power of two choices. Round robin and random ignore the back ends' state, so a slow request piles up on one machine: the queueing p99 in this chapter's simulation is 241 ms and 253 ms respectively, against 62 ms for least connections and 140 ms for power of two choices. Least connections is the most accurate but needs centralised state; power of two choices (pick two at random and take the emptier) needs only local information and is the more common choice in a distributed load balancer.
    3. By the M/M/c formula, the average queueing time on one server goes from 10 ms to 90 ms, a factor of 9; on 4 machines from 0.9 ms to 19.7 ms, over 20 times. More machines absorb more: at the same 90% utilisation, 1 machine queues 90 ms and 16 only 3.7 ms. A small pool has to keep more headroom.
    4. Retries add more than another full load exactly when the back end is weakest: the peak in this chapter's simulation goes from 1000/s to 2876/s, while the number of requests that succeed during the outage does not change at all (it is capacity-bound). Exponential backoff with jitter stops retries arriving together, but under sustained overload it still amplifies the total. What actually works is a **retry budget** (retries per second no more than 10% of the normal traffic) plus circuit breaking, which brings the peak back to 1100/s in the simulation.
    5. Three things: request cost varies enormously (prefill lengths differ by tens of times and the output length is unknown), so the connection count is a poor load signal; streaming responses keep connections alive for a long time, so layer-4 load balancing goes uneven; and the prefix cache makes "send it to the instance that handled the same prefix" faster, which conflicts with load balancing and needs a compromise.

A six-panel strip first, then the chapter:

<!-- comic ../assets/comics/load-balance.webp is in Chinese; put it back once the English version exists -->

## Layer 4 and layer 7 {#四层与七层}

| | Layer 4 | Layer 7 |
| --- | --- | --- |
| What it sees | IP addresses, ports | the HTTP method, path, headers, body |
| When it decides | once, at connection time | per request |
| What it can do | forwarding, connection-level health checks | content-based routing, retries, timeouts, rate limiting, circuit breaking, rewriting, observability |
| Cost | very low (can live in the kernel or in hardware) | needs parsing and buffering, a little more latency |
| Typical implementations | LVS/IPVS, a cloud network load balancer, kube-proxy | Nginx, Envoy, HAProxy (in layer-7 mode), the various API gateways |

An inference service usually has both: an outer layer-4 balancer spreads the traffic over a number of gateways, and the gateways do layer-7 routing (authentication, rate limiting, choosing a back-end pool by model name, choosing an instance by cache affinity).

One trap to note: **HTTP/2 and long-lived connections defeat layer-4 load balancing**. A connection picks its back end when it is opened, and the hundreds or thousands of requests on it all go there. If the client opens only one connection, only one back end does any work. The answers are layer-7 load balancing (which distributes per request), or having the client rebuild connections periodically (a maximum connection lifetime, a maximum request count).

## Which machine: four algorithms {#选哪台四种算法}

With identical back ends and a long tail in request cost, how the four common algorithms differ:

```python title="balance.py"
# how four load-balancing policies behave under a long tail of service times: round robin, random, least connections and power of two choices
import heapq
import random


def simulate(policy, n_servers=8, n_reqs=20000, seed=7):
    rng = random.Random(seed)
    busy_until = [0.0] * n_servers        # how long each server is busy until
    inflight = [0] * n_servers            # the number of requests in flight
    finish = []                           # the scheduled completion events (time, server)
    waits = []
    t = 0.0
    for i in range(n_reqs):
        t += rng.expovariate(1 / 1.0)     # requests arrive 1 ms apart on average
        while finish and finish[0][0] <= t:
            _, s = heapq.heappop(finish)
            inflight[s] -= 1
        if policy == "轮询":
            s = i % n_servers
        elif policy == "随机":
            s = rng.randrange(n_servers)
        elif policy == "最少连接":
            s = min(range(n_servers), key=lambda k: (inflight[k], k))
        else:                             # power of two choices: pick two at random and take whichever has fewer in flight
            a, b = rng.randrange(n_servers), rng.randrange(n_servers)
            s = a if inflight[a] <= inflight[b] else b
        service = rng.expovariate(1 / 4.0) if rng.random() > 0.05 else rng.expovariate(1 / 40.0)
        start = max(t, busy_until[s])     # this machine handles one request at a time
        busy_until[s] = start + service
        inflight[s] += 1
        heapq.heappush(finish, (busy_until[s], s))
        waits.append(start - t)           # the time spent queueing
    waits.sort()
    return waits[len(waits) // 2], waits[int(len(waits) * 0.99)]


print("到达 1000 请求/秒，8 台服务器，平均服务 5.8 ms（5% 的慢请求平均 40 ms），利用率约 72%\n")
print("策略      排队 p50   排队 p99")
for policy in ["轮询", "随机", "最少连接", "二选一"]:
    p50, p99 = simulate(policy)
    print(f"{policy:6s} {p50:7.2f} ms {p99:7.1f} ms")
```

```text title="output"
到达 1000 请求/秒，8 台服务器，平均服务 5.8 ms（5% 的慢请求平均 40 ms），利用率约 72%

策略      排队 p50   排队 p99
轮询        3.48 ms   241.3 ms
随机        9.97 ms   253.4 ms
最少连接      0.00 ms    62.5 ms
二选一       0.99 ms   139.6 ms
```

- **Round robin** and **random** ignore the back ends' state. When one machine meets a slow request, every request dealt to it afterwards waits behind it, and the p99 is the worst.
- **Least connections** sends the request to the machine currently handling the fewest, which confines a long-tail request's effect to itself, and the p99 is the best. The price is that the balancer has to keep live state for every back end; with several balancer instances each keeping their own, they fight because what they see differs (all of them think the same machine is emptiest and rush it together).
- **Power of two choices**: pick two at random and take whichever is handling fewer. It needs only local information, several balancer instances will not overload in lockstep, and the result is close to least connections. This is the default or recommended policy in Envoy, gRPC and others.

Two variants are common in practice as well: **weighting** (distributing by weight when back ends differ in capacity, or during a canary rollout) and **EWMA latency** (choosing by a moving average of recent response times, which reflects a back end's real health).

## Queueing theory: utilisation and latency {#排队论利用率与延迟}

Why is the p99 so bad when average utilisation is only 80%? Because arrivals are random, and queueing time grows non-linearly with utilisation.

Drag the utilisation and watch the queueing time grow:

<div class="aig-widget" data-widget="mm1-latency"></div>

```python title="queueing.py"
# queueing theory: the closer utilisation gets to 1, the faster the queueing time grows. M/M/c's waiting probability comes from the Erlang C formula
from math import factorial


def erlang_c(c, rho):
    """c 台服务器、利用率 rho 时，一个请求到达就要排队的概率"""
    a = c * rho                                        # the arrival rate / one server's service rate
    top = a ** c / factorial(c) / (1 - rho)
    bottom = sum(a ** k / factorial(k) for k in range(c)) + top
    return top / bottom


def wait_ms(c, rho, service_ms):
    """平均排队时间 = 排队概率 × 服务时间 / (c × (1 - 利用率))"""
    return erlang_c(c, rho) * service_ms / (c * (1 - rho))


print("平均服务时间 10 ms 时，不同利用率下的平均排队时间：")
print("利用率   1 台      4 台      16 台")
for rho in (0.5, 0.7, 0.8, 0.9, 0.95, 0.99):
    row = "".join(f"{wait_ms(c, rho, 10):8.1f} ms" for c in (1, 4, 16))
    print(f"{rho:5.0%} {row}")
print()
print("同样 90% 的利用率，机器越多排队越短（小池子更怕长尾）：")
for c in (1, 2, 4, 8, 16, 32):
    print(f"  {c:2d} 台：排队 {wait_ms(c, 0.9, 10):6.2f} ms，排队概率 {erlang_c(c, 0.9):4.0%}")
print()
print("要把排队压到 5 ms 以内，利用率最高能到多少：")
for c in (1, 4, 16, 64):
    hi = max((r for r in (i / 1000 for i in range(1, 1000)) if wait_ms(c, r, 10) <= 5), default=0)
    print(f"  {c:2d} 台：{hi:.1%}")
```

```text title="output"
平均服务时间 10 ms 时，不同利用率下的平均排队时间：
利用率   1 台      4 台      16 台
  50%     10.0 ms     0.9 ms     0.0 ms
  70%     23.3 ms     3.6 ms     0.3 ms
  80%     40.0 ms     7.5 ms     1.0 ms
  90%     90.0 ms    19.7 ms     3.7 ms
  95%    190.0 ms    44.6 ms     9.8 ms
  99%    990.0 ms   244.5 ms    59.6 ms

同样 90% 的利用率，机器越多排队越短（小池子更怕长尾）：
   1 台：排队  90.00 ms，排队概率  90%
   2 台：排队  42.63 ms，排队概率  85%
   4 台：排队  19.69 ms，排队概率  79%
   8 台：排队   8.77 ms，排队概率  70%
  16 台：排队   3.70 ms，排队概率  59%
  32 台：排队   1.43 ms，排队概率  46%

要把排队压到 5 ms 以内，利用率最高能到多少：
   1 台：33.3%
   4 台：74.7%
  16 台：91.8%
  64 台：97.5%
```

Three conclusions to remember:

- **Past 80% utilisation, queueing time takes off**. From 50% to 90%, one machine's average queue goes from 10 ms to 90 ms; at 99% it is 990 ms. Capacity planning sets the utilisation from the queueing latency you can accept, not from a wish to keep the machines busy.
- **A bigger pool absorbs more**. At the same 90% utilisation, 1 machine queues 90 ms and 16 only 3.7 ms. So a small cluster has to keep more headroom, which is also the argument for one large pool over several small ones (sharding breaks the pool up).
- **And this is only the average**. The p99 is worse, and the larger the variance of the service time (inference request lengths vary enormously) the longer the distribution's tail.

For an inference service, be clear about what the service time even is: a request does not occupy a whole machine but a slot in a continuous batch shared with others. The engine's internal queue (waiting to be scheduled into a batch) and the external queue (waiting to be assigned to an instance) are two layers, and [Load testing and capacity planning](serving://perf/benchmark/) covers measuring them together.

## Rate limiting, retries and circuit breaking {#限流重试与熔断}

**Rate limiting** keeps the requests beyond your capacity outside the door instead of making everyone slower together. Two common algorithms:

- **The token bucket**: tokens are added at a fixed rate and each request takes one; the bucket has a capacity, so a certain amount of burst is allowed. Use this in most situations.
- **The leaky bucket**: requests flow out at a fixed rate, perfectly smoothed, with no burst allowed.

Rate limiting for inference has to count in the right unit: requests per second is very inaccurate (one request may be 100 tokens or 100,000), while **token count** or **concurrency** is more sensible, usually both together (a concurrency cap protects the device memory, a token rate protects the throughput).

**Retries** need great care. How much extra load each retry policy puts on a back end whose capacity dips briefly:

```python title="retry.py"
# how much a retry policy amplifies the load when a back end briefly slows down
from collections import defaultdict

RPS, CAPACITY, SECONDS, OUTAGE = 1000, 1200, 60, range(10, 20)
MAX_ATTEMPTS, BUDGET = 3, 0.1              # at most 2 retries; the retry budget: retries per second no more than 10% of the normal traffic


def simulate(policy):
    """返回（峰值负载, 总请求数, 故障期间成功的请求数, 恢复到正常负载用了几秒）"""
    queued = defaultdict(lambda: defaultdict(int))     # second -> {which attempt: request count}
    peak = total = served_in_outage = 0
    back_to_normal = None
    for sec in range(SECONDS):
        arrivals = dict(queued.pop(sec, {}))
        retries = sum(arrivals.values())
        if policy.endswith("重试预算"):                # retries beyond the budget are dropped (fail fast)
            allowed = int(RPS * BUDGET)
            for attempt in sorted(arrivals, reverse=True):
                take = min(arrivals[attempt], allowed)
                arrivals[attempt], allowed = take, allowed - take
            retries = sum(arrivals.values())
        arrivals[0] = arrivals.get(0, 0) + RPS         # the requests newly arriving this second
        load = sum(arrivals.values())
        cap = CAPACITY if sec not in OUTAGE else CAPACITY // 10
        peak, total = max(peak, load), total + load
        if sec in OUTAGE:
            served_in_outage += min(load, cap)
        elif sec > OUTAGE[-1] and back_to_normal is None and load <= RPS * 1.05:
            back_to_normal = sec - OUTAGE[-1]
        if load <= cap or policy == "不重试":
            continue
        share = (load - cap) / load                    # spread the failures proportionally
        for attempt, n in list(arrivals.items()):
            failed = int(n * share)
            if not failed or attempt + 1 >= MAX_ATTEMPTS:
                continue
            if policy == "立即重试":
                queued[sec + 1][attempt + 1] += failed
            else:                                      # exponential backoff with jitter: wait 2^(k+1) seconds, spread flat across that window
                window = 2 ** (attempt + 1)
                for d in range(window):
                    queued[sec + window + d][attempt + 1] += failed // window
    return peak, total, served_in_outage, back_to_normal


print(f"正常 {RPS} 请求/秒、容量 {CAPACITY}，第 {OUTAGE[0]}～{OUTAGE[-1] + 1} 秒容量掉到 {CAPACITY // 10}：")
print("策略                 峰值负载  打到后端的总量  故障期间成功  恢复用时")
for policy in ["不重试", "立即重试", "指数退避 + 抖动", "退避 + 抖动 + 重试预算"]:
    peak, total, served, back = simulate(policy)
    print(f"{policy:20s} {peak:6d}/s {total:11d} {served:12d} {back if back is not None else '-':>7} 秒")
```

```text title="output"
正常 1000 请求/秒、容量 1200，第 10～20 秒容量掉到 120：
策略                 峰值负载  打到后端的总量  故障期间成功  恢复用时
不重试                    1000/s       60000         1200       1 秒
立即重试                   2876/s       80068         1200       6 秒
指数退避 + 抖动              2820/s       84556         1200      20 秒
退避 + 抖动 + 重试预算         1100/s       61170         1200       4 秒
```

- Retrying immediately takes the peak load from 1000/s to 2876/s, putting three times the pressure on the back end exactly when it is weakest.
- Exponential backoff with jitter stops the retries arriving together, but under sustained overload it only pushes the load back: the total is larger and the recovery slower.
- **A retry budget** is what actually works: cap retries per second at a small fraction of the normal traffic (10% here) and fail the rest fast. The peak comes back to 1100/s and the recovery is quick.
- The most important row: **the number of requests that succeed during the outage is exactly the same under all four policies**. The back end handles what it handles, and retries do not conjure capacity out of nothing.

**Circuit breaking** complements retries: when one back end's consecutive failures reach a threshold, stop sending to it for a while, then let a few requests through to probe. It avoids sending to a back end you know will fail, and it leaves the faulty instance room to recover.

Use the three together, and **retry only at the outermost layer**. Three retries at every layer is a 27-fold amplification over three layers.

## Health checks and graceful shutdown {#健康检查与优雅退出}

The load balancer has to know which instances are usable:

- **A liveness probe** (is the process still there) and **a readiness probe** (can it take new requests) are separate things. An inference instance takes tens of seconds to minutes to load its weights, and during that time the process is alive but cannot serve. Get the readiness probe wrong and traffic lands on an instance that has not finished loading (see [Production deployment and operations](serving://ops/deploy/)).
- **A readiness probe can also reflect overload**: report not-ready when the queue is too long or device memory is tight, so the balancer gives the traffic to someone else. But guard against flapping, where every instance takes turns reporting not-ready.
- **Graceful shutdown**: on a termination signal, first come out of the load balancer (let the readiness probe fail), then finish the requests in hand before exiting. An inference request may run for minutes, so the termination grace period has to be set from the longest answer, or users will see their answers cut off.

## What is particular to inference {#推理服务的特殊之处}

| Trait | Consequence | Common practice |
| --- | --- | --- |
| Request cost differs by tens of times (prefill length, unknown output length) | the connection count is a poor load signal | use pending token count, or the queue length the engine reports, as the load signal |
| The response is a long stream | layer-4 load balancing per connection goes badly uneven | use layer 7; cap the connection lifetime |
| The prefix cache | sending the same prefix to the same instance saves a whole prefill | cache-aware routing (consistent hashing with a load cap), see [The prefix cache](serving://engine/prefix-cache/) |
| Prefill-decode disaggregation | prefill and decode are two pools, with KV transfer to account for | a global scheduler, see [Global scheduling for a disaggregated architecture](serving://frontier/disagg-sched/) |
| The queueing happens inside the engine | the response time the balancer sees includes the queueing | have the engine report its queue depth and consult it when routing |

Cache affinity and load balancing are a contradictory pair: send everything to the instance with the highest hit rate and it overloads. The usual compromise is affinity with a load cap: prefer the instance with the high hit rate, but switch when its queue passes a threshold. The consistent hashing in the next chapter is the basic tool for building that affinity.

!!! interview "How to explain it"
    To explain load balancing: start by separating layer 4 (per connection, IP and port, low cost) from layer 7 (per request, sees the content, can retry, rate limit and break circuits), and point out that long-lived connections and HTTP/2 defeat layer 4. On algorithms: round robin and random ignore the back ends' state and a long-tail request drags the p99 down; least connections is the most accurate but needs centralised state; power of two choices needs only local information for a comparable result, which is the common choice for distributed load balancing. Then use queueing theory to explain capacity planning: past 80% utilisation, queueing time takes off, and a bigger pool absorbs more (at the same 90% utilisation, 1 machine queues 90 ms and 16 queue 3.7 ms). The three overload protections: rate limiting (counted in tokens or concurrency for inference, not QPS), retries (a retry budget is mandatory and only the outermost layer retries, otherwise the peak multiplies while the success count does not move at all) and circuit breaking. Finish with what is particular to inference: request costs differing by tens of times, so pending token count is the load signal; prefix-cache affinity to be traded off against balancing; and prefill-decode disaggregation needing a global scheduler.

## Exercises {#练习}

**1. Choosing the signal.** An inference cluster picks the instance with the fewest current connections. One instance is handling 2 requests with very long contexts (100K tokens each), another is handling 8 short requests. Which is busier? What signal should be used?

??? success "Answer"
    Almost certainly the first. Prefill compute is proportional to the token count, so 2 requests of 100K are 200K tokens of prefill while 8 short ones may be only a few thousand; the KV cache occupancy of the first is tens of times larger too.

    Better signals: the pending token count the engine reports (queued prefill tokens plus the context lengths of running requests), the KV cache utilisation, and a moving average of the recent time to first token. Many gateways have each instance expose its queue depth and KV utilisation through a metrics endpoint, and the router scores by those values.

**2. Capacity and latency.** A service has an average processing time of 50 ms and requires a p50 queueing time of at most 10 ms. By this chapter's M/M/c formula, what is the highest utilisation with 4 machines? With 16? If the business doubles, would you add machines or allow the utilisation to rise?

??? success "Answer"
    Change `service_ms` in this chapter's program to 50 and the target to 10 ms, then run it again: the utilisation ceiling rises with the machine count (about 55% at 4 machines and about 80% at 16; take the program's output as the exact figure).

    On doubling, keeping the machine count means the utilisation doubles toward 1 and the queueing time grows several-fold or diverges, so add machines. Adding machines has a bonus: a larger pool queues less at the same utilisation, so the target utilisation can go up a little as well, and the cost does not grow linearly.

**3. The retry account.** A gateway is configured to retry a failing inference instance twice. During an incident, a batch of instances starts erroring because they are out of device memory. Draw what happens next, and give three improvements.

??? success "Answer"
    The chain reaction: errors, the gateway retries, the load on the healthy instances triples, their queues lengthen and the time to first token times out, the timeouts trigger retries too, more instances are dragged down, and it avalanches.

    Improvements: (1) add a retry budget, say retries capped at 10% of the total, returning an error beyond that; (2) retry only clearly retryable errors (a connection failure, a 503) and not timeouts (the request may already be generating, and retrying doubles the GPU it occupies); (3) add circuit breaking so an instance that keeps failing is taken out for a while; (4) combine with rate limiting so requests beyond capacity are rejected at the entrance instead of queueing inside; (5) make sure only the outermost layer retries.

**4. Affinity against balance.** In a multi-turn conversation service, sending a session's later requests to the same instance hits the prefix cache and saves a few hundred milliseconds of prefill. But a popular session overloads one instance. Design a routing policy and say how it behaves when instances are scaled up or down.

??? success "Answer"
    Use consistent hashing to map the session id to an instance (the next chapter's subject), which gives "the same session goes to the same machine by default"; then add a load cap: if the target instance's queue depth or KV utilisation passes a threshold, move along the hash ring to the next one (consistent hashing with bounded loads), and record the reassignment so later requests also go to the new machine until the session ends.

    On a scale-up or scale-down, consistent hashing changes the ownership of only a small fraction of sessions (about 1/N); those sessions lose their prefix cache and pay one extra prefill on their next request, but not all of them are invalidated. Virtual nodes make the migration more even. Also add hysteresis to the reassignment, so it does not flap around the threshold.

## Summary {#小结}

- [x] Layer 4 forwards per connection at low cost, layer 7 routes per request and can retry, rate limit and break circuits; long-lived connections and HTTP/2 make layer 4 badly uneven.
- [x] Under a long tail of service times, least connections and power of two choices clearly beat round robin and random; power of two choices needs only local information, which suits a distributed balancer.
- [x] Queueing time grows non-linearly with utilisation and takes off past 80%; a bigger pool absorbs more, and capacity planning sets the utilisation from the queueing latency you can accept.
- [x] The three overload protections: rate limiting (counted in tokens or concurrency for inference), retries (a budget is mandatory, and only at the outermost layer) and circuit breaking; retries do not conjure capacity.
- [x] Keep liveness and readiness probes separate, and set the graceful-shutdown grace period from the longest answer.
- [x] An inference service uses pending token count as its load signal; cache affinity and load balancing need a compromise (consistent hashing with a load cap).
