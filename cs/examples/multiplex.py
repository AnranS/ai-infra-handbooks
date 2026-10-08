# 同一条连接上并发发 N 个请求：HTTP/1.1 只能一个接一个（队头阻塞），HTTP/2 可以交错
def http11(reqs, conns):
    """conns 条连接，每条串行处理排给它的请求。reqs 是每个请求的服务端耗时（ms）"""
    queues = [0.0] * conns
    done = []
    for i, cost in enumerate(reqs):
        k = i % conns                          # 轮流放进各条连接
        queues[k] += cost
        done.append(queues[k])
    return done


def http2(reqs):
    """一条连接，所有请求同时开始（服务端并发处理），各自在自己的耗时后完成"""
    return list(reqs)


slow, fast = 500.0, 20.0
reqs = [slow] + [fast] * 5                     # 一个慢请求排在前面，后面五个很快
for name, done in [("HTTP/1.1，1 条连接", http11(reqs, 1)),
                   ("HTTP/1.1，6 条连接", http11(reqs, 6)),
                   ("HTTP/2，1 条连接", http2(reqs))]:
    print(f"{name:20s} 快请求的完成时间：{[round(t) for t in done[1:]]} ms，"
          f"最后一个 {max(done):.0f} ms")
