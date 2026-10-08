# 一个请求在网络上要花几个往返？（RTT 按跨城 30 ms 算）
rtt = 30
steps = [
    (1, "DNS 查询", "有缓存时省掉"),
    (1, "TCP 三次握手", "连接复用时省掉"),
    (1, "TLS 1.3 握手", "会话复用或 0-RTT 时省掉；TLS 1.2 要 2 个往返"),
    (1, "发请求、等第一个响应字节", "这一段里还包含服务端的处理时间"),
]
print(f"第一次请求（RTT = {rtt} ms）：")
for n, name, note in steps:
    print(f"  {n * rtt:3d} ms  {name}（{note}）")
print(f"  {sum(n for n, _, _ in steps) * rtt:3d} ms  合计，还没算服务端生成第一个 token 的时间")
print(f"\n连接复用 + DNS 缓存之后只剩最后一个往返：{rtt} ms\n")
print("推理服务的 TTFT 里网络占多少（服务端按 200 ms 算）：")
for name, r in [("同机房", 0.5), ("同城", 5), ("跨城", 30), ("跨洲", 200)]:
    print(f"  {name:4s}RTT {r:5.1f} ms：新建连接要 {4 * r:6.1f} ms，复用连接 {r:5.1f} ms，"
          f"网络占总 TTFT 的 {r / (r + 200):4.1%}")
