import torch

x = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
w = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
for _ in range(3):                     # 预热：第一次调用有 cuBLAS 初始化、选择算法的开销
    x @ w

start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
start.record()                         # 事件也被放进队列，GPU 执行到这里时记下时间
for _ in range(10):
    y = x @ w
end.record()
end.synchronize()                      # 等 end 事件完成
ms = start.elapsed_time(end) / 10
print(f"每次 {ms:.3f} ms，{2 * 8192**3 / ms / 1e9:.0f} TFLOPS")
