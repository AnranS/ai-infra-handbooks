import torch

compute = torch.cuda.current_stream()
copy = torch.cuda.Stream()
w = torch.randn(4096, 4096, device="cuda")
batches = [torch.randn(4096, 4096).pin_memory() for _ in range(4)]   # 锁页内存：才能真正异步地拷贝

next_gpu = batches[0].to("cuda", non_blocking=True)
for i in range(len(batches)):
    cur = next_gpu
    if i + 1 < len(batches):
        with torch.cuda.stream(copy):                   # 在拷贝 stream 上预取下一批
            next_gpu = batches[i + 1].to("cuda", non_blocking=True)
    y = cur @ w                                         # 在计算 stream 上算这一批
    compute.wait_stream(copy)                           # 下一轮用 next_gpu 之前，等它拷完
    next_gpu.record_stream(compute)                     # 告诉分配器：这块显存还会被计算 stream 用
torch.cuda.synchronize()
