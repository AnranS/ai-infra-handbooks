import torch

torch.cuda.memory._record_memory_history(max_entries=100_000)   # 记录每次分配和释放的调用栈
torch.cuda.reset_peak_memory_stats()

x = torch.randn(4096, 4096, device="cuda")
y = x @ x
del x

print(f"allocated {torch.cuda.memory_allocated() / 2**20:.0f} MiB，"
      f"reserved {torch.cuda.memory_reserved() / 2**20:.0f} MiB，"
      f"峰值 {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB")
torch.cuda.memory._dump_snapshot("mem_snapshot.pickle")        # 拖到 https://pytorch.org/memory_viz 里查看时间线
