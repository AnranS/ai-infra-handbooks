import torch

model = torch.nn.Sequential(torch.nn.Linear(4096, 11008), torch.nn.SiLU(), torch.nn.Linear(11008, 4096)).cuda().half()
static_in = torch.zeros(8, 4096, device="cuda", dtype=torch.half)   # 录制和重放都用这块固定的显存

side = torch.cuda.Stream()                    # 在非默认 stream 上预热，让分配器、cuBLAS 完成初始化
side.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(side), torch.inference_mode():
    for _ in range(3):
        model(static_in)
torch.cuda.current_stream().wait_stream(side)

g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g), torch.inference_mode():
    static_out = model(static_in)             # 只录制，不执行

for step in range(5):
    new_input = torch.randn(8, 4096, device="cuda", dtype=torch.half)
    static_in.copy_(new_input)                # 把新输入拷进固定的地址
    g.replay()                                # 一次提交整张图
    token_logits = static_out.clone()         # 结果在固定的输出地址上
