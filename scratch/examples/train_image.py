# CIFAR-10 上训一个流匹配图像生成模型。和 flow2d.py 的训练目标一字不差，只是换了网络和数据。
import time

import torch
import torchvision
from torch.utils.data import DataLoader

from unet import UNet

dev = torch.device("cuda")
torch.set_float32_matmul_precision("high")

tf = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.5] * 3, [0.5] * 3),          # 归一化到 [-1, 1]
])
ds = torchvision.datasets.CIFAR10("./data", train=True, download=True, transform=tf)
dl = DataLoader(ds, batch_size=256, shuffle=True, num_workers=8, pin_memory=True,
                drop_last=True, persistent_workers=True)

model = UNet().to(dev).to(memory_format=torch.channels_last)
model = torch.compile(model)
ema = torch.optim.swa_utils.AveragedModel(model, avg_fn=lambda a, b, _: 0.999 * a + 0.001 * b)
opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.0, fused=True)

EPOCHS = 200
for epoch in range(EPOCHS):
    t0, total, n = time.perf_counter(), 0.0, 0
    for x1, _ in dl:
        x1 = x1.to(dev, non_blocking=True).to(memory_format=torch.channels_last)
        x0 = torch.randn_like(x1)
        t = torch.rand(x1.shape[0], device=dev)
        xt = (1 - t[:, None, None, None]) * x0 + t[:, None, None, None] * x1
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = ((model(xt, t) - (x1 - x0)) ** 2).mean()          # 和二维版本完全一样的一行
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        ema.update_parameters(model)
        total, n = total + loss.item(), n + 1
    print(f"epoch {epoch:>3} loss {total / n:.4f} {time.perf_counter() - t0:.0f}s "
          f"显存 {torch.cuda.max_memory_allocated() / 1024 ** 3:.1f} GB")
    if epoch % 20 == 19:
        torch.save({"model": model.state_dict(), "ema": ema.state_dict()}, f"ckpt_{epoch}.pt")
