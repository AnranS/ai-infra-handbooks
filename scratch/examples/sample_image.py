# 采样：从噪声出发，沿速度场积分到 t=1。步数可以少到 20 步还能看
import torch
import torchvision

from unet import UNet

dev = torch.device("cuda")
model = UNet().to(dev)
model.load_state_dict({k.replace("module.", ""): v for k, v in torch.load("ckpt_199.pt")["ema"].items()
                       if not k.startswith("n_averaged")})
model.eval()


@torch.no_grad()
def sample(n=64, steps=50):
    x = torch.randn(n, 3, 32, 32, device=dev)
    for i in range(steps):                                   # 欧拉法；换成 Heun 法同样步数质量更好
        t = torch.full((n,), i / steps, device=dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            v = model(x, t).float()
        x = x + v / steps
    return (x.clamp(-1, 1) + 1) / 2


torchvision.utils.save_image(sample(), "samples.png", nrow=8)
print("已写出 samples.png")
