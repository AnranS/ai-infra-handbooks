"""nn.Module：参数怎么注册、buffer 是什么、state_dict 长什么样、train 和 eval 有什么区别"""
import torch
import torch.nn as nn

torch.manual_seed(0)


class Block(nn.Module):
    def __init__(self, d_in, d_out, p=0.5):
        super().__init__()                                    # 忘了这一句，后面注册参数会直接报错
        self.fc = nn.Linear(d_in, d_out)
        self.norm = nn.LayerNorm(d_out)
        self.drop = nn.Dropout(p)
        self.scale = nn.Parameter(torch.ones(d_out))          # 自己加的可训练参数
        self.register_buffer("calls", torch.zeros(1))         # 不训练但要随 state_dict 存取的状态

    def forward(self, x):
        self.calls += 1
        return self.drop(self.norm(self.fc(x)) * self.scale)


class Net(nn.Module):
    def __init__(self, d=4, n_block=2):
        super().__init__()
        self.blocks = nn.ModuleList(Block(d, d) for _ in range(n_block))   # 用 ModuleList，不要用 []
        self.head = nn.Linear(d, 2)

    def forward(self, x):
        for b in self.blocks:
            x = x + b(x)                                      # 残差
        return self.head(x)


net = Net()
print("—— 参数 ——")
total = sum(p.numel() for p in net.parameters())
print(f"参数总数 {total}，可训练的 {sum(p.numel() for p in net.parameters() if p.requires_grad)}")
for name, p in list(net.named_parameters())[:4]:
    print(f"  {name:24s} {tuple(p.shape)}")
print("  ……")
print("buffer 不在 parameters() 里，但在 state_dict 里：", [n for n, _ in net.named_buffers()])

print("\n—— 放进 list 就注册不上了 ——")
class Broken(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [nn.Linear(2, 2) for _ in range(2)]     # 错误示范

print("Broken 的参数个数：", sum(p.numel() for p in Broken().parameters()), "——优化器什么也拿不到，模型也不会跟着 .to(device) 搬")

print("\n—— state_dict ——")
sd = net.state_dict()
print("键的数量：", len(sd), "，前三个：", list(sd)[:3])
print("保存和加载就是它：torch.save(net.state_dict(), ...) / net.load_state_dict(...)")
print("load_state_dict 的返回值会告诉你缺了哪些、多了哪些：", net.load_state_dict(sd, strict=True))

print("\n—— 初始化 ——")
def init(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, std=0.02)
        nn.init.zeros_(m.bias)

net.apply(init)                                               # apply 会递归作用到每一个子模块
print("apply 之后第一层权重的标准差：", f"{net.blocks[0].fc.weight.std().item():.4f}")

print("\n—— train 和 eval ——")
x = torch.randn(3, 4)
net.train()
out_train = torch.stack([net(x) for _ in range(2)])
net.eval()
out_eval = torch.stack([net(x) for _ in range(2)])
print("train 模式下两次前向不同（dropout 在随机丢）：", not torch.allclose(out_train[0], out_train[1]))
print("eval 模式下两次前向相同：", torch.allclose(out_eval[0], out_eval[1]))
print("eval() 只影响 dropout、batchnorm 这类有「训练/推理两套行为」的层，它**不会**关掉梯度——那是 no_grad 的事")
print("buffer 记下来的调用次数：", net.blocks[0].calls.item())
