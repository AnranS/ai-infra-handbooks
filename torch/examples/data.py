"""Dataset、DataLoader 和 collate_fn：变长样本怎么拼成一个 batch"""
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


class ToyText(Dataset):
    """一个最小的 Dataset：实现 __len__ 和 __getitem__ 就够了"""

    def __init__(self, n=8, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.lens = torch.randint(2, 7, (n,), generator=g).tolist()     # 每条样本长度不同
        self.g = g

    def __len__(self):
        return len(self.lens)

    def __getitem__(self, i):
        return torch.arange(self.lens[i]) + 10 * i, self.lens[i] % 2    # (序列, 标签)


ds = ToyText()
print("样本长度：", ds.lens)
print("第 0 条：", ds[0])

print("\n—— 默认的 collate 要求形状一致 ——")
try:
    next(iter(DataLoader(ds, batch_size=3)))
except RuntimeError as e:
    print("直接用会报错：", str(e).split(".")[0])


def collate(batch):
    """自己拼：补齐到本 batch 里最长的那条，并给出 mask"""
    seqs, labels = zip(*batch)
    padded = pad_sequence(seqs, batch_first=True, padding_value=0)
    mask = padded.new_zeros(padded.shape, dtype=torch.bool)
    for i, s in enumerate(seqs):
        mask[i, :len(s)] = True
    return padded, mask, torch.tensor(labels)


loader = DataLoader(ds, batch_size=3, shuffle=False, collate_fn=collate)
for step, (x, mask, y) in enumerate(loader):
    print(f"batch {step}: x{tuple(x.shape)} 真实 token 数 {mask.sum().item():2d}/{mask.numel():2d}  标签 {y.tolist()}")
print("补齐到**本 batch** 的最长，而不是全局最长——所以按长度排序再分桶能少算很多 padding")

print("\n—— shuffle 与可复现 ——")
a = [b[2].tolist() for b in DataLoader(ds, batch_size=3, shuffle=True,
                                       generator=torch.Generator().manual_seed(7), collate_fn=collate)]
b = [b[2].tolist() for b in DataLoader(ds, batch_size=3, shuffle=True,
                                       generator=torch.Generator().manual_seed(7), collate_fn=collate)]
print("给 DataLoader 一个 generator，两次 shuffle 的顺序一样：", a == b, a)
print("num_workers>0 时每个 worker 还要单独设种子（worker_init_fn），否则各进程的数据增强会撞上")
print("drop_last=True 丢掉最后不满的一批；分布式训练里常开，省得各卡步数对不齐")
