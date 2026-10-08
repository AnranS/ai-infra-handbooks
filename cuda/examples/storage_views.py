import torch

x = torch.arange(12, dtype=torch.float32).reshape(3, 4)
print("原张量：shape", tuple(x.shape), "stride", x.stride(), "连续", x.is_contiguous())

t = x.t()                                  # 转置：只是交换了 shape 和 stride
print("转置：shape", tuple(t.shape), "stride", t.stride(), "连续", t.is_contiguous())
print("共享存储：", t.untyped_storage().data_ptr() == x.untyped_storage().data_ptr())

s = x[:, 1:3]                              # 切片：换了 offset 和 shape
print("切片：shape", tuple(s.shape), "stride", s.stride(), "offset", s.storage_offset(), "连续", s.is_contiguous())

t[0, 1] = 100.0                            # 通过视图写，原张量也变了
print("通过视图修改后 x[1, 0] =", x[1, 0].item())

try:
    t.view(12)
except RuntimeError as e:
    print("view 失败：", str(e).split(" (")[0])
r = t.reshape(12)                          # reshape：能做视图就做视图，做不了就拷贝
print("reshape 拷贝了：", r.untyped_storage().data_ptr() != x.untyped_storage().data_ptr())
print("contiguous() 之后 stride", t.contiguous().stride())
