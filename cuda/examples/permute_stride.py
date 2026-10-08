import torch

x = torch.arange(24).reshape(2, 3, 4)
p = x.permute(2, 0, 1)
print(tuple(p.shape), p.stride(), p.is_contiguous())
v = p.view(4, 6)
print(tuple(v.shape), v.stride(), v.untyped_storage().data_ptr() == x.untyped_storage().data_ptr())
