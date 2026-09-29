import numpy as np


def lora_delta(x, adapter_ids, A, B, scale):
    out = np.zeros((x.shape[0], B[0].shape[1]))
    for t, a in enumerate(adapter_ids):                    # 逐 token 计算，而且忘了 -1 的情况
        out[t] = scale[a] * (x[t] @ A[a]) @ B[a]
    return out


def schedule(requests, max_loras, gpu_slots, max_batch):
    pass
