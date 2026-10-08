import numpy as np

import kvpool_py as kp

pool = kp.BlockPool(8)
blocks = pool.allocate(3)
print("分配：", blocks, pool)
print("不够时：", pool.allocate(10))
for b in blocks:
    pool.release(b)
try:
    pool.release(blocks[0])
except RuntimeError as e:                 # std::logic_error 被翻译成 RuntimeError
    print("重复释放：", type(e).__name__, e)
print("公共前缀：", kp.common_prefix([1, 2, 3, 4], [1, 2, 9]))
print("EOS 个数：", kp.count_eos(np.array([5, 2, 7, 2, 2], dtype=np.int32), eos=2))
