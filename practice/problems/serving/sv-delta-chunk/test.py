import numpy as np

from checker import check, check_close
from solution import delta_chunked, delta_recurrent, ut_inverse


def data(seed, T, dk=8, dv=6):
    rng = np.random.default_rng(seed)
    k = rng.standard_normal((T, dk))
    k /= np.linalg.norm(k, axis=1, keepdims=True)              # 真实模型里 k 做了 L2 归一化
    q = rng.standard_normal((T, dk)) / np.sqrt(dk)
    v = rng.standard_normal((T, dv))
    beta = 1 / (1 + np.exp(-rng.standard_normal(T)))           # β = sigmoid(·) ∈ (0, 1)
    return q, k, v, beta


def test_example():
    q, k, v, beta = data(0, 32)
    o1, s1 = delta_recurrent(q, k, v, beta)
    o2, s2 = delta_chunked(q, k, v, beta, C=8)
    check_close(o2, o1, rtol=1e-9, atol=1e-10, what="输出与逐 token 递推一致")
    check_close(s2, s1, rtol=1e-9, atol=1e-10, what="最终状态一致")


def test_ut_inverse():
    q, k, v, beta = data(1, 10)
    T = ut_inverse(k, beta)
    check(T.shape, (10, 10), "T 的形状")
    L = np.tril(k @ k.T, -1)
    check_close(T, np.linalg.inv(np.eye(10) + beta[:, None] * L), rtol=1e-9, atol=1e-10, what="T = (I + diag(β) L)⁻¹")
    check(bool(np.allclose(np.triu(T, 1), 0) and np.allclose(np.diag(T), 1)), True, "T 是对角线为 1 的下三角矩阵")


def test_chunk_sizes():
    q, k, v, beta = data(2, 37)                                # 37 不是块大小的整数倍
    o1, s1 = delta_recurrent(q, k, v, beta)
    for C in (1, 2, 5, 16, 37, 64):
        o2, s2 = delta_chunked(q, k, v, beta, C=C)
        check(o2.shape, o1.shape, f"C={C} 时输出的形状")
        check_close(o2, o1, rtol=1e-9, atol=1e-10, what=f"C={C} 时的输出")
        check_close(s2, s1, rtol=1e-9, atol=1e-10, what=f"C={C} 时的最终状态")


def test_initial_state_and_split_prefill():
    q, k, v, beta = data(3, 48, dk=16, dv=16)
    S0 = np.random.default_rng(4).standard_normal((16, 16))
    o1, s1 = delta_recurrent(q, k, v, beta, S0)
    o2, s2 = delta_chunked(q, k, v, beta, C=16, S0=S0)
    check_close(o2, o1, rtol=1e-9, atol=1e-9, what="带初始状态")
    check_close(s2, s1, rtol=1e-9, atol=1e-9, what="带初始状态时的最终状态")
    oa, sa = delta_chunked(q[:20], k[:20], v[:20], beta[:20], C=16, S0=S0)     # 分块 prefill：前 20 个 token 一段
    ob, sb = delta_chunked(q[20:], k[20:], v[20:], beta[20:], C=16, S0=sa)     # 后 28 个接着上一段的状态
    check_close(np.concatenate([oa, ob]), o1, rtol=1e-9, atol=1e-9, what="切成两段、把状态交给下一段")
    check_close(sb, s1, rtol=1e-9, atol=1e-9, what="两段之后的状态")


def test_erases_old_memory():
    # β = 1 且 k 是单位向量时，写入之后 k S 应该正好等于 v：旧的记忆被完全替换
    dk, dv = 4, 3
    k = np.tile(np.eye(dk)[0], (3, 1))
    v = np.array([[1.0, 2.0, 3.0], [-5.0, 0.0, 5.0], [7.0, 7.0, 7.0]])
    q = k.copy()
    o, S = delta_chunked(q, k, v, np.ones(3), C=3)
    check_close(o, v, rtol=1e-12, atol=1e-12, what="同一个 k 反复写入：每次读出的都是最新的 v")
