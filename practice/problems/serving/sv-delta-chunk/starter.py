import numpy as np


def delta_recurrent(q, k, v, beta, S0=None):
    """逐 token 递推（decode 的算法）：S_t = S_{t-1} + β_t k_tᵀ (v_t - k_t S_{t-1})，o_t = q_t S_t"""
    S = np.zeros((k.shape[1], v.shape[1])) if S0 is None else S0.copy()
    out = []
    for t in range(len(q)):
        S = S + beta[t] * np.outer(k[t], v[t] - k[t] @ S)
        out.append(q[t] @ S)
    return np.array(out), S


def ut_inverse(k, beta):
    """一个块的 T = (I + diag(β)·strict_lower(K Kᵀ))⁻¹"""
    pass


def delta_chunked(q, k, v, beta, C, S0=None):
    """分块算法：块内用矩阵乘，块间传递状态"""
    S = np.zeros((k.shape[1], v.shape[1])) if S0 is None else S0.copy()
    out = []
    for s in range(0, len(q), C):
        qc, kc, vc, bc = q[s:s + C], k[s:s + C], v[s:s + C], beta[s:s + C]
        U = bc[:, None] * vc                                   # 当成普通的线性注意力：忽略了纠错项
        out.append(qc @ S + np.tril(qc @ kc.T) @ U)
        S = S + kc.T @ U
    return np.concatenate(out), S
