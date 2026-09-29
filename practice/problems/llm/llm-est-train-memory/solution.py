def model_state_bytes(n_params, zero_stage=0, dp=1):
    if zero_stage == 0:
        return 16 * n_params
    if zero_stage == 1:
        return 4 * n_params + 12 * n_params / dp
    if zero_stage == 2:
        return 2 * n_params + 14 * n_params / dp
    if zero_stage == 3:
        return 16 * n_params / dp
    raise ValueError(f"未知的 ZeRO 阶段：{zero_stage}")


def activation_bytes(seq, batch, hidden, heads, n_layers, flash=True, recompute=False):
    sbh = seq * batch * hidden
    if recompute:
        per_layer = 2 * sbh
    elif flash:
        per_layer = 34 * sbh
    else:
        per_layer = sbh * (34 + 5 * heads * seq / hidden)
    return per_layer * n_layers


def per_gpu_gib(n_params, dp, zero_stage, act_bytes):
    return (model_state_bytes(n_params, zero_stage, dp) + act_bytes) / 2 ** 30
