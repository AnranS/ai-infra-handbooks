import math


def colocated_s(weight_bytes, gpus_per_inst, nvlink_bw):
    return weight_bytes / gpus_per_inst / nvlink_bw


def disaggregated_s(weight_bytes, n_inst, nic_bw, gpus_per_inst, mode):
    if mode == "naive":
        return n_inst * weight_bytes / nic_bw
    if mode == "parallel":
        return n_inst * weight_bytes / (gpus_per_inst * nic_bw)
    if mode == "relay":
        return weight_bytes / (gpus_per_inst * nic_bw) * (1 + 0.05 * (n_inst > 1))
    raise ValueError(f"unknown mode {mode!r}")


def max_instances(budget_s, weight_bytes, nic_bw, gpus_per_inst, mode):
    if mode == "relay":
        if disaggregated_s(weight_bytes, 2, nic_bw, gpus_per_inst, mode) <= budget_s:
            return None
        return 1 if disaggregated_s(weight_bytes, 1, nic_bw, gpus_per_inst, mode) <= budget_s else 0
    per_inst = disaggregated_s(weight_bytes, 1, nic_bw, gpus_per_inst, mode)
    return math.floor(budget_s / per_inst + 1e-9)
