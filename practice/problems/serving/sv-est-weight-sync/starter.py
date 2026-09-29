def colocated_s(weight_bytes, gpus_per_inst, nvlink_bw):
    return weight_bytes / nvlink_bw                    # 忘了每张卡只需要拿自己那一片


def disaggregated_s(weight_bytes, n_inst, nic_bw, gpus_per_inst, mode):
    pass


def max_instances(budget_s, weight_bytes, nic_bw, gpus_per_inst, mode):
    pass
