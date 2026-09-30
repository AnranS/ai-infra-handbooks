def rollout_batches(replicas, max_surge, max_unavailable):
    return replicas // max(1, max_surge + max_unavailable)      # 整除：最后不满一批被漏掉


def rollout_seconds(replicas, ready_s, drain_s, max_surge, max_unavailable):
    return rollout_batches(replicas, max_surge, max_unavailable) * (ready_s + drain_s)   # 启动和退出其实是并行的


def extra_gpus(replicas, gpus_per_replica, max_surge):
    return replicas * gpus_per_replica                          # 算成了整个服务的卡数


def capacity_loss(replicas, max_unavailable):
    return max_unavailable / replicas                           # 返回的是损失比例，不是可用比例


def scale_lag_seconds(ready_s, image_pull_s=0, node_provision_s=0):
    return ready_s                                              # 忽略了拉镜像和开机器
