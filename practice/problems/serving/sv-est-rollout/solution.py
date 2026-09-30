def rollout_batches(replicas, max_surge, max_unavailable):
    per_batch = max(1, max_surge + max_unavailable)
    return -(-replicas // per_batch)


def rollout_seconds(replicas, ready_s, drain_s, max_surge, max_unavailable):
    return rollout_batches(replicas, max_surge, max_unavailable) * max(ready_s, drain_s)


def extra_gpus(replicas, gpus_per_replica, max_surge):
    return max_surge * gpus_per_replica


def capacity_loss(replicas, max_unavailable):
    if replicas <= 0:
        return 0.0
    return (replicas - min(max_unavailable, replicas)) / replicas


def scale_lag_seconds(ready_s, image_pull_s=0, node_provision_s=0):
    return ready_s + image_pull_s + node_provision_s
