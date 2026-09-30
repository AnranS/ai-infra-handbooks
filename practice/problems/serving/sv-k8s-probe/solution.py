def startup_budget(period_s, failure_threshold):
    return period_s * failure_threshold


def restart_loop(load_s, liveness_period_s, liveness_failures, has_startup_probe,
                 startup_period_s=10, startup_failures=30):
    if has_startup_probe:
        return load_s > startup_budget(startup_period_s, startup_failures)
    return load_s > startup_budget(liveness_period_s, liveness_failures)


def shutdown_seconds(prestop_s, drain_s, grace_s):
    return prestop_s + min(drain_s, grace_s)


def truncated(prestop_s, drain_s, grace_s):
    return drain_s > grace_s
