def startup_budget(period_s, failure_threshold):
    return period_s + failure_threshold           # 应该是相乘


def restart_loop(load_s, liveness_period_s, liveness_failures, has_startup_probe,
                 startup_period_s=10, startup_failures=30):
    return load_s > liveness_period_s * liveness_failures   # 有 startupProbe 时也按 liveness 算


def shutdown_seconds(prestop_s, drain_s, grace_s):
    return prestop_s + drain_s                    # 忘了宽限期到了会 SIGKILL


def truncated(prestop_s, drain_s, grace_s):
    return False                                  # 认为永远不会被截断
