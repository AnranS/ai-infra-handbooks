---
title: 探针与优雅退出的时序
chapter: k8s/deploy-scale.md
difficulty: 简单
tags: [Kubernetes,探针,优雅退出]
---
把探针和停止流程的时序算清楚。实现：

1. `startup_budget(period_s, failure_threshold)`：startupProbe 允许的最长启动时间；
2. `restart_loop(load_s, liveness_period_s, liveness_failures, has_startup_probe, startup_period_s=10, startup_failures=30)`：权重加载要 `load_s` 秒时，容器会不会被 liveness 探针杀掉重启（返回 `True` 表示会陷入重启循环）。有 startupProbe 时，只要 `load_s` 在 startup 的预算之内就不会被杀；没有 startupProbe 时，`liveness_period_s * liveness_failures` 小于 `load_s` 就会被杀；
3. `shutdown_seconds(prestop_s, drain_s, grace_s)`：一个 Pod 从被删除到真正消失的时间。先执行 preStop（`prestop_s`），然后进程处理完手上的请求（`drain_s`），但总的等待不超过 `prestop_s + grace_s`（超时就 SIGKILL）；
4. `truncated(prestop_s, drain_s, grace_s)`：是否会有请求被强杀截断。

```python
startup_budget(10, 30)                       # 300
restart_loop(180, 20, 3, has_startup_probe=False)   # True：60 秒就被杀了
restart_loop(180, 20, 3, has_startup_probe=True)    # False
shutdown_seconds(5, 120, 60)                 # 65
truncated(5, 120, 60)                        # True
```

<!-- 题解 -->
推理服务的两个特殊数字——启动几分钟、退出几分钟——把探针的默认配置全部推翻了。

**启动**：没有 startupProbe 时，liveness 的预算是 `period × failureThreshold`，权重加载比它慢就会被反复杀掉，表现为 CrashLoopBackOff 且日志停在"Loading model weights"。加上 startupProbe 之后，前两种探针在启动期间不生效，预算变成 `startup_period × startup_failures`。

**退出**：`terminationGracePeriodSeconds` 是从 **SIGTERM 之后**开始算的（preStop 不占用它），超时就 SIGKILL。推理服务手上可能有正在生成的长回答，宽限期要按"最长回答的生成时间"设，否则用户会看到回答被截断。preStop 里 sleep 几秒是为了盖住"端点摘除是异步的"这个窗口——这段时间新请求还可能进来。
