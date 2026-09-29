---
title: 平台插件：发现、选择与设备 API 的转发
chapter: ops/platforms.md
difficulty: 简单
tags: [多硬件, 插件, vLLM, SGLang]
---
推理框架用"检测函数"决定当前是哪种硬件：每个检测函数在硬件可用时返回平台类的全名（字符串），否则返回 `None`；检测函数抛出异常时当作不可用。实现：

1. `resolve(builtin, oot, target_device=None, allowed=None)`：vLLM 的规则。`builtin`、`oot` 都是 `{名字: 检测函数}`。`target_device == "cpu"` 时直接返回 `builtin["cpu"]()`；`allowed` 不为 `None` 时只考虑名字在其中的外部插件（相当于 `VLLM_PLUGINS`）。外部插件优先：恰好一个激活就返回它的结果，两个以上抛出 `RuntimeError`；没有外部插件激活时在内置平台里选，同样只允许一个，两个以上抛出 `RuntimeError`；都没有返回 `"vllm.platforms.interface.UnspecifiedPlatform"`；
2. `resolve_sglang(plugins, selected=None, fallbacks=())`：SGLang 的规则。`selected`（相当于 `SGLANG_PLATFORM`）不为空时只调用这一个插件：名字不存在或返回 `None` 都抛出 `RuntimeError`，其他插件一个都不许调用。没有指定时调用全部插件：恰好一个激活就用它，多个激活抛出 `RuntimeError`；一个都没有时按顺序检查 `fallbacks`（`[(全名, 是否可用的函数), ...]`），返回第一个可用的，都不可用返回 `"SRTPlatform"`；
3. `Platform` 类：类属性 `device_type = "cpu"`；实例上找不到的属性转发给 `DEVICE_MODULES[self.device_type]`（一个模块级的字典，值相当于 `torch.cuda`、`torch.npu` 这些设备模块），设备模块也没有或者值为 `None` 时返回 `None`；名字前后都是双下划线的属性不转发，直接抛出 `AttributeError`（否则 pickle、copy 会出问题）。

```python
resolve({"cuda": lambda: None, "cpu": lambda: "CpuPlatform"}, {"ascend": lambda: "NPUPlatform"})   # 'NPUPlatform'
```

<!-- 题解 -->
外部插件优先，是因为装了某个硬件的插件就说明用户想用它；两个外部插件同时激活时框架无从判断，只能报错，由用户卸载一个、用 `VLLM_PLUGINS` 限定，或者在 SGLang 里用 `SGLANG_PLATFORM` 指定。检测函数抛异常时当作不可用，而不是让整个启动失败——某个插件的依赖坏了，不应该影响在别的硬件上运行。SGLang 指定插件时只加载那一个，其他插件的模块根本不导入，避免拉进它们的依赖。

`__getattr__` 只在正常的属性查找失败时才被调用，所以平台类里显式定义的方法优先，没定义的才转发给设备模块：`current_platform.synchronize()` 在 CUDA 上就是 `torch.cuda.synchronize()`，在昇腾上就是 `torch.npu.synchronize()`。双下划线的名字必须照常抛出 `AttributeError`：pickle 会探测 `__getstate__`、`__reduce_ex__` 这类方法，如果转发后返回 `None`，它会把 `None` 当成真实的值去调用。
