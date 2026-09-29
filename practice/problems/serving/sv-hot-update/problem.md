---
title: 权重热更新：原地重新量化，以及暂停方式的代价
chapter: ops/weight-update.md
difficulty: 中等
tags: [权重更新, 量化, CUDA Graph, RL]
---
推理引擎里的层保存的是 kernel 用的格式：这里是按行量化的 int8 权重 `qweight`（形状 `(rows, cols)`）加每行一个 float32 缩放系数 `scale`（形状 `(rows, 1)`）。CUDA Graph 记住的是这两块存储的地址，所以热更新必须原地写入。实现：

1. `quantize(w)`：`scale = max(|w| 按行) / 127`（float32；整行都是 0 时该行的 scale 取 1），`qweight = np.round(w / scale)` 转成 int8，返回 `(qweight, scale)`；
2. `hot_update(layer, new_w)`：`layer` 有 `qweight`、`scale` 两个属性。把检查点格式的浮点权重 `new_w` 重新量化后**原地**写进 `layer.qweight` 和 `layer.scale`（数组对象和它们的存储都不能换），形状对不上时抛出 `ValueError`，而且不能改动原来的值；
3. `pause_costs(mode, done, left, prompt, step_ms)`：更新时暂停调度，在途请求的处理方式有四种。`done[i]`、`left[i]` 是第 i 个请求已经生成、还要生成的 token 数，每个请求的提示词都是 `prompt` 个 token，decode 一步 `step_ms` 毫秒。返回字典 `{"wait_s", "redecode", "reprefill", "mixed", "kv_fresh"}`（更新前要等多少秒、作废后要重新 decode 的 token 数、要重新 prefill 的 token 数、新旧权重混合的请求数、KV 是否全部由新权重算出）：
    - `"abort"`：中止全部请求，已生成的作废（重新 decode），提示词重新 prefill；
    - `"wait"`：等所有请求做完再更新，等待时间由最长的那个决定；
    - `"keep"`：原地冻结，更新后接着用旧 KV 生成；
    - `"retract"`：退回等待队列，更新后把提示词和已生成的 token 一起重新 prefill；

    其他 `mode` 抛出 `ValueError`。

```python
layer = Layer(w_old)                      # 测试里的简单对象：layer.qweight, layer.scale = quantize(w_old)
hot_update(layer, w_new)                  # 之后 layer.qweight 还是原来那个数组，内容是 quantize(w_new)[0]
pause_costs("wait", [10, 20], [100, 300], 2048, 25.0)["wait_s"]   # 7.5
```

<!-- 题解 -->
原地写入用切片赋值：`layer.qweight[...] = q`、`layer.scale[...] = s`，数组对象和底层存储都不变；而 `layer.qweight = q` 只是让属性指向一个新数组——模型的 Python 代码看到的是新值，但 CUDA Graph 重放时读的仍是旧存储。更新前要先检查形状：切片赋值遇到能广播的形状（比如一行）不会报错，会把错误的值悄悄写进去。

不能把浮点权重直接拷进 int8 数组：`qweight[...] = new_w` 会被截断成 0 附近的整数，缩放系数也还是旧的。必须走和加载时相同的处理（这里是量化，真实系统里还有切分、合并、重排），再原地拷贝。

四种暂停方式没有免费的：abort 作废已生成的内容，wait 被长尾拖住（最长的请求决定等待时间），keep 让一条轨迹混了两个版本的权重、旧 KV 还被新权重继续使用，retract 让 KV 用新权重重算，代价是一次（并行的、比重新 decode 便宜得多的）prefill。RL 系统按自己能接受的"策略陈旧度"来选。
