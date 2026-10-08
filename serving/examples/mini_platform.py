"""mini_platform.py —— vLLM 平台抽象的骨架：Platform 基类，以及选择平台的规则（vllm/platforms/__init__.py）。"""

import torch


class Platform:
    """硬件相关的名字都是类属性；基类里没有的方法，转发给 torch.<device_type>（比如 synchronize、empty_cache）"""
    device_type = "cpu"
    dispatch_key = "CPU"
    dist_backend = "gloo"
    device_control_env_var = ""

    def __getattr__(self, key):
        return getattr(getattr(torch, self.device_type), key)


def resolve(builtin, oot):
    """builtin、oot：名字 → 检测函数（硬件可用时返回平台类的全名，否则返回 None）。
    外部插件优先；同一类里激活了两个以上就报错；都没有就是 UnspecifiedPlatform"""
    act_builtin = {n: q for n, f in builtin.items() if (q := f()) is not None}
    act_oot = {n: q for n, f in oot.items() if (q := f()) is not None}
    for acts in (act_oot, act_builtin):
        if len(acts) >= 2:
            raise RuntimeError(f"Only one platform plugin can be activated, but got: {sorted(acts)}")
        if acts:
            return next(iter(acts.values()))
    return "vllm.platforms.interface.UnspecifiedPlatform"
