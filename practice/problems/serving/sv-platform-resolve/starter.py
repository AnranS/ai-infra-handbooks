UNSPECIFIED = "vllm.platforms.interface.UnspecifiedPlatform"
DEVICE_MODULES = {}


def resolve(builtin, oot, target_device=None, allowed=None):
    for name, fn in {**builtin, **oot}.items():      # 没有区分内置和外部插件，也没处理异常、多个激活
        q = fn()
        if q is not None:
            return q
    return UNSPECIFIED


def resolve_sglang(plugins, selected=None, fallbacks=()):
    pass


class Platform:
    device_type = "cpu"
