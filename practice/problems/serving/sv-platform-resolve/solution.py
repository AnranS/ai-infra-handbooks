UNSPECIFIED = "vllm.platforms.interface.UnspecifiedPlatform"
DEVICE_MODULES = {}


def _activate(detectors):
    out = {}
    for name, fn in detectors.items():
        try:
            q = fn()
        except Exception:
            continue
        if q is not None:
            out[name] = q
    return out


def resolve(builtin, oot, target_device=None, allowed=None):
    if target_device == "cpu":
        return builtin["cpu"]()
    if allowed is not None:
        oot = {n: f for n, f in oot.items() if n in allowed}
    for group in (_activate(oot), _activate(builtin)):
        if len(group) >= 2:
            raise RuntimeError(f"Only one platform plugin can be activated, but got: {sorted(group)}")
        if group:
            return next(iter(group.values()))
    return UNSPECIFIED


def resolve_sglang(plugins, selected=None, fallbacks=()):
    if selected:
        if selected not in plugins:
            raise RuntimeError(f"SGLANG_PLATFORM={selected!r} not found")
        result = plugins[selected]()
        if result is None:
            raise RuntimeError(f"platform plugin {selected!r} returned None")
        return result
    active = _activate(plugins)
    if len(active) >= 2:
        raise RuntimeError(f"Multiple platform plugins activated: {sorted(active)}")
    if active:
        return next(iter(active.values()))
    for qualname, available in fallbacks:
        if available():
            return qualname
    return "SRTPlatform"


class Platform:
    device_type = "cpu"

    def __getattr__(self, key):
        if key.startswith("__") and key.endswith("__"):
            raise AttributeError(key)
        module = DEVICE_MODULES.get(self.device_type)
        value = getattr(module, key, None) if module is not None else None
        return value
