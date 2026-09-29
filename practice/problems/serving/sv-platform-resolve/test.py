import copy
import pickle
from types import SimpleNamespace

import solution
from checker import check, raises
from solution import Platform, resolve, resolve_sglang

NONE = lambda: None


def boom():
    raise ImportError("broken plugin")


def test_example():
    check(resolve({"cuda": NONE, "cpu": lambda: "CpuPlatform"}, {"ascend": lambda: "NPUPlatform"}), "NPUPlatform",
          "外部插件优先")


def test_resolve():
    b = {"tpu": NONE, "cuda": lambda: "CudaPlatform", "rocm": NONE, "cpu": NONE}
    check(resolve(b, {}), "CudaPlatform", "只有内置的 CUDA")
    check(resolve({"cuda": NONE, "cpu": NONE}, {}), "vllm.platforms.interface.UnspecifiedPlatform", "都不可用")
    with raises(RuntimeError, "两个内置平台同时激活"):
        resolve({"cuda": lambda: "Cuda", "cpu": lambda: "Cpu"}, {})
    with raises(RuntimeError, "两个外部插件同时激活"):
        resolve(b, {"a": lambda: "A", "b": lambda: "B"})
    check(resolve(b, {"a": lambda: "A", "b": lambda: "B"}, allowed=["b"]), "B", "VLLM_PLUGINS 只允许 b")
    check(resolve(b, {"a": lambda: "A"}, allowed=[]), "CudaPlatform", "一个外部插件都不允许")
    check(resolve(b, {"broken": boom, "a": lambda: "A"}), "A", "检测函数抛异常时当作不可用")
    check(resolve({"cuda": boom, "cpu": lambda: "Cpu"}, {}), "Cpu", "内置的检测函数抛异常也一样")
    check(resolve({"cuda": lambda: "Cuda", "cpu": lambda: "Cpu"}, {"a": lambda: "A"}, target_device="cpu"), "Cpu",
          "VLLM_TARGET_DEVICE=cpu 时直接用 CPU")


def test_resolve_sglang():
    called = []

    def track(name, result):
        def fn():
            called.append(name)
            return result
        return fn

    plugins = {"npu": track("npu", "NPUPlatform"), "musa": track("musa", "MusaPlatform")}
    check(resolve_sglang(plugins, selected="musa"), "MusaPlatform", "SGLANG_PLATFORM 指定 musa")
    check(called, ["musa"], "指定时只调用这一个插件")
    with raises(RuntimeError, "指定的插件不存在"):
        resolve_sglang(plugins, selected="xpu")
    with raises(RuntimeError, "指定的插件返回 None（硬件不可用）"):
        resolve_sglang({"npu": NONE}, selected="npu")
    with raises(RuntimeError, "没指定时多个插件激活"):
        resolve_sglang({"npu": lambda: "N", "musa": lambda: "M"})
    check(resolve_sglang({"npu": NONE, "musa": lambda: "M"}), "M", "恰好一个激活")
    fb = [("CpuSRTPlatform", lambda: False), ("CudaSRTPlatform", lambda: True), ("NPUSRTPlatform", lambda: True)]
    check(resolve_sglang({"npu": NONE, "bad": boom}, fallbacks=fb), "CudaSRTPlatform", "回退到第一个可用的内置平台")
    check(resolve_sglang({}), "SRTPlatform", "什么都没有")


def test_platform_forwarding():
    solution.DEVICE_MODULES["npu"] = SimpleNamespace(synchronize=lambda: "npu-sync", device_count=lambda: 8,
                                                      maybe=None)
    solution.DEVICE_MODULES["cpu"] = SimpleNamespace(synchronize=lambda: "cpu-sync")

    class NPUPlatform(Platform):
        device_type = "npu"

        def device_count(self):
            return 16                                     # 平台类里显式定义的方法优先

    p = NPUPlatform()
    check(p.synchronize(), "npu-sync", "没定义的方法转发给设备模块")
    check(p.device_count(), 16, "显式定义的优先")
    check(p.missing_api, None, "设备模块里没有：返回 None")
    check(p.maybe, None, "值为 None")
    check(Platform().synchronize(), "cpu-sync", "基类默认 cpu")
    with raises(AttributeError, "双下划线的名字不转发"):
        getattr(p, "__not_a_real_dunder__")
    check(isinstance(copy.copy(p), NPUPlatform), True, "copy 能正常工作")
    check(type(pickle.loads(pickle.dumps(Platform()))).__name__, "Platform", "pickle 能正常工作")
    solution.DEVICE_MODULES.pop("xpu", None)

    class XpuPlatform(Platform):
        device_type = "xpu"

    check(XpuPlatform().synchronize, None, "设备模块不存在")
