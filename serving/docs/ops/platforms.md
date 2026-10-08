# 多硬件支持：platform 抽象与昇腾插件

<p class="lead">推理框架要跑在 NVIDIA、AMD、Intel、Google TPU、华为昇腾等不同的硬件上。它们的差别远不止把 "cuda" 换成另一个设备名：设备 API、可见设备的环境变量、集合通信库、注意力和 MoE 的 kernel、图模式、编译后端、量化格式都不一样。vLLM 和 SGLang 都把这些决定集中到一个 Platform 类里，并允许第三方用插件接入新硬件，不必改框架本身。这一章读两边的实现，用一个可运行的小例子演示插件是怎么被发现和选中的，再看昇腾的插件 vllm-ascend 具体接管了哪些东西，最后整理接入一种新硬件要做的事。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. 接入一种新硬件，推理引擎里哪些东西必须跟着换？
    2. vLLM 怎样发现平台插件？内置平台和外部插件同时可用时选哪个？
    3. `current_platform` 为什么要在第一次访问时才初始化？
    4. vllm-ascend 除了实现 Platform 类，为什么还要给 vLLM 打补丁？补丁是怎么组织的？
    5. PyTorch 本身不认识昇腾 NPU，`torch.npu`、`torch.device("npu")` 是怎么来的？

??? success "自测参考答案（先自己答，再展开对照）"
    1. PyTorch 的设备后端（`torch.cuda` / `torch.npu`……）和分发键；可见设备的环境变量；集合通信后端（NCCL、RCCL、HCCL、XCCL）和设备间通信器；注意力、MoE、量化等 kernel 及其选择逻辑；图模式（CUDA Graph 或各家的替代品）；编译后端；显存分配器与睡眠模式；LoRA 的 kernel；PD 分离的 KV 传输。
    2. 外部插件用 Python 的 entry point 登记在 `vllm.platform_plugins` 组下，vLLM 读已安装包的元数据找到它们，依次调用每个插件的检测函数（硬件可用就返回平台类的全名）。内置平台（cuda、rocm、tpu、xpu、cpu）也各有一个检测函数。外部插件优先：恰好一个外部插件激活时用它；两个以上报错；没有外部插件时在内置平台里选，同样不允许两个同时激活；都没有就是 `UnspecifiedPlatform`。
    3. 外部插件要 `from vllm.platforms import Platform` 来继承基类，如果导入 `vllm.platforms` 时就解析平台，插件还没加载，会选错平台；所以 `current_platform` 用模块级的 `__getattr__` 延迟到第一次访问时才解析。
    4. Platform 类只覆盖了 vLLM 预留的扩展点，很多行为（分布式组的建立与销毁、调度器的细节、某些模型的实现）没有钩子，插件只能在运行时替换 vLLM 的函数。vllm-ascend 把补丁分成两类：启动前生效的平台补丁（在 `pre_register_and_update` 里应用）和每个 worker 启动时生效的 worker 补丁，每类再按 vLLM 版本分目录，并逐条写明为什么打、怎么打、打算什么时候随上游合入而删掉。
    5. PyTorch 给树外设备预留了 `PrivateUse1` 分发键：torch_npu 把它改名为 "npu"、注册设备模块，于是有了 `torch.npu`、`torch.device("npu")`，算子通过这个分发键注册到昇腾的实现上。vllm-ascend 的平台类里 `dispatch_key = "PrivateUse1"` 就是这个原因。

## 硬件不同，差在哪

![图：多硬件支持的分层——引擎逻辑、Platform 抽象、平台插件、kernel](../assets/figures/platform-layers.svg){.aig-svg}

以 vLLM 各平台类上的几个属性为例（昇腾一列来自 vllm-ascend 0.9.1）：

| | CUDA | ROCm | Intel XPU | 昇腾 NPU |
| --- | --- | --- | --- | --- |
| `device_type`（PyTorch 的设备名） | `cuda` | `cuda`（PyTorch 的 ROCm 版沿用 `torch.cuda`） | `xpu` | `npu` |
| `dispatch_key` | `CUDA` | `CUDA` | `XPU` | `PrivateUse1` |
| 集合通信 | `nccl` | `nccl`（底层是 RCCL） | `xccl` | HCCL（torch_npu 的 `ProcessGroupHCCL`） |
| 可见设备的环境变量 | `CUDA_VISIBLE_DEVICES` | `CUDA_VISIBLE_DEVICES` | `ZE_AFFINITY_MASK` | `ASCEND_RT_VISIBLE_DEVICES` |
| Ray 里的资源名 | `GPU` | `GPU` | `GPU` | `NPU` |

这只是最表层的差别。往下还有：注意力后端（FlashAttention、FlashInfer / ROCm 的 aiter / 昇腾自己的注意力实现）、MoE 和量化 kernel、图模式（CUDA Graph / 昇腾的 ACL Graph 与 TorchAir 图模式）、编译后端（Inductor 能不能用）、显存分配器（睡眠模式依赖的 CuMem）、LoRA 的分段矩阵乘、PD 分离时的 KV 传输。一个引擎要支持多种硬件，就得把这些决定从业务代码里拿出来，集中到一处。

## vLLM：Platform 类与平台插件

vLLM 的做法是一个 `Platform` 类（`vllm/platforms/interface.py`），每种硬件一个子类（`vllm/platforms/` 下的 `cuda.py`、`rocm.py`、`xpu.py`、`tpu.py`、`cpu.py`），引擎里到处用的 `current_platform` 就是当前硬件那个子类的实例。它提供三类东西：

- **硬件相关的名字**：上表那些类属性，外加 `supported_quantization` 等；
- **返回组件全名的方法**：`get_attn_backend_cls`（选注意力后端）、`get_device_communicator_cls`（设备间通信器）、`get_punica_wrapper`（LoRA 的 kernel 封装）、`get_static_graph_wrapper_cls`（CUDA Graph 或它的替代品），返回的是字符串形式的类名，由引擎按需导入，于是插件可以把自己的实现"填"进去；
- **能力查询与配置钩子**：`get_device_capability`、`supports_fp8`、`is_arch_support_pdl` 这类问答，以及 `check_and_update_config`（在引擎配置确定后按硬件修正，比如块大小、图模式）、`pre_register_and_update`。另外，基类的 `__getattr__` 会把找不到的属性转发给 `torch.<device_type>`，所以引擎代码写 `current_platform.synchronize()`、`current_platform.empty_cache()` 就能在各种硬件上工作。

平台怎么选出来（`vllm/platforms/__init__.py` 的 `resolve_current_platform_cls_qualname`）：

1. `VLLM_TARGET_DEVICE=cpu` 时直接用 CPU 平台；
2. 否则逐个运行内置的检测函数（tpu、cuda、rocm、xpu、cpu），以及外部插件的检测函数。外部插件登记在 entry point 组 `vllm.platform_plugins`（`vllm/plugins/__init__.py`），`VLLM_PLUGINS` 可以限定只加载哪几个；
3. 外部插件优先：恰好一个外部插件激活就用它，两个以上报错；否则在内置平台里选，同样只允许一个；都没有就是 `UnspecifiedPlatform`；
4. `current_platform` 在第一次访问时才解析（模块级 `__getattr__`），因为外部插件要先 `from vllm.platforms import Platform` 继承基类。

下面把这套机制缩成一个可以运行的例子。基类和选择规则：

```python title="mini_platform.py"
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
```

在临时目录里"安装"一个假的硬件插件（一个模块，加上 dist-info 元数据，`entry_points.txt` 把它的 `register` 登记在 `vllm.platform_plugins` 组下），然后走一遍发现、选择、实例化：

```python
import importlib.metadata as md
import pkgutil
import sys
import tempfile
from pathlib import Path

from mini_platform import resolve

GROUP = "vllm.platform_plugins"

# 1. 在临时目录里"装"一个插件包：模块 + dist-info 元数据，entry_points.txt 把 register 登记在 vllm.platform_plugins 组下
site = Path(tempfile.mkdtemp())
(site / "mydev_plugin.py").write_text('''
from mini_platform import Platform

class MyDevPlatform(Platform):
    device_type = "cpu"                        # 假装是新硬件：设备 API 借用 torch.cpu
    dispatch_key = "PrivateUse1"
    dist_backend = "mydevccl"
    device_control_env_var = "MYDEV_VISIBLE_DEVICES"

def register():                                # 检测硬件：可用就返回平台类的全名，否则返回 None
    return "mydev_plugin.MyDevPlatform"
''')
dist = site / "mydev_plugin-0.1.dist-info"
dist.mkdir()
(dist / "METADATA").write_text("Metadata-Version: 2.1\nName: mydev-plugin\nVersion: 0.1\n")
(dist / "entry_points.txt").write_text(f"[{GROUP}]\nmydev = mydev_plugin:register\n")
sys.path.insert(0, str(site))

# 2. 发现：entry point 来自已安装包的元数据，列出它们不需要 import 插件模块
eps = md.entry_points(group=GROUP)
print("发现的插件：", [(ep.name, ep.value) for ep in eps])
oot = {ep.name: ep.load() for ep in eps}

# 3. 选择：这台机器上内置的检测函数只有 cpu 返回了结果
builtin = {"cuda": lambda: None, "rocm": lambda: None, "cpu": lambda: "vllm.platforms.cpu.CpuPlatform"}
print("只有内置平台：", resolve(builtin, {}))
qualname = resolve(builtin, oot)
print("装了插件之后：", qualname)
try:
    resolve(builtin, {**oot, "other": lambda: "other_plugin.OtherPlatform"})
except RuntimeError as e:
    print("两个外部插件同时可用：", e)

# 4. 实例化；基类里没有的方法转发给 torch.<device_type>
current_platform = pkgutil.resolve_name(qualname)()
print(type(current_platform).__name__, current_platform.dispatch_key, current_platform.dist_backend,
      current_platform.device_control_env_var)
print("current_platform.is_available() ->", current_platform.is_available(), "（实际调用的是 torch.cpu.is_available）")
```

```text title="输出"
发现的插件： [('mydev', 'mydev_plugin:register')]
只有内置平台： vllm.platforms.cpu.CpuPlatform
装了插件之后： mydev_plugin.MyDevPlatform
两个外部插件同时可用： Only one platform plugin can be activated, but got: ['mydev', 'other']
MyDevPlatform PrivateUse1 mydevccl MYDEV_VISIBLE_DEVICES
current_platform.is_available() -> True （实际调用的是 torch.cpu.is_available）
```

entry point 是 Python 打包的标准机制：插件包在 `pyproject.toml` 里写

```toml
[project.entry-points."vllm.platform_plugins"]
mydev = "mydev_plugin:register"
```

安装之后，这一行就进了包的元数据，`importlib.metadata.entry_points(group=...)` 不用导入任何插件模块就能列出它。除了平台插件，vLLM 还有 `vllm.general_plugins` 组：每个进程（引擎和每个 worker）启动时都会调用 `load_general_plugins()` 运行这一组的函数，插件用它注册自己的模型实现、打补丁。

## 昇腾：vllm-ascend

昇腾的支持不在 vLLM 仓库里，而是一个独立维护的插件 vllm-ascend。以 0.9.1 版为例，它的包元数据里登记了两个入口：

```ini
[vllm.general_plugins]
ascend_enhanced_model = vllm_ascend:register_model

[vllm.platform_plugins]
ascend = vllm_ascend:register
```

`register()` 只做一件事：返回 `"vllm_ascend.platform.NPUPlatform"`。这个平台类（`_enum = PlatformEnum.OOT`）接管了这些东西：

- **名字**：`device_type = "npu"`、`dispatch_key = "PrivateUse1"`、`device_control_env_var = "ASCEND_RT_VISIBLE_DEVICES"`、`ray_device_key = "NPU"`；0.9.1 里 `simple_compile_backend = "eager"`，也就是不用 torch.compile；
- **组件**：`get_attn_backend_cls` 返回昇腾自己的注意力后端（普通注意力和 MLA 各一个），`get_device_communicator_cls` 返回 `NPUCommunicator`，`get_punica_wrapper` 返回 NPU 版的 LoRA 封装，还有自己的 worker、模型运行器、分段编译后端；分布式进程组用 torch_npu 的 `ProcessGroupHCCL`；
- **图模式**：两条路。ACL Graph（昇腾的"CUDA Graph"，0.9.1 里主要在 Qwen 系列上测过）和 TorchAir 图模式（通过 `--additional-config` 里的 `torchair_graph_config` 打开，当时只支持 DeepSeek 系列等少数模型）。

`register_model()` 注册 vllm-ascend 自己的模型实现（比如 DeepSeek 系列），并应用补丁。补丁是插件生态的现实：Platform 类只覆盖了 vLLM 预留的扩展点，没有钩子的地方（分布式组的建立与销毁、数据并行的端口、调度器的细节……）只能在运行时替换 vLLM 的函数。vllm-ascend 把补丁分成两类：

- **平台补丁**：在 `NPUPlatform.pre_register_and_update()` 里应用，进程启动、worker 创建之前生效；
- **worker 补丁**：在每个 worker 的 `__init__` 里应用。

每一类再按 vLLM 版本分成 `patch_0_9_1`、`patch_main`、`patch_common` 三个目录，并在 `patch/__init__.py` 里逐条写明补的是哪个函数、为什么、怎么补、对应的上游 PR、打算什么时候删掉。补丁越少，说明上游的扩展点越完善；给新硬件做适配时，把需要的钩子推回上游，是减少长期维护成本的关键。

往下一层是 PyTorch 本身：PyTorch 为树外设备预留了 `PrivateUse1` 分发键，torch_npu 把它改名为 "npu"（`torch.utils.rename_privateuse1_backend`）并注册设备模块，于是有了 `torch.npu`、`torch.device("npu")`，算子通过这个分发键注册到 CANN 的实现上（分发键的机制见 CUDA 手册的 [dispatcher 与自定义算子](cuda://framework/dispatcher/)）。vLLM 的 `Platform.__getattr__` 转发给 `torch.npu`，靠的就是这一层。

## SGLang：SRTPlatform

SGLang 0.5.20 也有了类似的抽象（`srt/platforms/`）：基类 `SRTPlatform`，内置 CUDA、ROCm、NPU、XPU、CPU 五个子类，外部插件登记在 entry point 组 `sglang.srt.platforms`。它的方法更贴近 SGLang 自己的结构：`get_default_attention_backend`、`get_graph_runner_cls`、`get_mha_kv_pool_cls` / `get_mla_kv_pool_cls`、`get_paged_allocator_cls`、`get_compile_backend`、`support_cuda_graph`、`support_piecewise_cuda_graph`、`apply_server_args_defaults` 等。和 vLLM 的两点不同：

- **选择规则**：设置了 `SGLANG_PLATFORM` 时只加载并激活这一个插件（其他插件的模块根本不导入，避免拉进它们的依赖）；没设置时加载全部插件，恰好一个激活就用它，多个激活报错；没有插件激活时按 CPU（`SGLANG_USE_CPU_ENGINE=1`）、CUDA、NPU、ROCm、XPU 的顺序回退到内置平台；
- **昇腾在仓库里**：`NPUSRTPlatform` 是内置的，默认注意力后端是 `"ascend"`，图运行器是 `hardware_backend/npu/graph_runner` 里的 NPU 版本，不支持分段图；`get_device_capability` 固定返回 (0, 0)，因为 torch_npu 报告的计算能力只是为了兼容 PyTorch 而配置的值，并不代表硬件能力。昇腾相关的 kernel、显存池、量化、MoE 都在 `srt/hardware_backend/npu/` 下；同一层目录里还有 `musa`（摩尔线程）、`mlx`（苹果芯片）、`xpu`、`cpu` 等后端。

两种组织方式各有取舍：放在上游仓库里，改动能和引擎一起测试、一起发布，但主仓库的维护负担更重；放在插件里，硬件厂商可以独立迭代，代价是版本对齐和补丁维护。

## 接入一种新硬件要做什么

按依赖顺序大致是：

1. **PyTorch 设备后端**：没有官方支持的设备，先用 `PrivateUse1` 接进 PyTorch，让张量、基本算子、`torch.distributed` 的进程组（通信库）能用；
2. **平台类**：设备名、分发键、可见设备的环境变量、通信后端、Ray 的资源名；显存查询、设备名、计算能力；
3. **核心 kernel**：注意力（prefill 与 decode、分页 KV）、RMSNorm / RoPE / 激活、采样；然后是 MoE、量化 GEMM、LoRA；每一个都要和 CUDA 版的输出对齐（方法见[新模型接入与精度对齐](new-model.md)）；
4. **图模式与编译**：没有 CUDA Graph 等价物时 decode 会被发射开销拖垮（见 [CUDA Graphs 与 torch.compile](../engine/graphs-compile.md)），这是性能的第一道坎；
5. **分布式**：张量并行的 all-reduce、专家并行的 all-to-all、PD 分离的 KV 传输，都要有对应的通信实现；
6. **补丁与上游化**：先用补丁跑通，再把需要的扩展点推回上游，补丁越少越好维护。

!!! interview "怎么讲清楚"
    讲"推理框架怎么支持多种硬件"：把硬件相关的决定集中到一个 Platform 类——设备名、分发键、可见设备的环境变量、通信后端这些属性，选注意力后端、通信器、LoRA kernel、图模式封装这些返回组件类名的方法，以及能力查询和按硬件修正配置的钩子；引擎只和 `current_platform` 打交道。新硬件通过 Python entry point（vLLM 是 `vllm.platform_plugins`，SGLang 是 `sglang.srt.platforms`）以插件形式接入，外部插件优先、多个同时激活就报错；`current_platform` 延迟初始化，好让插件先继承基类。昇腾的 vllm-ascend 就是这样的插件：`PrivateUse1` 分发键、HCCL、自己的注意力后端和图模式（ACL Graph、TorchAir），没有扩展点的地方靠版本化的补丁。能讲出"补丁越少越好、钩子要推回上游"会加分。

## 练习

**1. 两个插件。** 一台机器上同时装了 vllm-ascend 和另一个硬件厂商的 vLLM 平台插件，两者的检测函数都返回了平台类名。vLLM 会怎样？怎么解决？

??? success "参考答案"
    两个外部插件同时激活，vLLM 直接报错（"Only one platform plugin can be activated"）。解决办法：卸载用不到的那个；或者用 `VLLM_PLUGINS` 只加载需要的插件（它限定的是按名字加载哪些 entry point，比如 `VLLM_PLUGINS=ascend`，注意这也会影响 general plugins 的加载）；或者让插件的检测函数在硬件不存在时返回 None——检测函数本来就应该真的检查硬件，而不是装了就返回。SGLang 里对应的做法是设置 `SGLANG_PLATFORM` 指定用哪一个。

**2. 为什么返回类名字符串。** `get_attn_backend_cls`、`get_device_communicator_cls` 为什么返回字符串形式的类名，而不是直接返回类？

??? success "参考答案"
    延迟导入：注意力后端、通信器这些模块往往依赖特定硬件的库（FlashAttention、NCCL、torch_npu……），在别的硬件上根本导入不了。返回类名、由引擎在真正需要时再导入，平台类本身就能在任何机器上被导入和检测；插件也只需要给出自己实现的全名，不必让 vLLM 的代码依赖插件的模块。

**3. 转发的边界。** `Platform.__getattr__` 把找不到的属性转发给 `torch.<device_type>`。这样做的好处和风险各是什么？

??? success "参考答案"
    好处：设备 API 大多同名（`synchronize`、`empty_cache`、`mem_get_info`、`Stream`……），引擎写一次 `current_platform.xxx()` 就能在各种硬件上工作，平台类不必逐个包装。风险：两个后端同名函数的语义可能不同（比如计算能力，torch_npu 返回的是配置出来的兼容值，SGLang 因此在 NPU 平台上固定返回 (0, 0)）；某个后端缺这个函数时，vLLM 只打印一次警告并返回 None，调用方如果直接调用就会在别处报错。重要的差异应该在平台类里显式实现，而不是依赖转发。

## 小结

- [x] 多硬件的差别包括设备 API、分发键、可见设备、通信后端、kernel、图模式、编译、分配器、KV 传输，要集中到一个 Platform 类里。
- [x] vLLM：`Platform` 子类提供名字、返回组件类名的方法和配置钩子；`current_platform` 延迟解析；外部插件通过 `vllm.platform_plugins` 接入，优先于内置平台，多个同时激活就报错。
- [x] vllm-ascend：`PrivateUse1`、HCCL、自己的注意力后端、ACL Graph 与 TorchAir 图模式；没有扩展点的地方用按版本组织、逐条说明的补丁。
- [x] SGLang：`SRTPlatform` + `sglang.srt.platforms` 插件组 + `SGLANG_PLATFORM`；昇腾支持在仓库里（`hardware_backend/npu`）。
- [x] 接入新硬件的顺序：PyTorch 后端 → 平台类 → 核心 kernel 与精度对齐 → 图模式 → 分布式 → 把补丁推回上游。
