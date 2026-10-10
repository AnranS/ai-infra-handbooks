import inspect
import warnings

warnings.filterwarnings("ignore")
from sglang.srt.managers.scheduler import Scheduler
from sglang_omni.scheduling.omni_scheduler import OmniScheduler

upstream = {n for n, v in inspect.getmembers(Scheduler) if callable(v) and not n.startswith("__")}
own = {n for n, v in vars(OmniScheduler).items() if callable(v) and not n.startswith("__")}
print("SGLang Scheduler 的方法（含 mixin）：", len(upstream))
print("OmniScheduler 自己定义的方法：      ", len(own))
print("其中覆盖了上游同名方法的：")
for name in sorted(upstream & own):
    print("  ", name)
