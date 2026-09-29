"""运行全部检查：MINISGL=<你的 mini-sglang 的 python/ 目录> python run_checks.py

每项检查在单独的进程里运行（CPU、float32，约 5～10 分钟）；只跑一项：python checks/check_generate.py
"""
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CHECKS = ["check_config.py", "check_weights.py", "check_generate.py", "check_prefix_cache.py"]
minisgl = os.environ.get("MINISGL", str(ROOT / "minisgl" / "python"))
env = dict(os.environ, PYTHONPATH=os.pathsep.join([minisgl, os.environ.get("PYTHONPATH", "")]),
           OMP_NUM_THREADS=os.environ.get("OMP_NUM_THREADS", "8"), TOKENIZERS_PARALLELISM="false",
           TRANSFORMERS_VERBOSITY="error", HF_HUB_OFFLINE="1")
print(f"被检查的 mini-sglang：{minisgl}\n")
failed = 0
for script in CHECKS:
    r = subprocess.run([sys.executable, script], cwd=HERE / "checks", capture_output=True, text=True, env=env)
    lines = [l for l in r.stdout.splitlines() if l.startswith(("PASS", "FAIL"))]
    print(lines[-1] if lines else f"FAIL  {script}  运行出错：\n{(r.stdout + r.stderr)[-1500:]}")
    failed += r.returncode != 0 or not lines or lines[-1].startswith("FAIL")
print(f"\n{len(CHECKS) - failed} / {len(CHECKS)} 项通过")
sys.exit(1 if failed else 0)
