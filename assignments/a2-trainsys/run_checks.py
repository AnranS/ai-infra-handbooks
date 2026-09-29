"""运行全部检查：python run_checks.py（用 torchrun 启动多个进程，gloo 后端，CPU 即可）"""
import os
import subprocess
import sys
from pathlib import Path

CHECKS = [("check_ddp.py", 2), ("check_zero1.py", 4), ("check_recompute.py", 1), ("check_together.py", 2)]
here = Path(__file__).parent / "checks"
env = dict(os.environ, OMP_NUM_THREADS="2")
failed = 0
for script, n in CHECKS:
    r = subprocess.run([sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc-per-node={n}", script],
                       cwd=here, capture_output=True, text=True, env=env)
    lines = [l for l in r.stdout.splitlines() if l.startswith(("PASS", "FAIL"))]
    print(lines[0] if lines else f"FAIL  {script}  运行出错：\n{(r.stdout + r.stderr)[-1500:]}")
    failed += r.returncode != 0 or not lines or lines[0].startswith("FAIL")
print(f"\n{len(CHECKS) - failed} / {len(CHECKS)} 项通过")
sys.exit(1 if failed else 0)
