#!/usr/bin/env bash
# Mac 环境诊断：工具链版本、pybind11 / PyTorch 扩展用两种编译器各编一次、torchrun + gloo 用两种启动方式各跑一次。
# 用法（仓库根目录）：bash env/mac_diag.sh
# 全部结果写到 build/mac_diag.log，最后打印一段"汇总"，把汇总贴给维护者即可。
cd "$(dirname "$0")/.."
mkdir -p build
LOG="build/mac_diag.log"
: > "$LOG"
PY="$PWD/.venv/bin/python"
BREW_CLANG=/opt/homebrew/opt/llvm/bin/clang++
SUMMARY=()

log() { printf '%s\n' "$*" | tee -a "$LOG"; }
step() {  # step <标签> <命令...>：跑一条命令，输出进日志，退出码记进汇总
  local label="$1"; shift
  log ""; log "\$ $*"
  "$@" >> "$LOG" 2>&1
  local rc=$?
  log "   -> 退出码 $rc"
  SUMMARY+=("$([[ $rc -eq 0 ]] && echo '✓' || echo '✗') $label（退出码 $rc）")
  return 0
}

log "===== 1. 工具链 ====="
step "xcode-select -p" xcode-select -p
step "ld 版本" /usr/bin/ld -v
step "SDK 列表" ls /Library/Developer/CommandLineTools/SDKs
step "苹果 clang 版本" /usr/bin/clang++ --version
if [[ -x "$BREW_CLANG" ]]; then
  step "Homebrew clang 版本" "$BREW_CLANG" --version
  step "Homebrew clang 默认链接器" "$BREW_CLANG" -v -Wl,-v -xc++ /dev/null -o /dev/null -fsyntax-only
else
  log "（没有 Homebrew LLVM：$BREW_CLANG 不存在）"
  SUMMARY+=("- 没有 Homebrew LLVM")
fi
step "Python / torch 版本" "$PY" -c "import sys, torch; print(sys.version.split()[0], 'torch', torch.__version__)"

log ""; log "===== 2. pybind11 / PyTorch 扩展：两种编译器各编一次（cpp 手册的第 14 章）====="
( cd cpp && CXX=/usr/bin/clang++ "$PY" tools/check_code.py docs/engineering/python-binding.md ) >> "$LOG" 2>&1
rc=$?; log "   苹果 clang -> 退出码 $rc"; SUMMARY+=("$([[ $rc -eq 0 ]] && echo '✓' || echo '✗') 扩展：苹果 clang（退出码 $rc）")
if [[ -x "$BREW_CLANG" ]]; then
  ( cd cpp && CXX="$BREW_CLANG" "$PY" tools/check_code.py docs/engineering/python-binding.md ) >> "$LOG" 2>&1
  rc=$?; log "   Homebrew clang -> 退出码 $rc"; SUMMARY+=("$([[ $rc -eq 0 ]] && echo '✓' || echo '✗') 扩展：Homebrew clang（退出码 $rc）")
fi

log ""; log "===== 3. torchrun + gloo：两种启动方式各跑一次（90 秒超时）====="
cat > build/tp_smoke.py <<'PYEOF'
import torch, torch.distributed as dist
dist.init_process_group("gloo")
t = torch.ones(1) * dist.get_rank()
dist.all_reduce(t)
print(f"rank {dist.get_rank()} ok, sum={t.item()}", flush=True)
dist.destroy_process_group()
PYEOF
"$PY" - "$LOG" <<'PYEOF'
import os, subprocess, sys, time
log = open(sys.argv[1], "a", encoding="utf-8")
py = sys.executable
env = dict(os.environ, GLOO_SOCKET_IFNAME="lo0", PYTHONUNBUFFERED="1")
variants = {
    "standalone": [py, "-m", "torch.distributed.run", "--standalone", "--nproc-per-node", "2", "build/tp_smoke.py"],
    "127.0.0.1 静态": [py, "-m", "torch.distributed.run", "--nproc-per-node", "2", "--master-addr", "127.0.0.1",
                      "--master-port", "29511", "build/tp_smoke.py"],
}
results = []
for name, cmd in variants.items():
    log.write(f"\n$ GLOO_SOCKET_IFNAME=lo0 {' '.join(cmd)}\n"); log.flush()
    t = time.time()
    try:
        r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=90)
        out = r.stdout + r.stderr
        ok = r.returncode == 0 and out.count("ok, sum=1.0") == 2
        log.write(out[-3000:] + f"\n   -> 退出码 {r.returncode}，{time.time() - t:.0f} 秒\n")
        results.append(f"{'✓' if ok else '✗'} torchrun {name}（{time.time() - t:.0f} 秒，退出码 {r.returncode}）")
    except subprocess.TimeoutExpired as e:
        log.write(((e.stdout or "") + (e.stderr or ""))[-3000:] + "\n   -> 90 秒超时，杀掉\n")
        results.append(f"✗ torchrun {name}（90 秒没跑完：挂住）")
print("\n".join(results))
PYEOF
# 把两种启动方式的结果从日志里抓出来进汇总
while IFS= read -r line; do SUMMARY+=("$line"); done < <("$PY" - "$LOG" <<'PYEOF'
import re, sys
txt = open(sys.argv[1], encoding="utf-8").read()
for name, key in (("standalone", "--standalone"), ("127.0.0.1 静态", "--master-addr")):
    seg = txt.split(key, 1)[1] if key in txt else ""
    seg = seg.split("\n$ ", 1)[0]
    if "90 秒超时" in seg:
        print(f"✗ torchrun {name}：90 秒没跑完（挂住）")
    elif seg.count("ok, sum=1.0") == 2:
        print(f"✓ torchrun {name}")
    else:
        print(f"✗ torchrun {name}：没跑通")
PYEOF
)

log ""; log "===== 汇总（把这一段贴给维护者）====="
for s in "${SUMMARY[@]}"; do log "$s"; done
log "完整日志：$LOG"
