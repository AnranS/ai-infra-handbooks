#!/usr/bin/env bash
# Mac 环境诊断：工具链版本、pybind11 / PyTorch 扩展用两种编译器各编一次、torchrun + gloo 用两种启动方式各跑一次。
# 用法（仓库根目录）：bash env/mac_diag.sh
# 全部结果写到 build/mac_diag.log，最后打印一段"汇总"，把汇总贴给维护者即可。
# 注意：macOS 自带的是 bash 3.2——变量后面紧跟中文时必须写成 ${var}，否则中文的字节会被当成变量名的一部分。
cd "$(dirname "$0")/.."
mkdir -p build
LOG="build/mac_diag.log"
: > "$LOG"
PY="$PWD/.venv/bin/python"
BREW_CLANG=/opt/homebrew/opt/llvm/bin/clang++
SUMMARY=()

log() { printf '%s\n' "$*" | tee -a "$LOG"; }
mark() { if [ "$1" -eq 0 ]; then printf '✓'; else printf '✗'; fi; }
step() {  # step <标签> <命令...>：跑一条命令，输出进日志；汇总里带上退出码和输出的第一行
  local label="$1"; shift
  log ""; log "\$ $*"
  local out rc
  out=$("$@" 2>&1); rc=$?
  printf '%s\n' "$out" >> "$LOG"
  log "   -> 退出码 ${rc}"
  SUMMARY+=("$(mark "$rc") ${label}：$(printf '%s\n' "$out" | head -n 1 | cut -c1-110)")
  return 0
}

log "===== 1. 工具链 ====="
step "xcode-select -p" xcode-select -p
step "ld 版本" /usr/bin/ld -v
step "SDK 列表" ls /Library/Developer/CommandLineTools/SDKs
step "苹果 clang" /usr/bin/clang++ --version
if [ -x "$BREW_CLANG" ]; then
  step "Homebrew clang" "$BREW_CLANG" --version
else
  log "（没有 Homebrew LLVM：${BREW_CLANG} 不存在）"
  SUMMARY+=("- 没有 Homebrew LLVM")
fi
step "Python / torch" "$PY" -c "import sys, torch; print(sys.version.split()[0], 'torch', torch.__version__)"

log ""; log "===== 2. pybind11 / PyTorch 扩展：两种编译器各编一次（cpp 手册的第 14 章）====="
ext_try() {  # ext_try <标签> <编译器>：跑那一章的检查，把最像报错的几行摘进汇总
  local label="$1" cxx="$2" out rc
  log ""; log "\$ CXX=${cxx} cpp/tools/check_code.py docs/engineering/python-binding.md"
  out=$(cd cpp && CXX="$cxx" "$PY" tools/check_code.py docs/engineering/python-binding.md 2>&1); rc=$?
  printf '%s\n' "$out" >> "$LOG"
  log "   -> 退出码 ${rc}"
  local key
  key=$(printf '%s\n' "$out" | grep -E "error:|错误|失败|不同|No module|not found" | head -n 3 | cut -c1-160 | tr '\n' ' ')
  SUMMARY+=("$(mark "$rc") 扩展：${label}（退出码 ${rc}）${key:+ —— ${key}}")
}
ext_try "苹果 clang" /usr/bin/clang++
if [ -x "$BREW_CLANG" ]; then ext_try "Homebrew clang" "$BREW_CLANG"; fi

log ""; log "===== 3. torchrun + gloo：两种启动方式各跑一次（各 90 秒超时）====="
cat > build/tp_smoke.py <<'PYEOF'
import torch, torch.distributed as dist
dist.init_process_group("gloo")
t = torch.ones(1) * dist.get_rank()
dist.all_reduce(t)
print(f"rank {dist.get_rank()} ok, sum={t.item()}", flush=True)
dist.destroy_process_group()
PYEOF
TR_SUMMARY=$("$PY" - "$LOG" <<'PYEOF'
import os, subprocess, sys, time
log = open(sys.argv[1], "a", encoding="utf-8")
py = sys.executable
env = dict(os.environ, GLOO_SOCKET_IFNAME="lo0", PYTHONUNBUFFERED="1")
variants = [
    ("standalone", [py, "-m", "torch.distributed.run", "--standalone", "--nproc-per-node", "2", "build/tp_smoke.py"]),
    ("127.0.0.1 静态", [py, "-m", "torch.distributed.run", "--nproc-per-node", "2", "--master-addr", "127.0.0.1",
                       "--master-port", "29511", "build/tp_smoke.py"]),
]
def text(b):
    return b.decode("utf-8", "replace") if isinstance(b, bytes) else (b or "")
for name, cmd in variants:
    log.write(f"\n$ GLOO_SOCKET_IFNAME=lo0 {' '.join(cmd)}\n"); log.flush()
    t = time.time()
    try:
        r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=90)
        out = r.stdout + r.stderr
        ok = r.returncode == 0 and out.count("ok, sum=1.0") == 2
        log.write(out[-3000:] + f"\n   -> 退出码 {r.returncode}，{time.time() - t:.0f} 秒\n")
        tail = "" if ok else " —— " + " | ".join(l for l in out.strip().splitlines()[-3:])[:200]
        print(f"{'✓' if ok else '✗'} torchrun {name}（{time.time() - t:.0f} 秒，退出码 {r.returncode}）{tail}")
    except subprocess.TimeoutExpired as e:
        out = text(e.stdout) + text(e.stderr)
        log.write(out[-3000:] + "\n   -> 90 秒超时，杀掉\n")
        tail = " | ".join(l for l in out.strip().splitlines()[-3:])[:200]
        print(f"✗ torchrun {name}：90 秒没跑完（挂住）{(' —— ' + tail) if tail else ''}")
    log.flush()
PYEOF
)
printf '%s\n' "$TR_SUMMARY"
while IFS= read -r line; do [ -n "$line" ] && SUMMARY+=("$line"); done <<EOF
$TR_SUMMARY
EOF

log ""; log "===== 汇总（把这一段贴给维护者）====="
for s in "${SUMMARY[@]}"; do log "$s"; done
log "完整日志：${LOG}"
