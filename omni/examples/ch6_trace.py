"""在子进程里跑 ch6_pipeline.py，打开 SGLANG_OMNI_COMM_TRACE，整理出每条边用了什么传输方式。"""
import json
import os
import subprocess
import sys

EDGES = [("payload", "src", "near"), ("payload", "src", "far"), ("stream", "src", "far")]


def run(mode: str) -> None:
    env = dict(os.environ, SGLANG_OMNI_COMM_TRACE="1")
    out = subprocess.run([sys.executable, "ch6_pipeline.py", mode], env=env,
                         capture_output=True, text=True, timeout=600).stdout
    chosen, sizes = {}, []
    for line in out.splitlines():
        if "COMM_TRACE " in line:
            rec = json.loads(line.split("COMM_TRACE ", 1)[1])
            if rec["event"] == "comm_transport_selected":
                chosen[(rec["direction"], rec["stage"], rec["peer_stage"])] = rec["transport"]
            elif rec["event"] == "comm_stream_send":
                sizes.append((rec["bytes"], rec["transport"]))
    print(f"== {mode}")
    for edge in EDGES:
        print(f"  {edge[0]:7s} {edge[1]} → {edge[2]:5s} {chosen.get(edge, '（没经过通信层）')}")
    for nbytes, transport in sizes:
        print(f"  流式块 {nbytes} 字节 → {transport}")
    result = [l for l in out.splitlines() if l.startswith("RESULT")]
    errors = [l.strip() for l in out.splitlines() if l.startswith("ValueError")]
    print("  结果：", result[0][7:] if result else errors[0] if errors else "（无）")


run("small")
run("big")
