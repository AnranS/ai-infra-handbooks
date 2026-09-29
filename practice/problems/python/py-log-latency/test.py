import random
from datetime import datetime, timedelta, timezone

from checker import check
from solution import summarize

LOG = """\
2026-05-17T10:00:01.250Z INFO req=a1 POST /v1/chat/completions status=200 latency_ms=812.5 tokens=256
2026-05-17T10:00:01.900Z WARN req=a2 POST /v1/completions status=429 latency_ms=3.1 tokens=0
2026-05-17T10:00:02.000Z INFO req=a3 POST /v1/chat/completions status=200 latency_ms=950.0 tokens=300
2026-05-17T10:00:03.000Z ERROR req=a4 POST /v1/completions status=503 latency_ms=10.0 tokens=0
garbage line
2026-05-17T10:00:04.000Z INFO req=a5 POST /v1/completions status=200 latency_ms=oops tokens=5
""".splitlines()


def test_example():
    got = summarize(LOG)
    check(sorted(got), ["/v1/chat/completions", "/v1/completions"], "接口列表")
    check(got["/v1/chat/completions"], {"count": 2, "errors": 0, "p50": 812.5, "p99": 950.0, "tokens_per_s": 315.5},
          "/v1/chat/completions 的统计")
    check(got["/v1/completions"], {"count": 2, "errors": 1, "p50": 3.1, "p99": 10.0, "tokens_per_s": 0.0},
          "/v1/completions 的统计（格式错误的行被跳过）")


def test_time_window():
    utc = timezone.utc
    got = summarize(LOG, start=datetime(2026, 5, 17, 10, 0, 1, 500000, tzinfo=utc),
                    end=datetime(2026, 5, 17, 10, 0, 3, tzinfo=utc))
    check(got["/v1/chat/completions"]["count"], 1, "窗口内 chat 请求数")
    check(got["/v1/completions"]["count"], 1, "窗口内 completions 请求数（end 不包含）")


def test_field_order_and_percentiles():
    rng = random.Random(0)
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    lat = [rng.uniform(1, 1000) for _ in range(1000)]
    lines = []
    for i, l in enumerate(lat):
        ts = (base + timedelta(seconds=i)).isoformat().replace("+00:00", "Z")
        fields = [f"status={200 if i % 10 else 500}", f"latency_ms={l:.3f}", "tokens=10", f"req=r{i}"]
        rng.shuffle(fields)
        lines.append(f"{ts} INFO GET /health " + " ".join(fields))
    got = summarize(lines)["/health"]
    s = sorted(float(f"{x:.3f}") for x in lat)
    check((got["count"], got["errors"]), (1000, 100), "count / errors")
    check(got["p50"], s[499], "p50（最近秩法：第 500 个）")
    check(got["p99"], s[989], "p99（第 990 个）")


def test_empty():
    check(summarize([]), {}, "没有日志")
    check(summarize(["not a log"]), {}, "全是格式错误的行")
