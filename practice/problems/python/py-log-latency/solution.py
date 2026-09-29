import math
import re
from collections import defaultdict
from datetime import datetime

_KV = re.compile(r"(\w+)=(\S+)")
_PATH = re.compile(r"\s(/\S*)")


def _percentile(sorted_values, q):
    return sorted_values[max(0, math.ceil(q * len(sorted_values)) - 1)]


def summarize(lines, start=None, end=None):
    groups = defaultdict(list)
    for line in lines:
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            continue
        try:
            ts = datetime.fromisoformat(parts[0].replace("Z", "+00:00"))
            fields = dict(_KV.findall(parts[1]))
            path = _PATH.search(" " + parts[1]).group(1)
            status = int(fields["status"])
            latency = float(fields["latency_ms"])
            tokens = int(fields["tokens"])
        except (ValueError, KeyError, AttributeError):
            continue
        if (start is not None and ts < start) or (end is not None and ts >= end):
            continue
        groups[path].append((status, latency, tokens))
    result = {}
    for path, rows in groups.items():
        lat = sorted(r[1] for r in rows)
        total_ms = sum(lat)
        result[path] = {
            "count": len(rows),
            "errors": sum(r[0] >= 500 for r in rows),
            "p50": _percentile(lat, 0.5),
            "p99": _percentile(lat, 0.99),
            "tokens_per_s": round(sum(r[2] for r in rows) / (total_ms / 1000), 1) if total_ms else 0.0,
        }
    return result
