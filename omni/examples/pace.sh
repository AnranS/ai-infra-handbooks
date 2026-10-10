echo "月份      提交   作者"
git log "$REF" --date=format:%Y-%m --format='%ad|%aN' | awk -F'|' '{c[$1]++; if (!seen[$0]++) a[$1]++} END {for (m in c) printf "%s  %5d  %5d\n", m, c[m], a[m]}' | sort
echo "提交总数：$(git rev-list --count "$REF")；标题以 (#PR号) 结尾的：$(git log "$REF" --format=%s | grep -cE '\(#[0-9]+\)$')"
echo "第一个提交：$(git log --reverse --date=short --format='%ad %s' "$REF" | head -1)"
