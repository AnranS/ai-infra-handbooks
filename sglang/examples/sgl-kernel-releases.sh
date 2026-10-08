REF=${REF:-29f6d408c0}
echo "2025 年里标题含 sgl-kernel 且含 bump/release/version 的提交：$(git log --date=short --format='%ad %s' "$REF" | grep '^2025' | grep -i 'sgl-kernel' | grep -icE 'bump|release|version')"
echo "sgl-kernel 相关提交总数（标题含 sgl-kernel）：$(git log --format=%s "$REF" | grep -ic 'sgl-kernel')"
