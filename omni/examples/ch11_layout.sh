for lane in unit_test test_model test_ci; do
  n=$(git ls-tree -r --name-only "$REF" "tests/$lane/" | grep -c '/test_.*\.py$')
  printf '%-10s %4d 个测试文件\n' "$lane" "$n"
done
echo "单测按模块分的目录（前 12 个，按文件数）："
git ls-tree -r --name-only "$REF" tests/unit_test/ | grep '/test_.*\.py$' | cut -d/ -f3 | grep -v '\.py$' \
  | sort | uniq -c | sort -rn | head -12 | awk '{printf "  %-24s %3d\n", $2, $1}'
