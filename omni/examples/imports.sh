total=$(git ls-tree -r --name-only "$REF" -- sglang_omni | grep -c '\.py$')
direct=$(git grep -lE '^\s*(from sglang(\.| )|import sglang(\.| |$))' "$REF" -- 'sglang_omni/*.py' | wc -l)
vendor=$(git grep -lE 'sglang_omni\.vendor\.sglang' "$REF" -- 'sglang_omni/*.py' | wc -l)
echo "sglang_omni/ 下的 .py 文件：$total"
echo "直接 import sglang 的：      $direct"
echo "经 vendor 层 import 的：     $vendor"
