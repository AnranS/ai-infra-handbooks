for d in $(git ls-tree -d --name-only "$REF" sglang_omni/models/ | sed 's#.*/##'); do
  n=$(git grep -lE '^\s*(from sglang(\.| )|import sglang(\.| |$))|sglang_omni\.(vendor\.sglang|scheduling\.(omni_scheduler|engine_factory)|model_runner)' "$REF" -- "sglang_omni/models/$d/" | wc -l)
  [ "$n" = 0 ] && echo "不用 SGLang：$d"
done
echo "模型目录总数：$(git ls-tree -d --name-only "$REF" sglang_omni/models/ | wc -l)"
