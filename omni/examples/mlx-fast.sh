git grep -hoE 'mx\.fast\.[a-z_]+' "$REF" -- sglang_omni sglang_omni_mlx | sort | uniq -c
echo ".metal 文件数：$(git ls-tree -r --name-only "$REF" | grep -c '\.metal$')"
