REF=${REF:-29f6d408c0}
echo "-- -S：引入和删除 'import rpyc' 的提交"
git log --reverse --date=short --format='%ad %h %s' -S'import rpyc' "$REF" -- python/sglang/srt | cut -c1-80
echo "-- -S：future_token_ids_map 第一次出现"
git log --reverse --date=short --format='%ad %h %s' -S'future_token_ids_map' "$REF" | head -1 | cut -c1-80
