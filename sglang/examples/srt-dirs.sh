for t in v0.2.0 v0.3.0 v0.4.0; do
  echo "== $t"
  git ls-tree -r --name-only $t -- python/sglang/srt | grep '\.py$' | awk -F/ '{print (NF > 4 ? $4 "/" : $4)}' | sort | uniq -c | sort -rn | awk '{printf "   %3d %s\n", $1, $2}'
done
