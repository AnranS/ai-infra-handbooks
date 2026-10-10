echo "sglang_omni/ 里的 TODO / FIXME：$(git grep -cE '\b(TODO|FIXME)\b' "$REF" -- 'sglang_omni/*.py' | awk -F: '{s += $NF} END {print s}') 处（下面不列 vendor/ 里的）"
git grep -nE '\b(TODO|FIXME)\b' "$REF" -- 'sglang_omni/*.py' ':!sglang_omni/vendor/' | sed "s/^$REF://" | cut -c1-110
