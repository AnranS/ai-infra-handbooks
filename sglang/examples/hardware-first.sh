REF=${REF:-29f6d408c0}
git log --reverse --date=short --format='%ad  %h  %s' "$REF" | grep -iE '^\S+\s+\S+\s+.*(\bamd\b|rocm)' | head -1 | cut -c1-96
git log --reverse --date=short --format='%ad  %h  %s' "$REF" | grep -iE 'ascend|\bnpu\b' | head -1 | cut -c1-96
git log --reverse --date=short --format='%ad  %h  %s' "$REF" | grep -iE '\bcpu backend\b|intel.*cpu|cpu.*intel' | head -1 | cut -c1-96
git log --reverse --date=short --format='%ad  %h  %s' "$REF" | grep -iE '\bxpu\b' | head -1 | cut -c1-96
echo "hardware_backend/ 子目录：$(git ls-tree --name-only "$REF" python/sglang/srt/hardware_backend/ | sed 's|.*/||' | tr '\n' ' ')"
