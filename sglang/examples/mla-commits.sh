REF=${REF:-29f6d408c0}
git log --reverse --date=short --format='%ad  %h  %s' "$REF" | awk '$1 >= "2024-07-20" && $1 <= "2024-09-10"' | grep -iE 'mla|deepseek' | cut -c1-96
