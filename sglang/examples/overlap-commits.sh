REF=${REF:-29f6d408c0}
git log --reverse --date=short --format='%ad  %h  %s' "$REF" | awk '$1 >= "2024-10-15" && $1 <= "2024-12-10"' | grep -iE 'overlap' | cut -c1-96
