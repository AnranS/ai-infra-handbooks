REF=${REF:-29f6d408c0}
git log --reverse --date=short --format='%ad  %h  %s' "$REF" | awk '$1 >= "2024-07-01" && $1 <= "2024-09-10"' | grep -iE 'torch.?compile|compile' | cut -c1-96
