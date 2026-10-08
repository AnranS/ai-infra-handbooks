REF=${REF:-29f6d408c0}
for d in $(git ls-tree --name-only "$REF" rust/ | grep -v '\.'); do printf '  %-24s %3d 个文件\n' "${d#rust/}" "$(git ls-tree -r --name-only "$REF" -- "$d" | wc -l)"; done
git log --reverse --date=short --format='%ad  %h  %s' "$REF" -- python/sglang/srt/rust_extensions | head -1 | cut -c1-96
git log --reverse --date=short --format='%ad  %h  %s' "$REF" -- rust/sglang-radix-tree | head -1 | cut -c1-96
