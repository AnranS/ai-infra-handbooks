git log --date=short --format='%ad %h %s' "$REF" | grep -E 'Bump SGLang to' | while read -r d h rest; do
  printf '%s %s %-40s %3d 个文件\n' "$d" "$h" "$rest" "$(git show --name-only --format= "$h" | grep -c .)"
done
