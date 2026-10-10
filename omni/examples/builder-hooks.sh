git show "$REF:sglang_omni/scheduling/engine_factory.py" | awk 'NR>=129 && NR<=375' \
  | grep -oE 'self\.[a-z_]+\(' | awk '!seen[$0]++' | sed 's/($//' | paste -sd' ' | fold -s -w 96
