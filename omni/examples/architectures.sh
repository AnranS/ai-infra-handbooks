git grep -hE '^\s+architecture: ClassVar\[str( \| None)?\] = "' "$REF" -- 'sglang_omni/models/*/config.py' \
  | sed -E 's/.*= "([^"]+)".*/\1/' | sort | paste -sd' ' | fold -s -w 96
