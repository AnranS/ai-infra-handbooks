git show "$REF:pyproject.toml" | grep -nE '^\s*"(sglang|torch|transformers|flashinfer_python\[cu13\])==' | cut -c1-90
