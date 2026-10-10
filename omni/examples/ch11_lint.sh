cd "$OMNI_TREE"
demo=sglang_omni/lint_demo.py
trap 'rm -f "$demo"' EXIT
cat > "$demo" <<'EOF'
class _Cache:
    def get(self, key):
        if key is None:
            return None
        return key
EOF
python3 scripts/check_if_else.py "$demo" 2>&1 || true
python3 scripts/check_leading_underscore.py "$demo" 2>&1 || true
echo "== --fix 之后的 if："
python3 scripts/check_if_else.py --fix "$demo" > /dev/null || true
sed -n '3,7p' "$demo"
