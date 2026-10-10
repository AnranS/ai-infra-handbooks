echo "build_relay 构造的后端："
git grep -n -A1 'create_relay($' "$REF" -- sglang_omni/comm/router.py | grep -oE '"[a-z_]+"' | tr '\n' ' '; echo
echo "relay/ 目录里注册的后端："
git grep -hoE '^@register_relay\("[a-z_]+"\)' "$REF" -- sglang_omni/relay/ | grep -oE '"[a-z_]+"' | sort | tr '\n' ' '; echo
