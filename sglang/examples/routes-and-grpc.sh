REF=${REF:-29f6d408c0}
for t in v0.4.6 v0.5.0rc0 "$REF"; do printf '%-11s %2d 个 HTTP 路由\n' "$t" "$(git show "$t:python/sglang/srt/entrypoints/http_server.py" | grep -c '^@app\.')"; done
echo "gRPC 相关文件：$(git ls-tree -r --name-only "$REF" -- python/sglang/srt/entrypoints python/sglang/srt/grpc proto | grep -iE 'grpc|\.proto$' | sed 's|.*/||' | tr '\n' ' ')"
