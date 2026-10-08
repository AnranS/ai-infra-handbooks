REF=${REF:-29f6d408c0}
for t in v0.1.5 v0.2.0 v0.4.0; do printf '%-8s %3d 个路由（srt/server.py）\n' $t "$(git show $t:python/sglang/srt/server.py | grep -c '^@app\.')"; done
printf '%-8s %3d 个路由（srt/entrypoints/http_server.py）\n' "$REF" "$(git show "$REF:python/sglang/srt/entrypoints/http_server.py" | grep -c '^@app\.')"
