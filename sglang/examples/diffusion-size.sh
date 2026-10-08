REF=${REF:-29f6d408c0}
echo "multimodal_gen/ 共 $(git ls-tree -r --name-only "$REF" -- python/sglang/multimodal_gen | wc -l) 个文件，.py $(git ls-tree -r --name-only "$REF" -- python/sglang/multimodal_gen | grep -c '\.py$') 个"
echo "runtime/ 一级目录：$(git ls-tree --name-only "$REF" python/sglang/multimodal_gen/runtime/ | sed 's|.*/||' | grep -v '\.py$' | tr '\n' ' ')"
echo "共享的入口：$(git show --stat=200 --format= 7bc1dae095 | awk '{print $1}' | grep -E '^python/sglang/(launch_server|cli)' | tr '\n' ' ')"
