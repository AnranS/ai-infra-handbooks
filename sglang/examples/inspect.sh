echo "-- 一个提交改了哪些文件（-M 识别改名）："
git show --stat=90 -M --format='%ad %an %s' --date=short cdcbde5fc3 | sed -n '1p;/=>/p' | head -8 | cut -c1-90
echo "-- 某个版本的某个文件有多少行："
printf '%s\n' "$(git show v0.2.0:python/sglang/srt/managers/controller/tp_worker.py | wc -l) 行（tp_worker.py @ v0.2.0）"
echo "-- 两个版本之间某目录的提交数："
echo "$(git rev-list --count v0.4.0..v0.4.6 -- python/sglang/srt/mem_cache) 个（mem_cache/，v0.4.0 → v0.4.6）"
