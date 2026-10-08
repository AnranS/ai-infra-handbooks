git show --stat=200 --format='%ad  %an  %s' --date=short 7bc1dae095 | sed -n '1p;$p' | cut -c1-96
echo "-- 按目录："; git show --stat=200 --format= 7bc1dae095 | awk '{print $1}' | grep '^python/sglang/multimodal_gen/' | cut -d/ -f4 | sort | uniq -c | sort -rn | head -8 | awk '{printf "   %3d  %s\n", $1, $2}'
git log --date=short --format='%ad  %h  %s' 29f6d408c0 | grep -i 'diffusion announcement' | cut -c1-96
