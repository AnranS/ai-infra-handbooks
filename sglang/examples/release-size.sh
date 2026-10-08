git show --stat=10 --format= 22085081bb | tail -1
git ls-tree -r --name-only 22085081bb | cut -d/ -f1 | sort | uniq -c | sort -rn
