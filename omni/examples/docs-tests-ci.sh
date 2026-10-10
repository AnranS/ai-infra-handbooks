echo "docs/ 的栏目："; git ls-tree -d --name-only "$REF" docs/ | sed 's#docs/#  #'
echo "cookbook 篇数：$(git ls-tree --name-only "$REF" docs/cookbook/ | grep -c '\.md$')"
echo "单测目录数：$(git ls-tree -d --name-only "$REF" tests/unit_test/ | wc -l)"
echo "单测文件数：$(git ls-tree -r --name-only "$REF" tests/unit_test/ | grep -c '/test_.*\.py$')"
echo "CI workflow 数：$(git ls-tree --name-only "$REF" .github/workflows/ | wc -l)"
