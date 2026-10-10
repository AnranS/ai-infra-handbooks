"""列出没人认领、没有 open PR 引用的 issue（需要 gh 已登录）。"""
import json
import re
import subprocess

REPO = "sgl-project/sglang-omni"


def gh(*args):
    return json.loads(subprocess.run(["gh", *args], capture_output=True, text=True, check=True).stdout)


issues = gh("issue", "list", "-R", REPO, "--state", "open", "--limit", "1000",
            "--json", "number,title,assignees,comments,createdAt")
prs = gh("pr", "list", "-R", REPO, "--state", "open", "--limit", "1000", "--json", "number,title,body")
referenced = {int(n) for p in prs for n in re.findall(r"#(\d+)", f"{p['title']} {p['body'] or ''}")}
claim = re.compile(r"(?i)i'?d like to (take|work)|i'?ll take|working on (it|this)|assign (it|this) to me")
for i in sorted(issues, key=lambda i: -i["number"]):
    if re.search(r"(?i)roadmap|tracking|rfc", i["title"]):
        continue
    if i["assignees"] or i["number"] in referenced or any(claim.search(c["body"] or "") for c in i["comments"]):
        continue
    print(f"#{i['number']} {i['createdAt'][:10]} {i['title'][:90]}")
