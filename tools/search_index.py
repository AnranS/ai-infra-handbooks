"""把七本手册的 MkDocs 搜索索引、练习题和总入口页合并成一份全站搜索索引。

用法：python tools/search_index.py _site    （build.sh 在各手册和练习题都构建完之后调用）
输出 _site/search/index.json：{"books": [...], "docs": [[book, url, 页面标题, 小节标题, 正文], ...]}
"""

from __future__ import annotations

import html
import json
import re
import sys
from pathlib import Path

BOOKS = [("python", "Python 进阶"), ("cpp", "C++ 进阶"), ("llm", "大模型原理"), ("cuda", "CUDA 进阶"), ("train", "分布式训练"), ("serving", "推理系统"),
         ("minisgl", "手写 mini-sglang"), ("practice", "练习题"), ("site", "总览")]
PORTAL = [
    ("roadmap/", "学习路线图", "", "把七本手册的 196 章按 17 周排好的学习路线（与冲刺计划逐周对应）：必学、推荐、选学，按推理框架、推理优化、推理平台三个方向标出重点，记录学习进度"),
    ("setup/", "学习环境：在 Mac 上学", "", "Mac 上一键准备全部手册的学习环境（env/setup-macos.sh）与自检（tools/mac_check.py）；每本书在 Mac 上的运行情况：C++ 的 clang 与 LeakSanitizer、CUDA 的 CPU 模拟器、Triton、gloo、MPS；需要 NVIDIA GPU 的部分与租云 GPU、WSL2 的替代方案"),
    ("plan/", "17 周冲刺计划", "", "推理系统岗求职冲刺计划：目标能力、学习目标分级与结果验证、三类岗位的侧重、参考课程与教程（CS336、Scaling Book、Ultra-Scale Playbook、LeetCUDA、GPU MODE 等）、逐周学习与练习、里程碑、作品与开源贡献、算法题、手撕组件、系统设计、论文精读清单、简历与面试节奏"),
    ("practice/#/?q=估算", "估算题库", "", "26 道估算题：训练算力与 MFU、训练显存与 ZeRO、KV Cache 账本、MoE 激活参数、decode 注意力计算访存比、decode 与 prefill 下限、GEMM 波次量化与分块访存、张量并行与专家并行通信、PD 分离 KV 传输、流水线气泡、投机解码加速比、部署规模与容量规划、端侧 decode 速度与量化格式"),
    ("practice/", "练习题", "", "七本手册配套的编程题，浏览器里写代码一键判题，GPU 模拟器检查合并访存与 bank conflict，支持 macOS 与 WSL2 本地判题"),
    ("cards/", "学习卡", "", "从七本手册的自测题、练习和面试题库里抽出的学习卡：间隔重复复习、按手册筛选、导出到 Anki"),
    ("playground/", "Playground", "", "在浏览器里直接运行 Python 的沙盒：numpy、matplotlib、CUDA 模拟器 gpusim、Triton 模拟器，代码补全、函数签名与语法检查，模板与分享链接"),
]


def clean(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text).replace("​", "")
    return re.sub(r"\s+", " ", text).strip()


def main(site: Path) -> None:
    ids = {b: i for i, (b, _) in enumerate(BOOKS)}
    docs = []
    chapter_titles = {}
    for book, _ in BOOKS[:7]:
        index = json.loads((site / book / "search" / "search_index.json").read_text())
        pages = {}
        for d in index["docs"]:
            if "#" not in d["location"]:
                pages[d["location"]] = clean(d["title"])
                chapter_titles[f"{book}/{d['location']}"] = pages[d["location"]]
        for d in index["docs"]:
            loc = d["location"]
            page = loc.split("#")[0]
            title = pages.get(page, "")
            section = clean(d["title"]) if "#" in loc else ""
            text = clean(d.get("text", ""))
            if not text and not section:
                continue
            docs.append([ids[book], f"{book}/{loc}", title, section, text])
    practice = json.loads((site / "practice" / "data" / "index.json").read_text())
    for p in practice["problems"]:
        chapter = p.get("chapter") or ""
        chapter = re.sub(r"\.md$", "/", re.sub(r"(^|/)index\.md$", r"\1", chapter))
        chapter = chapter_titles.get(f"{p['book']}/{chapter}", "")
        text = " ".join(filter(None, [chapter, " ".join(p.get("tags", [])), p.get("difficulty", "")]))
        docs.append([ids["practice"], f"practice/#/p/{p['slug']}", f"{p['number']}. {p['title']}", "", text])
    for url, title, section, text in PORTAL:
        docs.append([ids["site"], url, title, section, text])
    out = site / "search" / "index.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"books": [name for _, name in BOOKS], "docs": docs}, ensure_ascii=False, separators=(",", ":")))
    print(f"search: {len(docs)} 条 -> {out}（{out.stat().st_size // 1024} KB）")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "_site"))
