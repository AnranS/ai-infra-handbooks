# 练习题

五本手册配套的编程练习，网页版在站点的 `practice/`（顶栏「练习题」）。每道题都有描述、模板、测试和参考解答；每章末尾会自动列出本章的练习题。

## 三种做题方式

| 方式 | 适合 | 说明 |
| --- | --- | --- |
| 浏览器 | 所有标注"浏览器判题"的题 | Python 跑在 WebAssembly（Pyodide）里，打开网页就能写、能判题，进度保存在本地浏览器 |
| macOS（Apple Silicon） | 另外加上需要 PyTorch 的题 | PyTorch 用 MPS；Triton 题自动用模拟器；CUDA C++ 题用 CPU 模拟器只检查正确性 |
| WSL2 + NVIDIA GPU | 全部 | PyTorch / Triton / nvcc 跑在真卡上，CUDA C++ 题报告耗时和带宽 |

本地环境的安装见 [LOCAL.md](LOCAL.md)，一键脚本在 `env/`：

```bash
bash practice/env/setup-macos.sh                # macOS
INSTALL_CUDA=1 bash practice/env/setup-wsl2.sh  # WSL2 + NVIDIA GPU
source practice/.venv/bin/activate
python practice/judge.py doctor                 # 看本机能跑哪些题
python practice/judge.py start 12               # 复制第 12 题的模板到 practice/workspace/
python practice/judge.py test 12                # 判题
```

## 目录结构

```text
practice/
├── problems/<手册>/<题目 id>/
│   ├── problem.md      元数据（标题、章节、难度、标签、依赖）+ 题目描述 + <!-- 题解 --> 之后的讲解
│   ├── starter.py      模板（交给做题的人）
│   ├── solution.py     参考解答
│   ├── test.py         测试：test_* 函数（可以是 async），名字以 test_example 开头的是样例
│   └── *.cu            可选：CUDA C++ 版本的 starter / solution / test（test.cu 用 #include "user.cu"）
├── runtime/            判题内核，浏览器和本地共用
│   ├── judge_runner.py 执行测试、报告失败的断言和相关变量
│   ├── checker.py      check / check_close / raises / time_limit，以及运行环境判断
│   ├── gpusim.py       SIMT 模拟器：CUDA 风格的 kernel，检查越界、数据竞争、屏障，统计合并访存和 bank conflict
│   ├── minitl.py       Triton 模拟器（没有 NVIDIA GPU 时代替 triton.language）
│   ├── tritonkit.py    在真 Triton 和模拟器之间切换
│   └── cuda/judge.cuh  CUDA C++ 题的测试工具
├── app/                网页（题库列表 + 做题页 + Pyodide worker）
├── judge.py            本地判题命令行
├── cudajudge.py        CUDA C++ 题的判题：nvcc 真卡，或 CUDA 手册的 CPU 模拟器
├── build.py            生成 _site/practice（build.sh 会调用）
└── env/                macOS、WSL2 的安装脚本
```

## 新增一道题

1. 在 `problems/<手册>/` 下新建目录，写好四个文件；`chapter` 必须是该手册 `mkdocs.yml` 导航里的章节；
2. `python practice/judge.py check <id>`：参考解答必须通过所有测试、模板不能通过、至少有一个 `test_example*` 样例；
3. `python practice/build.py _site/practice` 生成网页数据；浏览器里跑一遍（第一次加载 Pyodide 需要联网）。

测试里的时间限制在浏览器里会自动放宽 3 倍；需要 PyTorch 的题在元数据里写 `requires: [torch]`，需要 NVIDIA GPU 的写 `requires: [cuda]`，需要线程等浏览器没有的功能写 `requires: [local]`。
