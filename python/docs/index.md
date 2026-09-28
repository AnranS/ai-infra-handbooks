# Python 进阶手册

<p class="lead">写给已经会写 Python 的人。目标是让你从"能写出来"走到"写得地道、写得稳，并且知道为什么这样写"。</p>

## 这份手册适合谁

你大概处在这个阶段：

- 函数、类、循环、推导式都会写，能用 pip 装包，能写脚本解决手头的问题。
- 但看到装饰器、生成器、描述符、`asyncio` 时心里没底，读别人的库代码有点吃力。
- 代码能跑，但总觉得不够 Pythonic；不太清楚项目该怎么组织、测试该怎么写、类型标注怎么用。

读完并练完这份手册，你应该能做到：

- 说清楚 Python 对象模型：名字、引用、可变性、拷贝、哈希，写代码时不再踩这些坑。
- 熟练使用装饰器、生成器、上下文管理器、dataclass、类型标注，能读懂主流库的源码。
- 根据问题选对并发模型（线程、进程、`asyncio`），并能写出正确的并发代码。
- 用 uv、ruff、mypy、pytest 搭起一个像样的工程，写出可维护、可测试的代码。

## 怎么用

1. **先做自测。** 每章开头有一个自测框，全部能答上来就可以跳过这一章，直接去做练习。
2. **代码亲手敲。** 不要只看。把示例敲进 REPL，改几个参数看结果怎么变。
3. **先做练习再看答案。** 每章末尾有练习，答案默认折叠。卡住超过 20 分钟再展开。
4. **每个阶段做一个项目。** [综合项目](practice/projects.md)里按阶段列了 6 个项目，这是从"看懂"到"会用"的关键一步。

## 学习路线

四本手册合在一起的逐章路线（12 周计划、每章是必学还是选学、不同岗位方向的重点、跨书的知识依赖）见[学习路线图](root://roadmap/)。下面是本书内部的顺序。

按顺序读最稳妥。如果时间紧，第一阶段和第四阶段优先。

<div class="roadmap" markdown>

| 阶段 | 章节 | 学完能做什么 | 建议用时 |
| --- | --- | --- | --- |
| 一、核心语言 | [对象模型](core/data-model.md) · [容器](core/containers.md) · [函数](core/functions.md) · [装饰器](core/decorators.md) · [迭代器与生成器](core/iterators.md) | 写出地道的 Python，不踩引用和可变性的坑 | 1～2 周 |
| 二、面向对象与建模 | [面向对象](core/oop.md) · [数据建模](core/data-classes.md) · [异常与上下文管理器](core/errors-context.md) · [模式匹配](core/pattern-matching.md) | 设计清晰的类和数据结构，正确处理错误和资源 | 1 周 |
| 三、类型与抽象 | [类型标注](types/typing.md) · [协议与抽象基类](types/protocols.md) · [元编程](types/metaprogramming.md) | 给代码加上有用的类型，读懂框架里的"魔法" | 1 周 |
| 四、工程化 | [常用标准库](engineering/stdlib.md) · [项目与工具链](engineering/tooling.md) · [pytest](engineering/testing.md) | 独立搭建、测试、发布一个 Python 项目 | 1 周 |
| 五、并发与性能 | [线程与进程](concurrency/threads-processes.md) · [asyncio](concurrency/asyncio.md) · [性能优化](concurrency/performance.md) | 选对并发模型，定位并解决性能问题 | 1～2 周 |
| 六、查漏补缺 | [惯用法与常见坑](practice/pitfalls.md) · [新版本特性](practice/whats-new.md) · [延伸阅读](practice/resources.md) | 查漏补缺，跟上语言的变化 | 随时 |

</div>

## 版本约定

- 以 **Python 3.12** 为基线。只在更新版本里才有的特性会标出版本，比如 <span class="since">3.13+</span>。
- 所有 `python` 代码块和 `>>>` 交互示例都在 **Python 3.14.7** 上自动跑过，输出与页面一致。
- 带 `>>>` 的是交互式会话，照着敲进 REPL 就能看到同样的结果；不带的是完整代码，可以存成 `.py` 文件运行。

## 开始之前：准备环境

推荐用 [uv](https://docs.astral.sh/uv/) 管理 Python 版本和虚拟环境，它比 pyenv + pip 快得多，也更省心。

```bash
# 安装 uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 安装最新的 Python，并建一个练习用的虚拟环境
uv python install 3.14
mkdir py-practice && cd py-practice
uv venv --python 3.14
source .venv/bin/activate

# 3.13 起的新 REPL 支持多行编辑、历史记录和彩色输出
python
```

REPL 里几个好用的东西：

- `help(obj)` 看文档，`dir(obj)` 看有哪些属性。
- `_` 是上一个表达式的结果。
- 在新 REPL 里按 ++f3++ 进入粘贴模式，可以一次粘贴一大段代码。
- `python -i script.py` 运行完脚本后停在 REPL 里，方便检查变量。
