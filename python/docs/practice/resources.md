# 延伸阅读

<p class="lead">这份手册覆盖了"熟练使用"需要的核心内容。想继续深入，下面是经过筛选的资料：官方文档是最权威的参考，几本书能帮你建立系统的理解，练习平台让你保持手感。</p>

## 官方文档

[docs.python.org](https://docs.python.org/zh-cn/3/)（有中文版）是最准确、最新的资料，值得花时间熟悉它的结构：

| 部分 | 什么时候看 |
| --- | --- |
| [教程](https://docs.python.org/zh-cn/3/tutorial/index.html) | 查漏补缺，快速过一遍 |
| [标准库参考](https://docs.python.org/zh-cn/3/library/index.html) | 日常查阅最多的部分；用一个模块前先看看它的文档 |
| [语言参考](https://docs.python.org/zh-cn/3/reference/index.html) | 想知道某个语法的精确含义时 |
| [HOWTO 指南](https://docs.python.org/zh-cn/3/howto/index.html) | 专题深入，强烈推荐：[描述符指南](https://docs.python.org/zh-cn/3/howto/descriptor.html)、[日志指南](https://docs.python.org/zh-cn/3/howto/logging.html)、[函数式编程](https://docs.python.org/zh-cn/3/howto/functional.html)、[排序指南](https://docs.python.org/zh-cn/3/howto/sorting.html)、[asyncio 概念](https://docs.python.org/zh-cn/3/howto/a-conceptual-overview-of-asyncio.html) |
| [What's New](https://docs.python.org/zh-cn/3/whatsnew/index.html) | 每次升级版本时 |
| [数据模型](https://docs.python.org/zh-cn/3/reference/datamodel.html) | 所有特殊方法的权威说明 |

值得读的 PEP：

- [PEP 8](https://peps.python.org/pep-0008/)：代码风格指南
- [PEP 20](https://peps.python.org/pep-0020/)：Python 之禅（在 REPL 里 `import this`）
- [PEP 257](https://peps.python.org/pep-0257/)：文档字符串约定
- [PEP 484](https://peps.python.org/pep-0484/)：类型标注的起点
- [PEP 636](https://peps.python.org/pep-0636/)：模式匹配教程

类型系统的规范和最佳实践集中在 [typing.python.org](https://typing.python.org/)。

## 书

| 书 | 适合 |
| --- | --- |
| 《流畅的 Python》（Fluent Python，第 2 版），Luciano Ramalho | **最推荐**。深入讲解数据模型、函数、对象、并发，和本手册的方向完全一致，是进阶的必读书 |
| 《Effective Python》，Brett Slatkin | 一条一条的最佳实践，每条都很短，适合碎片时间读，选最新版 |
| 《Robust Python》，Patrick Viafore | 类型标注和可维护的代码设计 |
| 《Architecture Patterns with Python》，Harry Percival & Bob Gregory | 在 Python 里做领域驱动设计、依赖倒置、仓储模式，适合写业务系统的人；有免费在线版 |
| 《High Performance Python》，Micha Gorelick & Ian Ozsvald | 性能分析和优化 |
| 《Python Testing with pytest》，Brian Okken | pytest 的系统讲解 |

## 演讲

这些演讲都能在 YouTube 上找到，每一个都值得看：

- **Ned Batchelder — Facts and Myths about Python Names and Values**：把[对象模型](../core/data-model.md)讲得最清楚的演讲。
- **Raymond Hettinger — Transforming Code into Beautiful, Idiomatic Python**：几十个地道写法的例子。
- **Raymond Hettinger — Beyond PEP 8**：什么才是真正的好代码。
- **David Beazley — Generators: The Final Frontier** 及他的其他生成器系列：把生成器用到极致。
- **Łukasz Langa — AsyncIO 系列视频**（EdgeDB 频道）：从零讲 asyncio。

## 读源码

读优秀的代码是提高最快的方法之一。标准库本身就是很好的材料（在编辑器里跳转到定义就能看到）：

| 模块 | 能学到什么 |
| --- | --- |
| `collections/__init__.py` | `namedtuple`、`OrderedDict`、`Counter`、`ChainMap` 的实现 |
| `functools.py` | `wraps`、`partial`、`lru_cache`、`singledispatch`、`cached_property` |
| `contextlib.py` | `contextmanager`、`ExitStack` 的实现 |
| `dataclasses.py` | 类装饰器如何动态生成方法 |
| `pathlib/` | 一个设计良好的面向对象 API |

第三方库推荐读 [httpx](https://github.com/encode/httpx)（清晰的同步/异步双 API 设计）、[rich](https://github.com/Textualize/rich)（大量特殊方法和协议的运用）、[attrs](https://github.com/python-attrs/attrs)（dataclass 的前身）。

## 练习平台

- [Exercism Python 路线](https://exercism.org/tracks/python)：有导师点评，特别适合练习"地道的写法"。
- [LeetCode](https://leetcode.cn/)：刷算法的同时练习 `collections`、`heapq`、`bisect`、`itertools` 的使用。
- [Advent of Code](https://adventofcode.com/)：每年 12 月的编程谜题，题目有趣，很适合练习数据处理和生成器。
- [Project Euler](https://projecteuler.net/)：数学向的编程题。

## 保持更新

- [Python 官方博客](https://blog.python.org/) 和 [discuss.python.org](https://discuss.python.org/)：新版本和 PEP 的讨论。
- [Real Python](https://realpython.com/)：高质量的教程文章。
- [PyCoder's Weekly](https://pycoders.com/)、[Python Weekly](https://www.pythonweekly.com/)：每周的文章和项目精选。
- [Talk Python To Me](https://talkpython.fm/) 播客。
