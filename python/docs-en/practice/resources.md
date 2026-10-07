# Further reading

<p class="lead">This handbook covers the core of what "using Python fluently" takes. To go deeper, here is a filtered list: the official documentation is the authoritative reference, a few books build a systematic understanding, and the practice sites keep your hand in.</p>

## The official documentation {#官方文档}

[docs.python.org](https://docs.python.org/3/) is the most accurate and most current material, and its structure is worth becoming familiar with:

| Part | When to read it |
| --- | --- |
| [the tutorial](https://docs.python.org/3/tutorial/index.html) | filling gaps, as a quick pass |
| [the library reference](https://docs.python.org/3/library/index.html) | what you look at most day to day; read a module's documentation before using it |
| [the language reference](https://docs.python.org/3/reference/index.html) | when you want a piece of syntax's exact meaning |
| [the HOWTOs](https://docs.python.org/3/howto/index.html) | deep dives on one topic, strongly recommended: [the descriptor guide](https://docs.python.org/3/howto/descriptor.html), [the logging guide](https://docs.python.org/3/howto/logging.html), [functional programming](https://docs.python.org/3/howto/functional.html), [the sorting guide](https://docs.python.org/3/howto/sorting.html), [a conceptual overview of asyncio](https://docs.python.org/3/howto/a-conceptual-overview-of-asyncio.html) |
| [What's New](https://docs.python.org/3/whatsnew/index.html) | every time you upgrade |
| [the data model](https://docs.python.org/3/reference/datamodel.html) | the authoritative account of every special method |

The PEPs worth reading:

- [PEP 8](https://peps.python.org/pep-0008/): the style guide
- [PEP 20](https://peps.python.org/pep-0020/): the Zen of Python (`import this` in the REPL)
- [PEP 257](https://peps.python.org/pep-0257/): docstring conventions
- [PEP 484](https://peps.python.org/pep-0484/): where type annotations began
- [PEP 636](https://peps.python.org/pep-0636/): the pattern matching tutorial

The type system's specification and best practices are collected at [typing.python.org](https://typing.python.org/).

## Books {#书}

| Book | Suits |
| --- | --- |
| *Fluent Python* (2nd edition), Luciano Ramalho | **the first recommendation**. A deep treatment of the data model, functions, objects and concurrency, pointing in exactly this handbook's direction and essential reading to go further |
| *Effective Python*, Brett Slatkin | best practices one item at a time, each short enough to read in a spare moment; take the newest edition |
| *Robust Python*, Patrick Viafore | type annotations and designing maintainable code |
| *Architecture Patterns with Python*, Harry Percival & Bob Gregory | domain-driven design, dependency inversion and the repository pattern in Python, which suits people writing business systems; there is a free online edition |
| *High Performance Python*, Micha Gorelick & Ian Ozsvald | profiling and optimization |
| *Python Testing with pytest*, Brian Okken | a systematic treatment of pytest |

## Talks {#演讲}

All of these are on YouTube and every one is worth watching:

- **Ned Batchelder, Facts and Myths about Python Names and Values**: the clearest talk there is on [the object model](../core/data-model.md).
- **Raymond Hettinger, Transforming Code into Beautiful, Idiomatic Python**: dozens of examples of idiomatic form.
- **Raymond Hettinger, Beyond PEP 8**: what genuinely good code is.
- **David Beazley, Generators: The Final Frontier** and the rest of his generator series: generators taken to their limit.
- **Łukasz Langa, the AsyncIO video series** (the EdgeDB channel): asyncio from the beginning.

## Reading source {#读源码}

Reading good code is one of the fastest ways to improve. The standard library itself is excellent material (jump to the definition in your editor and there it is):

| Module | What there is to learn |
| --- | --- |
| `collections/__init__.py` | how `namedtuple`, `OrderedDict`, `Counter` and `ChainMap` are implemented |
| `functools.py` | `wraps`, `partial`, `lru_cache`, `singledispatch`, `cached_property` |
| `contextlib.py` | how `contextmanager` and `ExitStack` are implemented |
| `dataclasses.py` | how a class decorator generates methods dynamically |
| `pathlib/` | a well-designed object-oriented API |

Among third-party libraries, [httpx](https://github.com/encode/httpx) (a clear design with parallel synchronous and asynchronous APIs), [rich](https://github.com/Textualize/rich) (heavy use of the special methods and protocols) and [attrs](https://github.com/python-attrs/attrs) (the dataclass's ancestor) are worth reading.

## Places to practise {#练习平台}

- [Exercism's Python track](https://exercism.org/tracks/python): with mentor feedback, which suits practising "the idiomatic form" particularly well.
- [LeetCode](https://leetcode.com/): algorithms alongside practice with `collections`, `heapq`, `bisect` and `itertools`.
- [Advent of Code](https://adventofcode.com/): December's programming puzzles, which are enjoyable and suit practising data processing and generators.
- [Project Euler](https://projecteuler.net/): mathematically inclined programming problems.

## Keeping up {#保持更新}

- [The official Python blog](https://blog.python.org/) and [discuss.python.org](https://discuss.python.org/): new versions and the PEP discussions.
- [Real Python](https://realpython.com/): high-quality tutorial articles.
- [PyCoder's Weekly](https://pycoders.com/) and [Python Weekly](https://www.pythonweekly.com/): a weekly selection of articles and projects.
- The [Talk Python To Me](https://talkpython.fm/) podcast.
