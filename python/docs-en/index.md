# Advanced Python

<p class="lead">Written for people who already write Python. The aim is to take you from "I can get it working" to "I write it idiomatically and soundly, and I know why".</p>

## Who this handbook is for {#这份手册适合谁}

You are probably around here:

- you can write functions, classes, loops and comprehensions, install packages with pip, and write a script to solve the problem in front of you.
- but decorators, generators, descriptors and `asyncio` leave you unsure, and reading other people's library code is a bit of a struggle.
- your code runs, but never feels Pythonic enough; and how a project should be organized, how tests should be written and how type annotations are used are not clear.

Having read this handbook and done the exercises, you should be able to:

- explain Python's object model: names, references, mutability, copying and hashing, so that these traps stop catching you.
- use decorators, generators, context managers, dataclasses and type annotations fluently, and read the source of the mainstream libraries.
- pick the right concurrency model for the problem (threads, processes, `asyncio`) and write concurrent code that is correct.
- build a proper project with uv, ruff, mypy and pytest, and write code that is maintainable and testable.

## How to use it {#怎么用}

1. **Take the self-test first.** Every chapter opens with one, and if you can answer all of it, skip the chapter and go straight to the exercises.
2. **Type the code yourself.** Do not just read it. Put the examples into the REPL and change a few arguments to see what happens.
3. **Do the exercises before reading the answers.** Every chapter ends with exercises whose answers are collapsed. Expand one only after being stuck for 20 minutes.
4. **Do a project per stage.** [The projects](practice/projects.md) lists 6 of them by stage, which is the step from "I follow it" to "I can use it".

## The learning path {#学习路线}

For the chapter-by-chapter route across all the handbooks (17 weeks, matching the sprint plan week by week, with core and optional chapters, key chapters for different directions, and dependencies across books), see the [roadmap](root://roadmap/). Below is the order within this book.

Reading in order is the safest. If time is short, stages one and four come first.

<div class="roadmap" markdown>

| Stage | Chapters | What you can do afterwards | Suggested time |
| --- | --- | --- | --- |
| 1. The core language | [the object model](core/data-model.md) · [containers](core/containers.md) · [functions](core/functions.md) · [decorators](core/decorators.md) · [iterators and generators](core/iterators.md) | write idiomatic Python without falling into the traps of references and mutability | 1-2 weeks |
| 2. Objects and modelling | [object orientation](core/oop.md) · [modelling data](core/data-classes.md) · [exceptions and context managers](core/errors-context.md) · [pattern matching](core/pattern-matching.md) | design clear classes and data structures, and handle errors and resources correctly | 1 week |
| 3. Types and abstraction | [type annotations](types/typing.md) · [protocols and abstract base classes](types/protocols.md) · [metaprogramming](types/metaprogramming.md) | give your code types that earn their keep, and read the "magic" in a framework | 1 week |
| 4. Engineering | [the standard library](engineering/stdlib.md) · [projects and the toolchain](engineering/tooling.md) · [pytest](engineering/testing.md) | build, test and publish a Python project on your own | 1 week |
| 5. Concurrency and performance | [threads and processes](concurrency/threads-processes.md) · [asyncio](concurrency/asyncio.md) · [optimizing performance](concurrency/performance.md) | pick the right concurrency model, and locate and fix a performance problem | 1-2 weeks |
| 6. Filling the gaps | [idioms and common traps](practice/pitfalls.md) · [what is new in recent versions](practice/whats-new.md) · [further reading](practice/resources.md) | fill the gaps and keep up with the language | any time |

</div>

## Version conventions {#版本约定}

- **Python 3.12** is the baseline. Anything that needs a newer version is marked with it, like <span class="since">3.13+</span>.
- Every `python` code block and `>>>` session has been run automatically on **Python 3.14.7**, and the output matches the page.
- A block with `>>>` is an interactive session, and typing it into the REPL gives the same result; one without is complete code that can be saved as a `.py` file and run.

## Before you start: setting up {#开始之前准备环境}

[uv](https://docs.astral.sh/uv/) is the recommended way to manage Python versions and virtual environments; it is far faster than pyenv + pip and takes less looking after.

```bash
# install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# install the latest Python and create a virtual environment to practise in
uv python install 3.14
mkdir py-practice && cd py-practice
uv venv --python 3.14
source .venv/bin/activate

# the new REPL from 3.13 supports multi-line editing, history and colour
python
```

A few useful things in the REPL:

- `help(obj)` shows the documentation and `dir(obj)` the attributes.
- `_` is the last expression's result.
- In the new REPL, ++f3++ enters paste mode, where a large block can be pasted at once.
- `python -i script.py` runs the script and stops in the REPL, which is handy for inspecting the variables.
