# The projects

<p class="lead">A great deal of practice separates following something from being able to use it. The 6 projects below are ordered by stage and each covers several chapters' material. Every one states the requirements, the acceptance criteria and some hints, and none gives a complete answer: when you get stuck, go back to the chapter for the idea, which is the whole point of practising.</p>

!!! tip "How to approach a project"
    - **Make it work, then make it good.** The first version implements only the core and confirms it is usable, and error handling, tests and type annotations come afterwards.
    - **Build every project as a full project with uv**: the src layout, ruff, mypy, pytest; see [projects and the toolchain](../engineering/tooling.md).
    - **Review your own code when it is finished**, against the checklist at the end of [idioms and common traps](pitfalls.md).
    - **Only the "going further" part counts as really having it.** Going further usually exposes a problem in the first version's design and forces a refactor.

## Project 1: a command-line ledger {#项目一命令行记账本}

**Stage**: 1, the core language · **Topics**: dataclasses, Enum, `pathlib`, `json`, `argparse`, `datetime`, exception handling

A command-line tool recording personal income and spending, with the data in a JSON file.

<!-- i18n:diagram 68d18de4c9 -->
```text
$ ledger add -50 --category food --note "lunch"
$ ledger add 8000 --category salary
$ ledger list --month 2026-09
$ ledger report --month 2026-09
income 8000.00 | spending 50.00 | balance 7950.00
food      50.00  100.0%
```

**Requirements**

- `add`: an amount (positive for income, negative for spending), a category, a note and a date (today by default)
- `list`: filtered by month and category, sorted by date
- `report`: one month's income, spending and balance, plus the spending by category as percentages
- `delete ID`: remove one record

**Acceptance criteria**

- Amounts are stored and computed as `Decimal` and written to JSON as strings
- The data file is created when it does not exist; a corrupted file gives a clear error message rather than a traceback
- Categories come from an `Enum` or a configurable list, and an unknown category raises and lists the valid ones
- The core logic (statistics, filtering) is kept apart from the command line and the file I/O, and is tested

**Going further**

- Support exporting CSV and importing a bank statement's CSV
- Replace the storage with `sqlite3` without the command-line code changing one line (a hint: define a Protocol for the storage)

## Project 2: a large log file analyzer {#项目二大日志文件分析器}

**Stage**: 1, the core language · **Topics**: generator pipelines, `re`, `Counter`, `itertools`, `heapq`, performance

Analyze Nginx access logs (write a script of your own to generate a few million lines of test data).

**Requirements**

- Statistics: the total requests, the count per status code, the top 10 URLs and IPs, the requests per minute
- Find the 20 slowest requests by response time
- Support `.gz` compressed logs and analyzing several files at once
- Support filtering by a time range

**Acceptance criteria**

- Processed through a generator pipeline, with **memory use independent of the file's size** (a 1 GB file in under 100 MB)
- A malformed line does not crash the program, and the report says how many lines were skipped
- Use `cProfile` to find the slowest step and optimize it, recording the time before and after

**Going further**

- Process several files in parallel with a `ProcessPoolExecutor` and merge the statistics (a `Counter` supports addition)
- Support a `tail -f` mode: follow a log file being written and refresh the statistics every 5 seconds

## Project 3: an asynchronous crawler with caching and retries {#项目三带缓存和重试的异步爬虫}

**Stage**: 3, concurrency · **Topics**: `asyncio`, `TaskGroup`, `Semaphore`, timeouts, decorators, `httpx`

Starting from one URL, fetch the pages under the same domain and extract the titles and the links.

**Requirements**

- At most N requests at once, a timeout per request, and exponential backoff on failure
- Never fetch the same URL twice; limit the maximum depth and the maximum number of pages
- Obey `robots.txt`
- Save the results as JSON Lines (one JSON object per line)

**Acceptance criteria**

- The retry logic is an asynchronous decorator, `@retry(times=3, exceptions=(httpx.TransportError,))`
- Ctrl+C exits gracefully: every task is cancelled and what has been fetched is saved
- The tests use a local `http.server` or a mocked transport and never reach the real network

**Going further**

- Add a rate limit per domain (at most K requests per second)
- Support resuming: skip the URLs already fetched after a restart

## Project 4: a pluggable task scheduler {#项目四插件式任务调度器}

**Stage**: 2 and 3 · **Topics**: `__init_subclass__`, Protocol, type annotations, `threading` or `asyncio`, `logging`

A light scheduler for periodic tasks, where the tasks are defined as plugins.

```py
class CleanupTask(Task, name="cleanup", every="10m"):
    def run(self, ctx: Context) -> None:
        ...
```

**Requirements**

- Defining a `Task` subclass registers it automatically (`__init_subclass__`), with intervals written as `every="30s"`/`"10m"`/`"1h"`
- The scheduler runs each task at its interval and does not start one that has not finished its previous run
- A failed task is logged and retried by policy without affecting the others
- A command shows each task's state (the last run, how long it took, whether it succeeded)

**Acceptance criteria**

- No errors under mypy's strict mode
- The scheduling logic can be driven by a "fake clock" in tests, with no real waiting
- Plugin modules can be loaded automatically from a directory (`importlib`)

**Going further**

- Support cron expressions
- Persist the task states to SQLite and restore them after a restart

## Project 5: a miniature key-value database {#项目五迷你键值数据库}

**Stage**: 3 and 4 · **Topics**: asyncio networking, protocol design, persistence, testing, performance

Implement an in-memory key-value database like a simplified Redis, served over TCP.

```text
$ nc localhost 7379
SET name amy
OK
GET name
amy
EXPIRE name 10
OK
INCR counter
1
```

**Requirements**

- The server uses `asyncio.start_server` and supports several clients at once
- The commands: `GET`, `SET`, `DEL`, `EXISTS`, `INCR`, `EXPIRE`, `TTL`, `KEYS pattern`
- Persistence: snapshot the data to disk periodically (writing a temporary file and `rename`ing it atomically) and load it at startup
- A Python client library supporting `async with Client(...) as c: await c.set("k", "v")`

**Acceptance criteria**

- The commands are parsed with a `match` statement
- End-to-end tests: start the server in the test and verify every command through the client library
- It can carry 1000 concurrent connections (write a load script to prove it)

**Going further**

- Support append-only (AOF) persistence, losing no data after a crash
- Support publish and subscribe (`SUBSCRIBE`/`PUBLISH`)

## Project 6: publish a library of your own {#项目六发布一个自己的库}

**Stage**: 4, engineering · **Topics**: packaging, documentation, CI, semantic versioning, type annotations

Pick a module from an earlier project that is worth reusing (the retry decorator, the log parser, the configuration loader) and publish it as a real library.

**Requirements**

- A complete `pyproject.toml` supporting Python 3.12 and above
- The public API is fully annotated and ships a `py.typed` marker file
- The README covers installation, a quick start and the API
- GitHub Actions: ruff, mypy and pytest on several Python versions, and publishing automatically on a tag
- A documentation site written with MkDocs (like the one you are reading)

**Acceptance criteria**

- In a fresh virtual environment, `uv add <your library>` and the README get it working
- Test coverage above 90%
- The version numbers follow semantic versioning and a CHANGELOG is maintained

**Going further**

- Publish to TestPyPI first to verify the process, then to PyPI proper
- Ask a friend to use it and release 0.2.0 from their feedback
