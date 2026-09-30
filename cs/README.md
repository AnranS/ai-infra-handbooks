# 计算机基础手册

推理工程师需要的那部分计算机基础：操作系统（进程与调度、虚拟内存与大页、锁页内存与 NUMA、I/O 栈、epoll 与 io_uring、进程间通信、容器、性能分析工具）、体系结构（CPU 流水线与缓存、GPU 的 SM 与 Tensor Core、GPU 内存系统、架构演进、多卡系统），计算机网络（TCP、HTTP 与流式输出、负载均衡与排队）、分布式系统（一致性哈希与分片、复制与共识），数据结构与算法（六章，配 63 道浏览器判题的练习）。每章都从推理系统里的真实现象讲起，配有在 Linux 上实际运行过的 Python 和 C 程序。

## 目录

- `docs/`：正文，按部分分为 `os/`（操作系统）、`arch/`（体系结构）、`net/`（计算机网络）、`dist/`（分布式系统）、`algo/`（数据结构与算法）
- `tools/check_code.py`：运行正文里所有标了文件名的程序，确认页面上的输出与实际运行一致

## 校验

```bash
# Linux；Python 3.10+（pyzmq、numpy）和 gcc；性能工具一章用到 perf 和 strace
python3 tools/check_code.py                  # 所有页面
python3 tools/check_code.py docs/os/ipc.md   # 指定页面
```

约定（详见 `tools/check_code.py` 的说明）：

- ```` ```python title="x.py" ```` 和 ```` ```c title="x.c" ```` 是完整程序，紧跟的 ```` ```text title="输出" ```` 必须与标准输出逐行一致；C 程序用 `gcc -O2 -Wall -Wextra -Werror -pthread` 编译，`flags="..."` 追加编译参数；
- ```` ```text title="输出（本机示例）" ```` 是和机器相关的测量结果，只要求程序跑通；
- `run="no"` 的程序需要 GPU 等本机没有的条件，只做语法检查；
- 同一页的程序写在同一个目录里、按页面顺序运行，可以读前面的程序留下的文件。

构建站点：在仓库根目录运行 `./build.sh`，或单独预览 `mkdocs serve -f cs/mkdocs.yml`。
