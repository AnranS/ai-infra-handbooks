"""从这里开始：随便写点 Python"""
# Playground 里的代码跑在浏览器中的 Python（Pyodide）上，不需要安装任何东西。
# Ctrl / ⌘ + Enter 运行；输入 "." 或名字时会弹出补全，Ctrl + 空格 或 ⌥ + / 手动补全。
# 可以 import numpy、matplotlib，以及练习题里的 CUDA 模拟器 gpusim 和 Triton 模拟器（import triton）。
import sys

print("Python", sys.version.split()[0])
squares = {n: n * n for n in range(1, 6)}
print("平方表：", squares)
