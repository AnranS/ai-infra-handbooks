from parallel_plan import llama70b, search

search("70B，1024 卡，8K 序列，每步 4M token", llama70b, 8192, 512, 1024)
search("70B，1024 卡，8K 序列，每步 16M token", llama70b, 8192, 2048, 1024)
