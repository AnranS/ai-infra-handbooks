import multiprocessing as mp
import threading

counter = {"n": 0}


def bump():
    counter["n"] += 1


t = threading.Thread(target=bump)
t.start()
t.join()
print("线程改完之后，主线程看到：", counter["n"])

p = mp.get_context("fork").Process(target=bump)
p.start()
p.join()
print("子进程改完之后，父进程看到：", counter["n"])
