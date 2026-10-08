# 需要 GPU。两个进程共享同一块显存：生产者导出 IPC 句柄，消费者打开它，拿到的是同一块显存，不拷贝
import torch
import torch.multiprocessing as mp


def consumer(q):
    t = q.get()                    # 收到的张量底下是 cudaIpcOpenMemHandle 打开的同一块显存
    t.add_(1)                      # 改的是生产者那块显存
    q.put("done")


if __name__ == "__main__":
    mp.set_start_method("spawn")   # CUDA 进程只能用 spawn（见"进程、线程与调度"一章）
    q = mp.Queue()
    x = torch.zeros(4, device="cuda")
    p = mp.Process(target=consumer, args=(q,))
    p.start()
    q.put(x)                       # 传过去的只是 IPC 句柄和元数据
    q.get()                        # 等对方用完：生产者必须保证 x 在对方使用期间一直有效
    print(x)                       # tensor([1., 1., 1., 1.], device='cuda:0')
    p.join()
