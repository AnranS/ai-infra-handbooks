import threading
import time

from checker import check, raises
from solution import BlockingQueue


def test_example_fifo():
    q = BlockingQueue(3)
    for i in range(3):
        q.put(i)
    check(len(q), 3, "len(q)")
    check([q.get(), q.get(), q.get()], [0, 1, 2], "先进先出")


def test_get_timeout_on_empty():
    q = BlockingQueue(2)
    start = time.monotonic()
    with raises(TimeoutError, "空队列 get(timeout=0.1)"):
        q.get(timeout=0.1)
    used = time.monotonic() - start
    assert 0.08 <= used < 1.0, f"应该等待约 0.1 秒，实际 {used:.3f} 秒"


def test_put_blocks_when_full():
    """队列满时 put 阻塞，直到消费者取走元素"""
    q = BlockingQueue(1)
    q.put("a")
    done = threading.Event()

    def producer():
        q.put("b")
        done.set()

    t = threading.Thread(target=producer, daemon=True)
    t.start()
    time.sleep(0.1)
    assert not done.is_set(), "队列已满，put 应该阻塞"
    check(q.get(), "a", "第一个元素")
    assert done.wait(1.0), "取走元素后，阻塞的 put 应该完成"
    check(q.get(), "b", "第二个元素")
    with raises(TimeoutError, "满队列 put(timeout=0.05)"):
        q.put("x")
        q.put("y", timeout=0.05)


def test_many_producers_consumers():
    q = BlockingQueue(4)
    n_prod, per = 4, 500
    got, lock = [], threading.Lock()

    def producer(k):
        for i in range(per):
            q.put((k, i))

    def consumer():
        while True:
            try:
                item = q.get(timeout=2)
            except (EOFError, TimeoutError):
                return
            except Exception as e:  # noqa: BLE001  实现有 bug 时，别让线程把异常打印到控制台
                with lock:
                    got.append(("error", repr(e)))
                return
            with lock:
                got.append(item)

    prods = [threading.Thread(target=producer, args=(k,)) for k in range(n_prod)]
    cons = [threading.Thread(target=consumer) for _ in range(3)]
    for t in prods + cons:
        t.start()
    for t in prods:
        t.join(5)
    q.close()
    for t in cons:
        t.join(5)
    check(len(got), n_prod * per, "取到的元素总数")
    check(sorted(got), sorted((k, i) for k in range(n_prod) for i in range(per)), "取到的元素")
    for k in range(n_prod):
        seq = [i for kk, i in got if kk == k]
        assert seq == sorted(seq), f"同一个生产者的元素顺序被打乱了（生产者 {k}）"


def test_close_wakes_waiters():
    q = BlockingQueue(1)
    errors = []

    def waiter():
        try:
            q.get()
        except EOFError:
            errors.append("EOF")
        except Exception as e:  # noqa: BLE001
            errors.append(type(e).__name__)

    t = threading.Thread(target=waiter, daemon=True)
    t.start()
    time.sleep(0.05)
    q.close()
    t.join(1.0)
    check(errors, ["EOF"], "close() 应该唤醒等待的 get 并抛出 EOFError")
    with raises(RuntimeError, "关闭后 put"):
        q.put(1)


def test_drain_after_close():
    q = BlockingQueue(3)
    q.put(1)
    q.put(2)
    q.close()
    check([q.get(), q.get()], [1, 2], "关闭后仍能取完剩下的元素")
    with raises(EOFError, "取完之后再 get"):
        q.get(timeout=1)
