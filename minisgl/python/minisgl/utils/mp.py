from __future__ import annotations

from typing import Callable, Dict, Generic, TypeVar

import msgpack
import zmq
import zmq.asyncio

T = TypeVar("T")

# 五种队列都是对 ZMQ socket 的薄封装：put 时 encoder 把消息对象变成 dict，
# msgpack 再把 dict 变成字节；get 时反过来。bind（create=True）的一方先存在，connect 的一方去连它。


class ZmqPushQueue(Generic[T]):
    def __init__(self, addr: str, create: bool, encoder: Callable[[T], Dict]):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUSH)
        self.socket.bind(addr) if create else self.socket.connect(addr)
        self.encoder = encoder

    def put(self, obj: T) -> None:
        self.socket.send(msgpack.packb(self.encoder(obj), use_bin_type=True), copy=False)

    def stop(self) -> None:
        self.socket.close(linger=0)
        self.context.term()


class ZmqAsyncPushQueue(Generic[T]):
    def __init__(self, addr: str, create: bool, encoder: Callable[[T], Dict]):
        self.context = zmq.asyncio.Context()
        self.socket = self.context.socket(zmq.PUSH)
        self.socket.bind(addr) if create else self.socket.connect(addr)
        self.encoder = encoder

    async def put(self, obj: T) -> None:
        await self.socket.send(msgpack.packb(self.encoder(obj), use_bin_type=True), copy=False)

    def stop(self) -> None:
        self.socket.close(linger=0)
        self.context.term()


class ZmqPullQueue(Generic[T]):
    def __init__(self, addr: str, create: bool, decoder: Callable[[Dict], T]):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PULL)
        self.socket.bind(addr) if create else self.socket.connect(addr)
        self.decoder = decoder

    def get(self) -> T:
        return self.decode(self.socket.recv())

    def get_raw(self) -> bytes:
        return self.socket.recv()

    def decode(self, raw: bytes) -> T:
        return self.decoder(msgpack.unpackb(raw, raw=False))

    def empty(self) -> bool:
        return self.socket.poll(timeout=0) == 0

    def stop(self) -> None:
        self.socket.close(linger=0)
        self.context.term()


class ZmqAsyncPullQueue(Generic[T]):
    def __init__(self, addr: str, create: bool, decoder: Callable[[Dict], T]):
        self.context = zmq.asyncio.Context()
        self.socket = self.context.socket(zmq.PULL)
        self.socket.bind(addr) if create else self.socket.connect(addr)
        self.decoder = decoder

    async def get(self) -> T:
        return self.decoder(msgpack.unpackb(await self.socket.recv(), raw=False))

    def stop(self) -> None:
        self.socket.close(linger=0)
        self.context.term()


class ZmqPubQueue(Generic[T]):
    """广播队列。用 XPUB 而不是 PUB：XPUB 能收到"有订阅者订阅了"的通知，于是可以等订阅者到齐再发。

    PUB/SUB 的一个经典陷阱（"慢订阅者"）：订阅在连接建立后才异步生效，在此之前发布的消息会被直接丢弃。
    """

    def __init__(self, addr: str, create: bool, encoder: Callable[[T], Dict]):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.XPUB)
        self.socket.setsockopt(zmq.XPUB_VERBOSE, 1)  # 每个订阅者的订阅通知都要，不去重
        self.socket.bind(addr) if create else self.socket.connect(addr)
        self.encoder = encoder

    def wait_for_subscribers(self, n: int) -> None:
        for _ in range(n):
            self.socket.recv()  # 订阅通知：b"\x01" + 主题

    def put_raw(self, raw: bytes) -> None:
        self.socket.send(raw, copy=False)

    def put(self, obj: T) -> None:
        self.socket.send(msgpack.packb(self.encoder(obj), use_bin_type=True), copy=False)

    def stop(self) -> None:
        self.socket.close(linger=0)
        self.context.term()


class ZmqSubQueue(Generic[T]):
    def __init__(self, addr: str, create: bool, decoder: Callable[[Dict], T]):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.bind(addr) if create else self.socket.connect(addr)
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self.decoder = decoder

    def get(self) -> T:
        return self.decoder(msgpack.unpackb(self.socket.recv(), raw=False))

    def empty(self) -> bool:
        return self.socket.poll(timeout=0) == 0

    def stop(self) -> None:
        self.socket.close(linger=0)
        self.context.term()
