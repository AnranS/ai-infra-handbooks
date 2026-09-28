"""Radix Cache：用基数树组织所有已缓存的 token 序列，实现跨请求的前缀复用。

树的每个节点保存一段 token（key）以及这些 token 的 KV 在池中的位置（value）。从根走到某个节点，
经过的所有 key 拼起来就是一个被缓存的前缀。节点上的 ref_count 表示有多少个正在运行的请求
依赖这个节点，ref_count 为 0 的叶子可以按最近最少使用（LRU）的顺序淘汰。
"""

from __future__ import annotations

import heapq
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple

import torch
from minisgl.core import get_global_ctx
from minisgl.kernel import fast_compare_key
from minisgl.utils import align_down

from .base import BaseCacheHandle, BasePrefixCache, InsertResult, MatchResult, SizeInfo

KeyFn = Callable[[torch.Tensor], Any]


class RadixTreeNode:
    counter: int = 0

    def __init__(self, key_fn: KeyFn, tic: int | None = None) -> None:
        self.key_fn = key_fn
        self.children: Dict[Any, RadixTreeNode] = {}
        self._parent: RadixTreeNode | None = None
        self.ref_count = 0
        self.uuid = RadixTreeNode.counter
        RadixTreeNode.counter += 1
        self.timestamp = tic or time.monotonic_ns()
        self._key: torch.Tensor
        self._value: torch.Tensor
        self._length: int

    def set_key_value(self, key: torch.Tensor, value: torch.Tensor) -> None:
        assert len(key) == len(value)
        self._key, self._value, self._length = key, value, len(key)

    def set_parent(self, parent: RadixTreeNode) -> None:
        self._parent = parent
        parent.children[self.key_fn(self._key)] = self  # 孩子按第一页的 token 索引

    @property
    def length(self) -> int:
        return self._length

    @property
    def parent(self) -> RadixTreeNode:
        assert self._parent is not None
        return self._parent

    @property
    def value(self) -> torch.Tensor:
        return self._value

    def is_root(self) -> bool:
        return self._parent is None

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def get_match_len(self, input_ids: torch.Tensor) -> int:
        return fast_compare_key(self._key, input_ids)

    def split_at(self, pos: int) -> RadixTreeNode:
        """把本节点在 pos 处一分为二：新建的前半段成为本节点的父节点，返回前半段。"""
        assert 0 < pos < self.length
        parent = self.parent
        new_node = RadixTreeNode(self.key_fn, self.timestamp)
        new_node.set_key_value(self._key[:pos], self._value[:pos])
        new_node.set_parent(parent)  # 替换掉 parent.children 中原来指向自己的那一项
        new_node.ref_count = self.ref_count
        self.set_key_value(self._key[pos:], self._value[pos:])
        self.set_parent(new_node)
        return new_node

    def __lt__(self, other: RadixTreeNode) -> bool:  # 供堆排序：时间戳越早越先淘汰
        return self.timestamp < other.timestamp


@dataclass(frozen=True)
class RadixCacheHandle(BaseCacheHandle):
    node: RadixTreeNode

    def get_matched_indices(self) -> torch.Tensor:
        values: List[torch.Tensor] = []
        node = self.node
        while not node.is_root():
            values.append(node.value)
            node = node.parent
        values.reverse()
        return torch.cat(values) if values else torch.empty(0, dtype=torch.int32)


class RadixPrefixCache(BasePrefixCache):
    def __init__(self, device: torch.device):
        self.device = device
        self.page_size = get_global_ctx().page_size
        self.key_fn = _get_key_fn(self.page_size)
        self.empty_tensor = torch.empty(0, dtype=torch.int32, device=device)
        self.evictable_size = 0
        self.protected_size = 0
        self.root_node = RadixTreeNode(self.key_fn)
        self.root_node.ref_count = 1  # 根节点永远不被淘汰

    def lock_handle(self, handle: BaseCacheHandle, unlock: bool = False) -> None:
        assert isinstance(handle, RadixCacheHandle)
        node = handle.node
        while not node.is_root():  # 从命中的节点一直走到根：整条路径都被这个请求依赖
            if unlock:
                node.ref_count -= 1
                assert node.ref_count >= 0
                if node.ref_count == 0:
                    self.evictable_size += node.length
                    self.protected_size -= node.length
            else:
                if node.ref_count == 0:
                    self.evictable_size -= node.length
                    self.protected_size += node.length
                node.ref_count += 1
            node = node.parent

    def match_prefix(self, input_ids: torch.Tensor) -> MatchResult:
        node, prefix_len = self._tree_walk(input_ids)
        return MatchResult(RadixCacheHandle(prefix_len, node))

    def insert_prefix(self, input_ids: torch.Tensor, indices: torch.Tensor) -> InsertResult:
        insert_len = align_down(len(input_ids), self.page_size)  # 只缓存完整的页
        input_ids, indices = input_ids[:insert_len], indices[:insert_len]
        node, prefix_len = self._tree_walk(input_ids)
        if prefix_len != insert_len:  # 还有没缓存的部分：挂一个新叶子
            new_node = RadixTreeNode(self.key_fn)
            new_node.set_key_value(input_ids[prefix_len:], indices[prefix_len:].clone())
            new_node.set_parent(node)
            self.evictable_size += new_node.length
            node = new_node
        return InsertResult(prefix_len, RadixCacheHandle(insert_len, node))

    def evict(self, size: int) -> torch.Tensor:
        if size == 0:
            return self.empty_tensor
        assert size <= self.evictable_size, f"Cannot evict {size}, only {self.evictable_size}"
        leaves = self._collect_evictable_leaves()
        heapq.heapify(leaves)
        evicted: List[torch.Tensor] = []
        evicted_size = 0
        while evicted_size < size:
            assert leaves, f"Cannot evict enough cache: need {size}, got {evicted_size}"
            node = heapq.heappop(leaves)
            evicted_size += node.length
            evicted.append(node.value)
            self.evictable_size -= node.length
            parent = node.parent
            del parent.children[self.key_fn(node._key)]
            if parent.is_leaf() and parent.ref_count == 0:  # 父节点变成了可淘汰的叶子
                heapq.heappush(leaves, parent)
        return torch.cat(evicted)

    def reset(self) -> None:
        raise NotImplementedError

    @property
    def size_info(self) -> SizeInfo:
        return SizeInfo(evictable_size=self.evictable_size, protected_size=self.protected_size)

    def check_integrity(self) -> None:
        """重新遍历整棵树，核对计数器：可淘汰量 = ref_count 为 0 的节点长度之和。"""
        evictable = protected = 0
        stack = list(self.root_node.children.values())
        while stack:
            node = stack.pop()
            assert node.ref_count >= 0
            for child in node.children.values():
                assert child.ref_count <= node.ref_count, "child locked more than its parent"
                stack.append(child)
            if node.ref_count == 0:
                evictable += node.length
            else:
                protected += node.length
        assert (evictable, protected) == (self.evictable_size, self.protected_size), (
            f"size mismatch: tree=({evictable}, {protected}), "
            f"counter=({self.evictable_size}, {self.protected_size})"
        )

    def _collect_evictable_leaves(self) -> List[RadixTreeNode]:
        nodes, leaves = [self.root_node], []
        while nodes:
            node = nodes.pop()
            if node.is_leaf():
                if node.ref_count == 0:
                    leaves.append(node)
            else:
                nodes.extend(node.children.values())
        return leaves

    def _tree_walk(self, input_ids: torch.Tensor) -> Tuple[RadixTreeNode, int]:
        prefix_len, total = 0, len(input_ids)
        node = self.root_node
        tic = time.monotonic_ns()
        while prefix_len < total:
            child = node.children.get(self.key_fn(input_ids[prefix_len:]))
            if child is None:
                return node, prefix_len
            node = child
            # 进入孩子说明至少第一页匹配上了；再按页对齐算出实际匹配长度
            match_len = align_down(node.get_match_len(input_ids[prefix_len:]), self.page_size)
            prefix_len += match_len
            if match_len != node.length:  # 只匹配了节点的一部分：在匹配处分裂
                node = node.split_at(match_len)
                node.timestamp = tic
                return node, prefix_len
            node.timestamp = tic  # 访问过的节点刷新时间戳（LRU）
        return node, prefix_len


def _get_key_fn(page_size: int) -> KeyFn:
    if page_size == 1:
        return lambda x: int(x[0])
    return lambda x: tuple(x[:page_size].tolist())
