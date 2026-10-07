# The RDMA programming model: verbs, one-sided operations and memory registration

<p class="lead">KV transfer in PD disaggregation, cross-machine expert parallelism and distributed KV caches all sit on RDMA: the NIC reads and writes a remote machine's memory (even a remote GPU's) directly, bypassing that machine's CPU and kernel. NCCL wraps up these details, but systems like KV transfer engines and DeepEP are written directly on RDMA. This chapter covers RDMA's programming model (memory registration, queue pairs, one-sided and two-sided operations, completion notification), walks through a KV push with a semantic simulator, and uses a cost model to show why "requests must be merged".</p>

!!! question "Self-test: if you can answer these, skip this chapter"
    1. Why is RDMA faster than TCP? What do "kernel bypass" and "zero copy" each mean?
    2. What does memory registration do? Why should a KV pool be registered once at startup?
    3. How do a one-sided WRITE and a two-sided SEND / RECV differ? How does the receiver know a write has finished?
    4. In PD disaggregation, does prefill push the KV (WRITE) or does decode pull it (READ)? What are the considerations for each?
    5. Why can't transferring many small blocks saturate the bandwidth?

??? success "Answers (try first, then expand to compare)"
    1. Kernel bypass: the application talks to the NIC directly, and sending and receiving data does not go through the operating system kernel (no system calls or context switches); zero copy: the NIC reads and writes the application's memory directly, without copies between kernel buffers and user memory. Add protocol processing done in NIC hardware, and you get low latency, high bandwidth and almost no CPU use.
    2. It pins a range of memory in physical memory (so it cannot be swapped out), builds the virtual-to-physical address mapping in the NIC, and generates access keys (`lkey` / `rkey`). Registration is slow (pinning pages, writing the NIC's page tables), so large regions like a KV pool are registered once at startup rather than on the fly for each transfer.
    3. One-sided WRITE: the initiator writes the peer's memory directly, and the peer's CPU neither participates nor knows; two-sided SEND / RECV: the receiver must post receive buffers in advance, and gets a completion event when data arrives. How the receiver learns a write has finished: the last write carries an immediate value (WRITE_WITH_IMM), producing a completion event on the peer; or a separate message is sent.
    4. This book pushes: the decode instance pre-allocates blocks and shares their addresses, and the prefill instance writes each layer over with one-sided WRITEs as soon as it is computed, overlapping with compute. Pulling (READ) has the decode instance read on its own once it is ready, so decode sets the pace, but it must wait for the prefill side's data to be ready, with one more notification.
    5. A NIC can process only so many messages per second (the message rate), and each message has a fixed processing cost; with very small blocks, the limit is the message rate, not bandwidth. Remedies: merge contiguous blocks, arrange the layout so one layer's data is contiguous, post in batches, and use several QPs and NICs in parallel.

<!-- comic ../assets/comics/rdma.webp is in Chinese; put it back once the English version exists -->

## Why RDMA {#为什么是-rdma}

Sending data over TCP goes through the sender kernel's protocol stack, copies data from the user buffer to a kernel buffer, has the NIC interrupt the receiver's CPU, then copies back to a user buffer; every step burns CPU and adds latency. RDMA removes all of this:

- **Kernel bypass**: a user-space program writes requests straight into the NIC's queues (through a memory-mapped "doorbell" register), with no system call;
- **Zero copy**: the NIC reads data straight from the user's memory and writes straight into the peer user's memory, with no kernel buffer in between;
- **CPU offload**: reliable delivery, retransmission and packetization are all done in NIC hardware. **One-sided** operations (READ / WRITE) do not even need the peer's CPU.

Add [GPUDirect RDMA](interconnect.md#gpudirect让数据不绕道-cpu), and "the user's memory" can be GPU memory: the NIC reads from one machine's GPU and writes into another machine's GPU.

## The verbs programming model {#verbs-编程模型}

![Figure: RDMA's verbs model: register memory, build queue pairs, and the NIC DMAs straight into the peer's memory](../assets/figures/rdma-verbs.svg){.aig-svg}

RDMA's programming interface is called **verbs** (`libibverbs`), with these core objects:

| Object | Role |
| --- | --- |
| **PD** (protection domain) | groups the resources below; only QPs in a PD can access memory registered in the same PD |
| **MR** (memory region) | `ibv_reg_mr` registers a range of memory: pins the physical pages (no swapping), has the NIC build an address translation table, and returns a local key `lkey` and a remote key `rkey` |
| **QP** (queue pair) | a send queue + a receive queue, roughly one "connection". The common type is RC (reliable connection: one-to-one, with ordering and delivery guaranteed) |
| **CQ** (completion queue) | when a request completes, the NIC writes a completion entry (CQE) here, which the program polls |
| **WR** (work request) | one operation the program posts to a QP: SEND, RECV, WRITE, READ, atomics |

Setting up a connection needs one **out-of-band** exchange: the two sides use ordinary TCP, HTTP or a metadata service such as etcd to swap QP numbers, NIC addresses (LID / GID), and the addresses and `rkey` of the memory to access, then move the QP's state through INIT → RTR (ready to receive) → RTS (ready to send). The "bootstrap service" in inference frameworks does exactly this.

Operations come in two kinds:

- **Two-sided**: SEND / RECV. The receiver must post a RECV request **in advance** with a buffer ready; when data arrives, the receiver gets a completion entry. Suited to control messages;
- **One-sided**: WRITE / READ. The initiator only needs the peer memory's address and `rkey` to read or write directly, and **the peer's CPU knows nothing** of it. Suited to bulk data.
- **WRITE_WITH_IMM**: writes data while attaching a 32-bit immediate value, and the peer gets a completion entry; this is the standard way to do "one-sided write + tell the peer it's done". An RC connection guarantees that operations on one QP take effect in order, so if the last write carries the immediate, all earlier data has certainly arrived when the peer gets the notification.

## Walking through a KV push with a simulator {#用模拟器走一遍-kv-推送}

The simulator below models only the semantics (registration, one-sided writes, completion queues, key checks), not the network. The scenario is PD disaggregation: the decode instance registers its whole KV pool once at startup; when a request arrives it allocates blocks and tells the prefill instance their addresses and `rkey` over an out-of-band channel; after computing, prefill writes the KV straight into those blocks with one-sided writes, the last one carrying an immediate to notify decode:

```python
import itertools


class RemoteAccessError(Exception):
    pass


class Nic:
    """一张网卡的替身：只模拟 RDMA 的语义（注册内存、单边写、完成队列），不模拟网络"""
    _keys = itertools.count(0x1000)

    def __init__(self, name):
        self.name, self.mrs, self.cq = name, {}, []

    def reg_mr(self, buf):
        """注册内存：固定物理页、让网卡能直接访问，返回 (起始地址, rkey)；这里用 rkey 当"地址空间"的编号"""
        rkey = next(self._keys)
        self.mrs[rkey] = buf
        return 0, rkey

    def post_write(self, peer, data, remote_addr, rkey, imm=None, signaled=False):
        """单边写：直接写进对端已注册的内存，对端 CPU 不参与；imm 不为空时对端收到一条完成通知"""
        buf = peer.mrs.get(rkey)
        if buf is None or remote_addr + len(data) > len(buf):
            raise RemoteAccessError(f"{peer.name} 拒绝访问：rkey={rkey:#x}，地址 {remote_addr}+{len(data)}")
        buf[remote_addr:remote_addr + len(data)] = data
        if imm is not None:
            peer.cq.append(("RECV_RDMA_WITH_IMM", imm))
        if signaled:
            self.cq.append(("RDMA_WRITE", len(data)))

    def poll_cq(self):
        done, self.cq = self.cq, []
        return done


BLOCK = 8                                        # 8 bytes per KV block (illustrative)
decode, prefill = Nic("decode"), Nic("prefill")
pool = bytearray(BLOCK * 6)                      # the decode instance's KV pool: registered once, whole, at startup
addr, rkey = decode.reg_mr(pool)

# out-of-band handshake (bootstrap): decode allocates blocks 4, 1, 5 for the request and tells prefill the address and rkey
blocks = [4, 1, 5]
kv = [bytes([65 + i]) * BLOCK for i in range(len(blocks))]      # the 3 KV blocks prefill computed: b"AAAAAAAA", b"BBBB...", ...
for i, (b, data) in enumerate(zip(blocks, kv)):
    last = i == len(blocks) - 1
    prefill.post_write(decode, data, addr + b * BLOCK, rkey, imm=7 if last else None, signaled=last)

print("decode 的完成队列：", decode.poll_cq())             # only the last write, carrying an immediate, produces a notification
print("prefill 的完成队列：", prefill.poll_cq())           # a completion was requested only for the last request (selective signaling)
print("decode 的 KV 池：", bytes(pool).decode().replace("\x00", "."))
for bad in [(rkey + 1, 0), (rkey, BLOCK * 6)]:
    try:
        prefill.post_write(decode, b"x" * BLOCK, bad[1], bad[0])
    except RemoteAccessError as e:
        print("出错：", e)
```

```text title="输出"
decode 的完成队列： [('RECV_RDMA_WITH_IMM', 7)]
prefill 的完成队列： [('RDMA_WRITE', 8)]
decode 的 KV 池： ........BBBBBBBB................AAAAAAAACCCCCCCC
出错： decode 拒绝访问：rkey=0x1001，地址 0+8
出错： decode 拒绝访问：rkey=0x1000，地址 48+8
```

A few things worth noticing:

- decode's CPU gets only one notification at the end; the three writes are "invisible" to it;
- prefill asked for a completion event only on the last write request (**selective signaling**): having every request produce a completion entry adds NIC and CPU overhead, and a later completion on the same QP implies the earlier ones completed too;
- a write out of bounds or with the wrong `rkey` is rejected by the NIC outright (the real error code is `IBV_WC_REM_ACCESS_ERR`, and after an error the QP enters the error state and the connection must be rebuilt). The `rkey` is the "access credential" for remote memory, and leaking it lets the other side read and write that memory at will;
- in real systems, the "done writing" notification can also skip the immediate: write a flag for the peer to poll, or send a control message with a two-sided SEND.

**Push or pull?** Above, prefill pushes (WRITE). Decode can also pull (READ): after computing, prefill tells decode where the KV is, and decode reads it. Pushing lets prefill write **layer by layer**, overlapping with later layers' compute, and free the KV once written; pulling lets decode decide when to fetch based on its own pace and spare memory. Both are used: in SGLang's PD transfer, decode allocates blocks first and prefill pushes; vLLM's NIXL connector pulls by default: after computing, prefill hands the block IDs to decode, which issues READs (`NixlPullConnector`), and version 0.30 added `NixlPushConnector`, where prefill pushes with WRITEs.

## The cost of memory registration {#内存注册的代价}

`ibv_reg_mr` pins every page in physical memory and has the NIC build a virtual-to-physical translation table; registering tens of GB can take hundreds of milliseconds to seconds, and the more memory registered, the more often the NIC's translation cache misses. So:

- **Register once at startup** the whole KV pool and the whole weight buffer, rather than once per request;
- Registering GPU memory goes through the GPUDirect RDMA driver (`nvidia-peermem` or dma-buf);
- Use huge pages (2 MB / 1 GB) to cut translation entries;
- This also means the KV pool's size and location are best fixed after startup; a KV pool that grows and shrinks dynamically must either reserve space or pay for re-registration.

## Request count and message rate {#请求数与消息速率}

Besides a bandwidth limit, a NIC has a **message rate** limit: each work request has a fixed processing cost, and posting requests (writing WQEs, ringing the doorbell) takes the program time too. KV is stored in pages, and with small pages a transfer becomes a flood of tiny requests. Estimate how many requests the KV of one 4K prompt (70B, TP=8, 160 MiB per GPU to transfer) takes under different arrangements:

```python
LAYERS, PROMPT, HEAD_BYTES = 80, 4096, 128 * 2   # 70B: 80 layers; 4K prompt; at TP=8 each GPU has 1 KV head, so one token's K is 256 bytes
BW, PER_WR = 50e9, 0.1e-6                        # one 400G NIC; posting cost per WR (order of magnitude for one thread and one QP, illustrative)
total = LAYERS * PROMPT * 2 * HEAD_BYTES         # K and V
print(f"一个 4K 提示词每卡要传 {total / 2**20:.0f} MiB，纯带宽时间 {total / BW * 1e3:.1f} ms")

plans = [                                        # (page size, pages per WR, whether K and V travel together)
    ("页大小 1，每层每 token 的 K、V 各一条", 1, 1, False),
    ("页大小 16，每层每页的 K、V 各一条", 16, 1, False),
    ("页大小 16，K、V 相邻存放，每层每页一条", 16, 1, True),
    ("页大小 16，合并地址连续的 8 页", 16, 8, True),
    ("每层一条（两边的页都连续）", PROMPT, 1, True),
]
for name, page, merge, kv_together in plans:
    n_wr = LAYERS * (PROMPT // (page * merge)) * (1 if kv_together else 2)
    t = max(total / BW, n_wr * PER_WR)           # whichever is slower, bandwidth or posting rate, sets the time
    print(f"{name}：{n_wr} 条 WR，每条 {total / n_wr / 1024:g} KiB，约 {t * 1e3:.1f} ms")
```

```text title="输出"
一个 4K 提示词每卡要传 160 MiB，纯带宽时间 3.4 ms
页大小 1，每层每 token 的 K、V 各一条：655360 条 WR，每条 0.25 KiB，约 65.5 ms
页大小 16，每层每页的 K、V 各一条：40960 条 WR，每条 4 KiB，约 4.1 ms
页大小 16，K、V 相邻存放，每层每页一条：20480 条 WR，每条 8 KiB，约 3.4 ms
页大小 16，合并地址连续的 8 页：2560 条 WR，每条 64 KiB，约 3.4 ms
每层一条（两边的页都连续）：80 条 WR，每条 2048 KiB，约 3.4 ms
```

With a page size of 1 (managing KV per token; SGLang's default when `--page-size` is not set, though some attention backends change it to 64 or similar), transferring token by token makes the request time 20 times the bandwidth time; here the bottleneck is not network bandwidth at all. The usual remedies:

- **Merge contiguous blocks**: a run whose source and destination addresses are both contiguous becomes one request. When decode allocates KV, giving contiguous blocks where possible pays off clearly; SGLang's PD transfer first merges runs that are contiguous on both sides (`group_concurrent_contiguous` in `disaggregation/common/utils.py`);
- **Adjust the layout**: storing K and V next to each other, and a layer's pages next to each other, both cut the request count; some systems prepare a dedicated staging buffer for transfers that is "layer-major, contiguous per request", rearranging on the GPU first and then sending in one go;
- **Post in batches**: post a chain of requests with a single doorbell (`ibv_post_send` accepts a linked list of requests), and ask for a completion only on the last one;
- **Several QPs and NICs in parallel**: one NIC and one QP have a limited message rate, so transfer engines spread requests across several QPs and the machine's several NICs (choosing the nearest by GPU-NIC topology).

The model's cost of 0.1 µs per request is only illustrative; the real message rate depends on the NIC model, the number of QPs and the number of posting threads, and should be measured on your own machines with `ib_write_bw` and `ib_write_lat` (from the perftest suite).

## GPUs issuing RDMA themselves {#gpu-自己发起-rdma}

In the flow above, the CPU issues the RDMA: the GPU computes the KV, then the CPU posts the write requests. That is enough for PD transfer, but for communication like MoE's all-to-all, which is "needed every layer, small, and latency-sensitive", CPU involvement becomes a bottleneck: the GPU must first sync back to the CPU, which then operates the NIC.

**IBGDA** (InfiniBand GPUDirect Async) lets GPU threads build work requests and ring the NIC's doorbell themselves, so the whole communication happens inside a GPU kernel with no CPU involvement at all. NVSHMEM uses it to offer "put / get from inside a kernel", and DeepEP's low-latency mode is built on top of that; that is the [next chapter](nvshmem-deepep.md).

!!! interview "In an interview"
    When asked about RDMA, start with three keywords: kernel bypass, zero copy, one-sided operations; then the programming model: register memory to get an `rkey`, build a QP, exchange addresses and keys out of band, post WRITE / READ, poll the completion queue; and finally land on inference: register the KV pool once at startup, have prefill push layer by layer after decode pre-allocates blocks, notify with an immediate on the last write, merge small blocks to avoid the message-rate bottleneck, and run several NICs in parallel by topology. A quantitative judgment like "with a page size of 1, token-by-token transfer is throttled by the message rate" is far more convincing than reciting concepts.

## Exercises {#练习}

**1. Why is "an immediate on the last write" enough?** If prefill writes KV over two different QPs in parallel (to use two NICs fully), does this still hold?

??? success "Answer"
    An RC QP guarantees that writes on the same QP take effect at the peer in posting order, so when the last write's notification arrives, all earlier writes on that QP are done. There is no ordering across QPs: QP1's notification may arrive while QP2's data is still in flight. With several QPs, either each QP sends its own notification and the receiver waits for all of them (counting), or a separate notification is sent after all QPs' writes complete (the sender sees each one's completion event).

**2. The page-size trade-off.** If larger pages transfer more efficiently, why don't inference engines make pages very large (say, 256 tokens)?

??? success "Answer"
    Page size affects more than transfers: larger pages mean more internal fragmentation (each request's last page wastes half a page on average), coarser prefix-sharing granularity in the prefix cache (prefixes can only be shared in whole pages), and paged attention kernels must adapt as well. SGLang's default page size of 1 exists for token-level prefix sharing (Radix Cache). So the more common approach is to keep small pages and merge contiguous blocks during transfer, or rearrange into a staging buffer on the GPU before sending; some systems also pick a larger page size just for PD disaggregation.

## Summary {#小结}

- [x] RDMA gets low latency and high bandwidth from kernel bypass, zero copy and hardware offload; one-sided READ / WRITE need no CPU on the peer.
- [x] The core verbs objects: PD, MR (`lkey` / `rkey`), QP (usually RC), CQ, WR; connecting requires exchanging QP information, addresses and keys out of band.
- [x] A KV push: decode pre-allocates and shares addresses, prefill does one-sided writes layer by layer, and the last write notifies with an immediate; ordering holds within one QP, while across QPs it needs separate synchronization.
- [x] Memory registration is expensive, so KV pools and buffers are registered once at startup; registering GPU memory goes through the GPUDirect RDMA driver.
- [x] With many small blocks the bottleneck is message rate, not bandwidth: merge contiguous blocks, adjust the layout, post in batches, and run several QPs and NICs in parallel.
- [x] IBGDA lets GPUs issue RDMA themselves, the foundation of NVSHMEM and DeepEP's low-latency mode.
