# 从读代码到提 MR：找切入点、定范围、写测试、写描述

<p class="lead">前面十一章是"读懂"，这一章是"动手"。在 sglang-omni 这种节奏的项目里，标着 good first issue 的问题几个小时就被认领，所以<b>能提出有分量的 PR 的人，几乎都是自己读代码找到问题的人</b>。这一章先给一套找切入点的方法，然后完整地走一个案例——Rust router 的"连接失败时换一个 worker 重试"：从第十章实验里那个 502 出发，读代码、发现第一个障碍（请求体可能没法重放）、定范围、写测试计划、写认领评论和 PR 描述。最后用第六章自己撞到的那个 CPU 边界，演示"一个发现不一定是 bug，先确认再提"。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. 除了 issue 列表，还有哪些地方能找到值得做的改动？
    2. 一个 issue 看起来没人做，动手之前要确认哪几件事？
    3. 第十章实验里那个 502 发生在 router 的哪条路径上？为什么"换个 worker 重试"在那条路径上比看起来难？
    4. 一个"重试"PR 的范围应该怎么划，才不会引入重复执行的风险？
    5. 认领评论应该写什么？

??? success "自测参考答案（先自己答，再展开对照）"
    1. 路线图 / RFC 里没勾的项、代码里的 TODO / FIXME / note 注释、默认关闭的新开关（还在验证的路径）、文档和代码不一致的地方、自己做实验时撞到的边界、已经被别的 PR 解决但还开着的过期 issue。
    2. 有没有人留言认领、有没有 open PR 引用它、代码在当前 main 上是否已经修了、维护者是否同意这个方向（尤其是有设计取舍的改动）。
    3. 直接路径：同质的 worker 池不读请求体，`OutgoingRequest::direct` 把客户端的请求体以流的方式交给 reqwest。连接失败时这个流已经被 `send()` 拿走了，没有现成的副本可以重发；只有分类路径的 `BufferedUpload` 握着完整的字节（`Bytes` 可以零拷贝克隆）。
    4. 只对"连接没有建立"的失败重试（reqwest 的 `is_connect()`，请求一个字节都没发出去）；只重试能重放的请求体；换一个健康且兼容的 worker（排除刚失败的那个）；有次数上限和带抖动的退避；不超过请求原来的截止时间；默认关闭，和熔断一样加配置项；加指标和文档。
    5. 你打算做哪一项（引用路线图的原话）、打算怎么做（范围和不做什么）、和相关 PR 的关系（基于谁、会不会冲突），最后问一句"范围对吗"。

## 找切入点的六个地方

### 一、issue：先过滤，再判断

issue 列表要过滤掉三类：已有人认领的、已有 open PR 引用的、路线图 / 跟踪类的大 issue。用 GitHub CLI 可以批量做（需要联网和登录，所以这里只给脚本）：

```python title="scan_issues.py" run="no"
"""列出没人认领、没有 open PR 引用的 issue（需要 gh 已登录）。"""
import json
import re
import subprocess

REPO = "sgl-project/sglang-omni"


def gh(*args):
    return json.loads(subprocess.run(["gh", *args], capture_output=True, text=True, check=True).stdout)


issues = gh("issue", "list", "-R", REPO, "--state", "open", "--limit", "1000",
            "--json", "number,title,assignees,comments,createdAt")
prs = gh("pr", "list", "-R", REPO, "--state", "open", "--limit", "1000", "--json", "number,title,body")
referenced = {int(n) for p in prs for n in re.findall(r"#(\d+)", f"{p['title']} {p['body'] or ''}")}
claim = re.compile(r"(?i)i'?d like to (take|work)|i'?ll take|working on (it|this)|assign (it|this) to me")
for i in sorted(issues, key=lambda i: -i["number"]):
    if re.search(r"(?i)roadmap|tracking|rfc", i["title"]):
        continue
    if i["assignees"] or i["number"] in referenced or any(claim.search(c["body"] or "") for c in i["comments"]):
        continue
    print(f"#{i['number']} {i['createdAt'][:10]} {i['title'][:90]}")
```

过滤之后剩下的，还要**在当前 main 上验证一遍**。本书写作时（2026-10-10）这样过滤剩下 60 个，逐个看下来大多已经过期，例如：

```bash title="stale-issues.sh"
echo "#2269 说 audio_encoder 引用了不存在的 get_feat_extract_output_lengths，基准提交里是："
git grep -n 'feat_extract_output_lengths' "$REF" -- sglang_omni/models/qwen3_omni/components/audio_encoder.py | sed "s/^$REF://"
echo "#1147 说编码器固定等 50 ms，基准提交里的默认值："
git show "$REF:sglang_omni/models/qwen3_omni/stages.py" | sed -n '356,361p'
```

```text title="输出"
#2269 说 audio_encoder 引用了不存在的 get_feat_extract_output_lengths，基准提交里是：
sglang_omni/models/qwen3_omni/components/audio_encoder.py:197:            hf_modeling._get_feat_extract_output_lengths
#1147 说编码器固定等 50 ms，基准提交里的默认值：
def encoder_batch_wait_ms() -> int:
    raw = os.getenv("SGLANG_OMNI_ENCODER_BATCH_WAIT_MS", "")
    if not raw:
        return 0
    else:
        pass
```

两个都已经在代码里解决了（#2269 被改回了 `_get_...`；#1147 被 #1628 解决，默认窗口变成 0），issue 却还开着。在这样的 issue 下留一条"已由某某提交解决，可以关闭"的评论（附上证据），不算 PR，但对维护者有用，也让你的名字先出现在项目里。

### 二、路线图和 RFC 里没勾的项

大功能的 RFC 往往带一份清单，例如 #1623（Rust router）的第一阶段，"Production hardening"那一节里有熔断、连接失败重试、Prometheus 指标、首包计时、OpenTelemetry、优雅退出。清单是维护者认可过的方向，比自己凭空想的改动更容易被接受。但清单会过时——第十章读代码时看到，"按路由限制请求体大小"之类的项其实已经实现了。**动手前对照代码确认，再看有没有 open PR**（熔断是 #2508、首包计时是 #2541）。

### 三、代码里的 TODO、FIXME 和"默认关闭"的开关

```bash title="todos.sh"
echo "sglang_omni/ 里的 TODO / FIXME：$(git grep -cE '\b(TODO|FIXME)\b' "$REF" -- 'sglang_omni/*.py' | awk -F: '{s += $NF} END {print s}') 处（下面不列 vendor/ 里的）"
git grep -nE '\b(TODO|FIXME)\b' "$REF" -- 'sglang_omni/*.py' ':!sglang_omni/vendor/' | sed "s/^$REF://" | cut -c1-110
```

```text title="输出"
sglang_omni/ 里的 TODO / FIXME：10 处（下面不列 vendor/ 里的）
sglang_omni/models/fun_cosyvoice3/config.py:217:        # TODO (chenyang): Indeed, TRT and Torch compile confl
sglang_omni/models/llada2_uni/merge.py:22:    # TODO: add streaming support
sglang_omni/models/moss_transcribe_diarize/sglang_model.py:155:        TODO(yichi): investigate whether reduce
sglang_omni/models/qwen3_omni/config.py:36:# FIXME (Ratish): Replace this with a bounded/pre-ready SGLang Deep
sglang_omni/scheduling/pre_lm_encoder.py:157:        # TODO(Jeffro): once every service stages its own host co
sglang_omni/scheduling/session.py:226:            # TODO (chenyang): This error handling is a bit rough here.
sglang_omni/serve/transcriptions.py:126:        # TODO(Ratish): add the same pre-parser body limit used by voi
```

数量很少，每一条都值得读。比如 `serve/transcriptions.py` 那条："等转录上传的限制定下来之后，加上和音色上传一样的解析前大小限制"——这说明转录接口的上传目前在解析前没有大小上限，而"限制是多少"还没有定。这种 TODO 适合先开 issue 讨论数值，而不是直接提 PR。

默认关闭的开关是另一类线索：第九章的 `ENABLE_TALKER_START_TOPOLOGY = False`，新的"一个 chunk 就开工"的路径已经写好但没打开——它附近通常需要正确性测试、评测数据。

### 四、文档和代码不一致的地方

第六章发现 README 写数据面支持"SHM、NCCL、NIXL、Mooncake"，而路由只会构造 `cuda_ipc`、`shm`、`mooncake`。改文档是最小的 PR，但先问一句"是计划接回去还是文档该改"更稳妥。

### 五、自己做实验撞到的边界

本书的实验跑在一个不常见的环境里（CPU、没有 SGLang 的内核），撞到了两个边界：pytest 的临时目录太长导致套接字路径超限（第十一章）、CPU 上大的流式 chunk 让发送方崩溃（第六章）。这类发现最有价值，因为别人很可能没注意到——但也最需要先确认"这是不是一个受支持的场景"（本章最后一节）。

### 六、把一个模型上的优化推广到另一个模型

第八、九章看到，Qwen3-TTS 的声码器有 1、2、4、8 帧的递增 chunk，Qwen3-Omni 的声码器首段还要等满 10 帧；公共函数 `resolve_initial_codec_chunk_frames` 已经在 Higgs、MOSS-TTS Local 上用了。#1148 就是这一类。这类改动的难点不在代码，而在**验证**：要证明拼接无缝、首包变快、吞吐不降，需要 GPU 和评测——可以和有卡的贡献者合作。

## 案例：router 的连接失败重试

### 第一步：现象

第十章的实验：两个健康的 worker，w2 突然下线，下一个分给 w2 的请求得到 `502 upstream_protocol_error`，之后健康检查把 w2 摘掉。这个请求从未到达任何 worker，换一个 worker 就能成功。worker 滚动重启、扩缩容、某个副本崩溃时，这样的 502 会成批出现。

### 第二步：读代码，找到 502 的出处

第十章读过 `HttpRelay::send`：`request.send()` 失败一律归成 `UpstreamProtocolError`，底层错误 `_source` 被丢弃。再往上看是谁调用 `send`——聊天接口的处理函数：

```rust title="sglang_omni_router/rust/src/http_generation/mod.rs @ 921ea2c8 L82-121"
    let deadline = tokio::time::Instant::now() + generation.request_timeout;
    let framing = validate_request(request.headers())?;
    let proof = generation
        .pool
        .content_blind_generation_http(&generation.trust);
    let maximum = if proof.is_some() {
        generation.streamed_max
    } else {
        generation.buffered_max
    };
    if framing
        .content_length
        .is_some_and(|length| length > maximum)
    {
        return Err(HttpFault::RequestBodyTooLarge);
    }
    let admission = generation
        .pool
        .try_admit(CapacityClass::GenerationHttp, 1)
        .map_err(map_admission)?;

    if let Some(proof) = proof {
        let lease = proof.dispatch(admission).map_err(map_dispatch)?;
        let outgoing = OutgoingRequest::direct(
            CHAT_PATH,
            canonical_content_type(),
            request.into_body(),
            framing.content_length,
            generation.streamed_max,
        );
        return Arc::clone(&generation.relay)
            .send(
                outgoing,
                lease,
                request_id,
                deadline,
                sanitize_response_headers,
            )
            .await;
    }
```

这里出现了第一个关键事实：**同质的 worker 池走直接路径**（`content_blind_generation_http` 返回了一个"不用看内容"的证明），请求体是 `request.into_body()`——客户端的请求体流，被包进 `OutgoingRequest::direct` 交给 reqwest。第十章的实验配置正是同质池，所以那个 502 发生在直接路径上。

再看两种 `OutgoingRequest` 的构造：

```rust title="sglang_omni_router/rust/src/http_relay/mod.rs @ 921ea2c8 L238-279"
    pub(crate) fn buffered(
        path: &'static str,
        content_type: HeaderValue,
        upload: BufferedUpload,
    ) -> Result<Self, HttpFault> {
        let content_length =
            u64::try_from(upload.bytes.len()).map_err(|_| HttpFault::InternalError)?;
        Ok(Self {
            method: Method::POST,
            path: OutgoingPath::Static(path),
            query: None,
            content_type: Some(content_type),
            body: Some(reqwest::Body::wrap(BufferedBody::new(
                upload.bytes,
                upload.budget,
            ))),
            content_length: Some(content_length),
            upload: None,
        })
    }

    pub(crate) fn direct(
        path: &'static str,
        content_type: HeaderValue,
        body: Body,
        expected: Option<u64>,
        maximum: u64,
    ) -> Self {
        let state = SharedUploadState::new(UploadState::Incomplete);
        let direct = DirectRequestBody::new(body, expected, maximum, state.clone());
        Self {
            method: Method::POST,
            path: OutgoingPath::Static(path),
            query: None,
            content_type: Some(content_type),
            body: Some(reqwest::Body::wrap(direct)),
            content_length: expected,
            upload: Some(state),
        }
    }

    pub(crate) fn control(
```

- `buffered`：完整的字节在 `upload.bytes` 里（`Bytes`，引用计数，克隆不拷贝），还带着一个缓冲字节预算的许可 `upload.budget`——**可以重放**，只要在重试期间一直持有字节和许可；
- `direct`：一个流式的 `DirectRequestBody`，旁边一个 `SharedUploadState` 记录上传进度。流交给 `send()` 之后，连接失败时没有现成的副本——**不能直接重放**。

另外，HTTP 客户端在构建时明确关掉了 reqwest 自带的重试：

```rust title="sglang_omni_router/rust/src/worker_pool/resolver.rs @ 921ea2c8 L108-117"
pub(super) fn build_http_client(
    connect_timeout: Duration,
    pool_idle_timeout: Duration,
    pool_max_idle_per_host: usize,
) -> Result<Client, reqwest::Error> {
    Client::builder()
        .no_proxy()
        .redirect(Policy::none())
        .retry(reqwest::retry::never())
        .http1_only()
```

（文档里也写着"Redirects, ambient proxies, retries, and automatic decompression are disabled"。）所以重试必须在 router 自己的逻辑里做，而且要换 worker——reqwest 的重试只会对同一个地址重发。

### 第三步：定范围

把"安全"和"有用"拆开想：

| 问题 | 结论 |
| --- | --- |
| 哪些失败可以重试？ | 只有连接阶段的失败（`reqwest::Error::is_connect()`：连接被拒、连接超时）。请求体一个字节都没发出去，不存在重复执行的风险。读响应时断开、上传到一半出错，都不重试——生成类的 POST 可能已经在 worker 上跑了一半 |
| 哪些请求体能重放？ | 缓冲的（分类路径、控制类请求）可以；直接路径的流不行 |
| 换哪个 worker？ | 重新走一次选择，排除刚失败的 worker；没有其他兼容的健康 worker 就返回原来的错误 |
| 重试几次、等多久？ | 次数有上限（比如 1～2 次），每次之前加带抖动的退避（full jitter），总时间不超过请求原来的截止时间 |
| 默认开吗？ | 默认关闭，加配置项（和 #2508 的熔断一样的风格），给运维选择权 |
| 怎么观察？ | 一个按结果分类的重试计数指标（标签用固定词表，不带 worker id）；文档 `docs/basic_usage/omni_router.md` 加一节 |

直接路径怎么办？有三个选项，各有代价：

1. **只做缓冲路径**：范围最小、最安全，但第十章实验里那种最常见的同质池场景不受益；
2. **小请求体先缓冲**：直接路径上 `content-length` 小于某个阈值时先读进内存，变成可重放的——代价是这部分请求失去流式转发、多一次内存占用；
3. **可回收的请求体**：给直接路径的流包一层，在它**第一次被 poll 之前**可以把原始的流取回来（连接失败时 reqwest 还没开始读请求体）。最优雅，但要仔细证明"连接失败时流一定没被读过"，和 `SharedUploadState`、截止时间的交互也要处理。

这种取舍**不该一个人在本地决定**。合理的做法是：第一个 PR 只做选项 1（附上测试和指标，并在描述里说明直接路径的限制），同时在 issue 里提出 2 和 3 让维护者选；或者在认领评论里就把这个问题摆出来。

还要处理和熔断（#2508）的关系：它改的是同一片代码（`http_relay/mod.rs`、`worker_pool/`），并且明确写了"requests are not retried"。重试应该基于它的分支做，重试选 worker 时要尊重熔断状态（熔断打开的 worker 不参与选择）。

### 第四步：测试计划

router 的集成测试用真实的回环 socket（第十章）。重试至少要覆盖：

- 拒绝连接的 worker + 一个正常 worker → 请求成功，落在正常 worker 上，重试计数 +1；
- 所有 worker 都拒绝连接 → 达到上限后返回 502，重试次数等于上限；
- 直接路径（或流式上传）→ 不重试（如果第一个 PR 只做缓冲路径）；
- 截止时间很短 → 不会超过截止时间才返回；
- worker 接受连接后中途断开 → 不重试（非连接错误）；
- 熔断打开的 worker 不会被重试选中（基于 #2508）。

每一条都要在去掉实现之后失败——#2508 的描述里专门写了"去掉熔断过滤，回归测试就失败；恢复之后通过"，这是证明测试真的在测东西的好习惯。

### 第五步：认领评论

```text title="认领评论（贴在 #1623 下）"
I'd like to take the "Retry with jitter on upstream connect failure" item of Phase 1c.

Reading the current code, a refused connection surfaces as 502 upstream_protocol_error even though
no byte reached any worker (HttpRelay::send maps every send error to UpstreamProtocolError). I can
reproduce it with two loopback workers: stop one, and the next request routed to it fails while the
other worker is healthy.

Proposed scope, building on #2508 (which explicitly leaves retries out):
- retry only reqwest connect errors (is_connect), never after any byte was sent
- re-run worker selection excluding the failed worker and respecting open circuits
- bounded attempts with full-jitter backoff, never past the request deadline
- opt-in config like the breaker, a retry outcome counter, docs in omni_router.md
- real-TCP integration tests (refused -> failover, all refused -> 502, deadline, mid-stream failure not retried)

One open question: homogeneous pools use the direct path, whose streamed body cannot be replayed.
Should the first PR cover buffered bodies only, or would you prefer buffering small direct bodies
(or a reclaimable body wrapper) so that homogeneous pools also benefit?
```

注意它做了四件事：说明你读过代码（指出具体的函数）、说明你能复现、给出范围和"不做什么"、把需要维护者拍板的问题摆出来。

### 第六步：PR 描述

等维护者回复、方向确定之后再写代码。PR 描述照仓库的习惯写：

```text title="PR 描述的骨架"
[Router] Retry upstream connect failures on another worker

## Motivation
A refused upstream connection returns 502 even though the request never reached a worker.
During rolling restarts this turns a healthy cluster into a burst of client-visible errors.
Part of #1623 Phase 1c; builds on #2508.

## Modifications
- http_relay: distinguish connect failures from other send errors
- worker_pool: re-select excluding the failed worker; open circuits are not eligible
- config: http.connect_retry_attempts (default 0 = off), http.connect_retry_base_backoff_ms
- metrics: retry outcome counter with a fixed label set
- docs: "Connect retries" section in omni_router.md

## Verification
- cargo test --locked: N passed (M new real-TCP tests: ...)
- removing the retry makes the failover test fail; restoring it passes
- cargo clippy --all-targets -D warnings, cargo fmt --check, pre-commit run --all-files
- default config: behavior unchanged (retries disabled)
```

## 一个发现不一定是 bug：CPU 上的大流式 chunk

第六章的实验里，纯 CPU 部署下一个 64 KB 的流式 chunk 让发送方进程退出：`ValueError: cpu stream chunk cannot be sent from 'src' to non-GPU target 'far'`。动手之前，先回答三个问题：

1. **这是受支持的场景吗？** 仓库有 Intel CPU 平台的安装文档（`docs/get_started/installation_cpu.md`）和 CPU CI（`omni-cpu-ci.yaml`），所以"纯 CPU 部署"是受支持的。但本书的环境是 `OmniSRTPlatform`，不是真正的 Intel CPU 平台（`CPUOmniPlatform`），要先在 CPU 平台上复现。
2. **真实的模型会走到吗？** Qwen3-TTS 的码每帧 16 个 int64，一个 chunk 远小于 16 KB 的内联上限；thinker 的隐状态一帧几 KB，攒几帧就可能超过。要找一个 CPU 平台上支持的、会产生大流式 chunk 的模型。
3. **修法有没有争议？** `outbound_payload` 对 CPU 张量会选 SHM，`outbound_stream` 却没有这个分支——两者对齐看起来很自然，但也可能是有意的（比如 CPU 平台上不希望用 SHM 传流）。

所以这里正确的第一步是**开一个 issue**：现象、最小复现（第六章的 `ch6_pipeline.py big`）、你认为的原因、你不确定的地方、一个可能的修法，问维护者这是不是预期行为。等确认之后再提 PR，PR 里带上单测（在 `tests/unit_test/comm/` 下构造一个两端都不是 GPU stage 的 `CommRouter`，断言大 chunk 选 SHM）。

## 交出 PR 之后

- **保持小**：一个 PR 只做一件事；顺手发现的问题另开 issue。
- **跟进 review**：每条评论都回复（改了就说改了，不同意就讲理由），在同一个分支上追加提交；仓库是 squash 合并，提交历史不用太讲究。
- **CI 失败先自己看**：区分"我的代码"、"性能阈值本身不稳"和"基础设施问题"（第十一章），在 PR 里写清判断。
- **等待是正常的**：这个仓库合并的 PR 中位数是一天，但需要设计讨论的会更久；一周没动静可以礼貌地 @ 相关维护者（CODEOWNERS 里能找到）。

## 练习

**1. 跑一次扫描。** 用本章的 `scan_issues.py`（或者网页上的筛选）找出三个候选 issue，对每个在基准提交或当前 main 上验证"问题还在不在"，写下你的判断和证据。

??? success "参考思路"
    验证的方式取决于 issue 的类型：报错类看报错的那行代码还在不在（`git grep`）；性能类看相关的默认值和注释；功能类看有没有已合入的 PR（`git log --grep='#编号'`）。只有"问题还在、没人认领、方向清楚"的才值得动手。

**2. 实现重试的第一步。** 在本地的 router 里，只做"区分连接失败"：把 `send` 里的 `Err(_source)` 改成检查 `source.is_connect()`，连接失败时打一条带 worker 地址的 warn 日志，并让第十章的实验能在日志里看到它。跑 `cargo test` 和 `cargo clippy -D warnings`。

??? success "参考思路"
    `Err(source) => { if source.is_connect() { tracing::warn!(worker = %lease.target().base_url(), error = %source, "upstream connect failed"); } ... }`。注意 clippy 的 lint 配置禁止 `unwrap` / `expect`；日志字段不要带请求内容。这一步本身行为不变，可以作为一个独立的小 PR（"让连接失败在日志里可见"），也是后续重试 PR 的基础。

**3. 写 issue。** 按本章最后一节的三个问题，把 CPU 大流式 chunk 的发现写成一个完整的英文 issue（现象、复现、原因、不确定的地方、建议的修法）。

??? success "参考思路"
    标题像 "[Bug][CPU] Stream chunks above 16 KiB crash the producer stage when neither stage is on a GPU"。复现用第六章的两个文件；原因指向 `CommRouter.outbound_stream` 的最后一个分支，并对比 `outbound_payload`；不确定的地方写"是否有 CPU 平台上的模型会产生这么大的 chunk、SHM 是否是期望的传输"；修法写"非 GPU 两端的 CPU chunk 走 SHM，加单测"。

!!! interview "怎么讲清楚"
    讲"你怎么给一个陌生的开源项目做贡献"，按这一章的顺序讲：先读懂架构（这个项目的 Stage / Coordinator / 通信层），再从六个地方找切入点（路线图未勾项、TODO、默认关闭的开关、文档代码不一致、自己实验撞到的边界、跨模型推广），每个候选都在 main 上验证并查重；动手前在 issue 里认领，写清范围、不做什么和需要维护者拍板的问题；PR 小而完整，测试能证明自己在测东西，描述按动机 / 改动 / 验证写，带数字。用 router 重试做例子：一个看起来只是"加个重试"的改动，读完代码才发现直接路径的请求体不能重放、要和熔断协调、只能对连接错误重试。

## 小结

- [x] 简单 issue 会被秒认领，有分量的贡献靠自己读代码找：路线图未勾项、TODO、默认关闭的开关、文档代码不一致、实验撞到的边界、跨模型推广、过期 issue 清理。
- [x] 动手前确认：没人认领、没有 open PR、main 上问题还在、维护者认可方向。
- [x] router 重试案例：502 来自 `send()` 不区分错误种类；同质池走直接路径，流式请求体不能重放；范围限定为连接错误 + 可重放的请求体 + 换 worker + 有界抖动退避 + 默认关闭；基于 #2508。
- [x] 认领评论四要素：读过的代码、复现、范围与不做什么、待决问题；PR 描述按动机 / 改动 / 验证写。
- [x] 自己的发现先确认是否受支持、是否真实可达、修法是否有争议，再开 issue、再提 PR。
