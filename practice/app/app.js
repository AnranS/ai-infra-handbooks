// 练习题前端：#/ 是题库列表，#/p/<id> 是做题页。数据由 practice/build.py 生成在 data/ 下。
/* global marked, hljs, katex, CodeMirror */
(function () {
  "use strict";

  const $ = (sel, el = document) => el.querySelector(sel);
  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const app = $("#app");
  const DIFF = { 简单: 1, 中等: 2, 困难: 3 };
  const ENV = { browser: "浏览器判题", local: "需要本地 Python", torch: "需要 PyTorch", cuda: "需要 NVIDIA GPU" };
  const ICON = {
    solved: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12.5l4.5 4.5L19 7.5"/></svg>',
    tried: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><circle cx="12" cy="12" r="6.5"/></svg>',
    play: '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M8 5.5v13l11-6.5z"/></svg>',
    up: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 19V6M6 11l6-6 6 6"/></svg>',
  };

  // ---------------------------------------------------------------- 存储（每个人自己的浏览器里）
  const store = {
    get(key, dflt) { try { const v = localStorage.getItem("aig-practice:" + key); return v === null ? dflt : JSON.parse(v); } catch (e) { return dflt; } },
    set(key, v) { try { localStorage.setItem("aig-practice:" + key, JSON.stringify(v)); } catch (e) { /* 隐私模式等 */ } },
  };
  const statusOf = (slug) => (store.get("status", {})[slug] || "");
  function setStatus(slug, st) {
    const all = store.get("status", {});
    if (all[slug] === "solved" && st !== "solved") return;
    all[slug] = st;
    store.set("status", all);
  }

  // ---------------------------------------------------------------- 主题
  const root = document.documentElement;
  function isDark() {
    return root.dataset.theme ? root.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
  }
  function syncHl() { $("#hl-dark").disabled = !isDark(); $("#hl-light").disabled = isDark(); }
  $("#theme").addEventListener("click", () => {
    root.dataset.theme = isDark() ? "light" : "dark";
    try { localStorage.setItem("aig-theme", root.dataset.theme); } catch (e) { /* ignore */ }
    syncHl();
  });
  syncHl();

  // ---------------------------------------------------------------- Markdown + 公式
  function renderMd(md) {
    const math = [];
    const keep = (tex, display) => { math.push([tex, display]); return `@@MATH${math.length - 1}@@`; };
    let src = md.replace(/```[\s\S]*?```|`[^`\n]*`/g, (m) => m.replace(/\$/g, "@@DOLLAR@@"));
    src = src.replace(/\$\$([\s\S]+?)\$\$/g, (_, t) => keep(t, true)).replace(/\$([^$\n]+?)\$/g, (_, t) => keep(t, false));
    src = src.replace(/@@DOLLAR@@/g, "$");
    let html = marked.parse(src, { gfm: true });
    html = html.replace(/@@MATH(\d+)@@/g, (_, i) => {
      const [tex, display] = math[+i];
      try { return katex.renderToString(tex, { displayMode: display, throwOnError: false }); } catch (e) { return esc(tex); }
    });
    return html;
  }
  function highlight(el) { el.querySelectorAll("pre code").forEach((c) => { try { hljs.highlightElement(c); } catch (e) { /* ignore */ } }); }

  // ---------------------------------------------------------------- 数据
  let INDEX = null;
  const cache = {};
  async function loadIndex() {
    if (!INDEX) {
      const r = await fetch("data/index.json", { cache: "no-cache" });
      INDEX = await r.json();
      INDEX.bookName = Object.fromEntries(INDEX.books.map((b) => [b.id, b.name]));
      INDEX.chapter = {};
      for (const [book, list] of Object.entries(INDEX.chapters)) for (const c of list) INDEX.chapter[book + "/" + c.path] = c;
    }
    return INDEX;
  }
  async function loadProblem(slug) {
    if (!cache[slug]) {
      const r = await fetch(`data/p/${slug}.json?v=${INDEX.version}`);
      if (!r.ok) throw new Error("题目不存在");
      cache[slug] = await r.json();
    }
    return cache[slug];
  }

  // ---------------------------------------------------------------- 判题 worker
  const judge = {
    worker: null, ready: null, seq: 0, pending: {}, version: "",
    status(text, cls) {
      const el = $("#pystatus");
      if (el) { el.className = "pystatus " + (cls || ""); el.lastChild.textContent = text; }
    },
    start() {
      if (this.worker) return this.ready;
      this.worker = new Worker("worker.js?v=" + INDEX.version);
      this.worker.onmessage = (e) => {
        const m = e.data, p = this.pending[m.id];
        if (!p) return;
        if (m.type === "progress") { this.status(m.text, "busy"); return; }
        delete this.pending[m.id];
        clearTimeout(p.timer);
        if (m.type === "error") p.reject(new Error(m.error)); else p.resolve(m);
      };
      this.status("正在加载 Python 运行环境（首次约 10 MB）…", "busy");
      const base = location.href.replace(/[#?].*$/, "").replace(/[^/]*$/, "");
      this.ready = this.call({ type: "init", base, version: INDEX.version }, 120000)
        .then((m) => { this.status(`Python ${m.version ? "(Pyodide " + m.version + ")" : ""} 已就绪`, "ready"); })
        .catch((e) => { this.status("运行环境加载失败：" + e.message, ""); this.worker = null; throw e; });
      return this.ready;
    },
    call(msg, timeout) {
      const id = ++this.seq;
      return new Promise((resolve, reject) => {
        const timer = setTimeout(() => {
          delete this.pending[id];
          this.kill();
          reject(Object.assign(new Error("timeout"), { timeout: true }));
        }, timeout);
        this.pending[id] = { resolve, reject, timer };
        this.worker.postMessage({ ...msg, id });
      });
    },
    kill() { if (this.worker) this.worker.terminate(); this.worker = null; this.ready = null; },
    async run(problem, code, mode) {
      await this.start();
      const limit = mode === "run" ? 30000 : 60000;
      try {
        const m = await this.call({ type: "run", code, tests: problem.tests, mode, requires: problem.requires }, limit);
        this.status("已就绪", "ready");
        return m.result;
      } catch (e) {
        if (e.timeout) {
          this.start();
          return { status: "timeout", cases: [], passed: 0, total: 0, error: `运行超过 ${limit / 1000} 秒还没结束：可能有死循环，或者算法太慢。` };
        }
        this.status("已就绪", "ready");
        return { status: "runtime_error", cases: [], passed: 0, total: 0, error: e.message };
      }
    },
  };

  // ---------------------------------------------------------------- 列表页
  const filt = Object.assign({ book: "all", diff: "all", status: "all", env: "all", q: "" }, store.get("filters", {}));

  async function renderList() {
    document.title = "练习题 · AI Infra 学习手册";
    const idx = await loadIndex();
    const st = store.get("status", {});
    const solved = idx.problems.filter((p) => st[p.slug] === "solved");
    const byDiff = (d) => idx.problems.filter((p) => p.difficulty === d).length;
    const solvedDiff = (d) => solved.filter((p) => p.difficulty === d).length;
    app.innerHTML = `
      <div class="list-page">
        <div class="intro">
          <div>
            <h1>练习题</h1>
            <p>五本手册每个章节配套的编程题。在浏览器里直接写代码、跑测试（Python 运行在 WebAssembly 里，不需要安装任何东西），
            CUDA 题用 Python 版的 GPU 模拟器判题，会检查合并访存、bank conflict 和数据竞争。
            需要 PyTorch 或 NVIDIA GPU 的题在本地用命令行判题，支持 macOS 和 WSL2，见<a href="#/local">本地环境</a>。</p>
          </div>
          <div class="stats">
            <div class="stat"><b>${solved.length}<small style="font-size:14px;color:var(--faint)"> / ${idx.problems.length}</small></b><span>已通过</span></div>
            <div class="stat easy"><b>${solvedDiff("简单")}/${byDiff("简单")}</b><span>简单</span></div>
            <div class="stat medium"><b>${solvedDiff("中等")}/${byDiff("中等")}</b><span>中等</span></div>
            <div class="stat hard"><b>${solvedDiff("困难")}/${byDiff("困难")}</b><span>困难</span></div>
          </div>
        </div>
        <div class="filters">
          <div class="seg" id="f-book">${[["all", "全部"], ...idx.books.map((b) => [b.id, b.name])]
            .map(([v, n]) => `<button data-v="${v}" class="${filt.book === v ? "on" : ""}">${esc(n)}</button>`).join("")}</div>
          <select id="f-diff"><option value="all">全部难度</option><option>简单</option><option>中等</option><option>困难</option></select>
          <select id="f-status"><option value="all">全部状态</option><option value="todo">未开始</option><option value="tried">尝试过</option><option value="solved">已通过</option></select>
          <select id="f-env"><option value="all">全部环境</option><option value="browser">浏览器判题</option><option value="local">需要本地 Python</option><option value="torch">需要 PyTorch</option><option value="cuda">需要 NVIDIA GPU</option></select>
          <input id="f-q" type="search" placeholder="搜索题目、标签或章节" value="${esc(filt.q)}">
          <span class="count" id="f-count"></span>
        </div>
        <div id="plist"></div>
        <div class="local-box">
          <h3>在本地做题</h3>
          所有题目都可以在本地用自己的编辑器做：<code>python practice/judge.py start 12</code> 复制模板，
          <code>python practice/judge.py test 12</code> 判题。macOS（Apple Silicon，PyTorch 跑在 MPS 上）和 WSL2 + NVIDIA GPU（PyTorch / Triton / nvcc 跑在真卡上）的安装步骤见
          <a href="#/local" style="color:var(--blue)">本地环境</a>。
        </div>
      </div>`;
    $("#f-diff").value = filt.diff; $("#f-status").value = filt.status; $("#f-env").value = filt.env;
    const update = () => { store.set("filters", filt); drawList(); };
    $("#f-book").addEventListener("click", (e) => {
      const b = e.target.closest("button"); if (!b) return;
      filt.book = b.dataset.v;
      $("#f-book").querySelectorAll("button").forEach((x) => x.classList.toggle("on", x === b));
      update();
    });
    $("#f-diff").onchange = (e) => { filt.diff = e.target.value; update(); };
    $("#f-status").onchange = (e) => { filt.status = e.target.value; update(); };
    $("#f-env").onchange = (e) => { filt.env = e.target.value; update(); };
    $("#f-q").oninput = (e) => { filt.q = e.target.value.trim(); update(); };
    $("#plist").addEventListener("click", (e) => {
      const row = e.target.closest(".row");
      if (row && !e.target.closest("a")) location.hash = "#/p/" + row.dataset.slug;
    });
    drawList();
  }

  function drawList() {
    const idx = INDEX, st = store.get("status", {});
    const q = filt.q.toLowerCase();
    const list = idx.problems.filter((p) => {
      if (filt.book !== "all" && p.book !== filt.book) return false;
      if (filt.diff !== "all" && p.difficulty !== filt.diff) return false;
      if (filt.env !== "all" && p.env !== filt.env) return false;
      const s = st[p.slug] || "todo";
      if (filt.status !== "all" && s !== filt.status) return false;
      if (q) {
        const ch = idx.chapter[p.book + "/" + p.chapter];
        const hay = [p.title, p.slug, String(p.number), ...p.tags, ch ? ch.title : ""].join(" ").toLowerCase();
        if (!hay.includes(q)) return false;
      }
      return true;
    });
    $("#f-count").textContent = `${list.length} 道题`;
    if (!list.length) { $("#plist").innerHTML = '<div class="empty">没有符合条件的题目</div>'; return; }
    let html = "", book = null, chap = null;
    for (const p of list) {
      if (p.book !== book) {
        if (chap) html += "</div>";
        chap = null; book = p.book;
        const n = idx.problems.filter((x) => x.book === book);
        const done = n.filter((x) => st[x.slug] === "solved").length;
        html += `<div class="book-h"><h2>${esc(idx.bookName[book])}</h2><span>${done} / ${n.length} 已通过</span></div>`;
      }
      if (p.chapter !== chap) {
        if (chap) html += "</div>";
        chap = p.chapter;
        const c = idx.chapter[p.book + "/" + p.chapter];
        html += `<div class="chap"><div class="chap-h"><span>${esc(c.part)} · <b>${esc(c.title)}</b></span><a href="../${c.url}">读这一章 →</a></div>`;
      }
      const s = st[p.slug];
      html += `<div class="row" data-slug="${p.slug}">
        <span class="st ${s || ""}">${s ? ICON[s] : ""}</span>
        <span class="num">${p.number}.</span>
        <span class="ttl">${esc(p.title)}</span>
        <span class="tags">${p.env !== "browser" ? `<span class="tag env ${p.env}">${ENV[p.env]}</span>` : ""}${p.hasCuda ? '<span class="tag env cuda">+ CUDA C++</span>' : ""}${p.tags.slice(0, 3).map((t) => `<span class="tag">${esc(t)}</span>`).join("")}</span>
        <span class="diff d${DIFF[p.difficulty]}">${p.difficulty}</span></div>`;
    }
    html += "</div>";
    $("#plist").innerHTML = html;
  }

  // ---------------------------------------------------------------- 本地环境说明页
  async function renderLocal() {
    const idx = await loadIndex();
    document.title = "本地环境 · 练习题";
    app.innerHTML = `<div class="list-page"><p><a href="#/" style="color:var(--blue)">← 返回题库</a></p><div class="md" id="localmd"></div></div>`;
    $("#localmd").innerHTML = renderMd(idx.localGuide);
    highlight($("#localmd"));
  }

  // ---------------------------------------------------------------- 做题页
  let cm = null;
  let renderSeq = 0;                 // 切换题目后，旧题目的运行结果不要写到新页面上

  async function renderProblem(slug) {
    const seq = ++renderSeq;
    const idx = await loadIndex();
    const meta = idx.problems.find((p) => p.slug === slug);
    if (!meta) { app.innerHTML = '<div class="empty">题目不存在。<a href="#/">返回题库</a></div>'; return; }
    const p = await loadProblem(slug);
    const ch = idx.chapter[p.book + "/" + p.chapter];
    const i = idx.problems.indexOf(meta);
    const prev = idx.problems[i - 1], next = idx.problems[i + 1];
    document.title = `${p.number}. ${p.title} · 练习题`;
    const browserOk = p.env === "browser";
    app.innerHTML = `
      <div class="ws">
        <section class="pane left" id="left">
          <div class="ptabs" id="ltabs">
            <button data-t="desc" class="on">题目描述</button>
            <button data-t="sol">题解</button>
            <button data-t="subs">提交记录</button>
            <button data-t="local">本地运行</button>
            <span class="grow"></span>
            <a class="navbtn" href="#/" title="题库">题库</a>
            ${prev ? `<a class="navbtn" href="#/p/${prev.slug}" title="上一题：${esc(prev.title)}">‹</a>` : ""}
            ${next ? `<a class="navbtn" href="#/p/${next.slug}" title="下一题：${esc(next.title)}">›</a>` : ""}
          </div>
          <div class="scroll" id="lbody"></div>
        </section>
        <div class="splitter" id="vsplit"></div>
        <section class="pane right" id="right">
          <div class="edbar">
            <span class="lang">Python 3</span>
            <span class="pystatus" id="pystatus"><i></i><span>${browserOk ? "点击运行时加载 Python" : "这道题需要在本地判题"}</span></span>
            <span class="grow"></span>
            <button class="btn" id="b-reset" title="恢复成模板代码">重置</button>
            <button class="btn" id="b-dl" title="下载当前代码，用于本地判题">下载</button>
            <button class="btn run" id="b-run" title="只跑样例（Ctrl/⌘ + Enter）" ${browserOk ? "" : "disabled"}>${ICON.play}运行</button>
            <button class="btn primary" id="b-submit" title="跑全部测试（Ctrl/⌘ + Shift + Enter）" ${browserOk ? "" : "disabled"}>${ICON.up}提交</button>
          </div>
          <div class="editor" id="editor"></div>
          <div class="hsplit" id="hsplit"></div>
          <div class="console" id="console">
            <div class="ptabs"><button class="on">测试结果</button><span class="grow"></span><span class="hint"><kbd>Ctrl</kbd>/<kbd>⌘</kbd> + <kbd>Enter</kbd> 运行样例</span></div>
            <div class="scroll" id="cbody"><div class="hint">${browserOk ? "写完代码后点「运行」跑样例，点「提交」跑全部测试。" :
              "这道题需要本地环境，浏览器里跑不了：点「下载」保存代码，再按左边「本地运行」里的命令判题。"}</div></div>
          </div>
        </section>
      </div>`;

    const tabs = {
      desc() {
        const envNote = p.env === "browser" ? "" : `<div class="note warn">这道题${ENV[p.env]}，需要在本地判题：见左上角「本地运行」。</div>`;
        const cudaNote = p.cuda ? `<div class="note">这道题还有 <b>CUDA C++ 版本</b>：在 WSL2 + NVIDIA GPU 上用 nvcc 编译运行，见「本地运行」。</div>` : "";
        return `<div class="ph"><h1>${p.number}. ${esc(p.title)}</h1>
          <div class="meta"><span class="diff d${DIFF[p.difficulty]}">${p.difficulty}</span>
            ${p.tags.map((t) => `<span class="tag">${esc(t)}</span>`).join("")}
            ${p.env !== "browser" ? `<span class="tag env ${p.env}">${ENV[p.env]}</span>` : ""}
            <span style="color:var(--faint)">·</span><a href="../${ch.url}">${esc(idx.bookName[p.book])} › ${esc(ch.title)}</a></div></div>
          ${envNote}${cudaNote}<div class="md">${renderMd(p.description)}</div>`;
      },
      sol() {
        if (!store.get("reveal:" + slug, false)) {
          return `<div class="reveal"><p>先自己试试？看过题解再做，练习效果会打折扣。</p><button class="btn" id="b-reveal">查看参考解答</button></div>`;
        }
        return `<div class="md">${p.explanation ? renderMd(p.explanation) : ""}<h3>参考解答</h3><pre><code class="language-python">${esc(p.solution)}</code></pre>
          ${p.cuda ? `<h3>CUDA C++ 参考解答</h3><pre><code class="language-cpp">${esc(p.cuda.solution)}</code></pre>` : ""}</div>`;
      },
      subs() {
        const subs = store.get("subs:" + slug, []);
        if (!subs.length) return '<div class="hint">还没有提交过。</div>';
        return `<table class="subs"><tr><th>时间</th><th>结果</th><th>用例</th><th>耗时</th></tr>${subs.map((s) =>
          `<tr><td>${new Date(s.t).toLocaleString()}</td><td class="${s.status === "accepted" ? "ok" : "no"}">${VERDICT[s.status] || s.status}</td>
            <td>${s.passed} / ${s.total}</td><td>${s.ms ? Math.round(s.ms) + " ms" : ""}</td></tr>`).join("")}</table>`;
      },
      local() {
        const n = p.number;
        let md = `在本地用自己的编辑器做这道题（先按[本地环境](#/local)装好依赖）：\n\n` +
          "```bash\n" + `python practice/judge.py start ${n}      # 复制模板到 practice/workspace/\n` +
          `python practice/judge.py test ${n}       # 判题（--run 只跑样例）\n` + `python practice/judge.py solution ${n}   # 参考解答\n` + "```\n\n" +
          "也可以点右上角「下载」保存浏览器里的代码，然后 `python practice/judge.py test " + n + " 下载的文件.py`。\n";
        if (p.env === "torch") md += "\n这道题需要 PyTorch：macOS（Apple Silicon）上自动用 **MPS**，WSL2 + NVIDIA GPU 上用 **CUDA**，都没有时退回 CPU。\n";
        if (p.env === "cuda") md += "\n这道题需要 **NVIDIA GPU**（例如 WSL2 + CUDA）。\n";
        if (p.env === "local") md += "\n这道题要用浏览器里没有的功能（例如线程），需要在本地用 CPython 判题，macOS 和 WSL2 都可以。\n";
        if ((p.requires || []).includes("triton")) md += "\n这道题用 Triton：WSL2 + NVIDIA GPU 上自动使用真 Triton；macOS 和浏览器里用模拟器 minitl，写法完全相同。设置 `PRACTICE_TRITON=emulate` 可以强制用模拟器。\n";
        if (p.cuda) {
          md += `\n### CUDA C++ 版本\n\n` + "```bash\n" + `python practice/judge.py start ${n} --cuda   # 复制 .cu 模板\n` +
            `python practice/judge.py test ${n}          # WSL2 + NVIDIA GPU：nvcc 编译，在真卡上运行并报告耗时\n` + "```\n\n" +
            "没有 NVIDIA GPU 时（macOS、普通 Linux），同一命令会用仓库里的 CPU 模拟器编译运行，只检查正确性、不测性能。\n\n模板：\n\n```cpp\n" + p.cuda.starter + "\n```\n";
        }
        return `<div class="md">${renderMd(md)}</div>`;
      },
    };
    const showTab = (t) => {
      $("#ltabs").querySelectorAll("button").forEach((b) => b.classList.toggle("on", b.dataset.t === t));
      $("#lbody").innerHTML = tabs[t]();
      highlight($("#lbody"));
      const rv = $("#b-reveal");
      if (rv) rv.onclick = () => { store.set("reveal:" + slug, true); showTab("sol"); };
      $("#lbody").scrollTop = 0;
    };
    $("#ltabs").addEventListener("click", (e) => { const b = e.target.closest("button"); if (b) showTab(b.dataset.t); });
    showTab("desc");

    // 编辑器
    const saved = store.get("code:" + slug, null);
    cm = CodeMirror($("#editor"), {
      value: saved !== null ? saved : p.starter, mode: "python", lineNumbers: true, indentUnit: 4, tabSize: 4,
      matchBrackets: true, autoCloseBrackets: true, styleActiveLine: true, viewportMargin: 50,
      extraKeys: {
        Tab: (c) => (c.somethingSelected() ? c.indentSelection("add") : c.replaceSelection("    ", "end")),
        "Shift-Tab": (c) => c.indentSelection("subtract"),
        "Ctrl-Enter": () => doRun("run"), "Cmd-Enter": () => doRun("run"),
        "Shift-Ctrl-Enter": () => doRun("submit"), "Shift-Cmd-Enter": () => doRun("submit"),
        "Ctrl-/": "toggleComment", "Cmd-/": "toggleComment",
      },
    });
    let saveTimer = null;
    cm.on("change", () => { clearTimeout(saveTimer); saveTimer = setTimeout(() => store.set("code:" + slug, cm.getValue()), 400); });
    $("#b-reset").onclick = () => { if (confirm("恢复成模板代码？当前代码会丢失。")) { cm.setValue(p.starter); store.set("code:" + slug, null); } };
    $("#b-dl").onclick = () => {
      const a = document.createElement("a");
      a.href = URL.createObjectURL(new Blob([cm.getValue()], { type: "text/x-python" }));
      a.download = `${String(p.number).padStart(3, "0")}-${slug}.py`;
      a.click();
      setTimeout(() => URL.revokeObjectURL(a.href), 1000);
    };
    $("#b-run").onclick = () => doRun("run");
    $("#b-submit").onclick = () => doRun("submit");

    let busy = false;
    async function doRun(mode) {
      if (busy || !browserOk) return;
      busy = true;
      $("#b-run").disabled = $("#b-submit").disabled = true;
      $("#cbody").innerHTML = `<div class="hint">${mode === "run" ? "正在运行样例…" : "正在运行全部测试…"}</div>`;
      const code = cm.getValue();
      store.set("code:" + slug, code);
      let res;
      try { res = await judge.run(p, code, mode); } finally {
        busy = false;
        if (seq === renderSeq) $("#b-run").disabled = $("#b-submit").disabled = false;
      }
      if (seq === renderSeq) showResult(res, mode);
      if (mode === "submit") {
        setStatus(slug, res.status === "accepted" ? "solved" : "tried");
        const subs = store.get("subs:" + slug, []);
        subs.unshift({ t: Date.now(), status: res.status, passed: res.passed, total: res.total, ms: res.ms });
        store.set("subs:" + slug, subs.slice(0, 20));
      } else if (!statusOf(slug)) setStatus(slug, "tried");
    }

    // 拖动分隔条
    drag($("#vsplit"), (e) => {
      const w = $(".ws").getBoundingClientRect();
      $("#left").style.flex = `0 0 ${Math.min(75, Math.max(22, ((e.clientX - w.left) / w.width) * 100))}%`;
      cm.refresh();
    });
    drag($("#hsplit"), (e) => {
      const r = $("#right").getBoundingClientRect();
      const h = Math.min(r.height - 120, Math.max(80, r.bottom - e.clientY));
      $("#console").style.flex = `0 0 ${h}px`;
      cm.refresh();
    });
    if (browserOk) judge.start().catch(() => {});
    if (window.innerWidth > 900) cm.focus();           // 手机上不自动聚焦，免得弹出键盘、页面跳动
  }

  const VERDICT = { accepted: "通过", wrong_answer: "解答错误", runtime_error: "执行出错", compile_error: "语法错误", timeout: "超出时间限制" };

  function showResult(res, mode) {
    const ok = res.status === "accepted";
    const cls = ok ? "accepted" : res.status === "timeout" ? "timeout" : "bad";
    const skipped = res.skipped ? `，跳过 ${res.skipped} 个（需要本地环境）` : "";
    let title = VERDICT[res.status] || res.status;
    if (ok && mode === "run") title = "样例通过";
    let html = `<div class="verdict ${cls}"><b>${title}</b><span>${res.total ? `${res.passed} / ${res.total} 个用例通过${skipped}` : ""}${res.ms ? ` · ${Math.round(res.ms)} ms` : ""}</span></div>`;
    if (ok && mode === "run") html += `<div class="hint" style="margin-bottom:10px">样例通过了，点「提交」跑全部测试。</div>`;
    if (res.error) html += `<pre class="out err">${esc(res.error)}</pre>`;
    if (res.stdout) html += `<div class="hint" style="margin-top:8px">导入你的代码时的输出：</div><pre class="out">${esc(res.stdout)}</pre>`;
    if (res.cases && res.cases.length) {
      html += '<div class="cases">' + res.cases.map((c) => {
        const mk = { pass: "✓", fail: "✗", error: "✗", skip: "–" }[c.status];
        const open = c.status !== "pass" ? "open" : "";
        return `<div class="case ${c.status} ${open}"><div class="case-h"><span class="mk">${mk}</span><span class="nm">${esc(c.name)}</span>
          <span class="dc">${esc(c.doc || "")}</span><span class="ms">${Math.round(c.ms)} ms</span></div>
          <div class="case-b">${c.message ? `<pre class="out ${c.status === "skip" ? "" : "err"}">${esc(c.message)}</pre>` : ""}
          ${c.stdout ? `<div class="hint" style="margin-top:6px">输出：</div><pre class="out">${esc(c.stdout)}</pre>` : ""}
          ${!c.message && !c.stdout ? '<div class="hint">通过</div>' : ""}</div></div>`;
      }).join("") + "</div>";
    }
    $("#cbody").innerHTML = html;
    $("#cbody").querySelectorAll(".case-h").forEach((h) => { h.onclick = () => h.parentElement.classList.toggle("open"); });
  }

  function drag(handle, onMove) {
    handle.addEventListener("pointerdown", (e) => {
      e.preventDefault();
      handle.classList.add("drag");
      handle.setPointerCapture(e.pointerId);
      const move = (ev) => onMove(ev);
      const up = () => { handle.classList.remove("drag"); handle.removeEventListener("pointermove", move); handle.removeEventListener("pointerup", up); };
      handle.addEventListener("pointermove", move);
      handle.addEventListener("pointerup", up);
    });
  }

  // ---------------------------------------------------------------- 路由
  async function route() {
    const h = location.hash || "#/";
    try {
      if (h.startsWith("#/p/")) await renderProblem(decodeURIComponent(h.slice(4)));
      else if (h === "#/local") await renderLocal();
      else {
        const q = /^#\/\?q=(.*)$/.exec(h);          // #/?q=估算：带着搜索词打开题库
        if (q) Object.assign(filt, { q: decodeURIComponent(q[1]), book: "all", diff: "all", status: "all", env: "all" });
        await renderList();
      }
      if (!h.startsWith("#/p/")) window.scrollTo(0, 0);
    } catch (e) {
      app.innerHTML = `<div class="empty">加载失败：${esc(e.message)}。<a href="#/">返回题库</a></div>`;
    }
  }
  window.addEventListener("hashchange", route);
  route();
})();
