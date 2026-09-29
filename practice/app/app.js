// 练习题前端：#/ 是题库列表，#/p/<id> 是做题页。数据由 practice/build.py 生成在 data/ 下。
/* global marked, hljs, katex, CodeMirror */
(function () {
  "use strict";

  const $ = (sel, el = document) => el.querySelector(sel);
  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const app = $("#app");
  const DIFF = { 简单: 1, 中等: 2, 困难: 3 };
  const ENV = { browser: "浏览器判题", local: "需要本地 Python", cpp: "C++ 本地判题", torch: "需要 PyTorch", cuda: "需要 NVIDIA GPU" };
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
    // 题解里可以用 cpp://basics/move/ 这样的跨手册链接，和手册正文的写法一致
    return html.replace(/href="(python|cpp|llm|cuda|train|serving|minisgl):\/\//g, 'href="../$1/').replace(/href="root:\/\//g, 'href="../');
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
    worker: null, ready: null, seq: 0, pending: {}, version: "", pyVersion: "", assistOn: false,
    readyText: "",                   // 当前页面就绪时显示什么：能判题的题和只在本地判题的题不一样
    status(text, cls) {
      const el = $("#pystatus");
      if (el) { el.className = "pystatus " + (cls || ""); el.lastChild.textContent = text; }
    },
    showReady() {
      this.status((this.readyText || `Python ${this.pyVersion ? "(Pyodide " + this.pyVersion + ")" : ""} 已就绪`) +
        (this.assistOn ? " · 代码补全已开启" : ""), "ready");
    },
    start() {
      if (this.worker) { if (this.pyVersion) this.showReady(); return this.ready; }
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
        .then((m) => { this.pyVersion = m.version || "?"; this.showReady(); })
        .catch((e) => { this.status("运行环境加载失败：" + e.message, ""); this.worker = null; throw e; });
      return this.ready;
    },
    // 代码补全的请求：超时只返回 null（结果晚到就丢掉），不像判题那样重启 worker
    assist(op, payload, timeout = 8000) {
      if (!this.worker) return Promise.resolve(null);
      const id = ++this.seq;
      return new Promise((resolve) => {
        const timer = setTimeout(() => { delete this.pending[id]; resolve(null); }, timeout);
        this.pending[id] = { timer, reject: () => resolve(null), resolve: (m) => { try { resolve(JSON.parse(m.result)); } catch (e) { resolve(null); } } };
        this.worker.postMessage({ type: "assist", op, ...payload, id });
      });
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
    kill() { if (this.worker) this.worker.terminate(); this.worker = null; this.ready = null; this.assistOn = false; },
    async run(problem, code, mode) {
      await this.start();
      const limit = mode === "run" ? 30000 : 60000;
      try {
        const m = await this.call({ type: "run", code, tests: problem.tests, mode, requires: problem.requires }, limit);
        this.showReady();
        return m.result;
      } catch (e) {
        if (e.timeout) {
          this.start().then(() => assistWarm(problem)).catch(() => {});
          return { status: "timeout", cases: [], passed: 0, total: 0, error: `运行超过 ${limit / 1000} 秒还没结束：可能有死循环，或者算法太慢。` };
        }
        this.showReady();
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
            <p>七本手册每个章节配套的编程题。在浏览器里直接写代码、跑测试（Python 运行在 WebAssembly 里，不需要安装任何东西），
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
          <select id="f-env"><option value="all">全部环境</option><option value="browser">浏览器判题</option><option value="local">需要本地 Python</option><option value="cpp">C++ 本地判题</option><option value="torch">需要 PyTorch</option><option value="cuda">需要 NVIDIA GPU</option></select>
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

  // ---------------------------------------------------------------- 代码补全、函数签名与语法检查
  // Python：补全和签名由 worker 里的 Jedi 做静态分析（不执行代码），语法检查用 CPython 自己的解析器；
  // C++：浏览器里没有编译器，只补关键词、标准库常用名和当前代码里出现过的名字。
  const TYPE_ICON = { function: "ƒ", class: "C", module: "M", instance: "v", statement: "v", param: "p", keyword: "k", property: "@", path: "/", std: "s", word: "w" };
  const TYPE_NAME = { function: "函数", class: "类", module: "模块", instance: "变量", statement: "变量", param: "参数", keyword: "关键字", property: "属性", path: "路径", std: "标准库", word: "本文件" };
  const IDENT = /[A-Za-z_]\w*$/;
  let hintCache = null;        // 上一次向 Jedi 要到的候选：继续输入同一个名字时在本地过滤，不再请求
  let docBox = null, docHideTimer = null, sigBox = null, sigTimer = null, sigSeq = 0;

  async function assistWarm(problem) {
    if (!judge.worker || problem.lang === "cpp") return;
    const ok = await judge.assist("warm", { requires: problem.requires || [] }, 120000);
    if (ok) { judge.assistOn = true; judge.showReady(); }
  }

  function identAt(cm, pos) {
    const text = cm.getLine(pos.line).slice(0, pos.ch);
    const m = IDENT.exec(text), prefix = m ? m[0] : "";
    return { prefix, fromCh: pos.ch - prefix.length, before: text.slice(0, pos.ch - prefix.length) };
  }

  function renderHintItem(el, data, cur) {
    el.innerHTML = `<span class="hi-ic t-${cur.type}">${TYPE_ICON[cur.type] || "·"}</span><span class="hi-nm">${esc(cur.displayText)}</span>` +
      `<span class="hi-ty">${TYPE_NAME[cur.type] || ""}</span>`;
  }

  function hintList(c, cur, id) {
    const low = id.prefix.toLowerCase();
    const list = c.items.filter((it) => it.name !== id.prefix && it.name.toLowerCase().startsWith(low))
      .map((it) => ({ text: it.insert, displayText: it.name, type: it.type, index: it.index, render: renderHintItem }));
    const data = { list, from: CodeMirror.Pos(cur.line, id.fromCh), to: cur };
    attachDoc(data, c.token);
    return data;
  }

  function pyHint(cm, callback) {
    const cur = cm.getCursor(), id = identAt(cm, cur), c = hintCache;
    if (c && c.line === cur.line && c.fromCh === id.fromCh && c.before === id.before && id.prefix.startsWith(c.prefix)) {
      callback(hintList(c, cur, id));
      return;
    }
    judge.assist("complete", { code: cm.getValue(), line: cur.line + 1, col: cur.ch }).then((r) => {
      if (!r || !r.items) { callback(null); return; }
      hintCache = { line: cur.line, fromCh: id.fromCh, before: id.before, prefix: r.prefix, token: r.token, items: r.items.map((it, i) => ({ ...it, index: i })) };
      const now = cm.getCursor(), id2 = identAt(cm, now);          // 等结果期间可能又打了字
      if (now.line !== cur.line || id2.fromCh !== id.fromCh || !id2.prefix.startsWith(r.prefix)) { callback(null); return; }
      callback(hintList(hintCache, now, id2));
    });
  }
  pyHint.async = true;

  // 选中一个候选时，在列表旁边显示它的签名和文档（选中时才向 Jedi 要，避免一次算几十个）
  function hideDoc() { if (docBox) docBox.hidden = true; }
  function attachDoc(data, token) {
    let timer = null;
    CodeMirror.on(data, "select", (item, el) => {
      clearTimeout(timer); clearTimeout(docHideTimer);
      if (item.index === undefined || window.innerWidth < 700) { hideDoc(); return; }
      timer = setTimeout(async () => {
        const d = await judge.assist("detail", { token, index: item.index }, 3000);
        const ul = el.parentNode;
        if (!d || !ul || !ul.isConnected || !el.classList.contains("CodeMirror-hint-active")) return;
        if (!d.signatures.length && !d.doc) { hideDoc(); return; }
        if (!docBox) { docBox = document.createElement("div"); docBox.className = "hint-doc"; document.body.appendChild(docBox); }
        docBox.innerHTML = (d.signatures.length ? d.signatures.slice(0, 2) : [d.name]).map((s) => `<code>${esc(s)}</code>`).join("") +
          (d.signatures.length > 2 ? `<span class="more">共 ${d.signatures.length} 种签名</span>` : "") + (d.doc ? `<p>${esc(d.doc)}</p>` : "");
        // 优先放在列表右边（不挡题目描述），右边放不下再放左边，都不够就放在下面
        const r = ul.getBoundingClientRect(), right = window.innerWidth - r.right - 14, left = r.left - 14;
        let x, y = r.top;
        if (right >= 260) { docBox.style.maxWidth = Math.min(520, right) + "px"; x = r.right + 6; }
        else if (left >= 260) { docBox.style.maxWidth = Math.min(520, left) + "px"; x = null; }
        else { docBox.style.maxWidth = ""; x = Math.max(8, r.left); y = r.bottom + 6; }
        docBox.hidden = false;
        const w = docBox.offsetWidth, h = docBox.offsetHeight;
        docBox.style.left = (x === null ? r.left - w - 6 : x) + "px";
        docBox.style.top = Math.max(8, Math.min(y, window.innerHeight - h - 8)) + "px";
      }, 60);
    });
    CodeMirror.on(data, "update", () => { clearTimeout(timer); docHideTimer = setTimeout(hideDoc, 200); });
    CodeMirror.on(data, "close", () => { clearTimeout(timer); hideDoc(); });
  }

  // 光标在函数调用的括号里时，在上方显示签名，当前参数加粗
  function insideCall(cm, cur) {
    let text = cm.getRange(CodeMirror.Pos(Math.max(0, cur.line - 30), 0), cur);
    text = text.replace(/'''[\s\S]*?'''|"""[\s\S]*?"""|'(?:\\.|[^'\\\n])*'|"(?:\\.|[^"\\\n])*"/g, '""').replace(/#[^\n]*/g, "");
    let depth = 0;
    for (let i = text.length - 1; i >= 0; i--) {
      const ch = text[i];
      if (ch === ")" || ch === "]" || ch === "}") depth++;
      else if (ch === "(" || ch === "[" || ch === "{") { if (depth === 0) return ch === "("; depth--; }
    }
    return false;
  }
  function hideSig() { clearTimeout(sigTimer); sigSeq++; if (sigBox) sigBox.hidden = true; }
  function updateSig(cm, delay = 120) {
    clearTimeout(sigTimer);
    sigTimer = setTimeout(async () => {
      const cur = cm.getCursor();
      if (!cm.hasFocus() || !insideCall(cm, cur)) { hideSig(); return; }
      const seq = ++sigSeq;
      const sigs = await judge.assist("signatures", { code: cm.getValue(), line: cur.line + 1, col: cur.ch }, 4000);
      if (seq !== sigSeq) return;
      if (!sigs || !sigs.length) { hideSig(); return; }
      const s = sigs[0];
      const params = s.params.map((p, i) => (i === s.index ? `<b>${esc(p)}</b>` : esc(p))).join(", ");
      if (!sigBox) { sigBox = document.createElement("div"); sigBox.className = "sig-tip"; document.body.appendChild(sigBox); }
      sigBox.innerHTML = `<code>${esc(s.name)}(${params})</code>` + (sigs.length > 1 ? `<span class="more">共 ${sigs.length} 种签名</span>` : "") +
        (s.doc ? `<p>${esc(s.doc.split("\n\n")[0])}</p>` : "");
      sigBox.hidden = false;
      const c = cm.cursorCoords(null, "window"), w = sigBox.offsetWidth, h = sigBox.offsetHeight;
      sigBox.style.left = Math.max(8, Math.min(c.left - 12, window.innerWidth - w - 8)) + "px";
      sigBox.style.top = (c.top - h - 6 > 64 ? c.top - h - 6 : c.bottom + 6) + "px";
    }, delay);
  }

  // 停止输入 0.7 秒后做一次语法检查：出错的地方画波浪线，编辑器右下角显示错误，点一下跳过去
  function setupLint(cm) {
    let timer = null, seq = 0, marks = [];
    const bar = $("#lintbar");
    const run = async () => {
      const my = ++seq;
      const errs = await judge.assist("check", { code: cm.getValue() }, 4000);
      if (my !== seq || !errs || !bar.isConnected) return;
      marks.forEach((m) => m.clear());
      marks = [];
      if (!errs.length) { bar.hidden = true; return; }
      const e = errs[0], last = cm.lastLine();
      const line = Math.min(e.line - 1, last), len = cm.getLine(line).length;
      let from = CodeMirror.Pos(line, Math.min(e.col, len)), to = CodeMirror.Pos(Math.min(e.end_line - 1, last), e.end_col);
      if (from.ch >= len) {                                  // 错误落在行尾：标出这一行的内容
        from = CodeMirror.Pos(line, Math.min(cm.getLine(line).search(/\S|$/), Math.max(0, len - 1)));
        to = CodeMirror.Pos(line, len);
      }
      if (CodeMirror.cmpPos(to, from) <= 0) to = CodeMirror.Pos(from.line, from.ch + 1);
      marks.push(cm.markText(from, to, { className: "cm-lint-err", attributes: { title: e.msg } }));
      bar.innerHTML = `<b>第 ${e.line} 行</b>${esc(e.msg)}`;
      bar.hidden = false;
      bar.onclick = () => { cm.setCursor(from); cm.focus(); };
    };
    cm.on("change", () => { clearTimeout(timer); timer = setTimeout(run, 700); });
    return run;
  }

  function showPyHint(cm) { cm.showHint({ hint: pyHint, completeSingle: false }); }

  function setupPythonAssist(cm, problem, browserOk) {
    const lint = setupLint(cm);
    const ensure = () => {
      if (judge.worker) return;
      judge.start().then(() => { lint(); return assistWarm(problem); }).catch(() => {});
    };
    if (!browserOk) cm.on("focus", ensure);                 // 本地判题的题：开始写代码时才加载 Python（补全要用）
    let trig = null;
    cm.on("inputRead", (c, change) => {
      const typed = change.text.join("\n");
      if (typed.length !== 1 || !/[A-Za-z_.]/.test(typed)) return;   // 粘贴、输入法上屏的多个字不触发
      const cur = c.getCursor(), tok = c.getTokenAt(cur);
      if (/string|comment|number/.test(tok.type || "")) return;
      if (typed === "." && /\d\.$/.test(c.getLine(cur.line).slice(0, cur.ch))) return;
      clearTimeout(trig);
      trig = setTimeout(() => {
        // 等待期间可能又打了空格、逗号之类：光标前已经不是名字或 "." 就不弹了
        const now = c.getCursor(), before = c.getLine(now.line).slice(0, now.ch);
        if (!c.state.completionActive && /(\.|[A-Za-z_]\w*)$/.test(before)) showPyHint(c);
      }, typed === "." ? 20 : 120);
    });
    cm.on("endCompletion", () => { hintCache = null; });
    cm.on("cursorActivity", (c) => {
      const cur = c.getCursor();
      if ((sigBox && !sigBox.hidden) || /[(,]\s*$/.test(c.getLine(cur.line).slice(0, cur.ch))) updateSig(c);
    });
    cm.on("blur", hideSig);
    cm.on("keydown", (c, e) => { if (e.key === "Escape") hideSig(); });
    return { ensure, lint };
  }

  const CPP_KEYWORDS = ("alignas alignof auto bool break case catch char class concept const consteval constexpr constinit const_cast continue co_await " +
    "co_return co_yield decltype default delete do double dynamic_cast else enum explicit extern false final float for friend if inline int long " +
    "mutable namespace new noexcept nullptr operator override private protected public reinterpret_cast requires return short signed sizeof " +
    "static static_assert static_cast struct switch template this thread_local throw true try typedef typename union unsigned using virtual void " +
    "volatile while size_t ptrdiff_t int8_t int16_t int32_t int64_t uint8_t uint16_t uint32_t uint64_t include").split(" ");
  const CPP_STD = ("vector array deque list forward_list map multimap unordered_map set unordered_set string string_view span optional nullopt variant " +
    "visit get tuple pair make_pair unique_ptr shared_ptr weak_ptr make_unique make_shared enable_shared_from_this move forward swap exchange " +
    "thread jthread stop_token mutex shared_mutex recursive_mutex lock_guard unique_lock scoped_lock shared_lock condition_variable " +
    "condition_variable_any atomic atomic_flag atomic_thread_fence memory_order memory_order_relaxed memory_order_acquire memory_order_release " +
    "memory_order_acq_rel memory_order_seq_cst future promise async packaged_task launch function bind invoke ref cref reference_wrapper " +
    "sort stable_sort partial_sort nth_element lower_bound upper_bound equal_range binary_search min max minmax clamp accumulate reduce " +
    "iota fill fill_n copy copy_n move_backward transform find find_if count count_if all_of any_of none_of remove_if erase erase_if unique " +
    "reverse rotate priority_queue queue stack greater less hash begin end size data ssize chrono milliseconds microseconds nanoseconds " +
    "steady_clock this_thread cout cerr endl printf runtime_error logic_error invalid_argument out_of_range bad_alloc exception " +
    "numeric_limits byte launder align aligned_alloc memcpy memset memmove allocator allocator_traits construct_at destroy_at " +
    "is_same_v is_trivially_copyable_v enable_if_t conditional_t decay_t remove_reference_t declval integral_constant size_t").split(" ");

  function cppHint(cm) {
    const cur = cm.getCursor(), id = identAt(cm, cur);
    let pool;
    if (/std::$/.test(id.before)) pool = CPP_STD.map((w) => [w, "std"]);
    else {
      const words = new Set(), re = /[A-Za-z_]\w{2,}/g, text = cm.getValue();
      for (let m = re.exec(text); m; m = re.exec(text)) words.add(m[0]);
      const member = /(\.|->)$/.test(id.before);
      pool = [...(member ? [] : CPP_KEYWORDS.map((w) => [w, "keyword"])), ...[...words].map((w) => [w, "word"])];
    }
    const seen = new Set(), list = [];
    for (const [w, type] of pool) {
      if (w === id.prefix || seen.has(w) || !w.startsWith(id.prefix)) continue;
      seen.add(w);
      list.push({ text: w, displayText: w, type, render: renderHintItem });
      if (list.length >= 60) break;
    }
    return { list, from: CodeMirror.Pos(cur.line, id.fromCh), to: cur };
  }

  function setupCppAssist(cm) {
    let trig = null;
    cm.on("inputRead", (c, change) => {
      const typed = change.text.join("\n");
      if (typed.length !== 1) return;
      const cur = c.getCursor(), before = c.getLine(cur.line).slice(0, cur.ch), tok = c.getTokenAt(cur);
      if (/string|comment|number|meta/.test(tok.type || "")) return;
      if (!/(::|\.|->)$/.test(before) && !/(^|\W)[A-Za-z_]\w+$/.test(before)) return;   // 名字至少打了两个字符，或刚打完 :: . ->
      clearTimeout(trig);
      trig = setTimeout(() => {
        const now = c.getCursor(), b2 = c.getLine(now.line).slice(0, now.ch);
        if (!c.state.completionActive && /(::|\.|->|[A-Za-z_]\w+)$/.test(b2)) c.showHint({ hint: cppHint, completeSingle: false });
      }, 80);
    });
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
    const isCpp = p.lang === "cpp";
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
            <span class="lang">${isCpp ? "C++20" : "Python 3"}</span>
            <span class="pystatus" id="pystatus"><i></i><span>${browserOk ? "点击运行时加载 Python" : "这道题需要在本地判题"}</span></span>
            <span class="grow"></span>
            <button class="btn" id="b-reset" title="恢复成模板代码">重置</button>
            <button class="btn" id="b-dl" title="下载当前代码，用于本地判题">下载</button>
            <button class="btn run" id="b-run" title="只跑样例（Ctrl/⌘ + Enter）" ${browserOk ? "" : "disabled"}>${ICON.play}运行</button>
            <button class="btn primary" id="b-submit" title="跑全部测试（Ctrl/⌘ + Shift + Enter）" ${browserOk ? "" : "disabled"}>${ICON.up}提交</button>
          </div>
          <div class="editor" id="editor"><div class="lintbar" id="lintbar" hidden></div></div>
          <div class="hsplit" id="hsplit"></div>
          <div class="console" id="console">
            <div class="ptabs"><button class="on">测试结果</button><span class="grow"></span><span class="hint keys"><kbd>Ctrl</kbd>/<kbd>⌘</kbd> + <kbd>Enter</kbd> 运行样例<span class="sep">·</span><kbd>Ctrl</kbd> + <kbd>空格</kbd> 或 <kbd>⌥</kbd> + <kbd>/</kbd> 代码补全</span></div>
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
        return `<div class="md">${p.explanation ? renderMd(p.explanation) : ""}<h3>参考解答</h3><pre><code class="language-${isCpp ? "cpp" : "python"}">${esc(p.solution)}</code></pre>
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
        if (isCpp) {
          md = `C++ 题在本地判题（浏览器里没有 C++ 编译器），macOS（Apple clang）和 Linux / WSL2（g++ 12 以上）都可以：\n\n` +
            "```bash\n" + `python practice/judge.py start ${n}      # 复制 .cpp 模板到 practice/workspace/\n` +
            `python practice/judge.py test ${n}       # C++20 编译，在 ${p.sanitize === "thread" ? "ThreadSanitizer" : "AddressSanitizer + UBSan"} 下运行测试\n` +
            `python practice/judge.py solution ${n}   # 参考解答\n` + "```\n\n" +
            "也可以在右边的编辑器里写，点「下载」保存成 `.cpp` 文件，再用 `python practice/judge.py test " + n + " 下载的文件.cpp` 判题。" +
            "sanitizer 报告的任何问题（越界、泄漏、未定义行为、数据竞争）都算不通过。\n\n测试代码：\n\n```cpp\n" + p.tests + "\n```\n";
        }
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
      value: saved !== null ? saved : p.starter, mode: isCpp ? "text/x-c++src" : "python", lineNumbers: true,
      indentUnit: isCpp ? 2 : 4, tabSize: isCpp ? 2 : 4,
      matchBrackets: true, autoCloseBrackets: true, styleActiveLine: true, viewportMargin: 50,
      extraKeys: {
        Tab: (c) => (c.somethingSelected() ? c.indentSelection("add") : c.replaceSelection(isCpp ? "  " : "    ", "end")),
        "Shift-Tab": (c) => c.indentSelection("subtract"),
        "Ctrl-Enter": () => doRun("run"), "Cmd-Enter": () => doRun("run"),
        "Shift-Ctrl-Enter": () => doRun("submit"), "Shift-Cmd-Enter": () => doRun("submit"),
        "Ctrl-/": "toggleComment", "Cmd-/": "toggleComment",
        // 手动补全：Mac 上 Ctrl + 空格常被用来切换输入法，所以再给 ⌥ + / 和 ⌘ + I
        "Ctrl-Space": (c) => completeNow(c), "Alt-/": (c) => completeNow(c), "Cmd-I": (c) => completeNow(c),
      },
    });
    const assist = isCpp ? { lint() {}, ensure() {} } : setupPythonAssist(cm, p, browserOk);
    if (isCpp) setupCppAssist(cm);
    function completeNow(c) {
      if (isCpp) { c.showHint({ hint: cppHint, completeSingle: false }); return; }
      assist.ensure();
      showPyHint(c);
    }
    let saveTimer = null;
    cm.on("change", () => { clearTimeout(saveTimer); saveTimer = setTimeout(() => store.set("code:" + slug, cm.getValue()), 400); });
    $("#b-reset").onclick = () => { if (confirm("恢复成模板代码？当前代码会丢失。")) { cm.setValue(p.starter); store.set("code:" + slug, null); } };
    $("#b-dl").onclick = () => {
      const a = document.createElement("a");
      a.href = URL.createObjectURL(new Blob([cm.getValue()], { type: isCpp ? "text/x-c++src" : "text/x-python" }));
      a.download = `${String(p.number).padStart(3, "0")}-${slug}${isCpp ? ".cpp" : ".py"}`;
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
    judge.readyText = browserOk ? "" : "这道题在本地判题";
    if (browserOk) judge.start().then(() => { assist.lint(); return assistWarm(p); }).catch(() => {});
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
