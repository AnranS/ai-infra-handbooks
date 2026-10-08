// 学习卡：题目来自构建时生成的 cards.json（tools/cards.py），复习安排保存在 localStorage 的 aig-cards 里。
// 调度是简化的 SM-2：每张卡有"难易系数"和"间隔"，答"重来"间隔清零，答"良好""简单"间隔按系数放大。
(function () {
  var KEY = "aig-cards", NEW_PER_DAY = 20, DAY = 86400000;
  var EN = /^en/.test(document.documentElement.lang || "");   // 英文卡片页（en/cards/）：界面文字与书名用英文
  function zhen(zh, en) { return EN ? en : zh; }
  var BOOK = EN
    ? { python: "Advanced Python", cpp: "Advanced C++", llm: "LLM Internals", cuda: "Advanced CUDA", scratch: "Train a Small Model", train: "Distributed Training",
        serving: "Inference Systems", minisgl: "mini-sglang from Scratch", cs: "CS Fundamentals", math: "Math Fundamentals",
        media: "Image & Video Generation", sglang: "SGLang Design Evolution" }
    : { python: "Python 进阶", cpp: "C++ 进阶", llm: "大模型原理", cuda: "CUDA 进阶", scratch: "从零训练一个小模型", train: "分布式训练",
        serving: "推理系统", minisgl: "手写 mini-sglang", cs: "计算机基础", math: "数学基础", media: "图像与视频生成", sglang: "SGLang 设计演进" };
  var TAGCLS = { "面试题": " iv", "自测": " st", Interview: " iv", "Self-test": " st" };
  var $ = function (id) { return document.getElementById(id); };
  var view = $("view"), cards = [], byId = {};
  var ui = { mode: "review", book: "all", kind: "all", src: "all", q: "", shown: false, current: null, limit: 30 };

  function load() {
    try { var s = JSON.parse(localStorage.getItem(KEY) || "null"); if (s && typeof s === "object") return Object.assign({ s: {}, day: "", fresh: 0 }, s); } catch (e) {}
    return { s: {}, day: "", fresh: 0 };
  }
  var state = load();
  function save() { try { localStorage.setItem(KEY, JSON.stringify(state)); } catch (e) {} }
  function today() { var d = new Date(); return d.getFullYear() + "-" + (d.getMonth() + 1) + "-" + d.getDate(); }
  function esc(s) { return String(s).replace(/[&<>"]/g, function (c) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]; }); }

  function inScope(c) { return (ui.book === "all" || c.b === ui.book) && (ui.kind === "all" || c.k === ui.kind); }
  function due(now) {
    return cards.filter(function (c) { var r = state.s[c.id]; return inScope(c) && r && r.due <= now; })
      .sort(function (a, b) { return state.s[a.id].due - state.s[b.id].due; });
  }
  // 2026-10 搬过家的章节：数学进了数学基础手册，从零训练的四章拆成了单独一本
  function movedId(d) {
    if (d.indexOf("llm/math/") === 0) return "math/" + d.slice(9);
    if (d.indexOf("train/scratch/") === 0) return "scratch/" + d.slice(14);
    return d === "train/practice/one-gpu" ? "scratch/one-gpu" : d;
  }
  function learned() {                                        // 路线图和各章学习条上标为"已学"的章节
    try { return ((JSON.parse(localStorage.getItem("aig-roadmap") || "null") || {}).done || []).map(movedId); } catch (e) { return []; }
  }
  function fresh() {
    var done = ui.src === "done" ? learned() : null;
    return cards.filter(function (c) { return inScope(c) && !state.s[c.id] && (!done || done.indexOf(c.b + "/" + c.p) >= 0); });
  }

  // 下一次的间隔（天）：again 当天再来（10 分钟后），其余按系数放大
  function schedule(r, grade) {
    r = Object.assign({ ease: 2.5, ivl: 0, reps: 0 }, r || {});
    if (grade === 0) { r.ease = Math.max(1.3, r.ease - 0.2); r.ivl = 0; r.reps = 0; }
    else if (grade === 1) { r.ease = Math.max(1.3, r.ease - 0.15); r.ivl = Math.max(1, Math.round(r.ivl * 1.2)); r.reps++; }
    else if (grade === 2) { r.ivl = r.reps === 0 ? 1 : r.reps === 1 ? 3 : Math.round(r.ivl * r.ease); r.reps++; }
    else { r.ease += 0.15; r.ivl = r.reps === 0 ? 4 : Math.round(r.ivl * r.ease * 1.3); r.reps++; }
    r.due = grade === 0 ? Date.now() + 10 * 60000 : Date.now() + r.ivl * DAY;
    r.last = Date.now();
    return r;
  }
  function label(r, grade) {
    var n = schedule(r, grade);
    if (grade === 0) return zhen("10 分钟后", "in 10 min");
    return EN ? "in " + n.ivl + " d" : n.ivl + " 天后";
  }

  function stats() {
    var now = Date.now(), all = cards.filter(inScope);
    var seen = all.filter(function (c) { return state.s[c.id]; }).length;
    if (state.day !== today()) { state.day = today(); state.fresh = 0; save(); }
    $("stats").innerHTML = '<span><b>' + due(now).length + "</b>" + zhen("张到期", " due") + "</span>" +
      "<span><b>" + Math.max(0, Math.min(NEW_PER_DAY - state.fresh, all.length - seen)) + "</b>" + zhen("张今天的新卡", " new today") + "</span>" +
      "<span><b>" + seen + "</b>/ " + all.length + zhen(" 张学过", " seen") + "</span>" +
      '<span class="tools"><button class="btn" id="anki" type="button" title="' +
      zhen("导出当前筛选范围内的卡片，Anki 里用「文件 → 导入」", "Export the cards in the current filter; in Anki use File → Import") + '">' +
      zhen("导出到 Anki", "Export to Anki") + "</button></span>";
    $("anki").onclick = exportAnki;
  }

  function pick() {
    var d = due(Date.now());
    if (d.length) return d[0];
    if (state.fresh >= NEW_PER_DAY) return null;
    var f = fresh();
    return f.length ? f[0] : null;
  }

  function meta(c) {
    return '<div class="meta"><span class="tag' + (TAGCLS[c.k] || "") + '">' + c.k + "</span>" +
      '<a href="../' + c.b + "/" + c.p + '/">' + esc(BOOK[c.b]) + " · " + esc(c.t) + "</a></div>";
  }
  function math(el) {
    if (window.renderMathInElement) renderMathInElement(el, { delimiters: [{ left: "\\(", right: "\\)", display: false },
      { left: "\\[", right: "\\]", display: true }, { left: "$$", right: "$$", display: true }, { left: "$", right: "$", display: false }], throwOnError: false });
  }

  function renderReview() {
    var c = ui.current || pick();
    ui.current = c;
    if (!c) {
      view.innerHTML = '<div class="card empty">' + zhen('这个范围里今天没有要复习的卡了。<br>可以换一本手册，或者切到"浏览"模式随便翻翻。',
        'Nothing left to review in this range today.<br>Try another handbook, or switch to "Browse" and look around.') + "</div>";
      return;
    }
    var r = state.s[c.id];
    var html = '<div class="card">' + meta(c) + '<div class="q">' + c.q + "</div>";
    if (!ui.shown) {
      html += '<div class="actions"><button class="show" id="show" type="button">' + zhen("显示答案", "Show answer") + "</button></div>" +
        '<div class="keys">' + zhen("空格：显示答案", "Space: show the answer") + "</div>";
    } else {
      html += '<div class="a">' + c.a + "</div>" + '<div class="actions">' +
        [["again", 0, zhen("重来", "Again")], ["hard", 1, zhen("困难", "Hard")], ["good", 2, zhen("良好", "Good")], ["easy", 3, zhen("简单", "Easy")]].map(function (b) {
          return '<button class="rate ' + b[0] + '" data-g="' + b[1] + '" type="button">' + b[2] + "<small>" + label(r, b[1]) + "</small></button>";
        }).join("") + '</div><div class="keys">' +
        zhen("键盘：1 重来 · 2 困难 · 3 良好 · 4 简单", "Keys: 1 Again · 2 Hard · 3 Good · 4 Easy") + "</div>";
    }
    view.innerHTML = html + "</div>";
    math(view);
    if ($("show")) $("show").onclick = function () { ui.shown = true; renderReview(); };
    [].forEach.call(view.querySelectorAll(".rate"), function (b) { b.onclick = function () { rate(+b.dataset.g); }; });
  }
  function rate(g) {
    var c = ui.current;
    if (!c || !ui.shown) return;
    if (!state.s[c.id]) state.fresh++;
    state.s[c.id] = schedule(state.s[c.id], g);
    save();
    ui.current = null;
    ui.shown = false;
    render();
  }

  function renderBrowse() {
    var q = ui.q.trim().toLowerCase();
    var list = cards.filter(function (c) {
      return inScope(c) && (!q || (c.t + " " + c.q + " " + c.a).toLowerCase().indexOf(q) >= 0);
    });
    view.innerHTML = '<div class="list">' + list.slice(0, ui.limit).map(function (c) {
      return "<details><summary>" + meta(c) + '<div class="q">' + c.q + '</div></summary><div class="a">' + c.a + "</div></details>";
    }).join("") + (list.length > ui.limit
      ? '<div class="more"><button class="btn" id="more" type="button">' +
        (EN ? "Show 30 more (" + list.length + " in all)" : "再显示 30 张（共 " + list.length + " 张）") + "</button></div>"
      : '<div class="more">' + (EN ? list.length + " in all" : "共 " + list.length + " 张") + "</div>") + "</div>";
    math(view);
    if ($("more")) $("more").onclick = function () { ui.limit += 30; renderBrowse(); };
  }

  function render() {
    stats();
    [].forEach.call(document.querySelectorAll(".seg button"), function (b) { b.setAttribute("aria-pressed", String(b.dataset.mode === ui.mode)); });
    if (ui.mode === "review") renderReview(); else renderBrowse();
  }

  function exportAnki() {
    var rows = ["#separator:tab", "#html:true", "#tags column:3"];
    cards.filter(inScope).forEach(function (c) {
      var src = '<div style="font-size:12px;color:#888">' + esc(BOOK[c.b] + " · " + c.t) + "</div>";
      var clean = function (s) { return s.replace(/\t/g, "    ").replace(/\n/g, "<br>"); };
      rows.push([clean(src + c.q), clean(c.a), "ai-infra::" + c.b + " ai-infra::" + c.k].join("\t"));
    });
    var a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([rows.join("\n")], { type: "text/tab-separated-values" }));
    a.download = "ai-infra-cards" + (ui.book === "all" ? "" : "-" + ui.book) + ".txt";
    document.body.appendChild(a);
    a.click();
    setTimeout(function () { URL.revokeObjectURL(a.href); a.remove(); }, 1000);
  }

  [].forEach.call(document.querySelectorAll(".seg button"), function (b) {
    b.onclick = function () { ui.mode = b.dataset.mode; ui.limit = 30; render(); };
  });
  $("book").onchange = function () { ui.book = this.value; ui.current = null; ui.shown = false; render(); };
  $("kind").onchange = function () { ui.kind = this.value; ui.current = null; ui.shown = false; render(); };
  $("src").onchange = function () { ui.src = this.value; ui.current = null; ui.shown = false; render(); };
  $("q").oninput = function () { ui.q = this.value; ui.limit = 30; if (ui.mode === "browse") renderBrowse(); };
  document.addEventListener("keydown", function (e) {
    if (ui.mode !== "review" || /input|select|textarea/i.test(e.target.tagName)) return;
    if (e.key === " " && !ui.shown && ui.current) { e.preventDefault(); ui.shown = true; renderReview(); }
    else if (ui.shown && "1234".indexOf(e.key) >= 0) rate(+e.key - 1);
  });

  Promise.all([fetch("cards.json").then(function (r) { return r.json(); }),
               fetch("../roadmap/chapters.json").then(function (r) { return r.json(); }).catch(function () { return { order: [] }; })])
  .then(function (res) {
    var data = res[0], rank = {};
    res[1].order.forEach(function (id, i) { rank[id] = i; });  // 新卡按学习路线图的顺序出现
    cards = data.map(function (c, i) { c._i = i; return c; }).sort(function (a, b) {
      var ra = rank[a.b + "/" + a.p], rb = rank[b.b + "/" + b.p];
      ra = ra == null ? 1e9 : ra; rb = rb == null ? 1e9 : rb;
      return ra - rb || a._i - b._i;
    });
    cards.forEach(function (c) { byId[c.id] = c; });
    // 2026-10 数学章节搬进了数学基础手册，卡片 id 变了：把旧 id（字段 o）下的复习记录挪到新 id
    var moved = false;
    cards.forEach(function (c) { if (c.o && state.s[c.o] && !state.s[c.id]) { state.s[c.id] = state.s[c.o]; delete state.s[c.o]; moved = true; } });
    if (moved) save();
    $("total").textContent = cards.length;
    Object.keys(BOOK).forEach(function (b) {
      var n = cards.filter(function (c) { return c.b === b; }).length;
      if (n) $("book").insertAdjacentHTML("beforeend", '<option value="' + b + '">' + BOOK[b] + (EN ? " (" + n + ")" : "（" + n + "）") + "</option>");
    });
    var hash = location.hash.slice(1);
    if (BOOK[hash]) { ui.book = hash; $("book").value = hash; }
    render();
  }).catch(function () { view.innerHTML = '<div class="card empty">' + zhen("卡片数据加载失败。", "Could not load the cards.") + "</div>"; });
})();
