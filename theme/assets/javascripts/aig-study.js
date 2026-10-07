// 各章页面上的学习条：这一章排在学习路线图的第几周、是必学还是选学；一键标为已学（和路线图页面共用进度）；
// 读完之后按路线图的顺序去下一章。数据来自构建时生成的 roadmap/chapters.json，进度只存在本浏览器。
(function () {
  var BOOKS = ["python", "cpp", "math", "llm", "cuda", "train", "serving", "minisgl", "cs", "media", "sglang"];
  var EN = (document.documentElement.lang || "").indexOf("en") === 0;   // 英文站（en/ 下）用英文界面
  var DIRS = EN ? { f: "framework", o: "optimization", p: "platform" } : { f: "推理框架", o: "推理优化", p: "推理平台" };
  var LEVELS = EN ? { 1: "Core", 2: "Recommended", 3: "Optional" } : { 1: "必学", 2: "推荐", 3: "选学" };
  var T = EN ? { done: "✓ Done", mark: "Mark as done", see: "See it in the roadmap", hot: function (d) { return "Key for " + d; },
                 read: "Finished?", prog: function (n, m) { return "Roadmap progress " + n + " / " + m + " chapters"; }, next: "Next in the roadmap: " }
           : { done: "✓ 已学", mark: "标为已学", see: "在学习路线图中查看", hot: function (d) { return d + "重点"; },
               read: "读完了？", prog: function (n, m) { return "路线图进度 " + n + " / " + m + " 章"; }, next: "按路线图，下一章：" };
  var KEY = "aig-roadmap";
  var top = document.querySelector(".aig-study"), end = document.querySelector(".aig-study-end");
  if (!top || !window.fetch) return;
  var segs = location.pathname.split("/");
  var bi = -1;
  for (var i = 0; i < segs.length; i++) if (BOOKS.indexOf(segs[i]) >= 0) { bi = i; break; }
  if (bi < 0) return;
  var root = segs.slice(0, bi).join("/") + "/";
  var rest = segs.slice(bi).filter(function (s) { return s && s !== "index.html"; });
  var id = rest.join("/");

  function load() {
    try {
      var s = JSON.parse(localStorage.getItem(KEY) || "null");
      if (s && typeof s === "object") {
        // 2026-10 数学章节从大模型原理搬到了数学基础手册：llm/math/x → math/x
        if ((s.done || []).some(function (d) { return d.indexOf("llm/math/") === 0; })) {
          s.done = s.done.map(function (d) { return d.indexOf("llm/math/") === 0 ? "math/" + d.slice(9) : d; });
          save(s);
        }
        return s;
      }
    } catch (e) {}
    return { done: [], dir: "all" };
  }
  function save(s) { try { localStorage.setItem(KEY, JSON.stringify(s)); } catch (e) {} }
  function esc(s) { return String(s).replace(/[&<>"]/g, function (c) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]; }); }

  fetch(root + "roadmap/chapters.json").then(function (r) { return r.ok ? r.json() : null; }).then(function (data) {
    if (!data || !data.ch[id]) return;
    var info = data.ch[id], stage = info[0], weeks = info[1], lv = info[2], dirs = info[3];
    var pos = data.order.indexOf(id), next = data.order[pos + 1];
    var st = load();
    var hot = st.dir && st.dir !== "all" && dirs.indexOf(st.dir) >= 0;

    function isDone() { return (load().done || []).indexOf(id) >= 0; }
    function toggle() {
      var s = load(), done = s.done || [];
      var k = done.indexOf(id);
      if (k >= 0) done.splice(k, 1); else done.push(id);
      s.done = done;
      save(s);
      render();
    }
    function button() {
      return '<button type="button" class="aig-study__btn' + (isDone() ? " is-done" : "") + '">' + (isDone() ? T.done : T.mark) + "</button>";
    }
    function render() {
      var n = (load().done || []).filter(function (x) { return data.ch[x]; }).length;
      top.innerHTML = '<a class="aig-study__wk" href="' + root + "roadmap/#" + stage + '" title="' + T.see + '">' + esc(weeks) + "</a>" +
        '<span class="aig-study__lv lv' + lv + '">' + LEVELS[lv] + "</span>" +
        (hot ? '<span class="aig-study__dir">' + T.hot(DIRS[st.dir]) + "</span>" : "") + button();
      top.hidden = false;
      if (end) {
        end.innerHTML = '<div class="aig-study-end__row"><span>' + T.read + '</span>' + button() +
          '<span class="aig-study-end__count">' + T.prog(n, data.order.length) + "</span></div>" +
          (next ? '<a class="aig-study-end__next" href="' + root + next + '/">' + T.next + esc(data.ch[next][4]) +
            '<span class="aig-study-end__nwk">' + esc(data.ch[next][1]) + " · " + LEVELS[data.ch[next][2]] + "</span> →</a>" : "");
        end.hidden = false;
      }
      [].forEach.call(document.querySelectorAll(".aig-study__btn"), function (b) { b.onclick = toggle; });
    }
    render();
  }).catch(function () {});
})();
