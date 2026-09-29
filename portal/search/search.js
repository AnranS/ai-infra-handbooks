// 全站搜索：索引由 tools/search_index.py 在构建时生成（search/index.json），首次输入时才加载。
(function () {
  "use strict";
  var COLORS = ["#007aff", "#af52de", "#ff9500", "#34c759", "#ff2d55", "#5856d6", "#30b0c7"];
  var HINTS = ["PagedAttention", "Radix Cache", "bank conflict", "online softmax", "张量并行", "PD 分离", "投机解码", "量化", "asyncio"];
  var q = document.getElementById("q"), results = document.getElementById("results"), meta = document.getElementById("meta"), filters = document.getElementById("filters");
  var data = null, loading = null, book = -1, timer = 0;

  function esc(s) { return String(s).replace(/[&<>"]/g, function (c) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]; }); }
  function load() {
    if (!loading) {
      loading = fetch("index.json").then(function (r) { return r.json(); }).then(function (d) {
        d.docs.forEach(function (x) { x.push((x[2] + " " + x[3]).toLowerCase(), x[4].toLowerCase()); });
        data = d;
        return d;
      });
    }
    return loading;
  }
  function terms(s) { return s.toLowerCase().split(/\s+/).filter(Boolean); }
  function count(hay, t) { var n = 0, i = hay.indexOf(t); while (i >= 0 && n < 20) { n++; i = hay.indexOf(t, i + t.length); } return n; }
  function highlight(text, ts) {
    var re = new RegExp("(" + ts.map(function (t) { return t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"); }).join("|") + ")", "gi");
    return esc(text).replace(re, "<mark>$1</mark>");
  }
  function snippet(text, lower, ts) {
    var at = -1;
    ts.forEach(function (t) { var i = lower.indexOf(t); if (i >= 0 && (at < 0 || i < at)) at = i; });
    if (at < 0) return text.slice(0, 120);
    var start = Math.max(0, at - 50), end = Math.min(text.length, at + 110);
    return (start > 0 ? "…" : "") + text.slice(start, end) + (end < text.length ? "…" : "");
  }
  function search() {
    var raw = q.value.trim(), ts = terms(raw);
    try { history.replaceState(null, "", raw ? "?q=" + encodeURIComponent(raw) : location.pathname); } catch (e) { /* ignore */ }
    if (!ts.length) { results.innerHTML = ""; meta.textContent = ""; filters.innerHTML = ""; showHints(); return; }
    if (!data) { meta.textContent = "正在加载索引…"; load().then(search); return; }
    var hits = [], per = data.books.map(function () { return 0; });
    data.docs.forEach(function (d) {
      var head = d[5], body = d[6], score = 0;
      for (var i = 0; i < ts.length; i++) {
        var h = count(head, ts[i]), b = count(body, ts[i]);
        if (!h && !b) return;
        score += h * 20 + Math.min(b, 10);
      }
      if (!d[3]) score += 5;                       // 整页（而不是小节）略微靠前
      per[d[0]]++;
      if (book < 0 || d[0] === book) hits.push([score, d]);
    });
    hits.sort(function (a, b) { return b[0] - a[0]; });
    var total = per.reduce(function (a, b) { return a + b; }, 0);
    filters.innerHTML = '<button type="button" data-b="-1"' + (book < 0 ? ' class="on"' : "") + ">全部<small>" + total + "</small></button>" +
      data.books.map(function (name, i) {
        return per[i] ? '<button type="button" data-b="' + i + '"' + (book === i ? ' class="on"' : "") + ">" + esc(name) + "<small>" + per[i] + "</small></button>" : "";
      }).join("");
    meta.textContent = hits.length ? "找到 " + hits.length + " 条" + (hits.length > 60 ? "，显示前 60 条" : "") : "";
    if (!hits.length) { results.innerHTML = '<li class="empty">没有找到同时包含这些关键词的内容，换个说法或减少关键词试试。</li>'; return; }
    results.innerHTML = hits.slice(0, 60).map(function (h) {
      var d = h[1];
      return '<li><a href="../' + d[1] + '"><div class="t"><span class="badge" style="--c:' + COLORS[d[0]] + '">' + esc(data.books[d[0]]) + "</span>" +
        highlight(d[2], ts) + (d[3] ? " <span>› " + highlight(d[3], ts) + "</span>" : "") + '</div><div class="s">' + highlight(snippet(d[4], d[6], ts), ts) + "</div></a></li>";
    }).join("");
  }
  function showHints() {
    results.innerHTML = '<li class="empty">试试这些：<div class="hint">' + HINTS.map(function (h) { return '<a href="?q=' + encodeURIComponent(h) + '">' + esc(h) + "</a>"; }).join("") + "</div></li>";
  }

  q.addEventListener("input", function () { clearTimeout(timer); timer = setTimeout(search, 120); });
  q.addEventListener("focus", load, { once: true });
  q.addEventListener("keydown", function (e) {
    if (e.key === "Enter") { var a = results.querySelector("a[href^='../']"); if (a) location.href = a.href; }
  });
  filters.addEventListener("click", function (e) {
    var b = e.target.closest("button");
    if (b) { book = +b.dataset.b; search(); }
  });
  results.addEventListener("click", function (e) {
    var a = e.target.closest(".hint a");
    if (a) { e.preventDefault(); q.value = a.textContent; search(); }
  });
  var init = new URLSearchParams(location.search).get("q");
  if (init) { q.value = init; search(); } else showHints();
  if (!("ontouchstart" in window)) q.focus();

  document.getElementById("theme").addEventListener("click", function () {
    var root = document.documentElement;
    var dark = root.dataset.theme ? root.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
    root.dataset.theme = dark ? "light" : "dark";
    try { localStorage.setItem("aig-theme", root.dataset.theme); } catch (e) { /* ignore */ }
  });
})();
