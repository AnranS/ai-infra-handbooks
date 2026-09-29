// 冲刺计划页的渲染。数据在 data.js（window.AIG_PLAN）；打卡和开始日期保存在本地浏览器（localStorage），练习题的完成状态读自练习题页。
(function () {
  "use strict";

  var P = window.AIG_PLAN;
  var BOOK = P.BOOK, U = P.U, ABILITIES = P.ABILITIES, DIRS = P.DIRS, LEVELS = P.LEVELS, TOOLBOX = P.TOOLBOX;
  var REFS = P.REFS, EDGE = P.EDGE, PHASES = P.PHASES, WEEKS = P.WEEKS, MILESTONES = P.MILESTONES, PROJECTS = P.PROJECTS;
  var ALGO = P.ALGO, HANDWRITE = P.HANDWRITE, DESIGN = P.DESIGN, PAPERS = P.PAPERS, TIMELINE = P.TIMELINE, RESUME = P.RESUME;
  var RHYTHM = P.RHYTHM, BUILD = P.BUILD;

  // ---------------------------------------------------------------- 工具
  function $(s) { return document.querySelector(s); }
  function esc(s) { return String(s).replace(/[&<>"]/g, function (c) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]; }); }
  function href(l) {
    if (l[0] === "practice") return "../practice/" + (!l[1] ? "" : l[1][0] === "?" ? "#/" + l[1] : "#/p/" + l[1]);
    return "../" + l[0] + "/" + (l[1] ? l[1] + "/" : "");
  }
  function linkHTML(l) {
    if (l[0] === "todo") return '<span>' + esc(l[2]) + '<span class="tag todo">建设中</span></span>';
    if (l[0] === "ext") return '<a href="' + l[1] + '" target="_blank" rel="noopener">' + esc(l[2]) + '</a><span class="tag ext">外部</span>';
    var book = BOOK[l[0]] ? BOOK[l[0]] + " · " : "";
    return '<a href="' + href(l) + '">' + esc(book + l[2]) + "</a>";
  }
  var state = { start: "", checks: {}, open: {} };
  try { var saved = JSON.parse(localStorage.getItem("aig-plan") || "null"); if (saved) state = Object.assign(state, saved); } catch (e) { /* ignore */ }
  function save() { try { localStorage.setItem("aig-plan", JSON.stringify(state)); } catch (e) { /* ignore */ } }
  var practiceStatus = {};
  try { practiceStatus = JSON.parse(localStorage.getItem("aig-practice:status") || "{}") || {}; } catch (e) { /* ignore */ }

  function cards(el, list) {
    el.innerHTML = list.map(function (a) {
      return '<div class="card ' + a.c + '"><h3><i></i>' + esc(a.t) + "</h3>" + (a.why ? '<p class="why">' + esc(a.why) + "</p>" : "") +
        "<ul>" + a.pass.map(function (p) { return "<li>" + esc(p) + "</li>"; }).join("") + "</ul>" +
        (a.verify ? '<p class="verify"><b>怎么验证</b>' + esc(a.verify) + "</p>" : "") +
        (a.links ? '<div class="links">' + a.links.map(linkHTML).join(" · ") + "</div>" : "") + "</div>";
    }).join("");
  }

  // ---------------------------------------------------------------- 渲染
  var totalChecks = WEEKS.reduce(function (n, w) { return n + w.check.length; }, 0);
  var practiceIds = {};
  WEEKS.forEach(function (w) { w.practice.forEach(function (p) { practiceIds[p] = 1; }); });
  $("#stats").innerHTML = [["17", "周"], ["4", "个阶段"], ["3 + 1", "个作品"], ["约 200", "道算法题"], [Object.keys(practiceIds).length, "道配套练习题"], ["10", "道系统设计"]]
    .map(function (s) { return '<div class="stat"><b>' + s[0] + "</b><span>" + s[1] + "</span></div>"; }).join("");
  cards($("#abilities"), ABILITIES);
  cards($("#edgeCards"), EDGE);
  cards($("#projCards"), PROJECTS);
  cards($("#buildCards"), BUILD);
  $("#msCards").innerHTML = MILESTONES.map(function (m) {
    return '<div class="card ' + m.c + '"><div class="when">' + esc(m.when) + "</div><h3>" + esc(m.t) + "</h3><ul>" +
      m.items.map(function (i) { return "<li>" + esc(i) + "</li>"; }).join("") + "</ul></div>";
  }).join("");
  $("#cmp").innerHTML = "<tr><th></th><th>推理框架</th><th>推理优化</th><th>推理平台</th></tr>" + DIRS.map(function (r) {
    return "<tr><td>" + esc(r[0]) + "</td><td>" + esc(r[1]) + "</td><td>" + esc(r[2]) + "</td><td>" + esc(r[3]) + "</td></tr>";
  }).join("");
  $("#levels").innerHTML = LEVELS.map(function (l) {
    return '<div class="card ' + l.c + '"><div class="when">' + l.k + "</div><h3>" + esc(l.t) + '</h3><p class="why">' + esc(l.d) + "</p><ul>" +
      l.how.map(function (h) { return "<li>" + esc(h) + "</li>"; }).join("") + "</ul></div>";
  }).join("");
  $("#toolbox").innerHTML = "<tr><th>验证方式</th><th>验证什么</th><th>工具与做法</th><th>什么时候</th></tr>" + TOOLBOX.map(function (r) {
    return "<tr><td>" + r.map(esc).join("</td><td>") + "</td></tr>";
  }).join("");
  $("#refs").innerHTML = "<tr><th>课程 / 教程</th><th>用在</th><th>怎么用</th><th>自带验证</th></tr>" + REFS.map(function (r) {
    return '<tr><td><a href="' + r[1] + '" target="_blank" rel="noopener">' + esc(r[0]) + "</a></td><td>" + esc(r[2]) + "</td><td>" + esc(r[3]) + "</td><td>" + esc(r[4] || "—") + "</td></tr>";
  }).join("");
  function ol(el, list) { el.innerHTML = list.map(function (x) { return "<li>" + esc(x) + "</li>"; }).join(""); }
  ol($("#algo"), ALGO);
  ol($("#design"), DESIGN);
  ol($("#papers"), PAPERS);
  ol($("#timeline"), TIMELINE);
  ol($("#resume"), RESUME);
  $("#handwrite").innerHTML = HANDWRITE.map(function (h) {
    return "<li>" + (h[1] ? '<a href="../practice/#/p/' + h[1] + '" style="color:var(--blue)">' + esc(h[0]) + "</a>" : esc(h[0]) + '<span class="tag todo">建设中</span>') + "</li>";
  }).join("");
  $("#rhythmCards").innerHTML = RHYTHM.map(function (r) {
    return '<div class="card ' + r[0] + '"><b>' + r[1] + "</b><span>" + esc(r[2]) + "</span></div>";
  }).join("");

  // 阶段条
  var ph = "";
  PHASES.forEach(function (p) {
    var nm = p.name.split(" · ");
    ph += '<div class="ph ' + p.c + '" style="grid-column:' + p.weeks[0] + " / " + (p.weeks[1] + 1) + '"><b><span class="pn">' + esc(nm[0]) + "<i> · </i></span>" + esc(nm[1]) +
      "</b><small>第 " + p.weeks[0] + "～" + p.weeks[1] + " 周</small></div>";
  });
  for (var i = 1; i <= 17; i++) ph += '<div class="wk" data-w="' + i + '">' + i + "</div>";
  $("#phases").innerHTML = ph;

  function phaseOf(n) { return PHASES.filter(function (p) { return n >= p.weeks[0] && n <= p.weeks[1]; })[0]; }
  var titles = {};
  function renderWeeks(names) {
    $("#weekList").innerHTML = WEEKS.map(function (w, idx) {
      var n = idx + 1, p = phaseOf(n);
      var practice = w.practice.length ? '<h4>练习题</h4><div class="chips">' + w.practice.map(function (id) {
        var meta = names[id];
        var label = meta ? meta.number + ". " + meta.title : id;
        return '<a class="chip' + (practiceStatus[id] === "solved" ? " solved" : "") + '" href="../practice/#/p/' + id + '">' + esc(label) + "</a>";
      }).join("") + "</div>" : "";
      var todo = (w.todo || []).map(function (t) { return "<li>" + esc(t) + '<span class="tag todo">建设中</span></li>'; }).join("");
      return '<div class="week ' + p.c + '" id="w' + n + '" data-w="' + n + '"><div class="wh"><div class="no">第 ' + n + ' 周<small class="date"></small></div>' +
        '<div class="tt"><b>' + esc(w.t) + "</b><span>" + esc(w.g) + '</span></div><div class="pc"></div><div class="arr">›</div></div>' +
        '<div class="wb"><h4>主线学习</h4><ul>' + w.learn.map(function (l) { return "<li>" + linkHTML(l) + "</li>"; }).join("") + todo + "</ul>" +
        practice + "<h4>算法题</h4><ul><li>" + esc(w.algo) + "</li></ul>" +
        "<h4>产出</h4><ul>" + w.out.map(function (o) { return "<li>" + esc(o) + "</li>"; }).join("") + "</ul>" +
        '<h4>验收清单</h4><ul class="checks">' + w.check.map(function (c, j) {
          var id = "w" + n + "-" + j;
          var m = /^L([1-4])(（可选）)? ?/.exec(c), badge = "";
          if (m) { var lv = LEVELS[+m[1] - 1]; badge = '<span class="lv ' + lv.c + '">' + lv.t + (m[2] ? " · 可选" : "") + "</span>"; c = c.slice(m[0].length); }
          else if (/^里程碑/.test(c)) badge = '<span class="lv c-pink">里程碑</span>';
          return '<li><input type="checkbox" id="' + id + '"' + (state.checks[id] ? " checked" : "") + '><label for="' + id + '">' + badge + esc(c) + "</label></li>";
        }).join("") + "</ul></div></div>";
    }).join("");
    document.querySelectorAll(".week .wh").forEach(function (h) {
      h.addEventListener("click", function () {
        var wk = h.parentElement;
        wk.classList.toggle("open");
        state.open[wk.dataset.w] = wk.classList.contains("open");
        save();
      });
    });
    document.querySelectorAll(".checks input").forEach(function (cb) {
      cb.addEventListener("change", function () { state.checks[cb.id] = cb.checked; save(); refresh(); });
    });
    refresh();
  }

  function currentWeek() {
    if (!state.start) return 0;
    var d0 = new Date(state.start + "T00:00:00");
    var days = Math.floor((Date.now() - d0.getTime()) / 86400000);
    return days < 0 ? 0 : Math.floor(days / 7) + 1;
  }
  function refresh() {
    var done = 0;
    WEEKS.forEach(function (w, idx) {
      var n = idx + 1, k = 0;
      w.check.forEach(function (_, j) { if (state.checks["w" + n + "-" + j]) k++; });
      done += k;
      var el = document.getElementById("w" + n);
      if (!el) return;
      el.querySelector(".pc").textContent = k + " / " + w.check.length;
      var cell = document.querySelector('.wk[data-w="' + n + '"]');
      cell.classList.toggle("done", k === w.check.length);
      var cw = currentWeek();
      el.classList.toggle("now", cw === n);
      cell.classList.toggle("now", cw === n);
      if (state.open[n] === undefined && cw === n) el.classList.add("open");
      else if (state.open[n]) el.classList.add("open");
      var date = el.querySelector(".date");
      if (state.start) {
        var d = new Date(state.start + "T00:00:00");
        d.setDate(d.getDate() + (n - 1) * 7);
        date.textContent = (d.getMonth() + 1) + "/" + d.getDate();
      } else date.textContent = "";
    });
    $("#bar").style.width = (done / totalChecks * 100).toFixed(1) + "%";
    $("#progText").textContent = done + " / " + totalChecks;
    var cw = currentWeek();
    $("#nowText").textContent = !state.start ? "（设置后会标出当前周）" : cw === 0 ? "（还没开始）" : cw > 17 ? "（计划已结束）" : "· 现在是第 " + cw + " 周";
  }

  $("#start").value = state.start || "";
  $("#start").addEventListener("change", function (e) { state.start = e.target.value; state.open = {}; save(); renderWeeks(titles); });
  $("#reset").addEventListener("click", function () {
    if (confirm("清空所有验收清单的打卡记录？")) { state.checks = {}; save(); renderWeeks(titles); }
  });
  $("#expand").addEventListener("click", function () {
    var all = document.querySelectorAll(".week");
    var open = Array.prototype.some.call(all, function (w) { return !w.classList.contains("open"); });
    all.forEach(function (w) { w.classList.toggle("open", open); state.open[w.dataset.w] = open; });
    $("#expand").textContent = open ? "全部收起" : "全部展开";
    save();
  });
  $("#phases").addEventListener("click", function (e) {
    var c = e.target.closest(".wk");
    if (!c) return;
    var w = document.getElementById("w" + c.dataset.w);
    w.classList.add("open");
    state.open[c.dataset.w] = true;
    save();
    w.scrollIntoView({ behavior: "smooth", block: "start" });
  });

  renderWeeks({});
  fetch("../practice/data/index.json").then(function (r) { return r.json(); }).then(function (idx) {
    idx.problems.forEach(function (p) { titles[p.slug] = { number: p.number, title: p.title }; });
    renderWeeks(titles);
  }).catch(function () { /* 离线时只显示 id */ });

  document.getElementById("theme").addEventListener("click", function () {
    var root = document.documentElement;
    var dark = root.dataset.theme ? root.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
    root.dataset.theme = dark ? "light" : "dark";
    try { localStorage.setItem("aig-theme", root.dataset.theme); } catch (e) { /* ignore */ }
  });
})();
