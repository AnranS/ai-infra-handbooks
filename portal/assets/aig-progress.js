// 全站学习进度的汇总、导出与导入。进度散在几处 localStorage：
//   aig-roadmap（路线图与各章学习条的"已学"）、aig-plan（冲刺计划的开始日期与打卡）、
//   aig-practice:*（练习题的状态、提交记录、代码草稿、Playground）、aig-cards（学习卡的复习安排）。
// 导出成一个 JSON 文件；导入时按类型合并（已学、打卡取并集，练习题取更好的状态，草稿以本机为准），不会覆盖掉本机的进度。
(function () {
  var SKIP = { "aig-theme": 1 };
  var RANK = { "": 0, tried: 1, solved: 2 };

  function parse(v, d) { try { var x = JSON.parse(v); return x == null ? d : x; } catch (e) { return d; } }
  function keys() {
    var out = [];
    try { for (var i = 0; i < localStorage.length; i++) { var k = localStorage.key(i); if (k && k.indexOf("aig-") === 0 && !SKIP[k]) out.push(k); } } catch (e) {}
    return out;
  }
  function dump() { var o = {}; keys().forEach(function (k) { o[k] = localStorage.getItem(k); }); return o; }

  function summary() {
    var r = parse(localStorage.getItem("aig-roadmap"), {}), p = parse(localStorage.getItem("aig-plan"), {});
    var st = parse(localStorage.getItem("aig-practice:status"), {}), c = parse(localStorage.getItem("aig-cards"), {});
    var checks = p.checks || {}, cs = c.s || {};
    return {
      chapters: (r.done || []).length,
      checks: Object.keys(checks).filter(function (k) { return checks[k]; }).length,
      solved: Object.keys(st).filter(function (k) { return st[k] === "solved"; }).length,
      tried: Object.keys(st).filter(function (k) { return st[k]; }).length,
      cards: Object.keys(cs).length
    };
  }
  function summaryText() {
    var s = summary();
    if ((document.documentElement.lang || "").indexOf("en") === 0)
      return s.chapters + " roadmap chapters done · " + s.checks + " plan checks ticked · " + s.solved + " exercises passed (" + s.tried + " tried) · " + s.cards + " flashcards reviewed";
    return "路线图已学 " + s.chapters + " 章 · 冲刺计划打卡 " + s.checks + " 项 · 练习题通过 " + s.solved + " 题（做过 " + s.tried + " 题）· 学习卡复习过 " + s.cards + " 张";
  }

  function merge(k, mine, theirs) {
    if (mine == null) return theirs;
    var a = parse(mine, null), b = parse(theirs, null);
    if (a == null || b == null) return mine;
    if (k === "aig-roadmap") {
      var done = (a.done || []).slice();
      (b.done || []).forEach(function (x) { if (done.indexOf(x) < 0) done.push(x); });
      return JSON.stringify({ done: done, dir: a.dir && a.dir !== "all" ? a.dir : (b.dir || "all") });
    }
    if (k === "aig-plan") {
      var checks = Object.assign({}, b.checks || {}, a.checks || {});
      Object.keys(b.checks || {}).forEach(function (x) { if (b.checks[x]) checks[x] = true; });
      return JSON.stringify(Object.assign({}, b, a, { start: a.start || b.start || "", checks: checks }));
    }
    if (k === "aig-practice:status") {
      var st = Object.assign({}, a);
      Object.keys(b).forEach(function (x) { if ((RANK[b[x]] || 0) > (RANK[st[x]] || 0)) st[x] = b[x]; });
      return JSON.stringify(st);
    }
    if (k.indexOf("aig-practice:subs:") === 0 && Array.isArray(a) && Array.isArray(b)) {
      var seen = {}, all = a.concat(b).filter(function (s) { var key = s.t + ":" + s.status; if (seen[key]) return false; seen[key] = 1; return true; });
      return JSON.stringify(all.sort(function (x, y) { return y.t - x.t; }).slice(0, 20));
    }
    if (k.indexOf("aig-practice:reveal:") === 0) return JSON.stringify(!!(a || b));
    if (k === "aig-cards") {
      var s = Object.assign({}, b.s || {}, a.s || {});
      Object.keys(b.s || {}).forEach(function (x) { if (!a.s || !a.s[x] || (b.s[x].last || 0) > (a.s[x].last || 0)) s[x] = b.s[x]; });
      return JSON.stringify(Object.assign({}, b, a, { s: s }));
    }
    return mine;                                               // 代码草稿、筛选条件等：以本机为准
  }

  function exportFile() {
    var payload = { app: "ai-infra-handbooks", version: 1, exported: new Date().toISOString(), data: dump() };
    var a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([JSON.stringify(payload, null, 1)], { type: "application/json" }));
    a.download = "ai-infra-progress-" + new Date().toISOString().slice(0, 10) + ".json";
    document.body.appendChild(a);
    a.click();
    setTimeout(function () { URL.revokeObjectURL(a.href); a.remove(); }, 1000);
  }
  function importObject(obj) {
    if (!obj || obj.app !== "ai-infra-handbooks" || typeof obj.data !== "object") throw new Error((document.documentElement.lang || "").indexOf("en") === 0 ? "not a progress file exported from this site" : "不是本站导出的进度文件");
    var n = 0;
    Object.keys(obj.data).forEach(function (k) {
      if (k.indexOf("aig-") !== 0 || SKIP[k]) return;
      var mine = localStorage.getItem(k), merged = merge(k, mine, obj.data[k]);
      if (merged != null && merged !== mine) { localStorage.setItem(k, merged); n++; }
    });
    return n;
  }
  function pickAndImport(done) {
    var input = document.createElement("input");
    input.type = "file";
    input.accept = "application/json,.json";
    input.onchange = function () {
      var f = input.files && input.files[0];
      if (!f) return;
      f.text().then(function (t) {
        var n = importObject(JSON.parse(t));
        alert(((document.documentElement.lang || "").indexOf("en") === 0 ? "Merged " + n + " progress items.\n" : "已合并 " + n + " 项进度。\n") + summaryText());
        if (done) done();
      }).catch(function (e) { alert(((document.documentElement.lang || "").indexOf("en") === 0 ? "Import failed: " : "导入失败：") + e.message); });
    };
    input.click();
  }

  window.AIGProgress = { summary: summary, summaryText: summaryText, merge: merge, dump: dump,
                         exportFile: exportFile, importObject: importObject, pickAndImport: pickAndImport };
})();
