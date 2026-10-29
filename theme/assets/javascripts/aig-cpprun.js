// C++ 例子的"运行"按钮：把代码交给 Compiler Explorer（godbolt.org）编译运行，结果显示在代码块下面。
// 构建钩子 hooks/cpp_run.py 在每个能单独运行的例子后面放了 <div class="aig-cpprun" data-flags=… data-sanitize=… data-expect=…>，
// 编译参数和 cpp/tools/check_code.py 构建时用的一样。同一页里的 .hpp / .h 代码块会作为头文件一起发过去。
(function () {
  "use strict";

  var API = "https://godbolt.org/api/compiler/";
  var COMPILER = "g152", COMPILER_NAME = "x86-64 gcc 15.2";       // gcc 13、14 在 godbolt 的机器上跑 TSan 会随机崩

  function el(tag, cls, text) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
  }

  function lines(arr) {
    return (arr || []).map(function (x) { return x.text; }).join("\n").replace(/\x1b\[[0-9;]*[A-Za-z]/g, "").replace(/\s+$/, "");
  }

  function filenameOf(block) {
    var f = block && block.querySelector(".filename");
    return f ? f.textContent.trim() : "";
  }

  // 缩进在提示框里的占位会被 Markdown 包进 <p>，浏览器解析时 <div> 把它撑开，前后各留下一个空的 <p>，跳过它们
  function isEmptyP(e) { return e && e.tagName === "P" && !e.children.length && !e.textContent.trim(); }

  function sibling(e, dir) {
    do { e = e[dir]; } while (isEmptyP(e));
    return e && e.querySelector && e.querySelector("pre code") ? e : null;
  }

  function headersOnPage() {
    var out = [];
    document.querySelectorAll(".highlight").forEach(function (b) {
      var name = filenameOf(b);
      if (/\.(hpp|h)$/.test(name)) out.push({ filename: name, contents: b.querySelector("pre code").textContent });
    });
    return out;
  }

  // 在 godbolt 网页里打开：它的网页不支持一起带头文件，就把 #include "x.hpp" 换成头文件的内容
  function inlineHeaders(code, headers) {
    headers.forEach(function (h) {
      code = code.replace(new RegExp('^#include\\s+"' + h.filename.replace(/[.*+?^${}()|[\]\\]/g, "\\$&") + '"\\s*$', "m"),
        "// ---- " + h.filename + " ----\n" + h.contents.replace(/^#pragma once\s*$/m, "") + "// ---- " + h.filename + " 结束 ----");
    });
    return code;
  }

  function godboltLink(code, flags) {
    var state = { sessions: [{ id: 1, language: "c++", source: code,
      compilers: [{ id: COMPILER, options: flags }],
      executors: [{ compiler: { id: COMPILER, libs: [], options: flags }, stdin: "", arguments: "" }] }] };
    var b64 = btoa(unescape(encodeURIComponent(JSON.stringify(state))));
    return "https://godbolt.org/clientstate/" + encodeURIComponent(b64);
  }

  function section(box, title, body, cls) {
    if (!body) return;
    box.appendChild(el("div", "aig-cpprun-h", title));
    box.appendChild(el("pre", "aig-cpprun-pre" + (cls ? " " + cls : ""), body));
  }

  function init(marker) {
    var block = sibling(marker, "previousElementSibling");
    if (!block || marker.dataset.ready) return;
    marker.dataset.ready = "1";
    var pre = block.querySelector("pre");
    var original = block.querySelector("pre code").textContent;
    var flags = marker.dataset.flags, san = marker.dataset.sanitize, expect = marker.dataset.expect;
    var expected = sibling(marker, "nextElementSibling");
    expected = expected && filenameOf(expected) === "输出" ? expected.querySelector("pre code").textContent.replace(/\s+$/, "") : null;

    var bar = el("div", "aig-cpprun-bar");
    var run = el("button", "aig-cpprun-btn aig-cpprun-run", "▶ 运行");
    var edit = el("button", "aig-cpprun-btn", "改代码");
    var open = el("a", "aig-cpprun-btn", "在 Compiler Explorer 打开");
    open.target = "_blank";
    open.rel = "noopener";
    var info = el("span", "aig-cpprun-info",
      "g++ 15.2" + (san === "none" ? "，不开 sanitizer" : "，-fsanitize=" + san) + (expect === "fail" ? "，这个例子故意写了 bug" : expect === "compile-error" ? "，这个例子故意编译不过" : ""));
    [run, edit, open, info].forEach(function (b) { bar.appendChild(b); });
    var out = el("div", "aig-cpprun-out");
    out.hidden = true;
    marker.appendChild(bar);
    marker.appendChild(out);

    var area = null;
    function code() { return area ? area.value : original; }

    edit.addEventListener("click", function () {
      if (!area) {
        area = el("textarea", "aig-cpprun-edit");
        area.value = original;
        area.spellcheck = false;
        area.wrap = "off";
        area.rows = Math.min(40, original.split("\n").length + 1);
        area.addEventListener("keydown", function (e) {             // Tab 插入四个空格，而不是跳到下一个控件
          if (e.key !== "Tab") return;
          e.preventDefault();
          var s = area.selectionStart;
          area.value = area.value.slice(0, s) + "    " + area.value.slice(area.selectionEnd);
          area.selectionStart = area.selectionEnd = s + 4;
        });
        pre.style.display = "none";                                // Material 的样式会盖掉 hidden 属性
        pre.parentNode.insertBefore(area, pre.nextSibling);
        edit.textContent = "还原";
        area.focus();
      } else {
        area.remove();
        area = null;
        pre.style.display = "";
        edit.textContent = "改代码";
      }
    });

    open.href = "https://godbolt.org/";
    function refreshLink() { open.href = godboltLink(inlineHeaders(code(), headersOnPage()), flags); }
    ["mouseenter", "focus", "click"].forEach(function (ev) { open.addEventListener(ev, refreshLink); });   // 改过代码也带上最新的

    run.addEventListener("click", function () {
      var src = code();
      var userFlags = flags + (/undefined/.test(san) ? " -fno-sanitize-recover=undefined" : "");   // 同书里的 UBSAN_OPTIONS=halt_on_error=1
      run.disabled = true;
      run.textContent = "编译运行中…";
      out.hidden = false;
      out.innerHTML = "";
      out.appendChild(el("div", "aig-cpprun-note", "正在交给 godbolt.org 编译运行，一般要一两秒……"));
      var t0 = Date.now();
      fetch(API + COMPILER + "/compile", {
        method: "POST",
        headers: { "Content-Type": "application/json", "Accept": "application/json" },
        body: JSON.stringify({
          source: src, lang: "c++", allowStoreCodeDebug: false, files: headersOnPage(),
          options: { userArguments: userFlags, executeParameters: { args: [], stdin: "" },
                     compilerOptions: { executorRequest: true }, filters: { execute: true }, tools: [], libraries: [] }
        })
      }).then(function (r) {
        if (r.status === 429) throw new Error("godbolt.org 说请求太频繁了，过一会儿再试");
        if (!r.ok) throw new Error("godbolt.org 返回了 HTTP " + r.status);
        return r.json();
      }).then(function (d) {
        out.innerHTML = "";
        var build = d.buildResult || {};
        var buildMsg = lines(build.stderr) || lines(build.stdout);
        if (!d.didExecute) {
          section(out, build.code ? "编译失败" : "没有运行", buildMsg || lines(d.stderr) || "（没有输出）", "aig-cpprun-err");
        } else {
          if (buildMsg) section(out, "编译器的提示", buildMsg, "aig-cpprun-warn");
          var stdout = lines(d.stdout), stderr = lines(d.stderr);
          section(out, "程序输出", stdout || "（没有输出）");
          section(out, /Sanitizer|runtime error/.test(stderr) ? "sanitizer 的报告" : "标准错误输出", stderr, "aig-cpprun-err");
          if (d.timedOut || /SIGKILL/.test(stderr))
            out.appendChild(el("div", "aig-cpprun-note aig-cpprun-err", "超出了 godbolt 给每次运行的时间或内存限制，被它停掉了。性能测试类的例子常会这样，看页面上的输出就好"));
          if (expected != null && !area) {
            var same = stdout.split("\n").map(function (s) { return s.replace(/\s+$/, ""); }).join("\n") ===
                       expected.split("\n").map(function (s) { return s.replace(/\s+$/, ""); }).join("\n");
            out.appendChild(el("div", "aig-cpprun-note " + (same ? "aig-cpprun-ok" : ""),
              same ? "✓ 和书里的输出一致" : "和书里的输出不完全一样：书里是用 Debian 的 g++ 12 跑的，编译器版本或运行环境不同时，个别数字和顺序会有差别"));
          }
        }
        out.appendChild(el("div", "aig-cpprun-foot",
          "返回码 " + d.code + (d.execTime != null ? " · 运行 " + d.execTime + " ms" : "") + " · 往返 " + ((Date.now() - t0) / 1000).toFixed(1) +
          " s · 在 godbolt.org 上用 " + COMPILER_NAME + " 编译运行"));
      }).catch(function (e) {
        out.innerHTML = "";
        out.appendChild(el("div", "aig-cpprun-note aig-cpprun-err",
          (e && e.message && /godbolt/.test(e.message) ? e.message : "连不上 godbolt.org（网络不通或服务暂时不可用）") +
          "。页面上的输出是构建网站时实际运行的结果，可以先看它。"));
      }).then(function () {
        run.disabled = false;
        run.textContent = "▶ 运行";
      });
    });
  }

  function start() { document.querySelectorAll(".aig-cpprun").forEach(init); }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start);
  else start();
})();
