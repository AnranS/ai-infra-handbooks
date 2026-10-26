/* 专注阅读：顶栏右侧的按钮隐藏顶栏和左侧目录；收起后左下角浮出一个按钮，点它恢复。
   选择存在 localStorage 的 aig-focus 里，所有手册共用、换页也保持；首屏前由 theme/main.html 的内联脚本
   写到 <html data-aig-focus> 上，不会先闪出顶栏再收起。样式见 aig.css 的「专注阅读」一节（只在宽屏上生效）。 */
(function () {
  var EN = (document.documentElement.lang || "").indexOf("en") === 0;
  var SHOW = EN ? "Show the header and navigation" : "显示顶栏和目录";
  // 四角向内：恢复
  var SHRINK = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M9 4v5H4M15 4v5h5M9 20v-5H4M15 20v-5h5"/></svg>';

  function set(v) {
    if (v) document.documentElement.setAttribute("data-aig-focus", "1");
    else document.documentElement.removeAttribute("data-aig-focus");
    try { localStorage.setItem("aig-focus", v ? "1" : "0"); } catch (e) {}
    window.dispatchEvent(new Event("resize"));      // 让 Material 按新的版面重算侧栏高度
  }

  function mount() {
    var toggle = document.querySelector(".aig-focus-toggle");
    if (!toggle || document.querySelector(".aig-focus-btn")) return;   // 门户页没有这个按钮
    toggle.addEventListener("click", function () { set(true); });
    var b = document.createElement("button");
    b.type = "button";
    b.className = "aig-focus-btn";
    b.title = SHOW;
    b.setAttribute("aria-label", SHOW);
    b.innerHTML = SHRINK;
    b.addEventListener("click", function () { set(false); });
    document.body.appendChild(b);
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", mount);
  else mount();
})();
