/* 正文字号：顶栏里的「小 / 标准 / 大 / 特大」。
   选择存在 localStorage 的 aig-font 里，手册页和门户页共用同一个值（同源）。
   手册页放大的是 .md-typeset 的基准字号（其余标题、代码、表格都按 em 跟着变），
   门户页用 main 上的 zoom，因为那几页的字号是写死的 px。 */
(function () {
  var EN = (document.documentElement.lang || "").indexOf("en") === 0;
  var STEPS = EN ? [["s", "S"], ["m", "M"], ["l", "L"], ["xl", "XL"]] : [["s", "小"], ["m", "标准"], ["l", "大"], ["xl", "特大"]];
  var LBL = EN ? { aria: "Text size", label: "Text", title: "Text size: " } : { aria: "正文字号", label: "字号", title: "正文字号：" };
  function read() {
    try { var v = localStorage.getItem("aig-font"); return /^(s|m|l|xl)$/.test(v) ? v : "m"; } catch (e) { return "m"; }
  }
  function apply(v) {
    document.documentElement.setAttribute("data-aig-font", v);
    try { localStorage.setItem("aig-font", v); } catch (e) {}
    [].forEach.call(document.querySelectorAll(".aig-font__btn"), function (b) {
      b.setAttribute("aria-pressed", b.dataset.font === v ? "true" : "false");
    });
  }
  function build() {
    var wrap = document.createElement("div");
    wrap.className = "aig-font";
    wrap.setAttribute("role", "group");
    wrap.setAttribute("aria-label", LBL.aria);
    wrap.innerHTML = '<span class="aig-font__label">' + LBL.label + '</span>' + STEPS.map(function (s) {
      return '<button type="button" class="aig-font__btn" data-font="' + s[0] + '" title="' + LBL.title + s[1] +
             '" aria-pressed="false">' + s[1] + "</button>";
    }).join("");
    wrap.addEventListener("click", function (e) {
      var b = e.target.closest("button[data-font]");
      if (b) { apply(b.dataset.font); e.stopPropagation(); }
    });
    return wrap;
  }
  function mount() {
    if (document.querySelector(".aig-font")) return;
    // 手册页：放进顶栏「更多」菜单的最后；门户页：放进「更多」菜单里
    var panel = document.querySelector(".aig-menu__panel--right") || document.querySelector(".menu-panel--right");
    if (panel) { panel.appendChild(build()); return; }
    var host = document.querySelector(".aig-tabs") || document.querySelector("header nav");
    if (host) host.appendChild(build());
  }
  apply(read());
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", mount);
  else mount();
  if (window.document$ && window.document$.subscribe) window.document$.subscribe(function () { mount(); apply(read()); });
})();
