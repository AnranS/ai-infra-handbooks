/* 顶栏下拉菜单（<details>）：点别处、按 Esc 或点了菜单里的链接就收起 */
(function () {
  function closeAll(except) {
    document.querySelectorAll("header details.menu[open]").forEach(function (d) { if (d !== except) d.open = false; });
  }
  document.addEventListener("click", function (e) {
    var menu = e.target.closest && e.target.closest("header details.menu");
    closeAll(menu);
    if (menu && e.target.closest(".menu-item")) menu.open = false;
  });
  document.addEventListener("keydown", function (e) { if (e.key === "Escape") closeAll(null); });
})();
