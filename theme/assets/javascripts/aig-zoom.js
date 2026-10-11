/* 点图放大：mini-sglang 的内联架构图（figure.dg svg）、各书的配图（figure.aig-fig svg，hooks/figures.py 内联的）和章首漫画（img.aig-comic）。
   点图进入全屏查看层：滚轮 / 触控板双指缩放，拖拽平移，双击在"适应窗口"和 2 倍之间切换，
   键盘 + - 0，Esc 或点空白处关闭。SVG 是矢量，放多大都清晰；颜色用的是页面的 CSS 变量，跟随明暗主题。 */
(function () {
  "use strict";
  var ZH = (document.documentElement.lang || "zh").toLowerCase().indexOf("zh") === 0;
  var T = ZH
    ? { open: "点击放大", close: "关闭", fit: "适应窗口", hint: "滚轮缩放 · 拖拽平移 · 双击切换 · Esc 关闭" }
    : { open: "Click to zoom", close: "Close", fit: "Fit", hint: "scroll to zoom · drag to pan · double-click to toggle · Esc to close" };
  var SEL = "figure.dg svg, figure.aig-fig svg, img.aig-svg, img.aig-comic";
  var overlay, stage, content, pct;
  var scale = 1, tx = 0, ty = 0, fitScale = 1, natW = 1, natH = 1, maxScale = 12;

  function build() {
    if (overlay) return;
    overlay = document.createElement("div");
    overlay.className = "aig-zoom";
    overlay.setAttribute("role", "dialog");
    overlay.setAttribute("aria-modal", "true");
    overlay.innerHTML =
      '<div class="aig-zoom-bar"><span class="aig-zoom-cap"></span><span class="aig-zoom-hint">' + T.hint + "</span>" +
      '<span class="aig-zoom-pct">100%</span>' +
      '<button type="button" class="aig-zoom-fit">' + T.fit + "</button>" +
      '<button type="button" class="aig-zoom-close" aria-label="' + T.close + '">✕</button></div>' +
      '<div class="aig-zoom-stage"><div class="aig-zoom-content"></div></div>';
    document.body.appendChild(overlay);
    stage = overlay.querySelector(".aig-zoom-stage");
    content = overlay.querySelector(".aig-zoom-content");
    pct = overlay.querySelector(".aig-zoom-pct");
    overlay.querySelector(".aig-zoom-close").addEventListener("click", close);
    overlay.querySelector(".aig-zoom-fit").addEventListener("click", fit);
    stage.addEventListener("wheel", onWheel, { passive: false });
    stage.addEventListener("pointerdown", onDown);
    stage.addEventListener("pointermove", onMove);
    stage.addEventListener("pointerup", onUp);
    stage.addEventListener("pointercancel", onUp);
    stage.addEventListener("dblclick", onDouble);
    window.addEventListener("resize", function () { if (overlay.classList.contains("on")) fit(); });
  }

  function apply() {
    content.style.transform = "translate(" + tx + "px," + ty + "px) scale(" + scale + ")";
    pct.textContent = Math.round(scale / fitScale * 100) + "%";
  }

  function fit() {
    var r = stage.getBoundingClientRect();
    fitScale = Math.min((r.width - 32) / natW, (r.height - 32) / natH);
    scale = fitScale;
    tx = (r.width - natW * scale) / 2;
    ty = (r.height - natH * scale) / 2;
    apply();
  }

  function zoomAt(factor, cx, cy) {            // 以 (cx, cy)（相对 stage）为中心缩放
    var next = Math.max(fitScale * 0.5, Math.min(fitScale * maxScale, scale * factor));
    factor = next / scale;
    tx = cx - (cx - tx) * factor;
    ty = cy - (cy - ty) * factor;
    scale = next;
    apply();
  }

  function stagePoint(e) {
    var r = stage.getBoundingClientRect();
    return { x: e.clientX - r.left, y: e.clientY - r.top };
  }

  function onWheel(e) {
    e.preventDefault();
    var p = stagePoint(e);
    var k = e.ctrlKey ? 0.01 : 0.0015;          // 触控板双指捏合发来的是 ctrl + wheel，步子更细
    zoomAt(Math.exp(-e.deltaY * k), p.x, p.y);
  }

  var pointers = {}, dragStart = null, moved = false, pinchDist = 0;
  function onDown(e) {
    pointers[e.pointerId] = { x: e.clientX, y: e.clientY };
    stage.setPointerCapture(e.pointerId);
    var ids = Object.keys(pointers);
    if (ids.length === 1) { dragStart = { x: e.clientX - tx, y: e.clientY - ty }; moved = false; stage.classList.add("drag"); }
    if (ids.length === 2) pinchDist = dist();
  }
  function dist() {
    var ids = Object.keys(pointers), a = pointers[ids[0]], b = pointers[ids[1]];
    return Math.hypot(a.x - b.x, a.y - b.y);
  }
  function onMove(e) {
    if (!pointers[e.pointerId]) return;
    pointers[e.pointerId] = { x: e.clientX, y: e.clientY };
    var ids = Object.keys(pointers);
    if (ids.length === 2) {                      // 双指捏合
      var d = dist(), a = pointers[ids[0]], b = pointers[ids[1]];
      var r = stage.getBoundingClientRect();
      if (pinchDist) zoomAt(d / pinchDist, (a.x + b.x) / 2 - r.left, (a.y + b.y) / 2 - r.top);
      pinchDist = d;
      return;
    }
    if (dragStart) {
      var nx = e.clientX - dragStart.x, ny = e.clientY - dragStart.y;
      if (Math.abs(nx - tx) + Math.abs(ny - ty) > 2) moved = true;
      tx = nx; ty = ny; apply();
    }
  }
  function onUp(e) {
    delete pointers[e.pointerId];
    if (Object.keys(pointers).length === 0) {
      stage.classList.remove("drag");
      if (!moved && e.target === stage) close();   // 点空白处（没有拖动）关闭
      dragStart = null;
    }
    pinchDist = 0;
  }
  function onDouble(e) {
    var p = stagePoint(e);
    if (scale > fitScale * 1.05) fit(); else zoomAt(2, p.x, p.y);
  }

  function open(el) {
    build();
    content.innerHTML = "";
    var node;
    if (el.tagName.toLowerCase() === "svg") {
      var vb = el.viewBox && el.viewBox.baseVal, rect = el.getBoundingClientRect();
      natW = vb && vb.width ? vb.width : rect.width || 800;
      natH = vb && vb.height ? vb.height : rect.height || 600;
      node = el.cloneNode(true);
      node.removeAttribute("style");
      node.removeAttribute("id");
      node.setAttribute("width", natW);
      node.setAttribute("height", natH);
      maxScale = 12;
      var host = el.closest("figure.dg, figure.aig-fig");   // 配图的颜色类（.aig-fig .tx 等）按父元素选择，克隆时把父元素的 class 带上
      if (host) {
        var wrap = document.createElement("div");
        wrap.className = host.className;
        wrap.style.width = natW + "px";
        wrap.style.height = natH + "px";
        wrap.appendChild(node);
        content.appendChild(wrap);
      }
    } else {
      node = document.createElement("img");
      node.src = el.currentSrc || el.src;
      node.alt = el.alt || "";
      natW = el.naturalWidth || el.width || 800;
      natH = el.naturalHeight || el.height || 600;
      maxScale = /\.svg(\?|$)/i.test(node.src) ? 12 : 4;   // 位图放太大只会糊
    }
    node.style.width = natW + "px";
    node.style.height = natH + "px";
    if (!node.parentNode) content.appendChild(node);
    var fig = el.closest("figure"), cap = fig && fig.querySelector("figcaption");
    overlay.querySelector(".aig-zoom-cap").textContent = cap ? cap.textContent : (el.alt || "");
    overlay.classList.add("on");
    document.body.classList.add("aig-zoom-open");
    fit();
    overlay.querySelector(".aig-zoom-close").focus();
  }

  function close() {
    if (!overlay) return;
    overlay.classList.remove("on");
    document.body.classList.remove("aig-zoom-open");
    content.innerHTML = "";
  }

  document.addEventListener("keydown", function (e) {
    if (!overlay || !overlay.classList.contains("on")) return;
    var r = stage.getBoundingClientRect();
    if (e.key === "Escape") close();
    else if (e.key === "+" || e.key === "=") zoomAt(1.25, r.width / 2, r.height / 2);
    else if (e.key === "-") zoomAt(0.8, r.width / 2, r.height / 2);
    else if (e.key === "0") fit();
    else return;
    e.preventDefault();
  });

  function init(root) {
    (root || document).querySelectorAll(SEL).forEach(function (el) {
      if (el.dataset.aigZoom) return;
      el.dataset.aigZoom = "1";
      if (el.tagName.toLowerCase() === "img") el.setAttribute("title", T.open);
      el.setAttribute("tabindex", "0");
      el.addEventListener("click", function (e) { e.preventDefault(); open(el); });
      el.addEventListener("keydown", function (e) { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); open(el); } });
    });
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", function () { init(); });
  else init();
  if (window.document$ && window.document$.subscribe) window.document$.subscribe(function () { init(); });
})();
