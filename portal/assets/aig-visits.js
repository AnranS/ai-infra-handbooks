/* 页脚的访问计数：向不蒜子（busuanzi.ibruce.info）发一次 JSONP 请求，拿到本站访客数、访问量和本页访问量后
   填进 busuanzi_value_* 并显示 busuanzi_container_*。服务偶尔 502 或长时间不响应，所以自己实现加载：
   10 秒没回或报错就重发，间隔 2、4、8、16、30… 秒，最多试 9 次（约两分半）；一直失败就什么都不显示。请求用 script 标签发出，Referer 带页面地址，不蒜子按
   域名统计本站、按页面地址统计本页；不设 cookie。只需要计数、不显示的页面也可以只引这个脚本。 */
(function () {
  var KEYS = ["site_uv", "site_pv", "page_pv"];
  var tries = 0;
  function fill(data) {
    KEYS.forEach(function (k) {
      var box = document.getElementById("busuanzi_container_" + k), val = document.getElementById("busuanzi_value_" + k);
      if (!box || !val || data[k] == null) return;
      val.textContent = Number(data[k]).toLocaleString("en-US");
      box.style.display = "inline";
    });
  }
  function load() {
    var name = "BusuanziCallback_" + Math.floor(Math.random() * 1e12), done = false;
    var tag = document.createElement("script");
    var timer = setTimeout(function () { if (!done) { done = true; cleanup(); retry(); } }, 10000);
    function cleanup() { clearTimeout(timer); delete window[name]; if (tag.parentNode) tag.parentNode.removeChild(tag); }
    function retry() { if (++tries < 9) setTimeout(load, Math.min(30000, 1000 * Math.pow(2, tries))); }   // 2、4、8、16、30、30、30、30 秒后再试，约两分半
    window[name] = function (data) { if (done) return; done = true; cleanup(); if (data && typeof data === "object") fill(data); };
    tag.onerror = function () { if (done) return; done = true; cleanup(); retry(); };
    tag.async = true;
    tag.referrerPolicy = "no-referrer-when-downgrade";
    tag.src = "https://busuanzi.ibruce.info/busuanzi?jsonpCallback=" + name;
    document.head.appendChild(tag);
  }
  // 等页面 load 之后再发：动态插入的 script 在 load 之前会把 load 事件拖到它返回或超时为止
  if (document.readyState === "complete") load(); else window.addEventListener("load", load);
})();
