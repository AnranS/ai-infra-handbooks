// 判题 worker：在 Web Worker 里加载 Pyodide（WebAssembly 版 CPython），执行用户代码和测试。
// 放在 worker 里，死循环不会卡住页面：主线程超时后直接 terminate() 再重建一个。
/* global loadPyodide */
const PYODIDE = "https://cdn.jsdelivr.net/pyodide/v0.28.3/full/";
const RUNTIME = ["judge_runner.py", "checker.py", "gpusim.py", "minitl.py", "tritonkit.py"];
let py = null;
let booting = null;

async function boot(base, version) {
  importScripts(PYODIDE + "pyodide.js");
  py = await loadPyodide({ indexURL: PYODIDE });
  py.FS.mkdirTree("/home/pyodide/runtime");
  await Promise.all(RUNTIME.map(async (name) => {
    const r = await fetch(`${base}runtime/${name}?v=${version}`);
    if (!r.ok) throw new Error(`加载 ${name} 失败：${r.status}`);
    py.FS.writeFile(`/home/pyodide/runtime/${name}`, await r.text());
  }));
  py.runPython("import sys; sys.path.insert(0, '/home/pyodide/runtime'); import judge_runner");
}

self.onmessage = async (e) => {
  const msg = e.data;
  try {
    if (msg.type === "init") {
      booting = booting || boot(msg.base, msg.version);
      await booting;
      self.postMessage({ id: msg.id, type: "ready", version: py.version });
      return;
    }
    if (msg.type === "run") {
      await booting;
      const need = (msg.requires || []).filter((r) => r === "numpy" || r === "triton" || r === "gpusim");
      if (need.length) {
        self.postMessage({ id: msg.id, type: "progress", text: "加载 numpy…" });
        await py.loadPackage("numpy");
      }
      if ((msg.requires || []).includes("triton")) {
        py.runPython("import tritonkit; tritonkit.install(prefer_real=False)");
      }
      py.globals.set("USER_CODE", msg.code);
      py.globals.set("TEST_CODE", msg.tests);
      py.globals.set("MODE", msg.mode);
      self.postMessage({ id: msg.id, type: "progress", text: "运行中…" });
      const out = await py.runPythonAsync(
        "import json, judge_runner\njson.dumps(await judge_runner.run(USER_CODE, TEST_CODE, MODE), ensure_ascii=False)");
      self.postMessage({ id: msg.id, type: "result", result: JSON.parse(out) });
    }
  } catch (err) {
    self.postMessage({ id: msg.id, type: "error", error: String(err && err.message || err) });
  }
};
