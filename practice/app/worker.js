// 判题 worker：在 Web Worker 里加载 Pyodide（WebAssembly 版 CPython），执行用户代码和测试。
// 放在 worker 里，死循环不会卡住页面：主线程超时后直接 terminate() 再重建一个。
// 编辑器的代码补全也在这里做（runtime/assist.py，用 Pyodide 自带的 Jedi 做静态分析）。
/* global loadPyodide */
const PYODIDE = "https://cdn.jsdelivr.net/pyodide/v0.28.3/full/";
const RUNTIME = ["judge_runner.py", "checker.py", "gpusim.py", "minitl.py", "tritonkit.py", "assist.py", "playground.py"];
let py = null;
let booting = null;
let running = false;          // 判题期间不做补全：两者共用一个解释器，补全要等判题结束
let assist = null;            // assist 模块（PyProxy）
let jediLoading = null;
let numpyLoaded = false;

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
  assist = py.pyimport("assist");
}

const idle = async () => { while (running) await new Promise((r) => setTimeout(r, 50)); };

async function loadNumpy() {
  if (!numpyLoaded) { await py.loadPackage("numpy"); numpyLoaded = true; }
}

// 第一次补全前加载 Jedi（含 parso，约 1.7 MB），再预热一次，把内置函数和常用模块的存根解析好
function loadJedi() {
  jediLoading = jediLoading || (async () => {
    await py.loadPackage("jedi");
    await idle();
    assist.warm();
  })();
  return jediLoading;
}

async function doAssist(msg) {
  if (msg.op === "check") return assist.check(msg.code);        // 语法检查只用 CPython 自己的解析器
  await loadJedi();
  if (/\bnumpy\b/.test(msg.code || "") || (msg.requires || []).includes("numpy")) await loadNumpy();
  await idle();
  if (msg.op === "warm") {
    if ((msg.requires || []).includes("numpy")) assist.complete("import numpy as np\nnp.z", 2, 4);
    return "true";
  }
  if (msg.op === "complete") return assist.complete(msg.code, msg.line, msg.col);
  if (msg.op === "signatures") return assist.signatures(msg.code, msg.line, msg.col);
  if (msg.op === "detail") return assist.detail(msg.token, msg.index);
  return "null";
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
    if (msg.type === "assist") {
      if (!booting) { self.postMessage({ id: msg.id, type: "assist", result: "null" }); return; }
      await booting;
      // 判题进行中就直接返回空结果：补全是锦上添花，不能拖慢判题
      const result = running ? "null" : await doAssist(msg);
      self.postMessage({ id: msg.id, type: "assist", result });
      return;
    }
    if (msg.type === "exec") {                        // Playground：运行一段自由的代码
      await booting;
      running = true;
      try {
        const imports = (name) => new RegExp("^\\s*(import|from)\\s+(" + name + ")\\b", "m").test(msg.code);
        const pkgs = [];
        if (imports("numpy|gpusim|triton|tritonkit|minitl|matplotlib") && !numpyLoaded) pkgs.push("numpy");
        if (imports("matplotlib")) pkgs.push("matplotlib");
        if (pkgs.length) {
          self.postMessage({ id: msg.id, type: "progress", text: `加载 ${pkgs.join("、")}…` });
          await py.loadPackage(pkgs);
          if (pkgs.includes("numpy")) numpyLoaded = true;
        }
        if (imports("triton")) py.runPython("import tritonkit; tritonkit.install(prefer_real=False)");
        py.globals.set("PG_CODE", msg.code);
        self.postMessage({ id: msg.id, type: "progress", text: "运行中…" });
        const out = await py.runPythonAsync("import json, playground\njson.dumps(await playground.run(PG_CODE), ensure_ascii=False)");
        self.postMessage({ id: msg.id, type: "result", result: JSON.parse(out) });
      } finally {
        running = false;
      }
      return;
    }
    if (msg.type === "run") {
      await booting;
      running = true;
      try {
        const need = (msg.requires || []).filter((r) => r === "numpy" || r === "triton" || r === "gpusim");
        if (need.length && !numpyLoaded) {
          self.postMessage({ id: msg.id, type: "progress", text: "加载 numpy…" });
          await loadNumpy();
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
      } finally {
        running = false;
      }
    }
  } catch (err) {
    self.postMessage({ id: msg.id, type: "error", error: String(err && err.message || err) });
  }
};
