"""把 tools/videos/<名字>.js 渲染成带中文配音和字幕的教学视频。

    python render.py lifecycle              # TTS → 逐帧渲染 → 编码，输出 docs/assets/videos/lifecycle.mp4 与封面 .jpg
    python render.py lifecycle --preview 3 12.5   # 只导出几张静帧到 build/<名字>/preview/，调画面用

旁白写在场景文件的 SEGMENTS 里（唯一来源）。每段用 edge-tts 合成语音，量出时长后加上停顿，
作为这一段动画的时长传给页面（window.SEG_DUR），所以画面和配音天然对齐。
依赖：playwright + Chromium、edge-tts、imageio-ffmpeg（自带 ffmpeg，含 libx264 / aac）。
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import subprocess
import sys
import wave
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import imageio_ffmpeg
from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
OUT = HERE.parent.parent / "docs" / "assets" / "videos"
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
CHROMIUM = "/usr/bin/chromium"
FPS = 24
VOICE = "zh-CN-XiaoxiaoNeural"
GAP = 0.55  # 每段语音之后的停顿（秒）
RATE = 24000


def open_page(p, name: str, durations=None):
    browser = p.chromium.launch(executable_path=CHROMIUM, args=["--no-sandbox", "--disable-gpu"])
    page = browser.new_page(viewport={"width": 1280, "height": 720})
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    if durations is not None:
        page.add_init_script(f"window.SEG_DUR = {json.dumps(durations)};")
    page.goto((HERE / "scene.html").as_uri() + f"?v={name}")
    page.wait_for_function("window.sceneReady === true")
    if errors:
        raise RuntimeError(errors)
    return browser, page


def narration(name: str):
    with sync_playwright() as p:
        browser, page = open_page(p, name)
        segs = page.evaluate("SEGMENTS")
        mins = page.evaluate("typeof SEG_MIN === 'undefined' ? null : SEG_MIN")
        browser.close()
    return segs, mins


async def _tts(text: str, path: Path) -> None:
    import edge_tts

    await edge_tts.Communicate(text, VOICE, rate="+4%").save(str(path))


def to_pcm(mp3: Path) -> bytes:
    return subprocess.run([FFMPEG, "-v", "error", "-i", str(mp3), "-f", "s16le", "-ac", "1", "-ar", str(RATE), "-"],
                          capture_output=True, check=True).stdout


def synthesize(name: str, build: Path):
    segs, mins = narration(name)
    pcm_all, durations = b"", []
    for i, text in enumerate(segs):
        key = hashlib.sha1(f"{VOICE}|{text}".encode()).hexdigest()[:12]
        mp3 = build / f"seg{i:02d}-{key}.mp3"
        if not mp3.exists():
            asyncio.run(_tts(text, mp3))
        pcm = to_pcm(mp3)
        speech = len(pcm) / 2 / RATE
        dur = speech + GAP
        if mins and i < len(mins) and mins[i]:
            dur = max(dur, mins[i])
        pad = int(round(dur * RATE)) - len(pcm) // 2
        pcm_all += pcm + b"\0\0" * pad
        durations.append(round(len(pcm + b"\0\0" * pad) / 2 / RATE, 4))
    with wave.open(str(build / "audio.wav"), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(RATE); w.writeframes(pcm_all)
    (build / "durations.json").write_text(json.dumps(durations))
    return durations


def _render(args):
    name, durations, start, stop, frames = args
    with sync_playwright() as p:
        browser, page = open_page(p, name, durations)
        for i in range(start, stop):
            data = page.evaluate("(t) => { renderFrame(t); return document.getElementById('c').toDataURL('image/png'); }",
                                 i / FPS)
            (Path(frames) / f"f{i:05d}.png").write_bytes(base64.b64decode(data.split(",", 1)[1]))
        browser.close()
    return stop - start


def render(name: str, workers: int = 8) -> None:
    build = HERE / "build" / name
    build.mkdir(parents=True, exist_ok=True)
    durations = synthesize(name, build)
    total = sum(durations)
    n = int(total * FPS) + 1
    frames = build / "frames"
    frames.mkdir(exist_ok=True)
    for f in frames.glob("*.png"):
        f.unlink()
    chunk = (n + workers - 1) // workers
    jobs = [(name, durations, s, min(n, s + chunk), str(frames)) for s in range(0, n, chunk)]
    with ProcessPoolExecutor(workers) as ex:
        done = sum(ex.map(_render, jobs))
    OUT.mkdir(parents=True, exist_ok=True)
    mp4 = OUT / f"{name}.mp4"
    subprocess.run([FFMPEG, "-v", "error", "-y", "-framerate", str(FPS), "-i", str(frames / "f%05d.png"),
                    "-i", str(build / "audio.wav"), "-c:v", "libx264", "-preset", "slow", "-crf", "25",
                    "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "80k", "-movflags", "+faststart",
                    "-shortest", str(mp4)], check=True)
    poster = frames / f"f{min(n - 1, int(durations[0] * FPS * 0.8)):05d}.png"
    subprocess.run([FFMPEG, "-v", "error", "-y", "-i", str(poster), "-q:v", "4", str(OUT / f"{name}.jpg")], check=True)
    print(f"{name}: {done} frames, {total:.1f}s, {mp4.stat().st_size / 1e6:.1f} MB")


def preview(name: str, times) -> None:
    build = HERE / "build" / name
    durations = json.loads((build / "durations.json").read_text()) if (build / "durations.json").exists() else None
    out = build / "preview"
    out.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser, page = open_page(p, name, durations)
        for t in times:
            data = page.evaluate("(t) => { renderFrame(t); return document.getElementById('c').toDataURL('image/png'); }", t)
            (out / f"t{t:06.2f}.png").write_bytes(base64.b64decode(data.split(",", 1)[1]))
        print("total", page.evaluate("window.TOTAL"))
        browser.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+")
    ap.add_argument("--preview", nargs="*", type=float)
    ap.add_argument("--tts-only", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    for name in a.names:
        if a.preview is not None:
            preview(name, a.preview)
        elif a.tts_only:
            build = HERE / "build" / name
            build.mkdir(parents=True, exist_ok=True)
            print(name, synthesize(name, build))
        else:
            render(name, a.workers)
