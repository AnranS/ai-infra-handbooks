import asyncio
from pathlib import PurePosixPath

from ch3_terminals import make_config
from sglang_omni.pipeline.mp_runner import MultiProcessPipelineRunner


async def main() -> None:
    runner = MultiProcessPipelineRunner(make_config())
    await runner.start(timeout=120)
    try:
        for name, endpoint in sorted(runner.prep.endpoints.items()):
            scheme, _, path = endpoint.partition("://")
            print(f"{name:18s} {scheme}://<运行目录>/{PurePosixPath(path).name}")
    finally:
        await runner.stop()


if __name__ == "__main__":
    asyncio.run(main())
