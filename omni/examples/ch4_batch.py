import asyncio
import tempfile
import time

from sglang_omni.config.schema import EndpointsConfig, FactoryArgs, PipelineConfig, StageConfig
from sglang_omni.pipeline.mp_runner import MultiProcessPipelineRunner


def make_config(wait_when_idle: bool) -> PipelineConfig:
    return PipelineConfig(
        model_path="toy",
        entry_stage="encoder",
        stages=[StageConfig(name="encoder", process="p_enc", terminal=True,
                            factory_path="ch4_batch_stages.create_encoder",
                            factory=FactoryArgs(max_batch_wait_ms=200, batch_wait_when_idle=wait_when_idle))],
        endpoints=EndpointsConfig(base_path=tempfile.mkdtemp(prefix="omni-")),
    )


async def run(wait_when_idle: bool) -> None:
    runner = MultiProcessPipelineRunner(make_config(wait_when_idle))
    await runner.start(timeout=120)
    coord = runner.coordinator
    try:
        results = await asyncio.gather(*(coord.submit(f"burst-{i}", i) for i in range(4)))
        print(f"batch_wait_when_idle={wait_when_idle}")
        print("  同时到达的 4 个请求，各自所在批的大小：", [r["batch_size"] for r in results])
        t0 = time.perf_counter()
        lone = await coord.submit("lone", 0)
        waited = (time.perf_counter() - t0) * 1000
        print(f"  单独一个请求：批大小 {lone['batch_size']}，多等了 200 ms 吗：{waited >= 180}")
    finally:
        await runner.stop()


async def main() -> None:
    await run(True)
    await run(False)


if __name__ == "__main__":
    asyncio.run(main())
