import asyncio
import tempfile

from sglang_omni.config.schema import EndpointsConfig, PipelineConfig, StageConfig
from sglang_omni.pipeline.mp_runner import MultiProcessPipelineRunner

M = "ch3_stages"


def make_config() -> PipelineConfig:
    return PipelineConfig(
        model_path="toy",
        entry_stage="slow",
        stages=[StageConfig(name="slow", process="p_slow", factory_path=f"{M}.create_slow", terminal=True)],
        endpoints=EndpointsConfig(base_path=tempfile.mkdtemp(prefix="omni-")),
    )


async def main() -> None:
    runner = MultiProcessPipelineRunner(make_config())
    await runner.start(timeout=120)
    coord = runner.coordinator
    try:
        task = asyncio.create_task(coord.submit("r1", "sleep please"))
        await asyncio.sleep(0.5)
        print("提交后的状态：", coord.get_request_info("r1").state.value)
        print("abort 返回：", await coord.abort("r1"))
        try:
            await task
        except asyncio.CancelledError as exc:
            print("submit 的结果：CancelledError", exc)
        print("abort 之后还在跟踪吗：", "r1" in coord.requests)
        try:
            await coord.submit("r1", "again")
        except Exception as exc:
            print("同一个 id 再提交：", type(exc).__name__, str(exc)[:70])
        print("换个 id：", await coord.submit("r2", "again"))
    finally:
        await runner.stop()


if __name__ == "__main__":
    asyncio.run(main())
