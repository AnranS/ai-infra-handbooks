import asyncio
import tempfile

from sglang_omni.config.schema import EndpointsConfig, PipelineConfig, StageConfig
from sglang_omni.pipeline.mp_runner import MultiProcessPipelineRunner

M = "ch4_fanin_stages"


def make_config() -> PipelineConfig:
    return PipelineConfig(
        model_path="toy",
        entry_stage="split",
        stages=[
            StageConfig(name="split", process="p_split", factory_path=f"{M}.create_split",
                        next=["upper", "count"],
                        project_payload={"upper": f"{M}.project_to_upper", "count": f"{M}.project_to_count"}),
            StageConfig(name="upper", process="p_upper", factory_path=f"{M}.create_upper", next="join"),
            StageConfig(name="count", process="p_count", factory_path=f"{M}.create_count", next="join"),
            StageConfig(name="join", process="p_join", factory_path=f"{M}.create_join",
                        wait_for=["upper", "count"], merge_fn=f"{M}.merge", terminal=True),
        ],
        endpoints=EndpointsConfig(base_path=tempfile.mkdtemp(prefix="omni-")),
    )


async def main() -> None:
    runner = MultiProcessPipelineRunner(make_config())
    await runner.start(timeout=120)
    try:
        results = await asyncio.gather(*(runner.coordinator.submit(f"r{i}", t)
                                         for i, t in enumerate(["hello omni world", "stage by stage"])))
        for r in results:
            print(r)
    finally:
        await runner.stop()


if __name__ == "__main__":
    asyncio.run(main())
