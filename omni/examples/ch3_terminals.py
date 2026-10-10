import asyncio
import tempfile

from sglang_omni.config.schema import EndpointsConfig, PipelineConfig, StageConfig
from sglang_omni.pipeline.mp_runner import MultiProcessPipelineRunner
from sglang_omni.proto import OmniRequest

M = "ch3_stages"


def make_config() -> PipelineConfig:
    return PipelineConfig(
        model_path="toy",
        entry_stage="split",
        stages=[
            StageConfig(name="split", process="p_split", factory_path=f"{M}.create_split",
                        next=["text", "audio"], route_fn=f"{M}.route_split"),
            StageConfig(name="text", process="p_text", factory_path=f"{M}.create_text", terminal=True),
            StageConfig(name="audio", process="p_audio", factory_path=f"{M}.create_audio", terminal=True),
        ],
        terminal_stages_fn=f"{M}.resolve_terminals",
        endpoints=EndpointsConfig(base_path=tempfile.mkdtemp(prefix="omni-")),
    )


async def main() -> None:
    runner = MultiProcessPipelineRunner(make_config())
    await runner.start(timeout=120)
    coord = runner.coordinator
    try:
        print("静态终点：", sorted(coord.terminal_stages))
        r1 = await coord.submit("r1", OmniRequest(inputs="hi omni", params={"want_audio": False}))
        print("只要文本：", r1)
        r2 = await coord.submit("r2", OmniRequest(inputs="hi omni", params={"want_audio": True}))
        print("文本+音频：", r2)
        print("还在跟踪的请求：", list(coord.requests))
    finally:
        await runner.stop()


if __name__ == "__main__":
    asyncio.run(main())
