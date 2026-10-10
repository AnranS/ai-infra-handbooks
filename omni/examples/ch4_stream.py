import asyncio
import tempfile

from sglang_omni.config.schema import EndpointsConfig, PipelineConfig, StageConfig
from sglang_omni.pipeline.mp_runner import MultiProcessPipelineRunner
from sglang_omni.proto import CompleteMessage, StreamMessage

M = "ch4_stream_stages"


def make_config() -> PipelineConfig:
    return PipelineConfig(
        model_path="toy",
        entry_stage="talker",
        stages=[
            StageConfig(name="talker", process="p_talker", factory_path=f"{M}.create_talker",
                        next="vocoder", stream_to=["vocoder"]),
            StageConfig(name="vocoder", process="p_vocoder", factory_path=f"{M}.create_vocoder",
                        terminal=True, can_accept_stream_before_payload=True),
        ],
        endpoints=EndpointsConfig(base_path=tempfile.mkdtemp(prefix="omni-")),
    )


async def main() -> None:
    runner = MultiProcessPipelineRunner(make_config())
    await runner.start(timeout=120)
    try:
        async for msg in runner.coordinator.stream("r1", "say something"):
            if isinstance(msg, StreamMessage):
                print(f"流式块 #{msg.chunk_id} 来自 {msg.from_stage}：{msg.chunk}")
            elif isinstance(msg, CompleteMessage):
                print(f"完成 来自 {msg.from_stage}：{msg.result}")
    finally:
        await runner.stop()


if __name__ == "__main__":
    asyncio.run(main())
