"""python ch6_pipeline.py small|big：跑一次 src → (near, far)；日志打到标准输出，供 ch6_trace.py 解析。"""
import asyncio
import logging
import sys
import tempfile

from sglang_omni.config.schema import EndpointsConfig, FactoryArgs, PipelineConfig, StageConfig
from sglang_omni.pipeline.mp_runner import MultiProcessPipelineRunner

M = "ch6_stages"


def make_config(big_chunk: bool) -> PipelineConfig:
    return PipelineConfig(
        model_path="toy",
        entry_stage="src",
        stages=[
            StageConfig(name="src", process="p_main", factory_path=f"{M}.create_src", next=["near", "far"],
                        factory=FactoryArgs(big_chunk=big_chunk),
                        stream_to=["far"], project_payload={"near": f"{M}.project", "far": f"{M}.project"}),
            StageConfig(name="near", process="p_main", factory_path=f"{M}.create_near", terminal=True),
            StageConfig(name="far", process="p_far", factory_path=f"{M}.create_far", terminal=True,
                        can_accept_stream_before_payload=True),
        ],
        endpoints=EndpointsConfig(base_path=tempfile.mkdtemp(prefix="omni-")),
    )


async def main(big_chunk: bool) -> None:
    runner = MultiProcessPipelineRunner(make_config(big_chunk))
    await runner.start(timeout=120)
    try:
        print("RESULT", await runner.coordinator.submit("r0", "go"))
    finally:
        await runner.stop()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)   # 子进程继承这个日志级别
    asyncio.run(main(sys.argv[1] == "big"))
