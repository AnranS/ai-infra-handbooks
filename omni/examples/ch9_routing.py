import warnings

warnings.filterwarnings("ignore")
from sglang_omni.models.qwen3_omni import request_builders as rb
from sglang_omni.proto import OmniRequest, StagePayload

for modalities in (None, ["text"], ["text", "audio"]):
    metadata = {} if modalities is None else {"output_modalities": modalities}
    request = OmniRequest(inputs="描述一下这张图", metadata=metadata)
    payload = StagePayload("r1", request, {})
    print(f"output_modalities={modalities}")
    print("  编码器发给：      ", rb.resolve_encoder_next_stages("r1", payload))
    print("  thinker 流结束通知：", rb.resolve_thinker_stream_done_targets("r1", payload))
    print("  Coordinator 等待：  ", rb.resolve_terminal_stages(request))
