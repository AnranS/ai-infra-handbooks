import os
import warnings

warnings.filterwarnings("ignore")
from sglang_omni.config.manager import ConfigManager
from sglang_omni.models.qwen3_omni.config import (
    Qwen3OmniSpeechColocatedPipelineConfig,
    Qwen3OmniSpeechPipelineConfig,
)
from sglang_omni.models.qwen3_tts.config import Qwen3TTSPipelineConfig
from sglang_omni.pipeline.runtime_config import prepare_pipeline_runtime


def show(title, config):
    print(f"== {title}")
    prep = prepare_pipeline_runtime(config)
    try:
        for group in prep.process_plan.groups:
            where = "CPU " if group.gpu_id is None else f"GPU{group.gpu_id}"
            print(f"  进程 {group.name:14s} {where}  {', '.join(group.stage_names)}")
        for gpu_id, gpu in sorted(prep.placement_plan.gpus.items()):
            missing = f"；未声明比例：{', '.join(gpu.missing_fraction_stage_names)}" if gpu.missing_fraction_stage_names else ""
            print(f"  GPU{gpu_id}：显存比例合计 {gpu.total_gpu_memory_fraction:.2f}{missing}")
    finally:
        prep.runtime_dir.close()


show("Qwen3-TTS", Qwen3TTSPipelineConfig(model_path="Qwen/Qwen3-TTS-12Hz-0.6B-Base"))
show("Qwen3-Omni 语音，默认（两张卡）", Qwen3OmniSpeechPipelineConfig(model_path="Qwen/Qwen3-Omni-30B-A3B-Instruct"))
yaml_path = os.path.join(os.environ["OMNI_TREE"], "examples/configs/qwen3_omni_colocated_h100_bf16.yaml")
show("Qwen3-Omni 语音，单卡（examples/configs 里的 H100 配置）", ConfigManager.from_file(yaml_path).merge_config([]))
try:
    show("Qwen3-Omni 语音，单卡但不写显存比例", Qwen3OmniSpeechColocatedPipelineConfig(model_path="x"))
except ValueError as exc:
    print("  规划失败：", exc)
