from __future__ import annotations

import functools
import os
from typing import Any

from transformers import AutoConfig, AutoTokenizer, PretrainedConfig, PreTrainedTokenizerBase


def load_tokenizer(model_path: str) -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(model_path)


@functools.cache
def _load_hf_config(model_path: str) -> Any:
    return AutoConfig.from_pretrained(model_path)


def cached_load_hf_config(model_path: str) -> PretrainedConfig:
    config = _load_hf_config(model_path)
    return type(config)(**config.to_dict())  # 返回副本，调用方可以放心修改


def download_hf_weight(model_path: str) -> str:
    """本地目录直接返回；否则从 Hugging Face 下载 safetensors 权重。"""
    if os.path.isdir(model_path):
        return model_path
    from huggingface_hub import snapshot_download

    return snapshot_download(model_path, allow_patterns=["*.safetensors", "*.json"])
