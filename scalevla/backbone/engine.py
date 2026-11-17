"""Canonical DeepSeek-V4.1-Flash feature engine: vLLM 0.31 with batch-invariant kernels, linear emulation, DeepGEMM off,
data-parallel vision encoder and an early exit after the layer-20 feature (scripts/apply_vllm_patch.py). One place for the
LLM(...) arguments and the observation prompt, shared by feature extraction and the feature server.

The language model is a causal encoder (layers 0-19) followed by a decoder (layers 20-39). The feature is the input of
layers[20].attn: the encoder output after the model's own collapse of the hyper-connection streams and layer 20's
attention norm, i.e. exactly the tensor the decoder reads. Layer 20's attention and everything after it never run.
"""
import os
import sys

import numpy as np

IMAGE_TOKEN_ID = 129264  # config.json image_token_id
FEATURE_DIM = 5120
FEATURE_LAYER = 20
CANONICAL_ENV = {"VLLM_BATCH_INVARIANT": "1", "VLLM_USE_DEEP_GEMM": "0", "SCALEVLA_DS_FEATURE_LAYER": str(FEATURE_LAYER),
                 "VLLM_ALLOW_INSECURE_SERIALIZATION": "1"}
ENGINE_KW = dict(tensor_parallel_size=4, enable_expert_parallel=True, tokenizer_mode="deepseek_v41",
                 limit_mm_per_prompt={"image": 3}, mm_processor_cache_gb=0, engram_config={"cpu_offload": False},
                 cpu_offload_gb=0, max_model_len=4096, max_num_seqs=64, max_num_batched_tokens=20480,
                 gpu_memory_utilization=0.95, enforce_eager=True, enable_prefix_caching=False,
                 moe_backend="humming", linear_backend="emulation", mm_encoder_tp_mode="data")
# placeholder content of one image in the official chat encoding (the pixels go through multi_modal_data)
_IMAGE_SLOT = {"type": "image_url", "image_url": {"url": "data:image/png;base64,"}}


def check_env():
    bad = {k: os.environ.get(k) for k, v in CANONICAL_ENV.items() if os.environ.get(k) != v}
    if bad:
        raise RuntimeError(f"not the canonical DeepSeek feature environment (use scripts/run_deepseek.sh): {bad}")


def make_llm(model, **override):
    check_env()
    from vllm import LLM
    return LLM(model=str(model), **{**ENGINE_KW, **override})


def engine_spec(model):
    """What identifies the feature: recorded in every cache shard."""
    return {"model": os.path.basename(os.path.normpath(str(model))), "engine_kw": {k: str(v) for k, v in ENGINE_KW.items()},
            "env": {k: os.environ.get(k) for k in CANONICAL_ENV}, "feature": f"layers[{FEATURE_LAYER}].attn input",
            "image_token_id": IMAGE_TOKEN_ID}


def build_request(frames, instruction, encoding_dir):
    """frames: HxWx3 uint8 per camera (store order); instruction: task text. The model's official encode_messages
    (<model>/encoding/encoding.py), chat mode, images then text."""
    if encoding_dir not in sys.path:
        sys.path.insert(0, encoding_dir)
    from encoding import encode_messages
    from PIL import Image
    content = [_IMAGE_SLOT] * len(frames) + [{"type": "text", "text": str(instruction)}]
    prompt = encode_messages([{"role": "user", "content": content}], thinking_mode="chat", return_multi_modal_data=True)[0]
    return {"prompt": prompt, "multi_modal_data": {"image": [Image.fromarray(np.ascontiguousarray(f)) for f in frames]}}
