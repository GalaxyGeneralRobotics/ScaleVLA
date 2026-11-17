#!/usr/bin/env bash
# Run a DeepSeek-V4.1-Flash driver in the DeepSeek vLLM environment under the canonical feature environment:
# batch-invariant kernels, DeepGEMM off, early exit after the layer-20 feature.
#   bash scripts/run_deepseek.sh -m scripts.extract_features ...
#   bash scripts/run_deepseek.sh -m scripts.serve_features ...
#   DEEPSEEK_PYTHON     python of the DeepSeek vLLM environment, patched with scripts.apply_vllm_patch (required)
#   DS_GPUS             the 4 GPUs (default 0,1,2,3)
#   DEEPSEEK_CUDA_HOME  CUDA toolkit for the JIT-compiled kernels (optional)
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
PY=${DEEPSEEK_PYTHON:?set DEEPSEEK_PYTHON to the python of the DeepSeek vLLM environment}
if [ -n "${DEEPSEEK_CUDA_HOME:-}" ]; then
  export CUDA_HOME=$DEEPSEEK_CUDA_HOME PATH=$DEEPSEEK_CUDA_HOME/bin:$PATH LD_LIBRARY_PATH=$DEEPSEEK_CUDA_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
fi
export CUDA_VISIBLE_DEVICES=${DS_GPUS:-0,1,2,3} TILELANG_DEFAULT_TARGET=cuda OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export VLLM_BATCH_INVARIANT=1 VLLM_USE_DEEP_GEMM=0 SCALEVLA_DS_FEATURE_LAYER=20 VLLM_ALLOW_INSECURE_SERIALIZATION=1
export NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost
export PYTHONPATH=$REPO
exec "$PY" "$@"
