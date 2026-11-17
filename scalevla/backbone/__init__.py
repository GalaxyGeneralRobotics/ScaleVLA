"""DeepSeek-V4.1-Flash backbone through vLLM: engine (canonical engine arguments and observation prompt), worker
(row capture inside the vLLM workers) and deepseek (the backbone used by feature extraction and the feature server).
Runs in the DeepSeek vLLM environment with the early-exit patch (scripts/apply_vllm_patch.py), under
scripts/run_deepseek.sh."""
