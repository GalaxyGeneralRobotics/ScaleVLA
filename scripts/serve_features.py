"""Serve DeepSeek-V4.1-Flash features to policy servers (DeepSeek vLLM environment, 4 GPUs).

    bash scripts/run_deepseek.sh -m scripts.serve_features --model <DeepSeek-V4.1-Flash dir> --port 5601 \
        [--gpu-util 0.90] [--max-batched-tokens 4096] [--batch-window-ms 20]

--gpu-util and --max-batched-tokens leave GPU memory for action heads and simulators on the same GPUs; the batch-invariant
kernels keep the features unchanged (scripts.check_features verifies it against a feature cache).
"""
import argparse
import os
import sys
import time

_t = next((sys.argv[i + 1] for i, x in enumerate(sys.argv[:-1]) if x == "--threads"), "4")  # set before torch loads
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, _t)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True, help="local DeepSeek-V4.1-Flash model dir")
    ap.add_argument("--port", type=int, default=5601)
    ap.add_argument("--gpu-util", type=float, default=None, help="vLLM gpu_memory_utilization")
    ap.add_argument("--max-batched-tokens", type=int, default=None, help="vLLM max_num_batched_tokens")
    ap.add_argument("--batch-window-ms", type=float, default=0,
                    help="merge the requests of concurrent policy servers that arrive within this window into one engine call")
    ap.add_argument("--threads", default="4", help="CPU threads (OMP / MKL)")
    a = ap.parse_args()
    from scalevla.backbone.deepseek import DeepSeekBackbone
    from scalevla.features.server import serve
    t0 = time.time()
    over = {**({} if a.gpu_util is None else {"gpu_memory_utilization": a.gpu_util}),
            **({} if a.max_batched_tokens is None else {"max_num_batched_tokens": a.max_batched_tokens})}
    bb = DeepSeekBackbone(a.model, **over)
    print(f"{bb.name} up in {time.time() - t0:.0f}s (overrides {over})", flush=True)
    serve(bb, a.port, over, batch_window_s=a.batch_window_ms / 1e3)


if __name__ == "__main__":
    main()
