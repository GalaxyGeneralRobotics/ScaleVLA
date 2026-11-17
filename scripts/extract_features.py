"""Extract the DeepSeek-V4.1-Flash features of a sample store into a feature cache (DeepSeek vLLM environment, 4 GPUs).

    bash scripts/run_deepseek.sh -m scripts.extract_features --model <DeepSeek-V4.1-Flash dir> <samples> <cache>

Resumable. --part k/n writes every n-th shard, so several 4-GPU jobs can share one store; join their outputs with
scripts.merge_features. At the end, --verify K samples are recomputed one at a time and must equal the cache bitwise.
"""
import argparse
import os
import sys

# CPU threads of the preprocessing, set before torch loads: the default (every core) oversubscribes a shared node
_t = next((sys.argv[i + 1] for i, x in enumerate(sys.argv[:-1]) if x == "--threads"), "4")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, _t)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("samples")
    ap.add_argument("cache_dir")
    ap.add_argument("--model", required=True, help="local DeepSeek-V4.1-Flash model dir")
    ap.add_argument("--batch", type=int, default=64, help="requests per engine call (batch-invariant)")
    ap.add_argument("--shard", type=int, default=1024, help="samples per shard")
    ap.add_argument("--verify", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--part", default="0/1")
    ap.add_argument("--threads", default="4", help="CPU threads (OMP / MKL)")
    a = ap.parse_args()
    from scalevla.backbone.deepseek import DeepSeekBackbone
    from scalevla.features.writer import WriterOptions, write_cache
    bb = DeepSeekBackbone(a.model, batch=a.batch)
    write_cache(bb, a.samples, a.cache_dir, WriterOptions(shard=a.shard, verify=a.verify, limit=a.limit, part=a.part))
    bb.close()


if __name__ == "__main__":
    main()
