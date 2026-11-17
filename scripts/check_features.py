"""Check that a running feature server reproduces a feature cache bitwise (run it before a closed-loop evaluation).

    python -m scripts.check_features --samples <samples> --cache <cache> --feature-port 5601 --out check.json [--n 16]

For --n random samples, the features and token labels from the server (one request per sample) must equal the cached rows
bitwise; then the same samples are sent by --concurrent clients at once (a batching server merges them into one engine
call) and must again equal the cache. Exit code 0 = all equal.
"""
import argparse
import json
import sys
import threading
import time


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--samples", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--feature-port", type=int, required=True)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--concurrent", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()
    import numpy as np
    import torch
    from scalevla.data.cache import CacheReader
    from scalevla.data.samples import SampleStore
    from scalevla.features.client import FeatureClient
    store, cache = SampleStore(a.samples), CacheReader(a.cache)
    idx = sorted(np.random.default_rng(a.seed).choice(len(cache), a.n, replace=False).tolist())
    ref = {}
    for i in idx:
        f, t = cache.get(i)
        ref[i] = (torch.as_tensor(f), torch.as_tensor(np.asarray(t, np.int8)))

    def check(i, out):
        f, t, _ = out
        return {"sample": i, "rows": int(f.shape[0]), "bitwise": bool(torch.equal(f.view(torch.int16), ref[i][0].view(torch.int16))),
                "labels_equal": bool(torch.equal(t, ref[i][1]))}

    cl, single, lat = FeatureClient(a.feature_port), [], []
    for i in idx:
        t0 = time.perf_counter()
        single.append(check(i, cl.features([store.images(i)], [store.instruction(i)])[0]))
        lat.append(time.perf_counter() - t0)
    conc = []
    for b in range(0, len(idx), a.concurrent):
        part, res = idx[b:b + a.concurrent], {}

        def run(i):
            res[i] = FeatureClient(a.feature_port).features([store.images(i)], [store.instruction(i)])[0]
        th = [threading.Thread(target=run, args=(i,)) for i in part]
        [x.start() for x in th]
        [x.join() for x in th]
        conc += [check(i, res[i]) for i in part]
    ok = all(r["bitwise"] and r["labels_equal"] for r in single + conc)
    rep = {"samples": a.samples, "cache": a.cache, "spec": cl.call(cmd="spec"), "n": a.n, "concurrent": a.concurrent,
           "single_median_s": float(np.median(lat)), "single": single, "concurrent_results": conc, "all_pass": ok}
    json.dump(rep, open(a.out, "w"), indent=1, default=str)
    print(f"features: {sum(r['bitwise'] and r['labels_equal'] for r in single)}/{len(single)} single, "
          f"{sum(r['bitwise'] and r['labels_equal'] for r in conc)}/{len(conc)} concurrent bitwise equal -> {'PASS' if ok else 'FAIL'}",
          flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
