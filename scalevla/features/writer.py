"""Feature extraction: sample store -> feature cache (scalevla.data.cache).

The store is cut into shards of `shard` consecutive samples (shard_<j:05d>.safetensors). Writing is resumable: finished
shards are listed in manifest.json and skipped, and a cache is only resumed with the identical backbone spec and sample
store. part k/n writes only the shards with index % n == k, so n processes can write separate cache dirs that
scripts.merge_features joins. After writing, `verify` random samples are recomputed alone through the live path
and must equal their cached rows bitwise.
"""
import hashlib
import json
import os
import random
import time
from dataclasses import dataclass

import numpy as np
import torch

from ..data import cache as C
from ..data.samples import SampleStore


@dataclass
class WriterOptions:
    shard: int = 1024  # samples per shard
    verify: int = 8  # samples recomputed alone after writing (bitwise check)
    limit: int = 0  # write only the first `limit` samples (smoke tests)
    part: str = "0/1"  # k/n: this process writes the shards j with j % n == k


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(64 << 20), b""):
            h.update(b)
    return h.hexdigest()


def plan(n_samples, shard):
    """[(file name, sample indices)] of consecutive shards."""
    return [(f"shard_{first // shard:05d}.safetensors", list(range(first, min(first + shard, n_samples))))
            for first in range(0, n_samples, shard)]


def check_sample(backbone, store, i, feat, tt):
    """Recompute sample i alone through the live path and compare it with its cached rows."""
    f, t, _ = backbone.features([(store.images(i), store.instruction(i))])[0]
    t = torch.from_numpy(np.asarray(t, np.int8))
    return {"sample": i, "bitwise": bool(torch.equal(f, feat)), "labels_equal": bool(torch.equal(t, tt)), "rows": int(feat.shape[0]),
            "max_abs": float((f.float() - feat.float()).abs().max()) if f.shape == feat.shape else None}


def write_cache(backbone, samples, cache_dir, o: WriterOptions):
    S = SampleStore(samples)
    n_total = min(len(S), o.limit) if o.limit else len(S)
    os.makedirs(cache_dir, exist_ok=True)
    spec, index_sha = backbone.spec, sha256_file(f"{samples}/index.json")
    if os.path.exists(os.path.join(cache_dir, "manifest.json")):
        m = C.load_manifest(cache_dir)
        if m["spec"] != spec or m["samples"] != os.path.abspath(samples):
            raise SystemExit("the existing cache was written with a different backbone spec or sample store")
        if m.get("samples_index_sha256", index_sha) != index_sha:
            raise SystemExit("the sample store's index.json changed since the cache was started")
    else:
        m = {"schema": C.SCHEMA, "backbone": backbone.name, "samples": os.path.abspath(samples), "samples_schema": S.idx["schema"],
             "spec": spec, "shards": [], "complete": False, "log": []}
    m["samples_index_sha256"], m["part"] = index_sha, o.part
    done = {s["first"] for s in m["shards"]}
    meta = {"schema": C.SCHEMA, "backbone": backbone.name, "spec": json.dumps(spec, sort_keys=True)}
    pk, pn = (int(x) for x in o.part.split("/"))
    t_all, n_new = time.time(), 0
    for j, (fn, idxs) in enumerate(plan(n_total, o.shard)):
        if idxs[0] in done or j % pn != pk:
            continue
        t0 = time.time()
        path = os.path.join(cache_dir, fn)
        res = backbone.write_shard(S, idxs, path, meta)
        dt = time.time() - t0
        m["shards"].append({"file": fn, "first": idxs[0], "n": len(idxs), "rows": res["rows"], "sha256": sha256_file(path),
                            "s": round(dt, 1), "tokens_per_sample": res["tokens_per_sample"]})
        C.save_manifest(cache_dir, m)
        n_new += len(idxs)
        print(f"{fn}: {len(idxs)} samples, {res['rows'] / len(idxs):.1f} tokens / sample, {len(idxs) / dt:.2f} samples/s; "
              f"total {sum(s['n'] for s in m['shards'])}/{n_total}", flush=True)
    m["complete"] = sum(s["n"] for s in m["shards"]) == len(S)  # a limited run or a single part is never complete
    m["log"].append({"time": time.strftime("%F %T"), "new_samples": n_new, "wall_s": round(time.time() - t_all, 1)})
    if o.verify and m["shards"]:  # a part may own no shard (more parts than shards): nothing to verify
        R = C.CacheReader(cache_dir, require_complete=False)
        mine = [i for s in m["shards"] for i in range(s["first"], s["first"] + s["n"])]
        ver = [check_sample(backbone, S, i, *R.get(i)) for i in sorted(random.Random(0).sample(mine, min(o.verify, len(mine))))]
        m["verify_bitwise"] = {"n": len(ver), "all": all(v["bitwise"] and v["labels_equal"] for v in ver), "samples": ver}
        print("verify (recomputed alone == cached, bitwise):", m["verify_bitwise"]["all"], flush=True)
    C.save_manifest(cache_dir, m)
    print("DONE complete" if m["complete"] else "DONE partial", flush=True)
    return m
