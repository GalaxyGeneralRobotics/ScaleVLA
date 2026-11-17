"""Join cache parts written with scripts.extract_features --part k/n (one dir per part, global shard names) into one cache.

Checks: the same backbone, spec and sample store in every part, no shard twice, every sample of the store covered exactly
once (partial=True allows gaps; complete = full coverage) and each shard's sha256. Shard files are moved (same file
system); re-running with more parts into the same output dir adds them.
"""
import json
import os

from ..data import cache as C
from .writer import sha256_file


def merge(out, parts, partial=False):
    prev = C.load_manifest(out) if os.path.exists(os.path.join(out, "manifest.json")) else None
    ms = [C.load_manifest(p) for p in parts]
    for k in ("backbone", "spec", "samples", "samples_schema"):
        assert all(m[k] == ms[0][k] for m in ms + ([prev] if prev else [])), f"parts differ in {k}"
    with open(os.path.join(ms[0]["samples"], "index.json")) as f:
        n_store = json.load(f)["n"]
    shards, seen = [], set()
    if prev:
        for s in prev["shards"]:
            seen.add(s["file"])
            shards.append((out, s))
    for p, m in zip(parts, ms):
        for s in m["shards"]:
            if prev and any(q["file"] == s["file"] and q["sha256"] == s["sha256"] for q in prev["shards"]):
                continue  # merged earlier
            assert s["file"] not in seen, f"shard twice: {s['file']}"
            seen.add(s["file"])
            shards.append((p, s))
    shards.sort(key=lambda x: x[1]["first"])
    nxt, gaps = 0, []
    for _, s in shards:
        assert s["first"] >= nxt, f"overlap at {s['first']}"
        if s["first"] > nxt:
            gaps.append((nxt, s["first"]))
        nxt = s["first"] + s["n"]
    if nxt != n_store:
        gaps.append((nxt, n_store))
    assert partial or not gaps, f"coverage gaps {gaps[:5]}"
    os.makedirs(out, exist_ok=True)
    for p, s in shards:
        src, dst = os.path.join(p, s["file"]), os.path.join(out, s["file"])
        if src != dst and not os.path.exists(dst):
            os.replace(src, dst)
        assert sha256_file(dst) == s["sha256"], f"sha256 mismatch {s['file']}"
    m = {k: ms[0][k] for k in ("schema", "backbone", "samples", "samples_schema", "spec")}
    m.update(shards=[s for _, s in shards], complete=not gaps, gaps=gaps,
             parts=(prev or {}).get("parts", []) + [{"dir": p, "part": mm.get("part"), "n_shards": len(mm["shards"]),
                                                     "verify_bitwise": mm.get("verify_bitwise", {}).get("all")}
                                                    for p, mm in zip(parts, ms)])
    C.save_manifest(out, m)
    print("MERGED", len(shards), "shards,", sum(s["n"] for _, s in shards), "samples, complete", not gaps, "->", out, flush=True)
    return m
