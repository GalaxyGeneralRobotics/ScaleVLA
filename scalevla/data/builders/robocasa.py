"""RoboCasa sample store (schema v2) from GR00T's X-Embodiment-Sim Panda data (HF nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim,
the 24 single_panda_gripper.<Task> LeRobot v2.0 sets; needs pyarrow, torchcodec and cv2).

Demonstrations: per task (task dirs sorted by name), the episodes marked "Valid", sorted by episode_index, permuted with
np.random.default_rng([seed, task position]); the first N are kept (Gen-30: N = 30, Gen-100: N = 100; the subsets of one
seed are nested). N = 0 keeps every valid episode.
Frames: the mp4 frames decoded with torchcodec (GR00T's video backend) at the parquet row timestamps, RGB, stored as
lossless PNG (cv2.imencode; the round trip is checked on sampled frames). Cameras left_view, right_view, wrist_view =
GR00T's res256_image_side_0, side_1, wrist_0.
Rows: state = observation.state (53), action = action (12) of the same row (no shift); instruction = the episode's task
description. State / action key ranges = meta/modality.json (recorded; scalevla.data.embodiments checks it).
"""
import hashlib
import json
import os
import time
from multiprocessing import get_context

import numpy as np

from ..samples import SCHEMA_V2

CAMS = ("left_view", "right_view", "wrist_view")
GR00T_KEYS = {"left_view": "res256_image_side_0", "right_view": "res256_image_side_1", "wrist_view": "res256_image_wrist_0"}
TASK_PREFIX = "single_panda_gripper."


def select_demos(data, demos_per_task, seed):
    """-> {task dir: (task position, sorted episode indices)} for every task dir of the dataset."""
    names = sorted(x for x in os.listdir(data) if x.startswith(TASK_PREFIX) and os.path.isdir(f"{data}/{x}"))
    out = {}
    for pos, task in enumerate(names):
        eps = sorted((json.loads(l) for l in open(f"{data}/{task}/meta/episodes.jsonl") if l.strip()),
                     key=lambda e: e["episode_index"])
        valid = [e["episode_index"] for e in eps if "Valid" in e["tasks"]]
        if demos_per_task:
            if len(valid) < demos_per_task:
                raise ValueError(f"{task}: {len(valid)} valid episodes, fewer than {demos_per_task}")
            valid = [valid[i] for i in np.random.default_rng([seed, pos]).permutation(len(valid))[:demos_per_task]]
        out[task] = (pos, sorted(valid))
    return out


def episode(args):
    data, task, ep = args
    import cv2
    import pyarrow.parquet as pq
    from torchcodec.decoders import VideoDecoder
    ch = ep // 1000
    t = pq.read_table(f"{data}/{task}/data/chunk-{ch:03d}/episode_{ep:06d}.parquet",
                      columns=["observation.state", "action", "timestamp", "episode_index"])
    assert set(t.column("episode_index").to_pylist()) == {ep}
    ts = np.asarray(t.column("timestamp").to_pylist(), np.float64)
    st = np.asarray(t.column("observation.state").to_pylist(), np.float32)
    ac = np.asarray(t.column("action").to_pylist(), np.float32)
    out = {}
    for c in CAMS:
        dec = VideoDecoder(f"{data}/{task}/videos/chunk-{ch:03d}/observation.images.{c}/episode_{ep:06d}.mp4")
        fb = dec.get_frames_played_at(seconds=ts.tolist())
        assert float(np.abs(fb.pts_seconds.numpy() - ts).max()) < 1e-6, (task, ep, c)
        fr = fb.data.permute(0, 2, 3, 1).contiguous().numpy()
        assert fr.shape == (len(ts), 256, 256, 3) and fr.dtype == np.uint8, fr.shape
        enc = [cv2.imencode(".png", fr[i])[1].tobytes() for i in range(len(fr))]
        for i in range(0, len(fr), 37):
            assert np.array_equal(cv2.imdecode(np.frombuffer(enc[i], np.uint8), cv2.IMREAD_COLOR), fr[i]), (task, ep, c, i)
        out[c] = enc
    return task, ep, out, st, ac


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def build(data, out, demos_per_task=0, seed=42, tasks="", workers=16):
    """data: the dataset root holding the single_panda_gripper.<Task> dirs; tasks: comma task names (default all 24)."""
    t0 = time.time()
    sel = select_demos(data, demos_per_task, seed)
    names = sorted(sel, key=lambda t: sel[t][0])
    if tasks:
        want = {t if t.startswith(TASK_PREFIX) else TASK_PREFIX + t for t in tasks.split(",")}
        assert want <= set(names), sorted(want - set(names))
        names = [t for t in names if t in want]
    eps = []
    for ti, task in enumerate(names):
        meta = {json.loads(l)["episode_index"]: json.loads(l) for l in open(f"{data}/{task}/meta/episodes.jsonl") if l.strip()}
        for ep in sel[task][1]:
            m = meta[ep]
            assert m["tasks"][2] == "Valid", (task, ep, m["tasks"])
            eps.append({"task": task, "src_episode": ep, "instruction": m["tasks"][0], "length": m["length"], "task_index": ti})
    instructions = sorted({e["instruction"] for e in eps})
    off = 0
    for i, e in enumerate(eps):
        e.update(episode_index=i, instruction_index=instructions.index(e["instruction"]), start=off)
        off += e["length"]
    n = off
    mod = json.load(open(f"{data}/{names[0]}/meta/modality.json"))
    for task in names[1:]:
        assert json.load(open(f"{data}/{task}/meta/modality.json")) == mod, task
    os.makedirs(out, exist_ok=False)
    bins = {c: open(f"{out}/jpeg_{c}.bin", "wb") for c in CAMS}
    offs = {c: [0] for c in CAMS}
    state, action = np.zeros((n, 53), np.float32), np.zeros((n, 12), np.float32)
    with get_context("spawn").Pool(workers) as p:
        for j, (task, ep, fr, st, ac) in enumerate(p.imap(episode, [(data, e["task"], e["src_episode"]) for e in eps], chunksize=1)):
            e = eps[j]
            assert (task, ep) == (e["task"], e["src_episode"]) and len(st) == e["length"], (task, ep, len(st), e["length"])
            for c in CAMS:
                for b in fr[c]:
                    bins[c].write(b)
                    offs[c].append(offs[c][-1] + len(b))
            state[e["start"]:e["start"] + e["length"]], action[e["start"]:e["start"] + e["length"]] = st, ac
            if j % 50 == 0:
                print(f"{j + 1}/{len(eps)} episodes, {time.time() - t0:.0f}s", flush=True)
    for c in CAMS:
        bins[c].close()
        o = np.asarray(offs[c], np.int64)
        assert len(o) == n + 1 and o[-1] == os.path.getsize(f"{out}/jpeg_{c}.bin")
        np.save(f"{out}/jpeg_{c}.offsets.npy", o)
    col = lambda key: np.concatenate([np.full(e["length"], e[key], np.int64) for e in eps])  # noqa: E731
    np.savez(f"{out}/table.npz", episode_index=col("episode_index"), task_index=col("task_index"),
             instruction_index=col("instruction_index"), state=state, action=action,
             frame_index=np.concatenate([np.arange(e["length"], dtype=np.int64) for e in eps]))
    files = {f: {"sha256": sha_file(f"{out}/{f}"), "size": os.path.getsize(f"{out}/{f}")} for f in sorted(os.listdir(out))}
    json.dump({"schema": SCHEMA_V2, "frames": "png", "benchmark": "robocasa", "source": "nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim",
               "demo_selection": {"demos_per_task": demos_per_task, "seed": seed,
                                  "rule": "per task: 'Valid' episodes sorted by episode_index, "
                                          "np.random.default_rng([seed, task position]).permutation, first demos_per_task"},
               "cameras": list(CAMS), "gr00t_video_keys": GR00T_KEYS, "frame_shape": [256, 256, 3],
               "n": n, "state_dim": 53, "action_dim": 12, "modality": mod, "instruction_mode": "per_episode",
               "tasks": {i: t for i, t in enumerate(names)}, "instructions": instructions, "episodes": eps, "files": files},
              open(f"{out}/index.json", "w"), indent=1)
    print(f"DONE {n} samples, {len(eps)} episodes, {len(names)} tasks -> {out} in {time.time() - t0:.0f}s", flush=True)
