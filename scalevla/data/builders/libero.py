"""LIBERO sample store (schema v1) for one suite, from the LeRobot v2.1 release of the no-noop demonstrations
(HF IPEC-COMMUNITY/libero_<suite>_no_noops_1.0.0_lerobot; needs pyarrow and ffmpeg with libdav1d).

<out>/frames_<cam>.npy  uint8 [N, 256, 256, 3], cameras image, wrist_image (GR00T's libero_sim video keys)
<out>/table.npz         episode_index, frame_index, task_index [N] int64; state [N, 8], action [N, 7] float32
<out>/index.json        schema, cameras, tasks {task_index: instruction}, episodes (start, length), source
Frames are decoded once to rgb24 (no resize, no augmentation); every episode's decoded frame count must equal its row count.
"""
import hashlib
import json
import os
import subprocess

import numpy as np

from ..samples import SCHEMA_V1

CAMS = ("image", "wrist_image")
H = W = 256


def decode(path):
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-threads", "4", "-c:v", "libdav1d", "-i", path,
                          "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, H, W, 3)


def build(suite_dir, out):
    import pyarrow.parquet as pq
    meta = f"{suite_dir}/meta"
    info = json.load(open(f"{meta}/info.json"))
    assert info["codebase_version"] == "v2.1" and info["fps"] == 20, info["codebase_version"]
    tasks = {json.loads(l)["task_index"]: json.loads(l)["task"] for l in open(f"{meta}/tasks.jsonl")}
    eps = sorted((json.loads(l) for l in open(f"{meta}/episodes.jsonl")), key=lambda e: e["episode_index"])
    n_total = sum(e["length"] for e in eps)
    assert n_total == info["total_frames"], (n_total, info["total_frames"])
    os.makedirs(out, exist_ok=True)
    mm = {c: np.lib.format.open_memmap(f"{out}/frames_{c}.npy", "w+", np.uint8, (n_total, H, W, 3)) for c in CAMS}
    cols = {k: [] for k in ("episode_index", "frame_index", "task_index", "state", "action")}
    ep_index, off = [], 0
    for k, e in enumerate(eps):
        ei = e["episode_index"]
        chunk = ei // info["chunks_size"]
        t = pq.read_table(f"{suite_dir}/" + info["data_path"].format(episode_chunk=chunk, episode_index=ei)).to_pydict()
        n = len(t["frame_index"])
        assert n == e["length"] and t["frame_index"] == list(range(n)), (ei, n, e["length"])
        assert set(t["episode_index"]) == {ei} and len(set(t["task_index"])) == 1
        for c in CAMS:
            fr = decode(f"{suite_dir}/" + info["video_path"].format(episode_chunk=chunk, episode_index=ei,
                                                                   video_key=f"observation.images.{c}"))
            assert len(fr) == n, f"episode {ei} camera {c}: {len(fr)} decoded frames vs {n} rows"
            mm[c][off:off + n] = fr
        cols["episode_index"] += [ei] * n
        cols["frame_index"] += t["frame_index"]
        cols["task_index"] += t["task_index"]
        cols["state"] += t["observation.state"]
        cols["action"] += t["action"]
        ep_index.append({"episode_index": ei, "start": off, "length": n, "task_index": t["task_index"][0]})
        off += n
        if k % 50 == 0:
            print(f"{k}/{len(eps)} episodes, {off}/{n_total} frames", flush=True)
    for c in CAMS:
        mm[c].flush()
    np.savez(f"{out}/table.npz", episode_index=np.asarray(cols["episode_index"], np.int64),
             frame_index=np.asarray(cols["frame_index"], np.int64), task_index=np.asarray(cols["task_index"], np.int64),
             state=np.asarray(cols["state"], np.float32), action=np.asarray(cols["action"], np.float32))
    src = f"{suite_dir}/data.manifest.json"  # optional record of the downloaded HF repo / revision
    source = json.load(open(src)) if os.path.exists(src) else {}
    json.dump({"schema": SCHEMA_V1, "benchmark": "libero", "source_repo": source.get("repo"),
               "source_revision": source.get("revision"), "cameras": list(CAMS), "frame_shape": [H, W, 3],
               "decode": "ffmpeg libdav1d rgb24, no resize", "n": n_total, "state_dim": 8, "action_dim": 7, "fps": info["fps"],
               "tasks": tasks, "episodes": ep_index,
               "frames_sha256_first_episode": {c: hashlib.sha256(np.ascontiguousarray(mm[c][:eps[0]["length"]])).hexdigest()
                                               for c in CAMS}},
              open(f"{out}/index.json", "w"), indent=1)
    print(f"DONE {n_total} frames, {len(eps)} episodes -> {out}", flush=True)
