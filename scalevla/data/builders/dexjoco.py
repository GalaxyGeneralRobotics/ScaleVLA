"""DexJoCo sample store (schema v2) from the official LeRobot v3.0 release (HF DexJoCo/DexJoCo-Datasets-LeRobot; needs
PyAV with libdav1d, pandas, scipy, cv2 and PIL). One store per (variant, robot type): single-arm and dual-arm tasks differ
in state width and cameras, so each robot type gets its own policy.

Conventions of the official evaluator (dexjoco_openpi_client):
  - cameras = the evaluator's camera_mapping keys (single arm: base, wrist; dual arm: base, wrist_left, wrist_right); the
    dataset video of key k is observation.images.<camera_mapping[k]> of configs/<variant>/<task>.yaml
  - frames: AV1 640x640 decoded with libdav1d, then the evaluator's preprocessing (openpi_client.image_tools
    resize_with_pad to 224x224, PIL bilinear), stored as JPEG quality 95 of the RGB array (cv2.imdecode returns RGB)
  - state[t] = observation.state[t] (23 / 46 dims, the proprioception the evaluator sends)
  - action[t] = the dataset rotation-vector action[t] in the env layout (embodiments.DEXJOCO_LAYOUT): rotation vector ->
    quaternion (scalar first) on the hemisphere of state[t]'s quaternion; action_rotvec keeps the dataset vector
  - instruction = the prompt of the task config (what the evaluator sends)
<out>/: jpeg_<cam>.bin + jpeg_<cam>.offsets.npy, table.npz (episode_index, frame_index, task_index, instruction_index,
src_episode [N]; state, action [N, D]; action_rotvec [N, D - 2 * arms]), index.json.
"""
import hashlib
import json
import os
import time
from multiprocessing import get_context

import numpy as np

from ..embodiments import DEXJOCO_LAYOUT, DEXJOCO_VIDEO
from ..samples import SCHEMA_V2

SIZE = 224
JPEG_QUALITY = 95
DIMS = {"single_arm": (23, 22), "dual_arm": (46, 44)}  # state, dataset (rotation-vector) action


def resize_with_pad(img, height=SIZE, width=SIZE):
    """openpi_client.image_tools._resize_with_pad_pil (PIL bilinear, zero padding) on one HWC uint8 array."""
    from PIL import Image
    im = Image.fromarray(img)
    cw, ch = im.size
    if (cw, ch) == (width, height):
        return np.asarray(im)
    ratio = max(cw / width, ch / height)
    rh, rw = int(ch / ratio), int(cw / ratio)
    r = im.resize((rw, rh), resample=Image.BILINEAR)
    z = Image.new(r.mode, (width, height), 0)
    z.paste(r, (max(0, int((width - rw) / 2)), max(0, int((height - rh) / 2))))
    return np.asarray(z)


def to_env_action(rv_action, state, robot_type):
    """Dataset action [xyz, rotvec, hand] per arm (right first) -> env layout [xyz, quat] per arm, then the hands, each
    quaternion on the hemisphere of the observed state quaternion. -> (env action [N, D], max rotation round-trip error)."""
    from scipy.spatial.transform import Rotation as R
    rv_action, state = np.asarray(rv_action, np.float64), np.asarray(state, np.float64)
    arms = 1 if robot_type == "single_arm" else 2
    poses, hands, err = [], [], 0.0
    for k in range(arms):  # dataset arm k = [xyz 3, rotvec 3, hand 16]; state arm k quaternion at 7k + 3 .. 7k + 7
        a = rv_action[:, 22 * k:22 * (k + 1)]
        q = R.from_rotvec(a[:, 3:6]).as_quat(scalar_first=True)
        sq = state[:, 7 * k + 3:7 * k + 7]
        q *= np.where((q * sq).sum(1, keepdims=True) < 0, -1.0, 1.0)
        back = R.from_quat(q, scalar_first=True)
        err = max(err, float(np.max((back.inv() * R.from_rotvec(a[:, 3:6])).magnitude(), initial=0.0)))
        poses.append(np.concatenate([a[:, :3], q], 1))
        hands.append(a[:, 6:22])
    return np.concatenate(poses + hands, 1).astype(np.float32), err


def from_env_action(env_action, robot_type):
    """Inverse of to_env_action (evaluator side): env layout -> dataset / OpenPI rotation-vector layout."""
    from scipy.spatial.transform import Rotation as R
    x = np.asarray(env_action, np.float64)
    arms = 1 if robot_type == "single_arm" else 2
    out = []
    for k in range(arms):
        pose, hand = x[..., 7 * k:7 * k + 7], x[..., 7 * arms + 16 * k:7 * arms + 16 * (k + 1)]
        rv = R.from_quat(pose[..., 3:7].reshape(-1, 4), scalar_first=True).as_rotvec().reshape(*pose.shape[:-1], 3)
        out += [pose[..., :3], rv, hand]
    return np.concatenate(out, -1).astype(np.float32)


def load_config(configs, task):
    import yaml
    return yaml.safe_load(open(f"{configs}/{task}.yaml"))


def read_task(root, task):
    """-> (info, episodes DataFrame sorted by episode_index, data DataFrame sorted by index, dataset task string)."""
    import pandas as pd
    d = f"{root}/{task}"
    info = json.load(open(f"{d}/meta/info.json"))
    assert info["codebase_version"] == "v3.0", info["codebase_version"]
    meta_eps = sorted(os.path.join(dp, f) for dp, _, fs in os.walk(f"{d}/meta/episodes") for f in fs if f.endswith(".parquet"))
    eps = pd.concat([pd.read_parquet(p) for p in meta_eps]).sort_values("episode_index").reset_index(drop=True)
    data_files = sorted(os.path.join(dp, f) for dp, _, fs in os.walk(f"{d}/data") for f in fs if f.endswith(".parquet"))
    cols = ["index", "episode_index", "frame_index", "observation.state", "action"]
    data = pd.concat([pd.read_parquet(p, columns=cols) for p in data_files]).sort_values("index").reset_index(drop=True)
    tasks = pd.read_parquet(f"{d}/meta/tasks.parquet")
    assert len(tasks) == 1, tasks
    assert len(eps) == info["total_episodes"] and len(data) == info["total_frames"], (task, len(eps), len(data))
    assert np.array_equal(data["index"].to_numpy(), np.arange(len(data))), task
    return info, eps, data, str(tasks.index[0])


def decode_file(args):
    """One video file -> {episode_index: [JPEG bytes per frame]} for the episodes stored in it (frames in decode order;
    episode e = frames round(from_timestamp * fps) .. + length)."""
    import av
    import cv2
    path, fps, spans = args  # spans: [(episode_index, from_timestamp, length)]
    start = {round(ft * fps): (e, L) for e, ft, L in spans}
    assert len(start) == len(spans), path
    want = sum(L for _, _, L in spans)
    out, cur, n = {e: [] for e, _, _ in spans}, None, 0
    with av.open(path) as c:
        s = c.streams.video[0]
        ctx = av.CodecContext.create("libdav1d", "r")
        if s.codec_context.extradata:
            ctx.extradata = s.codec_context.extradata
        ctx.thread_count = 4

        def take(fr):
            nonlocal cur, n
            if n in start:
                cur = start[n]
            if cur is not None and len(out[cur[0]]) < cur[1]:
                ok, b = cv2.imencode(".jpg", resize_with_pad(fr.to_ndarray(format="rgb24")), [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                assert ok
                out[cur[0]].append(b.tobytes())
            n += 1
        for pkt in c.demux(s):
            if pkt.size == 0:
                continue
            for fr in ctx.decode(pkt):
                take(fr)
        for fr in ctx.decode(None):
            take(fr)
    got = sum(len(v) for v in out.values())
    assert got == want and all(len(out[e]) == L for e, _, L in spans), (path, got, want, n)
    return out


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def build(root, out, robot_type, configs, tasks="ALL", workers=16, max_eps_per_task=0):
    """root = the LeRobot dataset dir of one variant; configs = <dexjoco checkout>/configs/<rand_obj|rand_full>."""
    t0 = time.time()
    assert robot_type in DIMS, robot_type
    Ds, Da = DIMS[robot_type]
    cams = DEXJOCO_VIDEO[robot_type]
    all_tasks = sorted(t for t in os.listdir(root) if os.path.exists(f"{root}/{t}/meta/info.json"))
    tasks = all_tasks if tasks == "ALL" else tasks.split(",")
    tasks = [t for t in tasks if load_config(configs, t)["robot_type"] == robot_type]
    assert tasks, f"no {robot_type} tasks under {root}"
    os.makedirs(out, exist_ok=False)
    bins = {c: open(f"{out}/jpeg_{c}.bin", "wb") for c in cams}
    offs = {c: [0] for c in cams}
    eps_out, instructions, cam_src, ds_tasks, max_err = [], [], {}, {}, 0.0
    cols = {k: [] for k in ("state", "action", "action_rotvec")}
    with get_context("spawn").Pool(workers) as pool:
        for ti, task in enumerate(tasks):
            cfg = load_config(configs, task)
            info, eps, data, ds_task = read_task(root, task)
            if max_eps_per_task:
                eps = eps.iloc[:max_eps_per_task]
            prompt = cfg["prompt"]
            instructions.append(prompt)
            fps = info["fps"]
            cam_src[task] = {c: cfg["camera_mapping"][c] for c in cams}
            # frames: per camera, every video file in parallel; a file holds consecutive episodes in episode order
            frames = {}
            for c in cams:
                vk = f"observation.images.{cam_src[task][c]}"
                assert vk in info["features"], (task, vk)
                g = eps.groupby([f"videos/{vk}/chunk_index", f"videos/{vk}/file_index"], sort=True)
                jobs = [(f"{root}/{task}/" + info["video_path"].format(video_key=vk, chunk_index=ci, file_index=fi), fps,
                         [(int(r.episode_index), float(r[f"videos/{vk}/from_timestamp"]), int(r.length)) for _, r in grp.iterrows()])
                        for (ci, fi), grp in g]
                frames[c] = {}
                for res in pool.imap(decode_file, jobs):
                    frames[c].update(res)
            st_all = np.stack(data["observation.state"].to_numpy()).astype(np.float32)
            ac_all = np.stack(data["action"].to_numpy()).astype(np.float32)
            assert st_all.shape[1] == Ds and ac_all.shape[1] == Da, (task, st_all.shape, ac_all.shape)
            for _, r in eps.iterrows():
                e, a, L = int(r.episode_index), int(r.dataset_from_index), int(r.length)
                assert int(r.dataset_to_index) - a == L
                rows = data.iloc[a:a + L]
                assert (rows["episode_index"].to_numpy() == e).all() and np.array_equal(rows["frame_index"].to_numpy(), np.arange(L))
                st, rv = st_all[a:a + L], ac_all[a:a + L]
                env_a, err = to_env_action(rv, st, robot_type)
                max_err = max(max_err, err)
                start = sum(x["length"] for x in eps_out)
                for c in cams:
                    fr = frames[c].pop(e)
                    assert len(fr) == L, (task, e, c, len(fr), L)
                    for b in fr:
                        bins[c].write(b)
                        offs[c].append(offs[c][-1] + len(b))
                cols["state"].append(st)
                cols["action"].append(env_a)
                cols["action_rotvec"].append(rv)
                eps_out.append({"task": task, "src_episode": e, "episode_index": len(eps_out), "task_index": ti,
                                "instruction_index": ti, "instruction": prompt, "start": start, "length": L})
            print(f"{task}: {len(eps)} episodes, {int(eps.length.sum())} frames, {time.time() - t0:.0f}s", flush=True)
            ds_tasks[task] = ds_task
    n = sum(e["length"] for e in eps_out)
    for c in cams:
        bins[c].close()
        o = np.asarray(offs[c], np.int64)
        assert len(o) == n + 1 and o[-1] == os.path.getsize(f"{out}/jpeg_{c}.bin")
        np.save(f"{out}/jpeg_{c}.offsets.npy", o)
    col = lambda key: np.concatenate([np.full(e["length"], e[key], np.int64) for e in eps_out])  # noqa: E731
    np.savez(f"{out}/table.npz", episode_index=col("episode_index"), task_index=col("task_index"),
             instruction_index=col("instruction_index"), src_episode=col("src_episode"),
             frame_index=np.concatenate([np.arange(e["length"], dtype=np.int64) for e in eps_out]),
             state=np.concatenate(cols["state"]), action=np.concatenate(cols["action"]),
             action_rotvec=np.concatenate(cols["action_rotvec"]))
    assert max_err < 1e-5, f"rotation vector -> quaternion round-trip error {max_err}"
    files = {f: {"sha256": sha_file(f"{out}/{f}"), "size": os.path.getsize(f"{out}/{f}")} for f in sorted(os.listdir(out))}
    json.dump({"schema": SCHEMA_V2, "frames": "jpeg", "benchmark": "dexjoco", "robot_type": robot_type,
               "variant": os.path.basename(os.path.normpath(configs)), "source_repo": "DexJoCo/DexJoCo-Datasets-LeRobot",
               "cameras": list(cams), "camera_sources": cam_src, "frame_shape": [SIZE, SIZE, 3],
               "decode": f"AV1 (libdav1d) -> openpi resize_with_pad {SIZE}x{SIZE} PIL bilinear -> cv2 JPEG q{JPEG_QUALITY}",
               "n": n, "state_dim": Ds, "action_dim": Ds, "layout": DEXJOCO_LAYOUT[robot_type],
               "rotation_roundtrip_max_rad": max_err, "instruction_mode": "per_episode",
               "tasks": {i: t for i, t in enumerate(tasks)}, "instructions": instructions, "dataset_tasks": ds_tasks,
               "max_eps_per_task": max_eps_per_task, "episodes": eps_out, "files": files},
              open(f"{out}/index.json", "w"), indent=1)
    print(f"DONE {n} samples, {len(eps_out)} episodes, {len(tasks)} tasks -> {out} in {time.time() - t0:.0f}s", flush=True)
