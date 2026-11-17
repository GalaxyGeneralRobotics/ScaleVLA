"""DexJoCo evaluation with the official evaluator (DexJoCo environment with pyzmq and msgpack-numpy).

    python -m scripts.eval_dexjoco --dexjoco-root <checkout> --tasks dual_arm --variant rand_obj --seeds 0,1,2 \
        --episodes 50 --ports 5555 --out <dir>

The official evaluator runs unchanged: dexjoco_openpi_client.eval_dexjoco_openpi.main, i.e. env seeding, the asynchronous
inference process, chunk blending, replanning when fewer than replan_ratio * 30 actions are buffered, stay() while
nothing is buffered, success = the terminal "succeed" of the env, N episodes per (task, seed), per-episode videos.
--variant rand_obj and rand_full select the object and full randomization configs.
Only the transport is swapped: openpi_client's WebsocketClientPolicy -> Gr00tAsOpenPI, a client of the ScaleVLA policy
server:
  OpenPI observation {camera key: 224x224 uint8, "state": 23 / 46, "prompt"}
  -> GR00T sim observation (video.<key>, state.<key> of the DexJoCo profile layout, annotation.human.task_description)
  -> served chunk [30, 23 / 46] in the env layout -> rotation-vector layout (builders.dexjoco.from_env_action)
  -> {"actions": [30, 22 / 44]}
Each (task, seed) runs in its own process (one MuJoCo EGL context, one inference process) and writes the official
<out>/<task>_seed<S>/ tree, latency.txt (seconds per policy call) and one summary.jsonl line; summary.json holds the
per-task success over the seeds and the mean task success rate.
"""
import argparse
import glob
import json
import os
import socket
import subprocess
import sys
import time
import types
from pathlib import Path

import numpy as np

from scalevla.data.builders.dexjoco import from_env_action
from scalevla.data.embodiments import DEXJOCO_LAYOUT, DEXJOCO_VIDEO
from scalevla.serving.client import PolicyClient

LATENCY_ENV = "SCALEVLA_DEXJOCO_LATENCY"
REPO = str(Path(__file__).resolve().parents[1])


class Gr00tAsOpenPI:
    """Drop-in for WebsocketClientPolicy(host, port): same constructor and infer(obs) -> {"actions": [H, D_rotvec]}."""

    def __init__(self, host="127.0.0.1", port=5555, api_key=None):
        self.cl = PolicyClient(port, host=host)
        self.cl.call("reset", {"options": None})
        self.lat = os.environ.get(LATENCY_ENV)

    def get_server_metadata(self):
        return {"server": "scalevla"}

    def infer(self, obs):
        state = np.asarray(obs["state"], np.float32)
        rt = {23: "single_arm", 46: "dual_arm"}[state.shape[0]]
        g = {f"video.{c}": np.ascontiguousarray(obs[c], np.uint8)[None, None] for c in DEXJOCO_VIDEO[rt]}
        for k, (a, b) in DEXJOCO_LAYOUT[rt].items():
            g[f"state.{k}"] = state[a:b][None, None]
        g["annotation.human.task_description"] = [str(obs["prompt"])]
        t = time.time()
        act, _ = self.cl.call("get_action", {"observation": g, "options": None})
        dt = time.time() - t
        env = np.concatenate([np.asarray(act[f"action.{k}"], np.float32)[0] for k in DEXJOCO_LAYOUT[rt]], -1)
        if self.lat:
            with open(self.lat, "a") as f:
                f.write(f"{dt:.4f}\n")
        return {"actions": from_env_action(env, rt)}

    def reset(self):
        pass


def configs_dir(root, variant):
    return Path(root) / "configs" / variant


def tasks_of(spec, root, variant):
    import yaml
    names = sorted(Path(p).stem for p in glob.glob(str(configs_dir(root, variant) / "*.yaml")))
    if spec in ("ALL", "single_arm", "dual_arm"):
        return [t for t in names if spec == "ALL" or yaml.safe_load(open(configs_dir(root, variant) / f"{t}.yaml"))["robot_type"] == spec]
    out = spec.split(",")
    assert set(out) <= set(names), sorted(set(out) - set(names))
    return out


def job_dir(out, task, seed, variant):
    return Path(out) / f"{task}{'_rand_full' if variant == 'rand_full' else ''}_seed{seed}"


def run_one(task, seed, port, out, root, variant, episodes, replan_ratio):
    """One (task, seed) through the official main() in this process."""
    from dexjoco_openpi_client import eval_dexjoco_openpi as ev
    ev.websocket_client_policy = types.SimpleNamespace(WebsocketClientPolicy=Gr00tAsOpenPI)  # inherited by the forked worker
    stats = {"steps": 0, "stay": 0, "step_s": 0.0, "stale": 0}

    class CountedEnv(ev.DexJoCoOpenPIEnv):
        """Counts only (the official loop is unchanged): env steps, stay() steps and the wall time of env.step (the env paces
        itself to 30 Hz, so step_s / steps > 1/30 means the simulator ran slower than real time)."""

        def step(self, action):
            t = time.time()
            r = super().step(action)
            stats["steps"] += 1
            stats["step_s"] += time.time() - t
            return r

        def stay(self, continue_stay=False):
            stats["stay"] += 1
            return super().stay(continue_stay)
    ev.DexJoCoOpenPIEnv = CountedEnv
    orig_receive = ev.receive_actions

    def receive_actions(action_queue, actions_buffer, now_timestamp, dual_arm):
        """The official receive_actions, except that a chunk stamped later than now is dropped (and counted) instead of
        failing its assert. Such a chunk can only come from the previous episode: mp.Queue.put hands a chunk to a feeder
        thread, so under CPU load it can land after the official end-of-episode drain."""
        class Fresh:
            def get_nowait(self):
                while True:
                    c = action_queue.get_nowait()
                    if c.timestamp <= now_timestamp:
                        return c
                    stats["stale"] += 1
        return orig_receive(Fresh(), actions_buffer, now_timestamp, dual_arm)
    ev.receive_actions = receive_actions
    d = job_dir(out, task, seed, variant)
    d.mkdir(parents=True, exist_ok=True)
    os.environ[LATENCY_ENV] = str(d / "latency.txt")
    t0 = time.time()
    ev.main(config=configs_dir(root, variant) / f"{task}.yaml", seed=seed, rand_full=variant == "rand_full", port=port,
            host="127.0.0.1", output=d, episodes=episodes, replan_ratio=replan_ratio)
    rate = glob.glob(str(d / "success_rate_*_*.txt"))
    assert len(rate) == 1, rate
    k, n = map(int, Path(rate[0]).stem.split("_")[-2:])
    lat = np.loadtxt(d / "latency.txt", ndmin=1) if (d / "latency.txt").exists() else np.zeros(0)
    rec = {"task": task, "variant": variant, "seed": seed, "success": k, "episodes": n, "rate": k / n, "port": port,
           "host": socket.gethostname(), "s": round(time.time() - t0, 1), "calls": int(lat.size), "env_steps": stats["steps"],
           "stay_steps": stats["stay"], "stale_chunks": stats["stale"], "sim_hz": round(stats["steps"] / max(stats["step_s"], 1e-9), 2),
           "latency_median_ms": round(float(np.median(lat)) * 1e3, 1) if lat.size else None}
    with open(Path(out) / "summary.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    print("RESULT", json.dumps(rec), flush=True)


def set_aside(out, task, seed, variant, tag):
    """Move the partial output of a (task, seed) job to <out>/crashed/<name>.<tag>, so that a rerun starts from nothing."""
    (Path(out) / "crashed").mkdir(exist_ok=True)
    for x in (job_dir(out, task, seed, variant), Path(out) / f"{job_dir(out, task, seed, variant).name}.log"):
        if x.exists():
            x.rename(Path(out) / "crashed" / f"{x.name}.{tag}")


def summarize(out):
    by, path = {}, Path(out) / "summary.jsonl"
    for r in map(json.loads, open(path) if path.exists() else []):
        by.setdefault((r["variant"], r["task"]), []).append(r)
    rows = {f"{v}/{t}": {"success": sum(r["success"] for r in rs), "episodes": sum(r["episodes"] for r in rs),
                         "seeds": sorted(r["seed"] for r in rs)} for (v, t), rs in sorted(by.items())}
    for r in rows.values():
        r["rate"] = round(r["success"] / r["episodes"], 4)
    mean = {v: round(float(np.mean([r["rate"] for k, r in rows.items() if k.startswith(v + "/")])), 4) for v in sorted({v for v, _ in by})}
    json.dump({"tasks": rows, "mean_task_rate": mean}, open(Path(out) / "summary.json", "w"), indent=1)
    return rows, mean


def main(argv=None):
    ap = argparse.ArgumentParser(description="DexJoCo evaluation with the official evaluator")
    ap.add_argument("--dexjoco-root", default=os.environ.get("DEXJOCO_ROOT"), help="DexJoCo checkout (env DEXJOCO_ROOT)")
    ap.add_argument("--tasks", default="dual_arm", help="comma task names | single_arm | dual_arm | ALL")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--variant", choices=["rand_obj", "rand_full"], default="rand_obj")
    ap.add_argument("--ports", default="5555", help="job j talks to ports[j % n] (one policy server per concurrent job)")
    ap.add_argument("--workers", type=int, default=0, help="concurrent (task, seed) jobs (default: one per port)")
    ap.add_argument("--render-gpus", default="", help="GPU per port slot for MuJoCo EGL rendering (same length as --ports)")
    ap.add_argument("--episodes", type=int, default=50); ap.add_argument("--replan-ratio", type=float, default=0.8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--retries", type=int, default=2, help="rerun a crashed (task, seed) job this many times")
    ap.add_argument("--skip-done", action="store_true", help="resume: skip the (task, seed) jobs that have a summary.jsonl line")
    ap.add_argument("--job", default="", help=argparse.SUPPRESS)  # internal: task:seed:port
    a = ap.parse_args(argv)
    assert a.dexjoco_root, "--dexjoco-root (or DEXJOCO_ROOT) is required"
    if a.job:
        task, seed, port = a.job.split(":")
        run_one(task, int(seed), int(port), a.out, a.dexjoco_root, a.variant, a.episodes, a.replan_ratio)
        return
    ports = [int(p) for p in a.ports.split(",")]
    rgpu = a.render_gpus.split(",") if a.render_gpus else []
    assert not rgpu or len(rgpu) == len(ports), "--render-gpus needs one GPU per port"
    workers = a.workers or len(ports)
    jobs = [(t, int(s)) for s in a.seeds.split(",") for t in tasks_of(a.tasks, a.dexjoco_root, a.variant)]
    os.makedirs(a.out, exist_ok=True)
    if a.skip_done and (Path(a.out) / "summary.jsonl").exists():
        done = {(r["task"], r["seed"]) for r in map(json.loads, open(Path(a.out) / "summary.jsonl"))}
        jobs = [j for j in jobs if j not in done]
        for t, s in jobs:  # unfinished jobs of the interrupted run
            set_aside(a.out, t, s, a.variant, f"interrupted{int(time.time())}")
        print(f"resume: {len(done)} jobs done, {len(jobs)} to run", flush=True)
    running, free, failed, tries = [], list(range(workers)), [], {}
    while jobs or running:
        while jobs and free:
            w = free.pop(0)
            t, s = jobs.pop(0)
            log = open(Path(a.out) / f"{job_dir(a.out, t, s, a.variant).name}.log", "w")
            cmd = [sys.executable, "-m", "scripts.eval_dexjoco", "--job", f"{t}:{s}:{ports[w % len(ports)]}", "--out", a.out,
                   "--dexjoco-root", a.dexjoco_root, "--variant", a.variant, "--episodes", str(a.episodes),
                   "--replan-ratio", str(a.replan_ratio)]
            env = dict(os.environ)
            env["PYTHONPATH"] = os.pathsep.join(x for x in (REPO, env.get("PYTHONPATH")) if x)  # the job runs `-m scripts...`
            if rgpu:  # EGL enumerates physical GPUs and ignores CUDA_VISIBLE_DEVICES: the id must be the physical index
                env.update(CUDA_VISIBLE_DEVICES=rgpu[w % len(ports)], MUJOCO_EGL_DEVICE_ID=rgpu[w % len(ports)])
            running.append((subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env), w, t, s))
            print(f"start {t} seed {s} on port {ports[w % len(ports)]}", flush=True)
        time.sleep(5)
        for j in list(running):
            if j[0].poll() is None:
                continue
            running.remove(j)
            free.append(j[1])
            t, s, rc = j[2], j[3], j[0].returncode
            print(f"done {t} seed {s} rc {rc}", flush=True)
            if rc:
                tries[(t, s)] = tries.get((t, s), 0) + 1
                set_aside(a.out, t, s, a.variant, f"try{tries[(t, s)]}")
                if tries[(t, s)] <= a.retries:
                    jobs.append((t, s))
                    print(f"requeue {t} seed {s} (crash {tries[(t, s)]})", flush=True)
                else:
                    failed.append((t, s, rc))
    rows, mean = summarize(a.out)
    for k, r in rows.items():
        print(f"{k:40s} {r['success']:4d}/{r['episodes']:<4d} {100 * r['rate']:5.1f}%")
    print("mean task success rate:", {v: f"{100 * m:.1f}%" for v, m in mean.items()}, "| failed jobs:", failed, flush=True)
    if failed:  # an evaluation with missing jobs is not done (rerun with --skip-done)
        sys.exit(1)


if __name__ == "__main__":
    main()
