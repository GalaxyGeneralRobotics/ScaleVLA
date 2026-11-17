"""RoboCasa evaluation on registered initial states (GR00T RoboCasa simulation environment).

    python -m scripts.eval_robocasa --tasks ALL --episodes 50 --ports 5555,5556,5557 --workers 15 --out <dir>

storage/robocasa_initial_states/<Task>.json lists, for each of the 24 tasks, 50 initial states in unseen kitchens (10 in
each of 5 evaluation scenes) by id, with the sha256 of the realised simulator state.
Differences from GR00T's rollout_policy.py, which seeds only the first reset and counts whichever episodes finish first:
  - every episode starts from env.reset(seed=<id>) with RoboCasa's env.rng reseeded by the same id, so every policy sees
    exactly the same layouts, objects and textures; the realised state is checked against the registry
  - invalid or failed episodes are recorded with their reason, never dropped; a rerun resumes per episode
Environment construction, wrappers (MultiStepWrapper n_action_steps 8, 720 env steps), the policy contract and the success
rule are GR00T's, imported from gr00t.eval.rollout_policy.
Output: <out>/<Task>/episodes.jsonl per task and <out>/summary.json (per-task success, macro success rate).
"""
import argparse
import glob
import hashlib
import json
import multiprocessing as mp
import os
import time
import traceback
from functools import partial
from pathlib import Path

import numpy as np

from scalevla import STORAGE

REGISTRY = str(STORAGE / "robocasa_initial_states")
TERMINATIONS = ("success", "max_steps_or_failure")


def _any(v):
    return bool(np.any(v)) if isinstance(v, (list, tuple, np.ndarray)) else bool(v)


def _base_env(env):
    base = env.envs[0].unwrapped
    return getattr(base, "env", base)  # gymnasium adapter -> robosuite / RoboCasa env


def initial_state_record(env):
    base, rec = _base_env(env), {}
    try:
        rec["sim_state_sha256"] = hashlib.sha256(np.asarray(base.sim.get_state().flatten(), dtype=np.float64).tobytes()).hexdigest()
    except Exception as e:  # keep evaluating; record why the hash is missing
        rec["sim_state_sha256"], rec["sim_state_error"] = None, repr(e)
    try:
        m = base.get_ep_meta()
        rec.update(layout_id=m.get("layout_id"), style_id=m.get("style_id"), lang=m.get("lang"))
    except Exception as e:
        rec["ep_meta_error"] = repr(e)
    return rec


def worker(wid, ids, a, port, q):
    import gymnasium as gym
    from gr00t.eval import rollout_policy as rp
    from gr00t.policy.server_client import PolicyClient
    policy = PolicyClient(host="127.0.0.1", port=port)
    contract = rp.PolicyHorizonSpec.from_policy(policy, n_action_steps=a.n_action_steps)
    video_dir = os.path.join(a.task_out, "videos", f"w{wid}") if a.video else None
    cfg = rp.WrapperConfigs(multistep=rp.MultiStepConfig(contract=contract, max_episode_steps=a.max_episode_steps, terminate_on_success=True),
                            video=rp.VideoConfig(video_dir=video_dir, max_episode_steps=a.max_episode_steps))
    env = gym.vector.SyncVectorEnv([partial(rp.create_eval_env, env_name=a.env_name, env_idx=wid, total_n_envs=a.workers, wrapper_configs=cfg)])
    try:
        for sid in ids:
            row, t0 = {"env_name": a.env_name, "initial_state_id": sid, "worker": wid}, time.time()
            try:
                # RoboCasa samples the scene from its own env.rng (gym reset(seed) does not reach it): reseeding it with
                # the id reproduces layout, objects, textures and cameras exactly
                _base_env(env).rng = np.random.default_rng(sid)
                obs, _ = env.reset(seed=[sid])
                row["initial_state"] = initial_state_record(env)
                ref = a.registry.get(sid)
                if ref is not None:
                    row["initial_state_matches_registry"] = ref.get("sim_state_sha256") == row["initial_state"].get("sim_state_sha256")
                policy.reset()
                success, length, lat, valid, done, steps = False, 0, [], True, False, 0
                while not done:
                    t1 = time.perf_counter()
                    actions, _ = policy.get_action(obs)
                    lat.append(time.perf_counter() - t1)
                    obs, _, term, trunc, info = env.step(actions)
                    steps += 1
                    if "success" in info:
                        success |= _any(info["success"][0])
                    fi = info.get("final_info")
                    if fi is not None and fi[0] is not None:
                        success |= _any(fi[0].get("success", False))
                    length += rp._macro_step_env_steps(info, 0)
                    done = bool(term[0] or trunc[0])
                    if done:  # GR00T's rule: an episode is valid iff all its "valid" flags are true
                        if "valid" in info and info["valid"][0] is not None:
                            valid = bool(np.all(info["valid"][0]))
                        elif fi is not None and fi[0] is not None and "valid" in fi[0]:
                            valid = bool(np.all(fi[0]["valid"]))
                row.update(success=bool(success), valid=bool(valid), env_steps=int(length), policy_calls=steps,
                           termination="success" if success else ("invalid" if not valid else "max_steps_or_failure"),
                           latency_median_s=float(np.median(lat)) if lat else None)
            except Exception as e:
                row.update(success=False, valid=False, termination="error", error=repr(e), traceback=traceback.format_exc())
            row["wall_s"] = round(time.time() - t0, 1)
            q.put(row)
    finally:
        try:
            env.close()
        except Exception:
            pass
        q.put(("done", wid))


def evaluate_task(task, a, ports):
    """Run the first a.episodes registered initial states of one task -> per-task record."""
    reg = json.load(open(os.path.join(a.registry_dir, f"{task}.json")))
    ids = [int(x["id"]) for x in reg["initial_states"]][:a.episodes]
    a.env_name, a.task_out = reg["env_name"], os.path.join(a.out, task)
    a.registry = {int(x["id"]): x for x in reg["initial_states"]}
    os.makedirs(a.task_out, exist_ok=True)
    path = os.path.join(a.task_out, "episodes.jsonl")
    rows = [json.loads(l) for l in open(path) if l.strip()] if os.path.exists(path) else []
    done_ids = {r["initial_state_id"] for r in rows}
    todo = [i for i in ids if i not in done_ids]
    print(f"{task}: {len(ids)} initial states, {len(done_ids)} done earlier, {len(todo)} to run", flush=True)
    if todo:
        ctx = mp.get_context("spawn")
        q = ctx.Queue()
        procs = [ctx.Process(target=worker, args=(w, todo[w::a.workers], a, ports[w % len(ports)], q))
                 for w in range(a.workers) if todo[w::a.workers]]
        for p in procs:
            p.start()
        live = len(procs)
        with open(path, "a") as f:
            while live:
                row = q.get()
                if isinstance(row, tuple):
                    live -= 1
                    continue
                f.write(json.dumps(row) + "\n")
                f.flush()
                rows.append(row)
                print(f"{task} id={row['initial_state_id']} success={row['success']} {row['termination']} steps={row.get('env_steps')}", flush=True)
        for p in procs:
            p.join()
    rows = [r for r in rows if r["initial_state_id"] in set(ids)]
    bad = [(r["initial_state_id"], r["termination"]) for r in rows
           if r["termination"] not in TERMINATIONS or r.get("initial_state_matches_registry") is not True]
    return {"n": len(rows), "successes": sum(bool(r["success"]) for r in rows), "sr": sum(bool(r["success"]) for r in rows) / len(ids),
            "complete": len({r["initial_state_id"] for r in rows}) == len(ids), "bad": bad}


def main(argv=None):
    ap = argparse.ArgumentParser(description="RoboCasa evaluation on registered initial states")
    ap.add_argument("--tasks", default="ALL", help="comma task names (e.g. OpenDrawer) or ALL (every registry file)")
    ap.add_argument("--registry-dir", default=REGISTRY)
    ap.add_argument("--episodes", type=int, default=50, help="first N registered initial states per task")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ports", default="5555", help="comma policy-server ports; env worker w uses ports[w % n]")
    ap.add_argument("--workers", type=int, default=5, help="env workers per task")
    ap.add_argument("--n-action-steps", type=int, default=8); ap.add_argument("--max-episode-steps", type=int, default=720)
    ap.add_argument("--video", action="store_true")
    a = ap.parse_args(argv)
    names = sorted(Path(p).stem for p in glob.glob(os.path.join(a.registry_dir, "*.json")))
    tasks = names if a.tasks == "ALL" else a.tasks.split(",")
    assert set(tasks) <= set(names), f"no registry for {sorted(set(tasks) - set(names))}"
    ports = [int(p) for p in a.ports.split(",")]
    per = {t: evaluate_task(t, a, ports) for t in tasks}
    valid = all(v["complete"] and not v["bad"] for v in per.values())
    summ = {"tasks": per, "episodes_per_task": a.episodes, "n_action_steps": a.n_action_steps, "max_episode_steps": a.max_episode_steps,
            "valid": valid, "macro_sr": float(np.mean([v["sr"] for v in per.values()]))}
    json.dump(summ, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    print(f"{'VALID' if valid else 'INVALID'}: macro success rate {100 * summ['macro_sr']:.1f}% over {len(tasks)} tasks", flush=True)


if __name__ == "__main__":
    main()
