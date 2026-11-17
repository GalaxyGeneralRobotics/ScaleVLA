"""LIBERO evaluation on the official fixed initial states (GR00T LIBERO simulation environment).

    python -m scripts.eval_libero --suite libero_goal --episodes 50 --workers 5 --port 5555 --out <dir>

  - episodes = (task, init_state_id) for ids 0..N-1 of the fixed initial states of LIBERO (N = --episodes), so every
    policy sees exactly the same starts; every listed episode runs to termination
  - episode start = LIBERO's official procedure: reset, set_init_state(ids[k]), 5 zero-action physics steps (not counted)
  - control loop = GR00T's MultiStepWrapper (n_action_steps 8, max_episode_steps 720 env steps, terminate on success)
    around GR00T's LiberoEnv: the env success check and action conversion are unchanged
  - per episode: env steps, success, termination reason, policy calls, action finiteness and a heuristic failure
    category; the first and last agent-view frames are saved
Episodes are spread over --workers processes, each with its own env and PolicyClient. Output: <out>/episodes.jsonl,
<out>/summary.json (per-task success, macro success rate).
"""
import argparse
import json
import os
import time

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import numpy as np  # noqa: E402

STATE = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
ACT = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
FAILURES = ("static", "divergent", "no_grasp", "grasp_not_placed")


def characterise(ep):
    """Heuristic failure category of one episode from its (state [7], action chunk [H, 7]) calls."""
    S, A = np.stack([s for s, _ in ep]), np.stack([a for _, a in ep])
    path = float(np.linalg.norm(np.diff(S[:, :3], axis=0), axis=1).sum()) if len(S) > 1 else 0.0
    finite = bool(np.isfinite(A).all())
    out_ws = bool((np.abs(S[:, :2]) > 0.5).any() or (S[:, 2] < 0.0).any() or (S[:, 2] > 1.6).any())
    closed = bool((A[..., 6] < 0.5).any())
    cat = "divergent" if (not finite or out_ws) else "static" if path < 0.02 else "grasp_not_placed" if closed else "no_grasp"
    return {"calls": len(ep), "ee_path_m": round(path, 4), "finite": finite, "left_workspace": out_ws,
            "gripper_closed_ever": closed, "category": cat}


def batch_obs(obs):
    """What the official vector env sends: batch dim 1, state float32, language as a tuple."""
    out = {}
    for k, v in obs.items():
        if k.startswith("video"):
            out[k] = np.asarray(v, np.uint8)[None]
        elif k.startswith("state"):
            out[k] = np.asarray(v, np.float32)[None]
        else:
            out[k] = (str(v),)
    return out


def rollout(env, client, reset_options, max_steps):
    """One episode through a MultiStepWrapper env -> (success, termination, env steps, trace, first frame, last frame)."""
    obs, info = env.reset(options=reset_options)
    first = np.asarray(obs["video.image"][-1])
    # MultiStepWrapper.reset wraps info values in lists: bool([False]) would be True
    trace, success, reason = [], bool(np.any(info["success"])), None
    while True:
        ob = batch_obs(obs)
        act, _ = client.get_action(ob)
        a = {key: np.asarray(v)[0] for key, v in act.items()}
        trace.append((np.concatenate([ob[f"state.{s}"][0, -1] for s in STATE]), np.concatenate([a[f"action.{s}"] for s in ACT], -1)))
        obs, _, done, truncated, info = env.step(a)
        success |= bool(np.any(info["success"]))
        n = len(env.reward)
        if success:
            reason = "success"
        elif n >= max_steps:
            reason = "max_steps"
        elif done or truncated:
            reason = "env_done"
        if reason:
            return success, reason, n, trace, first, np.asarray(obs["video.image"][-1])


def init_states(suite, tid):
    """benchmark.get_task_init_states, loaded with weights_only=False (torch >= 2.6 refuses the pickled numpy arrays of the
    official LIBERO init files otherwise)."""
    import torch
    from libero.libero import get_libero_path
    t = suite.tasks[tid]
    return torch.load(os.path.join(get_libero_path("init_states"), t.problem_folder, t.init_states_file), weights_only=False)


def make_env(env_name, states, contract, max_steps):
    import gymnasium as gym
    from gr00t.eval.sim.wrapper.multistep_wrapper import MultiStepWrapper

    class FixedInitState(gym.Wrapper):
        """reset(options={"init_state_id": k}) -> LIBERO's official episode start from fixed initial state k."""

        def reset(self, seed=None, options=None):
            k = int(options["init_state_id"])
            self.env.reset(seed=k)
            le = self.env.unwrapped
            le._env.set_init_state(states[k])
            for _ in range(5):  # official LIBERO: zero-action physics settle
                raw, _, _, _ = le._env.step(np.zeros(7))
            return le._process_observation(raw), {"success": le._env.check_success()}

    return MultiStepWrapper(FixedInitState(gym.make(env_name)), contract=contract, max_episode_steps=max_steps,
                            terminate_on_success=True)


def worker(wid, jobs, port, suite_name, max_steps, n_action_steps, out_dir, results):
    import imageio
    from gr00t.eval._horizon_contract import PolicyHorizonSpec
    from gr00t.eval.sim.LIBERO.libero_env import register_libero_envs
    from gr00t.policy.server_client import PolicyClient
    from libero.libero import benchmark
    register_libero_envs()
    suite = benchmark.get_benchmark_dict()[suite_name]()
    client = PolicyClient(host="127.0.0.1", port=port)
    contract = PolicyHorizonSpec.from_policy(client, n_action_steps=n_action_steps)
    envs = {}
    for tid, k in jobs:
        task = suite.get_task(tid)
        if tid not in envs:
            envs[tid] = make_env(f"libero_sim/{task.name}", init_states(suite, tid), contract, max_steps)
        t0 = time.time()
        success, reason, n, trace, first, last = rollout(envs[tid], client, {"init_state_id": k}, max_steps)
        imageio.imwrite(f"{out_dir}/frames/{task.name}__init{k:02d}.png", np.concatenate([first, last], 1))
        r = {"task_id": tid, "task": task.name, "init_state_id": k, "env_steps": int(n), "success": success, "termination": reason,
             "policy_calls": len(trace), "s": round(time.time() - t0, 1), **characterise(trace)}
        if success:
            r["category"] = "success"
        results.put(r)
    for e in envs.values():
        e.close()
    results.put(("done", wid))


def main(argv=None):
    ap = argparse.ArgumentParser(description="LIBERO evaluation on the official fixed initial states")
    ap.add_argument("--suite", default="libero_spatial", help="libero_spatial | libero_object | libero_goal | libero_10")
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", type=int, default=50, help="initial states per task (ids 0..N-1)")
    ap.add_argument("--workers", type=int, default=5); ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--tasks", default="", help="comma task ids (default all)")
    ap.add_argument("--max-steps", type=int, default=720); ap.add_argument("--n-action-steps", type=int, default=8)
    a = ap.parse_args(argv)
    import multiprocessing as mp
    from libero.libero import benchmark
    suite = benchmark.get_benchmark_dict()[a.suite]()
    tids = [int(x) for x in a.tasks.split(",")] if a.tasks else list(range(suite.get_num_tasks()))
    jobs = [(t, k) for k in range(a.episodes) for t in tids]  # interleaved, so the workers share the load evenly
    os.makedirs(f"{a.out}/frames", exist_ok=True)
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=worker, args=(w, jobs[w::a.workers], a.port, a.suite, a.max_steps, a.n_action_steps, a.out, q))
             for w in range(a.workers)]
    for p in procs:
        p.start()
    eps, done, t0 = [], 0, time.time()
    with open(f"{a.out}/episodes.jsonl", "w") as log:
        while done < a.workers:
            r = q.get()
            if isinstance(r, tuple):
                done += 1
                continue
            eps.append(r)
            log.write(json.dumps(r) + "\n")
            log.flush()
            print(json.dumps({k: r[k] for k in ("task_id", "init_state_id", "env_steps", "success", "termination", "category")}), flush=True)
    for p in procs:
        p.join()
    missing = sorted(set(jobs) - {(r["task_id"], r["init_state_id"]) for r in eps})
    per = {}
    for t in tids:
        te = [r for r in eps if r["task_id"] == t]
        per[suite.get_task(t).name] = {"n": len(te), "successes": sum(r["success"] for r in te),
                                       "sr": float(np.mean([r["success"] for r in te])) if te else None,
                                       "failure_categories": {c: sum(r["category"] == c for r in te) for c in FAILURES}}
    summ = {"suite": a.suite, "episodes_per_task": a.episodes, "max_steps": a.max_steps, "n_action_steps": a.n_action_steps,
            "n_episodes": len(eps), "missing": missing, "complete": not missing, "per_task": per,
            "macro_sr": float(np.mean([v["sr"] for v in per.values()])) if not missing else None,
            "all_actions_finite": all(r["finite"] for r in eps), "wall_s": round(time.time() - t0, 1)}
    json.dump(summ, open(f"{a.out}/summary.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in summ.items() if k != "per_task"}, indent=1), flush=True)


if __name__ == "__main__":
    main()
