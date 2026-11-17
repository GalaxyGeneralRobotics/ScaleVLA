"""Build a sample store: the frames, states, actions and instructions that feature extraction and training read.

    python -m scripts.prepare_data libero   <suite_dir> <out>
    python -m scripts.prepare_data robocasa <dataset_root> <out> --demos-per-task 30|100 [--seed 42] [--tasks a,b]
    python -m scripts.prepare_data dexjoco  <lerobot_root> <out> --robot-type single_arm|dual_arm --configs <dexjoco>/configs/<variant>
"""
import argparse


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="benchmark", required=True)
    p = sub.add_parser("libero", help="one suite of IPEC-COMMUNITY/libero_<suite>_no_noops_1.0.0_lerobot")
    p.add_argument("suite_dir")
    p.add_argument("out")
    p = sub.add_parser("robocasa", help="the single_panda_gripper.* sets of GR00T X-Embodiment-Sim")
    p.add_argument("data")
    p.add_argument("out")
    p.add_argument("--demos-per-task", type=int, default=0, help="N demonstrations per task (0 = every valid one)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tasks", default="")
    p.add_argument("--workers", type=int, default=16)
    p = sub.add_parser("dexjoco", help="one variant and robot type of DexJoCo/DexJoCo-Datasets-LeRobot")
    p.add_argument("root")
    p.add_argument("out")
    p.add_argument("--robot-type", required=True, choices=["single_arm", "dual_arm"])
    p.add_argument("--configs", required=True, help="the evaluator configs: <dexjoco checkout>/configs/<rand_obj|rand_full>")
    p.add_argument("--tasks", default="ALL")
    p.add_argument("--max-eps-per-task", type=int, default=0)
    p.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    if a.benchmark == "libero":
        from scalevla.data.builders.libero import build
        build(a.suite_dir, a.out)
    elif a.benchmark == "robocasa":
        from scalevla.data.builders.robocasa import build
        build(a.data, a.out, a.demos_per_task, a.seed, a.tasks, a.workers)
    else:
        from scalevla.data.builders.dexjoco import build
        build(a.root, a.out, a.robot_type, a.configs, a.tasks, a.workers, a.max_eps_per_task)


if __name__ == "__main__":
    main()
