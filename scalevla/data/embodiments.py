"""Embodiment profiles: GR00T embodiment tag, state / action key -> column range of the sample-store vectors, action-chunk
horizon and multi-task sampling scheme. The trainer and the policy server both take the profile from here.

  libero    GR00T's libero_sim config                                              horizon 16, uniform sampling
  robocasa  GR00T's robocasa_panda_omron config; state 53 / action 12 columns       horizon 8, task-equal sampling
            from the dataset's meta/modality.json
  dexjoco   Franka + Allegro (DexJoCo), registered as NEW_EMBODIMENT; one profile    horizon 30 (the official evaluator's
            per robot type (single_arm / dual_arm), absolute actions in the env's    action horizon), task-equal sampling
            quaternion layout

NEW_EMBODIMENT is a single global slot of GR00T's registry, so a process holds one DexJoCo robot type at a time.
"""
from dataclasses import dataclass, field

from . import samples as S


@dataclass
class Profile:
    name: str
    tag: str
    ranges: dict
    horizon: int = 16
    sampling: str = "uniform"  # "uniform" (random permutation of chunk starts) | "task_equal" (every task the same share)
    video_keys: list = field(default_factory=list)

    def describe(self):  # recorded in run.json and policy.json
        return {"name": self.name, "tag": self.tag, "ranges": self.ranges, "horizon": self.horizon, "sampling": self.sampling}

    @property
    def state_dim(self):
        return sum(b - a for a, b in self.ranges["state"].values())

    @property
    def action_dim(self):
        return sum(b - a for a, b in self.ranges["action"].values())


# ---------------------------------------------------------------------------------------------------------------- LIBERO
LIBERO_TAG = "libero_sim"
LIBERO_RANGES = {
    "state": {"x": (0, 1), "y": (1, 2), "z": (2, 3), "roll": (3, 4), "pitch": (4, 5), "yaw": (5, 6), "gripper": (6, 8)},
    "action": {"x": (0, 1), "y": (1, 2), "z": (2, 3), "roll": (3, 4), "pitch": (4, 5), "yaw": (5, 6), "gripper": (6, 7)},
}


def libero():
    return Profile("libero", LIBERO_TAG, LIBERO_RANGES, horizon=16, sampling="uniform", video_keys=["image", "wrist_image"])


# -------------------------------------------------------------------------------------------------------------- RoboCasa
ROBOCASA_TAG = "robocasa_panda_omron"
ROBOCASA_STATE = {
    "base_position": (0, 3), "base_rotation": (3, 7), "end_effector_position_absolute": (7, 10),
    "end_effector_position_relative": (10, 13), "end_effector_rotation_absolute": (13, 17),
    "end_effector_rotation_relative": (17, 21), "gripper_qpos": (21, 23), "gripper_qvel": (23, 25),
    "joint_position": (25, 32), "joint_position_cos": (32, 39), "joint_position_sin": (39, 46), "joint_velocity": (46, 53)}
ROBOCASA_ACTION = {"base_motion": (0, 4), "control_mode": (4, 5), "end_effector_position": (5, 8),
                   "end_effector_rotation": (8, 11), "gripper_close": (11, 12)}


def robocasa():
    """Key order = GR00T's modality_keys order, not the store column order (gripper_close (11, 12) comes first). The head
    output is laid out in this key order; see scalevla.serving.layout. All actions are absolute."""
    from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
    mc = MODALITY_CONFIGS[ROBOCASA_TAG]
    ranges = {"state": {k: ROBOCASA_STATE[k] for k in mc["state"].modality_keys},
              "action": {k: ROBOCASA_ACTION[k] for k in mc["action"].modality_keys}}
    horizon = len(mc["action"].delta_indices)
    assert horizon == 8 and set(ranges["state"]) == set(ROBOCASA_STATE) and set(ranges["action"]) == set(ROBOCASA_ACTION)
    assert not any(str(getattr(c, "rep", "")).endswith("RELATIVE") for c in (mc["action"].action_configs or []))
    return Profile("robocasa", ROBOCASA_TAG, ranges, horizon=horizon, sampling="task_equal",
                   video_keys=list(mc["video"].modality_keys))


def check_robocasa_modality(modality):
    """modality = a store's index.json["modality"] (the dataset's meta/modality.json)."""
    for g, R in (("state", ROBOCASA_STATE), ("action", ROBOCASA_ACTION)):
        assert {k: (v["start"], v["end"]) for k, v in modality[g].items()} == R, g
    return True


# --------------------------------------------------------------------------------------------------------------- DexJoCo
# state  = the policy proprioception: the first 23 (single arm) / 46 (dual arm) dims of the env state.
# action = the env action layout, which equals the state layout. The dataset's rotation-vector actions are converted to
#          the env quaternion (scalar first) on the hemisphere of the observed state quaternion: the rotation vector wraps
#          at |angle| = pi and the end effector points down, the quaternion is continuous. The evaluator converts the
#          served quaternion back to a rotation vector (builders.dexjoco.from_env_action).
DEXJOCO_LAYOUT = {
    "single_arm": {"eef_pos": (0, 3), "eef_quat": (3, 7), "hand": (7, 23)},
    "dual_arm": {"right_eef_pos": (0, 3), "right_eef_quat": (3, 7), "left_eef_pos": (7, 10), "left_eef_quat": (10, 14),
                 "right_hand": (14, 30), "left_hand": (30, 46)},
}
DEXJOCO_VIDEO = {"single_arm": ["base", "wrist"], "dual_arm": ["base", "wrist_left", "wrist_right"]}  # evaluator camera keys
DEXJOCO_HORIZON = 30


def register_new_embodiment(cfg):
    """Register a layout as NEW_EMBODIMENT in GR00T's global registry (idempotent; a second registration in the same process
    must have the same layout)."""
    from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS, register_modality_config
    from gr00t.data.embodiment_tags import EmbodimentTag
    tag = EmbodimentTag.NEW_EMBODIMENT.value
    if tag not in MODALITY_CONFIGS:
        register_modality_config(cfg, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
    else:
        have = {m: (c.modality_keys, c.delta_indices) for m, c in MODALITY_CONFIGS[tag].items()}
        want = {m: (c.modality_keys, c.delta_indices) for m, c in cfg.items()}
        assert have == want, f"NEW_EMBODIMENT already registered with another layout: {have}"
    return tag


def dexjoco(robot_type, horizon=DEXJOCO_HORIZON):
    assert robot_type in DEXJOCO_LAYOUT, robot_type
    from gr00t.data.types import ModalityConfig
    keys = list(DEXJOCO_LAYOUT[robot_type])
    tag = register_new_embodiment({  # no action_configs = absolute actions
        "video": ModalityConfig(delta_indices=[0], modality_keys=list(DEXJOCO_VIDEO[robot_type])),
        "state": ModalityConfig(delta_indices=[0], modality_keys=keys),
        "action": ModalityConfig(delta_indices=list(range(horizon)), modality_keys=keys),
        "language": ModalityConfig(delta_indices=[0], modality_keys=["annotation.human.task_description"]),
    })
    layout = dict(DEXJOCO_LAYOUT[robot_type])
    return Profile(f"dexjoco_{robot_type}", tag, {"state": layout, "action": dict(layout)}, horizon=horizon,
                   sampling="task_equal", video_keys=list(DEXJOCO_VIDEO[robot_type]))


# ------------------------------------------------------------------------------------------------------------- dispatch
def from_name(name, horizon=None):
    """Profile from a checkpoint's profile name (policy.json); checkpoints without the field are LIBERO."""
    if name.startswith("dexjoco_"):
        robot_type = name[len("dexjoco_"):]
        if robot_type not in DEXJOCO_LAYOUT:
            raise ValueError(f"unsupported DexJoCo profile {name!r}")
        return dexjoco(robot_type, horizon or DEXJOCO_HORIZON)
    return {"libero": libero, "robocasa": robocasa}[name]()


def for_store(store, horizon=None):
    """Profile of a sample store from its index.json "benchmark" (default libero). horizon: DexJoCo only."""
    idx = S.load_index(store)
    benchmark = idx.get("benchmark", "libero")
    if benchmark == "dexjoco":
        return dexjoco(idx["robot_type"], horizon or DEXJOCO_HORIZON)
    if benchmark == "robocasa":
        check_robocasa_modality(idx["modality"])
        return robocasa()
    if benchmark != "libero":
        raise ValueError(f"{store}: unsupported benchmark {benchmark!r}")
    return libero()
