"""The action head: feature adapter, state and action encoders, a 32-layer action DiT and an action decoder, trained from
scratch with flow matching. The architecture is configured by storage/configs/action_head/config.json and implemented by
the Gr00tN1d7ActionHead of Isaac GR00T; no pretrained action weights are loaded. Every training run starts from the same
random initialization (init_seed)."""
import hashlib
import json
import os

import torch

from .. import HEAD_CONFIG


def build_head(config_dir=HEAD_CONFIG, horizon=0):
    """config_dir: a dir holding the action-head config.json (storage/configs/action_head or a checkpoint dir).
    horizon > the configured action_horizon (40) widens the head; no parameter shape depends on it."""
    from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
    from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7ActionHead
    cfg = Gr00tN1d7Config.from_pretrained(str(config_dir))
    if horizon > cfg.action_horizon:
        cfg.action_horizon = horizon
    return Gr00tN1d7ActionHead(cfg), cfg


def sha256_state(sd):
    h = hashlib.sha256()
    for k in sorted(sd):
        h.update(k.encode())
        h.update(sd[k].detach().cpu().contiguous().float().numpy().tobytes())
    return h.hexdigest()


def init_head_state(config_dir=HEAD_CONFIG, init_seed=0, horizon=0):
    """The common initialization: the head built right after torch.manual_seed(init_seed)."""
    torch.manual_seed(init_seed)
    head, _ = build_head(config_dir, horizon)
    return head.state_dict()


def ensure_head_init(path, config_dir=HEAD_CONFIG, init_seed=0, horizon=0):
    """Save the common initialization once, with a .json sidecar holding its sha256; an existing file is kept, so runs on
    different machines or software versions can share one initialization file."""
    if os.path.exists(path):
        return
    sd = init_head_state(config_dir, init_seed, horizon)
    torch.save(sd, path + ".tmp")
    os.replace(path + ".tmp", path)
    with open(path + ".json", "w") as f:
        json.dump({"init_seed": init_seed, "sha256": sha256_state(sd)}, f, indent=1)
