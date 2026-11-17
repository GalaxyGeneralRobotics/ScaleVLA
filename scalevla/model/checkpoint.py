"""The trained module (feature mapper + action head on cached features) and its checkpoint format, shared by the trainer
and the policy server.

<run>/checkpoint-<step>/policy.pt    {"mapper": state dict, "head": state dict}
<run>/checkpoint-<step>/policy.json  step, mapper kind, rank, d_in, normalization stats, embodiment_id, profile, cache spec
<run>/checkpoint-<step>/config.json  the action-head configuration the head was built from
"""
import json
import os
import shutil

import torch
from torch import nn

from .. import HEAD_CONFIG
from .head import build_head
from .mapper import MAPPER_KIND, FeatureMapper


class Policy(nn.Module):
    def __init__(self, mapper, head):
        super().__init__()
        self.mapper, self.head = mapper, head

    def forward(self, b):
        """Flow-matching loss of a collated batch (scalevla.data.dataset.collate)."""
        from transformers.feature_extraction_utils import BatchFeature
        bo = BatchFeature(data=self.mapper(b["features"], b["token_type"]))
        ai = BatchFeature(data={k: b[k] for k in ("state", "action", "action_mask", "embodiment_id")})
        return self.head(bo, ai)["loss"]


def _durable(path, mode, write):
    """Write to path.tmp, fsync, check, rename: a full disk quota surfaces at flush / close and must not leave an empty
    checkpoint behind."""
    with open(path + ".tmp", mode) as f:
        write(f)
        f.flush()
        os.fsync(f.fileno())
    if os.path.getsize(path + ".tmp") == 0:
        raise OSError(f"{path}: empty after write")
    os.replace(path + ".tmp", path)


def save_policy(ck, policy, meta, head_config=HEAD_CONFIG):
    os.makedirs(ck, exist_ok=True)
    _durable(f"{ck}/policy.pt", "wb", lambda f: torch.save({"mapper": policy.mapper.state_dict(), "head": policy.head.state_dict()}, f))
    _durable(f"{ck}/policy.json", "w", lambda f: json.dump({"mapper": MAPPER_KIND, **meta}, f, indent=1))
    shutil.copyfile(os.path.join(head_config, "config.json"), f"{ck}/config.json")


def load_policy(ck, horizon, head_config=None):
    """-> (Policy on cpu in eval mode, policy.json, head config). The head configuration comes from the checkpoint's own
    config.json, else from head_config (default storage/configs/action_head)."""
    with open(f"{ck}/policy.json") as f:
        pj = json.load(f)
    assert pj.get("mapper", MAPPER_KIND) == MAPPER_KIND, f"unsupported mapper {pj.get('mapper')}"
    config_dir = ck if os.path.exists(f"{ck}/config.json") else (head_config or HEAD_CONFIG)
    head, cfg = build_head(config_dir, horizon)
    mapper = FeatureMapper(pj["d_in"], cfg.backbone_embedding_dim, rank=pj["rank"])
    sd = torch.load(f"{ck}/policy.pt", map_location="cpu")
    mapper.load_state_dict(sd["mapper"])
    head.load_state_dict(sd["head"])
    return Policy(mapper, head).eval(), pj, cfg
