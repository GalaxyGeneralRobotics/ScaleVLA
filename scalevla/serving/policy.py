"""The ScaleVLA policy for closed-loop rollouts (GR00T environment), served with the GR00T PolicyServer and
Gr00tSimPolicyWrapper so that the rollout clients of each benchmark connect directly. One class for every benchmark; the
embodiment comes from the checkpoint.

observation -> action, mirroring training exactly:
  camera frames (modality-config order == sample-store camera order) + instruction -> live feature server (the request
  path of feature extraction) -> feature mapper -> action head (get_action: flow matching, bf16 autocast) -> first
  `horizon` steps, split per key with the trainer's layout (scalevla.serving.layout) -> StateActionProcessor.unapply with
  the checkpoint's statistics.
The state is normalized by the same processor (eval mode: no state dropout). options["noise_seed"] fixes the
flow-matching noise. The intermediates of the last call are kept in .last (offline checks).
"""
import json

import numpy as np
import torch

from ..data import embodiments as E
from ..data.dataset import MAX_DIM
from ..data.stats import processor_from_stats
from ..features.client import FeatureClient
from ..model.checkpoint import load_policy
from .layout import head_offsets, self_test


def profile_from_checkpoint(pj):
    """Checkpoints without a profile record are LIBERO."""
    pr = pj.get("profile", {})
    prof = E.from_name(pr.get("name", "libero"), int(pr["horizon"]) if "horizon" in pr else None)
    if pr:
        assert json.loads(json.dumps(prof.describe())) == pr, (prof.describe(), pr)  # compared after the JSON round trip
    return prof


class GraphedHead:
    """mapper + head.get_action for one observation as one CUDA graph per token length T (static inputs, one shared memory
    pool). The kernels are those of the eager path, so actions are bitwise equal. Capture does not consume the caller's
    RNG stream."""

    def __init__(self, run, device):
        self.run, self.device, self.graphs, self.pool = run, device, {}, None

    def _capture(self, F, TT, S):
        st = {"F": F.to(self.device).clone(), "TT": TT.to(self.device).clone(), "S": S.to(self.device).clone()}
        cur = torch.cuda.current_stream(self.device)
        side = torch.cuda.Stream(self.device)
        side.wait_stream(cur)
        rng = torch.cuda.get_rng_state(self.device)
        with torch.cuda.stream(side):
            for _ in range(3):  # warm-up outside the graph (allocator, lazy initialization)
                self.run(st["F"], st["TT"], st["S"])
        cur.wait_stream(side)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=self.pool):
            out = self.run(st["F"], st["TT"], st["S"])
        if self.pool is None:
            self.pool = g.pool()
        torch.cuda.set_rng_state(rng, self.device)
        return {"graph": g, "st": st, "out": out}

    @torch.inference_mode()
    def __call__(self, F, TT, S):
        e = self.graphs.get(F.shape[1])
        if e is None:
            e = self.graphs[F.shape[1]] = self._capture(F, TT, S)
        e["st"]["F"].copy_(F)
        e["st"]["TT"].copy_(TT)
        e["st"]["S"].copy_(S)
        e["graph"].replay()
        return {k: v.clone() for k, v in e["out"].items()}


def _gr00t_policy_base():
    from gr00t.policy.gr00t_policy import Gr00tPolicy
    return Gr00tPolicy


class CachedHeadPolicy(_gr00t_policy_base()):
    """Gr00tPolicy interface (observation checks, unbatching and action checks are inherited unchanged)."""

    def __init__(self, ckpt, feature_port=5601, device="cuda:0", strict=True, cuda_graph=False, flow_steps=0, features=None,
                 head_config=None):
        from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
        from gr00t.data.embodiment_tags import EmbodimentTag
        from gr00t.policy.policy import BasePolicy
        BasePolicy.__init__(self, strict=strict)
        with open(f"{ckpt}/policy.json") as f:
            self.profile = profile_from_checkpoint(json.load(f))
        net, self.pj, _ = load_policy(ckpt, self.profile.horizon, head_config)
        self.device = torch.device(device)
        self.net = net.to(self.device)
        if flow_steps:
            self.net.head.num_inference_timesteps = flow_steps
        self.flow_steps = self.net.head.num_inference_timesteps
        self.tag = self.profile.tag
        assert self_test(self.profile.ranges["action"]), self.profile.ranges["action"]
        self.ranges = {"state": self.profile.ranges["state"], "action": head_offsets(self.profile.ranges["action"])}
        self.proc = processor_from_stats(self.pj["stats"], self.profile, train=False)
        self.embodiment_tag = EmbodimentTag.resolve(self.tag)
        self.modality_configs = {k: v for k, v in MODALITY_CONFIGS[self.tag].items() if k != "rl_info"}
        self.language_key = self.modality_configs["language"].modality_keys[0]
        self.cams = list(self.modality_configs["video"].modality_keys)
        self.horizon = len(self.modality_configs["action"].delta_indices)
        assert self.horizon == self.profile.horizon, (self.horizon, self.profile.horizon)
        self.emb_id = self.pj["embodiment_id"]
        self.ds = features or FeatureClient(feature_port)
        self.graph = GraphedHead(self._run, self.device) if cuda_graph else None
        self.last = None

    def describe(self):
        return {"profile": self.profile.describe(), "head_offsets": self.ranges["action"], "cams": self.cams,
                "language_key": self.language_key, "horizon": self.horizon, "flow_steps": self.flow_steps,
                "cuda_graph": self.graph is not None, "embodiment_id": self.emb_id}

    def _run(self, F, TT, S):
        """mapper + head.get_action under bf16 autocast (the eager path; also what the CUDA graph captures)."""
        from transformers.feature_extraction_utils import BatchFeature
        with torch.autocast("cuda", dtype=torch.bfloat16):
            bo = self.net.mapper(F, TT)
            ai = BatchFeature(data={"state": S, "embodiment_id": torch.full((F.shape[0],), self.emb_id, dtype=torch.long, device=self.device)})
            pred = self.net.head.get_action(BatchFeature(data=bo), ai)["action_pred"].float()
        return {"pred": pred, "backbone_attention_mask": bo["backbone_attention_mask"], "image_mask": bo["image_mask"]}

    def _get_action(self, observation, options=None):
        obs = self._unbatch_observation(observation)
        frames = [[np.asarray(o["video"][c][-1]) for c in self.cams] for o in obs]
        texts = [str(o["language"][self.language_key][0]) for o in obs]
        H = self.ds.features(frames, texts)
        B, T = len(obs), max(h[0].shape[0] for h in H)
        F = torch.zeros(B, T, H[0][0].shape[1], dtype=torch.bfloat16)
        TT = torch.full((B, T), -1, dtype=torch.int8)
        S = torch.zeros(B, 1, MAX_DIM)
        for b, ((f, tt, _), o) in enumerate(zip(H, obs)):
            F[b, :f.shape[0]], TT[b, :f.shape[0]] = f, tt
            ns = self.proc.apply_state({k: np.asarray(o["state"][k], np.float32) for k in self.ranges["state"]}, self.tag)
            st = np.concatenate([ns[k] for k in self.ranges["state"]], -1)[-1]
            S[b, 0, :st.shape[0]] = torch.from_numpy(st.astype(np.float32))
        if options and options.get("noise_seed") is not None:
            torch.manual_seed(int(options["noise_seed"]))
        if self.graph is not None and B == 1:
            out = self.graph(F, TT, S)
        else:
            with torch.inference_mode():
                out = self._run(F.to(self.device), TT.to(self.device), S.to(self.device))
        pred = out["pred"]
        na = pred[:, :self.horizon].cpu().numpy()
        un = self.proc.unapply_action({k: na[..., a:b] for k, (a, b) in self.ranges["action"].items()}, self.tag)
        self.last = {"features": [h[0] for h in H], "token_type": [h[1] for h in H], "prompt_token_ids": [h[2] for h in H],
                     "instructions": texts, "state_norm": S, "action_pred": pred.cpu(),
                     "backbone_attention_mask": out["backbone_attention_mask"].cpu(), "image_mask": out["image_mask"].cpu()}
        res = {k: v.astype(np.float32) for k, v in un.items()}
        return res, {}

    def reset(self, options=None):
        return {}
