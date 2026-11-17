"""Training dataset over (sample store, feature cache): one item = one action-chunk start.

  - chunk starts: every step with a full action chunk inside its episode (GR00T's effective length L - horizon + 1); with
    pad_end every frame is a start and actions past the episode end repeat its last action (the mask stays 1)
  - state / action normalized by GR00T's StateActionProcessor (scalevla.data.stats) with statistics over the whole store;
    processor-level state dropout (the whole state vector is zeroed with probability state_dropout)
  - padding to MAX_DIM dims and max(MAX_H, horizon) steps; action_mask marks the valid block
  - demos_per_task N > 0: train on N demonstrations per task (select_demos); the normalization statistics stay those of
    the full store, so the subset only changes which chunks are drawn
"""
import random

import numpy as np
import torch

from . import cache as C
from . import embodiments as E
from . import samples as S
from .stats import make_processor

MAX_DIM, MAX_H = 132, 40  # padded state / action width and action horizon of the action head


def select_demos(episodes, n, seed):
    """N demonstrations per task: per task, its episodes sorted by episode_index, permuted with
    np.random.default_rng([seed, task_index]), first N. Subsets of one seed are nested (N = 5 within 10 within 25)."""
    by_task = {}
    for e in episodes:
        by_task.setdefault(int(e["task_index"]), []).append(e)
    keep = []
    for t, eps in sorted(by_task.items()):
        eps = sorted(eps, key=lambda e: int(e["episode_index"]))
        if len(eps) < n:
            raise ValueError(f"task {t} has {len(eps)} demonstrations, fewer than {n}")
        perm = np.random.default_rng([seed, t]).permutation(len(eps))
        keep += [eps[i] for i in perm[:n]]
    return sorted(keep, key=lambda e: e["start"])


def task_equal_weights(task_ids):
    """Per chunk start: 1 / (n_tasks * chunk starts of its task) -> every task has the same expected share of a batch,
    uniform over the chunk starts within a task."""
    tasks, counts = np.unique(task_ids, return_counts=True)
    w = 1.0 / (len(tasks) * counts[np.searchsorted(tasks, task_ids)])
    return torch.from_numpy(w), {int(k): int(c) for k, c in zip(tasks, counts)}


class CachedSteps(torch.utils.data.Dataset):
    def __init__(self, samples, cache, profile=None, horizon=None, state_dropout=0.2, pad_end=False, reader="mmap",
                 demos_per_task=0, demo_seed=42):
        self.idx = S.load_index(samples)
        self.tab = S.load_table(samples)
        self.cache_dir, self.cache, self.reader = cache, None, reader
        self.profile = profile or E.for_store(samples, horizon)
        self.h, self.state_dropout, self.pad_end = self.profile.horizon, state_dropout, pad_end
        assert horizon in (None, self.h), (horizon, self.h)
        self.max_h = max(MAX_H, self.h)
        self.proc, self.stats = make_processor(self.tab, self.profile)
        eps = self.idx["episodes"]
        self.demos = None
        if demos_per_task:
            eps = select_demos(eps, demos_per_task, demo_seed)
            self.demos = {"per_task": demos_per_task, "seed": demo_seed, "episodes": [int(e["episode_index"]) for e in eps]}
        self.valid = np.concatenate([np.arange(e["start"], e["start"] + e["length"] - (0 if pad_end else self.h - 1)) for e in eps])
        self.ep_end = np.zeros(len(self.tab["action"]), np.int64)
        for e in self.idx["episodes"]:
            self.ep_end[e["start"]:e["start"] + e["length"]] = e["start"] + e["length"]
        from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import EMBODIMENT_TAG_TO_PROJECTOR_INDEX
        self.emb_id = EMBODIMENT_TAG_TO_PROJECTOR_INDEX[self.profile.tag]  # the head's projector slot
        self.n_task_ids = len(self.idx["tasks"])

    def task_ids(self):
        """Task index of every chunk start."""
        return self.tab["task_index"][self.valid]

    def meta(self):
        """What run.json and policy.json record about the data (the policy server rebuilds the processor from it)."""
        out = {"stats": self.stats, "embodiment_id": self.emb_id, "profile": self.profile.describe()}
        return {**out, "demos": self.demos} if self.demos else out

    def __len__(self):
        return len(self.valid)

    def __getitem__(self, k):
        if self.cache is None:  # opened per worker process
            self.cache = C.open_cache(self.cache_dir, self.reader)
        i = int(self.valid[k])
        feats, tt = self.cache.ref(i) if self.reader == "pread" else self.cache.get(i)
        st = self.tab["state"][i:i + 1]
        steps = np.arange(i, i + self.h)
        ac = self.tab["action"][np.minimum(steps, self.ep_end[i] - 1) if self.pad_end else steps]
        R = self.profile.ranges
        ns, na = self.proc.apply(state={key: st[:, a:b] for key, (a, b) in R["state"].items()},
                                 action={key: ac[:, a:b] for key, (a, b) in R["action"].items()}, embodiment_tag=self.profile.tag)
        action = np.concatenate([na[key] for key in R["action"]], -1).astype(np.float32)  # head layout = key order
        if self.state_dropout > 0 and random.random() < self.state_dropout:
            state = np.zeros((1, self.profile.state_dim), np.float32)
        else:
            state = np.concatenate([ns[key] for key in R["state"]], -1).astype(np.float32)
        s = torch.zeros(1, MAX_DIM); s[:, :state.shape[1]] = torch.from_numpy(state)
        a = torch.zeros(self.max_h, MAX_DIM); a[:self.h, :action.shape[1]] = torch.from_numpy(action)
        m = torch.zeros(self.max_h, MAX_DIM); m[:self.h, :action.shape[1]] = 1
        return {"features": feats, "token_type": tt, "state": s, "action": a, "action_mask": m, "embodiment_id": self.emb_id}


def collate(batch):
    T = max(b["features"].shape[0] for b in batch)
    d = batch[0]["features"].shape[1]
    F = torch.zeros(len(batch), T, d, dtype=batch[0]["features"].dtype)
    TT = torch.full((len(batch), T), -1, dtype=torch.int8)  # -1 = padding
    for j, b in enumerate(batch):
        n = b["features"].shape[0]
        if isinstance(b["features"], C.FeatRef):
            b["features"].read_into(F[j, :n])
        else:
            F[j, :n] = b["features"]
        TT[j, :n] = b["token_type"]
    out = {k: torch.stack([b[k] for b in batch]) for k in ("state", "action", "action_mask")}
    out.update(features=F, token_type=TT, embodiment_id=torch.tensor([b["embodiment_id"] for b in batch], dtype=torch.long))
    return out
