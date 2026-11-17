"""The global sample stream: every rank draws the same global sequence of chunk-start indices and keeps its slice of each
global batch (step s = seq[s * gbs:(s + 1) * gbs], rank r takes [r * per:(r + 1) * per], per = gbs / world).

  uniform     torch RandomSampler (no replacement within a pass, generator seeded with run.seed) over the chunk starts
  task_equal  torch.multinomial(w, block * gbs, replacement=True, generator seeded with run.seed), w = 1 / (n_tasks *
              chunk starts of the task): every task the same expected share of a batch
  blocks      data.stream_block B draws the stream in blocks of B steps from one generator, so the first B steps of a long
              run equal those of a B-step run
"""
import hashlib

import numpy as np
import torch

from ..data.dataset import task_equal_weights


def build_stream(ds, cfg, rank, world, start=0):
    """-> (this rank's per-step index lists for steps start + 1 .. run.steps, sampling record for run.json)."""
    gbs, seed, steps = cfg.run.gbs, cfg.run.seed, cfg.run.steps
    assert gbs % world == 0, f"run.gbs {gbs} is not divisible by the world size {world}"
    per = gbs // world
    g = torch.Generator().manual_seed(seed)
    blk = cfg.data.stream_block or steps
    n_blk = -(-steps // blk)
    if ds.profile.sampling == "task_equal":
        w, per_task = task_equal_weights(ds.task_ids())
        idx_all = [i for _ in range(n_blk) for i in torch.multinomial(w, blk * gbs, replacement=True, generator=g).tolist()][:steps * gbs]
        rec = {"scheme": "task_equal", "chunk_starts_per_task": per_task}
    else:
        sampler = torch.utils.data.RandomSampler(ds, replacement=False, generator=g, num_samples=blk * gbs)
        idx_all = [i for _ in range(n_blk) for i in sampler][:steps * gbs]
        rec = {"scheme": "uniform"}
    rec.update(seed=seed, gbs=gbs, world=world, per_rank=per, block_steps=blk,
               stream_sha256=hashlib.sha256(np.asarray(idx_all, dtype=np.int64).tobytes()).hexdigest())
    my = [idx_all[s * gbs + rank * per: s * gbs + (rank + 1) * per] for s in range(steps)]
    return my[start:], rec
