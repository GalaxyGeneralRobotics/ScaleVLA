"""DeepSeek-V4.1-Flash backbone: the input of layers[20].attn through the canonical vLLM engine (TP 4, expert parallel).
Every observation becomes feature rows: features [L, 5120] bf16, token_type [L] int8 (0 text, 1 + c image tokens of
camera c) and the prompt token ids.

Request path (feature extraction and live server alike): engine.build_request (images in store camera order, then the
instruction) -> llm.generate(max_tokens=1) -> rows recovered from vLLM's batch layout (worker.arm_layout / take /
write_shard). Batch-invariant kernels make the features independent of batching and chunking, so the writer may batch
(batch=64) while the live server serves whatever the rollout sends. Memory-only engine overrides for rollouts
(gpu_memory_utilization, max_num_batched_tokens) do not change the features; they are recorded as live_overrides.
"""
import os
from functools import partial

import numpy as np

from ..data.cache import token_types
from . import engine
from .worker import arm_layout, disarm_layout, take, write_shard


def check_labels(tt, n_cams):
    """Every sample: exactly one contiguous image-token run per camera, cameras in order, text elsewhere."""
    runs = [int(c) for c, prev in zip(tt, np.concatenate([[0], tt[:-1]])) if c > 0 and c != prev]
    if runs != list(range(1, n_cams + 1)):
        raise RuntimeError(f"image-token runs {runs}, expected cameras 1..{n_cams} in order")
    return tt


class DeepSeekBackbone:
    name = "deepseek_v41_flash"

    def __init__(self, model, batch=64, encoding_dir=None, **overrides):
        """model: the vLLM model dir of DeepSeek-V4.1-Flash; encoding_dir: the model's official chat encoding
        (default <model>/encoding)."""
        from vllm import SamplingParams
        self.sp = SamplingParams(temperature=0.0, max_tokens=1)
        self.batch = batch
        self.encoding_dir = str(encoding_dir or os.path.join(str(model), "encoding"))
        self.llm = engine.make_llm(model, **overrides)
        self.spec = {**engine.engine_spec(model), "prompt": "official encode_messages chat; images (store camera order) then instruction",
                     "dtype": "bfloat16", "d": engine.FEATURE_DIM}
        self.live_overrides = overrides

    def _request(self, frames, instruction):
        return engine.build_request(frames, instruction, self.encoding_dir)

    def _arm(self):
        if not any(r["recording"] for r in self.llm.collective_rpc(arm_layout)):
            raise RuntimeError("no rank records the layout")

    def features(self, items):
        n_cams = {len(im) for im, _ in items}
        if len(n_cams) != 1:
            raise ValueError(f"items have different camera counts {n_cams}")
        n_cams = n_cams.pop()
        self._arm()
        reqs = [self._request([np.asarray(x, np.uint8) for x in im], t) for im, t in items]
        outs = self.llm.generate(reqs, self.sp, use_tqdm=False)
        rq = [(o.request_id, list(o.prompt_token_ids)) for o in outs]
        feats = [r for r in self.llm.collective_rpc(partial(take, reqs=rq)) if r is not None][0]
        return [(f, token_types(p, engine.IMAGE_TOKEN_ID, n_cams), p) for f, (_, p) in zip(feats, rq)]

    def write_shard(self, store, idxs, path, metadata):
        n_cams = len(store.cams)
        self._arm()
        reqs, tts = [], []
        for b in range(0, len(idxs), self.batch):
            outs = self.llm.generate([self._request(store.images(i), store.instruction(i)) for i in idxs[b:b + self.batch]],
                                     self.sp, use_tqdm=False)
            for o in outs:
                pid = list(o.prompt_token_ids)
                reqs.append((o.request_id, pid))
                tts.append(check_labels(token_types(pid, engine.IMAGE_TOKEN_ID, n_cams), n_cams).tolist())
        res = [r for r in self.llm.collective_rpc(partial(write_shard, reqs=reqs, sample_idx=idxs, token_types=tts,
                                                          path=path, metadata=metadata)) if r]
        assert len(res) == 1, res
        return {"rows": res[0]["rows"], "tokens_per_sample": [len(p) for _, p in reqs]}

    def close(self):
        self.llm.collective_rpc(disarm_layout)
