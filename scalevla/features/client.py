"""Client of a live feature server (scalevla.features.server) and an offline stand-in that serves cached rows."""
import pickle

import numpy as np
import torch


class FeatureClient:
    def __init__(self, port, timeout_s=600):
        import zmq
        self.sock = zmq.Context.instance().socket(zmq.REQ)
        self.sock.setsockopt(zmq.RCVTIMEO, timeout_s * 1000)
        self.sock.connect(f"tcp://127.0.0.1:{port}")

    def call(self, **msg):
        self.sock.send(pickle.dumps(msg))
        r = pickle.loads(self.sock.recv())
        if not r["ok"]:
            raise RuntimeError(f"feature server: {r['error']}\n{r.get('tb', '')}")
        return r["out"]

    def features(self, frames, instructions):
        """-> [(features [L, d] bf16, token_type [L] int8, prompt token ids)] per item."""
        out = self.call(cmd="features", frames=frames, instructions=instructions)
        return [(torch.from_numpy(f).view(torch.bfloat16), torch.from_numpy(np.asarray(t, np.int8)), p)
                for f, t, p in zip(out["features"], out["token_type"], out["prompt_token_ids"])]


class CachedFeatures:
    """Offline stand-in for FeatureClient: returns the cached rows of given sample indices (tests, offline checks).
    Call bind(sample_indices) before each features() call; frames and instructions are ignored."""

    def __init__(self, cache_dir):
        from ..data.cache import CacheReader
        self.reader, self.next = CacheReader(cache_dir, require_complete=False), None

    def bind(self, sample_indices):
        self.next = list(sample_indices)
        return self

    def call(self, cmd, **_):
        assert cmd == "ping"
        return "pong (cached)"

    def features(self, frames, instructions):
        assert self.next is not None and len(self.next) == len(frames), "bind() the sample indices first"
        out = [(*self.reader.get(i), None) for i in self.next]
        self.next = None
        return [(f.clone(), t.clone(), p) for f, t, p in out]
