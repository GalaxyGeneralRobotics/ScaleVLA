"""Feature cache: the DeepSeek-V4.1-Flash features of every sample of a sample store.

<cache>/manifest.json   {"schema", "backbone", "samples": <sample store dir>, "spec": {...feature identity...},
                         "shards": [{"file", "first", "n", "rows", "sha256"}], "complete": bool}
<cache>/shard_XXXXX.safetensors   samples [first, first + n) of the store:
  features    [rows, d]  bf16   backbone tokens of the n samples, concatenated
  offsets     [n + 1]    int64  sample k = rows offsets[k]:offsets[k + 1]
  token_type  [rows]     int8   TOKEN_TEXT (0) = text / special token, 1 + c = image token of camera c (store order)
  sample_idx  [n]        int64  row index into the sample store
State, action and instruction stay in the sample store, never in the cache.

Readers (same bytes): CacheReader slices the memory-mapped safetensors lazily; PreadCacheReader reads the rows of one
sample with a single os.preadv, optionally straight into the collated batch tensor (FeatRef), which is faster on network
file systems.
"""
import bisect
import json
import os
import struct

import numpy as np

SCHEMA = "scalevla-cache-v1"
LEGACY_SCHEMAS = ("exact-p1-cache-v1",)  # the same format under the name used by earlier versions of this code
TOKEN_TEXT = 0


def load_manifest(cache_dir):
    with open(os.path.join(cache_dir, "manifest.json")) as f:
        m = json.load(f)
    if m.get("schema") not in (SCHEMA, *LEGACY_SCHEMAS):
        raise ValueError(f"{cache_dir}: schema {m.get('schema')} is not {SCHEMA}")
    return m


def save_manifest(cache_dir, m):
    tmp = os.path.join(cache_dir, "manifest.json.tmp")
    with open(tmp, "w") as f:
        json.dump(m, f, indent=1)
    os.replace(tmp, os.path.join(cache_dir, "manifest.json"))


def token_types(prompt_token_ids, image_token_id, n_cams):
    """Token labels of one sample: 0 = text / special, 1 + c = image token of camera c. The image tokens of one image form
    one contiguous run of image_token_id; runs are assigned to the cameras in order."""
    ids = np.asarray(prompt_token_ids)
    img = ids == image_token_id
    first = img & ~np.concatenate([[False], img[:-1]])
    if int(first.sum()) != n_cams:
        raise ValueError(f"{int(first.sum())} image-token runs, expected {n_cams}")
    tt = np.zeros(len(ids), np.int8)
    tt[img] = np.cumsum(first)[img]
    return tt


def save_shard(path, feats, token_types_, sample_idx, metadata):
    """Write one shard atomically, as a single sequential write (works on object-store mounts that reject the seeks of
    safetensors.save_file). -> {"rows", "tokens_per_sample"}."""
    import torch
    from safetensors.torch import save
    lens = torch.tensor([f.shape[0] for f in feats], dtype=torch.int64)
    tensors = {"features": torch.cat(feats).contiguous(),
               "offsets": torch.cat([torch.zeros(1, dtype=torch.int64), lens.cumsum(0)]),
               "token_type": torch.cat([torch.as_tensor(np.asarray(t), dtype=torch.int8) for t in token_types_]).contiguous(),
               "sample_idx": torch.tensor(sample_idx, dtype=torch.int64)}
    if tensors["token_type"].numel() != tensors["features"].shape[0]:
        raise RuntimeError("token_type length != feature rows")
    data = save(tensors, metadata=metadata)
    with open(path + ".tmp", "wb") as f:
        f.write(data)
    del data
    os.replace(path + ".tmp", path)
    return {"rows": int(tensors["features"].shape[0]), "tokens_per_sample": lens.tolist()}


class CacheReader:
    """Random access to cached features by sample index. Shards are opened lazily and memory-mapped."""

    def __init__(self, cache_dir, require_complete=True):
        self.dir = cache_dir
        self.m = load_manifest(cache_dir)
        if require_complete and not self.m.get("complete"):
            raise ValueError(f"{cache_dir}: cache not complete")
        self.shards = sorted(self.m["shards"], key=lambda s: s["first"])
        self._firsts = [s["first"] for s in self.shards]
        self._open = {}

    def __len__(self):
        return sum(s["n"] for s in self.shards)

    def _shard(self, k):
        # one safe_open handle per shard and process; feature rows are read lazily by slice
        if k not in self._open:
            from safetensors import safe_open
            f = safe_open(os.path.join(self.dir, self.shards[k]["file"]), framework="pt")
            self._open[k] = (f, f.get_tensor("offsets"), f.get_tensor("sample_idx"), f.get_tensor("token_type"))
        return self._open[k]

    def _locate(self, sample_idx):
        k = bisect.bisect_right(self._firsts, sample_idx) - 1
        if k < 0 or not 0 <= sample_idx - self.shards[k]["first"] < self.shards[k]["n"]:
            raise KeyError(sample_idx)
        j = sample_idx - self.shards[k]["first"]
        f, off, sidx, tt = self._shard(k)
        a, b = int(off[j]), int(off[j + 1])
        assert int(sidx[j]) == sample_idx
        return k, f, a, b, tt

    def get(self, sample_idx):
        """-> (features [T, d] bf16, token_type [T] int8) of one sample."""
        _, f, a, b, tt = self._locate(sample_idx)
        return f.get_slice("features")[a:b], tt[a:b]


class PreadCacheReader(CacheReader):
    """Same lookup and bytes as CacheReader, but the rows of one sample are read with one os.preadv of their byte range
    (from the safetensors header). The shard file is opened per read, so many loader workers do not exhaust the open-file
    limit. ref() returns a FeatRef that collate reads straight into the batch tensor."""

    def __init__(self, cache_dir, require_complete=True):
        super().__init__(cache_dir, require_complete=require_complete)
        self._hdr = {}

    def _features_header(self, k):
        if k not in self._hdr:
            import torch
            dtypes = {"BF16": (torch.bfloat16, 2), "F16": (torch.float16, 2), "F32": (torch.float32, 4)}
            p = os.path.join(self.dir, self.shards[k]["file"])
            with open(p, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                h = json.loads(f.read(n))["features"]
            dt, nb = dtypes[h["dtype"]]
            d = int(h["shape"][1])
            self._hdr[k] = (p, 8 + n + int(h["data_offsets"][0]), d * nb, d, dt)
        return self._hdr[k]

    def ref(self, sample_idx):
        """-> (FeatRef of the feature rows, token_type [T] int8); nothing is read until FeatRef.read_into."""
        k, _, a, b, tt = self._locate(sample_idx)
        p, base, row_bytes, d, dt = self._features_header(k)
        return FeatRef(p, base + a * row_bytes, b - a, d, dt), tt[a:b]

    def get(self, sample_idx):
        import torch
        ref, tt = self.ref(sample_idx)
        out = torch.empty(ref.shape, dtype=ref.dtype)
        ref.read_into(out)
        return out, tt


class FeatRef:
    """The feature rows of one sample in a shard file; .shape / .dtype like the tensor they stand for."""
    __slots__ = ("path", "offset", "shape", "dtype")

    def __init__(self, path, offset, n, d, dtype):
        self.path, self.offset, self.shape, self.dtype = path, offset, (n, d), dtype

    def read_into(self, dst):
        """dst: a contiguous [n, d] tensor of self.dtype."""
        import torch
        assert tuple(dst.shape) == self.shape and dst.dtype == self.dtype and dst.is_contiguous()
        if self.shape[0] == 0:
            return
        mv = memoryview(dst.view(torch.uint8).numpy()).cast("B")
        pos = 0
        fd = os.open(self.path, os.O_RDONLY)
        try:
            while pos < len(mv):
                got = os.preadv(fd, [mv[pos:]], self.offset + pos)
                if got <= 0:
                    raise IOError(f"{self.path}: short read at {pos} / {len(mv)} bytes (offset {self.offset})")
                pos += got
        finally:
            os.close(fd)


def open_cache(cache_dir, reader="mmap", require_complete=True):
    return {"mmap": CacheReader, "pread": PreadCacheReader}[reader](cache_dir, require_complete=require_complete)
