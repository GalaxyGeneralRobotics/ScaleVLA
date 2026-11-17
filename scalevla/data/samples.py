"""Sample stores: one directory per (benchmark, split) that holds every training frame once. Feature extraction and the
training loader read only this store, so the cached features, states, actions and instructions always line up.

<store>/index.json   {"schema", "benchmark", "cameras", "tasks": {task_index: name}, "episodes": [{"start", "length",
                      "task_index", ...}], "n", ...}; benchmark-specific fields are documented in builders/<benchmark>.py
<store>/table.npz    state [N, Ds], action [N, Da], episode_index, frame_index, task_index [N], ... (row = sample index)
frames, schema v1    frames_<cam>.npy                          decoded uint8 [N, H, W, 3] (memory-mapped)
frames, schema v2    jpeg_<cam>.bin + jpeg_<cam>.offsets.npy   one encoded image (JPEG or lossless PNG) per sample, decoded
                                                               with cv2.imdecode
"""
import json

import numpy as np

SCHEMA_V1 = "scalevla-samples-v1"
SCHEMA_V2 = "scalevla-samples-v2"
LEGACY_SCHEMAS = {"exact-p1-samples-v1": SCHEMA_V1, "exact-p1-samples-v2": SCHEMA_V2}  # names used by earlier versions


def load_index(store):
    with open(f"{store}/index.json") as f:
        return json.load(f)


def load_table(store):
    return dict(np.load(f"{store}/table.npz"))


def schema(index):
    return LEGACY_SCHEMAS.get(index["schema"], index["schema"])


class EncodedFrames:
    """Frames of one camera stored as encoded images: obj[i] -> uint8 [H, W, 3]."""

    def __init__(self, store, cam):
        self.off = np.load(f"{store}/jpeg_{cam}.offsets.npy")
        self.bin = np.memmap(f"{store}/jpeg_{cam}.bin", np.uint8, "r")

    def __len__(self):
        return len(self.off) - 1

    def __getitem__(self, i):
        import cv2
        i = int(i)
        return cv2.imdecode(np.asarray(self.bin[self.off[i]:self.off[i + 1]]), cv2.IMREAD_COLOR)


def open_frames(store, index=None):
    """-> {cam: obj} with obj[i] -> uint8 [H, W, 3], for both frame schemas."""
    index = index or load_index(store)
    if schema(index) == SCHEMA_V2:
        return {c: EncodedFrames(store, c) for c in index["cameras"]}
    return {c: np.load(f"{store}/frames_{c}.npy", mmap_mode="r") for c in index["cameras"]}


class SampleStore:
    """Read view of one store for feature extraction: the frames of every camera (store order) and the instruction of
    each sample. Stores with per-episode instructions (index.json "instruction_mode": "per_episode") carry
    instruction_index in the table; the others use the task name."""

    def __init__(self, store):
        self.dir = store
        self.idx = load_index(store)
        self.tab = load_table(store)  # loaded once: NpzFile re-reads an array on every access
        self.cams = list(self.idx["cameras"])
        self.frames = open_frames(store, self.idx)
        self.tasks = {int(k): v for k, v in self.idx["tasks"].items()}
        self.per_episode = self.idx.get("instruction_mode") == "per_episode"

    def __len__(self):
        return self.idx["n"]

    def instruction(self, i):
        if self.per_episode:
            return self.idx["instructions"][int(self.tab["instruction_index"][i])]
        return self.tasks[int(self.tab["task_index"][i])]

    def images(self, i):
        """uint8 [H, W, 3] per camera, in store camera order."""
        return [self.frames[c][i] for c in self.cams]
