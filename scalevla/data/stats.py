"""State / action normalization as GR00T fine-tuning does it: statistics computed like gr00t.data.stats over the training
frames, then GR00T's StateActionProcessor with the 1st / 99th percentiles mapped to [-1, 1] and clipped.

stats = {embodiment tag: {"state": {key: {mean, std, min, max, q01, q99}}, "action": {...}}} (recorded in policy.json).
"""
import numpy as np


def gr00t_stats(arr):
    """Same statistics as gr00t.data.stats.calculate_dataset_statistics (float32, numpy)."""
    x = np.asarray(arr, dtype=np.float32)
    return dict(mean=np.mean(x, 0).tolist(), std=np.std(x, 0).tolist(), min=np.min(x, 0).tolist(),
                max=np.max(x, 0).tolist(), q01=np.quantile(x, 0.01, axis=0).tolist(), q99=np.quantile(x, 0.99, axis=0).tolist())


def split_stats(full, ranges):
    return {k: {s: v[a:b] for s, v in full.items()} for k, (a, b) in ranges.items()}


def compute_stats(table, profile):
    """Statistics over every row of the sample store."""
    return {profile.tag: {"state": split_stats(gr00t_stats(table["state"]), profile.ranges["state"]),
                          "action": split_stats(gr00t_stats(table["action"]), profile.ranges["action"])}}


def processor_from_stats(stats, profile, train=True):
    """clip_outliers=True: the released GR00T LIBERO processor config has no clip_outliers key, so GR00T's from_pretrained
    fills in True."""
    from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
    from gr00t.data.state_action.state_action_processor import StateActionProcessor
    mc = {profile.tag: {m: MODALITY_CONFIGS[profile.tag][m] for m in ("state", "action")}}
    p = StateActionProcessor(modality_configs=mc, statistics=stats, use_percentiles=True, clip_outliers=True,
                             apply_sincos_state_encoding=False, use_relative_action=True)
    p.train() if train else p.eval()
    return p


def make_processor(table, profile, train=True):
    stats = compute_stats(table, profile)
    return processor_from_stats(stats, profile, train), stats
