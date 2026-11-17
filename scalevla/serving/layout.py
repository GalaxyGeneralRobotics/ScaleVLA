"""Head-output action layout. The trainer concatenates the normalized actions in the profile's key order
(profile.ranges["action"]), so in the head output key k sits at the cumulative width of the keys before it, not at its
store column range (a, b). LIBERO and DexJoCo ranges are contiguous in key order (both splits agree); RoboCasa's are not
(gripper_close (11, 12) comes first). Always split head outputs with split_head_output().
"""
import numpy as np


def head_offsets(ranges_action):
    off, out = 0, {}
    for k, (a, b) in ranges_action.items():
        out[k] = (off, off + b - a)
        off += b - a
    return out


def split_head_output(pred, ranges_action):
    return {k: pred[..., x:y] for k, (x, y) in head_offsets(ranges_action).items()}


def self_test(ranges_action):
    """Emulate the trainer on a store vector with a distinct value per column (store -> per-key slices -> concatenation in
    key order); split_head_output must give every key its own store columns back."""
    D = max(b for _, b in ranges_action.values())
    assert sorted(c for a, b in ranges_action.values() for c in range(a, b)) == list(range(D)), ranges_action
    v = np.arange(D, dtype=np.float64)[None, None] + 100.0
    trained = np.concatenate([v[..., a:b] for a, b in ranges_action.values()], -1)
    got = split_head_output(trained, ranges_action)
    return all(np.array_equal(got[k], v[..., a:b]) for k, (a, b) in ranges_action.items())
