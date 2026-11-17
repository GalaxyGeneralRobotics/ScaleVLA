"""ScaleVLA: a frontier vision-language model, DeepSeek-V4.1-Flash, read once per action chunk and grounded in continuous
actions by an action head trained from scratch.

    backbone   DeepSeek-V4.1-Flash features through vLLM (engine, worker-side capture, backbone)
    data       sample stores and their builders, feature-cache format, embodiment profiles, normalization, dataset
    features   feature extraction into a cache, merge of cache parts, live feature server and client
    model      feature mapper, action head, checkpoint format
    training   configuration, sample stream, optimizer and the training loop
    serving    policy served through the GR00T PolicyServer interface, policy client
Modules import torch lazily, so the simulator environments can use the data layouts and the policy client.
"""
from pathlib import Path

__version__ = "1.0.0"
STORAGE = Path(__file__).resolve().parents[1] / "storage"  # assets, configs, evaluation initial states
HEAD_CONFIG = STORAGE / "configs" / "action_head"  # config.json of the action-head architecture
