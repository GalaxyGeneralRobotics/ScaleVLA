"""Optimizer and learning-rate schedule: AdamW (gradient clipping at 1.0 and bf16 autocast are in the training loop).
  cosine  transformers' cosine schedule with linear warmup (ceil(warmup_ratio * run.steps) steps unless warmup_steps)
  wsd     warmup-stable-decay: linear warmup, constant, then with cooldown_start C a linear decay to 0 from C to run.steps
"""
import math

import torch


def make_optimizer(cfg, module):
    params = [(n, p) for n, p in module.named_parameters() if p.requires_grad]
    if cfg.wd_no_bias_norm:  # the Hugging Face / GR00T trainer rule: no decay on LayerNorm modules, biases and norms
        from transformers import Trainer
        decay = set(Trainer.get_decay_parameter_names(None, module))
        return torch.optim.AdamW([{"params": [p for n, p in params if n in decay], "weight_decay": cfg.wd},
                                  {"params": [p for n, p in params if n not in decay], "weight_decay": 0.0}],
                                 lr=cfg.lr, weight_decay=cfg.wd)
    return torch.optim.AdamW([p for _, p in params], lr=cfg.lr, weight_decay=cfg.wd)


def make_scheduler(cfg, steps, opt):
    """-> (scheduler, record for run.json)."""
    n_warm = cfg.warmup_steps if cfg.warmup_steps >= 0 else math.ceil(cfg.warmup_ratio * steps)
    if cfg.lr_schedule == "cosine":
        from transformers import get_cosine_schedule_with_warmup
        return get_cosine_schedule_with_warmup(opt, n_warm, steps), {"type": "cosine", "warmup_steps": n_warm, "total_steps": steps}
    if cfg.lr_schedule != "wsd":
        raise SystemExit(f"unknown optim.lr_schedule {cfg.lr_schedule}")
    if cfg.warmup_steps < 0 or not 0 <= cfg.cooldown_start < steps:
        raise SystemExit("wsd needs optim.warmup_steps and 0 <= optim.cooldown_start < run.steps")
    c0 = cfg.cooldown_start

    def wsd(s):
        if s < n_warm:
            return s / max(1, n_warm)  # the warmup of transformers' cosine schedule
        if c0 and s >= c0:
            return max(0.0, (steps - s) / (steps - c0))
        return 1.0
    return torch.optim.lr_scheduler.LambdaLR(opt, wsd), \
        {"type": "wsd", "warmup_steps": n_warm, "cooldown_start": c0 or None, "total_steps": steps}
