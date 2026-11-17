"""Training configuration: a YAML recipe (storage/configs/<benchmark>.yaml) with the sections below, plus single-field
overrides:

    torchrun --nproc_per_node 4 -m scripts.train storage/configs/libero.yaml --data.samples S --data.cache C --run.output_dir R

Strings in the YAML and on the command line may use environment variables (${VAR}). `python -m scripts.train --help` lists
every field with its default.
"""
from __future__ import annotations

import dataclasses
import os
import typing
from dataclasses import dataclass, field
from pathlib import Path

from .. import HEAD_CONFIG


def _f(default, help_, required=False):
    return field(default=default, metadata={"help": help_, "required": required})


@dataclass
class DataConfig:
    samples: str = _f("", "sample store dir (scripts.prepare_data)", required=True)
    cache: str = _f("", "feature cache of that store (scripts.extract_features)", required=True)
    horizon: int = _f(16, "action-chunk horizon; must equal the embodiment profile's (LIBERO 16, RoboCasa 8, DexJoCo 30)")
    pad_end: bool = _f(False, "every frame is a chunk start; actions past the episode end repeat its last action")
    state_dropout: float = _f(0.2, "probability of zeroing the whole state vector of a training item")
    demos_per_task: int = _f(0, "train on this many demonstrations per task (0 = all); normalization stays full-data")
    demo_seed: int = _f(42, "seed of the per-task demonstration permutation (nested subsets)")
    cache_reader: str = _f("mmap", "mmap | pread (one os.preadv per sample; same bytes, faster on network storage)")
    workers: int = _f(8, "DataLoader workers per rank")
    stream_block: int = _f(0, "draw the sample stream in blocks of this many steps (0 = one block of run.steps)")


@dataclass
class ModelConfig:
    head_config: str = _f(str(HEAD_CONFIG), "dir of the action-head config.json")
    head_init: str = _f("", "optional file of the common head initialization (created from init_seed if missing)")
    init_seed: int = _f(0, "seed of the common head initialization")
    rank: int = _f(2048, "feature-mapper bottleneck width")
    mapper_seed: int = _f(1, "seed of the feature-mapper initialization")


@dataclass
class OptimConfig:
    lr: float = _f(1e-4, "AdamW learning rate")
    wd: float = _f(1e-5, "AdamW weight decay")
    wd_no_bias_norm: bool = _f(False, "no weight decay on bias and norm parameters (Hugging Face Trainer rule)")
    lr_schedule: str = _f("cosine", "cosine | wsd (warmup, constant, linear cooldown)")
    warmup_ratio: float = _f(0.05, "cosine warmup = ceil(warmup_ratio * run.steps) unless warmup_steps is set")
    warmup_steps: int = _f(-1, "-1: from warmup_ratio; required for wsd")
    cooldown_start: int = _f(0, "wsd: linear decay to 0 from this step to run.steps (0 = none)")


@dataclass
class RunConfig:
    output_dir: str = _f("", "run dir: run.json, train_log.jsonl, checkpoint-<step>/", required=True)
    steps: int = _f(5000, "optimizer steps")
    gbs: int = _f(640, "global batch size (split evenly over the ranks)")
    seed: int = _f(42, "seed of the sample stream and of every rank's RNG")
    save: str = _f("every:1000", "policy checkpoints: comma steps and / or every:N")
    save_state: str = _f("", "resumable training states: comma steps and / or every:N")
    resume: str = _f("", "a checkpoint dir with a resumable training state to continue from")


@dataclass
class TrainConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    run: RunConfig = field(default_factory=RunConfig)

    def to_dict(self):
        return dataclasses.asdict(self)


SECTIONS = {f.name: f.default_factory for f in dataclasses.fields(TrainConfig)}


def _cast(tp, value, where):
    if isinstance(value, str):
        if tp is bool:
            if value.lower() not in ("true", "false", "1", "0", "yes", "no"):
                raise SystemExit(f"{where}: expected a boolean, got {value!r}")
            return value.lower() in ("true", "1", "yes")
        if tp in (int, float):
            return tp(value)
        return os.path.expanduser(os.path.expandvars(value))
    if tp is float and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    if tp is str and isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)  # e.g. save: 5000 in YAML
    if not isinstance(value, tp):
        raise SystemExit(f"{where}: expected {tp.__name__}, got {value!r}")
    return value


def _parse_overrides(overrides):
    out, it = [], iter(overrides)
    for tok in it:
        if not tok.startswith("--") or "." not in tok:
            raise SystemExit(f"override {tok!r}: expected --section.field value")
        key, eq, val = tok[2:].partition("=")
        if not eq:
            val = next(it, None)
            if val is None:
                raise SystemExit(f"override --{key}: missing value")
        out.append((key, val))
    return out


def load_config(path: str | Path | None, overrides=()) -> TrainConfig:
    """YAML recipe (sections data / model / optim / run) + ``--section.field value`` overrides."""
    raw = {}
    if path:
        import yaml
        with open(os.path.expandvars(str(path))) as f:
            raw = yaml.safe_load(f) or {}
    for key, val in _parse_overrides(overrides):
        sec, name = key.split(".", 1)
        raw.setdefault(sec, {})[name.replace("-", "_")] = val
    unknown = set(raw) - set(SECTIONS)
    if unknown:
        raise SystemExit(f"{path}: unknown sections {sorted(unknown)} (known: {sorted(SECTIONS)})")
    built = {}
    for sec, factory in SECTIONS.items():
        obj, given = factory(), raw.get(sec) or {}
        hints = typing.get_type_hints(type(obj))
        fields = {f.name: f for f in dataclasses.fields(obj)}
        bad = set(given) - set(fields)
        if bad:
            raise SystemExit(f"{sec}: unknown fields {sorted(bad)} (known: {sorted(fields)})")
        for k, v in given.items():
            setattr(obj, k, _cast(hints[k], v, f"{sec}.{k}"))
        missing = [k for k, f in fields.items() if f.metadata.get("required") and not getattr(obj, k)]
        if missing:
            raise SystemExit(f"missing required fields: {', '.join(f'{sec}.{k}' for k in missing)}")
        built[sec] = obj
    return TrainConfig(**built)


def describe() -> str:
    """Every field with its default and help (python -m scripts.train --help)."""
    lines = []
    for sec, factory in SECTIONS.items():
        lines.append(f"{sec}:")
        for f in dataclasses.fields(factory()):
            req = " (required)" if f.metadata.get("required") else ""
            lines.append(f"  {f.name:<16} {f.default!r:<24} {f.metadata.get('help', '')}{req}")
    return "\n".join(lines)
